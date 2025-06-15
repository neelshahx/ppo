import numpy as np
import gymnasium as gym
import minigrid
from matplotlib import pyplot as plt

plt.ion()

# https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/core/constants.py

OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}
IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}

COLOR_TO_IDX = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "purple": 3,
    "yellow": 4,
    "grey": 5,
}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}

STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}
IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

ACTION_TO_IDX = {
    "left": 0,
    "right": 1,
    "forward": 2,
    "pickup": 3,
    "drop": 4,
    "toggle": 5,
    "done": 6,
}

IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}

KEY_OBJECT_IDS = {
    OBJECT_TO_IDX["key"],
    OBJECT_TO_IDX["door"],
    OBJECT_TO_IDX["goal"],
}


def get_object_description(obj_id, color_id, state_id):
    """Generates a human-readable description of an object."""
    obj_name = IDX_TO_OBJECT.get(obj_id, f"UnknownObject({obj_id})")
    color_name = IDX_TO_COLOR.get(color_id, f"UnknownColor({color_id})")

    desc = f"{color_name.capitalize()} {obj_name}"

    if obj_name == "door":
        state_name = IDX_TO_STATE.get(state_id, f"UnknownState({state_id})")
        desc += f" ({state_name})"
    elif obj_name == "empty" or obj_name == "unseen":
        return obj_name.capitalize()  # No need for color for these

    return desc


def interpret_agent_view(observation_image):
    """
    Interprets the agent's 7x7x3 field of vision.
    Identifies key objects and their relative position to the agent.

    Args:
        observation_image (np.ndarray): The 7x7x3 array from obs['image'].

    Returns:
        list: A list of strings, each describing a found key object.
    """
    if observation_image is None or observation_image.shape != (7, 7, 3):
        return ["Invalid observation image format."]

    found_objects_descriptions = []
    view_height, view_width, _ = observation_image.shape  # Should be 7, 7, 3

    # The agent is at the "bottom-center" of this view, looking "up" into the array.
    # Row 6 is 1 step ahead, Row 0 is 7 steps ahead.
    # Column 3 is straight ahead.

    center_col_idx = (view_width - 1) // 2  # Should be 3 for width 7

    for r in range(view_height):  # Iterate rows from 0 (furthest) to 6 (closest)
        for c in range(view_width):  # Iterate columns from 0 (leftmost) to 6 (rightmost)
            obj_id, color_id, state_id = observation_image[r, c]

            if obj_id in KEY_OBJECT_IDS:
                obj_desc = get_object_description(obj_id, color_id, state_id)

                # Calculate distance ahead
                # Rows are indexed 0 (furthest) to 6 (closest)
                # So, row 6 is 1 step ahead, row 5 is 2 steps ahead, ..., row 0 is 7 steps ahead.
                steps_ahead = view_height - (r + 1)

                # Calculate horizontal position
                if c == center_col_idx:
                    horizontal_pos_desc = "straight ahead"
                elif c < center_col_idx:
                    cells_to_left = center_col_idx - c
                    s = "s" if cells_to_left > 1 else ""
                    horizontal_pos_desc = f"{cells_to_left} cell{s} to the left"
                else:  # c > center_col_idx
                    cells_to_right = c - center_col_idx
                    s = "s" if cells_to_right > 1 else ""
                    horizontal_pos_desc = f"{cells_to_right} cell{s} to the right"

                s_ahead = "s" if steps_ahead > 1 else ""
                description = (f"Sees: {obj_desc} - {steps_ahead} step{s_ahead} forward, "
                               f"{horizontal_pos_desc}.")
                found_objects_descriptions.append(description)
            elif obj_id == OBJECT_TO_IDX["unseen"]:  # Skip unseen cells
                continue
            # Optionally, you could log other non-key objects if needed for full context

    if not found_objects_descriptions:
        return ["Sees: No key objects in view. Mostly empty space or walls."]
    return found_objects_descriptions


def get_orientation_summary(observation_image):
    """
    Provides a qualitative summary of the agent's local orientation
    based on walls in its field of view.
    """
    if observation_image is None or observation_image.shape != (7, 7, 3):
        return "Invalid observation image for orientation."

    WALL_ID = OBJECT_TO_IDX["wall"]
    EMPTY_ID = OBJECT_TO_IDX["empty"]
    UNSEEN_ID = OBJECT_TO_IDX["unseen"]

    view_height, view_width, _ = observation_image.shape
    center_col_idx = (view_width - 1) // 2

    # Helper to get object ID at a relative coordinate in the view
    # (r_fwd, c_rel): r_fwd=1 is 1 step ahead, c_rel=0 is center, -1 is 1 left, 1 is 1 right
    def get_id_rel(steps_fwd, cells_sideways):
        if not (1 <= steps_fwd <= view_height):
            return UNSEEN_ID
        r_idx = (view_height - 1) - (steps_fwd - 1)
        c_idx = center_col_idx + cells_sideways
        if not (0 <= c_idx < view_width):
            return UNSEEN_ID
        return observation_image[r_idx, c_idx, 0]

    # Rule-based inference for orientation
    # 1. Check for wall immediately in front
    if get_id_rel(steps_fwd=1, cells_sideways=0) == WALL_ID:
        desc = "Facing a wall 1 step forward."
        wall_left1 = get_id_rel(steps_fwd=1, cells_sideways=-1) == WALL_ID
        wall_right1 = get_id_rel(steps_fwd=1, cells_sideways=1) == WALL_ID
        if wall_left1 and wall_right1:
            desc += " It appears solid directly in front."
        elif wall_left1:
            desc += " Wall extends to the front-left, potential opening to the front-right."
        elif wall_right1:
            desc += " Wall extends to the front-right, potential opening to the front-left."
        return desc

    # 2. Check for corridor-like structure
    path_clear_2_steps = (get_id_rel(steps_fwd=1, cells_sideways=0) == EMPTY_ID and \
                          get_id_rel(steps_fwd=2, cells_sideways=0) == EMPTY_ID)

    if path_clear_2_steps:
        wall_on_left_flank = (get_id_rel(steps_fwd=1, cells_sideways=-1) == WALL_ID or \
                              get_id_rel(steps_fwd=2, cells_sideways=-1) == WALL_ID or \
                              get_id_rel(steps_fwd=1, cells_sideways=-2) == WALL_ID or \
                              get_id_rel(steps_fwd=2, cells_sideways=-2) == WALL_ID)
        wall_on_right_flank = (get_id_rel(steps_fwd=1, cells_sideways=1) == WALL_ID or \
                               get_id_rel(steps_fwd=2, cells_sideways=1) == WALL_ID or \
                               get_id_rel(steps_fwd=1, cells_sideways=2) == WALL_ID or \
                               get_id_rel(steps_fwd=2, cells_sideways=2) == WALL_ID)
        if wall_on_left_flank and wall_on_right_flank:
            return "Path ahead looks like a corridor (walls to left and right)."
        if wall_on_left_flank:
            return "Path ahead with a wall to the left; open to the right."
        if wall_on_right_flank:
            return "Path ahead with a wall to the right; open to the left."

    # 3. Check for general open space ahead
    path_clear_3_steps = path_clear_2_steps and (get_id_rel(steps_fwd=3, cells_sideways=0) == EMPTY_ID)
    if path_clear_3_steps:
        nearby_walls = 0
        for fwd in range(1, 4):
            for side in range(-1, 2):
                if get_id_rel(steps_fwd=fwd, cells_sideways=side) == WALL_ID:
                    nearby_walls += 1
        if nearby_walls <= 1:
            return "Appears to be in a relatively open space ahead."

    # 4. Fallback
    wall_count = 0
    empty_count = 0
    for r_idx_arr in range(view_height - 3, view_height):
        for c_idx_arr in range(view_width):
            obj_id = observation_image[r_idx_arr, c_idx_arr, 0]
            if obj_id == WALL_ID:
                wall_count += 1
            elif obj_id == EMPTY_ID:
                empty_count += 1

    if wall_count == 0 and empty_count > 0:
        return "In a very open area; no walls detected nearby."
    if wall_count == 0 and empty_count == 0:
        return "Limited visibility or no clear wall/empty patterns nearby."

    return "Orientation context is ambiguous based on nearby walls."


def update_and_get_history_entry(
        observation_image: np.ndarray,
        action_taken_before_this_obs=None,  # Action that led to this observation
        step_number: int = 0,
        env=None
):
    interpretation_strings = interpret_agent_view(observation_image)
    orientation_summary = get_orientation_summary(observation_image)

    history_entry = {
        "step": step_number,
        "action_leading_to_state": action_taken_before_this_obs,  # Can be numeric or string
        "view_interpretation": interpretation_strings,
        "orientation_summary": orientation_summary,
        "raw_view": env.render()
    }
    return history_entry


def manage_history(
        history_log: list,
        new_entry: dict,
        max_history_steps: int = None  # Max number of past steps to keep
):
    history_log.append(new_entry)
    if max_history_steps is not None and len(history_log) > max_history_steps:
        history_log.pop(0)


# %% free explore

env = gym.make("MiniGrid-DoorKey-16x16-v0", max_episode_steps=10, render_mode="rgb_array")

obs, _ = env.reset()
initial_obs_image = obs['image']

agent_history = []
MAX_HISTORY = 10
current_step_count = 0

# Record interpretation of the initial state
initial_entry = update_and_get_history_entry(
    initial_obs_image.transpose(1, 0, 2),  # TODO(nshah, 2025/05/23): hack
    action_taken_before_this_obs="<initial_state>",
    step_number=current_step_count,
    env=env
)
manage_history(agent_history, initial_entry, MAX_HISTORY)
current_step_count += 1

for i in range(10):
    random_action = env.action_space.sample() % 3  # TMP: only allow left, right, forward
    action_taken = IDX_TO_ACTION[random_action]

    obs, _, _, _, _ = env.step(random_action)
    new_obs_image = obs['image']
    history_entry = update_and_get_history_entry(
        new_obs_image.transpose(1, 0, 2),  # TODO(nshah, 2025/05/23): hack
        action_taken_before_this_obs=action_taken,
        step_number=current_step_count,
        env=env
    )
    manage_history(agent_history, history_entry, MAX_HISTORY)

plt.imshow(agent_history[0]['raw_view'])

# %% collect interpretation

for i in range(len(agent_history)):
    print(
        agent_history[i]['view_interpretation'],
        agent_history[i]['orientation_summary'],
        "Previous action =", agent_history[i]['action_leading_to_state']
    )
# %% test code

print("\n--- Example 1 ---")

example_obs_image = np.array([
    [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [5, 4, 0], [1, 0, 0]],  # Key [5,4,0] is at row 5, col 5
    [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]
], dtype=np.uint8)

descriptions = interpret_agent_view(example_obs_image)
for desc in descriptions:
    print(desc)

orientations = get_orientation_summary(example_obs_image)
print(orientations)

# %%
print("\n--- Example 2 ---")
# Agent is 1 step in front of a locked red door, directly ahead.
# A green key is 2 steps ahead, 1 cell to the left.
# A blue goal is 3 steps ahead, 2 cells to the right.
example_obs_image_2 = np.zeros((7, 7, 3), dtype=np.uint8)
example_obs_image_2[:, :, 0] = OBJECT_TO_IDX["empty"]  # Fill with empty first

# Closest row (1 step ahead)
example_obs_image_2[6, 3] = [OBJECT_TO_IDX["door"], COLOR_TO_IDX["red"], STATE_TO_IDX["locked"]]

# Second closest row (2 steps ahead)
example_obs_image_2[5, 2] = [OBJECT_TO_IDX["key"], COLOR_TO_IDX["green"], 0]  # Key (green)

# Third closest row (3 steps ahead)
example_obs_image_2[4, 5] = [OBJECT_TO_IDX["goal"], COLOR_TO_IDX["blue"], 0]  # Goal (blue)

descriptions2 = interpret_agent_view(example_obs_image_2)
for desc in descriptions2:
    print(desc)

orientations = get_orientation_summary(example_obs_image)
print(orientations)

# %%
print("\n--- Example 3 ---")
# This was the one with walls
example_obs_image_3 = np.array([
    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0]],
    [[0, 0, 0], [2, 5, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[0, 0, 0], [2, 5, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[0, 0, 0], [2, 5, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[0, 0, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0]],
], dtype=np.uint8)

# Let's make walls "key objects" for this example to see them reported
KEY_OBJECT_IDS_temp = KEY_OBJECT_IDS.copy()
KEY_OBJECT_IDS_temp.add(OBJECT_TO_IDX["wall"])
# Temporarily override KEY_OBJECT_IDS for this specific call
original_key_objects = KEY_OBJECT_IDS.copy()  # Save original
globals()['KEY_OBJECT_IDS'] = KEY_OBJECT_IDS_temp  # Modify global for the call
descriptions3 = interpret_agent_view(example_obs_image_3)
globals()['KEY_OBJECT_IDS'] = original_key_objects  # Restore original

for desc in descriptions3:
    print(desc)

orientations = get_orientation_summary(example_obs_image)
print(orientations)
