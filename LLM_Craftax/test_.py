import sys
import os
import jax
from PIL import Image
import jax.numpy as jnp
import numpy as np
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional



PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_ma.renderer.renderer_pixels import render_craftax_pixels
from craftax.craftax_ma.constants import *

from LLM_Craftax.craftax_coop_sym_obs_parser import CraftaxObservationParser
from craftax_ma_sym_obs_parser import CraftaxMAObservationParser
from craftax.craftax_ma.craftax_state import StaticEnvParams
from craftax.craftax_ma.constants import OBS_DIM, MAX_OBS_DIM, BlockType
from craftax.craftax_ma.craftax_state import EnvState
from astar import astar

from craftax.craftax_ma.envs.craftax_symbolic_env import CraftaxMASymbolicEnv

# def get_map_view(state: EnvState):
#     """Returns (player_count, 9, 11) integer array of BlockType values."""
#     obs_dim_array = jnp.array(OBS_DIM, dtype=jnp.int32)
#     padded = jnp.pad(
#         state.map[state.player_level],
#         (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
#         constant_values=BlockType.OUT_OF_BOUNDS.value,
#     )
#     tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2
#     return jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
#         padded, tl_corner, OBS_DIM
#     )  # shape: (player_count, 9, 11)

plan= [5]#,4,5,1,8,0]

env_name = "Craftax-Coop-Symbolic"
# env = make_craftax_env_from_name(env_name)
env = CraftaxMASymbolicEnv(num_agents=2)
rng = jax.random.PRNGKey(2026)
static_env_params = StaticEnvParams()
obs, env_state = env.reset(rng)
print(env.agents)
# exit()
# for agent_name in env.agents:
#     print(f"Observation for {agent_name}: {obs[agent_name].shape}")
# print(obs_.shape) for agent_name in env.agents:
# print(SOLID_BLOCKS)
pixel_size = 64
# print(TEXTURES.keys())
# print(pixel_size)
# exit()
player_specific_textures = load_player_specific_textures(
    TEXTURES[pixel_size],
    static_env_params.player_count
)
parser = CraftaxMAObservationParser(num_agents=2)
# parser = CraftaxObservationParser(num_agents=2)
# print(env.agents)
# print(obs)
imgs = []
START_POS = (4, 5)
for i in range(len(plan)):
    step_imgs = []
    # print(static_env_params.player_count)
    # print(player_specific_textures)
    pixels = render_craftax_pixels(
                env_state, 
                pixel_size, 
                static_env_params,
                player_specific_textures
            ) / 255.0
    # print(pixels.shape)
    print(f"\n=== STEP {i} ===")
    for id, agent_name in enumerate(env.agents):
        # print(agent_name)
        img = Image.fromarray((np.array(pixels)[id] * 255).astype(np.uint8))
        step_imgs.append(img)
        # print(f"Observation for {agent_name}: {obs[agent_name]}")
        parsed = parser.parse_observation(obs[agent_name], agent_id=id)
        short_term_obs, long_term_obs = parser.to_text(parsed, id)
        
        grid = parsed["map"]["passable_grid"]
        facing = parsed["inventory"]["state"]["facing"]
        # print(grid)
        print(facing)
        # goal = None
        # while goal is None:
        #     goal_x = np.random.randint(0, grid.shape[0])
        #     goal_y = np.random.randint(0, grid.shape[1])
        #     if grid[goal_x, goal_y] != 1:
        #         goal = (goal_x, goal_y)
        # print(f"Selected random goal: {goal}")
        # path = astar(grid, START_POS, goal)
        # print(f"Path from player to goal: {path}")
        # print(f"Parsed observation for {agent_name}: {parsed}")
        print(f"Short-term observation for {agent_name}: {short_term_obs}")
        print(f"Long-term observation for {agent_name}: {long_term_obs}")

    actions_dict = {agent_name: plan[i] for agent_name in env.agents}
    rng, step_rng = jax.random.split(rng)

    obs, env_state, rewards, dones, info = env.step(step_rng, env_state, actions_dict)
    # print(info)
    for agent_name in env.agents:
        print(f"Reward for {agent_name}: {rewards[agent_name]}")
    imgs.append(step_imgs)

# for step_idx, step_imgs in enumerate(imgs):
#     for agent_id, img in enumerate(step_imgs):
#         img.save(f"test_img/step_{step_idx}_agent_{agent_id}.png")

# parsed = parser.parse_observation(obs[agent_name], agent_id=agent_id)

# class CraftaxLanguageWrapper:
#     def __init__(self, env, parser):
#         self.env = env
#         self.parser = parser
    
#     def reset(self, rng):
#         obs, env_state = self.env.reset(rng)
#         textual_obs = {}
#         for id, agent_name in enumerate(self.env.agents):
#             parsed = self.parser.parse_observation(obs[agent_name], agent_id=id)
#             textual_obs[agent_name] = self.parser.to_text(parsed, id)
#         return textual_obs, env_state
    
#     def step(self, actions: Dict[str, str], env_state):
#         obs, rewards, dones, infos = self.env.step(actions, env_state)
#         textual_obs = {}
#         for id, agent_name in enumerate(self.env.agents):
#             parsed = self.parser.parse_observation(obs[agent_name], agent_id=id)
#             textual_obs[agent_name] = self.parser.to_text(parsed, id)
#         return textual_obs, rewards, dones, infos