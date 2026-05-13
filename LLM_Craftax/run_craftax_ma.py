import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import json
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import hydra
import time
import imageio
from datetime import datetime
from omegaconf import DictConfig
from hydra.utils import instantiate, get_original_cwd
from PIL import Image, ImageDraw, ImageFont
# import omegaconf
# from omegaconf import open_dict
# from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "craftax"))
sys.path.insert(0, os.path.dirname(__file__))

# from craftax.craftax_env import make_craftax_env_from_name
from env_wrapper import CraftaxLanguageWrapper  
from llm_client import LLMClientWrapper
from prompt_builder import HistoryPromptBuilder, HistoryandPlansPromptBuilder
from craftax_agents import CoTAgent, NaiveAgent, PlannerAgent
from prompts import get_ma_system_prompt
from craftax.craftax_ma.envs.craftax_symbolic_env import CraftaxMASymbolicEnv
from craftax_ma_sym_obs_parser import CraftaxMAObservationParser
from craftax.craftax_ma.constants import *
from craftax_agents import ACTIONS
from craftax.craftax_ma.craftax_state import EnvState
from craftax.craftax_ma.craftax_state import StaticEnvParams
from craftax.craftax_ma.renderer.renderer_pixels import render_craftax_pixels

PIXEL_SIZE = 64


def log_episode(stats ,filepath=None):
    with open(filepath + "episode_stats.json", "w") as f:
        json.dump(stats, f, indent=4)


CHAT_PANEL_WIDTH = 500
CHAT_PANEL_BG = (30, 30, 30)
CHAT_DIVIDER_COLOR = (80, 80, 80)
CHAT_TITLE_COLOR = (100, 200, 255)
CHAT_TEXT_COLOR = (200, 200, 200)
CHAT_ACTION_COLOR = (100, 255, 100)
CHAT_FONT_SIZE = 11
CHAT_LINE_HEIGHT = 14
CHAT_MAX_LINES_PER_SECTION = 8
CHAT_PADDING = 5

def _load_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _render_chat_panel(step_data: dict, height: int) -> Image.Image:
    """Render a text panel showing agent input (observation) and output (reasoning/action)."""
    panel = Image.new("RGB", (CHAT_PANEL_WIDTH, height), color=CHAT_PANEL_BG)
    draw = ImageDraw.Draw(panel)
    font = _load_font(CHAT_FONT_SIZE)
    font_bold = _load_font(CHAT_FONT_SIZE + 1)
    chars_per_line = max(1, (CHAT_PANEL_WIDTH - CHAT_PADDING * 2) // 7)

    y = CHAT_PADDING

    def draw_section(title, text, text_color=CHAT_TEXT_COLOR, max_lines=CHAT_MAX_LINES_PER_SECTION):
        nonlocal y
        if y + CHAT_LINE_HEIGHT >= height - CHAT_PADDING:
            return
        draw.text((CHAT_PADDING, y), title, font=font_bold, fill=CHAT_TITLE_COLOR)
        y += CHAT_LINE_HEIGHT
        if text:
            lines = textwrap.wrap(str(text), width=chars_per_line)
            for line in lines[:max_lines]:
                if y + CHAT_LINE_HEIGHT >= height - CHAT_PADDING:
                    break
                draw.text((CHAT_PADDING, y), line, font=font, fill=text_color)
                y += CHAT_LINE_HEIGHT
        y += 4

    def draw_divider():
        nonlocal y
        if y < height - CHAT_PADDING:
            draw.line([(CHAT_PADDING, y), (CHAT_PANEL_WIDTH - CHAT_PADDING, y)], fill=CHAT_DIVIDER_COLOR, width=1)
            y += 6

    draw_section("INPUT (OBSERVATION):", step_data.get("observation", ""), max_lines=10)
    draw_divider()
    draw_section("OBJECTIVE:", step_data.get("objective", ""), max_lines=2)
    draw_section("OBJECTIVE REASONING:", step_data.get("objective_reasoning", ""))
    draw_section("PLAN:", step_data.get("plan", ""), max_lines=3)
    draw_section("PLAN REASONING:", step_data.get("plan_reasoning", ""))
    draw_divider()
    draw_section("ACTION:", step_data.get("actions", ""), text_color=CHAT_ACTION_COLOR, max_lines=1)

    return panel


def save_episode_video(img_list, episode_dict=None, filepath=None):
    for agent_name in img_list:
        images = img_list[agent_name]
        frames = []
        for step_idx, img in enumerate(images):
            if episode_dict is not None and step_idx in episode_dict:
                step_data = episode_dict[step_idx].get(agent_name, {})
                chat_panel = _render_chat_panel(step_data, img.height)
                composite = Image.new("RGB", (img.width + CHAT_PANEL_WIDTH, img.height))
                composite.paste(img, (0, 0))
                composite.paste(chat_panel, (img.width, 0))
                frames.append(composite)
            else:
                frames.append(img)
        imageio.mimsave(
            filepath + f"{agent_name}_episode_video.mp4",
            [np.array(f) for f in frames],
            fps=2,
            codec="libx264",
            quality=8,
        )

def run_episode(env, agents, config):

    obs, env_state = env.reset()
    for agent_name in env.env.agents:
        agents[agent_name].reset()
    episode_dict = {}
    actions_dict = {}
    # plans_dict = {}
    # obj_dict = {}
    # invalid_act = [False for _ in env.env.agents]
    # episode_stats = {}
    if config.human_render:
        static_env_params = StaticEnvParams()
        player_specific_textures = load_player_specific_textures(
            TEXTURES[PIXEL_SIZE],
            static_env_params.player_count
        )

    img_list = {agent_name: [] for agent_name in env.env.agents}


    for step in range(config.max_episode_steps):
        print(f"\n=== STEP {step} ===")
        actions_dict = {}
        # actions_dict_verb = {}
        episode_dict[step] = {}
        # episode_stats[step] = {}
        episode_dict[step]["stats"] = {}
        invalid_act = {agent_name: False for agent_name in env.env.agents}
        if config.human_render:
            obs_pixels = render_craftax_pixels(
                    env_state, 
                    PIXEL_SIZE, 
                    static_env_params,
                    player_specific_textures
                ) / 255.0
        
        for ind, agent_name in enumerate(env.env.agents):
            actions_dict[agent_name] = None          
            obs_agent = obs[agent_name]["short_term_context"] + "\n" + obs[agent_name]["long_term_context"]
            
            episode_dict[step][agent_name] = {
                "observation": obs_agent,
                # "actions" : actions_dict[agent_name],
                "achievements": ", ".join(env.achievement_by_agent[agent_name]),
                "reasoning": None
            }
            if config.human_render:
                img = Image.fromarray((np.array(obs_pixels)[ind] * 255).astype(np.uint8))

                img_list[agent_name].append(img)

        
            
        if config.parallel_execution:
            with ThreadPoolExecutor(max_workers=len(env.agents)) as executor:
                futures = {}
                for agent_name in env.env.agents:
                    future = executor.submit(agents[agent_name].act, obs[agent_name], step)
                    futures[future] = agent_name
                
                for future in as_completed(futures):
                    agent_name = futures[future]
                    try:
                        action, response = future.result()
                        actions_dict[agent_name] = action
                        # actions_dict_verb[agent_name] = answer.completion
                        if response is not None:
                            episode_dict[step][agent_name]['objective'] = response["objective"]
                            episode_dict[step][agent_name]['plan'] = str(response["plan"])
                            episode_dict[step][agent_name]['objective_reasoning'] = response["reasoning"]['objective_reasoning']
                            episode_dict[step][agent_name]['plan_reasoning'] = response["reasoning"]['plan_reasoning']
                        else:
                            episode_dict[step][agent_name]['objective'] = episode_dict[step - 1][agent_name]['objective']
                            episode_dict[step][agent_name]['plan'] = episode_dict[step - 1][agent_name]['plan']
                            episode_dict[step][agent_name]['objective_reasoning'] = episode_dict[step - 1][agent_name]['objective_reasoning']
                            episode_dict[step][agent_name]['plan_reasoning'] = episode_dict[step - 1][agent_name]['plan_reasoning']
                    except Exception as e:
                        print(f"[Agent {agent_name}] Error: {e}")
                        actions_dict[agent_name] = 0
                        # actions_dict_verb[agent_name] = "NOOP"
        
        else:
            for agent_name in env.env.agents:
                action, response  = agents[agent_name].act(obs[agent_name], step)
                if response is not None:
                    episode_dict[step][agent_name]['objective'] = response["objective"]
                    episode_dict[step][agent_name]['plan'] = str(response["plan"])
                    episode_dict[step][agent_name]['objective_reasoning'] = response["reasoning"]['objective_reasoning'].reasoning
                    episode_dict[step][agent_name]['plan_reasoning'] = response["reasoning"]['plan_reasoning'].reasoning
                    # print("Objective Reasoning: ",episode_dict[step][agent_name]['objective_reasoning'])
                    # print("Plan Reasoning: ",episode_dict[step][agent_name]['plan_reasoning'])
                    # print("Plan :", episode_dict[step][agent_name]['plan'])
                else:
                    episode_dict[step][agent_name]['objective'] = episode_dict[step - 1][agent_name]['objective']
                    episode_dict[step][agent_name]['plan'] = episode_dict[step - 1][agent_name]['plan']
                    episode_dict[step][agent_name]['objective_reasoning'] = episode_dict[step - 1][agent_name]['objective_reasoning']
                    episode_dict[step][agent_name]['plan_reasoning'] = episode_dict[step - 1][agent_name]['plan_reasoning']
                
                act_verb = None
                for act in ACTIONS.keys():
                    if ACTIONS[act] == action:
                        act_verb = act
                        break
                episode_dict[step][agent_name]['actions'] = act_verb        
                actions_dict[agent_name] = action
                # actions_dict_verb[agent_name] = action_verb
                # if action_verb != answer.completion and config.feedback_on_invalid_action:
                #     invalid_act[agent_name] = True

            # print(f"Action for {agent_name}: {action}")

        # print(actions_dict)
        # exit()
        
        obs, env_state, rewards, dones, infos = env.step(actions_dict, env_state)

        # for agent_name in env.env.agents:
        #     obs[agent_name]["long_term_context"] = (
        #             f"\n\nYour previous output did not contain a valid action. Defaulted to action: {action_verb}\n"
        #             + obs[agent_name]["long_term_context"]
        #             if invalid_act[agent_name]
        #             else obs[agent_name]["long_term_context"]
        #         )

        env.update_progress(infos, rewards)

        # print(infos)
        
        for agent_name in env.env.agents:
            # episode_dict[step][agent_name]["actions"] = actions_dict_verb[agent_name]
            # episode_dict[step][agent_name]["reasoning"] = rewards[agent_name]
            episode_dict[step][agent_name]["achievements"] = ", ".join(env.achievement_by_agent[agent_name])
            print(f"\n==={agent_name} ===")
            # print(f"Input No of Tokens: {answer.input_tokens}")
            # print(f"Output No of Tokens: {answer.output_tokens}")
            print(f"Observation: {episode_dict[step][agent_name]['observation']}")
            # print(f"Reasoning: {episode_dict[step][agent_name]['reasoning']}")
            print("Objective Reasoning: ",episode_dict[step][agent_name]['objective_reasoning'])
            print("Plan Reasoning: ",episode_dict[step][agent_name]['plan_reasoning'])
            print("Plan :", episode_dict[step][agent_name]['plan'])
            print(f"Action taken: {episode_dict[step][agent_name]['actions']}")
        stats = env.get_achievements()
        episode_dict[step]["stats"]["score"] =int(stats["score"])
        episode_dict[step]["stats"]["achievements"] = stats["achievements"]
        print(f"\n=== Environment Stats ===")
        print(f"Score: {stats['score']}")
        print(rewards)
        achivements = [ach for ach in stats["achievements"].keys() if stats["achievements"][ach] != 0]
        print(f"Achievements: {' '.join(achivements)}")
        if all(dones.values()):
            break

    return episode_dict, img_list#, episode_stats




@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    original_cwd = get_original_cwd()
    log_path = os.path.join(original_cwd, "results")

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{config.env_name}_{config.seed}_{config.model_id}_{timestamp}"

# env_name = "Craftax-Coop-Symbolic"
    env_ = CraftaxMASymbolicEnv(num_agents=config.num_agents)
    
    # if config.human_render:
    #     static_env_params = StaticEnvParams()
    #     player_specific_textures = load_player_specific_textures(
    #         TEXTURES[PIXEL_SIZE],
    #         static_env_params.player_count
    #     )

    parser = CraftaxMAObservationParser(num_agents=len(env_.agents))
    rng = jax.random.PRNGKey(config.seed)
    env = CraftaxLanguageWrapper(env_, parser, rng)
    # obs, env_state = env.reset(rng)
    LLM_client = LLMClientWrapper(config)
    LLM_client._initialize_client()

    if config.agent_type == "cot":
        AgentClass = CoTAgent
    elif config.agent_type == "naive":
        AgentClass = NaiveAgent
    elif config.agent_type == "planner":
        AgentClass = PlannerAgent

    else:
        raise ValueError(f"Unknown agent type: {config.agent_type}")
    agents = {agent_name: AgentClass(agent_name, LLM_client, HistoryandPlansPromptBuilder(config)) for agent_name in env.agents}


    for aid, agent_name in enumerate(env.agents):
        
        system_prompt = get_ma_system_prompt(aid)
        agents[agent_name].prompt_builder.update_instruction_prompt(system_prompt)

    episode_dict, img_lst = run_episode(env, agents, config)
    # print(episode_dict)]
    os.mkdir(os.path.join(log_path, f"{run_name}_craftax_ma_{config.num_agents}agents/"))
    log_episode(episode_dict, filepath=os.path.join(log_path, f"{run_name}_craftax_ma_{config.num_agents}agents/"))
    if config.human_render:
        save_episode_video(img_lst, episode_dict=episode_dict, filepath=os.path.join(log_path, f"{run_name}_craftax_ma_{config.num_agents}agents/"))

# # print(env.agents)
# # print(obs)
# for i in range(10):
#     print(f"\n=== STEP {i} ===")
#     for id, agent_name in enumerate(env.agents):
#         # print(f"Observation for {agent_name}: {obs[agent_name]}")
#         parsed = parser.parse_observation(obs[agent_name], agent_id=id)
#         short_term_obs, long_term_obs = parser.to_text(parsed, id)
#         # print(f"Parsed observation for {agent_name}: {parsed}")
#         print(f"Short-term observation for {agent_name}: {short_term_obs}")
#         print(f"Long-term observation for {agent_name}: {long_term_obs}")

#     actions_dict = {agent_name: 1 for agent_name in env.agents}
#     rng, step_rng = jax.random.split(rng)

#     obs, env_state, rewards, dones, info = env.step(step_rng, env_state, actions_dict)

if __name__ == "__main__":
    main()