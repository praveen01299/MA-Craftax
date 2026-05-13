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

from craftax.craftax_env import make_craftax_env_from_name
from env_wrapper import CraftaxLanguageWrapper  
from LLM_Craftax.craftax_coop_sym_obs_parser import CraftaxObservationParser
from llm_client import LLMClientWrapper
from prompt_builder import HistoryPromptBuilder, HistoryandPlansPromptBuilderv2, DialogueHistoryPromptBuilder
from craftax_agents import CoTAgent, NaiveAgent, PlannerAgent, PlannerAgentMA
from prompts import get_system_prompt, get_dialogue_system_prompt
# from craftax.craftax_coop.envs.craftax_symbolic_env import CraftaxMASymbolicEnv
from craftax.craftax_coop.constants import *
from craftax_agents import ACTIONS
from craftax.craftax_coop.craftax_state import EnvState
from craftax.craftax_coop.craftax_state import StaticEnvParams
from craftax.craftax_coop.renderer.renderer_pixels import render_craftax_pixels

PIXEL_SIZE = 64

SPECIALIZATION = {
    0: "Warrior",
    1: "Forager",
    2: "Miner"
}



def log_episode(stats ,filepath=None):
    with open(filepath + "episode_stats.json", "w") as f:
        json.dump(stats, f, indent=4)


CHAT_PANEL_WIDTH = 500
CHAT_PANEL_BG = (30, 30, 30)
CHAT_DIVIDER_COLOR = (80, 80, 80)
CHAT_TITLE_COLOR = (100, 200, 255)
CHAT_TEXT_COLOR = (200, 200, 200)
CHAT_ACTION_COLOR = (100, 255, 100)
CHAT_FONT_SIZE = 20
CHAT_LINE_HEIGHT = 23
CHAT_MAX_LINES_PER_SECTION = 10
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


def _render_chat_panel(step_data: dict, height: int, agent_name: str = "", agent_idx: int = 0) -> Image.Image:
    """Render a text panel showing agent input (observation) and output (reasoning/action)."""
    panel = Image.new("RGB", (CHAT_PANEL_WIDTH, height), color=CHAT_PANEL_BG)
    draw = ImageDraw.Draw(panel)
    font = _load_font(CHAT_FONT_SIZE)
    font_bold = _load_font(CHAT_FONT_SIZE + 1)
    font_header = _load_font(CHAT_FONT_SIZE + 3)
    char_width = font.getlength("x")
    chars_per_line = max(1, int((CHAT_PANEL_WIDTH - CHAT_PADDING * 2) // char_width))

    y = CHAT_PADDING

    # Draw agent name and role header
    role = SPECIALIZATION.get(agent_idx, "Unknown")
    header_text = f"{agent_name}  |  {role}"
    draw.text((CHAT_PADDING, y), header_text, font=font_header, fill=CHAT_TITLE_COLOR)
    y += CHAT_FONT_SIZE + 6
    draw.line([(CHAT_PADDING, y), (CHAT_PANEL_WIDTH - CHAT_PADDING, y)], fill=CHAT_DIVIDER_COLOR, width=1)
    y += 8

    def draw_section(title, text, text_color=CHAT_TEXT_COLOR, max_lines=CHAT_MAX_LINES_PER_SECTION):
        nonlocal y
        if y + CHAT_LINE_HEIGHT >= height - CHAT_PADDING:
            return
        draw.text((CHAT_PADDING, y), title, font=font_bold, fill=CHAT_TITLE_COLOR)
        y += CHAT_LINE_HEIGHT
        if text:
            lines = []
            for paragraph in str(text).split("\n"):
                lines.extend(textwrap.wrap(paragraph, width=chars_per_line) or [""])
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

    # if "observation" in step_data.keys():

    #     draw_section("INPUT (OBSERVATION):", step_data.get("observation", ""), max_lines=30)
    #     draw_divider()
    if "dialogue_reply" in step_data.keys():
        draw_section("INITIATING DIALOGUE :", step_data.get("dialogue_reply", ""), max_lines=30)
        # draw_divider()
    else:
        draw_section("OBJECTIVE:", step_data.get("objective", ""), max_lines=20)
        draw_section("OBJECTIVE REASONING:", step_data.get("objective_reasoning", ""))
        draw_section("PLAN:", step_data.get("plan", ""), max_lines=3)
        draw_section("PLAN REASONING:", step_data.get("plan_reasoning", ""),max_lines=20)
        draw_divider()
        draw_section("ACTION:", step_data.get("actions", ""), text_color=CHAT_ACTION_COLOR, max_lines=1)
        if step_data.get("objective", "") is None and step_data.get("objective_completion", False):
            draw_divider()
            draw_section("READY TO INITIATE DIALOGUE, WAITING...."," ", text_color=CHAT_ACTION_COLOR, max_lines=2)

    return panel


def _render_info_panel(info_text: str, width: int, height: int, step_idx: int = 0, n_steps: int = 0) -> Image.Image:
    """Render a text panel showing custom information for unused grid cells."""
    panel = Image.new("RGB", (width, height), color=(20, 20, 20))
    draw = ImageDraw.Draw(panel)
    font = _load_font(CHAT_FONT_SIZE)
    title_font = _load_font(CHAT_FONT_SIZE + 4)

    padding = 20
    
    y = padding
    
    # Draw title
    draw.text((padding, y), "Episode Info", font=title_font, fill=CHAT_TITLE_COLOR)
    y += 40
    
    # Draw step counter if provided
    if n_steps > 0:
        step_text = f"Step: {step_idx + 1} / {n_steps}"
        draw.text((padding, y), step_text, font=font, fill=(150, 150, 255))
        y += 35
        
        # Draw progress bar
        bar_width = width - padding * 2
        bar_height = 20
        progress = (step_idx + 1) / n_steps
        draw.rectangle(
            [(padding, y), (padding + bar_width, y + bar_height)],
            outline=(80, 80, 80),
            width=2
        )
        draw.rectangle(
            [(padding + 2, y + 2), (padding + int(bar_width * progress) - 2, y + bar_height - 2)],
            fill=(100, 200, 100)
        )
        y += bar_height + 30
    
    # Draw info text fields, each separated by a divider
    if info_text:
        fields = str(info_text).split("\n")
        for field in fields:
            if y + CHAT_LINE_HEIGHT >= height - padding:
                break
            if ":" in field:
                label, _, value = field.partition(":")
                label_text = label.strip() + ":"
                value_text = value.strip()
                draw.text((padding, y), label_text, font=title_font, fill=CHAT_TITLE_COLOR)
                label_w = title_font.getlength(label_text + " ")
                draw.text((padding + int(label_w), y), value_text, font=font, fill=CHAT_TEXT_COLOR)
            else:
                draw.text((padding, y), field, font=font, fill=CHAT_TEXT_COLOR)
            y += CHAT_LINE_HEIGHT + 6
            draw.line([(padding, y), (width - padding, y)], fill=CHAT_DIVIDER_COLOR, width=1)
            y += 8
    
    return panel


def save_episode_video(img_list, episode_dict=None, filepath=None, info_text=None):
    agent_names = list(img_list.keys())
    n_agents = len(agent_names)
    n_cols = min(n_agents, 2)
    n_rows = (n_agents + 1) // 2

    # Build per-agent composite frames indexed by step
    agent_frames: dict[str, list[Image.Image]] = {}
    for agent_idx, agent_name in enumerate(agent_names):
        frames = []
        for step_idx, img in enumerate(img_list[agent_name]):
            if episode_dict is not None and step_idx in episode_dict:
                step_data = episode_dict[step_idx].get(agent_name, {})
                chat_panel = _render_chat_panel(step_data, img.height, agent_name=agent_name, agent_idx=agent_idx)
                composite = Image.new("RGB", (img.width + CHAT_PANEL_WIDTH, img.height))
                composite.paste(img, (0, 0))
                composite.paste(chat_panel, (img.width, 0))
                frames.append(composite)
            else:
                frames.append(img)
        agent_frames[agent_name] = frames

    n_steps = max(len(f) for f in agent_frames.values())
    cell_w = agent_frames[agent_names[0]][0].width
    cell_h = agent_frames[agent_names[0]][0].height

    # Create info panel for unused cells (if any)
    total_cells = n_cols * n_rows
    info_panel = None
    if n_agents < total_cells and info_text is not None:
        info_panel = _render_info_panel(info_text, cell_w, cell_h, step_idx=0, n_steps=n_steps)

    merged_frames = []
    for step_idx in range(n_steps):
        grid = Image.new("RGB", (cell_w * n_cols, cell_h * n_rows), color=(0, 0, 0))
        for i, agent_name in enumerate(agent_names):
            frames = agent_frames[agent_name]
            frame = frames[step_idx] if step_idx < len(frames) else frames[-1]
            col = i % n_cols
            row = i // n_cols
            grid.paste(frame, (col * cell_w, row * cell_h))
        
        # Fill unused cells with info panel
        if info_panel is not None:
            for i in range(n_agents, total_cells):
                col = i % n_cols
                row = i // n_cols
                # Update step counter if dynamic info is needed
                dynamic_panel = _render_info_panel(info_text, cell_w, cell_h, step_idx, n_steps)
                grid.paste(dynamic_panel, (col * cell_w, row * cell_h))
        
        merged_frames.append(grid)

    imageio.mimsave(
        filepath + "merged_episode_video.mp4",
        [np.array(f) for f in merged_frames],
        fps=0.33,  # 3 seconds per frame
        codec="libx264",
        quality=8,
    )

def run_decentralized_planning(obs, agents, config, episode_dict, joint_step, img_list, obs_pixels):
    print("=========== Inititating Multi-Agent Dialogue ===================")
    for aid,agent_name in enumerate(agents.keys()):
        instr_prompt = get_dialogue_system_prompt(aid)
        agents[agent_name].dialogue_reset()
        agents[agent_name].dialogue_prompt_builder.update_dialogue_instruction_prompt(instr_prompt)
        agents[agent_name].dialogue_prompt_builder.update_observation(obs[agent_name])
        # episode_dict[joint_step][agent_name] = {}

    dialogue_order = list(agents.keys())
    agent_inds = {agent_name: idx for idx, agent_name in enumerate(agents.keys())}
    np.random.shuffle(dialogue_order)
    


    for round in range(config.dialogue_rounds):
        print(f"===== Dialogue Round {round + 1}======")

        episode_dict[joint_step] = {}
        
        for agent_name in dialogue_order:
            print(f"=====  {agent_name}======")
            obs_agent = obs[agent_name]["short_term_context"] + "\n" + obs[agent_name]["long_term_context"]
            
            episode_dict[joint_step][agent_name] = {
                "observation": obs_agent,

                # "actions" : actions_dict[agent_name],
                # "achievements": ", ".join(env.achievement_by_agent[agent_name]),
                # "reasoning": None
            }

            if config.human_render:
                img = Image.fromarray((np.array(obs_pixels)[agent_inds[agent_name]] * 255).astype(np.uint8))

                img_list[agent_name].append(img)


            dialogue_info, dialogue_answer = agents[agent_name].dialogue_act(round)

            try:
                dialogue_json = json.loads(dialogue_answer.completion)
                dialogue_text = "\n".join(f"{k}: {v}" for k, v in dialogue_json.items())
            except (json.JSONDecodeError, AttributeError, TypeError):
                dialogue_text = dialogue_answer.completion
            episode_dict[joint_step][agent_name]['dialogue_reply'] = dialogue_text

            print(f"Dialogue Reply: {dialogue_answer.completion}")

            for other_agent_name in agents.keys():
                if other_agent_name != agent_name:
                    agents[other_agent_name].update_dialogue(dialogue_info, agent_name, round)
        joint_step += 1

    return dialogue_info, episode_dict, img_list


def run_episode(env, agents, config):

    obs, env_state = env.reset()
    for agent_name in env.env.agents:
        agents[agent_name].reset()
    episode_dict = {}
    actions_dict = {}
    final_plan = None
    agent_objectives = {agent_name: None for agent_name in env.agents}
    # actions_dict_verb = {}
    # invalid_act = [False for _ in env.env.agents]
    # episode_stats = {}
    if config.human_render:
        static_env_params = StaticEnvParams()
        player_specific_textures = load_player_specific_textures(
            TEXTURES[PIXEL_SIZE],
            static_env_params.player_count
        )

    img_list = {agent_name: [] for agent_name in env.env.agents}

    ready = {agent_name: True for agent_name in env.env.agents}
    joint_step = 0
    

    for step in range(config.max_episode_steps):
        print("Voting", ready)
        print(f"\n=== STEP {step} ===")
        
        
        if config.human_render:
            obs_pixels = render_craftax_pixels(
                    env_state, 
                    PIXEL_SIZE, 
                    static_env_params,
                    player_specific_textures
                ) / 255
        else:
            obs_pixels = None
        
        majority_vote = sum(ready.values()) > len(ready) / 2
        
        if majority_vote:
            final_plan, episode_dict, img_list = run_decentralized_planning(obs, agents, config, episode_dict, joint_step, img_list, obs_pixels)
            joint_step += config.dialogue_rounds
            agent_objectives = final_plan['agent_objectives']
            for agent_name in env.env.agents:
                agents[agent_name].reset()
                ready[agent_name] = False

        # exit()
        print(f"Agent Objectives: {agent_objectives}")
        print(f"Final Plan: {final_plan}")
        actions_dict = {}
        # actions_dict_verb = {}
        episode_dict[joint_step] = {}
        # episode_stats[step] = {}
        episode_dict[joint_step]["stats"] = {}
        invalid_act = {agent_name: False for agent_name in env.env.agents}
        
        # if config.human_render:
        #     obs_pixels = render_craftax_pixels(
        #             env_state, 
        #             PIXEL_SIZE, 
        #             static_env_params,
        #             player_specific_textures
        #         ) / 255
        
        
        for ind, agent_name in enumerate(env.env.agents):
            actions_dict[agent_name] = None  # Initialize with None to indicate no action taken yet
            
            obs_agent = obs[agent_name]["short_term_context"] + "\n" + obs[agent_name]["long_term_context"]
            
            episode_dict[joint_step][agent_name] = {
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
                    future = executor.submit(agents[agent_name].act, obs[agent_name], step, objective = agent_objectives[agent_name])
                    futures[future] = agent_name
                
                for future in as_completed(futures):
                    agent_name = futures[future]
                    try:
                        action, response = future.result()
                        actions_dict[agent_name] = action
                        # actions_dict_verb[agent_name] = answer.completion
                        if response is not None:
                            episode_dict[joint_step][agent_name]['objective'] = response["objective"]
                            episode_dict[joint_step][agent_name]['plan'] = str(response["plan"])
                            episode_dict[joint_step][agent_name]['objective_reasoning'] = response["reasoning"]['objective_reasoning']
                            episode_dict[joint_step][agent_name]['plan_reasoning'] = response["reasoning"]['plan_reasoning']
                            episode_dict[joint_step][agent_name]['objective_completion'] = response["objective_completion"]
                            ready[agent_name] = response["objective_completion"]
                        else:
                            episode_dict[joint_step][agent_name]['objective'] = episode_dict[joint_step - 1][agent_name]['objective']
                            episode_dict[joint_step][agent_name]['plan'] = episode_dict[joint_step - 1][agent_name]['plan']
                            episode_dict[joint_step][agent_name]['objective_reasoning'] = episode_dict[joint_step - 1][agent_name]['objective_reasoning']
                            episode_dict[joint_step][agent_name]['plan_reasoning'] = episode_dict[joint_step - 1][agent_name]['plan_reasoning']
                            episode_dict[joint_step][agent_name]['objective_completion'] = episode_dict[joint_step - 1][agent_name]['objective_completion']
                    except Exception as e:
                        print(f"[Agent {agent_name}] Error: {e}")
                        actions_dict[agent_name] = 0
                        # actions_dict_verb[agent_name] = "NOOP"
        
        else:
            for agent_name in env.env.agents:
                action, response  = agents[agent_name].act(obs[agent_name], step,  objective = agent_objectives[agent_name])
                if response is not None:
                    episode_dict[joint_step][agent_name]['objective'] = response["objective"]
                    if response["objective"] != agent_objectives[agent_name]:
                        agent_objectives[agent_name] = response["objective"]
                    episode_dict[joint_step][agent_name]['plan'] = str(response["plan"])
                    if response["reasoning"]['objective_reasoning'] is not None:
                        episode_dict[joint_step][agent_name]['objective_reasoning'] = response["reasoning"]['objective_reasoning'].reasoning
                    else:
                        episode_dict[joint_step][agent_name]['objective_reasoning'] = None
                    if response["reasoning"]['plan_reasoning'] is not None:
                        episode_dict[joint_step][agent_name]['plan_reasoning'] = response["reasoning"]['plan_reasoning'].reasoning
                    else:
                        episode_dict[joint_step][agent_name]['plan_reasoning'] = None
                    episode_dict[joint_step][agent_name]['objective_completion'] = response["objective_completion"]
                    # print("Objective Reasoning: ",episode_dict[step][agent_name]['objective_reasoning'])
                    # print("Plan Reasoning: ",episode_dict[step][agent_name]['plan_reasoning'])
                    # print("Plan :", episode_dict[step][agent_name]['plan'])
                    ready[agent_name] = response["objective_completion"]
                else:
                    if "objective" in episode_dict[joint_step - 1][agent_name].keys():
                        episode_dict[joint_step][agent_name]['objective'] = episode_dict[joint_step - 1][agent_name]['objective']
                        episode_dict[joint_step][agent_name]['plan'] = episode_dict[joint_step - 1][agent_name]['plan']
                        episode_dict[joint_step][agent_name]['objective_reasoning'] = episode_dict[joint_step - 1][agent_name]['objective_reasoning']
                        episode_dict[joint_step][agent_name]['plan_reasoning'] = episode_dict[joint_step - 1][agent_name]['plan_reasoning']
                        episode_dict[joint_step][agent_name]['objective_completion'] = episode_dict[joint_step - 1][agent_name]['objective_completion']
                    else:
                        episode_dict[joint_step][agent_name]['objective'] = agent_objectives[agent_name]
                        episode_dict[joint_step][agent_name]['plan'] = None
                        episode_dict[joint_step][agent_name]['objective_reasoning'] = None
                        episode_dict[joint_step][agent_name]['plan_reasoning'] = None
                        episode_dict[joint_step][agent_name]['objective_completion'] = None
                
                act_verb = None
                for act in ACTIONS.keys():
                    if ACTIONS[act] == action:
                        act_verb = act
                        break
                episode_dict[joint_step][agent_name]['actions'] = act_verb 
                # answer, action_verb, action = agents[agent_name].act(obs[agent_name], step)
                # episode_dict[step][agent_name]['reasoning'] = answer.reasoning
                actions_dict[agent_name] = action
                # ready[agent_name] = response["objective_completion"] if response is not None else False
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
        # joint_step += 1

        # print(infos)
        
        for agent_name in env.env.agents:
            # episode_dict[step][agent_name]["actions"] = actions_dict_verb[agent_name]
            # episode_dict[step][agent_name]["reasoning"] = rewards[agent_name]
            episode_dict[joint_step][agent_name]["achievements"] = ", ".join(env.achievement_by_agent[agent_name])
            print(f"\n==={agent_name} ===")
            # print(f"Input No of Tokens: {answer.input_tokens}")
            # print(f"Output No of Tokens: {answer.output_tokens}")
            print(f"Observation: {episode_dict[joint_step][agent_name]['observation']}")
            # print(f"Reasoning: {episode_dict[step][agent_name]['reasoning']}")
            print("Objective Reasoning: ",episode_dict[joint_step][agent_name]['objective_reasoning'])
            print("Plan Reasoning: ",episode_dict[joint_step][agent_name]['plan_reasoning'])
            print("Plan :", episode_dict[joint_step][agent_name]['plan'])
            print(f"Action taken: {episode_dict[joint_step][agent_name]['actions']}")
        stats = env.get_achievements()
        episode_dict[joint_step]["stats"]["score"] =int(stats["score"])
        episode_dict[joint_step]["stats"]["achievements"] = stats["achievements"]
        print(f"\n=== Environment Stats ===")
        print(f"Score: {stats['score']}")
        print(rewards)
        achivements = [ach for ach in stats["achievements"].keys() if stats["achievements"][ach] != 0]
        print(f"Achievements: {' '.join(achivements)}")
        joint_step += 1
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
    env_ = make_craftax_env_from_name(config.env_name)
    parser = CraftaxObservationParser(num_agents=len(env_.agents))
    rng = jax.random.PRNGKey(config.seed)
    env = CraftaxLanguageWrapper(env_, parser, rng)
    # obs, env_state = env.reset(rng)
    LLM_client = LLMClientWrapper(config)
    LLM_client._initialize_client()

    # if config.agent_type == "cot":
    #     AgentClass = CoTAgent
    # elif config.agent_type == "naive":
    #     AgentClass = NaiveAgent
    # elif config.agent_type == "planner":
    #     AgentClass = PlannerAgent
    # elif config.agent_type == "dialogue":    
    #     AgentClass = PlannerAgentMA
    # else:
        # raise ValueError(f"Unknown agent type: {config.agent_type}")
    agents = {agent_name: PlannerAgentMA(agent_name, LLM_client, HistoryandPlansPromptBuilderv2(config), DialogueHistoryPromptBuilder(config,agent_name)) for agent_name in env.agents}


    for aid, agent_name in enumerate(env.agents):
        
        system_prompt = get_system_prompt(aid)
        agents[agent_name].prompt_builder.update_instruction_prompt(system_prompt)
        agents[agent_name].dialogue_prompt_builder.update_instruction_prompt(system_prompt)

    episode_dict, img_lst = run_episode(env, agents, config)
    # print(episode_dict)
    os.mkdir(os.path.join(log_path, f"{run_name}_craftax_ma_{config.num_agents}agents/"))
    log_episode(episode_dict, filepath=os.path.join(log_path, f"{run_name}_craftax_ma_{config.num_agents}agents/"))
    if config.human_render:
        save_episode_video(img_lst, episode_dict=episode_dict, filepath=os.path.join(log_path, f"{run_name}_craftax_ma_{config.num_agents}agents/"), info_text=f"Environment: {config.env_name}\nSeed: {config.seed}\nModel: {config.model_id}\nAgent Type: {config.agent_type}\nNumber of Agents: {config.num_agents}")

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