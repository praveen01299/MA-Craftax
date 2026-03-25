import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import hydra
import time
from datetime import datetime
from omegaconf import DictConfig
from hydra.utils import instantiate, get_original_cwd
# import omegaconf
# from omegaconf import open_dict
# from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from craftax.craftax_env import make_craftax_env_from_name
from env_wrapper import CraftaxLanguageWrapper  
from craftax_sym_obs_parser import CraftaxObservationParser
from llm_client import LLMClientWrapper
from prompt_builder import HistoryPromptBuilder
from craftax_agents import CoTAgent, NaiveAgent
from prompts import get_system_prompt



def log_episode(stats, filename="episode_stats.json"):
    with open(filename, "w") as f:
        json.dump(stats, f, indent=4)

def run_episode(env, agents, config):

    obs, env_state = env.reset()
    for agent_name in env.env.agents:
        agents[agent_name].reset()
    episode_dict = {}
    actions_dict = {}
    actions_dict_verb = {}
    # invalid_act = [False for _ in env.env.agents]
    # episode_stats = {}

    for step in range(config.max_episode_steps):
        print(f"\n=== STEP {step} ===")
        # actions_dict = {}
        # actions_dict_verb = {}
        episode_dict[step] = {}
        # episode_stats[step] = {}
        episode_dict[step]["stats"] = {}
        invalid_act = {agent_name: False for agent_name in env.env.agents}
        for agent_name in env.env.agents:
            actions_dict[agent_name] = None 
            actions_dict[agent_name] = None  # Initialize with None to indicate no action taken yet
            
            obs_agent = obs[agent_name]["short_term_context"] + "\n" + obs[agent_name]["long_term_context"]
            
            episode_dict[step][agent_name] = {
                "observation": obs_agent,
                # "actions" : actions_dict[agent_name],
                "achievements": ", ".join(env.achievement_by_agent[agent_name]),
                "reasoning": None
            }
            
        if config.parallel_execution:
            with ThreadPoolExecutor(max_workers=len(env.agents)) as executor:
                futures = {}
                for agent_name in env.env.agents:
                    future = executor.submit(agents[agent_name].act, obs[agent_name], prev_action=actions_dict_verb.get(agent_name))
                    futures[future] = agent_name
                
                for future in as_completed(futures):
                    agent_name = futures[future]
                    try:
                        answer, action = future.result()
                        actions_dict[agent_name] = action
                        actions_dict_verb[agent_name] = answer.completion
                    except Exception as e:
                        print(f"[Agent {agent_name}] Error: {e}")
                        actions_dict[agent_name] = 0
                        actions_dict_verb[agent_name] = "NOOP"
        
        else:
            for agent_name in env.env.agents:
                answer, action_verb, action = agents[agent_name].act(obs[agent_name], prev_action=actions_dict_verb.get(agent_name))
                episode_dict[step][agent_name]['reasoning'] = answer.reasoning
                actions_dict[agent_name] = action
                actions_dict_verb[agent_name] = action_verb
                if action_verb != answer.completion and config.feedback_on_invalid_action:
                    invalid_act[agent_name] = True

            # print(f"Action for {agent_name}: {action}")

        # print(actions_dict)
        # exit()
        
        obs, env_state, rewards, dones, infos = env.step(actions_dict, env_state)

        for agent_name in env.env.agents:
            obs[agent_name]["long_term_context"] = (
                    f"\n\nYour previous output did not contain a valid action. Defaulted to action: {action_verb}\n"
                    + obs[agent_name]["long_term_context"]
                    if invalid_act[agent_name]
                    else obs[agent_name]["long_term_context"]
                )

        env.update_progress(infos, rewards)

        # print(infos)
        
        for agent_name in env.env.agents:
            episode_dict[step][agent_name]["actions"] = actions_dict_verb[agent_name]
            # episode_dict[step][agent_name]["reasoning"] = rewards[agent_name]
            episode_dict[step][agent_name]["achievements"] = ", ".join(env.achievement_by_agent[agent_name])
            print(f"\n==={agent_name} ===")
            print(f"Input No of Tokens: {answer.input_tokens}")
            print(f"Output No of Tokens: {answer.output_tokens}")
            print(f"Observation: {episode_dict[step][agent_name]['observation']}")
            print(f"Reasoning: {episode_dict[step][agent_name]['reasoning']}")
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

    return episode_dict#, episode_stats




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

    if config.agent_type == "cot":
        AgentClass = CoTAgent
    elif config.agent_type == "naive":
        AgentClass = NaiveAgent
    else:
        raise ValueError(f"Unknown agent type: {config.agent_type}")
    agents = {agent_name: AgentClass(agent_name, LLM_client, HistoryPromptBuilder(config)) for agent_name in env.agents}


    for aid, agent_name in enumerate(env.agents):
        
        system_prompt = get_system_prompt(aid)
        agents[agent_name].prompt_builder.update_instruction_prompt(system_prompt)

    episode_dict = run_episode(env, agents, config)
    # print(episode_dict)
    log_episode(episode_dict, filename=os.path.join(log_path, f"{run_name}_episode_stats.json"))

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