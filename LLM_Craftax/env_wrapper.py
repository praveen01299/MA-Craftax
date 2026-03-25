import sys
import os
import jax
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

class CraftaxLanguageWrapper:
    def __init__(self, env, parser, rng):
        self.env = env
        self.parser = parser
        self.rng = rng
        self.achievements = None
        self.achievement_by_agent = {agent: set() for agent in self.env.agents}
        self.score_tracker = 0
        self.agents = self.env.agents
    
    def reset(self):
        obs, env_state = self.env.reset(self.rng)
        textual_obs = {}
        for id, agent_name in enumerate(self.env.agents):
            parsed = self.parser.parse_observation(obs[agent_name], agent_id=id)
            parsed = self.parser.to_text(parsed, id)
            parsed_dict = {
                "long_term_context": parsed[1],
                "short_term_context": parsed[0],
            }
            textual_obs[agent_name] = parsed_dict
        
        return textual_obs, env_state
    
    def step(self, actions: Dict[str, str], env_state):
        self.rng, step_rng = jax.random.split(self.rng)
        obs, env_state, rewards, dones, infos = self.env.step(step_rng, env_state, actions)
        textual_obs = {}
        for id, agent_name in enumerate(self.env.agents):
            parsed = self.parser.parse_observation(obs[agent_name], agent_id=id)
            
            parsed = self.parser.to_text(parsed, id)
            parsed_dict = {
                "long_term_context": parsed[1],
                "short_term_context": parsed[0],
            }
            textual_obs[agent_name] = parsed_dict

        return textual_obs, env_state, rewards, dones, infos
    
    def update_progress(self, info, rewards):
        # Extract achievements from infos and update self.achievements
        if self.achievements is None:
           self.achievements = {} 
        user_info = info.get("user_info", {})
        # for agent in self.env.agents:
        # self.score_tracker += rewards["agent_0"]
        for ind,agent in enumerate(self.env.agents):
            self.score_tracker += rewards[agent] 
        for achievement in user_info.keys():
            if "Achievements/" in achievement:
                ach_name = achievement.split("Achievements/")[1]
                self.achievements[ach_name] = int(user_info[achievement].sum())
                # self.score_tracker += rewards.sum() 
                for ind,agent in enumerate(self.env.agents):
                    # self.score_tracker += rewards[agent] 
                    if user_info[achievement].at[ind].get() > 0:
                        self.achievement_by_agent[agent].add(ach_name)

    def get_achievements(self):
        return {"score": self.score_tracker, "achievements": self.achievements, "achievement_by_agent": self.achievement_by_agent}
    
