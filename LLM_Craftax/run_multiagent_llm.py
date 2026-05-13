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
from craftax.craftax_env import make_craftax_env_from_name
from LLM_Craftax.craftax_coop_sym_obs_parser import CraftaxObservationParser

from openai import OpenAI

# Read API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in the environment")

client = OpenAI(api_key=OPENAI_API_KEY)

# Setup
env_name = "Craftax-Coop-Symbolic"
env = make_craftax_env_from_name(env_name)
rng = jax.random.PRNGKey(0)
obs, env_state = env.reset(rng)
parser = CraftaxObservationParser(num_agents=len(env.agents))

# Action mappings
ACTIONS = {
    'NOOP': 0, 'LEFT': 1, 'RIGHT': 2, 'UP': 3, 'DOWN': 4, 'DO': 5, 'SLEEP': 6,
    'PLACE_STONE': 7, 'PLACE_TABLE': 8, 'PLACE_FURNACE': 9, 'PLACE_PLANT': 10,
    'MAKE_WOOD_PICKAXE': 11, 'MAKE_STONE_PICKAXE': 12, 'MAKE_IRON_PICKAXE': 13,
    'MAKE_WOOD_SWORD': 14, 'MAKE_STONE_SWORD': 15, 'MAKE_IRON_SWORD': 16,
    'REST': 17, 'DESCEND': 18, 'ASCEND': 19, 'MAKE_DIAMOND_PICKAXE': 20,
    'MAKE_DIAMOND_SWORD': 21, 'MAKE_IRON_ARMOUR': 22, 'MAKE_DIAMOND_ARMOUR': 23,
    'SHOOT_ARROW': 24, 'MAKE_ARROW': 25, 'CAST_SPELL': 26, 'PLACE_TORCH': 28,
    'DRINK_POTION_RED': 29, 'DRINK_POTION_GREEN': 30, 'DRINK_POTION_BLUE': 31,
    'DRINK_POTION_PINK': 32, 'DRINK_POTION_CYAN': 33, 'DRINK_POTION_YELLOW': 34,
    'READ_BOOK': 35, 'ENCHANT_SWORD': 36, 'ENCHANT_ARMOUR': 37, 'MAKE_TORCH': 38,
    'LEVEL_UP_DEXTERITY': 39, 'LEVEL_UP_STRENGTH': 40, 'LEVEL_UP_INTELLIGENCE': 41,
    'ENCHANT_BOW': 42, 'REQUEST_FOOD': 43, 'REQUEST_DRINK': 44, 'REQUEST_WOOD': 45,
    'REQUEST_STONE': 46, 'REQUEST_IRON': 47, 'REQUEST_COAL': 48,
    'REQUEST_DIAMOND': 49, 'REQUEST_RUBY': 50, 'REQUEST_SAPPHIRE': 51, 'GIVE': 52
}

ACTION_NAMES = {v: k for k, v in ACTIONS.items()}

STEP = 1
cumulative_rewards = {agent: 0.0 for agent in env.agents}
llm_call_count = 0
total_llm_time = 0.0

# Agent plans
agent_plans = {
    0: {'plan': [], 'target': None, 'target_coords': None, 'step_in_plan': 0},
    1: {'plan': [], 'target': None, 'target_coords': None, 'step_in_plan': 0},
    2: {'plan': [], 'target': None, 'target_coords': None, 'step_in_plan': 0}
}


class PathPlanner:
    """Plans movement sequences to reach targets"""
    
    @staticmethod
    def plan_path_to_coords(target_coords: Tuple[int, int]) -> List[str]:
        """
        Plan path to move TO target coordinates (not adjacent).
        This guarantees the agent ends up next to the target.
        
        Args:
            target_coords: (row_offset, col_offset) relative to current position
        
        Returns:
            List of movement actions (UP, DOWN, LEFT, RIGHT)
        """
        rel_row, rel_col = target_coords
        plan = []
        
        # Move vertically first
        if rel_row < 0:  # NORTH
            for _ in range(abs(rel_row)):
                plan.append('UP')
        elif rel_row > 0:  # SOUTH
            for _ in range(rel_row):
                plan.append('DOWN')
        
        # Then horizontally
        if rel_col < 0:  # WEST
            for _ in range(abs(rel_col)):
                plan.append('LEFT')
        elif rel_col > 0:  # EAST
            for _ in range(rel_col):
                plan.append('RIGHT')
        
        return plan


def get_agent_prompt(agent_id: int, agent_obs: str, step_num: int) -> str:
    specializations = {
        0: "Warrior (combat specialist, can craft advanced swords)",
        1: "Forager (gathers food/water, hunts passive mobs)",
        2: "Miner (mines resources, crafts pickaxes and torches)"
    }
    
    prompt = f"""You are Agent {agent_id}, a {specializations[agent_id]} in Craftax-Coop.

CURRENT OBSERVATION:
{agent_obs}

RESPONSE FORMAT - Choose ONE:

1. MOVE TO TARGET:
{{
  "agent_id": {agent_id},
  "type": "move_to",
  "target": "tree",
  "target_coords": [-3, -1],
  "reasoning": "Moving to tree to chop wood"
}}

2. IMMEDIATE ACTION:
{{
  "agent_id": {agent_id},
  "type": "action",
  "action": "MAKE_WOOD_PICKAXE",
  "reasoning": "Crafting pickaxe"
}}

VALID ACTIONS:
Interaction: DO (chop trees, mine stone, drink water, attack mobs)
Crafting: MAKE_WOOD_PICKAXE, MAKE_STONE_PICKAXE, MAKE_WOOD_SWORD, MAKE_STONE_SWORD, PLACE_TABLE, PLACE_FURNACE
Trading: REQUEST_WOOD, REQUEST_STONE, REQUEST_FOOD, REQUEST_DRINK, GIVE
Other: SLEEP, REST, NOOP

IMPORTANT:
- For "move_to": target_coords are [row, col] offsets from VISIBLE TILES (e.g. tree at "(-2, 1)" means 2 north, 1 east)
- After moving to target, you'll automatically use DO to interact
- **COORDINATE WITH TEAMMATES** - choose different trees/resources to avoid clustering
- Gather resources first, then craft tools when you have materials

Return ONLY valid JSON, no extra text.
"""
    return prompt


def call_agent_llm(agent_id: int, agent_obs_text: str, step_num: int) -> dict:
    """Call OpenAI API for agent decision"""
    global llm_call_count, total_llm_time
    
    prompt = get_agent_prompt(agent_id, agent_obs_text, step_num)
    
    print(f"[Agent {agent_id}] Calling LLM...")
    
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": "You are a game-playing AI."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        
        elapsed = time.time() - start_time
        llm_call_count += 1
        total_llm_time += elapsed
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON from markdown
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        
        parsed = json.loads(content)
        print(f"[Agent {agent_id}] LLM ({elapsed:.2f}s): {parsed.get('type')} - {parsed.get('reasoning', '')[:50]}")
        
        return parsed
        
    except Exception as e:
        print(f"[Agent {agent_id}] ERROR: {e}")
        return {"agent_id": agent_id, "type": "action", "action": "NOOP"}


def call_agent_llm_parallel(agent_id: int, agent_obs_text: str, step_num: int) -> Tuple[int, dict]:
    response = call_agent_llm(agent_id, agent_obs_text, step_num)
    return agent_id, response


def get_next_planned_action(agent_id: int, parsed_obs: dict) -> Optional[str]:
    """Get next action from plan or None if plan complete"""
    plan_info = agent_plans[agent_id]
    
    if not plan_info['plan']:
        return None
    
    if plan_info['step_in_plan'] >= len(plan_info['plan']):
        # Plan complete
        print(f"[Agent {agent_id}] Plan completed")
        agent_plans[agent_id] = {'plan': [], 'target': None, 'target_coords': None, 'step_in_plan': 0}
        return None
    
    # Execute action (no visibility check - coords become invalid after movement)
    action = plan_info['plan'][plan_info['step_in_plan']]
    plan_info['step_in_plan'] += 1
    
    return action


def process_agent_decision(agent_id: int, llm_response: dict, parsed_obs: dict) -> Tuple[str, bool]:
    """Process LLM decision and return (action_name, has_active_plan)"""
    response_type = llm_response.get('type')
    
    if response_type == 'move_to':
        target_coords = tuple(llm_response['target_coords'])
        target_name = llm_response.get('target', 'unknown')
        
        # Create path directly TO the target coordinates
        movement_plan = PathPlanner.plan_path_to_coords(target_coords)
        
        # Add DO at the end to interact
        movement_plan.append('DO')
        
        # Store plan with target info
        agent_plans[agent_id] = {
            'plan': movement_plan,
            'target': target_name,
            'target_coords': target_coords,
            'step_in_plan': 0
        }
        
        if movement_plan:
            plan_str = ' -> '.join(movement_plan[:6])
            if len(movement_plan) > 6:
                plan_str += f'... ({len(movement_plan)} total)'
            print(f"[Agent {agent_id}] Plan to {target_name} at {target_coords}: {plan_str}")
        
        # Execute first action
        if movement_plan:
            agent_plans[agent_id]['step_in_plan'] = 1
            return movement_plan[0], True
        else:
            # Already at target - just DO
            return 'DO', False
    
    else:  # type == 'action' or default
        action = llm_response.get('action', 'NOOP')
        
        if action not in ACTIONS:
            print(f"[Agent {agent_id}] Invalid action '{action}', using NOOP")
            action = 'NOOP'
        
        agent_plans[agent_id] = {'plan': [], 'target': None, 'target_coords': None, 'step_in_plan': 0}
        return action, False


def get_actions_for_step(obs: dict, step_num: int, use_parallel: bool = True) -> Dict[str, int]:
    """Get actions for all agents"""
    actions_dict = {}
    agents_needing_llm = []
    
    for agent_id, agent_name in enumerate(env.agents):
        parsed = parser.parse_observation(obs[agent_name], agent_id=agent_id)
        
        # Try to use planned action
        planned_action = get_next_planned_action(agent_id, parsed)
        
        if planned_action is None:
            # Need new LLM decision
            agents_needing_llm.append((agent_id, agent_name, parsed))
        else:
            # Use planned action
            actions_dict[agent_name] = ACTIONS[planned_action]
            remaining = len(agent_plans[agent_id]['plan']) - agent_plans[agent_id]['step_in_plan']
            print(f"[Agent {agent_id}] Executing: {planned_action} ({remaining} steps left)")
    
    # Call LLMs for agents needing decisions
    if agents_needing_llm:
        print(f"\n{'='*70}")
        print(f"Requesting LLM decisions for {len(agents_needing_llm)} agents...")
        print('='*70)
        
        if use_parallel and len(agents_needing_llm) > 1:
            with ThreadPoolExecutor(max_workers=len(agents_needing_llm)) as executor:
                futures = {}
                for agent_id, agent_name, parsed in agents_needing_llm:
                    obs_text = parser.to_text(parsed, agent_id)
                    future = executor.submit(call_agent_llm_parallel, agent_id, obs_text, step_num)
                    futures[future] = (agent_id, agent_name, parsed)
                
                for future in as_completed(futures):
                    agent_id, agent_name, parsed = futures[future]
                    try:
                        _, llm_response = future.result()
                        action_name, has_plan = process_agent_decision(agent_id, llm_response, parsed)
                        actions_dict[agent_name] = ACTIONS[action_name]
                    except Exception as e:
                        print(f"[Agent {agent_id}] Error: {e}")
                        actions_dict[agent_name] = ACTIONS['NOOP']
        else:
            for agent_id, agent_name, parsed in agents_needing_llm:
                obs_text = parser.to_text(parsed, agent_id)
                llm_response = call_agent_llm(agent_id, obs_text, step_num)
                action_name, has_plan = process_agent_decision(agent_id, llm_response, parsed)
                actions_dict[agent_name] = ACTIONS[action_name]
    
    return actions_dict


def print_state_summary(env_state, step_num):
    print(f"\n{'='*70}")
    print(f"STEP {step_num}")
    print('='*70)
    print(f"Positions: {env_state.player_position}")
    print(f"Health: {env_state.player_health}")
    print(f"Wood: [{env_state.inventory.wood[0]}, {env_state.inventory.wood[1]}, {env_state.inventory.wood[2]}]")
    
    print("\nACTIVE PLANS:")
    for agent_id in range(len(env.agents)):
        plan_info = agent_plans[agent_id]
        if plan_info['plan']:
            remaining = len(plan_info['plan']) - plan_info['step_in_plan']
            print(f"  Agent {agent_id}: {plan_info['target']} at {plan_info['target_coords']} ({remaining} steps left)")
        else:
            print(f"  Agent {agent_id}: No active plan")
    
    print(f"\nLLM Stats: {llm_call_count} calls, {total_llm_time:.2f}s")
    print('='*70)


def run_episode(max_steps: int = 1000, use_parallel: bool = True):
    """Run episode"""
    global STEP, obs, env_state, rng, cumulative_rewards, llm_call_count, total_llm_time
    
    print(f"\n{'='*70}")
    print(f"STARTING EPISODE (max {max_steps} steps)")
    print('='*70)
    
    episode_start = time.time()
    
    while STEP <= max_steps:
        print_state_summary(env_state, STEP)
        
        if not env_state.player_alive.any():
            print("\n🔴 All agents died!")
            break
        
        try:
            actions_dict = get_actions_for_step(obs, STEP, use_parallel=use_parallel)
            
            print("\n" + "-"*70)
            print("ACTIONS:")
            for agent_name, action_id in actions_dict.items():
                agent_id = int(agent_name.split('_')[1])
                print(f"  Agent {agent_id}: {ACTION_NAMES[action_id]}")
            print("-"*70)
            
            # Step environment
            rng, step_rng = jax.random.split(rng)
            obs, env_state, rewards, dones, info = env.step(step_rng, env_state, actions_dict)
            
            # Print rewards
            total_reward = sum(rewards.values())
            print(f"\nRewards: {total_reward:.2f}")
            for agent_name, reward in rewards.items():
                cumulative_rewards[agent_name] += reward
                agent_id = int(agent_name.split('_')[1])
                print(f"  Agent {agent_id}: +{reward:.2f} (Total: {cumulative_rewards[agent_name]:.2f})")
            
            STEP += 1
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted")
            break
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            break
    
    episode_time = time.time() - episode_start
    
    print(f"\n{'='*70}")
    print("EPISODE COMPLETE")
    print('='*70)
    print(f"Steps: {STEP - 1}")
    print(f"Time: {episode_time:.2f}s")
    print(f"LLM calls: {llm_call_count} ({total_llm_time:.2f}s)")
    print(f"Final Rewards: {[float(cumulative_rewards[a]) for a in env.agents]}")
    print('='*70)


def main():
    global STEP, obs, env_state, rng, cumulative_rewards, llm_call_count, total_llm_time
    
    print("\n" + "="*70)
    print("CRAFTAX-COOP MULTI-AGENT LLM")
    print("="*70)
    print("Commands: run [N] | reset | quit")
    print("="*70)
    
    while True:
        try:
            user_input = input("\nCommand: ").strip().lower()
            
            if user_input in ['quit', 'exit', 'q']:
                break
            
            elif user_input == 'reset':
                rng = jax.random.PRNGKey(0)
                obs, env_state = env.reset(rng)
                STEP = 1
                for agent in cumulative_rewards:
                    cumulative_rewards[agent] = 0.0
                for agent_id in agent_plans:
                    agent_plans[agent_id] = {'plan': [], 'target': None, 'target_coords': None, 'step_in_plan': 0}
                llm_call_count = 0
                total_llm_time = 0.0
                print("✅ Reset!")
                continue
            
            elif user_input.startswith('run'):
                parts = user_input.split()
                max_steps = int(parts[1]) if len(parts) > 1 else 50
                run_episode(max_steps=max_steps, use_parallel=True)
            
            else:
                print("Unknown command")
        
        except KeyboardInterrupt:
            print("\n\nUse 'quit' to exit")
            continue
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()