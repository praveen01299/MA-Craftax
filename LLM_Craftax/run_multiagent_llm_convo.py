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
from craftax_sym_obs_parser import CraftaxObservationParser

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
    0: {'plan': [], 'target': None, 'target_coords': None, 'step_in_plan': 0, 'last_position': None, 'stuck_count': 0},
    1: {'plan': [], 'target': None, 'target_coords': None, 'step_in_plan': 0, 'last_position': None, 'stuck_count': 0},
    2: {'plan': [], 'target': None, 'target_coords': None, 'step_in_plan': 0, 'last_position': None, 'stuck_count': 0}
}

# Individual agent memory - each agent tracks infrastructure they've personally seen
agent_memory = {
    0: {'crafting_tables': [], 'furnaces': [], 'last_position': None},  # Each entry: {'rel_coords': (row, col), 'first_seen_step': N}
    1: {'crafting_tables': [], 'furnaces': [], 'last_position': None},
    2: {'crafting_tables': [], 'furnaces': [], 'last_position': None}
}

# Conversation history
conversation_history = []
last_broadcast_step = 0


def update_agent_memory(agent_id: int, parsed_obs: dict, env_state, step_num: int):
    """
    Update agent's personal memory of infrastructure.
    - Scans observation for new infrastructure
    - Updates relative coordinates based on movement
    """
    current_pos = tuple(env_state.player_position[agent_id])
    last_pos = agent_memory[agent_id]['last_position']
    
    # Calculate how much agent moved since last step
    if last_pos is not None:
        delta_row = current_pos[0] - last_pos[0]
        delta_col = current_pos[1] - last_pos[1]
        
        # Update all remembered infrastructure relative coords
        for table in agent_memory[agent_id]['crafting_tables']:
            # Adjust relative coords based on movement
            # If I moved RIGHT (+col), table is now more to my LEFT (negative col)
            table['rel_coords'] = (
                table['rel_coords'][0] - delta_row,
                table['rel_coords'][1] - delta_col
            )
        
        for furnace in agent_memory[agent_id]['furnaces']:
            furnace['rel_coords'] = (
                furnace['rel_coords'][0] - delta_row,
                furnace['rel_coords'][1] - delta_col
            )
    
    # Scan current observation for infrastructure
    obs_text = parser.to_text(parsed_obs, agent_id)
    lines = obs_text.split('\n')
    
    # DEBUG: Print observation text to see what we're parsing
    if step_num <= 20:  # Only debug first 20 steps
        has_table_keyword = any('table' in line.lower() or 'crafting' in line.lower() for line in lines)
        if has_table_keyword:
            print(f"[Agent {agent_id}] 🔍 DEBUG: Observation contains 'table' keyword:")
            for line in lines:
                if 'table' in line.lower() or 'crafting' in line.lower():
                    print(f"  {line}")
    
    for line in lines:
        # Look for crafting tables
        if 'crafting_table' in line.lower() or 'table' in line.lower():
            # Try to extract coordinates from line like "crafting_table at (-2, 3)"
            if 'at (' in line:
                try:
                    coords_str = line.split('at (')[1].split(')')[0]
                    row, col = map(int, coords_str.split(','))
                    
                    # Check if this table is already in memory (roughly same location)
                    is_new = True
                    for table in agent_memory[agent_id]['crafting_tables']:
                        # If within 1 tile, consider it the same table
                        if abs(table['rel_coords'][0] - row) <= 1 and abs(table['rel_coords'][1] - col) <= 1:
                            # Update coords to current observation (more accurate)
                            table['rel_coords'] = (row, col)
                            is_new = False
                            break
                    
                    if is_new:
                        agent_memory[agent_id]['crafting_tables'].append({
                            'rel_coords': (row, col),
                            'first_seen_step': step_num
                        })
                        print(f"[Agent {agent_id}] 👁️  First time seeing crafting table at {(row, col)}")
                
                except:
                    pass
        
        # Look for furnaces
        if 'furnace' in line.lower():
            if 'at (' in line:
                try:
                    coords_str = line.split('at (')[1].split(')')[0]
                    row, col = map(int, coords_str.split(','))
                    
                    is_new = True
                    for furnace in agent_memory[agent_id]['furnaces']:
                        if abs(furnace['rel_coords'][0] - row) <= 1 and abs(furnace['rel_coords'][1] - col) <= 1:
                            furnace['rel_coords'] = (row, col)
                            is_new = False
                            break
                    
                    if is_new:
                        agent_memory[agent_id]['furnaces'].append({
                            'rel_coords': (row, col),
                            'first_seen_step': step_num
                        })
                        print(f"[Agent {agent_id}] 👁️  First time seeing furnace at {(row, col)}")
                
                except:
                    pass
    
    # Update last known position
    agent_memory[agent_id]['last_position'] = current_pos


def get_agent_memory_context(agent_id: int) -> str:
    """Get context about infrastructure this agent has personally seen"""
    memory = agent_memory[agent_id]
    
    if not memory['crafting_tables'] and not memory['furnaces']:
        return "You haven't seen any crafting tables or furnaces yet."
    
    context_lines = ["YOUR PERSONAL INFRASTRUCTURE MEMORY (things you've seen):"]
    
    for table in memory['crafting_tables']:
        coords = table['rel_coords']
        # Generate path
        path = PathPlanner.plan_path_to_coords(coords)
        path_str = " -> ".join(path[:6])
        if len(path) > 6:
            path_str += f"... ({len(path)} steps total)"
        
        context_lines.append(
            f"- Crafting table at {coords}: Path = {path_str if path else 'AT YOUR LOCATION'}"
        )
    
    for furnace in memory['furnaces']:
        coords = furnace['rel_coords']
        path = PathPlanner.plan_path_to_coords(coords)
        path_str = " -> ".join(path[:6])
        if len(path) > 6:
            path_str += f"... ({len(path)} steps total)"
        
        context_lines.append(
            f"- Furnace at {coords}: Path = {path_str if path else 'AT YOUR LOCATION'}"
        )
    
    return "\n".join(context_lines)


def update_infrastructure_memory(env_state, step_num):
    """Update shared memory of infrastructure locations"""
    global infrastructure_memory
    
    # Check for crafting tables at each agent's position
    for agent_id in range(len(env.agents)):
        pos = tuple(env_state.player_position[agent_id])
        
        # Check if this position already has a table recorded
        table_exists = any(table[0] == pos for table in infrastructure_memory['crafting_tables'])
        
        if not table_exists:
            # Check if agent just placed a table (we'd need to track this via actions)
            # For now, we'll detect tables from observations
            pass


def get_infrastructure_context(agent_id: int, env_state) -> str:
    """Get context about known infrastructure relative to agent's current position"""
    if not infrastructure_memory['crafting_tables']:
        return "No crafting tables have been placed yet by the team."
    
    agent_pos = env_state.player_position[agent_id]
    context_lines = ["TEAM INFRASTRUCTURE MEMORY:"]
    
    for table_pos, placed_by, step in infrastructure_memory['crafting_tables']:
        # Calculate relative position from agent
        rel_row = table_pos[0] - agent_pos[0]
        rel_col = table_pos[1] - agent_pos[1]
        
        # Describe direction
        direction_parts = []
        if rel_row < 0:
            direction_parts.append(f"{abs(rel_row)} tiles NORTH")
        elif rel_row > 0:
            direction_parts.append(f"{rel_row} tiles SOUTH")
        
        if rel_col < 0:
            direction_parts.append(f"{abs(rel_col)} tiles WEST")
        elif rel_col > 0:
            direction_parts.append(f"{rel_col} tiles EAST")
        
        direction = " and ".join(direction_parts) if direction_parts else "at your current location"
        
        context_lines.append(
            f"- Crafting table (placed by Agent {placed_by} on step {step}): {direction} at coordinates ({rel_row}, {rel_col})"
        )
    
    return "\n".join(context_lines)


def should_trigger_broadcast(agent_id: int, action_name: str, reward: float, step_num: int) -> bool:
    """
    Determine if agent should broadcast an update to the team.
    Triggers when:
    - Agent does a non-movement action and gets positive reward
    - This indicates a successful action (chopped tree, placed table, crafted item, etc.)
    """
    global last_broadcast_step
    
    # Don't spam broadcasts - at least 2 steps between broadcasts
    if step_num - last_broadcast_step < 2:
        return False
    
    non_movement_actions = ['DO', 'PLACE_TABLE', 'PLACE_FURNACE', 'MAKE_WOOD_PICKAXE', 
                           'MAKE_WOOD_SWORD', 'MAKE_STONE_PICKAXE', 'MAKE_STONE_SWORD',
                           'PLACE_STONE', 'SLEEP', 'REST']
    
    return action_name in non_movement_actions and reward > 0


def get_broadcast_prompt(agent_id: int, agent_obs: str, action_taken: str, reward: float) -> str:
    """Get prompt for agent to broadcast what they just accomplished"""
    specializations = {
        0: "Warrior",
        1: "Forager", 
        2: "Miner"
    }
    
    prompt = f"""You are Agent {agent_id}, a {specializations[agent_id]} in Craftax-Coop.

You just completed action: {action_taken} and received reward: {reward:.2f}

YOUR CURRENT STATE:
{agent_obs}

BROADCAST TO TEAM:
Tell your teammates EXACTLY what you just did and your current state. Be ACCURATE - only state facts from your observation.

RESPONSE FORMAT:
{{
  "agent_id": {agent_id},
  "broadcast": "I just chopped a tree. I now have 1 wood. I'm in the northeast area and will continue gathering.",
  "next_action_intent": "continue chopping trees in my region"
}}

CRITICAL:
- ONLY state facts from your observation above
- Mention your CURRENT wood count from the observation
- Do NOT make assumptions about what others are doing
- Do NOT hallucinate information not in your observation
- If you placed a table, SAY SO clearly with your location
- If you have 2+ wood, mention you should place table next

Return ONLY valid JSON, no extra text.
"""
    return prompt


def get_team_update_response_prompt(agent_id: int, agent_obs: str, broadcasts: List[dict]) -> str:
    """Get prompt for agent to respond to team broadcasts and adjust plan"""
    specializations = {
        0: "Warrior - ONLY you can craft swords (wood sword needs 1 wood at crafting table)",
        1: "Forager - Gathers food/water, hunts passive mobs for resources",
        2: "Miner - ONLY you can craft pickaxes (wood pickaxe needs 1 wood at crafting table)"
    }
    
    broadcasts_text = "\n".join([
        f"Agent {b['agent_id']}: {b['broadcast']}"
        for b in broadcasts
    ])
    
    prompt = f"""You are Agent {agent_id}, a {specializations[agent_id]} in Craftax-Coop.

TEAMMATE'S UPDATE:
{broadcasts_text}

YOUR CURRENT STATE:
{agent_obs}

TEAM COORDINATION - RESPONSE:
Based on the teammate's update and YOUR OWN observation, decide your next move.

RESPONSE FORMAT:
{{
  "agent_id": {agent_id},
  "response": "Noted. I'll continue in my northeast area to avoid clustering.",
  "plan_adjustment": "continue_gathering_in_my_region"
}}

CRITICAL RULES - PREVENT HALLUCINATIONS:
- ONLY use information from: (1) the teammate's broadcast, (2) YOUR observation above
- DO NOT assume things not stated (e.g., don't assume table is placed unless YOU see it in YOUR observation)
- Check YOUR observation for "crafting_table" - if YOU see one, you can mention it
- If you see other agents in YOUR observation, move away to avoid friendly fire
- If YOU have 2+ wood and no table exists, YOU should place it
- Base decisions on FACTS, not assumptions

Return ONLY valid JSON, no extra text.
"""
    return prompt


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


def get_proposal_prompt(agent_id: int, agent_obs: str) -> str:
    """Get prompt for initial proposal phase"""
    specializations = {
        0: "Warrior - ONLY you can craft swords (wood sword needs 1 wood at crafting table)",
        1: "Forager - Gathers food/water, hunts passive mobs for resources",
        2: "Miner - ONLY you can craft pickaxes (wood pickaxe needs 1 wood at crafting table)"
    }
    
    prompt = f"""You are Agent {agent_id}, a {specializations[agent_id]} in Craftax-Coop.

GAME OBJECTIVE:
Craftax-Coop is a cooperative survival game with agent specialization. Each agent has unique abilities:
- Warrior: Can craft swords (needs 1 wood at crafting table)
- Forager: Gathers food/water, hunts mobs
- Miner: Can craft pickaxes (needs 1 wood at crafting table)

IMMEDIATE GOAL:
1. Gather wood (chop trees with DO action)
2. Once ANY agent has 2 wood → that agent should PLACE_TABLE
3. Once table is placed → Miner crafts wood pickaxe (needs 1 wood), Warrior crafts wood sword (needs 1 wood)
4. Both Miner and Warrior need to move to the crafting table location to craft

CURRENT OBSERVATION:
{agent_obs}

TEAM COORDINATION - PROPOSAL PHASE:
Propose your high-level regional plan. Use DIRECTIONS (north, south, east, west, northeast, etc.) instead of specific coordinates.

RESPONSE FORMAT:
{{
  "agent_id": {agent_id},
  "proposal": "I will gather wood from trees in the NORTHEAST region. Once we have 2 wood total, whoever gets it first should place the table. I'll then move to the table to craft my pickaxe.",
  "target_region": "northeast",
  "backup_region": "east",
  "role_strategy": "As Miner, I need 1 wood at the crafting table to make a pickaxe"
}}

IMPORTANT:
- Use REGIONS/DIRECTIONS (north, south, east, west, northeast, etc.) NOT specific coordinates
- This allows teammates to understand your general area in THEIR coordinate system
- Explain your role-specific needs
- Keep proposal concise but clear

Return ONLY valid JSON, no extra text.
"""
    return prompt


def get_rebuttal_prompt(agent_id: int, agent_obs: str, all_proposals: List[dict]) -> str:
    """Get prompt for rebuttal phase"""
    specializations = {
        0: "Warrior - ONLY you can craft swords (wood sword needs 1 wood at crafting table)",
        1: "Forager - Gathers food/water, hunts passive mobs for resources",
        2: "Miner - ONLY you can craft pickaxes (wood pickaxe needs 1 wood at crafting table)"
    }
    
    proposals_text = "\n".join([
        f"Agent {p['agent_id']}: {p['proposal']}\nTarget region: {p.get('target_region', 'unknown')}"
        for p in all_proposals
    ])
    
    prompt = f"""You are Agent {agent_id}, a {specializations[agent_id]} in Craftax-Coop.

GAME OBJECTIVE REMINDER:
1. Gather wood (chop trees with DO action)
2. Once ANY agent has 2 wood → that agent PLACE_TABLE
3. Miner and Warrior move to table and craft (need 1 wood each)

TEAM PROPOSALS:
{proposals_text}

TEAM COORDINATION - REBUTTAL PHASE:
Review the proposals and adjust to avoid regional conflicts.

RESPONSE FORMAT:
{{
  "agent_id": {agent_id},
  "rebuttal": "I see Agent 1 is also targeting the northeast region. I'll shift to the east region instead to avoid clustering.",
  "conflicts_identified": ["Agent 1 and I both want northeast region"],
  "updated_plan": "I will focus on the EAST region for wood gathering, then watch for the crafting table placement"
}}

IMPORTANT:
- Identify any REGIONAL conflicts (e.g., multiple agents in same general area)
- Adjust your plan if there are conflicts
- If no conflicts, confirm your original plan
- Remember: whoever gets 2 wood first should place the table
- Be cooperative and spread out across regions

Return ONLY valid JSON, no extra text.
"""
    return prompt


def get_consensus_prompt(agent_id: int, agent_obs: str, proposals: List[dict], rebuttals: List[dict]) -> str:
    """Get prompt for consensus phase"""
    specializations = {
        0: "Warrior - ONLY you can craft swords (wood sword needs 1 wood at crafting table)",
        1: "Forager - Gathers food/water, hunts passive mobs for resources",
        2: "Miner - ONLY you can craft pickaxes (wood pickaxe needs 1 wood at crafting table)"
    }
    
    proposals_text = "\n".join([
        f"Agent {p['agent_id']} proposed: {p['proposal']}"
        for p in proposals
    ])
    
    rebuttals_text = "\n".join([
        f"Agent {r['agent_id']} responded: {r['rebuttal']}"
        for r in rebuttals
    ])
    
    prompt = f"""You are Agent {agent_id}, a {specializations[agent_id]} in Craftax-Coop.

PROPOSALS:
{proposals_text}

REBUTTALS:
{rebuttals_text}

INDEPENDENT DECISION - FINALIZE YOUR PLAN:
Based on the discussion, independently decide your final plan. You are making your own decision, not following a central authority.

RESPONSE FORMAT:
{{
  "agent_id": {agent_id},
  "final_plan": "I will gather wood from the EAST region. If I get 2 wood first, I'll place the table. Otherwise, I'll watch for the table and move to it to craft my pickaxe (need 1 wood).",
  "immediate_action": "move_to_trees_in_east_region",
  "coordination_notes": "Agent 0 takes west, Agent 1 takes north, I take east. Whoever gets 2 wood first places table."
}}

IMPORTANT:
- Make your OWN decision about your final plan
- State your conflict-free regional plan
- Acknowledge who places the table (whoever gets 2 wood first)
- If you're Miner or Warrior, note that you'll need to go to the table to craft
- Be specific about immediate first action

Return ONLY valid JSON, no extra text.
"""
    return prompt


def call_llm_conversation(prompt: str, agent_id: int, phase: str) -> dict:
    """Call LLM for conversation phases"""
    global llm_call_count, total_llm_time
    
    print(f"[Agent {agent_id}] {phase}...")
    
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a cooperative game-playing AI agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
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
        print(f"[Agent {agent_id}] {phase} completed ({elapsed:.2f}s)")
        
        return parsed
        
    except Exception as e:
        print(f"[Agent {agent_id}] ERROR in {phase}: {e}")
        return {"agent_id": agent_id, "error": str(e)}


def run_mini_conversation(obs: dict, triggering_agent: int, action_taken: str, reward: float, step_num: int) -> Dict[int, dict]:
    """
    Run a quick 2-phase conversation when an agent accomplishes something:
    1. Triggering agent broadcasts what they did
    2. Other agents respond and adjust plans
    """
    global last_broadcast_step
    last_broadcast_step = step_num
    
    print(f"\n{'='*70}")
    print(f"🔔 TEAM UPDATE (triggered by Agent {triggering_agent}'s {action_taken})")
    print('='*70)
    
    # Parse observations
    agent_observations = {}
    for agent_id, agent_name in enumerate(env.agents):
        parsed = parser.parse_observation(obs[agent_name], agent_id=agent_id)
        agent_observations[agent_id] = parser.to_text(parsed, agent_id)
    
    # PHASE 1: Broadcast from triggering agent
    print(f"\n📢 Agent {triggering_agent} Broadcasting...")
    broadcast_prompt = get_broadcast_prompt(
        triggering_agent, 
        agent_observations[triggering_agent],
        action_taken,
        reward
    )
    broadcast = call_llm_conversation(broadcast_prompt, triggering_agent, "Broadcast")
    
    print(f"\n[Agent {triggering_agent}] Broadcast: {broadcast.get('broadcast', 'ERROR')}")
    
    # PHASE 2: Other agents respond
    print(f"\n💭 Team Responses...")
    responses = {}
    for agent_id in range(len(env.agents)):
        if agent_id == triggering_agent:
            responses[agent_id] = broadcast
            continue
        
        response_prompt = get_team_update_response_prompt(
            agent_id,
            agent_observations[agent_id],
            [broadcast]
        )
        response = call_llm_conversation(response_prompt, agent_id, "Response")
        responses[agent_id] = response
        
        print(f"\n[Agent {agent_id}] Response: {response.get('response', 'ERROR')}")
        if response.get('plan_adjustment'):
            print(f"  Plan: {response.get('plan_adjustment')}")
    
    print(f"\n{'='*70}")
    print("✅ TEAM UPDATE COMPLETE")
    print('='*70)
    
    return responses


def run_team_conversation(obs: dict) -> Dict[int, dict]:
    """
    Run three-phase conversation: proposals, rebuttals, consensus
    Returns final plans for each agent
    """
    global conversation_history
    conversation_history = []
    
    print(f"\n{'='*70}")
    print("🗣️  TEAM COORDINATION CONVERSATION")
    print('='*70)
    
    # Parse observations for all agents
    agent_observations = {}
    for agent_id, agent_name in enumerate(env.agents):
        parsed = parser.parse_observation(obs[agent_name], agent_id=agent_id)
        agent_observations[agent_id] = parser.to_text(parsed, agent_id)
    
    # PHASE 1: PROPOSALS
    print("\n📋 PHASE 1: PROPOSALS")
    print("-" * 70)
    proposals = []
    
    for agent_id in range(len(env.agents)):
        prompt = get_proposal_prompt(agent_id, agent_observations[agent_id])
        proposal = call_llm_conversation(prompt, agent_id, "Proposal")
        proposals.append(proposal)
        print(f"\n[Agent {agent_id}] Proposal:")
        print(f"  {proposal.get('proposal', 'ERROR')}")
        print(f"  Targets: {proposal.get('primary_targets', [])}")
    
    conversation_history.append({"phase": "proposals", "data": proposals})
    
    # PHASE 2: REBUTTALS
    print("\n\n💬 PHASE 2: REBUTTALS")
    print("-" * 70)
    rebuttals = []
    
    for agent_id in range(len(env.agents)):
        prompt = get_rebuttal_prompt(agent_id, agent_observations[agent_id], proposals)
        rebuttal = call_llm_conversation(prompt, agent_id, "Rebuttal")
        rebuttals.append(rebuttal)
        print(f"\n[Agent {agent_id}] Rebuttal:")
        print(f"  {rebuttal.get('rebuttal', 'ERROR')}")
        if rebuttal.get('conflicts_identified'):
            print(f"  Conflicts: {rebuttal.get('conflicts_identified')}")
    
    conversation_history.append({"phase": "rebuttals", "data": rebuttals})
    
    # PHASE 3: CONSENSUS
    print("\n\n✅ PHASE 3: CONSENSUS")
    print("-" * 70)
    final_plans = {}
    
    for agent_id in range(len(env.agents)):
        prompt = get_consensus_prompt(agent_id, agent_observations[agent_id], proposals, rebuttals)
        consensus = call_llm_conversation(prompt, agent_id, "Consensus")
        final_plans[agent_id] = consensus
        print(f"\n[Agent {agent_id}] Final Plan:")
        print(f"  {consensus.get('final_plan', 'ERROR')}")
        print(f"  First action: {consensus.get('immediate_action', 'NOOP')}")
    
    conversation_history.append({"phase": "consensus", "data": final_plans})
    
    print(f"\n{'='*70}")
    print("✅ TEAM COORDINATION COMPLETE")
    print('='*70)
    
    return final_plans


def get_agent_prompt(agent_id: int, agent_obs: str, step_num: int, team_plan: Optional[dict] = None, parsed_obs: Optional[dict] = None) -> str:
    """Updated prompt that includes team consensus if available"""
    specializations = {
        0: "Warrior - ONLY you can craft swords (wood sword needs 1 wood at crafting table)",
        1: "Forager - Gathers food/water, hunts passive mobs for resources",
        2: "Miner - ONLY you can craft pickaxes (wood pickaxe needs 1 wood at crafting table)"
    }
    
    team_context = ""
    if team_plan:
        team_context = f"\n🤝 YOUR AGREED PLAN FROM TEAM DISCUSSION:\n{team_plan.get('final_plan', team_plan.get('response', 'No plan recorded'))}\n"
    
    # Add personal infrastructure memory context
    memory_context = get_agent_memory_context(agent_id)
    
    prompt = f"""You are Agent {agent_id}, a {specializations[agent_id]} in Craftax-Coop.
{team_context}
{memory_context}

GAME OBJECTIVE:
1. Gather wood (chop trees with DO action) 
2. Once ANY agent has 2 wood → that agent uses PLACE_TABLE action
3. Once table placed → Miner and Warrior move to table and craft:
   - Miner: MAKE_WOOD_PICKAXE (needs 1 wood, must be FACING table)
   - Warrior: MAKE_WOOD_SWORD (needs 1 wood, must be FACING table)

==============================================================================
CURRENT OBSERVATION - READ EVERY LINE CAREFULLY:
{agent_obs}
==============================================================================

⚠️ BEFORE MAKING ANY DECISION, ANSWER THESE QUESTIONS TO YOURSELF:

Q1: Do I see "crafting_table" in my CURRENT OBSERVATION above?
Q2: Do I have any tables in my PERSONAL INFRASTRUCTURE MEMORY above?
Q3: What is my current wood count from the observation?
Q4: Do I see OTHER AGENTS (agent_0, agent_1, agent_2) in my observation? ⚠️ CRITICAL FOR FRIENDLY FIRE

DECISION PROCESS:
- If answer to Q1 OR Q2 is YES → A table already exists, DO NOT place another
- If answer to Q4 is YES → ⚠️ DANGER! Other agents nearby - move away BEFORE using DO action
- If answer to Q1 or Q2 is YES AND I'm Miner/Warrior with wood → Move to table to craft
- If answer to Q1 and Q2 is NO AND Q3 shows I have 2+ wood → Place table
- Otherwise → Gather more wood by moving to a tree FIRST, then DO

⚠️ CRITICAL RULES:
- READ your observation carefully before deciding
- ONLY ONE table is needed for the entire team
- If a table exists (in observation OR memory) → NEVER place another
- AVOID FRIENDLY FIRE: Don't use DO if other agents are nearby

[SCENE ANALYSIS]
1. WHO IS NEARBY? (List all agents in your observation)
2. WHERE IS THE TABLE? (Check your observation and memory)
3. WHAT IS THE RISK? (e.g., "If I use DO now, will I hit Agent 1?")

[DECISION]
Based on the analysis above, provide your action.

RESPONSE FORMAT - Choose ONE:

1. MOVE TO TARGET:
{{
  "agent_id": {agent_id},
  "type": "move_to",
  "target": "tree",
  "target_coords": [-3, -1],
  "reasoning": "Checked observation - no table visible. No table in memory. Moving to tree to gather wood."
}}

2. IMMEDIATE ACTION:
{{
  "agent_id": {agent_id},
  "type": "action",
  "action": "PLACE_TABLE",
  "reasoning": "Checked observation - no table visible. Checked memory - no table stored. I have 2 wood. Placing table now."
}}

VALID ACTIONS:
Interaction: DO (chop trees, mine stone, drink water, attack mobs)
Crafting: MAKE_WOOD_PICKAXE (Miner only, 1 wood, at table), MAKE_WOOD_SWORD (Warrior only, 1 wood, at table)
Placement: PLACE_TABLE (needs 2 wood)
Trading: REQUEST_WOOD, REQUEST_STONE, REQUEST_FOOD, REQUEST_DRINK, GIVE
Other: SLEEP, REST, NOOP

IMPORTANT:
- Your reasoning MUST reference what you found in observation and memory
- Show you actually read the observation by mentioning specific details
- If placing table, explicitly state you checked both observation and memory

Return ONLY valid JSON, no extra text.
"""
    return prompt


def call_agent_llm(agent_id: int, agent_obs_text: str, step_num: int, team_plan: Optional[dict] = None, parsed_obs: Optional[dict] = None) -> dict:
    """Call OpenAI API for agent decision"""
    global llm_call_count, total_llm_time
    
    prompt = get_agent_prompt(agent_id, agent_obs_text, step_num, team_plan, parsed_obs)
    
    print(f"[Agent {agent_id}] Calling LLM...")
    
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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


def call_agent_llm_parallel(agent_id: int, agent_obs_text: str, step_num: int, team_plan: Optional[dict] = None, parsed_obs: Optional[dict] = None) -> Tuple[int, dict]:
    response = call_agent_llm(agent_id, agent_obs_text, step_num, team_plan, parsed_obs)
    return agent_id, response


def check_if_stuck(agent_id: int, env_state) -> bool:
    """
    Check if agent is stuck (position didn't change despite movement action).
    If stuck 2+ times in a row, abort the plan.
    """
    current_pos = tuple(env_state.player_position[agent_id])
    plan_info = agent_plans[agent_id]
    
    if plan_info['last_position'] is None:
        # First time tracking, just record position
        plan_info['last_position'] = current_pos
        return False
    
    # Check if position changed
    if current_pos == plan_info['last_position']:
        # Agent is stuck!
        plan_info['stuck_count'] += 1
        print(f"[Agent {agent_id}] ⚠️  STUCK at {current_pos} (count: {plan_info['stuck_count']})")
        
        if plan_info['stuck_count'] >= 2:
            # Been stuck for 2 steps, abort plan
            print(f"[Agent {agent_id}] ❌ ABORTING PLAN - stuck on obstacle")
            agent_plans[agent_id] = {'plan': [], 'target': None, 'target_coords': None, 'step_in_plan': 0, 'last_position': current_pos, 'stuck_count': 0}
            return True
    else:
        # Movement succeeded, reset stuck count
        plan_info['stuck_count'] = 0
    
    plan_info['last_position'] = current_pos
    return False


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
        
        # Get current position for stuck tracking
        current_pos = tuple(env_state.player_position[agent_id])
        
        # Store plan with target info
        agent_plans[agent_id] = {
            'plan': movement_plan,
            'target': target_name,
            'target_coords': target_coords,
            'step_in_plan': 0,
            'last_position': current_pos,
            'stuck_count': 0
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
        
        # No validation - let LLM make the decision
        if action not in ACTIONS:
            print(f"[Agent {agent_id}] Invalid action '{action}', using NOOP")
            action = 'NOOP'
        
        current_pos = tuple(env_state.player_position[agent_id])
        agent_plans[agent_id] = {'plan': [], 'target': None, 'target_coords': None, 'step_in_plan': 0, 'last_position': current_pos, 'stuck_count': 0}
        return action, False


def get_actions_for_step(obs: dict, step_num: int, use_parallel: bool = True, team_plans: Optional[Dict[int, dict]] = None) -> Dict[str, int]:
    """Get actions for all agents"""
    actions_dict = {}
    agents_needing_llm = []
    
    for agent_id, agent_name in enumerate(env.agents):
        parsed = parser.parse_observation(obs[agent_name], agent_id=agent_id)
        
        # Update agent's personal memory based on observation
        update_agent_memory(agent_id, parsed, env_state, step_num)
        
        # Check if agent is stuck (only if they have an active plan)
        if agent_plans[agent_id]['plan']:
            is_stuck = check_if_stuck(agent_id, env_state)
            if is_stuck:
                # Plan was aborted, need new LLM decision
                agents_needing_llm.append((agent_id, agent_name, parsed))
                continue
        
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
                    team_plan = team_plans.get(agent_id) if team_plans else None
                    future = executor.submit(call_agent_llm_parallel, agent_id, obs_text, step_num, team_plan, parsed)
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
                team_plan = team_plans.get(agent_id) if team_plans else None
                llm_response = call_agent_llm(agent_id, obs_text, step_num, team_plan, parsed)
                action_name, has_plan = process_agent_decision(agent_id, llm_response, parsed)
                actions_dict[agent_name] = ACTIONS[action_name]
    
    return actions_dict


def print_state_summary(env_state, step_num, obs=None):
    print(f"\n{'='*70}")
    print(f"STEP {step_num}")
    print('='*70)
    print(f"Positions: {env_state.player_position}")
    print(f"Health: {env_state.player_health}")
    print(f"Wood: [{env_state.inventory.wood[0]}, {env_state.inventory.wood[1]}, {env_state.inventory.wood[2]}]")
    
    # Print visible entities for each agent for debugging
    if obs is not None:
        print("\nVISIBLE ENTITIES:")
        for agent_id, agent_name in enumerate(env.agents):
            try:
                parsed = parser.parse_observation(obs[agent_name], agent_id=agent_id)
                obs_text = parser.to_text(parsed, agent_id)
                
                # Count entities from the text observation
                lines = obs_text.split('\n')
                trees = sum(1 for line in lines if 'tree' in line.lower() and 'at (' in line)
                agents_seen = sum(1 for line in lines if 'agent_' in line.lower() and f'agent_{agent_id}' not in line.lower() and 'at (' in line)
                mobs = sum(1 for line in lines if 'mob' in line.lower() or 'zombie' in line.lower() or 'skeleton' in line.lower() or 'cow' in line.lower())
                tables = sum(1 for line in lines if 'table' in line.lower() or 'crafting' in line.lower())
                
                nearby_info = []
                if trees > 0:
                    nearby_info.append(f"{trees} trees")
                if agents_seen > 0:
                    nearby_info.append(f"{agents_seen} other agents")
                if mobs > 0:
                    nearby_info.append(f"{mobs} mobs")
                if tables > 0:
                    nearby_info.append(f"{tables} tables")
                
                visible_str = ", ".join(nearby_info) if nearby_info else "nothing notable"
                print(f"  Agent {agent_id}: sees {visible_str}")
            except Exception as e:
                print(f"  Agent {agent_id}: [error parsing visibility: {e}]")
    
    # Print agent memory for debugging
    print("\nAGENT MEMORY (infrastructure):")
    for agent_id in range(len(env.agents)):
        memory = agent_memory[agent_id]
        if memory['crafting_tables']:
            tables_str = ", ".join([f"table at {t['rel_coords']}" for t in memory['crafting_tables']])
            print(f"  Agent {agent_id}: {tables_str}")
        else:
            print(f"  Agent {agent_id}: No tables in memory")
    
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


def run_episode(max_steps: int = 1000, use_parallel: bool = True, run_conversation: bool = True):
    """Run episode with optional team conversation at start"""
    global STEP, obs, env_state, rng, cumulative_rewards, llm_call_count, total_llm_time
    
    print(f"\n{'='*70}")
    print(f"STARTING EPISODE (max {max_steps} steps)")
    print('='*70)
    
    episode_start = time.time()
    
    # Run team conversation at step 1
    team_plans = None
    if run_conversation and STEP == 1:
        team_plans = run_team_conversation(obs)
    
    # Track last action and reward per agent for broadcast triggers
    last_actions = {i: 'NOOP' for i in range(len(env.agents))}
    last_rewards = {i: 0.0 for i in range(len(env.agents))}
    
    while STEP <= max_steps:
        print_state_summary(env_state, STEP, obs)
        
        if not env_state.player_alive.any():
            print("\n🔴 All agents died!")
            break
        
        try:
            # Pass team plans only for step 1, otherwise use responses from mini-convos
            current_team_plans = team_plans if STEP == 1 else None
            actions_dict = get_actions_for_step(obs, STEP, use_parallel=use_parallel, team_plans=current_team_plans)
            
            # Store actions taken
            for agent_name, action_id in actions_dict.items():
                agent_id = int(agent_name.split('_')[1])
                last_actions[agent_id] = ACTION_NAMES[action_id]
            
            print("\n" + "-"*70)
            print("ACTIONS:")
            for agent_name, action_id in actions_dict.items():
                agent_id = int(agent_name.split('_')[1])
                print(f"  Agent {agent_id}: {ACTION_NAMES[action_id]}")
            print("-"*70)
            
            # Step environment
            rng, step_rng = jax.random.split(rng)
            obs, env_state, rewards, dones, info = env.step(step_rng, env_state, actions_dict)
            
            # Print rewards and store them
            total_reward = sum(rewards.values())
            print(f"\nRewards: {total_reward:.2f}")
            for agent_name, reward in rewards.items():
                cumulative_rewards[agent_name] += reward
                agent_id = int(agent_name.split('_')[1])
                last_rewards[agent_id] = reward
                print(f"  Agent {agent_id}: +{reward:.2f} (Total: {cumulative_rewards[agent_name]:.2f})")
            
            # Check if any agent should trigger a broadcast
            if run_conversation and STEP > 1:  # Don't broadcast on step 1 (just had full convo)
                for agent_id in range(len(env.agents)):
                    if should_trigger_broadcast(agent_id, last_actions[agent_id], last_rewards[agent_id], STEP):
                        print(f"\n🔔 Agent {agent_id} triggered team update!")
                        team_plans = run_mini_conversation(
                            obs, agent_id, last_actions[agent_id], last_rewards[agent_id], STEP
                        )
                        break  # Only one broadcast per step
            
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
    print("CRAFTAX-COOP MULTI-AGENT LLM WITH TEAM CONVERSATION")
    print("="*70)
    print("Commands: run [N] [--no-convo] | reset | quit")
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
                    agent_plans[agent_id] = {'plan': [], 'target': None, 'target_coords': None, 'step_in_plan': 0, 'last_position': None, 'stuck_count': 0}
                for agent_id in agent_memory:
                    agent_memory[agent_id] = {'crafting_tables': [], 'furnaces': [], 'last_position': None}
                llm_call_count = 0
                total_llm_time = 0.0
                print("✅ Reset!")
                continue
            
            elif user_input.startswith('run'):
                parts = user_input.split()
                max_steps = 50
                run_conversation = True
                
                for part in parts[1:]:
                    if part.isdigit():
                        max_steps = int(part)
                    elif part == '--no-convo':
                        run_conversation = False
                
                run_episode(max_steps=max_steps, use_parallel=True, run_conversation=run_conversation)
            
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