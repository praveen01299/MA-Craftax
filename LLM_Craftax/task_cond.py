"""Task precondition checkers based on textual observations.

Each function returns True if the conditions to attempt the task are met.
All checkers accept a single `obs` string (the current textual observation).

Shared parsing utilities are imported from task_check to avoid duplication.
"""

import re
from typing import Optional

from task_check import parse_observation, _tiles_contain, _visible_tiles_text


# ---------------------------------------------------------------------------
# Distance-aware tile helper
# ---------------------------------------------------------------------------

def _tiles_within_distance(obs: str, max_dist: int, *keywords: str) -> bool:
    """Return True if any keyword appears in a visible tile at distance <= max_dist.

    Handles the HERE section (distance 0) and [Distance N tiles away] sections.
    """
    tiles_text = _visible_tiles_text(obs)
    current_dist = 0
    in_range = True

    for line in tiles_text.splitlines():
        dist_match = re.search(r'\[distance (\d+)', line, re.IGNORECASE)
        if dist_match:
            current_dist = int(dist_match.group(1))
            in_range = current_dist <= max_dist
        elif '[here' in line.lower():
            current_dist = 0
            in_range = True

        if in_range and any(kw.lower() in line.lower() for kw in keywords):
            return True

    return False


def _at_structure(obs: str, *keywords: str) -> bool:
    """Return True if a structure tile is adjacent to the agent (distance <= 1).

    Mirrors the game's is_near_block() check over the 8 adjacent + current tile.
    """
    return _tiles_within_distance(obs, 1, *keywords)


# ---------------------------------------------------------------------------
# Shared inventory / equipment helpers (thin wrappers for readability)
# ---------------------------------------------------------------------------

def _has(obs: str, **requirements: int) -> bool:
    """Return True if every inventory key meets its minimum amount.

    Example: _has(obs, wood=2, stone=1)
    """
    state = parse_observation(obs)
    return all(state.get(key, 0) >= amount for key, amount in requirements.items())


# ---------------------------------------------------------------------------
# Task precondition functions
# ---------------------------------------------------------------------------

def cond_collect_wood(obs: str) -> bool:
    return _tiles_contain(obs, 'tree')


def cond_place_table(obs: str) -> bool:
    # Cost: 2 wood
    return _has(obs, wood=2)


def cond_eat_cow(obs: str) -> bool:
    return _tiles_contain(obs, 'cow')


def cond_collect_sapling(obs: str) -> bool:
    # Saplings drop from grass tiles (10 % chance per hit)
    return _tiles_contain(obs, 'grass')


def cond_collect_drink(obs: str) -> bool:
    return _tiles_contain(obs, 'water', 'fountain')


def cond_make_wood_pickaxe(obs: str) -> bool:
    # Cost: 1 wood | Requires: crafting table adjacent | Upgrade guard: pickaxe < 1
    state = parse_observation(obs)
    return (
        state['wood'] >= 1
        and state['pickaxe_lvl'] < 1
        and _at_structure(obs, 'table', 'crafting_table')
    )


def cond_make_wood_sword(obs: str) -> bool:
    # Cost: 1 wood | Requires: crafting table adjacent | Upgrade guard: sword < 1
    state = parse_observation(obs)
    return (
        state['wood'] >= 1
        and state['sword_lvl'] < 1
        and _at_structure(obs, 'table', 'crafting_table')
    )


def cond_place_plant(obs: str) -> bool:
    # Cost: 1 sapling
    return _has(obs, saplings=1)


def cond_defeat_zombie(obs: str) -> bool:
    return _tiles_contain(obs, 'zombie')


def cond_collect_stone(obs: str) -> bool:
    # Requires: pickaxe lvl >= 1 AND stone tile visible
    state = parse_observation(obs)
    return state['pickaxe_lvl'] >= 1 and _tiles_contain(obs, 'stone')


def cond_place_stone(obs: str) -> bool:
    # Cost: 1 stone
    return _has(obs, stone=1)


def cond_eat_plant(obs: str) -> bool:
    # Only ripe plants can be eaten
    return _tiles_contain(obs, 'ripe_plant')


def cond_defeat_skeleton(obs: str) -> bool:
    return _tiles_contain(obs, 'skeleton')


def cond_make_stone_pickaxe(obs: str) -> bool:
    # Cost: 1 wood + 1 stone | Requires: crafting table adjacent | Upgrade guard: pickaxe < 2
    state = parse_observation(obs)
    return (
        state['wood'] >= 1
        and state['stone'] >= 1
        and state['pickaxe_lvl'] < 2
        and _at_structure(obs, 'table', 'crafting_table')
    )


def cond_make_stone_sword(obs: str) -> bool:
    # Cost: 1 wood + 1 stone | Requires: crafting table adjacent | Upgrade guard: sword < 2
    state = parse_observation(obs)
    return (
        state['wood'] >= 1
        and state['stone'] >= 1
        and state['sword_lvl'] < 2
        and _at_structure(obs, 'table', 'crafting_table')
    )


def cond_wake_up(obs: str) -> bool:
    # Agent must currently be sleeping; wake-up triggers automatically at full energy
    return parse_observation(obs)['sleeping']


def cond_place_furnace(obs: str) -> bool:
    # Cost: 1 stone
    return _has(obs, stone=1)


def cond_collect_coal(obs: str) -> bool:
    # Requires: pickaxe lvl >= 1 AND coal tile visible
    state = parse_observation(obs)
    return state['pickaxe_lvl'] >= 1 and _tiles_contain(obs, 'coal')


def cond_collect_iron(obs: str) -> bool:
    # Requires: pickaxe lvl >= 2 AND iron tile visible
    state = parse_observation(obs)
    return state['pickaxe_lvl'] >= 2 and _tiles_contain(obs, 'iron')


def cond_make_iron_pickaxe(obs: str) -> bool:
    # Cost: 1 wood + 1 stone + 1 iron + 1 coal
    # Requires: crafting table AND furnace adjacent | Upgrade guard: pickaxe < 3
    state = parse_observation(obs)
    return (
        state['wood'] >= 1
        and state['stone'] >= 1
        and state['iron'] >= 1
        and state['coal'] >= 1
        and state['pickaxe_lvl'] < 3
        and _at_structure(obs, 'table', 'crafting_table')
        and _at_structure(obs, 'furnace')
    )


def cond_make_iron_sword(obs: str) -> bool:
    # Cost: 1 wood + 1 stone + 1 iron + 1 coal
    # Requires: crafting table AND furnace adjacent | Upgrade guard: sword < 3
    state = parse_observation(obs)
    return (
        state['wood'] >= 1
        and state['stone'] >= 1
        and state['iron'] >= 1
        and state['coal'] >= 1
        and state['sword_lvl'] < 3
        and _at_structure(obs, 'table', 'crafting_table')
        and _at_structure(obs, 'furnace')
    )


def cond_make_arrow(obs: str) -> bool:
    # Cost: 1 wood + 1 stone | Requires: crafting table adjacent | Cap: arrows < 99
    state = parse_observation(obs)
    return (
        state['wood'] >= 1
        and state['stone'] >= 1
        and state['arrows'] < 99
        and _at_structure(obs, 'table', 'crafting_table')
    )


def cond_make_torch(obs: str) -> bool:
    # Cost: 1 wood + 1 coal | Requires: crafting table adjacent | Cap: torches < 99
    state = parse_observation(obs)
    return (
        state['wood'] >= 1
        and state['coal'] >= 1
        and state['torches'] < 99
        and _at_structure(obs, 'table', 'crafting_table')
    )


def cond_place_torch(obs: str) -> bool:
    # Cost: 1 torch from inventory
    return _has(obs, torches=1)


def cond_collect_food(obs: str) -> bool:
    # Food is collected by killing a passive mob (cow); same target as EAT_COW
    return _tiles_contain(obs, 'cow')


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

TASK_CONDITIONS = {
    'COLLECT_WOOD':       cond_collect_wood,
    'PLACE_TABLE':        cond_place_table,
    'EAT_COW':            cond_eat_cow,
    'COLLECT_SAPLING':    cond_collect_sapling,
    'COLLECT_DRINK':      cond_collect_drink,
    'MAKE_WOOD_PICKAXE':  cond_make_wood_pickaxe,
    'MAKE_WOOD_SWORD':    cond_make_wood_sword,
    'PLACE_PLANT':        cond_place_plant,
    'DEFEAT_ZOMBIE':      cond_defeat_zombie,
    'COLLECT_STONE':      cond_collect_stone,
    'PLACE_STONE':        cond_place_stone,
    'EAT_PLANT':          cond_eat_plant,
    'DEFEAT_SKELETON':    cond_defeat_skeleton,
    'MAKE_STONE_PICKAXE': cond_make_stone_pickaxe,
    'MAKE_STONE_SWORD':   cond_make_stone_sword,
    'WAKE_UP':            cond_wake_up,
    'PLACE_FURNACE':      cond_place_furnace,
    'COLLECT_COAL':       cond_collect_coal,
    'COLLECT_IRON':       cond_collect_iron,
    'MAKE_IRON_PICKAXE':  cond_make_iron_pickaxe,
    'MAKE_IRON_SWORD':    cond_make_iron_sword,
    'MAKE_ARROW':         cond_make_arrow,
    'MAKE_TORCH':         cond_make_torch,
    'PLACE_TORCH':        cond_place_torch,
    'COLLECT_FOOD':       cond_collect_food,
}


def is_task_feasible(task: str, obs: str) -> bool:
    """Return True if the preconditions for the named task are currently met.

    Args:
        task: Task name (case-insensitive), e.g. 'COLLECT_STONE'.
        obs:  Current textual observation string.
    """
    checker = TASK_CONDITIONS.get(task.upper())
    if checker is None:
        raise ValueError(f"Unknown task: {task!r}. Known tasks: {sorted(TASK_CONDITIONS)}")
    return checker(obs)
