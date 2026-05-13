"""Task completion checkers based on textual observations.

Each checker accepts:
  obs      - the current textual observation string
  prev_obs - (optional) the previous textual observation string

When prev_obs is supplied, comparison-based detection is used (more precise).
Without prev_obs, the current state is inspected as a fallback.
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Observation parser
# ---------------------------------------------------------------------------

def parse_observation(obs: str) -> dict:
    """Parse a textual observation string into a structured dictionary."""
    state = {
        'health': 0.0,
        'food': 0.0,
        'drink': 0.0,
        'energy': 0.0,
        'mana': 0.0,
        'sleeping': False,
        'resting': False,
        'wood': 0,
        'stone': 0,
        'coal': 0,
        'iron': 0,
        'diamond': 0,
        'torches': 0,
        'arrows': 0,
        'saplings': 0,
        'pickaxe_lvl': 0,
        'sword_lvl': 0,
        'bow': False,
        'xp': 0.0,
    }

    float_fields = {'health', 'food', 'drink', 'energy', 'mana', 'xp'}

    patterns = {
        'health':      r'Health:\s*([\d.]+)',
        'food':        r'Food:\s*([\d.]+)',
        'drink':       r'Drink:\s*([\d.]+)',
        'energy':      r'Energy:\s*([\d.]+)',
        'mana':        r'Mana:\s*([\d.]+)',
        'xp':          r'XP:\s*([\d.]+)',
        'wood':        r'Wood=(\d+)',
        'stone':       r'Stone=(\d+)',
        'coal':        r'Coal=(\d+)',
        'iron':        r'Iron=(\d+)',
        'diamond':     r'Diamond=(\d+)',
        'torches':     r'Torches=(\d+)',
        'arrows':      r'Arrows=(\d+)',
        'saplings':    r'Saplings=(\d+)',
        'pickaxe_lvl': r'Pickaxe Lvl (\d+)',
        'sword_lvl':   r'Sword Lvl (\d+)',
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, obs)
        if m:
            state[key] = float(m.group(1)) if key in float_fields else int(m.group(1))

    m = re.search(r'Sleeping:\s*(True|False)', obs)
    if m:
        state['sleeping'] = m.group(1) == 'True'

    m = re.search(r'Resting:\s*(True|False)', obs)
    if m:
        state['resting'] = m.group(1) == 'True'

    m = re.search(r'Bow=(True|False)', obs)
    if m:
        state['bow'] = m.group(1) == 'True'

    return state


def _visible_tiles_text(obs: str) -> str:
    """Return the VISIBLE TILES section of an observation, or empty string."""
    m = re.search(r'VISIBLE TILES.*', obs, re.DOTALL)
    return m.group().lower() if m else ''


def _tiles_contain(obs: str, *keywords: str) -> bool:
    """Return True if any keyword appears in the visible tiles section."""
    tiles = _visible_tiles_text(obs)
    return any(kw.lower() in tiles for kw in keywords)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _inventory_increased(key: str, obs: str, prev_obs: str) -> bool:
    return parse_observation(obs)[key] > parse_observation(prev_obs)[key]



def _stat_increased(key: str, obs: str, prev_obs: str) -> bool:
    return parse_observation(obs)[key] > parse_observation(prev_obs)[key]


def _enemy_defeated(enemy_keyword: str, obs: str, prev_obs: str) -> bool:
    """Enemy was visible before and is gone now, or XP increased."""
    state = parse_observation(obs)
    prev = parse_observation(prev_obs)
    enemy_was_visible = _tiles_contain(prev_obs, enemy_keyword)
    enemy_gone = not _tiles_contain(obs, enemy_keyword)
    xp_gained = state['xp'] > prev['xp']
    return (enemy_was_visible and enemy_gone) or xp_gained


def _consumed(key: str, amount: int, obs: str, prev_obs: str) -> bool:
    """Return True if inventory[key] decreased by at least `amount`."""
    return (parse_observation(prev_obs)[key] - parse_observation(obs)[key]) >= amount


def _ate(obs: str, prev_obs: str) -> bool:
    """Food stat increased (generic eat check)."""
    return _stat_increased('food', obs, prev_obs)


# ---------------------------------------------------------------------------
# Task checkers
# ---------------------------------------------------------------------------

def check_collect_wood(obs: str, prev_obs: Optional[str] = None) -> bool:
    if prev_obs:
        return _inventory_increased('wood', obs, prev_obs)
    return parse_observation(obs)['wood'] > 0


def check_place_table(obs: str, prev_obs: Optional[str] = None) -> bool:
    # Requires 2 wood. Wood is also spent crafting tools, so require tile visible to disambiguate.
    if prev_obs:
        return _consumed('wood', 2, obs, prev_obs) and _tiles_contain(obs, 'table', 'crafting_table')
    return _tiles_contain(obs, 'table', 'crafting_table')


def check_eat_cow(obs: str, prev_obs: Optional[str] = None) -> bool:
    if prev_obs:
        cow_was_near = _tiles_contain(prev_obs, 'cow')
        return cow_was_near and _ate(obs, prev_obs)
    return parse_observation(obs)['food'] >= 9.0


def check_collect_sapling(obs: str, prev_obs: Optional[str] = None) -> bool:
    if prev_obs:
        return _inventory_increased('saplings', obs, prev_obs)
    return parse_observation(obs)['saplings'] > 0


def check_collect_drink(obs: str, prev_obs: Optional[str] = None) -> bool:
    if prev_obs:
        return _stat_increased('drink', obs, prev_obs)
    return parse_observation(obs)['drink'] >= 9.0


def check_make_wood_pickaxe(obs: str, prev_obs: Optional[str] = None) -> bool:
    return parse_observation(obs)['pickaxe_lvl'] >= 1


def check_make_wood_sword(obs: str, prev_obs: Optional[str] = None) -> bool:
    return parse_observation(obs)['sword_lvl'] >= 1


def check_place_plant(obs: str, prev_obs: Optional[str] = None) -> bool:
    # Requires 1 sapling. Saplings have no other consumption path.
    if prev_obs:
        return _consumed('saplings', 1, obs, prev_obs)
    return _tiles_contain(obs, 'plant')


def check_defeat_zombie(obs: str, prev_obs: Optional[str] = None) -> bool:
    if prev_obs:
        return _enemy_defeated('zombie', obs, prev_obs)
    return parse_observation(obs)['xp'] > 0.0


def check_collect_stone(obs: str, prev_obs: Optional[str] = None) -> bool:
    if prev_obs:
        return _inventory_increased('stone', obs, prev_obs)
    return parse_observation(obs)['stone'] > 0


def check_place_stone(obs: str, prev_obs: Optional[str] = None) -> bool:
    # Requires 1 stone. Stone is also spent placing a furnace; that case is
    # covered by check_place_furnace which additionally requires the furnace tile.
    if prev_obs:
        return _consumed('stone', 1, obs, prev_obs)
    return False  # cannot reliably infer from a single observation


def check_eat_plant(obs: str, prev_obs: Optional[str] = None) -> bool:
    if prev_obs:
        plant_was_near = _tiles_contain(prev_obs, 'plant')
        return plant_was_near and _ate(obs, prev_obs)
    return parse_observation(obs)['food'] >= 9.0


def check_defeat_skeleton(obs: str, prev_obs: Optional[str] = None) -> bool:
    if prev_obs:
        return _enemy_defeated('skeleton', obs, prev_obs)
    return parse_observation(obs)['xp'] > 0.0


def check_make_stone_pickaxe(obs: str, prev_obs: Optional[str] = None) -> bool:
    return parse_observation(obs)['pickaxe_lvl'] >= 2


def check_make_stone_sword(obs: str, prev_obs: Optional[str] = None) -> bool:
    return parse_observation(obs)['sword_lvl'] >= 2


def check_wake_up(obs: str, prev_obs: Optional[str] = None) -> bool:
    state = parse_observation(obs)
    if prev_obs:
        prev = parse_observation(prev_obs)
        return prev['sleeping'] and not state['sleeping']
    return not state['sleeping']


def check_place_furnace(obs: str, prev_obs: Optional[str] = None) -> bool:
    # Requires 1 stone. Stone also decreases when crafting stone tools, so
    # require the furnace tile to be visible to disambiguate.
    if prev_obs:
        return _consumed('stone', 1, obs, prev_obs) and _tiles_contain(obs, 'furnace')
    return _tiles_contain(obs, 'furnace')


def check_collect_coal(obs: str, prev_obs: Optional[str] = None) -> bool:
    if prev_obs:
        return _inventory_increased('coal', obs, prev_obs)
    return parse_observation(obs)['coal'] > 0


def check_collect_iron(obs: str, prev_obs: Optional[str] = None) -> bool:
    if prev_obs:
        return _inventory_increased('iron', obs, prev_obs)
    return parse_observation(obs)['iron'] > 0


def check_make_iron_pickaxe(obs: str, prev_obs: Optional[str] = None) -> bool:
    return parse_observation(obs)['pickaxe_lvl'] >= 3


def check_make_iron_sword(obs: str, prev_obs: Optional[str] = None) -> bool:
    return parse_observation(obs)['sword_lvl'] >= 3


def check_make_arrow(obs: str, prev_obs: Optional[str] = None) -> bool:
    if prev_obs:
        return _inventory_increased('arrows', obs, prev_obs)
    return parse_observation(obs)['arrows'] > 0


def check_make_torch(obs: str, prev_obs: Optional[str] = None) -> bool:
    if prev_obs:
        return _inventory_increased('torches', obs, prev_obs)
    return parse_observation(obs)['torches'] > 0


def check_place_torch(obs: str, prev_obs: Optional[str] = None) -> bool:
    # Requires 1 torch from inventory. Torches have no other consumption path.
    if prev_obs:
        return _consumed('torches', 1, obs, prev_obs)
    return _tiles_contain(obs, 'torch')


def check_collect_food(obs: str, prev_obs: Optional[str] = None) -> bool:
    if prev_obs:
        return _stat_increased('food', obs, prev_obs)
    return parse_observation(obs)['food'] >= 9.0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

TASK_CHECKERS = {
    'COLLECT_WOOD':       check_collect_wood,
    'PLACE_TABLE':        check_place_table,
    'EAT_COW':            check_eat_cow,
    'COLLECT_SAPLING':    check_collect_sapling,
    'COLLECT_DRINK':      check_collect_drink,
    'MAKE_WOOD_PICKAXE':  check_make_wood_pickaxe,
    'MAKE_WOOD_SWORD':    check_make_wood_sword,
    'PLACE_PLANT':        check_place_plant,
    'DEFEAT_ZOMBIE':      check_defeat_zombie,
    'COLLECT_STONE':      check_collect_stone,
    'PLACE_STONE':        check_place_stone,
    'EAT_PLANT':          check_eat_plant,
    'DEFEAT_SKELETON':    check_defeat_skeleton,
    'MAKE_STONE_PICKAXE': check_make_stone_pickaxe,
    'MAKE_STONE_SWORD':   check_make_stone_sword,
    'WAKE_UP':            check_wake_up,
    'PLACE_FURNACE':      check_place_furnace,
    'COLLECT_COAL':       check_collect_coal,
    'COLLECT_IRON':       check_collect_iron,
    'MAKE_IRON_PICKAXE':  check_make_iron_pickaxe,
    'MAKE_IRON_SWORD':    check_make_iron_sword,
    'MAKE_ARROW':         check_make_arrow,
    'MAKE_TORCH':         check_make_torch,
    'PLACE_TORCH':        check_place_torch,
    'COLLECT_FOOD':       check_collect_food,
}


def is_task_complete(task: str, obs: str, prev_obs: Optional[str] = None) -> bool:
    """Return True if the named task appears complete in the given observation.

    Args:
        task:     Task name (case-insensitive), e.g. 'COLLECT_WOOD'.
        obs:      Current textual observation string.
        prev_obs: Previous textual observation string (improves precision for
                  transition-based tasks like eating, defeating enemies, etc.).
    """
    checker = TASK_CHECKERS.get(task.upper())
    if checker is None:
        raise ValueError(f"Unknown task: {task!r}. Known tasks: {sorted(TASK_CHECKERS)}")
    return checker(obs, prev_obs)
