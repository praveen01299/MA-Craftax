from craftax.craftax_ma.constants import *
import jax.numpy as jnp
import numpy as np

OBS_DIM = (9, 11)


class CraftaxMAObservationParser:
    """Parse Craftax-MA symbolic observations into human-readable text.

    MA observation layout (flat vector per agent):
        [ map_section ] [ inventory(16) ] [ potions(6) ] [ intrinsics(9) ]
        [ direction(4) ] [ armour(4) ] [ armour_enchants(4) ]
        [ special_per_player(3) ] [ special_level(4) ]

    Map section per tile (85 features):
        [ blocks(37) ] [ items(5) ] [ mobs(40) ] [ teammate(2) ] [ light(1) ]

    Key differences from Craftax-Coop parser:
    - Teammate map is 2 channels (location + alive), not per-agent one-hot
    - No teammate dashboard (health/specialization/requests)
    - No teammate direction encoding
    - Health is in intrinsics[0], not a separate dashboard
    - special_level has 4 values (adds boss_vulnerable)
    - inventory_size = 50 (vs 48 in coop)
    """

    def __init__(self, num_agents: int = 2):
        self.num_agents = num_agents

        self.num_blocks = len(BlockType)   # 37
        self.num_items = len(ItemType)     # 5
        self.num_mob_classes = 5
        self.num_mob_types = 8
        self.num_teammate_channels = 2     # location bit + alive bit

        # Map: 9x11 tiles, each with 85 features
        self.num_tile_features = (
            self.num_blocks
            + self.num_items
            + self.num_mob_classes * self.num_mob_types
            + self.num_teammate_channels
            + 1  # light
        )
        self.map_size = OBS_DIM[0] * OBS_DIM[1] * self.num_tile_features

        # Inventory section (50 total)
        # 16 inventory + 6 potions + 9 intrinsics + 4 direction
        # + 4 armour + 4 armour_enchants + 3 special_per_player + 4 special_level
        self.inventory_size = 16 + 6 + 9 + 4 + 4 + 4 + 3 + 4

    def parse_observation(self, obs, agent_id: int = 0) -> dict:
        """Parse a single agent's flat observation vector into structured data."""
        obs = np.array(obs) if isinstance(obs, jnp.ndarray) else obs

        idx = 0

        # 1. Map view
        map_data = obs[idx:idx + self.map_size]
        idx += self.map_size
        map_parsed = self._parse_map(map_data)

        # 2. Inventory + stats (no teammate dashboard in MA)
        inventory_data = obs[idx:idx + self.inventory_size]
        inventory_parsed = self._parse_inventory(inventory_data)

        return {
            'agent_id': agent_id,
            'map': map_parsed,
            'inventory': inventory_parsed,
            'raw_obs': obs,
        }

    # ------------------------------------------------------------------
    # Map parsing
    # ------------------------------------------------------------------

    def _parse_map(self, map_data: np.ndarray) -> dict:
        """Reshape and parse the map section."""
        map_grid = map_data.reshape(OBS_DIM[0], OBS_DIM[1], self.num_tile_features)

        # Feature slice offsets
        b_end = self.num_blocks
        i_end = b_end + self.num_items
        m_end = i_end + self.num_mob_classes * self.num_mob_types
        t_end = m_end + self.num_teammate_channels  # +2

        blocks_layer    = map_grid[:, :, :b_end]
        items_layer     = map_grid[:, :, b_end:i_end]
        mobs_layer      = map_grid[:, :, i_end:m_end]
        teammate_layer  = map_grid[:, :, m_end:t_end]  # shape (9,11,2)
        light_layer     = map_grid[:, :, t_end]         # shape (9,11)

        block_indices = np.argmax(blocks_layer, axis=-1)
        impassable = np.array(SOLID_BLOCK_MAPPING)[block_indices]
        passable_grid = impassable.astype(int)

        visible_tiles = []
        for i in range(OBS_DIM[0]):
            for j in range(OBS_DIM[1]):
                if light_layer[i, j] > 0:
                    tile_info = self._parse_tile(
                        i, j,
                        blocks_layer[i, j],
                        items_layer[i, j],
                        mobs_layer[i, j],
                        teammate_layer[i, j],
                    )
                    if tile_info:
                        visible_tiles.append(tile_info)

        return {
            'visible_tiles': visible_tiles,
            'passable_grid': passable_grid,
            'grid_shape': OBS_DIM,
            'center': (OBS_DIM[0] // 2, OBS_DIM[1] // 2),
        }

    def _parse_tile(self, i, j, blocks, items, mobs, teammate) -> dict | None:
        """Parse a single lit tile. Returns None if nothing notable is present."""
        tile_contents = []

        # Block type
        block_idx = np.argmax(blocks)
        block_type = list(BlockType)[block_idx]
        if block_type not in (BlockType.GRASS, BlockType.PATH, BlockType.INVALID):
            tile_contents.append(block_type.name.lower().replace('_', ' '))

        # Item
        item_idx = np.argmax(items)
        item_type = list(ItemType)[item_idx]
        if item_type != ItemType.NONE:
            tile_contents.append(item_type.name.lower().replace('_', ' '))

        # Mobs (5 classes × 8 types)
        mob_grid = mobs.reshape(self.num_mob_classes, self.num_mob_types)
        mob_class_names = ['melee', 'passive', 'ranged', 'mob_projectile', 'player_projectile']
        for cls_idx, cls_name in enumerate(mob_class_names):
            for type_idx in range(self.num_mob_types):
                if mob_grid[cls_idx, type_idx] > 0:
                    tile_contents.append(f"{cls_name}_mob_{type_idx}")

        # Teammate: channel 0 = present, channel 1 = alive
        # MA doesn't encode which agent — just that *a* teammate is here
        has_teammate = teammate[0] > 0
        if has_teammate:
            is_alive = teammate[1] > 0
            status = "alive" if is_alive else "dead"
            tile_contents.append(f"teammate ({status})")

        if not tile_contents:
            return None

        rel_i = i - OBS_DIM[0] // 2
        rel_j = j - OBS_DIM[1] // 2
        direction = self._get_direction(rel_i, rel_j)
        manhattan_dist = abs(rel_i) + abs(rel_j)

        return {
            'position': (i, j),
            'relative': (rel_i, rel_j),
            'direction': direction,
            'manhattan_distance': manhattan_dist,
            'contents': tile_contents,
        }

    def _get_direction(self, di: int, dj: int) -> str:
        """Convert relative grid offsets to a compass direction string."""
        if di == 0 and dj == 0:
            return "here"

        vert  = "north" if di < 0 else ("south" if di > 0 else "")
        horiz = "west"  if dj < 0 else ("east"  if dj > 0 else "")

        if abs(di) >= abs(dj):
            return f"{vert} {horiz}".strip()
        else:
            return f"{vert} {horiz}".strip()

    # ------------------------------------------------------------------
    # Inventory / stats parsing
    # ------------------------------------------------------------------

    def _parse_inventory(self, inv_data: np.ndarray) -> dict:
        """Parse the flat inventory + stats vector (50 values)."""
        idx = 0

        # --- Inventory items (16) ---
        # Stored as sqrt(x)/10, so recover as (val*10)^2
        wood      = (inv_data[idx] * 10.0) ** 2; idx += 1
        stone     = (inv_data[idx] * 10.0) ** 2; idx += 1
        coal      = (inv_data[idx] * 10.0) ** 2; idx += 1
        iron      = (inv_data[idx] * 10.0) ** 2; idx += 1
        diamond   = (inv_data[idx] * 10.0) ** 2; idx += 1
        sapphire  = (inv_data[idx] * 10.0) ** 2; idx += 1
        ruby      = (inv_data[idx] * 10.0) ** 2; idx += 1
        sapling   = (inv_data[idx] * 10.0) ** 2; idx += 1
        torches   = (inv_data[idx] * 10.0) ** 2; idx += 1
        arrows    = (inv_data[idx] * 10.0) ** 2; idx += 1
        books     = inv_data[idx];               idx += 1
        pickaxe_level = int(inv_data[idx] * 4); idx += 1
        sword_level   = int(inv_data[idx] * 4); idx += 1
        sword_enchant = inv_data[idx];           idx += 1
        bow_enchant   = inv_data[idx];           idx += 1
        has_bow       = inv_data[idx] > 0.5;     idx += 1

        # --- Potions (6) ---
        potions = [(inv_data[idx + i] * 10.0) ** 2 for i in range(6)]
        idx += 6

        # --- Intrinsics (9) — health is FIRST in MA, unlike Coop ---
        health       = inv_data[idx] * 10.0; idx += 1
        food         = inv_data[idx] * 10.0; idx += 1
        drink        = inv_data[idx] * 10.0; idx += 1
        energy       = inv_data[idx] * 10.0; idx += 1
        mana         = inv_data[idx] * 10.0; idx += 1
        xp           = inv_data[idx] * 10.0; idx += 1
        dexterity    = inv_data[idx] * 10.0; idx += 1
        strength     = inv_data[idx] * 10.0; idx += 1
        intelligence = inv_data[idx] * 10.0; idx += 1

        # --- Direction (4, one-hot) ---
        direction_onehot = inv_data[idx:idx + 4]; idx += 4
        facing = ['west', 'east', 'north', 'south'][np.argmax(direction_onehot)]

        # --- Armour (4) ---
        armour = [int(inv_data[idx + i] * 2) for i in range(4)]; idx += 4

        # --- Armour enchantments (4) ---
        armour_enchants = [inv_data[idx + i] for i in range(4)]; idx += 4

        # --- Special values per player (3) ---
        is_sleeping   = inv_data[idx] > 0.5; idx += 1
        is_resting    = inv_data[idx] > 0.5; idx += 1
        learned_spell = inv_data[idx] > 0.5; idx += 1

        # --- Special level values (4) — MA adds boss_vulnerable vs Coop's 3 ---
        light_level     = inv_data[idx];              idx += 1
        current_level   = int(inv_data[idx] * 10);   idx += 1
        level_cleared   = inv_data[idx] > 0.5;       idx += 1
        boss_vulnerable = inv_data[idx] > 0.5;       idx += 1

        return {
            'materials': {
                'wood':     int(wood),
                'stone':    int(stone),
                'coal':     int(coal),
                'iron':     int(iron),
                'diamond':  int(diamond),
                'sapphire': int(sapphire),
                'ruby':     int(ruby),
            },
            'items': {
                'sapling': int(sapling),
                'torches': int(torches),
                'arrows':  int(arrows),
                'books':   int(books),
            },
            'equipment': {
                'pickaxe_level': pickaxe_level,
                'sword_level':   sword_level,
                'has_bow':       has_bow,
                'armour_levels': armour,
            },
            'stats': {
                'health':       health,
                'food':         food,
                'drink':        drink,
                'energy':       energy,
                'mana':         mana,
                'xp':           xp,
                'dexterity':    dexterity,
                'strength':     strength,
                'intelligence': intelligence,
            },
            'state': {
                'facing':          facing,
                'sleeping':        is_sleeping,
                'resting':         is_resting,
                'learned_spell':   learned_spell,
                'current_level':   current_level,
                'light_level':     light_level,
                'level_cleared':   level_cleared,
                'boss_vulnerable': boss_vulnerable,
            },
        }

    # ------------------------------------------------------------------
    # Text rendering for LLM
    # ------------------------------------------------------------------

    def to_text(self, parsed_obs: dict, agent_id: int) -> tuple[str, str]:
        """Convert parsed observation to human-readable text for an LLM.

        Returns:
            stat_text: Agent status, inventory, and visible teammates summary.
            obs_text:  Visible map tiles sorted by distance.
        """
        inv      = parsed_obs['inventory']
        map_info = parsed_obs['map']

        stat_text = f"""=== AGENT {agent_id} ===

HEALTH & RESOURCES:
- Health: {inv['stats']['health']:.1f}/10
- Food: {inv['stats']['food']:.1f}/9
- Drink: {inv['stats']['drink']:.1f}/9
- Energy: {inv['stats']['energy']:.1f}/9
- Mana: {inv['stats']['mana']:.1f}/9

STATUS:
- Current Level: {inv['state']['current_level']}/8
- Facing: {inv['state']['facing']}
- Light Level: {inv['state']['light_level']:.2f}
- Sleeping: {inv['state']['sleeping']} | Resting: {inv['state']['resting']}
- Level Cleared: {inv['state']['level_cleared']} | Boss Vulnerable: {inv['state']['boss_vulnerable']}

INVENTORY:
Materials: Wood={inv['materials']['wood']}, Stone={inv['materials']['stone']}, Coal={inv['materials']['coal']}, Iron={inv['materials']['iron']}, Diamond={inv['materials']['diamond']}
Gems: Sapphire={inv['materials']['sapphire']}, Ruby={inv['materials']['ruby']}
Items: Torches={inv['items']['torches']}, Arrows={inv['items']['arrows']}, Saplings={inv['items']['sapling']}
Equipment: Pickaxe Lvl {inv['equipment']['pickaxe_level']}, Sword Lvl {inv['equipment']['sword_level']}, Bow={inv['equipment']['has_bow']}
Armour: {inv['equipment']['armour_levels']}

ATTRIBUTES:
- XP: {inv['stats']['xp']:.1f} | Str: {inv['stats']['strength']:.1f} | Dex: {inv['stats']['dexterity']:.1f} | Int: {inv['stats']['intelligence']:.1f}

"""

        obs_text = "VISIBLE TILES (sorted by distance):\n"

        if not map_info['visible_tiles']:
            obs_text += "- Nothing visible (too dark)\n"
        else:
            sorted_tiles = sorted(
                map_info['visible_tiles'],
                key=lambda t: (t['manhattan_distance'], t['direction']),
            )

            current_dist = None
            for tile in sorted_tiles:
                dist = tile['manhattan_distance']
                rel_i, rel_j = tile['relative']
                rel_i_disp = rel_i * -1  # flip so north is positive

                if dist != current_dist:
                    current_dist = dist
                    if dist == 0:
                        obs_text += "\n[HERE - your current tile]:\n"
                    else:
                        obs_text += f"\n[Distance {dist} tiles away]:\n"

                if tile['direction'] == "here":
                    dir_desc = "HERE"
                    pos_desc = f"Facing: {inv['state']['facing']}"
                else:
                    dir_components = tile['direction'].split()
                    dir_desc = tile['direction'].replace(' ', '').upper()
                    parts = []
                    if len(dir_components) > 0 and dir_components[0]:
                        parts.append(f"{abs(rel_i_disp)} tiles {dir_components[0]}")
                    if len(dir_components) > 1 and dir_components[1]:
                        parts.append(f"{abs(rel_j)} tiles {dir_components[1]}")
                    pos_desc = "(" + ", ".join(parts) + ")"

                contents = ", ".join(tile['contents'])
                obs_text += f"  {dir_desc} {pos_desc}: {contents}\n"

        return stat_text, obs_text


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = CraftaxMAObservationParser(num_agents=2)

    # obs, state = env.reset(key)
    # parsed = parser.parse_observation(obs['agent_0'], agent_id=0)
    # stat_text, obs_text = parser.to_text(parsed, agent_id=0)
    # print(stat_text)
    # print(obs_text)
