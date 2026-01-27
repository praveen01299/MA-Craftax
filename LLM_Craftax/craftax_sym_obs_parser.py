# # import jax.numpy as jnp
# # import numpy as np
# # from enum import Enum

# # # Copy the enums from constants.py
# # class BlockType(Enum):
# #     INVALID = 0
# #     OUT_OF_BOUNDS = 1
# #     GRASS = 2
# #     WATER = 3
# #     STONE = 4
# #     TREE = 5
# #     WOOD = 6
# #     PATH = 7
# #     COAL = 8
# #     IRON = 9
# #     DIAMOND = 10
# #     CRAFTING_TABLE = 11
# #     FURNACE = 12
# #     SAND = 13
# #     LAVA = 14
# #     PLANT = 15
# #     RIPE_PLANT = 16
# #     WALL = 17
# #     DARKNESS = 18
# #     WALL_MOSS = 19
# #     STALAGMITE = 20
# #     SAPPHIRE = 21
# #     RUBY = 22
# #     CHEST = 23
# #     FOUNTAIN = 24
# #     FIRE_GRASS = 25
# #     ICE_GRASS = 26
# #     GRAVEL = 27
# #     FIRE_TREE = 28
# #     ICE_SHRUB = 29
# #     ENCHANTMENT_TABLE_FIRE = 30
# #     ENCHANTMENT_TABLE_ICE = 31
# #     NECROMANCER = 32
# #     GRAVE = 33
# #     GRAVE2 = 34
# #     GRAVE3 = 35
# #     NECROMANCER_VULNERABLE = 36

# # class ItemType(Enum):
# #     NONE = 0
# #     TORCH = 1
# #     LADDER_DOWN = 2
# #     LADDER_UP = 3
# #     LADDER_DOWN_BLOCKED = 4

# # class Specialization(Enum):
# #     UNASSIGNED = 0
# #     FORAGER = 1
# #     WARRIOR = 2
# #     MINER = 3

# # class Action(Enum):
# #     REQUEST_FOOD = 43
# #     REQUEST_DRINK = 44
# #     REQUEST_WOOD = 45
# #     REQUEST_STONE = 46
# #     REQUEST_IRON = 47
# #     REQUEST_COAL = 48
# #     REQUEST_DIAMOND = 49
# #     REQUEST_RUBY = 50
# #     REQUEST_SAPPHIRE = 51

# # OBS_DIM = (9, 11)

# # class CraftaxObservationParser:
# #     """Parse Craftax-Coop symbolic observations into human-readable text"""
    
# #     def __init__(self, num_agents=3):
# #         self.num_agents = num_agents
        
# #         # Calculate observation component sizes
# #         self.num_blocks = len(BlockType)
# #         self.num_items = len(ItemType)
# #         self.num_mob_classes = 5
# #         self.num_mob_types = 8
        
# #         # Map section: blocks, items, mobs, teammates, light
# #         self.map_size = OBS_DIM[0] * OBS_DIM[1] * (
# #             self.num_blocks + self.num_items + 
# #             self.num_mob_classes * self.num_mob_types + 
# #             num_agents + 1 + 1  # teammates + dead/alive bit + light
# #         )
        
# #         # Teammate dashboard: health, alive, specialization, requested material, direction
# #         self.teammate_dashboard_size = num_agents * (1 + 1 + 3 + 9 + 8)
        
# #         # Inventory and stats
# #         # self.inventory_size = 48  # Based on the observation shape calculation
        
# #     def parse_observation(self, obs, agent_id=0):
# #         """
# #         Parse a single agent's observation into structured data
        
# #         Args:
# #             obs: The flat observation array from the environment
# #             agent_id: Which agent this observation belongs to (0, 1, or 2)
            
# #         Returns:
# #             Dictionary with parsed observation components
# #         """
# #         obs = np.array(obs) if isinstance(obs, jnp.ndarray) else obs
        
# #         idx = 0
        
# #         # 1. Parse map view (9x11 grid)
# #         map_data = obs[idx:idx + self.map_size]
# #         idx += self.map_size
# #         map_parsed = self._parse_map(map_data)
        
# #         # 2. Parse teammate dashboard
# #         teammate_data = obs[idx:idx + self.teammate_dashboard_size]
# #         idx += self.teammate_dashboard_size
# #         teammates_parsed = self._parse_teammates(teammate_data)
        
# #         # 3. Parse inventory and stats
# #         # inventory_data = obs[idx:idx + self.inventory_size]
# #         # inventory_parsed = self._parse_inventory(inventory_data)
# #         # 3. Parse inventory and stats
# #         # Dynamically compute inventory size
# #         inventory_data = obs[idx:]
# #         inventory_parsed = self._parse_inventory(inventory_data)

# #         return {
# #             'agent_id': agent_id,
# #             'map': map_parsed,
# #             'teammates': teammates_parsed,
# #             'inventory': inventory_parsed,
# #             'raw_obs': obs
# #         }
    
# #     def _parse_map(self, map_data):
# #         """Parse the map view section"""
# #         # Reshape to (9, 11, num_features)
# #         num_features = self.num_blocks + self.num_items + self.num_mob_classes * self.num_mob_types + self.num_agents + 1 + 1
# #         map_grid = map_data.reshape(OBS_DIM[0], OBS_DIM[1], num_features)
        
# #         # Extract different layers
# #         blocks_layer = map_grid[:, :, :self.num_blocks]
# #         items_layer = map_grid[:, :, self.num_blocks:self.num_blocks + self.num_items]
        
# #         mob_start = self.num_blocks + self.num_items
# #         mob_end = mob_start + self.num_mob_classes * self.num_mob_types
# #         mobs_layer = map_grid[:, :, mob_start:mob_end]
        
# #         teammate_start = mob_end
# #         teammate_end = teammate_start + self.num_agents + 1
# #         teammates_layer = map_grid[:, :, teammate_start:teammate_end]
        
# #         light_layer = map_grid[:, :, -1]
        
# #         # Find visible tiles (where light > 0)
# #         visible_tiles = []
# #         for i in range(OBS_DIM[0]):
# #             for j in range(OBS_DIM[1]):
# #                 if light_layer[i, j] > 0:
# #                     tile_info = self._parse_tile(i, j, blocks_layer[i, j], items_layer[i, j], 
# #                                                    mobs_layer[i, j], teammates_layer[i, j])
# #                     if tile_info:
# #                         visible_tiles.append(tile_info)
        
# #         return {
# #             'visible_tiles': visible_tiles,
# #             'grid_shape': OBS_DIM,
# #             'center': (OBS_DIM[0] // 2, OBS_DIM[1] // 2)  # Player is at center
# #         }
    
# #     def _parse_tile(self, i, j, blocks, items, mobs, teammates):
# #         """Parse a single tile"""
# #         tile_contents = []
        
# #         # Get block type
# #         block_idx = np.argmax(blocks)
# #         block_type = list(BlockType)[block_idx]
# #         if block_type not in [BlockType.GRASS, BlockType.PATH, BlockType.INVALID]:
# #             tile_contents.append(block_type.name.lower().replace('_', ' '))
        
# #         # Get item
# #         item_idx = np.argmax(items)
# #         item_type = list(ItemType)[item_idx]
# #         if item_type != ItemType.NONE:
# #             tile_contents.append(item_type.name.lower().replace('_', ' '))
        
# #         # Get mobs (reshape to 5 classes x 8 types)
# #         mob_grid = mobs.reshape(self.num_mob_classes, self.num_mob_types)
# #         mob_types = ['melee', 'passive', 'ranged', 'mob_projectile', 'player_projectile']
# #         for class_idx, class_name in enumerate(mob_types):
# #             for type_idx in range(self.num_mob_types):
# #                 if mob_grid[class_idx, type_idx] > 0:
# #                     tile_contents.append(f"{class_name}_mob_{type_idx}")
        
# #         # Get teammates
# #         for t_idx in range(self.num_agents):
# #             if teammates[t_idx] > 0:
# #                 is_alive = teammates[-1] > 0
# #                 status = "alive" if is_alive else "dead"
# #                 tile_contents.append(f"teammate_{t_idx} ({status})")


        
# #         if tile_contents:
# #             # Convert grid coords to relative position (center is player)
# #             rel_i = i - OBS_DIM[0] // 2
# #             rel_j = j - OBS_DIM[1] // 2
# #             direction = self._get_direction(rel_i, rel_j)
            
# #             return {
# #                 'position': (i, j),
# #                 'relative': (rel_i, rel_j),
# #                 'direction': direction,
# #                 'contents': tile_contents
# #             }
# #         return None
    
# #     def _get_direction(self, di, dj):
# #         """Convert relative position to cardinal direction"""
# #         if di == 0 and dj == 0:
# #             return "here"
        
# #         vert = "north" if di < 0 else ("south" if di > 0 else "")
# #         horiz = "west" if dj < 0 else ("east" if dj > 0 else "")
        
# #         return f"{vert}{horiz}" if vert or horiz else "center"
    
# #     def _parse_teammates(self, teammate_data):
# #         """Parse teammate dashboard"""
# #         # Each teammate has: health(1) + alive(1) + specialization(3) + requested(9) + direction(8) = 22
# #         teammate_info = []
        
# #         per_teammate = 22
# #         for t_idx in range(self.num_agents):
# #             start = t_idx * per_teammate
            
# #             health = teammate_data[start] * 10.0  # Unscale
# #             alive = teammate_data[start + 1] > 0.5
            
# #             spec_onehot = teammate_data[start + 2:start + 5]
# #             spec_idx = np.argmax(spec_onehot)
# #             specialization = ['Forager', 'Warrior', 'Miner'][spec_idx]
            
# #             req_onehot = teammate_data[start + 5:start + 14]
# #             requesting = None
# #             if np.any(req_onehot > 0):
# #                 req_idx = np.argmax(req_onehot)
# #                 requests = ['food', 'drink', 'wood', 'stone', 'iron', 'coal', 'diamond', 'ruby', 'sapphire']
# #                 requesting = requests[req_idx]
            
# #             dir_onehot = teammate_data[start + 14:start + 22]
# #             direction = None
# #             if np.any(dir_onehot > 0):
# #                 # Direction encoding: 0=NW, 1=N, 2=NE, 3=W, 4=E, 5=SW, 6=S, 7=SE
# #                 dir_idx = np.argmax(dir_onehot)
# #                 directions = ['northwest', 'north', 'northeast', 'west', 'east', 'southwest', 'south', 'southeast']
# #                 direction = directions[dir_idx]
            
# #             teammate_info.append({
# #                 'id': t_idx,
# #                 'health': health,
# #                 'alive': alive,
# #                 'specialization': specialization,
# #                 'requesting': requesting,
# #                 'direction': direction
# #             })
        
# #         return teammate_info
    
# #     def _parse_inventory(self, inv_data):
# #         """Parse inventory and player stats"""
# #         idx = 0
        
# #         # Inventory items (16 items)
# #         wood = (inv_data[idx] * 10.0) ** 2; idx += 1
# #         stone = (inv_data[idx] * 10.0) ** 2; idx += 1
# #         coal = (inv_data[idx] * 10.0) ** 2; idx += 1
# #         iron = (inv_data[idx] * 10.0) ** 2; idx += 1
# #         diamond = (inv_data[idx] * 10.0) ** 2; idx += 1
# #         sapphire = (inv_data[idx] * 10.0) ** 2; idx += 1
# #         ruby = (inv_data[idx] * 10.0) ** 2; idx += 1
# #         sapling = (inv_data[idx] * 10.0) ** 2; idx += 1
# #         torches = (inv_data[idx] * 10.0) ** 2; idx += 1
# #         arrows = (inv_data[idx] * 10.0) ** 2; idx += 1
# #         books = inv_data[idx]; idx += 1
# #         pickaxe_level = int(inv_data[idx] * 4); idx += 1
# #         sword_level = int(inv_data[idx] * 4); idx += 1
# #         sword_enchant = inv_data[idx]; idx += 1
# #         bow_enchant = inv_data[idx]; idx += 1
# #         has_bow = inv_data[idx] > 0.5; idx += 1
        
# #         # Potions (6)
# #         potions = [(inv_data[idx + i] * 10.0) ** 2 for i in range(6)]
# #         idx += 6
        
# #         # Intrinsics (8) - Note: health is in teammate dashboard
# #         food = inv_data[idx] * 10.0; idx += 1
# #         drink = inv_data[idx] * 10.0; idx += 1
# #         energy = inv_data[idx] * 10.0; idx += 1
# #         mana = inv_data[idx] * 10.0; idx += 1
# #         xp = inv_data[idx] * 10.0; idx += 1
# #         dexterity = inv_data[idx] * 10.0; idx += 1
# #         strength = inv_data[idx] * 10.0; idx += 1
# #         intelligence = inv_data[idx] * 10.0; idx += 1
        
# #         # Direction (4)
# #         direction_onehot = inv_data[idx:idx+4]
# #         idx += 4
# #         directions = ['left', 'right', 'up', 'down']
# #         facing = directions[np.argmax(direction_onehot)]
        
# #         # Armour (4)
# #         armour = [int(inv_data[idx + i] * 2) for i in range(4)]
# #         idx += 4
        
# #         # Armour enchantments (4)
# #         armour_enchants = [inv_data[idx + i] for i in range(4)]
# #         idx += 4
        
# #         # Special values (3)
# #         is_sleeping = inv_data[idx] > 0.5; idx += 1
# #         is_resting = inv_data[idx] > 0.5; idx += 1
# #         learned_spell = inv_data[idx] > 0.5; idx += 1
        
# #         # Level values (4)
# #         light_level = inv_data[idx]; idx += 1
# #         current_level = int(inv_data[idx] * 10); idx += 1
# #         level_cleared = inv_data[idx] > 0.5; idx += 1
# #         boss_vulnerable = inv_data[idx] > 0.5; idx += 1
        
# #         return {
# #             'materials': {
# #                 'wood': int(wood),
# #                 'stone': int(stone),
# #                 'coal': int(coal),
# #                 'iron': int(iron),
# #                 'diamond': int(diamond),
# #                 'sapphire': int(sapphire),
# #                 'ruby': int(ruby),
# #             },
# #             'items': {
# #                 'sapling': int(sapling),
# #                 'torches': int(torches),
# #                 'arrows': int(arrows),
# #                 'books': int(books),
# #             },
# #             'equipment': {
# #                 'pickaxe_level': pickaxe_level,
# #                 'sword_level': sword_level,
# #                 'has_bow': has_bow,
# #                 'armour_levels': armour,
# #             },
# #             'stats': {
# #                 'food': food,
# #                 'drink': drink,
# #                 'energy': energy,
# #                 'mana': mana,
# #                 'xp': xp,
# #                 'dexterity': dexterity,
# #                 'strength': strength,
# #                 'intelligence': intelligence,
# #             },
# #             'state': {
# #                 'facing': facing,
# #                 'sleeping': is_sleeping,
# #                 'resting': is_resting,
# #                 'learned_spell': learned_spell,
# #                 'current_level': current_level,
# #                 'light_level': light_level,
# #                 'level_cleared': level_cleared,
# #                 'boss_vulnerable': boss_vulnerable,
# #             }
# #         }
    
# #     def to_text(self, parsed_obs, agent_id):
# #         """Convert parsed observation to human-readable text for LLM"""
        
# #         teammate_info = parsed_obs['teammates'][agent_id]
# #         inv = parsed_obs['inventory']
# #         map_info = parsed_obs['map']
        
# #         text = f"""=== AGENT {agent_id} ({teammate_info['specialization']}) ===

# # HEALTH & RESOURCES:
# # - Health: {teammate_info['health']:.1f}/14
# # - Food: {inv['stats']['food']:.1f}/17
# # - Drink: {inv['stats']['drink']:.1f}/17
# # - Energy: {inv['stats']['energy']:.1f}/17
# # - Mana: {inv['stats']['mana']:.1f}/21

# # LOCATION & STATUS:
# # - Current Level: {inv['state']['current_level']}/8
# # - Facing: {inv['state']['facing']}
# # - Light Level: {inv['state']['light_level']:.2f}
# # - Level Cleared: {inv['state']['level_cleared']}
# # - Sleeping: {inv['state']['sleeping']} | Resting: {inv['state']['resting']}

# # INVENTORY:
# # Materials: Wood={inv['materials']['wood']}, Stone={inv['materials']['stone']}, Coal={inv['materials']['coal']}, Iron={inv['materials']['iron']}, Diamond={inv['materials']['diamond']}
# # Items: Torches={inv['items']['torches']}, Arrows={inv['items']['arrows']}, Saplings={inv['items']['sapling']}
# # Equipment: Pickaxe Lvl {inv['equipment']['pickaxe_level']}, Sword Lvl {inv['equipment']['sword_level']}, Bow={inv['equipment']['has_bow']}

# # ATTRIBUTES:
# # - XP: {inv['stats']['xp']:.1f} | Dexterity: {inv['stats']['dexterity']:.1f} | Strength: {inv['stats']['strength']:.1f} | Intelligence: {inv['stats']['intelligence']:.1f}

# # TEAMMATES:
# # """
        
# #         for t in parsed_obs['teammates']:
# #             if t['id'] == agent_id:
# #                 continue
# #             status = "ALIVE" if t['alive'] else "DEAD"
# #             req_text = f", requesting {t['requesting']}" if t['requesting'] else ""
# #             dir_text = f", located {t['direction']}" if t['direction'] else ", nearby"
# #             text += f"- Agent {t['id']} ({t['specialization']}): {status}, Health {t['health']:.1f}{req_text}{dir_text}\n"
        
# #         text += "\nVISIBLE SURROUNDINGS:\n"
        
# #         if not map_info['visible_tiles']:
# #             text += "- Nothing visible (too dark)\n"
# #         else:
# #             # Group tiles by direction
# #             by_direction = {}
# #             for tile in map_info['visible_tiles']:
# #                 direction = tile['direction']
# #                 if direction not in by_direction:
# #                     by_direction[direction] = []
# #                 by_direction[direction].append(tile)
            
# #             for direction in ['here', 'north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest']:
# #                 if direction in by_direction:
# #                     text += f"\n{direction.upper()}:\n"
# #                     for tile in by_direction[direction]:
# #                         contents = ", ".join(tile['contents'])
# #                         text += f"  - {contents}\n"
        
# #         return text


# # # Example usage:
# # if __name__ == "__main__":
# #     parser = CraftaxObservationParser(num_agents=3)
    
# #     # Example: parse observation from environment
# #     # obs, state = env.reset(key)
# #     # parsed = parser.parse_observation(obs['agent_0'], agent_id=0)
# #     # text = parser.to_text(parsed, agent_id=0)
# #     # print(text)


# import jax.numpy as jnp
# import numpy as np
# from enum import Enum

# # Copy the enums from constants.py
# class BlockType(Enum):
#     INVALID = 0
#     OUT_OF_BOUNDS = 1
#     GRASS = 2
#     WATER = 3
#     STONE = 4
#     TREE = 5
#     WOOD = 6
#     PATH = 7
#     COAL = 8
#     IRON = 9
#     DIAMOND = 10
#     CRAFTING_TABLE = 11
#     FURNACE = 12
#     SAND = 13
#     LAVA = 14
#     PLANT = 15
#     RIPE_PLANT = 16
#     WALL = 17
#     DARKNESS = 18
#     WALL_MOSS = 19
#     STALAGMITE = 20
#     SAPPHIRE = 21
#     RUBY = 22
#     CHEST = 23
#     FOUNTAIN = 24
#     FIRE_GRASS = 25
#     ICE_GRASS = 26
#     GRAVEL = 27
#     FIRE_TREE = 28
#     ICE_SHRUB = 29
#     ENCHANTMENT_TABLE_FIRE = 30
#     ENCHANTMENT_TABLE_ICE = 31
#     NECROMANCER = 32
#     GRAVE = 33
#     GRAVE2 = 34
#     GRAVE3 = 35
#     NECROMANCER_VULNERABLE = 36

# class ItemType(Enum):
#     NONE = 0
#     TORCH = 1
#     LADDER_DOWN = 2
#     LADDER_UP = 3
#     LADDER_DOWN_BLOCKED = 4

# class Specialization(Enum):
#     UNASSIGNED = 0
#     FORAGER = 1
#     WARRIOR = 2
#     MINER = 3

# class Action(Enum):
#     REQUEST_FOOD = 43
#     REQUEST_DRINK = 44
#     REQUEST_WOOD = 45
#     REQUEST_STONE = 46
#     REQUEST_IRON = 47
#     REQUEST_COAL = 48
#     REQUEST_DIAMOND = 49
#     REQUEST_RUBY = 50
#     REQUEST_SAPPHIRE = 51

# OBS_DIM = (9, 11)

# class CraftaxObservationParser:
#     """Parse Craftax-Coop symbolic observations into human-readable text"""
    
#     def __init__(self, num_agents=3):
#         self.num_agents = num_agents
        
#         # Calculate observation component sizes
#         self.num_blocks = len(BlockType)
#         self.num_items = len(ItemType)
#         self.num_mob_classes = 5
#         self.num_mob_types = 8
        
#         # Map section: blocks, items, mobs, teammates, light
#         self.map_size = OBS_DIM[0] * OBS_DIM[1] * (
#             self.num_blocks + self.num_items + 
#             self.num_mob_classes * self.num_mob_types + 
#             num_agents + 1 + 1  # teammates + dead/alive bit + light
#         )
        
#         # Teammate dashboard: health, alive, specialization, requested material
#         # NOTE: Directions are stored separately!
#         self.teammate_dashboard_size = num_agents * (1 + 1 + 3 + 9)
#         self.teammate_directions_size = num_agents * 8
        
#         # Inventory and stats
#         self.inventory_size = 48  # Based on the observation shape calculation
        
#     def parse_observation(self, obs, agent_id=0):
#         """
#         Parse a single agent's observation into structured data
        
#         Args:
#             obs: The flat observation array from the environment
#             agent_id: Which agent this observation belongs to (0, 1, or 2)
            
#         Returns:
#             Dictionary with parsed observation components
#         """
#         obs = np.array(obs) if isinstance(obs, jnp.ndarray) else obs
        
#         idx = 0
        
#         # 1. Parse map view (9x11 grid)
#         map_data = obs[idx:idx + self.map_size]
#         idx += self.map_size
#         map_parsed = self._parse_map(map_data)
        
#         # 2. Parse teammate dashboard
#         teammate_data = obs[idx:idx + self.teammate_dashboard_size]
#         idx += self.teammate_dashboard_size
        
#         # 3. Parse teammate directions (separate from dashboard!)
#         teammate_directions_data = obs[idx:idx + self.teammate_directions_size]
#         idx += self.teammate_directions_size
#         teammates_parsed = self._parse_teammates(teammate_data, teammate_directions_data, agent_id)
        
#         # 4. Parse inventory and stats
#         inventory_data = obs[idx:idx + self.inventory_size]
#         inventory_parsed = self._parse_inventory(inventory_data)
        
#         return {
#             'agent_id': agent_id,
#             'map': map_parsed,
#             'teammates': teammates_parsed,
#             'inventory': inventory_parsed,
#             'raw_obs': obs
#         }
    
#     def _parse_map(self, map_data):
#         """Parse the map view section"""
#         # Reshape to (9, 11, num_features)
#         num_features = self.num_blocks + self.num_items + self.num_mob_classes * self.num_mob_types + self.num_agents + 1 + 1
#         map_grid = map_data.reshape(OBS_DIM[0], OBS_DIM[1], num_features)
        
#         # Extract different layers
#         blocks_layer = map_grid[:, :, :self.num_blocks]
#         items_layer = map_grid[:, :, self.num_blocks:self.num_blocks + self.num_items]
        
#         mob_start = self.num_blocks + self.num_items
#         mob_end = mob_start + self.num_mob_classes * self.num_mob_types
#         mobs_layer = map_grid[:, :, mob_start:mob_end]
        
#         teammate_start = mob_end
#         teammate_end = teammate_start + self.num_agents + 1
#         teammates_layer = map_grid[:, :, teammate_start:teammate_end]
        
#         light_layer = map_grid[:, :, -1]
        
#         # Find visible tiles (where light > 0)
#         visible_tiles = []
#         for i in range(OBS_DIM[0]):
#             for j in range(OBS_DIM[1]):
#                 if light_layer[i, j] > 0:
#                     tile_info = self._parse_tile(i, j, blocks_layer[i, j], items_layer[i, j], 
#                                                    mobs_layer[i, j], teammates_layer[i, j])
#                     if tile_info:
#                         visible_tiles.append(tile_info)
        
#         return {
#             'visible_tiles': visible_tiles,
#             'grid_shape': OBS_DIM,
#             'center': (OBS_DIM[0] // 2, OBS_DIM[1] // 2)  # Player is at center
#         }
    
#     def _parse_tile(self, i, j, blocks, items, mobs, teammates):
#         """Parse a single tile"""
#         tile_contents = []
        
#         # Get block type
#         block_idx = np.argmax(blocks)
#         block_type = list(BlockType)[block_idx]
#         if block_type not in [BlockType.GRASS, BlockType.PATH, BlockType.INVALID]:
#             tile_contents.append(block_type.name.lower().replace('_', ' '))
        
#         # Get item
#         item_idx = np.argmax(items)
#         item_type = list(ItemType)[item_idx]
#         if item_type != ItemType.NONE:
#             tile_contents.append(item_type.name.lower().replace('_', ' '))
        
#         # Get mobs (reshape to 5 classes x 8 types)
#         mob_grid = mobs.reshape(self.num_mob_classes, self.num_mob_types)
#         mob_types = ['melee', 'passive', 'ranged', 'mob_projectile', 'player_projectile']
#         for class_idx, class_name in enumerate(mob_types):
#             for type_idx in range(self.num_mob_types):
#                 if mob_grid[class_idx, type_idx] > 0:
#                     tile_contents.append(f"{class_name}_mob_{type_idx}")
        
#         # Get teammates
#         for t_idx in range(self.num_agents):
#             if teammates[t_idx] > 0:
#                 is_alive = teammates[-1] > 0
#                 status = "alive" if is_alive else "dead"
#                 tile_contents.append(f"teammate_{t_idx} ({status})")
        
#         if tile_contents:
#             # Convert grid coords to relative position (center is player)
#             rel_i = i - OBS_DIM[0] // 2
#             rel_j = j - OBS_DIM[1] // 2
#             direction = self._get_direction(rel_i, rel_j)
            
#             return {
#                 'position': (i, j),
#                 'relative': (rel_i, rel_j),
#                 'direction': direction,
#                 'contents': tile_contents
#             }
#         return None
    
#     def _get_direction(self, di, dj):
#         """Convert relative position to cardinal direction"""
#         if di == 0 and dj == 0:
#             return "here"
        
#         vert = "north" if di < 0 else ("south" if di > 0 else "")
#         horiz = "west" if dj < 0 else ("east" if dj > 0 else "")
        
#         return f"{vert}{horiz}" if vert or horiz else "center"
    
#     def _parse_teammates(self, teammate_data, teammate_directions_data, current_agent_id):
#         """Parse teammate dashboard
        
#         IMPORTANT: The teammate data is REORDERED per agent!
#         - Index 0 = current agent's own info
#         - Index 1+ = other agents (maintaining relative order)
        
#         We need to map back to absolute agent IDs.
#         """
#         # Each teammate has: health(1) + alive(1) + specialization(3) + requested(9) = 14
#         teammate_info_reordered = []
        
#         per_teammate = 14  # Not 22! Directions are separate
#         for reordered_idx in range(self.num_agents):
#             start = reordered_idx * per_teammate
            
#             health = teammate_data[start] * 10.0  # Unscale
#             alive = teammate_data[start + 1] > 0.5
            
#             spec_onehot = teammate_data[start + 2:start + 5]
#             spec_idx = np.argmax(spec_onehot)
#             specialization = ['Forager', 'Warrior', 'Miner'][spec_idx]
            
#             req_onehot = teammate_data[start + 5:start + 14]
#             requesting = None
#             if np.any(req_onehot > 0):
#                 req_idx = np.argmax(req_onehot)
#                 requests = ['food', 'drink', 'wood', 'stone', 'iron', 'coal', 'diamond', 'ruby', 'sapphire']
#                 requesting = requests[req_idx]
            
#             teammate_info_reordered.append({
#                 'reordered_idx': reordered_idx,
#                 'health': health,
#                 'alive': alive,
#                 'specialization': specialization,
#                 'requesting': requesting,
#             })
        
#         # Parse directions separately
#         directions_per_teammate = 8
#         teammate_directions = []
#         for reordered_idx in range(self.num_agents):
#             dir_start = reordered_idx * directions_per_teammate
#             dir_onehot = teammate_directions_data[dir_start:dir_start + directions_per_teammate]
#             direction = None
#             if np.any(dir_onehot > 0):
#                 # Direction encoding: 0=NW, 1=N, 2=NE, 3=W, 4=E, 5=SW, 6=S, 7=SE
#                 dir_idx = np.argmax(dir_onehot)
#                 directions = ['northwest', 'north', 'northeast', 'west', 'east', 'southwest', 'south', 'southeast']
#                 direction = directions[dir_idx]
#             teammate_directions.append(direction)
        
#         # Reconstruct absolute ordering
#         # The reordering logic from the code:
#         # i1 = (arange == 0) * current_agent_id  --> index 0 gets current agent
#         # i2 = (arange > 0 & arange <= current_agent_id) * (arange - 1)  --> indices before current agent shift down
#         # i3 = (arange > current_agent_id) * arange  --> indices after stay the same
        
#         teammate_info = [None] * self.num_agents
#         for reordered_idx, info in enumerate(teammate_info_reordered):
#             if reordered_idx == 0:
#                 # Index 0 always contains current agent's data
#                 absolute_idx = current_agent_id
#             elif reordered_idx <= current_agent_id:
#                 # These are agents before the current agent
#                 absolute_idx = reordered_idx - 1
#             else:
#                 # These are agents after the current agent
#                 absolute_idx = reordered_idx
            
#             teammate_info[absolute_idx] = {
#                 'id': absolute_idx,
#                 'health': info['health'],
#                 'alive': info['alive'],
#                 'specialization': info['specialization'],
#                 'requesting': info['requesting'],
#                 'direction': teammate_directions[reordered_idx]
#             }
        
#         return teammate_info
    
#     def _parse_inventory(self, inv_data):
#         """Parse inventory and player stats"""
#         idx = 0
        
#         # Inventory items (16 items)
#         wood = (inv_data[idx] * 10.0) ** 2; idx += 1
#         stone = (inv_data[idx] * 10.0) ** 2; idx += 1
#         coal = (inv_data[idx] * 10.0) ** 2; idx += 1
#         iron = (inv_data[idx] * 10.0) ** 2; idx += 1
#         diamond = (inv_data[idx] * 10.0) ** 2; idx += 1
#         sapphire = (inv_data[idx] * 10.0) ** 2; idx += 1
#         ruby = (inv_data[idx] * 10.0) ** 2; idx += 1
#         sapling = (inv_data[idx] * 10.0) ** 2; idx += 1
#         torches = (inv_data[idx] * 10.0) ** 2; idx += 1
#         arrows = (inv_data[idx] * 10.0) ** 2; idx += 1
#         books = inv_data[idx]; idx += 1
#         pickaxe_level = int(inv_data[idx] * 4); idx += 1
#         sword_level = int(inv_data[idx] * 4); idx += 1
#         sword_enchant = inv_data[idx]; idx += 1
#         bow_enchant = inv_data[idx]; idx += 1
#         has_bow = inv_data[idx] > 0.5; idx += 1
        
#         # Potions (6)
#         potions = [(inv_data[idx + i] * 10.0) ** 2 for i in range(6)]
#         idx += 6
        
#         # Intrinsics (8) - Note: health is in teammate dashboard
#         food = inv_data[idx] * 10.0; idx += 1
#         drink = inv_data[idx] * 10.0; idx += 1
#         energy = inv_data[idx] * 10.0; idx += 1
#         mana = inv_data[idx] * 10.0; idx += 1
#         xp = inv_data[idx] * 10.0; idx += 1
#         dexterity = inv_data[idx] * 10.0; idx += 1
#         strength = inv_data[idx] * 10.0; idx += 1
#         intelligence = inv_data[idx] * 10.0; idx += 1
        
#         # Direction (4)
#         direction_onehot = inv_data[idx:idx+4]
#         idx += 4
#         directions = ['left', 'right', 'up', 'down']
#         facing = directions[np.argmax(direction_onehot)]
        
#         # Armour (4)
#         armour = [int(inv_data[idx + i] * 2) for i in range(4)]
#         idx += 4
        
#         # Armour enchantments (4)
#         armour_enchants = [inv_data[idx + i] for i in range(4)]
#         idx += 4
        
#         # Special values (3)
#         is_sleeping = inv_data[idx] > 0.5; idx += 1
#         is_resting = inv_data[idx] > 0.5; idx += 1
#         learned_spell = inv_data[idx] > 0.5; idx += 1
        
#         # Level values (4)
#         light_level = inv_data[idx]; idx += 1
#         current_level = int(inv_data[idx] * 10); idx += 1
#         level_cleared = inv_data[idx] > 0.5; idx += 1
#         boss_vulnerable = inv_data[idx] > 0.5; idx += 1
        
#         return {
#             'materials': {
#                 'wood': int(wood),
#                 'stone': int(stone),
#                 'coal': int(coal),
#                 'iron': int(iron),
#                 'diamond': int(diamond),
#                 'sapphire': int(sapphire),
#                 'ruby': int(ruby),
#             },
#             'items': {
#                 'sapling': int(sapling),
#                 'torches': int(torches),
#                 'arrows': int(arrows),
#                 'books': int(books),
#             },
#             'equipment': {
#                 'pickaxe_level': pickaxe_level,
#                 'sword_level': sword_level,
#                 'has_bow': has_bow,
#                 'armour_levels': armour,
#             },
#             'stats': {
#                 'food': food,
#                 'drink': drink,
#                 'energy': energy,
#                 'mana': mana,
#                 'xp': xp,
#                 'dexterity': dexterity,
#                 'strength': strength,
#                 'intelligence': intelligence,
#             },
#             'state': {
#                 'facing': facing,
#                 'sleeping': is_sleeping,
#                 'resting': is_resting,
#                 'learned_spell': learned_spell,
#                 'current_level': current_level,
#                 'light_level': light_level,
#                 'level_cleared': level_cleared,
#                 'boss_vulnerable': boss_vulnerable,
#             }
#         }
    
#     def to_text(self, parsed_obs, agent_id):
#         """Convert parsed observation to human-readable text for LLM"""
        
#         teammate_info = parsed_obs['teammates'][agent_id]
#         inv = parsed_obs['inventory']
#         map_info = parsed_obs['map']
        
#         text = f"""=== AGENT {agent_id} ({teammate_info['specialization']}) ===

# HEALTH & RESOURCES:
# - Health: {teammate_info['health']:.1f}/14
# - Food: {inv['stats']['food']:.1f}/17
# - Drink: {inv['stats']['drink']:.1f}/17
# - Energy: {inv['stats']['energy']:.1f}/17
# - Mana: {inv['stats']['mana']:.1f}/21

# LOCATION & STATUS:
# - Current Level: {inv['state']['current_level']}/8
# - Facing: {inv['state']['facing']}
# - Light Level: {inv['state']['light_level']:.2f}
# - Level Cleared: {inv['state']['level_cleared']}
# - Sleeping: {inv['state']['sleeping']} | Resting: {inv['state']['resting']}

# INVENTORY:
# Materials: Wood={inv['materials']['wood']}, Stone={inv['materials']['stone']}, Coal={inv['materials']['coal']}, Iron={inv['materials']['iron']}, Diamond={inv['materials']['diamond']}
# Items: Torches={inv['items']['torches']}, Arrows={inv['items']['arrows']}, Saplings={inv['items']['sapling']}
# Equipment: Pickaxe Lvl {inv['equipment']['pickaxe_level']}, Sword Lvl {inv['equipment']['sword_level']}, Bow={inv['equipment']['has_bow']}

# ATTRIBUTES:
# - XP: {inv['stats']['xp']:.1f} | Dexterity: {inv['stats']['dexterity']:.1f} | Strength: {inv['stats']['strength']:.1f} | Intelligence: {inv['stats']['intelligence']:.1f}

# TEAMMATES:
# """
        
#         for t in parsed_obs['teammates']:
#             if t['id'] == agent_id:
#                 continue
#             status = "ALIVE" if t['alive'] else "DEAD"
#             req_text = f", requesting {t['requesting']}" if t['requesting'] else ""
#             dir_text = f", located {t['direction']}" if t['direction'] else ", nearby"
#             text += f"- Agent {t['id']} ({t['specialization']}): {status}, Health {t['health']:.1f}{req_text}{dir_text}\n"
        
#         text += "\nVISIBLE SURROUNDINGS:\n"
        
#         if not map_info['visible_tiles']:
#             text += "- Nothing visible (too dark)\n"
#         else:
#             # Group tiles by direction
#             by_direction = {}
#             for tile in map_info['visible_tiles']:
#                 direction = tile['direction']
#                 if direction not in by_direction:
#                     by_direction[direction] = []
#                 by_direction[direction].append(tile)
            
#             for direction in ['here', 'north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest']:
#                 if direction in by_direction:
#                     text += f"\n{direction.upper()}:\n"
#                     for tile in by_direction[direction]:
#                         contents = ", ".join(tile['contents'])
#                         text += f"  - {contents}\n"
        
#         return text


# # Example usage:
# if __name__ == "__main__":
#     parser = CraftaxObservationParser(num_agents=3)
    
#     # Example: parse observation from environment
#     # obs, state = env.reset(key)
#     # parsed = parser.parse_observation(obs['agent_0'], agent_id=0)
#     # text = parser.to_text(parsed, agent_id=0)
#     # print(text)

# import jax.numpy as jnp
# import numpy as np
# from enum import Enum

# # Copy the enums from constants.py
# class BlockType(Enum):
#     INVALID = 0
#     OUT_OF_BOUNDS = 1
#     GRASS = 2
#     WATER = 3
#     STONE = 4
#     TREE = 5
#     WOOD = 6
#     PATH = 7
#     COAL = 8
#     IRON = 9
#     DIAMOND = 10
#     CRAFTING_TABLE = 11
#     FURNACE = 12
#     SAND = 13
#     LAVA = 14
#     PLANT = 15
#     RIPE_PLANT = 16
#     WALL = 17
#     DARKNESS = 18
#     WALL_MOSS = 19
#     STALAGMITE = 20
#     SAPPHIRE = 21
#     RUBY = 22
#     CHEST = 23
#     FOUNTAIN = 24
#     FIRE_GRASS = 25
#     ICE_GRASS = 26
#     GRAVEL = 27
#     FIRE_TREE = 28
#     ICE_SHRUB = 29
#     ENCHANTMENT_TABLE_FIRE = 30
#     ENCHANTMENT_TABLE_ICE = 31
#     NECROMANCER = 32
#     GRAVE = 33
#     GRAVE2 = 34
#     GRAVE3 = 35
#     NECROMANCER_VULNERABLE = 36

# class ItemType(Enum):
#     NONE = 0
#     TORCH = 1
#     LADDER_DOWN = 2
#     LADDER_UP = 3
#     LADDER_DOWN_BLOCKED = 4

# class Specialization(Enum):
#     UNASSIGNED = 0
#     FORAGER = 1
#     WARRIOR = 2
#     MINER = 3

# class Action(Enum):
#     REQUEST_FOOD = 43
#     REQUEST_DRINK = 44
#     REQUEST_WOOD = 45
#     REQUEST_STONE = 46
#     REQUEST_IRON = 47
#     REQUEST_COAL = 48
#     REQUEST_DIAMOND = 49
#     REQUEST_RUBY = 50
#     REQUEST_SAPPHIRE = 51

# OBS_DIM = (9, 11)

# class CraftaxObservationParser:
#     """Parse Craftax-Coop symbolic observations into human-readable text"""
    
#     def __init__(self, num_agents=3):
#         self.num_agents = num_agents
        
#         # Calculate observation component sizes
#         self.num_blocks = len(BlockType)
#         self.num_items = len(ItemType)
#         self.num_mob_classes = 5
#         self.num_mob_types = 8
        
#         # Map section: blocks, items, mobs, teammates, light
#         self.map_size = OBS_DIM[0] * OBS_DIM[1] * (
#             self.num_blocks + self.num_items + 
#             self.num_mob_classes * self.num_mob_types + 
#             num_agents + 1 + 1  # teammates + dead/alive bit + light
#         )
        
#         # Teammate dashboard: health, alive, specialization, requested material
#         # NOTE: Directions are stored separately!
#         self.teammate_dashboard_size = num_agents * (1 + 1 + 3 + 9)
#         self.teammate_directions_size = num_agents * 8
        
#         # Inventory and stats
#         self.inventory_size = 48  # Based on the observation shape calculation
        
#     def parse_observation(self, obs, agent_id=0):
#         """
#         Parse a single agent's observation into structured data
        
#         Args:
#             obs: The flat observation array from the environment
#             agent_id: Which agent this observation belongs to (0, 1, or 2)
            
#         Returns:
#             Dictionary with parsed observation components
#         """
#         obs = np.array(obs) if isinstance(obs, jnp.ndarray) else obs
        
#         idx = 0
        
#         # 1. Parse map view (9x11 grid)
#         map_data = obs[idx:idx + self.map_size]
#         idx += self.map_size
#         map_parsed = self._parse_map(map_data)
        
#         # 2. Parse teammate dashboard
#         teammate_data = obs[idx:idx + self.teammate_dashboard_size]
#         idx += self.teammate_dashboard_size
        
#         # 3. Parse teammate directions (separate from dashboard!)
#         teammate_directions_data = obs[idx:idx + self.teammate_directions_size]
#         idx += self.teammate_directions_size
#         teammates_parsed = self._parse_teammates(teammate_data, teammate_directions_data, agent_id)
        
#         # 4. Parse inventory and stats
#         inventory_data = obs[idx:idx + self.inventory_size]
#         inventory_parsed = self._parse_inventory(inventory_data)
        
#         return {
#             'agent_id': agent_id,
#             'map': map_parsed,
#             'teammates': teammates_parsed,
#             'inventory': inventory_parsed,
#             'raw_obs': obs
#         }
    
#     def _parse_map(self, map_data):
#         """Parse the map view section"""
#         # Reshape to (9, 11, num_features)
#         num_features = self.num_blocks + self.num_items + self.num_mob_classes * self.num_mob_types + self.num_agents + 1 + 1
#         map_grid = map_data.reshape(OBS_DIM[0], OBS_DIM[1], num_features)
        
#         # Extract different layers
#         blocks_layer = map_grid[:, :, :self.num_blocks]
#         items_layer = map_grid[:, :, self.num_blocks:self.num_blocks + self.num_items]
        
#         mob_start = self.num_blocks + self.num_items
#         mob_end = mob_start + self.num_mob_classes * self.num_mob_types
#         mobs_layer = map_grid[:, :, mob_start:mob_end]
        
#         teammate_start = mob_end
#         teammate_end = teammate_start + self.num_agents + 1
#         teammates_layer = map_grid[:, :, teammate_start:teammate_end]
        
#         light_layer = map_grid[:, :, -1]
        
#         # Find visible tiles (where light > 0)
#         visible_tiles = []
#         for i in range(OBS_DIM[0]):
#             for j in range(OBS_DIM[1]):
#                 if light_layer[i, j] > 0:
#                     tile_info = self._parse_tile(i, j, blocks_layer[i, j], items_layer[i, j], 
#                                                    mobs_layer[i, j], teammates_layer[i, j])
#                     if tile_info:
#                         visible_tiles.append(tile_info)
        
#         return {
#             'visible_tiles': visible_tiles,
#             'grid_shape': OBS_DIM,
#             'center': (OBS_DIM[0] // 2, OBS_DIM[1] // 2)  # Player is at center
#         }
    
#     def _parse_tile(self, i, j, blocks, items, mobs, teammates):
#         """Parse a single tile"""
#         tile_contents = []
        
#         # Get block type
#         block_idx = np.argmax(blocks)
#         block_type = list(BlockType)[block_idx]
#         if block_type not in [BlockType.GRASS, BlockType.PATH, BlockType.INVALID]:
#             tile_contents.append(block_type.name.lower().replace('_', ' '))
        
#         # Get item
#         item_idx = np.argmax(items)
#         item_type = list(ItemType)[item_idx]
#         if item_type != ItemType.NONE:
#             tile_contents.append(item_type.name.lower().replace('_', ' '))
        
#         # Get mobs (reshape to 5 classes x 8 types)
#         mob_grid = mobs.reshape(self.num_mob_classes, self.num_mob_types)
#         mob_types = ['melee', 'passive', 'ranged', 'mob_projectile', 'player_projectile']
#         for class_idx, class_name in enumerate(mob_types):
#             for type_idx in range(self.num_mob_types):
#                 if mob_grid[class_idx, type_idx] > 0:
#                     tile_contents.append(f"{class_name}_mob_{type_idx}")
        
#         # Get teammates
#         for t_idx in range(self.num_agents):
#             if teammates[t_idx] > 0:
#                 is_alive = teammates[-1] > 0
#                 status = "alive" if is_alive else "dead"
#                 tile_contents.append(f"teammate_{t_idx} ({status})")
        
#         if tile_contents:
#             # Convert grid coords to relative position (center is player)
#             rel_i = i - OBS_DIM[0] // 2
#             rel_j = j - OBS_DIM[1] // 2
#             direction = self._get_direction(rel_i, rel_j)
            
#             return {
#                 'position': (i, j),
#                 'relative': (rel_i, rel_j),
#                 'direction': direction,
#                 'contents': tile_contents
#             }
#         return None
    
#     def _get_direction(self, di, dj):
#         """Convert relative position to cardinal direction"""
#         if di == 0 and dj == 0:
#             return "here"
        
#         vert = "north" if di < 0 else ("south" if di > 0 else "")
#         horiz = "west" if dj < 0 else ("east" if dj > 0 else "")
        
#         return f"{vert}{horiz}" if vert or horiz else "center"
    
#     def _parse_teammates(self, teammate_data, teammate_directions_data, current_agent_id):
#         """Parse teammate dashboard
        
#         IMPORTANT: The teammate data is REORDERED per agent!
#         - Index 0 = current agent's own info
#         - Index 1+ = other agents (maintaining relative order)
        
#         We need to map back to absolute agent IDs.
#         """
#         # Each teammate has: health(1) + alive(1) + specialization(3) + requested(9) = 14
#         teammate_info_reordered = []
        
#         per_teammate = 14  # Not 22! Directions are separate
#         for reordered_idx in range(self.num_agents):
#             start = reordered_idx * per_teammate
            
#             health = teammate_data[start] * 10.0  # Unscale
#             alive = teammate_data[start + 1] > 0.5
            
#             spec_onehot = teammate_data[start + 2:start + 5]
#             spec_idx = np.argmax(spec_onehot)
#             specialization = ['Forager', 'Warrior', 'Miner'][spec_idx]
            
#             req_onehot = teammate_data[start + 5:start + 14]
#             requesting = None
#             if np.any(req_onehot > 0):
#                 req_idx = np.argmax(req_onehot)
#                 requests = ['food', 'drink', 'wood', 'stone', 'iron', 'coal', 'diamond', 'ruby', 'sapphire']
#                 requesting = requests[req_idx]
            
#             teammate_info_reordered.append({
#                 'reordered_idx': reordered_idx,
#                 'health': health,
#                 'alive': alive,
#                 'specialization': specialization,
#                 'requesting': requesting,
#             })
        
#         # Parse directions separately
#         directions_per_teammate = 8
#         teammate_directions = []
#         for reordered_idx in range(self.num_agents):
#             dir_start = reordered_idx * directions_per_teammate
#             dir_onehot = teammate_directions_data[dir_start:dir_start + directions_per_teammate]
#             direction = None
#             if np.any(dir_onehot > 0):
#                 # Direction encoding: 0=NW, 1=N, 2=NE, 3=W, 4=E, 5=SW, 6=S, 7=SE
#                 dir_idx = np.argmax(dir_onehot)
#                 directions = ['northwest', 'north', 'northeast', 'west', 'east', 'southwest', 'south', 'southeast']
#                 direction = directions[dir_idx]
#             teammate_directions.append(direction)
        
#         # Reconstruct absolute ordering
#         # The reordering logic from the code:
#         # i1 = (arange == 0) * current_agent_id  --> index 0 gets current agent
#         # i2 = (arange > 0 & arange <= current_agent_id) * (arange - 1)  --> indices before current agent shift down
#         # i3 = (arange > current_agent_id) * arange  --> indices after stay the same
        
#         teammate_info = [None] * self.num_agents
#         for reordered_idx, info in enumerate(teammate_info_reordered):
#             if reordered_idx == 0:
#                 # Index 0 always contains current agent's data
#                 absolute_idx = current_agent_id
#             elif reordered_idx <= current_agent_id:
#                 # These are agents before the current agent
#                 absolute_idx = reordered_idx - 1
#             else:
#                 # These are agents after the current agent
#                 absolute_idx = reordered_idx
            
#             teammate_info[absolute_idx] = {
#                 'id': absolute_idx,
#                 'health': info['health'],
#                 'alive': info['alive'],
#                 'specialization': info['specialization'],
#                 'requesting': info['requesting'],
#                 'direction': teammate_directions[reordered_idx]
#             }
        
#         return teammate_info
    
#     def _parse_inventory(self, inv_data):
#         """Parse inventory and player stats"""
#         idx = 0
        
#         try:
#             # Inventory items (16 items)
#             wood = (inv_data[idx] * 10.0) ** 2; idx += 1
#             stone = (inv_data[idx] * 10.0) ** 2; idx += 1
#             coal = (inv_data[idx] * 10.0) ** 2; idx += 1
#             iron = (inv_data[idx] * 10.0) ** 2; idx += 1
#             diamond = (inv_data[idx] * 10.0) ** 2; idx += 1
#             sapphire = (inv_data[idx] * 10.0) ** 2; idx += 1
#             ruby = (inv_data[idx] * 10.0) ** 2; idx += 1
#             sapling = (inv_data[idx] * 10.0) ** 2; idx += 1
#             torches = (inv_data[idx] * 10.0) ** 2; idx += 1
#             arrows = (inv_data[idx] * 10.0) ** 2; idx += 1
#             books = inv_data[idx]; idx += 1
#             pickaxe_level = int(inv_data[idx] * 4); idx += 1
#             sword_level = int(inv_data[idx] * 4); idx += 1
#             sword_enchant = inv_data[idx]; idx += 1
#             bow_enchant = inv_data[idx]; idx += 1
#             has_bow = inv_data[idx] > 0.5; idx += 1
#             # idx should be 16 here
            
#             # Potions (6)
#             potions = [(inv_data[idx + i] * 10.0) ** 2 for i in range(6)]
#             idx += 6
#             # idx should be 22 here
            
#             # Intrinsics (8) - Note: health is in teammate dashboard
#             food = inv_data[idx] * 10.0; idx += 1
#             drink = inv_data[idx] * 10.0; idx += 1
#             energy = inv_data[idx] * 10.0; idx += 1
#             mana = inv_data[idx] * 10.0; idx += 1
#             xp = inv_data[idx] * 10.0; idx += 1
#             dexterity = inv_data[idx] * 10.0; idx += 1
#             strength = inv_data[idx] * 10.0; idx += 1
#             intelligence = inv_data[idx] * 10.0; idx += 1
#             # idx should be 30 here
            
#             # Direction (4)
#             direction_onehot = inv_data[idx:idx+4]
#             idx += 4
#             directions = ['left', 'right', 'up', 'down']
#             facing = directions[np.argmax(direction_onehot)]
#             # idx should be 34 here
            
#             # Armour (4)
#             armour = [int(inv_data[idx + i] * 2) for i in range(4)]
#             idx += 4
#             # idx should be 38 here
            
#             # Armour enchantments (4)
#             armour_enchants = [inv_data[idx + i] for i in range(4)]
#             idx += 4
#             # idx should be 42 here
            
#             # Special values (3)
#             is_sleeping = inv_data[idx] > 0.5; idx += 1
#             is_resting = inv_data[idx] > 0.5; idx += 1
#             learned_spell = inv_data[idx] > 0.5; idx += 1
#             # idx should be 45 here
            
#             # Level values (4)
#             light_level = inv_data[idx]; idx += 1
#             current_level = int(inv_data[idx] * 10); idx += 1
#             level_cleared = inv_data[idx] > 0.5; idx += 1
#             # idx should be 48 here - this is the last value!
#             # So boss_vulnerable is NOT in the observation!
#             boss_vulnerable = False  # Not included in obs
            
#         except IndexError as e:
#             print(f"Error parsing inventory at index {idx}, total length {len(inv_data)}")
#             raise e
        
#         return {
#             'materials': {
#                 'wood': int(wood),
#                 'stone': int(stone),
#                 'coal': int(coal),
#                 'iron': int(iron),
#                 'diamond': int(diamond),
#                 'sapphire': int(sapphire),
#                 'ruby': int(ruby),
#             },
#             'items': {
#                 'sapling': int(sapling),
#                 'torches': int(torches),
#                 'arrows': int(arrows),
#                 'books': int(books),
#             },
#             'equipment': {
#                 'pickaxe_level': pickaxe_level,
#                 'sword_level': sword_level,
#                 'has_bow': has_bow,
#                 'armour_levels': armour,
#             },
#             'stats': {
#                 'food': food,
#                 'drink': drink,
#                 'energy': energy,
#                 'mana': mana,
#                 'xp': xp,
#                 'dexterity': dexterity,
#                 'strength': strength,
#                 'intelligence': intelligence,
#             },
#             'state': {
#                 'facing': facing,
#                 'sleeping': is_sleeping,
#                 'resting': is_resting,
#                 'learned_spell': learned_spell,
#                 'current_level': current_level,
#                 'light_level': light_level,
#                 'level_cleared': level_cleared,
#                 'boss_vulnerable': boss_vulnerable,
#             }
#         }
    
#     def to_text(self, parsed_obs, agent_id):
#         """Convert parsed observation to human-readable text for LLM"""
        
#         teammate_info = parsed_obs['teammates'][agent_id]
#         inv = parsed_obs['inventory']
#         map_info = parsed_obs['map']
        
#         text = f"""=== AGENT {agent_id} ({teammate_info['specialization']}) ===

# HEALTH & RESOURCES:
# - Health: {teammate_info['health']:.1f}/14
# - Food: {inv['stats']['food']:.1f}/17
# - Drink: {inv['stats']['drink']:.1f}/17
# - Energy: {inv['stats']['energy']:.1f}/17
# - Mana: {inv['stats']['mana']:.1f}/21

# LOCATION & STATUS:
# - Current Level: {inv['state']['current_level']}/8
# - Facing: {inv['state']['facing']}
# - Light Level: {inv['state']['light_level']:.2f}
# - Level Cleared: {inv['state']['level_cleared']}
# - Sleeping: {inv['state']['sleeping']} | Resting: {inv['state']['resting']} | Learned Spell: {inv['state']['learned_spell']}
# - Boss Vulnerable: {inv['state']['boss_vulnerable']}

# INVENTORY:
# Materials: Wood={inv['materials']['wood']}, Stone={inv['materials']['stone']}, Coal={inv['materials']['coal']}, Iron={inv['materials']['iron']}, Diamond={inv['materials']['diamond']}, Sapphire={inv['materials']['sapphire']}, Ruby={inv['materials']['ruby']}
# Items: Torches={inv['items']['torches']}, Arrows={inv['items']['arrows']}, Saplings={inv['items']['sapling']}, Books={inv['items']['books']}
# Equipment: Pickaxe Lvl {inv['equipment']['pickaxe_level']}, Sword Lvl {inv['equipment']['sword_level']}, Bow={inv['equipment']['has_bow']}, Armour Levels={inv['equipment']['armour_levels']}

# ATTRIBUTES:
# - XP: {inv['stats']['xp']:.1f} | Dexterity: {inv['stats']['dexterity']:.1f} | Strength: {inv['stats']['strength']:.1f} | Intelligence: {inv['stats']['intelligence']:.1f}

# TEAMMATES:
# """

        
#         for t in parsed_obs['teammates']:
#             if t['id'] == agent_id:
#                 continue
#             status = "ALIVE" if t['alive'] else "DEAD"
#             req_text = f", requesting {t['requesting']}" if t['requesting'] else ""
#             dir_text = f", located {t['direction']}" if t['direction'] else ", nearby"
#             text += f"- Agent {t['id']} ({t['specialization']}): {status}, Health {t['health']:.1f}{req_text}{dir_text}\n"
        
#         text += "\nVISIBLE SURROUNDINGS:\n"
        
#         if not map_info['visible_tiles']:
#             text += "- Nothing visible (too dark)\n"
#         else:
#             # Group tiles by direction
#             by_direction = {}
#             for tile in map_info['visible_tiles']:
#                 direction = tile['direction']
#                 if direction not in by_direction:
#                     by_direction[direction] = []
#                 by_direction[direction].append(tile)
            
#             for direction in ['here', 'north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest']:
#                 if direction in by_direction:
#                     text += f"\n{direction.upper()}:\n"
#                     for tile in by_direction[direction]:
#                         contents = ", ".join(tile['contents'])
#                         text += f"  - {contents}\n"
        
#         return text


# # Example usage:
# if __name__ == "__main__":
#     parser = CraftaxObservationParser(num_agents=3)
    
#     # Example: parse observation from environment
#     # obs, state = env.reset(key)
#     # parsed = parser.parse_observation(obs['agent_0'], agent_id=0)
#     # text = parser.to_text(parsed, agent_id=0)
#     # print(text)

import jax.numpy as jnp
import numpy as np
from enum import Enum

# Copy the enums from constants.py
class BlockType(Enum):
    INVALID = 0
    OUT_OF_BOUNDS = 1
    GRASS = 2
    WATER = 3
    STONE = 4
    TREE = 5
    WOOD = 6
    PATH = 7
    COAL = 8
    IRON = 9
    DIAMOND = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    SAND = 13
    LAVA = 14
    PLANT = 15
    RIPE_PLANT = 16
    WALL = 17
    DARKNESS = 18
    WALL_MOSS = 19
    STALAGMITE = 20
    SAPPHIRE = 21
    RUBY = 22
    CHEST = 23
    FOUNTAIN = 24
    FIRE_GRASS = 25
    ICE_GRASS = 26
    GRAVEL = 27
    FIRE_TREE = 28
    ICE_SHRUB = 29
    ENCHANTMENT_TABLE_FIRE = 30
    ENCHANTMENT_TABLE_ICE = 31
    NECROMANCER = 32
    GRAVE = 33
    GRAVE2 = 34
    GRAVE3 = 35
    NECROMANCER_VULNERABLE = 36

class ItemType(Enum):
    NONE = 0
    TORCH = 1
    LADDER_DOWN = 2
    LADDER_UP = 3
    LADDER_DOWN_BLOCKED = 4

class Specialization(Enum):
    UNASSIGNED = 0
    FORAGER = 1
    WARRIOR = 2
    MINER = 3

class Action(Enum):
    REQUEST_FOOD = 43
    REQUEST_DRINK = 44
    REQUEST_WOOD = 45
    REQUEST_STONE = 46
    REQUEST_IRON = 47
    REQUEST_COAL = 48
    REQUEST_DIAMOND = 49
    REQUEST_RUBY = 50
    REQUEST_SAPPHIRE = 51

OBS_DIM = (9, 11)

class CraftaxObservationParser:
    """Parse Craftax-Coop symbolic observations into human-readable text"""
    
    def __init__(self, num_agents=3):
        self.num_agents = num_agents
        
        # Calculate observation component sizes
        self.num_blocks = len(BlockType)
        self.num_items = len(ItemType)
        self.num_mob_classes = 5
        self.num_mob_types = 8
        
        # Map section: blocks, items, mobs, teammates, light
        self.map_size = OBS_DIM[0] * OBS_DIM[1] * (
            self.num_blocks + self.num_items + 
            self.num_mob_classes * self.num_mob_types + 
            num_agents + 1 + 1  # teammates + dead/alive bit + light
        )
        
        # Teammate dashboard: health, alive, specialization, requested material
        # NOTE: Directions are stored separately!
        self.teammate_dashboard_size = num_agents * (1 + 1 + 3 + 9)
        self.teammate_directions_size = num_agents * 8
        
        # Inventory and stats
        self.inventory_size = 48
        
    def parse_observation(self, obs, agent_id=0):
        """Parse a single agent's observation into structured data"""
        obs = np.array(obs) if isinstance(obs, jnp.ndarray) else obs
        
        idx = 0
        
        # 1. Parse map view (9x11 grid)
        map_data = obs[idx:idx + self.map_size]
        idx += self.map_size
        map_parsed = self._parse_map(map_data)
        
        # 2. Parse teammate dashboard
        teammate_data = obs[idx:idx + self.teammate_dashboard_size]
        idx += self.teammate_dashboard_size
        
        # 3. Parse teammate directions (separate from dashboard!)
        teammate_directions_data = obs[idx:idx + self.teammate_directions_size]
        idx += self.teammate_directions_size
        teammates_parsed = self._parse_teammates(teammate_data, teammate_directions_data, agent_id)
        
        # 4. Parse inventory and stats
        inventory_data = obs[idx:idx + self.inventory_size]
        inventory_parsed = self._parse_inventory(inventory_data)
        
        return {
            'agent_id': agent_id,
            'map': map_parsed,
            'teammates': teammates_parsed,
            'inventory': inventory_parsed,
            'raw_obs': obs
        }
    
    def _parse_map(self, map_data):
        """Parse the map view section"""
        # Reshape to (9, 11, num_features)
        num_features = self.num_blocks + self.num_items + self.num_mob_classes * self.num_mob_types + self.num_agents + 1 + 1
        map_grid = map_data.reshape(OBS_DIM[0], OBS_DIM[1], num_features)
        
        # Extract different layers
        blocks_layer = map_grid[:, :, :self.num_blocks]
        items_layer = map_grid[:, :, self.num_blocks:self.num_blocks + self.num_items]
        
        mob_start = self.num_blocks + self.num_items
        mob_end = mob_start + self.num_mob_classes * self.num_mob_types
        mobs_layer = map_grid[:, :, mob_start:mob_end]
        
        teammate_start = mob_end
        teammate_end = teammate_start + self.num_agents + 1
        teammates_layer = map_grid[:, :, teammate_start:teammate_end]
        
        light_layer = map_grid[:, :, -1]
        
        # Find visible tiles (where light > 0)
        visible_tiles = []
        for i in range(OBS_DIM[0]):
            for j in range(OBS_DIM[1]):
                if light_layer[i, j] > 0:
                    tile_info = self._parse_tile(i, j, blocks_layer[i, j], items_layer[i, j], 
                                                   mobs_layer[i, j], teammates_layer[i, j])
                    if tile_info:
                        visible_tiles.append(tile_info)
        
        return {
            'visible_tiles': visible_tiles,
            'grid_shape': OBS_DIM,
            'center': (OBS_DIM[0] // 2, OBS_DIM[1] // 2)  # Player is at center
        }
    
    def _parse_tile(self, i, j, blocks, items, mobs, teammates):
        """Parse a single tile"""
        tile_contents = []
        
        # Get block type
        block_idx = np.argmax(blocks)
        block_type = list(BlockType)[block_idx]
        if block_type not in [BlockType.GRASS, BlockType.PATH, BlockType.INVALID]:
            tile_contents.append(block_type.name.lower().replace('_', ' '))
        
        # Get item
        item_idx = np.argmax(items)
        item_type = list(ItemType)[item_idx]
        if item_type != ItemType.NONE:
            tile_contents.append(item_type.name.lower().replace('_', ' '))
        
        # Get mobs (reshape to 5 classes x 8 types)
        mob_grid = mobs.reshape(self.num_mob_classes, self.num_mob_types)
        mob_types = ['melee', 'passive', 'ranged', 'mob_projectile', 'player_projectile']
        for class_idx, class_name in enumerate(mob_types):
            for type_idx in range(self.num_mob_types):
                if mob_grid[class_idx, type_idx] > 0:
                    tile_contents.append(f"{class_name}_mob_{type_idx}")
        
        # Get teammates
        for t_idx in range(self.num_agents):
            if teammates[t_idx] > 0:
                is_alive = teammates[-1] > 0
                status = "alive" if is_alive else "dead"
                tile_contents.append(f"teammate_{t_idx} ({status})")
        
        if tile_contents:
            # Convert grid coords to relative position (center is player)
            rel_i = i - OBS_DIM[0] // 2
            rel_j = j - OBS_DIM[1] // 2
            direction = self._get_direction(rel_i, rel_j)
            manhattan_dist = abs(rel_i) + abs(rel_j)
            
            return {
                'position': (i, j),
                'relative': (rel_i, rel_j),
                'direction': direction,
                'manhattan_distance': manhattan_dist,
                'contents': tile_contents
            }
        return None
    
    def _get_direction(self, di, dj):
        """Convert relative position to cardinal direction with distance info"""
        if di == 0 and dj == 0:
            return "here"
        
        # Primary direction (stronger component)
        if abs(di) > abs(dj):
            primary = "north" if di < 0 else "south"
            secondary = "west" if dj < 0 else ("east" if dj > 0 else "")
        elif abs(dj) > abs(di):
            primary = "west" if dj < 0 else "east"
            secondary = "north" if di < 0 else ("south" if di > 0 else "")
        else:
            # Equal - use both
            vert = "north" if di < 0 else "south"
            horiz = "west" if dj < 0 else "east"
            return f"{vert}{horiz}"
        
        return f"{primary}{secondary}" if secondary else primary
    
    def _parse_teammates(self, teammate_data, teammate_directions_data, current_agent_id):
        """Parse teammate dashboard
        
        IMPORTANT: The teammate data is REORDERED per agent!
        - Index 0 = current agent's own info
        - Index 1+ = other agents (maintaining relative order)
        
        We need to map back to absolute agent IDs.
        """
        # Each teammate has: health(1) + alive(1) + specialization(3) + requested(9) = 14
        teammate_info_reordered = []
        
        per_teammate = 14  # Not 22! Directions are separate
        for reordered_idx in range(self.num_agents):
            start = reordered_idx * per_teammate
            
            health = teammate_data[start] * 10.0  # Unscale
            alive = teammate_data[start + 1] > 0.5
            
            spec_onehot = teammate_data[start + 2:start + 5]
            spec_idx = np.argmax(spec_onehot)
            specialization = ['Forager', 'Warrior', 'Miner'][spec_idx]
            
            req_onehot = teammate_data[start + 5:start + 14]
            requesting = None
            if np.any(req_onehot > 0):
                req_idx = np.argmax(req_onehot)
                requests = ['food', 'drink', 'wood', 'stone', 'iron', 'coal', 'diamond', 'ruby', 'sapphire']
                requesting = requests[req_idx]
            
            teammate_info_reordered.append({
                'reordered_idx': reordered_idx,
                'health': health,
                'alive': alive,
                'specialization': specialization,
                'requesting': requesting,
            })
        
        # Parse directions separately
        directions_per_teammate = 8
        teammate_directions = []
        for reordered_idx in range(self.num_agents):
            dir_start = reordered_idx * directions_per_teammate
            dir_onehot = teammate_directions_data[dir_start:dir_start + directions_per_teammate]
            direction = None
            if np.any(dir_onehot > 0):
                # Direction encoding: 0=NW, 1=N, 2=NE, 3=W, 4=E, 5=SW, 6=S, 7=SE
                dir_idx = np.argmax(dir_onehot)
                directions = ['northwest', 'north', 'northeast', 'west', 'east', 'southwest', 'south', 'southeast']
                direction = directions[dir_idx]
            teammate_directions.append(direction)
        
        # Reconstruct absolute ordering
        teammate_info = [None] * self.num_agents
        for reordered_idx, info in enumerate(teammate_info_reordered):
            if reordered_idx == 0:
                absolute_idx = current_agent_id
            elif reordered_idx <= current_agent_id:
                absolute_idx = reordered_idx - 1
            else:
                absolute_idx = reordered_idx
            
            teammate_info[absolute_idx] = {
                'id': absolute_idx,
                'health': info['health'],
                'alive': info['alive'],
                'specialization': info['specialization'],
                'requesting': info['requesting'],
                'direction': teammate_directions[reordered_idx]
            }
        
        return teammate_info
    
    def _parse_inventory(self, inv_data):
        """Parse inventory and player stats"""
        idx = 0
        
        # Inventory items (16 items)
        wood = (inv_data[idx] * 10.0) ** 2; idx += 1
        stone = (inv_data[idx] * 10.0) ** 2; idx += 1
        coal = (inv_data[idx] * 10.0) ** 2; idx += 1
        iron = (inv_data[idx] * 10.0) ** 2; idx += 1
        diamond = (inv_data[idx] * 10.0) ** 2; idx += 1
        sapphire = (inv_data[idx] * 10.0) ** 2; idx += 1
        ruby = (inv_data[idx] * 10.0) ** 2; idx += 1
        sapling = (inv_data[idx] * 10.0) ** 2; idx += 1
        torches = (inv_data[idx] * 10.0) ** 2; idx += 1
        arrows = (inv_data[idx] * 10.0) ** 2; idx += 1
        books = inv_data[idx]; idx += 1
        pickaxe_level = int(inv_data[idx] * 4); idx += 1
        sword_level = int(inv_data[idx] * 4); idx += 1
        sword_enchant = inv_data[idx]; idx += 1
        bow_enchant = inv_data[idx]; idx += 1
        has_bow = inv_data[idx] > 0.5; idx += 1
        
        # Potions (6)
        potions = [(inv_data[idx + i] * 10.0) ** 2 for i in range(6)]
        idx += 6
        
        # Intrinsics (8)
        food = inv_data[idx] * 10.0; idx += 1
        drink = inv_data[idx] * 10.0; idx += 1
        energy = inv_data[idx] * 10.0; idx += 1
        mana = inv_data[idx] * 10.0; idx += 1
        xp = inv_data[idx] * 10.0; idx += 1
        dexterity = inv_data[idx] * 10.0; idx += 1
        strength = inv_data[idx] * 10.0; idx += 1
        intelligence = inv_data[idx] * 10.0; idx += 1
        
        # Direction (4)
        direction_onehot = inv_data[idx:idx+4]
        idx += 4
        directions = ['left', 'right', 'up', 'down']
        facing = directions[np.argmax(direction_onehot)]
        
        # Armour (4)
        armour = [int(inv_data[idx + i] * 2) for i in range(4)]
        idx += 4
        
        # Armour enchantments (4)
        armour_enchants = [inv_data[idx + i] for i in range(4)]
        idx += 4
        
        # Special values (3)
        is_sleeping = inv_data[idx] > 0.5; idx += 1
        is_resting = inv_data[idx] > 0.5; idx += 1
        learned_spell = inv_data[idx] > 0.5; idx += 1
        
        # Level values (3) - boss_vulnerable is NOT in observation
        light_level = inv_data[idx]; idx += 1
        current_level = int(inv_data[idx] * 10); idx += 1
        level_cleared = inv_data[idx] > 0.5; idx += 1
        
        return {
            'materials': {
                'wood': int(wood),
                'stone': int(stone),
                'coal': int(coal),
                'iron': int(iron),
                'diamond': int(diamond),
                'sapphire': int(sapphire),
                'ruby': int(ruby),
            },
            'items': {
                'sapling': int(sapling),
                'torches': int(torches),
                'arrows': int(arrows),
                'books': int(books),
            },
            'equipment': {
                'pickaxe_level': pickaxe_level,
                'sword_level': sword_level,
                'has_bow': has_bow,
                'armour_levels': armour,
            },
            'stats': {
                'food': food,
                'drink': drink,
                'energy': energy,
                'mana': mana,
                'xp': xp,
                'dexterity': dexterity,
                'strength': strength,
                'intelligence': intelligence,
            },
            'state': {
                'facing': facing,
                'sleeping': is_sleeping,
                'resting': is_resting,
                'learned_spell': learned_spell,
                'current_level': current_level,
                'light_level': light_level,
                'level_cleared': level_cleared,
            }
        }
    
    def to_text(self, parsed_obs, agent_id):
        """Convert parsed observation to human-readable text for LLM"""
        
        teammate_info = parsed_obs['teammates'][agent_id]
        inv = parsed_obs['inventory']
        map_info = parsed_obs['map']
        
        text = f"""=== AGENT {agent_id} ({teammate_info['specialization']}) ===

HEALTH & RESOURCES:
- Health: {teammate_info['health']:.1f}/10
- Food: {inv['stats']['food']:.1f}/9
- Drink: {inv['stats']['drink']:.1f}/9
- Energy: {inv['stats']['energy']:.1f}/9
- Mana: {inv['stats']['mana']:.1f}/9

LOCATION & STATUS:
- Current Level: {inv['state']['current_level']}/8
- Facing: {inv['state']['facing']}
- Light Level: {inv['state']['light_level']:.2f}
- Sleeping: {inv['state']['sleeping']} | Resting: {inv['state']['resting']}

INVENTORY:
Materials: Wood={inv['materials']['wood']}, Stone={inv['materials']['stone']}, Coal={inv['materials']['coal']}, Iron={inv['materials']['iron']}, Diamond={inv['materials']['diamond']}
Items: Torches={inv['items']['torches']}, Arrows={inv['items']['arrows']}, Saplings={inv['items']['sapling']}
Equipment: Pickaxe Lvl {inv['equipment']['pickaxe_level']}, Sword Lvl {inv['equipment']['sword_level']}, Bow={inv['equipment']['has_bow']}

ATTRIBUTES:
- XP: {inv['stats']['xp']:.1f} | Str: {inv['stats']['strength']:.1f} | Dex: {inv['stats']['dexterity']:.1f} | Int: {inv['stats']['intelligence']:.1f}

TEAMMATES:
"""
        
        for t in parsed_obs['teammates']:
            if t['id'] == agent_id:
                continue
            status = "ALIVE" if t['alive'] else "DEAD"
            req_text = f", requesting {t['requesting']}" if t['requesting'] else ""
            dir_text = f", off-screen to {t['direction']}" if t['direction'] else ", nearby on screen"
            text += f"- Agent {t['id']} ({t['specialization']}): {status}, HP {t['health']:.1f}{req_text}{dir_text}\n"
        
        text += "\nVISIBLE TILES (sorted by distance):\n"
        
        if not map_info['visible_tiles']:
            text += "- Nothing visible (too dark)\n"
        else:
            # Sort tiles by manhattan distance, then by direction
            sorted_tiles = sorted(map_info['visible_tiles'], 
                                key=lambda t: (t['manhattan_distance'], t['direction']))
            
            current_dist = None
            for tile in sorted_tiles:
                dist = tile['manhattan_distance']
                rel_i, rel_j = tile['relative']
                
                # Show distance header when it changes
                if dist != current_dist:
                    current_dist = dist
                    if dist == 0:
                        text += f"\n[HERE - your current tile]:\n"
                    else:
                        text += f"\n[Distance {dist} tiles away]:\n"
                
                # Format position info
                dir_desc = tile['direction'].upper()
                pos_desc = f"({rel_i:+d} rows, {rel_j:+d} cols)"  # +d adds sign
                contents = ", ".join(tile['contents'])
                
                text += f"  {dir_desc} {pos_desc}: {contents}\n"
        
        return text


# Example usage:
if __name__ == "__main__":
    parser = CraftaxObservationParser(num_agents=3)
    
    # Example: parse observation from environment
    # obs, state = env.reset(key)
    # parsed = parser.parse_observation(obs['agent_0'], agent_id=0)
    # text = parser.to_text(parsed, agent_id=0)
    # print(text)