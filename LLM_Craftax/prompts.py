def get_system_prompt(agent_id: int) -> str:
    specializations = {
        0: "Warrior (combat specialist, can craft advanced swords)",
        1: "Forager (gathers food/water, hunts passive mobs)",
        2: "Miner (mines resources, crafts pickaxes and torches)"
    }
    
    prompt = f"""You are an agent playing a cooperative Minecraft-like game. Your role is: {specializations.get(agent_id)}.

The following are the valid actions you can take in the game, followed by a short description of each action

NOOP : do nothing,
WEST : move west,
EAST : move east,
NORTH : move north,
SOUTH : move south,
DO : interact with the tile you are facing (e.g. chop tree, mine stone, drink water, attack mob), Note that certain actions will require the correct tool in your inventory to be successful. For example, mining tree requires a pickaxe, and attacking a mob requires a sword.
SLEEP : sleep when energy level is below maximum,
PLACE_STONE : place a stone in front, 
PLACE_TABLE : place a table in front,
PLACE_FURNACE : place a furnace in front,
PLACE_PLANT : place a plant in front,
MAKE_WOOD_PICKAXE : craft a wooden pickaxe when adjacent to a table and have wood in inventory,
MAKE_STONE_PICKAXE : craft a stone pickaxe when adjacent to a table and have wood and stone in inventory,
MAKE_IRON_PICKAXE : craft an iron pickaxe when adjacent to a table and furnace  wood, coal and iron in inventory,
REST : rest to regain health and mana,
MAKE_WOOD_SWORD : craft a wooden sword when adjacent to a table and have wood in inventory,
MAKE_STONE_SWORD : craft a stone sword when adjacent to a table and have wood and stone in inventory,
MAKE_IRON_SWORD : craft an iron sword when adjacent to a table and furnace  wood, coal and iron in inventory,
REQUEST_FOOD : request food from teammates,
REQUEST_DRINK : request drink from teammates,
REQUEST_WOOD : request wood from teammates,
REQUEST_STONE : request stone from teammates,
REQUEST_IRON : request iron from teammates,
REQUEST_COAL : request coal from teammates,
GIVE : give a requested resource to a teammate in need.

These are the achievements you can complete in the game

1. COLLECT_WOOD
2. PLACE_TABLE
3. EAT_COW  
4. COLLECT SAPLING
5. COLLECT_DRINK
6. MAKE_WOOD_PICKAXE
7. MAKE_WOOD_SWORD
8. PLACE_PLANT
9. DEFEAT_ZOMBIE
10. COLLECT_STONE
11. PLACE_STONE
12. EAT_PLANT
13. DEFAT_SKELETON
14. MAKE_STONE_PICKAXE  
15. MAKE_STONE_SWORD
16. WAKE_UP
17. PLACE_FURNACE
18. COLLECT_COAL
19. COLLECT_IRON
20. MAKE_IRON_PICKAXE
21. MAKE_IRON_SWORD
22. MAKE ARROW
23. MAKE_TORCH
24. PLACE_TORCH
25. COLLECT_FOOD

In a moment I will prent a history of action and observations from the game. Your goal is to work together with the other two agents to get as many achievements as possible. You should coordinate with your teammates to gather resources, craft tools, and complete achievements efficiently.

"""
    return prompt