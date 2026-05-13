
def get_alice_prompt() -> str:
  prompt = f"""I am Alice. My teammate Bob and I want to complete as many tasks as possible in a cooperative Minecraft-like game

Assume Alice is an expert in designing plan outlines. Given our shared goal, previous plan, dialogue history, latest observation, Bob's suggestion,  generate/refine the global plan for Bob and yourself during task execution, guiding us to achieve the goal collaboratively as soon as possible. 

The generated action plan should strictly meet following requirements:
1.You may consider the following actions as intermediate steps to achieve the overall goal:

EXPLORE : explore the edge of the visible area to discover new tiles and resources,
COLLECT_WOOD : collect wood when adjacent to a tree,
COLLECT_STONE : collect wood when adjacent to a stone tile, requires WOODEN_PICKAXE <lvl1> or higher
COLLECT_COAL : collect wood when adjacent to a coal tile, requires WOODEN_PICKAXE <lvl1> or higher
COLLECT_IRON : collect wood when adjacent to a iron tile, requires STONE_PICKAXE <lvl2> or higher
EAT_COW : eat a cow to regain health and food levels when adjacent to a cow 
SLEEP : sleep when energy level is below maximum,
PLACE_STONE : place a stone in front, 
PLACE_TABLE : place a table in front, (requires two wood in inventory)
PLACE_FURNACE : place a furnace in front, (requires one stone in inventory)
PLACE_PLANT : place a plant in front,
MAKE_WOOD_PICKAXE : craft a wooden pickaxe when adjacent to a table and have wood in inventory,
MAKE_STONE_PICKAXE : craft a stone pickaxe when adjacent to a table and have wood and stone in inventory,
MAKE_IRON_PICKAXE : craft an iron pickaxe when adjacent to a table and furnace  and have wood, coal and iron in inventory,
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

2. Please keep your reasoning process, but the final action plan should be brief, reliable, authentic, and consistent with the latest observations of yourself and Bob. Don’t make random and meaningless plans.
3. The collaboration action plan should be detailed to each Agent. The action plan only needs to consider three steps at most at current time. When there is not much known information or content to be planned, the action plan can have only one or two steps. The action plan must be structured strictly in the format: <Action Plan: Step 1: Alice xxx, Bob xxx; Step 2: Alice xxx, Bob xxx; Step 3: Alice xxx, Bob xxx>. Here, 'xxx' represents one or multiple allowed actions.
5. In order to let Bob know about Alice's situation, you need to generate a short message to Bob. The message has to be concise, reliable, and helpful for assisting Bob and me to make an efficient and consistent action plan, and transport as many objects to the bed as possible. Don’t generate repetitive messages.
6.Alice and Bob act separately and can only exchange information. They can trade items via the REQUEST_<item> and GIVE actions. Please do not assign the same action to two agents, it is wasteful.

Here are an example for Alice:

Goal: [Craft a wooden sword and wooden pickaxe]
Reasoning: [According to Bob's suggestions and progress, the current plan is partially reasonable and needs minor adjustments. Bob has 2 wood so he should place a crafting table, Alice has no wood and should collect wood so she can craft the sword and pickaxe.]
Action plan: [Step 1: Alice COLLECT_WOOD, Bob PLACE_CRAFTING_TABLE 
Step 2: Alice go to CRAFTING_TABLE, Bob COLLECT WOOD
Step 3: Alice MAKE_WOODEN_SWORD,  Bob MAKE_WOODEN_PICKAXE]
Message: [Hi Bob, I have received your message and will adjust our action plan. You  should place the place the crafting table and collect wood and to craft the wooden pickaxe. I will collect wood and craft the wooden sword at the crafting table. Do you have any new suggestions for the updated plan?]

Following are provided information for Alice:
Goal: Craft a wooden sword
Previous action plan: None
Dialogue history:  None at the moment
Alice's Observation: {get_alice_obs()}

Think step by step, and generate a new action plan, in word format. """
    
  return prompt


def get_bob_prompt() -> str:
    prompt = f"""I am Bob. My teammate Alice and I want to complete as many tasks as possible in a cooperative Minecraft-like game
Because Alice may not understand Bob's current progress and information, and may not consider the plan comprehensively and perfectly, which wastes our action time. Given our shared goal, action plan, dialogue history, observations, and my previous actions, please help me analyze and score Alice's proposed action plan, point out the shortcomings of Alice's plan and reflect on it and finally generate a message to send to Alice, at the beginning of the message, I should first explain my findings. You should make full use of Alice and Bob to complete the task efficiently and not waste time. Important, Alice and Bob act separately and can only exchange information. They can trade items via the REQUEST_<item> and GIVE actions. 
The content Bob generate mainly consists of two parts: reasoning and message sent to Alice. Please strictly follow the following format:
Reasoning: [the reasoning process, analyze the unreasonableness of the current plan and consider how to make it more efficient]
Dis_Score: [the score between 0 and 5: 1, No consideration of distance 2, Minimal consideration of distance, overlooking key factors 3, Distance is considered but not entirely accurate 4, Distance is sufficiently considered, with only minor oversights 5, Comprehensive and accurate consideration of distance, only output the score.]
Task_Score: [the score between 0 and 5: 1, No consideration of work distribution between Alice and Bob 2, Minimal consideration of work distribution, leading to unreasonable allocation 3, Work distribution is considered but not entirely accurate 4, Work distribution is reasonable, with only minor oversights 5, Work distribution is highly effective, making full use of Alice and Bob's abilities, only output the score.]
Message: [the message sent to Alice, you need to first tell Alice about your findings]
Here are an example for your reference: :
Reasoning: [In the current plan, Alice may not know our progress, so some of Bob's plans are vague, which needs to be improved. In addition, it is a waste of time for Alice and Bob to both collect wood.]
Dis_Score: [4]
Task_Score: [2]
Message: [Hi, Alice, I don’t think it is efficient for us to both collect wood   Your plan needs to be adjusted and describe my actions as detailed as possible..]
The following is the information of Bob currently:
Bob's Previous action: None at the moment
Bob's Observation: {get_bob_obs()}
The following is the relevant information when Alice is planning her action plan, which can be used as a reference for Bob:
User: <user_input>\n
Alice's response: <candidate_content>
Think step by step, and generate the content sent to Alice, word format:"""
    return prompt

def get_alice_obs():
  return """HEALTH & RESOURCES:
- Health: 9.0/10
- Food: 9.0/9
- Drink: 9.0/9
- Energy: 9.0/9
- Mana: 9.0/9

STATUS:
- Current Level: 0/8
- Facing: north
- Light Level: 0.80
- Sleeping: False | Resting: False
- Level Cleared: True | Boss Vulnerable: False

INVENTORY:
Materials: Wood=0, Stone=0, Coal=0, Iron=0, Diamond=0
Gems: Sapphire=0, Ruby=0
Items: Torches=0, Arrows=0, Saplings=0
Equipment: Pickaxe Lvl 0, Sword Lvl 0, Bow=False
Armour: [0, 0, 0, 0]

ATTRIBUTES:
- XP: 0.0 | Str: 1.0 | Dex: 1.0 | Int: 1.0


Long-term observation for agent_0: VISIBLE TILES (sorted by distance):

[HERE - your current tile]:
  HERE Facing: north: teammate (alive)

[Distance 1 tiles away]:
  EAST (0 tiles east): teammate (alive)
  NORTH (1 tiles north): tree
  SOUTH (1 tiles south): tree

[Distance 2 tiles away]:
  NORTH (2 tiles north): tree
  SOUTHWEST (1 tiles south, 1 tiles west): tree

[Distance 3 tiles away]:
  NORTH (3 tiles north): tree
  NORTHEAST (1 tiles north, 2 tiles east): tree
  NORTHWEST (2 tiles north, 1 tiles west): tree
  NORTHWEST (1 tiles north, 2 tiles west): tree

[Distance 4 tiles away]:
  NORTHWEST (3 tiles north, 1 tiles west): tree
  SOUTH (4 tiles south): coal

[Distance 5 tiles away]:
  NORTHWEST (4 tiles north, 1 tiles west): tree
  SOUTHEAST (4 tiles south, 1 tiles east): stone
  SOUTHWEST (4 tiles south, 1 tiles west): stone
  WEST (0 tiles west): stone

[Distance 6 tiles away]:
  NORTHEAST (2 tiles north, 4 tiles east): tree
  NORTHWEST (1 tiles north, 5 tiles west): stone
  SOUTHEAST (4 tiles south, 2 tiles east): stone
  SOUTHWEST (1 tiles south, 5 tiles west): stone
  SOUTHWEST (2 tiles south, 4 tiles west): coal
  SOUTHWEST (3 tiles south, 3 tiles west): stone
  SOUTHWEST (4 tiles south, 2 tiles west): lava

[Distance 7 tiles away]:
  NORTHWEST (2 tiles north, 5 tiles west): stone
  SOUTHEAST (4 tiles south, 3 tiles east): stone
  SOUTHWEST (2 tiles south, 5 tiles west): iron

[Distance 8 tiles away]:
  NORTHEAST (4 tiles north, 4 tiles east): sand
  NORTHEAST (3 tiles north, 5 tiles east): sand
  NORTHWEST (4 tiles north, 4 tiles west): stone
  SOUTHEAST (3 tiles south, 5 tiles east): stone
  SOUTHWEST (3 tiles south, 5 tiles west): stone

[Distance 9 tiles away]:
  NORTHEAST (4 tiles north, 5 tiles east): sand
  NORTHWEST (4 tiles north, 5 tiles west): stone
  SOUTHEAST (4 tiles south, 5 tiles east): stone
  SOUTHWEST (4 tiles south, 5 tiles west): stone"""



def get_bob_obs():
  return """HEALTH & RESOURCES:
- Health: 9.0/10
- Food: 9.0/9
- Drink: 9.0/9
- Energy: 9.0/9
- Mana: 9.0/9

STATUS:
- Current Level: 0/8
- Facing: north
- Light Level: 0.80
- Sleeping: False | Resting: False
- Level Cleared: True | Boss Vulnerable: False

INVENTORY:
Materials: Wood=0, Stone=0, Coal=0, Iron=0, Diamond=0
Gems: Sapphire=0, Ruby=0
Items: Torches=0, Arrows=0, Saplings=0
Equipment: Pickaxe Lvl 0, Sword Lvl 0, Bow=False
Armour: [0, 0, 0, 0]

ATTRIBUTES:
- XP: 0.0 | Str: 1.0 | Dex: 1.0 | Int: 1.0

VISIBLE TILES (sorted by distance):

[HERE - your current tile]:
  HERE Facing: north: teammate (alive)

[Distance 1 tiles away]:
  WEST (0 tiles west): teammate (alive)

[Distance 2 tiles away]:
  NORTHEAST (1 tiles north, 1 tiles east): tree
  NORTHWEST (1 tiles north, 1 tiles west): tree
  SOUTHWEST (1 tiles south, 1 tiles west): tree

[Distance 3 tiles away]:
  NORTHWEST (2 tiles north, 1 tiles west): tree
  SOUTHWEST (1 tiles south, 2 tiles west): tree

[Distance 4 tiles away]:
  NORTHWEST (3 tiles north, 1 tiles west): tree
  NORTHWEST (2 tiles north, 2 tiles west): tree
  NORTHWEST (1 tiles north, 3 tiles west): tree
  SOUTH (4 tiles south): stone

[Distance 5 tiles away]:
  EAST (0 tiles east): tree
  NORTHEAST (2 tiles north, 3 tiles east): tree
  NORTHWEST (3 tiles north, 2 tiles west): tree
  SOUTHEAST (4 tiles south, 1 tiles east): stone
  SOUTHWEST (4 tiles south, 1 tiles west): coal

[Distance 6 tiles away]:
  NORTHEAST (1 tiles north, 5 tiles east): tree
  NORTHWEST (4 tiles north, 2 tiles west): tree
  SOUTHEAST (1 tiles south, 5 tiles east): tree
  SOUTHEAST (4 tiles south, 2 tiles east): stone
  SOUTHWEST (4 tiles south, 2 tiles west): stone

[Distance 7 tiles away]:
  NORTHEAST (4 tiles north, 3 tiles east): sand
  NORTHEAST (3 tiles north, 4 tiles east): sand
  NORTHEAST (2 tiles north, 5 tiles east): sand
  SOUTHEAST (2 tiles south, 5 tiles east): stone
  SOUTHEAST (3 tiles south, 4 tiles east): stone
  SOUTHWEST (2 tiles south, 5 tiles west): coal
  SOUTHWEST (3 tiles south, 4 tiles west): stone
  SOUTHWEST (4 tiles south, 3 tiles west): lava

[Distance 8 tiles away]:
  NORTHEAST (4 tiles north, 4 tiles east): sand
  NORTHEAST (3 tiles north, 5 tiles east): sand
  SOUTHEAST (3 tiles south, 5 tiles east): stone
  SOUTHEAST (4 tiles south, 4 tiles east): stone

[Distance 9 tiles away]:
  NORTHEAST (4 tiles north, 5 tiles east): sand
  NORTHWEST (4 tiles north, 5 tiles west): stone
  SOUTHEAST (4 tiles south, 5 tiles east): stone"""
