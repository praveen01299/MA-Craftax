
def get_alice_prompt(objective, obs_str) -> str:
  prompt = f"""You are Alice. you and your teammate Bob want to collaborate to complete tasks in an efficient manner in a cooperative Minecraft-like game

You are to generate a joint action plan in the form of an function "action_plan" of the "CraftaxJointPolicy" class. Given your shared goal, list of available actions, previous plan, dialogue history, latest observation, Bob's suggestion,  generate/refine the code for the action plan for Bob and yourself during task execution, guiding us to achieve the goal collaboratively as soon as possible. 

The generated function should strictly meet following requirements:

1. The function should take in as input a string indicating player's action to give (Alice's or Bob's) and a Python dictionary containing past observations from Alice and Bob in the following format:

{{
"Alice" : [obs_0, ..., obs_t]
"Bob" : [obs_0, ..., obs_t]
}}   

and select the following list of actions for Alice or Bob as output , and the actions should be only selected if the preconditions are satisfied. The preconditions for each action are can be checked with the "is_task_feasible(task, observation)" function. The valid actions are as follows:

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

2. The function should also determine when to terminate an action and switch to a new action, the termination condition for each action can be checked via the "is_task_complete(task, observation, previous_obsevation)" function.
3. The function should only output the string of the selected action. Here is the template of the function:

def action_plan(self, agent_name: str, observation_history: dict) -> str:
  ....
  return selected_action

4. Please keep your reasoning process, but the final action plan should take into account how you and Bob should distribute the actions in order to achieve the final goal, avoid redundant action distributions and ground the selection on the observation and states of each player. Keep in mind that certain goals would require efficient cooperation of the two players.
5. In order to let Bob know about your (Alice's) reasoning, you need to generate a short message to Bob. The message has to be concise, reliable, and helpful for assisting Bob and me to make an efficient and consistent action selection function. Don’t generate repetitive messages.
6. Note that you and Bob act separately and cannot trade items.

Following are provided information for you:
Goal: {objective}
Alice's Observation: {obs_str}

Think step by step, and generate a new action plan. Give your outputs in the following format:
Reasoning : <step by step reasoning>
Action plan : <code for joint task execution>
Message : <messgae to Bob>"""

  return prompt


def get_bob_prompt(obs_str) -> str:
    prompt = f"""Your are Bob. Your and your teammate Alice want to collaborate to complete tasks in an efficient manner in a cooperative Minecraft-like game.
Given your shared goal, action selection scheme in the form of of the 'action_plan()' function, and observations, please analyze and score Alice's proposed 'action_plan()' function, point out the shortcomings of Alice's proposed scheme and reflect on it and finally generate a message to send to Alice, at the beginning of the message, you should first explain your findings. You should make full use of Alice and Bob to complete the task efficiently. Important, you and Alice act separately and cannot trade held items. 
The content Bob generate mainly consists of two parts: reasoning and message sent to Alice. Please strictly follow the following format:
Reasoning: [the reasoning process, analyze the quality of the current plan and consider how to make it more efficient, check for any redundancy between you and Alice.]
Progress_Score: [the score between 0 and 5: 1, Task selection does not contribute to the progress of goal 2, Majority of tasks selected does not contribute to the overall goal, overlooking key factors 3, Significant number of tasks selected does not contribute to the overall goal.  4, Most of the task selected meaningfully contribute to the overall goal. 5, All the tasks selected meaningfully contribute to the overall goal.]
Task_Score: [the score between 0 and 5: 1, No consideration of task distribution between Alice and Bob 2, Minimal consideration of task distribution, leading to unreasonable allocation 3, Work distribution is considered but not entirely accurate 4, Task distribution is reasonable, with only minor oversights 5, Task distribution is highly effective, making full use of Alice and Bob's abilities, only output the score.]
Message: [the message sent to Alice, you need to first tell Alice about your findings]
Bob's Observation: {obs_str}
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

