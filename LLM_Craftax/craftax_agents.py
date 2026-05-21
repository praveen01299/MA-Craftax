import copy
import re
from llm_client import LLMResponse
from astar import astar
import numpy as np
import json

from task_check import is_task_complete
from task_cond import is_task_feasible
# from prompts import get_dialogue_system_prompt

ACTIONS = {
    'NOOP': 0, 'WEST': 1, 'EAST': 2, 'NORTH': 3, 'SOUTH': 4, 'DO': 5, 'SLEEP': 6,
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

OBJECTIVES = [
    'COLLECT_WOOD', 'PLACE_TABLE', 'EAT_COW', 'COLLECT_SAPLING', 'COLLECT_DRINK',
    'MAKE_WOOD_PICKAXE', 'MAKE_WOOD_SWORD', 'PLACE_PLANT', 'DEFEAT_ZOMBIE',
    'COLLECT_STONE', 'PLACE_STONE', 'EAT_PLANT', 'DEFEAT_SKELETON',
    'MAKE_STONE_PICKAXE', 'MAKE_STONE_SWORD', 'WAKE_UP', 'PLACE_FURNACE',
    'COLLECT_COAL', 'COLLECT_IRON', 'MAKE_IRON_PICKAXE', 'MAKE_IRON_SWORD',
    'MAKE_ARROW', 'MAKE_TORCH', 'PLACE_TORCH', 'COLLECT_FOOD'
]

START_POS = (4, 5)

_MOVES = [
    (-1, 0),  # NORTH
    ( 1, 0),  # SOUTH
    ( 0,-1),  # WEST
    ( 0, 1),  # EAST
]

DIR_TO_FACING = {
    (1, 0) : 5,
    ( -1, 0) : 4,
    ( 0, 1) : 3,
    ( 0, -1) : 2,
}

class BaseAgent:
    """Base class for agents in the Craftax environment."""


    def __init__(self, agent_name, client=None, prompt_builder=None):
        self.agent_name = agent_name
        self.client = client
        self.prompt_builder = prompt_builder

    def act(self, obs, prev_action=None):
        return NotImplementedError
    
    def reset(self):
        self.prompt_builder.reset()
    
    def update_prompt(self, observation, prev_action):
        self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(observation)
        
        


class NaiveAgent(BaseAgent):
    """An agent that generates actions based on observations without complex reasoning."""

    def __init__(self, agent_name, client, prompt_builder):
        super().__init__(agent_name, client, prompt_builder)
        self.default_action = 'NOOP'
        self.failed_candidates = []  # To track invalid actions suggested by the LLM

    def act(self, obs, prev_action=None):
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            str: The selected action from the LLM response.
        """
        # print(prev_action)
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        naive_instruction = """
You always have to output one of the above actions at a time and no other text. You always have to output an action until the episode terminates.
        """.strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction

        response = self.client.generate(messages)

        final_answer = self._extract_final_answer(response)
        final_action = self.check_action_validity(final_answer.completion)
        # final_answer = LLMResponse(
        #     model_id="",
        #     completion="NOOP",
        #     stop_reason="",
        #     input_tokens="",
        #     output_tokens="",
        #     reasoning=None,
        # )

        

        return final_answer, ACTIONS[final_action]
        # return final_answer, 0
    
    def check_action_validity(self, candidate_action):
        valid_action = None
        if candidate_action in ACTIONS:
            valid_action = candidate_action
        else:
            valid_action = self.default_action
            self.failed_candidates.append(candidate_action)
        return valid_action

    def _extract_final_answer(self, answer):
        """Sanitize the final answer, keeping only alphabetic characters.

        Args:
            answer (LLMResponse): The response from the LLM.

        Returns:
            LLMResponse: The sanitized response.
        """

        def filter_letters(input_string):
            return re.sub(r"[^a-zA-Z\s:]", "", input_string)

        final_answer = copy.deepcopy(answer)
        final_answer = final_answer._replace(completion=filter_letters(final_answer.completion))

        return final_answer
    
    def reset(self):
        self.prompt_builder.reset()



class ExecAgent(BaseAgent):
    """An agent that generates actions based on observations without complex reasoning."""

    def __init__(self, agent_name, client, prompt_builder):
        super().__init__(agent_name, client, prompt_builder)
        self.default_action = 'NOOP'
        self.current_plan = None
        self.current_objective = None
        self.failed_candidates = []  # To track invalid actions suggested by the LLM

    def generate_plan(self, grid, action=None, coord=None):

        def select_random_goal(grid):
            new_goal_coord = None
            while new_goal_coord is None:
                goal_x = np.random.randint(0, grid.shape[0])
                goal_y = np.random.randint(0, grid.shape[1])
                if grid[goal_x, goal_y] == 0:
                    new_goal_coord = (goal_x, goal_y)
            return new_goal_coord

        if coord is None:
            goal_coord = select_random_goal(grid)
        else:
            goal_coord = (START_POS[0] + coord[0], START_POS[1] + coord[1])

        if action is None:
            action_list = []
        else:
            action_list = [ACTIONS[action]]
        # print(grid)
        # print(goal_coord)
        facing = grid[START_POS[0]][START_POS[1]]
        # print(facing)
        grid[START_POS[0]][START_POS[1]] = 0
        if grid[goal_coord[0], goal_coord[1]] == 1:
            neighbor_coords = [(goal_coord[0] + dx, goal_coord[1] + dy) for dx, dy in _MOVES]
            valid_neighbors = [c for c in neighbor_coords if 0 <= c[0] < grid.shape[0] and 0 <= c[1] < grid.shape[1] and grid[c[0], c[1]] == 0]
            if valid_neighbors:
                new_goal_coord = min(valid_neighbors, key=lambda c: abs(c[0] - START_POS[0]) + abs(c[1] - START_POS[1]))
            
                goal_dir = (goal_coord[0] - new_goal_coord[0], goal_coord[1] - new_goal_coord[1])
                correct_facing = DIR_TO_FACING[goal_dir]
                goal_coord = new_goal_coord
            else:
                print("No valid neighboring cells to navigate to!, selecting random goal")
                new_goal_coord = select_random_goal(grid)
                # goal_dir = (goal_coord[0] - new_goal_coord[0], goal_coord[1] - new_goal_coord[1])
                # correct_facing = DIR_TO_FACING[goal_dir] 
                correct_facing = None       
                goal_coord = new_goal_coord
        else:
            correct_facing = None
        if goal_coord is not None:
            navigate_plan = astar(np.array(grid), START_POS, goal_coord)
            # print(navigate_plan)
            if navigate_plan is not None:
                if len(navigate_plan) > 0:
                    facing = navigate_plan[-1] + 1
                if correct_facing is not None and facing != correct_facing:
                    navigate_plan.append(correct_facing - 1)
            return navigate_plan + action_list if navigate_plan is not None else action_list 
        else:
            return action_list

    def execute_plan(self, plan):
        if self.plan_index < len(plan) - 1:
            action = plan[self.plan_index]
            # if self.check_action_validity(action):
            self.plan_index += 1
            return action
        else:
            action = plan[self.plan_index]
            self.current_plan = None
            self.plan_index = 0
            # else:
            #     self.failed_candidates.append(action)
            #     return self.default_action
        return action


    def act(self, obs, step):
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            str: The selected action from the LLM response.
        """
        # print(prev_action)
        if self.current_plan is not None:
            next_action = self.execute_plan(self.current_plan)
            return next_action, None
        else:
            # if prev_action:
            #     self.prompt_builder.update_action(prev_action)

            self.prompt_builder.update_observation(obs, step)

            plan_messages = self.prompt_builder.get_exec_prompt()

            if self.objective == "EXPLORE":
                self.current_plan = self.generate_plan(obs["passable_grid"])
                plan_reasoning = None
                action = None
            else:    

                if self.objective.startswith("PLACE"):
                    plan_proposal_instructions = f"""Your current objective is to: {self.current_objective}. Given the current observation, 
            Give the relative postion of an empty cell to navigate to, use positive integers to indicate south and east and negative intergers to indicate north and west. 
            Give your answer in the following format:
            THOUGHTS :  <thoughts>
            NAVIGATE TO: <(north/south, east/west)>""".strip()

                    action = self.objective

                elif self.objective.startswith("COLLECT"):
                    plan_proposal_instructions = f"""Your current objective is to: {self.current_objective}. Given the current observation, 
            Give the relative postion of the resource to navigate to, use positive integers to indicate south and east and negative intergers to indicate north and west. 
            Give your answer in the following format:
            THOUGHTS :  <thoughts>
            NAVIGATE TO: <(north/south, east/west)>""".strip()
                    
                    action = "DO"

                elif self.objective.startswith("MAKE"):    

                    plan_proposal_instructions = f"""Your current objective is to: {self.current_objective}. Given the current observation, 
            Give the relative postion of the crafting table to navigate to, use positive integers to indicate south and east and negative intergers to indicate north and west. 
            Give your answer in the following format:
            THOUGHTS :  <thoughts>
            NAVIGATE TO: <(north/south, east/west)>""".strip()
                    
                    action = self.objective
                
                elif self.objective.startswith("EAT"):    

                    plan_proposal_instructions = f"""Your current objective is to: {self.current_objective}. Given the current observation, 
            Give the relative postion of the location of the cow to navigate to, use positive integers to indicate south and east and negative intergers to indicate north and west. 
            After that, give by a valid action. 
            Give your answer in the following format:
            THOUGHTS :  <thoughts>
            NAVIGATE TO: <(north/south, east/west)>""".strip()
                    
                    action = "DO"
                
                elif self.objective.startswith("DEFEAT"):

                    plan_proposal_instructions = f"""Your current objective is to: {self.current_objective}. Given the current observation, 
            Give the relative postion of the enemy to navigate to, use positive integers to indicate south and east and negative intergers to indicate north and west. 
            After that, give by a valid action. 
            Give your answer in the following format:
            THOUGHTS :  <thoughts>
            NAVIGATE TO: <(north/south, east/west)>""".strip()
                    
                    action = "DO"

                else:
                    print(f"Unknown objective type for objective {self.current_objective}, defaulting to random navigation")
                    self.current_plan = self.generate_plan(obs["passable_grid"])
                    plan_reasoning = None
                    action = None
                    
                if action is not None:
                    plan_messages[-1].content += "\n\n" + plan_proposal_instructions

                    cot_plan_reasoning = self.client.generate(plan_messages)
                    coord, plan_reasoning = self._extract_plan(cot_plan_reasoning)

                    self.current_plan = self.generate_plan(obs["passable_grid"], coord, action)


            next_action = self.execute_plan(self.current_plan)

            planner_output = {
                "objective": self.current_objective,
                "plan": self.current_plan,
                "reasoning": plan_reasoning,
            }

            return next_action, planner_output
        # return final_answer, 0
    

    def _extract_plan(self, completion: str):
        """Extract NAVIGATE coordinates and ACTION from an LLM plan completion.

        Expects the format produced by ``achievemnet_proposal_instructions``:
            NAVIGATE TO (<x>, <y>), ACTION: <ACTION_NAME>

        Args:
            completion: Raw LLM completion string.

        Returns:
            tuple[tuple[int, int] | None, str | None]:
                ``(coords, action)`` where *coords* is ``(x, y)`` and *action*
                is a key from ``ACTIONS``, or ``None`` for either field when the
                corresponding part cannot be parsed.
        """
        coords = None

        nav_match = re.search(r"NAVIGATE\s+TO:\s+\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", completion.completion, re.IGNORECASE)
        if nav_match:
            coords = (int(nav_match.group(1)), int(nav_match.group(2)))
        else:
            coords = (0, 0)


        final_answer = copy.deepcopy(completion)
        self.prompt_builder.update_reasoning(completion.completion, type="plan")
        final_answer = final_answer._replace(reasoning=final_answer.completion)
        # final_answer = final_answer._replace(completion=filter_letters(final_answer.completion).split("OBJECTIVE:")[-1].strip())

        return coords, final_answer



class CoTAgent(BaseAgent):
    """An agent that generates actions based on observations without complex reasoning."""

    def __init__(self, agent_name, client, prompt_builder):
        super().__init__(agent_name, client, prompt_builder)
        self.default_action = 'NOOP'
        self.failed_candidates = []  # To track invalid actions suggested by the LLM

    def act(self, obs, prev_action=None):
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            str: The selected action from the LLM response.
        """
        # print(prev_action)
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        cot_instructions = """
First think about what's the best course of action step by step.
Finally, provide a single output action at the end of the message in the form of: ACTION: <action>
        """.strip()

        messages[-1].content += "\n\n" + cot_instructions

        # Generate the CoT reasoning
        cot_reasoning = self.client.generate(messages)

        final_answer = self._extract_final_answer(cot_reasoning)
        final_action = self.check_action_validity(final_answer.completion)
        # final_answer = LLMResponse(
        #     model_id="",
        #     completion="NOOP",
        #     stop_reason="",
        #     input_tokens="",
        #     output_tokens="",
        #     reasoning=None,
        # )


        return final_answer, final_action, ACTIONS[final_action]
        # return final_answer, 0
    
    def check_action_validity(self, candidate_action):
        valid_action = None
        print(f"Checking validity of action: {candidate_action}")
        print(candidate_action in ACTIONS)
        if candidate_action in ACTIONS:
            valid_action = candidate_action
        else:
            valid_action = self.default_action
            self.failed_candidates.append(candidate_action)
        return valid_action

    def _extract_final_answer(self, answer):
        """Sanitize the final answer, keeping only alphabetic characters.

        Args:
            answer (LLMResponse): The response from the LLM.

        Returns:
            LLMResponse: The sanitized response.
        """

        def filter_letters(input_string):
            return re.sub(r"[^a-zA-Z\s:_]", "", input_string)

        final_answer = copy.deepcopy(answer)
        self.prompt_builder.update_reasoning(answer.completion)
        final_answer = final_answer._replace(reasoning=final_answer.completion)
        final_answer = final_answer._replace(completion=filter_letters(final_answer.completion).split("ACTION:")[-1].strip())

        return final_answer
    

class PlannerAgent(BaseAgent):
    """An agent that generates actions based on observations without complex reasoning."""

    def __init__(self, agent_name, client, prompt_builder):
        super().__init__(agent_name, client, prompt_builder)
        self.default_action = 'NOOP'
        self.failed_objectives = []  # To track invalid actions suggested by the LLM
        self.failed_candidates = []
        self.current_objective = None
        self.current_plan = None
        self.plan_index = 0

    def execute_plan(self, plan):
        if self.plan_index < len(plan) - 1:
            action = plan[self.plan_index]
            # if self.check_action_validity(action):
            self.plan_index += 1
            return action
        else:
            action = plan[self.plan_index]
            self.current_plan = None
            self.plan_index = 0
            # else:
            #     self.failed_candidates.append(action)
            #     return self.default_action
        return action
    
    def generate_plan(self, grid, coord, action):
        goal_coord = (START_POS[0] + coord[0], START_POS[1] + coord[1])
        # print(grid)
        # print(goal_coord)
        facing = grid[START_POS[0]][START_POS[1]]
        # print(facing)
        grid[START_POS[0]][START_POS[1]] = 0
        if grid[goal_coord[0], goal_coord[1]] == 1:
            neighbor_coords = [(goal_coord[0] + dx, goal_coord[1] + dy) for dx, dy in _MOVES]
            valid_neighbors = [c for c in neighbor_coords if 0 <= c[0] < grid.shape[0] and 0 <= c[1] < grid.shape[1] and grid[c[0], c[1]] == 0]
            if valid_neighbors:
                new_goal_coord = min(valid_neighbors, key=lambda c: abs(c[0] - START_POS[0]) + abs(c[1] - START_POS[1]))
            
                goal_dir = (goal_coord[0] - new_goal_coord[0], goal_coord[1] - new_goal_coord[1])
                correct_facing = DIR_TO_FACING[goal_dir]
                goal_coord = new_goal_coord
            else:
                print("No valid neighboring cells to navigate to!, selecting random goal")
                new_goal_coord = None
                while new_goal_coord is None:
                    goal_x = np.random.randint(0, grid.shape[0])
                    goal_y = np.random.randint(0, grid.shape[1])
                    if grid[goal_x, goal_y] == 0:
                        new_goal_coord = (goal_x, goal_y)
                # goal_dir = (goal_coord[0] - new_goal_coord[0], goal_coord[1] - new_goal_coord[1])
                # correct_facing = DIR_TO_FACING[goal_dir]
                correct_facing = None       
                goal_coord = new_goal_coord
                
        else:
            correct_facing = None
        if goal_coord is not None:
            navigate_plan = astar(np.array(grid), START_POS, goal_coord)
            # print(navigate_plan)
            if navigate_plan is not None:
                if len(navigate_plan) > 0:
                    facing = navigate_plan[-1] + 1
                if correct_facing is not None and facing != correct_facing:
                    navigate_plan.append(correct_facing - 1)
            return navigate_plan + [ACTIONS[action]] if navigate_plan is not None else [ACTIONS[action]] 
        else:
            return [ACTIONS[action]]

    def act(self, obs, step):
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            str: The selected action from the LLM response.
        """

        self.prompt_builder.update_observation(obs,step)

        # print(prev_action)
        if self.current_plan is not None:
            action = self.execute_plan(self.current_plan)
            return action, None
        else:
        
        # if prev_action:
        #     self.prompt_builder.update_action(prev_action)
            # verification_messages = self.prompt_builder.get_verification_prompt()

            # verification_instructions = """

            
            # """.strip()

            if self.current_objective is None:
                proposal_messages = self.prompt_builder.get_proposal_prompt()

                achievemnet_proposal_instructions = """
            First think about what's the best acheivement you can unlock step by step.
            Finally, select a valid objective that you would like to execute in the following format: OBJECTIVE: <objective>
                    """.strip()

                proposal_messages[-1].content += "\n\n" + achievemnet_proposal_instructions
            
                while self.current_objective is None: 

                    # Generate the CoT reasoning
                    cot_prop_reasoning = self.client.generate(proposal_messages)
                    prop_final_answer = self._extract_curent_objective(cot_prop_reasoning)

                    self.current_objective = self.check_objective_validity(prop_final_answer.completion)

                    if self.current_objective is None:
                        proposal_messages[-1].content += f"\n\nYour previous output: {self.failed_objectives[-1]}, did not contain a valid objective. Please try again.\n"


                    print(cot_prop_reasoning.completion)
            else:
                proposal_messages = self.prompt_builder.get_proposal_prompt()

                achievemnet_proposal_instructions = f"""
            Your current objective is to: {self.current_objective}. Given the current observation,     
            First think step by step about whether the acheievement is complete or not.
            If the objective is complete, select a new valid objective, else continue with the current objective. Output the selected objective in the following format: OBJECTIVE: <objective>
                    """.strip()
                
                proposal_messages[-1].content += "\n\n" + achievemnet_proposal_instructions

                cot_prop_reasoning = self.client.generate(proposal_messages)
                prop_final_answer = self._extract_curent_objective(cot_prop_reasoning)

                new_objective = self.check_objective_validity(prop_final_answer.completion)
                if new_objective is not None and new_objective != self.current_objective:
                    self.current_objective = new_objective
                
                # print(cot_prop_reasoning.completion)


            plan_messages = self.prompt_builder.get_plan_prompt()

            plan_proposal_instructions = f"""
        Your current objective is to: {self.current_objective}. Given the current observation and your previous reasoning and action plans, 
        First think about what's the best course of action step by step to achieve this objective.
        Finally, select a new action plan that aligns with the objective. 
        First give the relative postion of the location to navigate to, use positive integers to indicate south and east and negative intergers to indicate north and west. 
        After that, give by a valid action. 
        Give your answer in the following format:
        THOUGHTS :  <thoughts>
        NAVIGATE TO: <(north/south, east/west)>
        ACTION: <action>

                    """.strip()
            
            plan_messages[-1].content += "\n\n" + plan_proposal_instructions

            cot_plan_reasoning = self.client.generate(plan_messages)
            coord, action, plan_final_answer = self._extract_plan(cot_plan_reasoning)
            # print(obs)

            # print(cot_prop_reasoning.completion)
            # print(cot_plan_reasoning.completion)
            # print(coord)
            # print(action)
            # exit()
            # print(obs["passable_grid"])

            self.current_plan = self.generate_plan(obs["passable_grid"], coord, action)
            # print(self.current_plan)
            # exit()

            action = self.execute_plan(self.current_plan)

            planner_output = {
                "objective": self.current_objective,
                "plan": self.current_plan,
                "reasoning": {
                    "objective_reasoning": prop_final_answer,
                    "plan_reasoning": plan_final_answer
                }
            }

            return action, planner_output
        # return final_answer, 0
    
    def check_objective_validity(self, candidate_obj):
        valid_objective = None
        print(f"Checking validity of objective: {candidate_obj}")
        # print(candidate_action in ACTIONS)
        if candidate_obj in OBJECTIVES:
            valid_objective = candidate_obj
            # return valid_objective
        else:
            # valid_action = self.default_action
            self.failed_objectives.append(candidate_obj)
        return valid_objective

    def _extract_plan(self, completion: str):
        """Extract NAVIGATE coordinates and ACTION from an LLM plan completion.

        Expects the format produced by ``achievemnet_proposal_instructions``:
            NAVIGATE TO (<x>, <y>), ACTION: <ACTION_NAME>

        Args:
            completion: Raw LLM completion string.

        Returns:
            tuple[tuple[int, int] | None, str | None]:
                ``(coords, action)`` where *coords* is ``(x, y)`` and *action*
                is a key from ``ACTIONS``, or ``None`` for either field when the
                corresponding part cannot be parsed.
        """
        coords = None
        action = None

        nav_match = re.search(r"NAVIGATE\s+TO:\s+\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", completion.completion, re.IGNORECASE)
        if nav_match:
            coords = (int(nav_match.group(1)), int(nav_match.group(2)))
        else:
            coords = (0, 0)

        action_match = re.search(r"ACTION:\s*([A-Z_]+)", completion.completion)
        if action_match:
            candidate = action_match.group(1).strip()
            if candidate in ACTIONS:
                action = candidate
            else:
                self.failed_candidates.append(candidate)
                action = self.default_action

        final_answer = copy.deepcopy(completion)
        self.prompt_builder.update_reasoning(completion.completion, type="plan")
        final_answer = final_answer._replace(reasoning=final_answer.completion)
        # final_answer = final_answer._replace(completion=filter_letters(final_answer.completion).split("OBJECTIVE:")[-1].strip())

        return coords, action, final_answer

    def _extract_curent_objective(self, answer):
        """Sanitize the final answer, keeping only alphabetic characters.

        Args:
            answer (LLMResponse): The response from the LLM.

        Returns:
            LLMResponse: The sanitized response.
        """

        def filter_letters(input_string):
            return re.sub(r"[^a-zA-Z\s:_]", "", input_string)

        final_answer = copy.deepcopy(answer)
        self.prompt_builder.update_reasoning(answer.completion, type="objective")
        final_answer = final_answer._replace(reasoning=final_answer.completion)
        final_answer = final_answer._replace(completion=filter_letters(final_answer.completion).split("OBJECTIVE:")[-1].strip())

        return final_answer
    

class PlannerAgentMA(BaseAgent):
    """An agent that generates actions based on observations without complex reasoning."""

    def __init__(self, agent_name, client, prompt_builder, dialogue_prompt_builder):
        super().__init__(agent_name, client, prompt_builder)
        self.default_action = 'NOOP'
        self.failed_objectives = []  # To track invalid actions suggested by the LLM
        self.failed_candidates = []
        self.current_objective = None
        self.current_plan = None
        self.plan_index = 0
        self.dialogue_prompt_builder = dialogue_prompt_builder

    def execute_plan(self, plan):
        if self.plan_index < len(plan) - 1:
            action = plan[self.plan_index]
            # if self.check_action_validity(action):
            self.plan_index += 1
            return action
        else:
            action = plan[self.plan_index]
            self.current_plan = None
            self.plan_index = 0
            # else:
            #     self.failed_candidates.append(action)
            #     return self.default_action
        return action
    
    def generate_plan(self, grid, action, coord=None):

        def select_random_goal(grid):
            new_goal_coord = None
            while new_goal_coord is None:
                goal_x = np.random.randint(0, grid.shape[0])
                goal_y = np.random.randint(0, grid.shape[1])
                if grid[goal_x, goal_y] == 0:
                    new_goal_coord = (goal_x, goal_y)
            return new_goal_coord

        if coord is None:
            goal_coord = select_random_goal(grid)
        else:
            goal_coord = (START_POS[0] + coord[0], START_POS[1] + coord[1])
        # print(grid)
        # print(goal_coord)
        facing = grid[START_POS[0]][START_POS[1]]
        # print(facing)
        grid[START_POS[0]][START_POS[1]] = 0
        if grid[goal_coord[0], goal_coord[1]] == 1:
            neighbor_coords = [(goal_coord[0] + dx, goal_coord[1] + dy) for dx, dy in _MOVES]
            valid_neighbors = [c for c in neighbor_coords if 0 <= c[0] < grid.shape[0] and 0 <= c[1] < grid.shape[1] and grid[c[0], c[1]] == 0]
            if valid_neighbors:
                new_goal_coord = min(valid_neighbors, key=lambda c: abs(c[0] - START_POS[0]) + abs(c[1] - START_POS[1]))
            
                goal_dir = (goal_coord[0] - new_goal_coord[0], goal_coord[1] - new_goal_coord[1])
                correct_facing = DIR_TO_FACING[goal_dir]
                goal_coord = new_goal_coord
            else:
                print("No valid neighboring cells to navigate to!, selecting random goal")
                new_goal_coord = select_random_goal(grid)
                # goal_dir = (goal_coord[0] - new_goal_coord[0], goal_coord[1] - new_goal_coord[1])
                # correct_facing = DIR_TO_FACING[goal_dir] 
                correct_facing = None       
                goal_coord = new_goal_coord
        else:
            correct_facing = None
        if goal_coord is not None:
            navigate_plan = astar(np.array(grid), START_POS, goal_coord)
            # print(navigate_plan)
            if navigate_plan is not None:
                if len(navigate_plan) > 0:
                    facing = navigate_plan[-1] + 1
                if correct_facing is not None and facing != correct_facing:
                    navigate_plan.append(correct_facing - 1)
            return navigate_plan + [ACTIONS[action]] if navigate_plan is not None else [ACTIONS[action]] 
        else:
            return [ACTIONS[action]]

    def dialogue_reset(self):
        self.dialogue_prompt_builder.reset()

    def dialogue_act(self, round):
        
        dialogue_messages = self.dialogue_prompt_builder.get_next_dialogue_prompt()
        
        dialogue_process_prompt = f"""

    You are current in round {round} of the dialogue. 
    Given the dialogue history and the current observation, you will have to generate your next dialogue turn. Your dialogue turn can be a proposal of an objective and a plan to achieve it, or a counter-proposal to the current proposed objective and plan, or an agreement with the current proposed objective and plan.
    
    "LOCAL OBSERVATION": Share Relevant information about your observation and status here to your teamates. E.g. I see a water tile 5 tiles west of me, I have 2 wood in my inventory, I have low health, etc.
    "OBJECTIVES" : Propose or ammend the objectives for youself and your teamates. The objectives must be selected from valid list of objectives.

    Ground your proposed objectives on your current obsevation, status (e.g. hunger and energy levels), current inventory, as well as whether you have the required equipment (if not you should select a prerequisite objective). Also take into account the information shared by your teamates, keep in mind the different roles of your and your teamates as well as the overall goal to complete the objectives efficeintly.

    Give your dialougue response in the following JSON format:
    {{
        "LOCAL OBSERVATION": <observation text>,
        "THOUGHTS": <thoughts>,
        "AGENT 0 OBJECTIVE": <agent 0's objective>,
        "AGENT 1 OBJECTIVE": <agent 1's objective>,
        "AGENT 2 OBJECTIVE": <agent 3's objective>
    }}
    
        """.strip()

        dialogue_messages[-1].content += "\n\n" + dialogue_process_prompt

        dialogue_response = self.client.generate(dialogue_messages)

        dialogue_info, dialogue_answer = self.extract_dialogue_info(dialogue_response, round)

        return dialogue_info, dialogue_answer
    
    def update_dialogue(self, response, agent_name, round):
        self.dialogue_prompt_builder.update_dialogue(response,agent_name, round)

    def act(self, obs, step, objective=None):
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            str: The selected action from the LLM response.
        """

        self.prompt_builder.update_observation(obs,step)

        # print(prev_action)
        if self.current_plan is not None:
            action = self.execute_plan(self.current_plan)
            return action, None
        else:
        
        # if prev_action:
        #     self.prompt_builder.update_action(prev_action)
            # verification_messages = self.prompt_builder.get_verification_prompt()

            # verification_instructions = """

            
            # """.strip()
            self.current_objective = objective

            if self.current_objective is None:
                planner_output = {
                    "objective": self.current_objective,
                    "plan": self.current_plan,
                    "reasoning": {
                        "objective_reasoning": None,
                        "plan_reasoning": None
                    },
                    "objective_completion": True
                }

                return ACTIONS["NOOP"], planner_output
            else:
                proposal_messages = self.prompt_builder.get_proposal_prompt()

                achievemnet_proposal_instructions = f"""
            Your current objective is to: {self.current_objective}. Given the current observation,     
            First think step by step about whether the acheievement is complete or not.
            Output the status of objective in the following format: 
            THOGHTS: <thoughts>
            OBJECTIVE COMPLETE: <objective status, either TRUE or FALSE>
                    """.strip()
                
                proposal_messages[-1].content += "\n\n" + achievemnet_proposal_instructions

                cot_prop_reasoning = self.client.generate(proposal_messages)
                prop_final_answer = self._extract_curent_objective_status(cot_prop_reasoning)

                # new_objective = self.check_objective_validity(prop_final_answer.completion)
                # if new_objective is not None and new_objective != self.current_objective:
                #     self.current_objective = new_objective
                
                # print(cot_prop_reasoning.completion)
                if "TRUE" in prop_final_answer.completion.upper():
                    self.current_objective = None
                    planner_output = {
                        "objective": self.current_objective,
                        "plan": self.current_plan,
                        "reasoning": {
                            "objective_reasoning": prop_final_answer,
                            "plan_reasoning": None
                        },
                        "objective_completion": True
                    }


                    return ACTIONS["NOOP"], planner_output
                else:


                    plan_messages = self.prompt_builder.get_plan_prompt()

                    plan_proposal_instructions = f"""
                Your current objective is to: {self.current_objective}. Given the current observation and your previous reasoning and action plans, 
                First think about what's the best course of action step by step to achieve this objective. 
                Finally, select a new action plan that aligns with the objective. 
                First give the relative postion of the location to navigate to, use positive integers to indicate south and east and negative intergers to indicate north and west. Select positions where you can reach, i.e. there no obstacles adjacent to the location.
                After that, give by a valid action to perform after navigation. 
                Give your answer in the following format:
                THOUGHTS :  <thoughts>
                NAVIGATE TO: <(north/south, east/west)>
                ACTION: <action>

                            """.strip()
                    
                    plan_messages[-1].content += "\n\n" + plan_proposal_instructions

                    cot_plan_reasoning = self.client.generate(plan_messages)
                    coord, action, plan_final_answer = self._extract_plan(cot_plan_reasoning)
                    # print(obs)

                    # print(cot_prop_reasoning.completion)
                    # print(cot_plan_reasoning.completion)
                    # print(coord)
                    # print(action)
                    # exit()
                    # print(obs["passable_grid"])

                    self.current_plan = self.generate_plan(obs["passable_grid"], coord, action)
                    # print(self.current_plan)
                    # exit()

                    action = self.execute_plan(self.current_plan)

                    planner_output = {
                        "objective": self.current_objective,
                        "plan": self.current_plan,
                        "reasoning": {
                            "objective_reasoning": prop_final_answer,
                            "plan_reasoning": plan_final_answer
                        },
                        "objective_completion": False
                    }

                    return action, planner_output
        # return final_answer, 0
    
    
    def check_objective_validity(self, candidate_obj):
        valid_objective = None
        print(f"Checking validity of objective: {candidate_obj}")
        # print(candidate_action in ACTIONS)
        if candidate_obj in OBJECTIVES:
            valid_objective = candidate_obj
            # return valid_objective
        else:
            # valid_action = self.default_action
            self.failed_objectives.append(candidate_obj)
        return valid_objective
    
    def extract_dialogue_info(self, dialogue_response, round):
        dialogue_info = {
            "local_observation": None,
            "thoughts": None,
            "agent_objectives": {}
        }

        try:
            content = dialogue_response.completion.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            dialogue_json = json.loads(content)
            dialogue_info["local_observation"] = dialogue_json.get("LOCAL OBSERVATION", None)
            dialogue_info["thoughts"] = dialogue_json.get("THOUGHTS", None)
            for key, value in dialogue_json.items():
                if key.startswith("AGENT") and key.endswith("OBJECTIVE"):
                    new_key = key.replace(" OBJECTIVE", "").replace(" ", "_").lower()  # e.g., "AGENT 0 OBJECTIVE" -> "agent_0"
                    dialogue_info["agent_objectives"][new_key] = value
        except json.JSONDecodeError:
            print(dialogue_response.completion)
            print("Failed to parse dialogue response as JSON.")

        dialogue_answer = copy.deepcopy(dialogue_response)
        self.dialogue_prompt_builder.update_dialogue(dialogue_info,self.agent_name, round)

        dialogue_answer = dialogue_answer._replace(reasoning=dialogue_answer.completion)
        
        return dialogue_info, dialogue_answer

    def _extract_plan(self, completion: str):
        """Extract NAVIGATE coordinates and ACTION from an LLM plan completion.

        Expects the format produced by ``achievemnet_proposal_instructions``:
            NAVIGATE TO (<x>, <y>), ACTION: <ACTION_NAME>

        Args:
            completion: Raw LLM completion string.

        Returns:
            tuple[tuple[int, int] | None, str | None]:
                ``(coords, action)`` where *coords* is ``(x, y)`` and *action*
                is a key from ``ACTIONS``, or ``None`` for either field when the
                corresponding part cannot be parsed.
        """
        coords = None
        action = None

        nav_match = re.search(r"NAVIGATE\s+TO:\s+\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", completion.completion, re.IGNORECASE)
        if nav_match:
            coords = (int(nav_match.group(1)), int(nav_match.group(2)))
        else:
            coords = (0, 0)

        action_match = re.search(r"ACTION:\s*([A-Z_]+)", completion.completion)
        if action_match:
            candidate = action_match.group(1).strip()
            if candidate in ACTIONS:
                action = candidate
            else:
                self.failed_candidates.append(candidate)
                action = self.default_action

        final_answer = copy.deepcopy(completion)
        self.prompt_builder.update_reasoning(completion.completion, type="plan")
        final_answer = final_answer._replace(reasoning=final_answer.completion)
        # final_answer = final_answer._replace(completion=filter_letters(final_answer.completion).split("OBJECTIVE:")[-1].strip())

        return coords, action, final_answer

    def _extract_curent_objective_status(self, answer):
        """Sanitize the final answer, keeping only alphabetic characters.

        Args:
            answer (LLMResponse): The response from the LLM.

        Returns:
            LLMResponse: The sanitized response.
        """

        def filter_letters(input_string):
            return re.sub(r"[^a-zA-Z\s:_]", "", input_string)

        final_answer = copy.deepcopy(answer)
        self.prompt_builder.update_reasoning(answer.completion, type="verification")
        final_answer = final_answer._replace(reasoning=final_answer.completion) 
        final_answer = final_answer._replace(completion=filter_letters(final_answer.completion).split("OBJECTIVE COMPLETE:")[-1].strip())

        return final_answer