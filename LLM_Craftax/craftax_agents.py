import copy
import re
from llm_client import LLMResponse

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
    
