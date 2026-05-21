from collections import deque
from typing import List, Optional
import copy


class Message:
    """Represents a conversation message with role, content, and optional attachment."""

    def __init__(self, role: str, content: str, attachment: Optional[object] = None):
        self.role = role  # 'system', 'user', 'assistant'
        self.content = content  # String content of the message
        self.attachment = attachment

    def __repr__(self):
        return f"Message(role={self.role}, content={self.content}, attachment={self.attachment})"


class HistoryPromptBuilder:
    """Builds a prompt with a history of observations, actions, and reasoning.

    Maintains a configurable history of text, images, and chain-of-thought reasoning to
    construct prompt messages for conversational agents.
    """

    def __init__(
        self,
        config
    ):
        self.max_text_history = config.max_text_history
        self.max_image_history = config.max_image_history
        self.max_history = max(self.max_text_history, self.max_image_history)
        self.system_prompt = None
        self._events = deque(maxlen=self.max_history * 2)  # Stores observations and actions
        self._last_short_term_obs = None  # To store the latest short-term observation
        self.previous_reasoning = None
        self.max_objective_history = config.max_objective_history
        self.max_cot_history = config.max_cot_history

    def update_instruction_prompt(self, instruction: str):
        """Set the system-level instruction prompt."""
        self.system_prompt = instruction

    def update_observation(self, obs: dict):
        """Add an observation to the prompt history, which can include text, an image, or both."""
        long_term_context = obs.get("long_term_context", "")
        self._last_short_term_obs = obs.get("short_term_context", "")
        text = long_term_context

        # Add observation to events
        self._events.append(
            {
                "type": "observation",
                "text": text,
            }
        )

    def update_action(self, action: str):
        """Add an action to the prompt history, including reasoning if available."""
        self._events.append(
            {
                "type": "action",
                "action": action,
                "reasoning": self.previous_reasoning,
            }
        )

    def update_reasoning(self, reasoning: str):
        """Set the reasoning text to be included with subsequent actions."""
        self.previous_reasoning = reasoning

    def reset(self):
        """Clear the event history."""
        self._events.clear()

    def get_prompt(self, icl_episodes=False) -> List[Message]:
        """Generate a list of Message objects representing the prompt.

        Returns:
            List[Message]: Messages constructed from the event history.
        """
        messages = []

        if self.system_prompt and not icl_episodes:
            messages.append(Message(role="system", content=self.system_prompt))

        # Determine which text observations to include
        text_needed = self.max_text_history
        for event in reversed(self._events):
            if event["type"] == "observation":
                if text_needed > 0 and event.get("text") is not None:
                    event["include_text"] = True
                    text_needed -= 1
                else:
                    event["include_text"] = False

        # Determine which image observations to include
        # images_needed = self.max_image_history
        # for event in reversed(self._events):
        #     if event["type"] == "observation":
        #         if images_needed > 0 and event.get("image") is not None:
        #             event["include_image"] = True
        #             images_needed -= 1
        #         else:
        #             event["include_image"] = False

        # determine the reasoning to include
        reasoning_needed = self.max_cot_history
        for event in reversed(self._events):
            if event["type"] == "action":
                if reasoning_needed > 0 and event.get("reasoning") is not None:
                    reasoning_needed -= 1
                else:
                    event["reasoning"] = None

        # Process events to create messages
        # print(self._events)
        # exit()
        for idx, event in enumerate(self._events):
            if event["type"] == "observation":
                message_parts = []

                if idx == len(self._events) - 1:
                    if self._last_short_term_obs:
                        message_parts.append("Current Status:")
                        message_parts.append(self._last_short_term_obs)
                    message_parts.append("Current Observation:")
                else:
                    message_parts.append(f"Observation from {int((len(self._events) - idx)/2)} steps ago:")

                if event.get("include_text", False):
                    message_parts.append(event["text"])
                    
                # image = None
                # if event.get("include_image", False):
                #     image = event["image"]
                #     message_parts.append("Image observation provided.")

                content = "\n".join(message_parts)
                message = Message(role="user", content=content)
                # message = Message(role="user", content=content, attachment=image)

                # Clean up temporary flags
                for flag in ["include_text"]:#, "include_image"]:
                    if flag in event:
                        del event[flag]
            elif event["type"] == "action":
                if event.get("reasoning") is not None:
                    content = f"Plan from {int((len(self._events) - idx)/2)} steps ago: \n" + event["reasoning"]
                else:
                    content = f"Action from {int((len(self._events) - idx)/2)} steps ago: \n" + event["action"]
                message = Message(role="user", content=content)
            messages.append(message)

        return messages


class HistoryandPlansPromptBuilder:
    def __init__(
        self,
        config
    ):
        self.max_text_history = config.max_text_history
        self.max_image_history = config.max_image_history
        self.max_history = max(self.max_text_history, self.max_image_history)
        self.proposal_system_prompt = None
        self.plans_system_prompt = None
        self._events = deque(maxlen=self.max_history * 2)  # Stores observations and actions
        self._last_short_term_obs = None  # To store the latest short-term observation
        self.previous_reasoning = None
        self.current_step = None
        self.max_objective_history = config.max_objective_history
        self.max_cot_history = config.max_cot_history

    def update_instruction_prompt(self, instruction: str):
        """Set the system-level instruction prompt."""
        self.proposal_system_prompt = instruction

    def update_observation(self, obs: dict, step : int):
        """Add an observation to the prompt history, which can include text, an image, or both."""
        long_term_context = obs.get("long_term_context", "")
        self._last_short_term_obs = obs.get("short_term_context", "")
        text = long_term_context
        self.current_step = step
        # Add observation to events
        self._events.append(
            {
                "type": "observation",
                "text": text,
                "step" : step
            }
        )

    # def update_action(self, action: str):
    #     """Add an action to the prompt history, including reasoning if available."""
        # self._events.append(
        #     {
        #         "type": "action",
        #         "action": action,
        #         "reasoning": self.previous_reasoning,
        #     }
        # )

    def update_reasoning(self, reasoning: str, type=None):
        """Set the reasoning text to be included with subsequent actions."""
        # self.previous_reasoning = reasoning
        self._events.append(
            {
                "type": "reasoning",
                # "action": action,
                "reasoning": reasoning,
                "reasoning_type": type

            }
        )

    def reset(self):
        """Clear the event history."""
        self._events.clear()

    def get_proposal_prompt(self, icl_episodes=False) -> List[Message]:
        """Generate a list of Message objects representing the prompt.

        Returns:
            List[Message]: Messages constructed from the event history.
        """
        messages = []

        if self.proposal_system_prompt and not icl_episodes:
            messages.append(Message(role="system", content=self.proposal_system_prompt))

        # Determine which text observations to include
        events = copy.deepcopy(self._events)
        text_needed = self.max_text_history
        for event in reversed(events):
            if event["type"] == "observation":
                if text_needed > 0 and event.get("text") is not None:
                    event["include_text"] = True
                    text_needed -= 1
                else:
                    event["include_text"] = False

        # Determine which image observations to include
        # images_needed = self.max_image_history
        # for event in reversed(self._events):
        #     if event["type"] == "observation":
        #         if images_needed > 0 and event.get("image") is not None:
        #             event["include_image"] = True
        #             images_needed -= 1
        #         else:
        #             event["include_image"] = False

        # determine the reasoning to include
        obj_reasoning_needed = self.max_objective_history
        cot_reasoning_needed = self.max_cot_history
        for event in reversed(events):
            if event["type"] == "reasoning":
                if event.get("reasoning_type") == "objective":
                    if obj_reasoning_needed > 0 and event.get("reasoning") is not None:
                        obj_reasoning_needed -= 1
                    else:
                        event["reasoning"] = None
                elif event.get("reasoning_type") == "plan":
                    if cot_reasoning_needed > 0 and event.get("reasoning") is not None:
                        cot_reasoning_needed -= 1
                    else:
                        event["reasoning"] = None
                else:
                    event["reasoning"] = None

        # Process events to create messages
        # print(self._events)
        # exit()
        for idx, event in enumerate(events):
            if event["type"] == "observation":
                message_parts = []

                if event["step"] == self.current_step:
                    if self._last_short_term_obs:
                        message_parts.append("Current Status:")
                        message_parts.append(self._last_short_term_obs)
                    message_parts.append("Current Observation:")
                else:
                    message_parts.append(f"Observation from {(self.current_step - event['step'])} steps ago:")

                if event.get("include_text", False):
                    message_parts.append(event["text"])
                    
                # image = None
                # if event.get("include_image", False):
                #     image = event["image"]
                #     message_parts.append("Image observation provided.")

                content = "\n".join(message_parts)
                message = Message(role="user", content=content)
                # message = Message(role="user", content=content, attachment=image)

                # Clean up temporary flags
                for flag in ["include_text"]:#, "include_image"]:
                    if flag in event:
                        del event[flag]
                messages.append(message)
            elif event["type"] == "reasoning":
                if event.get("reasoning") is not None:
                    if event.get("reasoning_type") == "objective":
                        content = f"Reasoning for selecting previous achievement: \n" + event["reasoning"]
                    # elif event.get("reasoning_type") == "plan":
                    #     content = f"Reasoning for previous actionplan {int((len(events) - idx)/2)} steps ago: \n" + event["reasoning"]
                # else:
                #     content = f"Action from {int((len(events) - idx)/2)} steps ago: \n" + event["action"]
                        message = Message(role="user", content=content)
                        messages.append(message)

        return messages
    
    def get_plan_prompt(self, icl_episodes=False) -> List[Message]:
        """Generate a list of Message objects representing the prompt.

        Returns:
            List[Message]: Messages constructed from the event history.
        """
        messages = []
        # print(self.proposal_system_prompt)
        if self.proposal_system_prompt and not icl_episodes:
            messages.append(Message(role="system", content=self.proposal_system_prompt))
        events = copy.deepcopy(self._events)
        
        # print(messages)
        # Determine which text observations to include
        text_needed = self.max_text_history
        for event in reversed(events):
            if event["type"] == "observation":
                if text_needed > 0 and event.get("text") is not None:
                    event["include_text"] = True
                    text_needed -= 1
                else:
                    event["include_text"] = False
        # print(events)
        # Determine which image observations to include
        # images_needed = self.max_image_history
        # for event in reversed(self._events):
        #     if event["type"] == "observation":
        #         if images_needed > 0 and event.get("image") is not None:
        #             event["include_image"] = True
        #             images_needed -= 1
        #         else:
        #             event["include_image"] = False

        # determine the reasoning to include
        obj_reasoning_needed = self.max_objective_history
        cot_reasoning_needed = self.max_cot_history
        for event in reversed(events):
            if event["type"] == "reasoning":
                if event.get("reasoning_type") == "objective":
                    if obj_reasoning_needed > 0 and event.get("reasoning") is not None:
                        obj_reasoning_needed -= 1
                    else:
                        event["reasoning"] = None
                elif event.get("reasoning_type") == "plan":
                    if cot_reasoning_needed > 0 and event.get("reasoning") is not None:
                        cot_reasoning_needed -= 1
                    else:
                        event["reasoning"] = None
                else:
                    event["reasoning"] = None

        # Process events to create messages
        # print(self._events)
        # exit()
        for idx, event in enumerate(events):
            if event["type"] == "observation":
                message_parts = []

                if event["step"] == self.current_step:
                    if self._last_short_term_obs:
                        message_parts.append("Current Status:")
                        message_parts.append(self._last_short_term_obs)
                    message_parts.append("Current Observation:")
                else:
                    message_parts.append(f"Observation from {(self.current_step - event['step'])} steps ago:")

                if event.get("include_text", False):
                    message_parts.append(event["text"])
                    
                # image = None
                # if event.get("include_image", False):
                #     image = event["image"]
                #     message_parts.append("Image observation provided.")

                content = "\n".join(message_parts)
                message = Message(role="user", content=content)
                # message = Message(role="user", content=content, attachment=image)

                # Clean up temporary flags
                for flag in ["include_text"]:#, "include_image"]:
                    if flag in event:
                        del event[flag]
                messages.append(message)
            elif event["type"] == "reasoning":
                if event.get("reasoning") is not None:
                    if event.get("reasoning_type") == "objective":
                        content = f"Reasoning for current achievement: \n" + event["reasoning"]
                    elif event.get("reasoning_type") == "plan":
                        content = f"Reasoning for previous action plan : \n" + event["reasoning"]
                # else:
                #     content = f"Action from {int((len(events) - idx)/2)} steps ago: \n" + event["action"]
                    message = Message(role="user", content=content)
                    messages.append(message)

        # print(messages)
        return messages


class HistoryandPlansPromptBuilderv2:
    def __init__(
        self,
        config
    ):
        self.max_text_history = config.max_text_history
        self.max_image_history = config.max_image_history
        self.max_history = max(self.max_text_history, self.max_image_history)
        self.proposal_system_prompt = None
        self.plans_system_prompt = None
        self._events = deque(maxlen=self.max_history * 2)  # Stores observations and actions
        self._last_short_term_obs = None  # To store the latest short-term observation
        self.previous_reasoning = None
        self.current_step = None
        self.max_objective_history = config.max_objective_history
        self.max_cot_history = config.max_cot_history

    def update_instruction_prompt(self, instruction: str):
        """Set the system-level instruction prompt."""
        self.proposal_system_prompt = instruction

    def update_observation(self, obs: dict, step : int):
        """Add an observation to the prompt history, which can include text, an image, or both."""
        long_term_context = obs.get("long_term_context", "")
        self._last_short_term_obs = obs.get("short_term_context", "")
        text = long_term_context
        self.current_step = step
        # Add observation to events
        self._events.append(
            {
                "type": "observation",
                "text": text,
                "step" : step
            }
        )

    # def update_action(self, action: str):
    #     """Add an action to the prompt history, including reasoning if available."""
        # self._events.append(
        #     {
        #         "type": "action",
        #         "action": action,
        #         "reasoning": self.previous_reasoning,
        #     }
        # )

    def update_reasoning(self, reasoning: str, type=None):
        """Set the reasoning text to be included with subsequent actions."""
        # self.previous_reasoning = reasoning
        self._events.append(
            {
                "type": "reasoning",
                # "action": action,
                "reasoning": reasoning,
                "reasoning_type": type

            }
        )

    def reset(self):
        """Clear the event history."""
        self._events.clear()

    def get_proposal_prompt(self, icl_episodes=False) -> List[Message]:
        """Generate a list of Message objects representing the prompt.

        Returns:
            List[Message]: Messages constructed from the event history.
        """
        messages = []

        if self.proposal_system_prompt and not icl_episodes:
            messages.append(Message(role="system", content=self.proposal_system_prompt))

        # Determine which text observations to include
        events = copy.deepcopy(self._events)
        text_needed = self.max_text_history
        for event in reversed(events):
            if event["type"] == "observation":
                if text_needed > 0 and event.get("text") is not None:
                    event["include_text"] = True
                    text_needed -= 1
                else:
                    event["include_text"] = False

        # Determine which image observations to include
        # images_needed = self.max_image_history
        # for event in reversed(self._events):
        #     if event["type"] == "observation":
        #         if images_needed > 0 and event.get("image") is not None:
        #             event["include_image"] = True
        #             images_needed -= 1
        #         else:
        #             event["include_image"] = False

        # determine the reasoning to include
        obj_reasoning_needed = self.max_objective_history
        cot_reasoning_needed = self.max_cot_history
        for event in reversed(events):
            if event["type"] == "reasoning":
                if event.get("reasoning_type") == "objective":
                    if obj_reasoning_needed > 0 and event.get("reasoning") is not None:
                        obj_reasoning_needed -= 1
                    else:
                        event["reasoning"] = None
                elif event.get("reasoning_type") == "plan":
                    if cot_reasoning_needed > 0 and event.get("reasoning") is not None:
                        cot_reasoning_needed -= 1
                    else:
                        event["reasoning"] = None
                else:
                    event["reasoning"] = None

        # Process events to create messages
        # print(self._events)
        # exit()
        for idx, event in enumerate(events):
            if event["type"] == "observation":
                message_parts = []

                if event["step"] == self.current_step:
                    if self._last_short_term_obs:
                        message_parts.append("Current Status:")
                        message_parts.append(self._last_short_term_obs)
                    message_parts.append("Current Observation:")
                else:
                    message_parts.append(f"Observation from {(self.current_step - event['step'])} steps ago:")

                if event.get("include_text", False):
                    message_parts.append(event["text"])
                    
                # image = None
                # if event.get("include_image", False):
                #     image = event["image"]
                #     message_parts.append("Image observation provided.")

                content = "\n".join(message_parts)
                message = Message(role="user", content=content)
                # message = Message(role="user", content=content, attachment=image)

                # Clean up temporary flags
                for flag in ["include_text"]:#, "include_image"]:
                    if flag in event:
                        del event[flag]
                messages.append(message)
            elif event["type"] == "reasoning":
                if event.get("reasoning") is not None:
                    if event.get("reasoning_type") == "objective":
                        content = f"Reasoning for previous evaluation status for current objective: \n" + event["reasoning"]
                    # elif event.get("reasoning_type") == "plan":
                    #     content = f"Reasoning for previous actionplan {int((len(events) - idx)/2)} steps ago: \n" + event["reasoning"]
                # else:
                #     content = f"Action from {int((len(events) - idx)/2)} steps ago: \n" + event["action"]
                        message = Message(role="user", content=content)
                        messages.append(message)

        return messages
    
    def get_plan_prompt(self, icl_episodes=False) -> List[Message]:
        """Generate a list of Message objects representing the prompt.

        Returns:
            List[Message]: Messages constructed from the event history.
        """
        messages = []
        # print(self.proposal_system_prompt)
        if self.proposal_system_prompt and not icl_episodes:
            messages.append(Message(role="system", content=self.proposal_system_prompt))
        events = copy.deepcopy(self._events)
        
        # print(messages)
        # Determine which text observations to include
        text_needed = self.max_text_history
        for event in reversed(events):
            if event["type"] == "observation":
                if text_needed > 0 and event.get("text") is not None:
                    event["include_text"] = True
                    text_needed -= 1
                else:
                    event["include_text"] = False
        # print(events)
        # Determine which image observations to include
        # images_needed = self.max_image_history
        # for event in reversed(self._events):
        #     if event["type"] == "observation":
        #         if images_needed > 0 and event.get("image") is not None:
        #             event["include_image"] = True
        #             images_needed -= 1
        #         else:
        #             event["include_image"] = False

        # determine the reasoning to include
        obj_reasoning_needed = self.max_objective_history
        cot_reasoning_needed = self.max_cot_history
        for event in reversed(events):
            if event["type"] == "reasoning":
                if event.get("reasoning_type") == "objective":
                    if obj_reasoning_needed > 0 and event.get("reasoning") is not None:
                        obj_reasoning_needed -= 1
                    else:
                        event["reasoning"] = None
                elif event.get("reasoning_type") == "plan":
                    if cot_reasoning_needed > 0 and event.get("reasoning") is not None:
                        cot_reasoning_needed -= 1
                    else:
                        event["reasoning"] = None
                else:
                    event["reasoning"] = None

        # Process events to create messages
        # print(self._events)
        # exit()
        for idx, event in enumerate(events):
            if event["type"] == "observation":
                message_parts = []

                if event["step"] == self.current_step:
                    if self._last_short_term_obs:
                        message_parts.append("Current Status:")
                        message_parts.append(self._last_short_term_obs)
                    message_parts.append("Current Observation:")
                else:
                    message_parts.append(f"Observation from {(self.current_step - event['step'])} steps ago:")

                if event.get("include_text", False):
                    message_parts.append(event["text"])
                    
                # image = None
                # if event.get("include_image", False):
                #     image = event["image"]
                #     message_parts.append("Image observation provided.")

                content = "\n".join(message_parts)
                message = Message(role="user", content=content)
                # message = Message(role="user", content=content, attachment=image)

                # Clean up temporary flags
                for flag in ["include_text"]:#, "include_image"]:
                    if flag in event:
                        del event[flag]
                messages.append(message)
            elif event["type"] == "reasoning":
                if event.get("reasoning") is not None:
                    # if event.get("reasoning_type") == "objective":
                    #     content = f"Reasoning for current achievement: \n" + event["reasoning"]
                    if event.get("reasoning_type") == "plan":
                        content = f"Reasoning for previous action plan : \n" + event["reasoning"]
                # else:
                #     content = f"Action from {int((len(events) - idx)/2)} steps ago: \n" + event["action"]
                    message = Message(role="user", content=content)
                    messages.append(message)

        # print(messages)
        return messages


class TaskExecutionPromptBuilder:
    def __init__(
        self,
        config
    ):
        self.max_history = config.max_text_history
        self.plans_system_prompt = None
        self._events = deque(maxlen=self.max_history * 2)  # Stores observations and actions
        self._last_short_term_obs = None  # To store the latest short-term observation

    def update_instruction_prompt(self, instruction: str):
        """Set the system-level instruction prompt."""
        self.proposal_system_prompt = instruction

    def update_observation(self, obs: dict, step : int):
        """Add an observation to the prompt history, which can include text, an image, or both."""
        long_term_context = obs.get("long_term_context", "")
        self._last_short_term_obs = obs.get("short_term_context", "")
        text = long_term_context
        self.current_step = step
        # Add observation to events
        self._events.append(
            {
                "type": "observation",
                "text": text,
                "step" : step
            }
        )

    def reset(self):
        """Clear the event history."""
        self._events.clear()

    def get_exec_prompt(self, icl_episodes=False) -> List[Message]:
        """Generate a list of Message objects representing the prompt.

        Returns:
            List[Message]: Messages constructed from the event history.
        """
        messages = []
        if self.proposal_system_prompt and not icl_episodes:
            messages.append(Message(role="system", content=self.proposal_system_prompt))
        events = copy.deepcopy(self._events)
        
        for idx, event in enumerate(events):
            if event["type"] == "observation":
                message_parts = []

                if event["step"] == self.current_step:
                    if self._last_short_term_obs:
                        message_parts.append("Current Status:")
                        message_parts.append(self._last_short_term_obs)
                    message_parts.append("Current Observation:")

                    message_parts.append(event["text"])
                    
                    content = "\n".join(message_parts)
                    message = Message(role="user", content=content)
                    messages.append(message)
            else:
                continue

        return messages


class DialogueHistoryPromptBuilder:
    def __init__(
        self,
        config,
        agent_name
    ):
        # self.max_text_history = config.max_text_history
        # self.max_image_history = config.max_image_history
        # self.max_history = max(self.max_text_history, self.max_image_history)
        self.system_prompt = None
        self.instruction_prompt = None
        self._events = deque(maxlen=config.num_agents * config.dialogue_rounds)  # Stores observations and actions
        self._last_short_term_obs = None  # To store the latest short-term observation
        # self.previous_reasoning = None
        # self.current_step = None
        # self.max_objective_history = config.max_objective_history
        # self.max_cot_history = config.max_cot_history

        self.agent_name = agent_name
        self.config = config

    def update_instruction_prompt(self, sysprompt: str):
        """Set the system-level instruction prompt."""
        self.system_prompt = sysprompt

    def update_dialogue_instruction_prompt(self, instruction: str):
        self.instruction_prompt = instruction

    def update_observation(self, obs: dict):
        """Add an observation to the prompt history, which can include text, an image, or both."""
        long_term_context = obs.get("long_term_context", "")
        self._last_short_term_obs = obs.get("short_term_context", "")
        text = long_term_context
        # self.current_step = step
        # Add observation to events
        self._events.append(
            {
                "type": "observation",
                "text": text,
                # "step" : step
            }
        )

    # def update_action(self, action: str):
    #     """Add an action to the prompt history, including reasoning if available."""
        # self._events.append(
        #     {
        #         "type": "action",
        #         "action": action,
        #         "reasoning": self.previous_reasoning,
        #     }
        # )

    def update_dialogue(self, dialogue: str, agent_name : int, round : int):
        
        plan_proposal = f""""
        AGENT 0 OBJECTIVE : {dialogue["agent_objectives"].get("agent_0", "N/A")}
        AGENT 1 OBJECTIVE : {dialogue["agent_objectives"].get("agent_1", "N/A")}
        AGENT 2 OBJECTIVE : {dialogue["agent_objectives"].get("agent_2", "N/A")}
        """
        
        self._events.append(
            {
                "type": "dialogue",
                'author' : agent_name,
                "proposal": plan_proposal,
                "thoughts": dialogue.get("thoughts", None),
                "obs_info": dialogue.get("local_observation", None),
                "round" : round
            }
        )

    # def update_reasoning(self, reasoning: str, type=None):
    #     """Set the reasoning text to be included with subsequent actions."""
    #     # self.previous_reasoning = reasoning
    #     self._events.append(
    #         {
    #             "type": "reasoning",
    #             # "action": action,
    #             "reasoning": reasoning,
    #             "reasoning_type": type

    #         }
    #     )

    def reset(self):
        """Clear the event history."""
        self._events.clear()
    
    def get_next_dialogue_prompt(self) -> List[Message]:
        """Generate a list of Message objects representing the prompt.

        Returns:
            List[Message]: Messages constructed from the event history.
        """
        messages = []
        # print(self.proposal_system_prompt)
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        if self.instruction_prompt:
            messages.append(Message(role="system", content=self.instruction_prompt))
        events = copy.deepcopy(self._events)
        
        # print(messages)
        # Determine which text observations to include
        # text_needed = self.max_text_history
        # for event in reversed(events):
        #     if event["type"] == "observation":
        #         if text_needed > 0 and event.get("text") is not None:
        #             event["include_text"] = True
        #             text_needed -= 1
        #         else:
        #             event["include_text"] = False
        # print(events)
        # Determine which image observations to include
        # images_needed = self.max_image_history
        # for event in reversed(self._events):
        #     if event["type"] == "observation":
        #         if images_needed > 0 and event.get("image") is not None:
        #             event["include_image"] = True
        #             images_needed -= 1
        #         else:
        #             event["include_image"] = False

        # determine the reasoning to include
        # obj_reasoning_needed = self.max_objective_history
        # cot_reasoning_needed = self.max_cot_history
        # for event in reversed(events):
        #     if event["type"] == "reasoning":
        #         if event.get("reasoning_type") == "objective":
        #             if obj_reasoning_needed > 0 and event.get("reasoning") is not None:
        #                 obj_reasoning_needed -= 1
        #             else:
        #                 event["reasoning"] = None
        #         elif event.get("reasoning_type") == "plan":
        #             if cot_reasoning_needed > 0 and event.get("reasoning") is not None:
        #                 cot_reasoning_needed -= 1
        #             else:
        #                 event["reasoning"] = None
        #         else:
        #             event["reasoning"] = None

        # Process events to create messages
        # print(self._events)
        # exit()
        obs_info_included = {f"agent_{ind}": False for ind in range(self.config.num_agents)}
        for idx, event in enumerate(events):
            if event["type"] == "observation":
                message_parts = []

                # if event["step"] == self.current_step:
                #     if self._last_short_term_obs:
                message_parts.append("Current Status:")
                message_parts.append(self._last_short_term_obs)
                message_parts.append("Current Observation:")
                # else:
                #     message_parts.append(f"Observation from {(self.current_step - event['step'])} steps ago:")

                # if event.get("include_text", False):
                message_parts.append(event["text"])
                    
                # image = None
                # if event.get("include_image", False):
                #     image = event["image"]
                #     message_parts.append("Image observation provided.")

                content = "\n".join(message_parts)
                message = Message(role="user", content=content)
                # message = Message(role="user", content=content, attachment=image)

                # Clean up temporary flags
                for flag in ["include_text"]:#, "include_image"]:
                    if flag in event:
                        del event[flag]
                messages.append(message)
                # print(messages)
            elif event["type"] == "dialogue":
                if event.get("proposal") is not None:
                    if event.get("author") == self.agent_name:
                        prop_content = f"Your proposal from round {event['round']}: \n" + event["proposal"]
                    else:
                        prop_content = f"{event['author']} proposal for round {event['round']}: \n" + event["proposal"]
                    message = Message(role="user", content=prop_content)
                    messages.append(message)
                if event.get("thoughts") is not None:
                    if event.get("author") == self.agent_name:
                        thought_content = f"Your thoughts from round {event['round']}: \n" + event["thoughts"]
                    else:
                        thought_content = f"{event['author']}'s thoughts for round {event['round']}: \n" + event["thoughts"]
                    message = Message(role="user", content=thought_content)
                    messages.append(message)
                if event.get("obs_info") is not None and not obs_info_included[event['author']]:
                    if event.get("author") != self.agent_name:
                        obs_content = f"{event['author']}'s local information: \n" + event["obs_info"]
                        message = Message(role="user", content=obs_content)
                        messages.append(message)
                        obs_info_included[event["author"]] = True



                # else:
                #     content = f"Action from {int((len(events) - idx)/2)} steps ago: \n" + event["action"]
                    # message = Message(role="user", content=content)
                    # messages.append(message)

        # print(messages)
        return messages
