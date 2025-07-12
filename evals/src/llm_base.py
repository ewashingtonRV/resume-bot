
from langchain_core.messages import BaseMessage
from typing import List
from langchain_openai import ChatOpenAI
import os
import json

# Azure OpenAI API version
azure_openai_version = "2024-02-15-preview"


class LLMBase:
    """Base class for LLM functionality that can be inherited by other classes."""
    
    def __init__(self, 
                 llm_model_name="gpt-4o",  
                 temperature=0,
                 return_json_bool=True):
        if llm_model_name not in ["gpt-4o", "gpt-4o-mini"]:
            raise ValueError(f"Invalid model name: {llm_model_name}")
        else:
            self.llm_model_name = llm_model_name
        self.temperature = temperature
        self.return_json_bool = return_json_bool
        self.llm = None

    def init_llm(self) -> "BaseLanguageModel":
        """
        Creates an LLM instance using either Azure or OpenAI service.

        Returns:
            BaseLanguageModel: Configured language model
        """
        
        llm = ChatOpenAI(model=self.llm_model_name, 
                            temperature=self.temperature, 
                            seed=42)
        if self.return_json_bool:
            self.llm = llm.bind(response_format={"type": "json_object"})
        else:
            self.llm = llm
    
    @staticmethod
    def convert_string_to_json(string: str) -> dict:
        """
        Converts a string to a JSON object.

        Parameters:
            string (str)

        Returns:
            dict: JSON object.
        """
        try:
            dictionary = json.loads(string)
            return dictionary
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

    def invoke_llm(self, messages: List[BaseMessage]) -> dict:
        """
        Invokes the language model with a list of messages and returns the response content.

        Parameters:
            messages (List[BaseMessage]): List of messages to be passed to the language model.

        Returns:
            str: The response content from the language model.
        """
        if self.llm is None:
            self.init_llm()
        response = self.llm.invoke(messages)    
        return self.convert_string_to_json(response.content) 
