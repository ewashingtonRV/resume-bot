from typing import List, Dict
import pandas as pd
from itertools import product
import random
import os
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore

from .llm_base import LLMBase


class SyntheticQueryGenerator(LLMBase):
    """
    A class for generating synthetic user queries.
    """
    def __init__(self,
                 pinecone_index_name=None,
                 pinecone_namespace=None,
                 llm_model_name="gpt-4o",  
                 temperature=0,
                 return_json_bool=True):
        """
        Initializes the SyntheticQueryGenerator with the given system prompt dictionary.
        """
        super().__init__(llm_model_name, temperature, return_json_bool)
        self.pinecone_namespace = pinecone_namespace
        self.pinecone_index_name = pinecone_index_name
        self.pinecone_client = None

    @staticmethod
    def interpolate_synthetic_query_example(sampled_persona_dict: dict, prompt_example_query_dict: dict):
        """
        Creates a template for the example query prompt.

        Args:
            None

        Returns:
            str: The template for the example query prompt.
        """
        example_query_prompt_template = f"""
            The information about the user asking the question will be in this dictionary format:
            '{{dimension: dimension_value}}' 

            Here are user dimensions and dimension values:
            #############################################
            {sampled_persona_dict}
            #############################################

            Here are example queries for each dimension_value:
            #############################################
            {prompt_example_query_dict}
            #############################################

            Your response should always be in this json format:
            {{
                "response": "The content of the LLM response",
            }}
            """
        return example_query_prompt_template

    def generate_synthetic_user_query_prompt(self, system_prompt_dict: dict) -> str:
        """
        Generates synthetic user queries based on the given system prompt dictionary.

        Args:
            system_prompt_dict (dict): A dictionary containing the system prompt information.

        Returns:
            str: A user system prompt.
        """
        if self.validate_persona_input_data(system_prompt_dict):
            persona_lod = system_prompt_dict["personas"]
            permutation_df = self.create_persona_permutation_df(persona_lod)
            sampled_persona_lod = self.sample_personas(permutation_df, sample_size=20)
            selected_persona_dict = random.choice(sampled_persona_lod)
            print(f"Sampled {selected_persona_dict} persona combinations")
            base_instructions = system_prompt_dict.get("system_prompt", "").strip()
            user_system_prompt = self.create_conversational_user_prompt(
                persona_dict=selected_persona_dict,
                base_instructions=base_instructions
            )
            return user_system_prompt
        else:
            raise ValueError("Invalid input data.")
        
    @staticmethod
    def create_conversational_user_prompt(persona_dict: dict, base_instructions: str) -> str:
        """
        Creates a conversational user prompt by interpolating persona information into base instructions.
        
        Args:
            persona_dict (dict): Dictionary of dimension: dimension_value pairs for this persona
            base_instructions (str): Base conversation instructions
            
        Returns:
            str: Interpolated prompt optimized for dynamic conversations
        """
        # Build persona description
        persona_description = []
        example_questions = [] 
        
        for dimension, dimension_value in persona_dict.items():
            persona_description.append(f"- {dimension.title()}: {dimension_value}")
            
            # Get example queries for this dimension value
            if dimension_value in example_queries_dict:
                examples = example_queries_dict[dimension_value]
                if examples:  # Only add if there are examples
                    example_questions.extend(examples)
        
        persona_text = "\n".join(persona_description)
        
        # Create a clean example section
        if example_questions:
            examples_text = "\n".join([f"- \"{q}\"" for q in example_questions])
            example_section = f"""
    Here are some example questions someone with your profile might ask:
    {examples_text}

    Use these as inspiration for your first question, but don't repeat them exactly."""
        else:
            example_section = ""
        
        # Combine everything into a clean, conversational prompt
        conversational_prompt = f"""{base_instructions}

    YOUR USER PROFILE:
    {persona_text}
    {example_section}

    Your response should always be in this json format:
    {{
        "response": "Your question as the user"
    }}"""
        return conversational_prompt

    @staticmethod
    def create_prompt_example_query_dict(sample_persona_dict: dict, dimension_value_example_query_dict: dict) -> dict:
        """
        Creates a dictionary of prompt-example query pairs based on the given sampled persona dictionary and dimension-value example query dictionary.

        Args:
            sample_persona_dict (dict): A dictionary containing the sampled persona information.
            dimension_value_example_query_dict (dict): A dictionary containing the dimension-value example query information.

        Returns:
            dict: A dictionary of prompt-example query pairs.
        """
        prompt_example_query_dict = {}
        dimension_value_list = list(sample_persona_dict.values())
        for dimension_value in dimension_value_list:
            prompt_example_query_dict[dimension_value] = ' '.join(dimension_value_example_query_dict[dimension_value])
        return prompt_example_query_dict

    @staticmethod
    def validate_persona_input_data(system_prompt_dict):
        """
        Validates the structure and content of the provided data.
        
        Args:
            system_prompt_dict (dict): The data to validate.

        Returns:
            - is_valid (bool): True if the data is valid, False otherwise.
        """
        if not isinstance(system_prompt_dict.get("system_prompt"), str):
            raise ValueError(f"Entry {system_prompt_dict} is missing the 'system_prompt' key.")

        if not isinstance(system_prompt_dict.get("personas"), list):
            raise ValueError(f"Entry {system_prompt_dict} is missing the 'personas' key.")
        required_persona_keys = ["dimension", "dimension_value"]
        persona_lod = system_prompt_dict["personas"]
        for persona_dict in persona_lod:
            if not isinstance(persona_dict, dict):
                raise ValueError(f"Entry {persona_dict} is not a dictionary.")
            for key in required_persona_keys:
                dict_value = persona_dict.get(key, None)
                if dict_value is None:
                    raise ValueError(f"Entry {persona_dict} is missing the required key '{key}'.")
            example_queries = persona_dict.get("example_queries", None)
            if not isinstance(example_queries, list):
                raise ValueError(f"Entry {persona_dict} has 'example_queries' that is not a list.")
        return True

    @staticmethod
    def create_persona_permutation_df(persona_lod: List[dict]) -> pd.DataFrame:
        """
        Create a DataFrame of all possible permutations of the given personas.
        
        Args:
            persona_lod (list): A list of dictionaries representing personas
        
        Returns:
            pd.DataFrame: A DataFrame with all possible permutations of the personas
        """
        dimension_groups = {}
        for entry in persona_lod:
            dimension = entry['dimension']
            dimension_value = entry['dimension_value']
            if dimension not in dimension_groups:
                dimension_groups[dimension] = []
            dimension_groups[dimension].append(dimension_value)
        # Extract dimension names and dimension values
        dimension_names = list(dimension_groups.keys())
        dimension_values = list(dimension_groups.values())
        # Generate all unique permutations using itertools.product
        permutations = list(product(*dimension_values))
        print(f"Created {len(permutations)} permutations")
        permutation_df = pd.DataFrame(permutations, columns=dimension_names)
        return permutation_df

    @staticmethod
    def sample_personas(permutation_df: pd.DataFrame, sample_size: int=20) -> List[dict]:
        """
        Randomly sample rows from the DataFrame without replacement and convert to a list of dictionaries.

        Args:
            df (pd.DataFrame): The input DataFrame.
            sample_size (int): Number of rows to sample.

        Returns:
            list: A list of dictionaries representing the sampled rows.
        """
        if sample_size > len(permutation_df):
            return permutation_df.to_dict(orient="records")
        else:
            sampled_df = permutation_df.sample(n=sample_size, replace=False)
            print(f"Sampled {len(sampled_df)} persona combinations")
            return sampled_df.to_dict(orient="records")
    
    def create_user_response(self, system_prompt: str, messages: list, turn: int=None) -> list:
        """
        Creates a user response based on the given system prompt and conversation history.

        Args:
            system_prompt (str): The system prompt to use for generating user queries.
            messages (list): The current conversation messages.
            sqg (SyntheticQueryGenerator): Instance of the query generator.

        Returns:
            list: Updated messages list with the new user message.
        """
        # Prepare messages for LLM call
        llm_messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history if it exists
        if messages:
            llm_messages.extend(messages)
        
        print("Generating user query...")
        response_dict = self.invoke_llm(llm_messages)
        user_response = response_dict["response"]
        # Add the new user message to the conversation
        if turn is None:
            new_message = HumanMessage(content=user_response)
        else:
            new_message = HumanMessage(content=user_response, additional_kwargs={"turn": turn})
        messages.append(new_message)
        return messages

    def create_ai_response(self, system_prompt: str, messages: list) -> list:
        """
        Creates an AI response based on the given system prompt and conversation history.

        Args:
            system_prompt (str): The system prompt to use for generating AI responses.
            messages (list): The current conversation messages.
            sqg (SyntheticQueryGenerator): Instance of the query generator.

        Returns:
            list: Updated messages list with the new AI message.
        """
        if len(messages) == 0:
            raise Exception("No messages provided. AI needs at least one user message to respond to.")
        
        # Prepare messages for LLM call
        llm_messages = [SystemMessage(content=system_prompt)]
        llm_messages.extend(messages)
        
        print("Generating AI response...")
        response_dict = self.invoke_llm(llm_messages)
        ai_response = response_dict["answer"]
        
        # Add the new AI message to the conversation
        new_message = AIMessage(content=ai_response)
        messages.append(new_message)
        return messages

    def run_bot_conversation_with_graph(self, user_system_prompt: str, graph, max_turns: int = 5):
        """
        Runs a conversation between a user bot and the graph-based AI.
        
        Args:
            user_system_prompt (str): System prompt for the user bot.
            graph: The LangGraph graph to use for AI responses.
            max_turns (int): Maximum number of conversation turns.
        """
        import uuid
        
        messages = []
        
        print("=" * 50)
        print("Starting Bot Conversation with Graph")
        print("=" * 50)
        
        for turn in range(max_turns):
            print(f"\n--- Turn {turn + 1} ---")
            
            # User bot generates a query
            try:
                messages = self.create_user_response(user_system_prompt, messages)
                print(f"USER: {messages[-1].content}")
            except Exception as e:
                print(f"Error generating user response: {e}")
                break
            
            # Graph responds to the conversation
            try:
                # Create state for graph invocation
                graph_state = {
                    "messages": messages.copy(),
                    "next": "intent_classification",
                    "intent": None,
                    "context": None,
                    "answer": None
                }
                
                # Create config with thread_id
                config = {
                    "configurable": {
                        "thread_id": str(uuid.uuid4())
                    }
                }
                
                # Invoke the graph
                result = graph.invoke(graph_state, config=config)
                
                # Update messages with the graph result
                messages = result["messages"]
                print(f"AI: {messages[-1].content}")
                
            except Exception as e:
                print(f"Error generating graph response: {e}")
                break
        
        print("\n" + "=" * 50)
        print("Conversation Complete")
        print("=" * 50)
        return messages

    def create_synthetic_trace(self, system_prompt_dict: dict, graph, max_turns: int = 5):
        """
        Creates a synthetic trace of a conversation between a user and an AI.

        Args:
            system_prompt_dict (dict): Dictionary containing system prompts
            graph: The graph object to use for conversation
            max_turns (int): The maximum number of turns in the conversation

        Returns:
            List[Message]: The conversation messages
        """
        user_system_prompt = self.generate_synthetic_user_query_prompt(system_prompt_dict)
        return self.run_bot_conversation_with_graph(user_system_prompt, graph, max_turns)

    @staticmethod
    def create_simple_synthetic_rag_query_prompt(text_chunk: str) -> str:
        """
        Creates a synthetic RAG query prompt for a given evaluation name and system prompt.

        Args:
            text_chunk (str): The text chunk to be used as the query

        Returns:
            str: The synthetic RAG query prompt
        """
        return f"""You are a helpful assistant generating synthetic QA pairs for 
        retrieval evaluation. 
        Given a chunk of text, extract a specific, self-contained
        fact from it. Then write a question that is directly and
        unambiguously answered by that fact alone. Return your output
        in the following JSON format:
        {{ "fact": "...", "question": "..." }}
        Chunk of text:
        #########################################################
        {text_chunk}
        #########################################################
        """

    @staticmethod
    def create_complex_synthetic_rag_query_prompt(target_text_chunk: str, non_target_text_chunks: List[str]) -> str:
        """
        Creates a synthetic RAG query prompt that generates a question specifically answerable by the target chunk,
        while incorporating terminology from non-target chunks.

        Args:
            target_text_chunk (str): The text chunk that should contain the answer
            non_target_text_chunks (List[str]): Other text chunks to draw terminology from

        Returns:
            str: The synthetic RAG query prompt
        """
        non_target_chunks_str = "\n".join(non_target_text_chunks)
        return f"""You are a helpful assistant generating synthetic QA pairs for 
        retrieval evaluation.

        You will be given two inputs:
        1. A target text chunk that contains a specific fact we want to test
        2. Other related text chunks that contain similar terminology and themes

        Your task is to:
        1. Extract a specific, self-contained fact from the target chunk
        2. Write a question that:
           - Is directly and unambiguously answered ONLY by the target chunk
           - Reuses terminology or themes from the other chunks where possible
           - Cannot be answered using only the other chunks

        Return your output in the following JSON format:
        {{ "fact": "...", "question": "..." }}

        Target text chunk:
        #########################################################
        {target_text_chunk}
        #########################################################

        Other related chunks:
        #########################################################
        {non_target_chunks_str}
        #########################################################
        """
    
    def generate_synthetic_rag_query(self, text_chunk: str) -> str:
        """
        Generates a synthetic RAG query for a given text chunk.

        Args:
            text_chunk (str): The text chunk to be used as the query
        """
        if self.pinecone_namespace is not None:
            non_target_text_chunk_list = self.retrieve_decoy_chunks(text_chunk, k=3)
            prompt = self.create_complex_synthetic_rag_query_prompt(target_text_chunk=text_chunk,
                                                                    non_target_text_chunks=non_target_text_chunk_list)
        else:
            prompt = self.create_simple_synthetic_rag_query_prompt(text_chunk)
        messages = [SystemMessage(content=prompt)]  
        response = self.invoke_llm(messages)
        return response
    
    def validate_synthetic_rag_query_response(self, response: dict) -> bool:
        """
        Validates the synthetic RAG query response.

        Args:
            response (dict): The response from the synthetic RAG query

        Returns:
            bool: True if the response is valid, False otherwise
        """
        if not isinstance(response, dict):
            raise Exception("Expected answer to be a dictionary")
        if "fact" not in response.keys():
            raise Exception("Expected answer to have 'fact' key")
        if "question" not in response.keys():
            raise Exception("Expected answer to have 'question' key")
        return True
    
    def generate_simple_synthetic_rag_queries(self, guideline_lod: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Generates synthetic RAG queries for a given list of guideline dictionaries.

        Args:
            guideline_lod (List[Dict[str, str]]): The list of guideline dictionaries

        Returns:
            List[Dict[str, str]]: The list of guideline dictionaries with 'fact' and 'question' keys
        """
        for guideline_dict in guideline_lod:
            text_chunk = guideline_dict.get("input_text", None)
            if text_chunk is not None:
                fact_query_dict = self.generate_synthetic_rag_query(text_chunk)
                if self.validate_synthetic_rag_query_response(fact_query_dict):
                    guideline_dict["fact"] = fact_query_dict.get("fact", None)
                    guideline_dict["question"] = fact_query_dict.get("question", None)
                else:
                    fact_query_dict = self.generate_synthetic_rag_query(text_chunk)
                    try:
                        guideline_dict["fact"] = fact_query_dict.get("fact", None)
                        guideline_dict["question"] = fact_query_dict.get("question", None) 
                    except:
                        pass
        return guideline_lod
    
    def init_pinecone(self):  
        """
        Initializes a Pinecone index.
        """
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        pinecone_client = PineconeVectorStore(
            index_name=self.pinecone_index_name,
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            embedding=embeddings,
            namespace=self.pinecone_namespace
        )
        self.pinecone_client = pinecone_client
    
    def similarity_search(self, query: str, k: int=10) -> List[str]:
        """
        Retrieves documents from Pinecone.

        Args:
            query (str): The query to retrieve documents for
            k (int): Number of documents to retrieve
        """
        if self.pinecone_client is None:
            self.init_pinecone()
        return self.pinecone_client.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int=10) -> List[tuple]:
        """
        Retrieves documents from Pinecone with similarity scores.

        Args:
            query (str): The query to retrieve documents for
            k (int): Number of documents to retrieve
            
        Returns:
            List[tuple]: List of (document, score) tuples where score is the similarity score
        """
        if self.pinecone_client is None:
            self.init_pinecone()
        return self.pinecone_client.similarity_search_with_score(query, k=k)
    
    def retrieve_decoy_chunks(self, query: str, k) -> list:
        """
        Retrieves documents from Pinecone with similarity scores and returns the retrieved text

        Args:
            query (str): The query to retrieve documents for
            k (int): Number of documents to retrieve
            
        Returns:
            list: List of document texts, excluding the first result
        """
        documents = self.similarity_search(query, k=k)
        page_contents = [doc.page_content for doc in documents]
        return page_contents[1:]
    