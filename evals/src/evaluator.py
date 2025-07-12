from typing import List, Dict
import os
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI, OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore

# Azure OpenAI API version
from .llm_base import LLMBase


class Evaluator(LLMBase):
    def __init__(self,
                 llm_model_name="gpt-4o",  
                 llm_service="azure", 
                 temperature=0,
                 return_json_bool=True,
                 pinecone_index_name="rvo-index",):
        super().__init__(llm_model_name, temperature, return_json_bool)
        self.pinecone_index_name = pinecone_index_name
        self.pinecone_client = None

    @staticmethod
    def compile_messages(df: pd.DataFrame, system_prompt: str=None, inference_message_bool: bool=False) -> List[dict]:
        """
        Compiles messages from a DataFrame into a list of dictionaries.

        Parameters:
            df (pandas.DataFrame): DataFrame containing messages.
            system_prompt (str): System prompt to be used as the first message in each conversation.
            inference_message_bool (bool): Whether to include the last AIMessage in the list.

        Returns:
            list: List of dictionaries containing messages.
        """
        if inference_message_bool is False and system_prompt is None:
            raise ValueError("system_prompt must be provided if inference_message_bool is False.")
        message_lod = []
        for index, row_i in df.iterrows():
            conversation_id = row_i["conversation_id"]
            message_idx = row_i["message_idx"]
            row_id = conversation_id + "_" + str(message_idx)
            
            # Get all messages up to and including the current message, in chronological order
            conversation_df = df[(df["conversation_id"] == conversation_id) & (df["message_idx"] <= message_idx)].sort_values(by=["message_idx"], ascending=[True])
            
            # Build messages in chronological order
            chronological_messages = []
            for _, row_j in conversation_df.iterrows():
                # Add HumanMessage first (the question)
                chronological_messages.append(HumanMessage(content=row_j['human_message']))
                # Add AIMessage second (the response) - but only if we're not excluding it
                if not (inference_message_bool and row_j["message_idx"] == message_idx):
                    chronological_messages.append(AIMessage(content=row_j['ai_message']))
            
            # If this is for inference, remove the last AIMessage (the one we want to predict)
            if inference_message_bool:
                # Remove the last AIMessage if it exists
                if chronological_messages and isinstance(chronological_messages[-1], AIMessage):
                    chronological_messages.pop()
            
            # Construct final message list
            if inference_message_bool:
                all_messages = chronological_messages
            else:
                all_messages = [SystemMessage(content=system_prompt)] + chronological_messages
                
            messages_dict = {"chat_id": row_id, "messages": all_messages}
            message_lod.append(messages_dict)
        return message_lod
    
    @staticmethod
    def validate_input_df_schema(df: pd.DataFrame) -> bool:
        """Validates the input DataFrame schema.

        Parameters:
            df (pandas.DataFrame): DataFrame to be validated.
        
        Returns:
            bool: True if the schema is valid, False otherwise.
        """
        expected_col_names = ['ai_message', 'chat_id', 'conversation_id', 'human_message', 'message_idx']
        input_col_names = set(df.columns) 
        if not set(expected_col_names).issubset(input_col_names):
            missing_cols = list(set(expected_col_names) - input_col_names)
            raise Exception(f"Missing expected columns: {missing_cols}")
        else:
            return True
        
    @staticmethod
    def validate_unique_id_per_row(df: pd.DataFrame) -> bool:
        """Validates that each row in the DataFrame has a unique ID.

        Parameters:
            df (pandas.DataFrame): DataFrame to be validated.
        
        Returns:
            bool: True if each row has a unique ID, False otherwise.
        """
        unique_id_count = df["chat_id"].unique().shape[0]
        if unique_id_count != df.shape[0]:
            raise Exception(f"Expected unique ids per row, got {unique_id_count})")
        else:
            return True
        
    def validate_input_df(self, df: pd.DataFrame) -> bool:
        """
        Validates the input DataFrame for the auto_annotate method.

        Parameters:
            df (pandas.DataFrame): DataFrame to be validated.
        
        Returns:
            bool: True if the DataFrame is valid, False otherwise.
        """
        if self.validate_input_df_schema(df) and self.validate_unique_id_per_row(df):
            return True
        else:
            return False
        
    @staticmethod
    def validate_prompt_string(text: str) -> bool:
        """
        Validates if a string contains the required JSON structure with 'answer' and 'reason' keys.

        Args:
            text (str): The input string

        Returns:
            bool: True if the string contains the required JSON structure, False otherwise
        """
        # Check for "answer" key with single or double quotes followed by colon
        answer_pattern = r'["\']answer["\']:\s*'
        # Check for "reason" key with single or double quotes followed by colon
        reason_pattern = r'["\']reason["\']:\s*'
        # Check for answer value containing "pass" or "fail"
        answer_value_pattern = r'["\']answer["\']:\s*["\'][^"\']*(?:pass|fail)[^"\']*["\']'
        # Check for the word "json" in the text
        json_pattern = r'json'
        # Check if all patterns are present
        has_answer_key = bool(re.search(answer_pattern, text, re.IGNORECASE))
        has_reason_key = bool(re.search(reason_pattern, text, re.IGNORECASE))
        has_valid_answer_value = bool(re.search(answer_value_pattern, text, re.IGNORECASE))
        has_json_word = bool(re.search(json_pattern, text, re.IGNORECASE))
        return has_answer_key and has_reason_key and has_valid_answer_value and has_json_word

    @staticmethod
    def validate_prompt_type(prompt_type: str) -> bool:
        """
        Validates that the prompt type is either 'eval' or 'persona'.

        Args:
            prompt_type (str): The prompt type to validate.

        Returns:
            bool: True if the prompt type is valid, False otherwise.
        """
        if prompt_type in ["eval", "persona"]:
            return True
        else:
            raise Exception(f"Invalid prompt type: {prompt_type}. Expected 'eval' or 'persona'.")
    
    @staticmethod
    def validate_persona_input_data(system_prompt_dict: dict) -> bool:
        """
        Validates that each dictionary in the list has the keys 'dimension' and 'dimension_values'.
        
        Args:
            system_prompt_dict (dict): List of dictionaries to validate.
            
        Returns:
            - is_valid (bool): True if all dictionaries are valid, False otherwise.
        """
        persona_lod = system_prompt_dict.get("personas", None)
        if persona_lod is None:
            raise ValueError(f"Entry {system_prompt_dict} is missing the 'personas' key.")
        for persona_dict in persona_lod:
            if not isinstance(persona_dict, dict):
                raise ValueError(f"Entry {persona_dict} is not a dictionary.")
            if "dimension" not in persona_dict.keys():
                raise ValueError(f"Entry {persona_dict} is missing the 'dimension' key.")
            if "dimension_values" not in persona_dict.keys():
                raise ValueError(f"Entry {persona_dict} is missing the 'dimension_values' key.")
        return True

    def validate_prompts(self, prompts: List[dict]) -> bool:
        """
        Validates the prompts list for the auto_annotate method.

        Parameters: 
            prompts (list): List of dictionaries containing prompts.

        Returns:
            bool: True if all prompts are valid, False otherwise.
        """
        if not isinstance(prompts, list):
            raise Exception("prompts must be a list.")
        
        for prompt_dict in prompts:
            if not isinstance(prompt_dict, dict):
                raise Exception("Each prompt must be a dictionary.")
            prompt_type = prompt_dict.get("prompt_type", "eval")
            # Validate prompt_type
            if not self.validate_prompt_type(prompt_type):
                raise Exception("Invalid prompt_type.")
            if prompt_type == "persona":
                if not self.validate_persona_input_data(prompt_dict):
                    raise Exception("Invalid persona input data.")
            if prompt_type == "eval":
                eval_system_prompt = prompt_dict.get("eval_system_prompt", None)
                if not self.validate_prompt_string(eval_system_prompt):
                    raise Exception("Invalid eval_system_prompt: Expected output format: {'answer': 'pass' or 'fail', 'reason': 'reason for your decision'}")
            return True

    def auto_annotate(self, df: pd.DataFrame, eval_system_prompt: str, eval_name: str) -> pd.DataFrame:
        """
        Automatically annotate a DataFrame with annotations based on the provided system prompt.

        Parameters:
            df (pandas.DataFrame): DataFrame containing messages.
            eval_system_prompt (str): System prompt to be used as the first message in each conversation
            eval_name (str): Name of the evaluation to be used for the annotations.

        Returns:
            list: List of dictionaries containing messages and annotations.
        """
        if self.validate_input_df(df):
            message_lod = self.compile_messages(df, eval_system_prompt)
            eval_lod = []
            for i, message_dict in enumerate(message_lod):
                if i % 100 == 0:
                    print(f"{eval_name} percent complete: " + str(round(i / len(message_lod), 2) * 100) + "%")
                messages = message_dict["messages"]
                response_dict = self.invoke_llm(messages)
                if not isinstance(response_dict, dict):
                    print(f"Expected answer to be a dictionary, trying again")
                    response_dict = self.invoke_llm(messages)
                if isinstance(response_dict, dict):
                    response_dict["chat_id"] = message_dict["chat_id"]
                    eval_lod.append(response_dict)
                else:
                    output_type = type(response_dict)
                    raise Exception(f"Expected answer to be a dictionary, got {output_type}")
            eval_df = pd.DataFrame(eval_lod)
            eval_df["eval_name"] = eval_name
            if self.validate_unique_id_per_row(eval_df):
                print(f"{eval_name} percent complete: 100%")
                return pd.merge(df, eval_df, on="chat_id", how="inner")
            else:
                raise Exception(f"Expected unique ids per row, got {len(eval_df['chat_id'].unique())}")
        else:
            raise Exception(f"Invalid input DataFrame")

    def create_eval_df(self, df: pd.DataFrame, eval_system_prompts: List[dict]) -> pd.DataFrame:
        """
        Creates an evaluation DataFrame from a DataFrame and a list of system prompts.

        Parameters:
            df (pandas.DataFrame): DataFrame containing messages.
            eval_system_prompts (List[dict]): List of dictionaries containing system prompts and their names.

        Returns:
            pandas.DataFrame: DataFrame containing messages and annotations.
        """
        evals_df = pd.DataFrame()
        if self.validate_input_df(df):
            for eval_system_prompt_dict in eval_system_prompts:
                assert sorted(list(eval_system_prompt_dict.keys())) == ["eval_name", "eval_system_prompt"], "Expected eval_system_prompt and eval_name keys"
                eval_name = eval_system_prompt_dict["eval_name"]
                print(f"Evaluating: {eval_name}")
                eval_df_i = self.auto_annotate(df, eval_system_prompt_dict["eval_system_prompt"], eval_name)
                evals_df = pd.concat([evals_df, eval_df_i], axis=0)
            return evals_df
        else:
            raise Exception(f"Invalid input DataFrame")

    def evaluate(self, df: pd.DataFrame, eval_system_prompts: List[dict], answers: List[dict]=None) -> pd.DataFrame:
        """
        Evaluates a DataFrame using a list of system prompts.

        Parameters:
            df (pandas.DataFrame): DataFrame containing messages.
            eval_system_prompts (List[dict]): List of dictionaries containing system prompts and their names.
            answers (List[dict]): List of dictionaries containing answers and their names. Defaults to None.
        
        Returns:
            pandas.DataFrame: DataFrame containing messages and annotations.
        """
        df_lod = [{"df": df, "df_name": "baseline"}]
        # Validate prompts before any processing to fail fast
        self.validate_prompts(eval_system_prompts)
        eval_df = pd.DataFrame()
        if answers is not None:
            inference_df = self.create_inference_df(df, answers)
            df_lod.append({"df": inference_df, "df_name": "inference"})
        for df_dict in df_lod:
            input_df = df_dict["df"]
            eval_df_i = self.create_eval_df(input_df, eval_system_prompts)
            eval_df_i["data_source"] = df_dict["df_name"]
            eval_df = pd.concat([eval_df, eval_df_i], axis=0)
        return eval_df

    @staticmethod
    def create_inference_answer_map(answers: List[dict]) -> dict:
        """
        Create a map of chat_id to answer.
        
        Args:
            answers (List[dict]) - A list of dictionaries containing the chat_id and answer.

        Returns:
            dict - A dictionary mapping chat_id to answer.
        """
        inference_answer_map = {}
        for inference_message_dict in answers:
            chat_id = inference_message_dict.get("chat_id", None)
            answer = inference_message_dict.get("answer", None)
            if chat_id is not None:
                inference_answer_map[chat_id] = answer
        return inference_answer_map

    def create_inference_df(self, df: pd.DataFrame, answers: List[dict]) -> pd.DataFrame:
        """
        Create a DataFrame with the AI answers.
        
        Args:
            df (pd.DataFrame) - The input DataFrame.
            answers (List[dict]) - A list of dictionaries containing the chat_id and answer.
        
        Returns:
            pd.DataFrame - The input DataFrame with an additional column containing the AI answers.
        """
        if self.validate_input_df(df):
            inference_answer_map = self.create_inference_answer_map(answers)
            df["ai_message"] = df["chat_id"].map(inference_answer_map)
            assert df["ai_message"].isna().sum() == 0, "Missing answers for some chat_ids."
            return df
        else:
            raise ValueError("Invalid input DataFrame.")
    
    @staticmethod
    def calculate_cohens_kappa(judge_1: list, judge_2: list) -> float:
        """
        Calculate Cohen's Kappa for two annotators

        Arg:
            judge_1 (list): List of annotation results from judge 1
            judge_2 (list): List of annotation results from judge 2

        Returns:
            float: Cohen's Kappa value
        """
        # Ensure the annotation lists have the same length
        assert len(judge_1) == len(judge_2), "Annotation lists must have the same length."
        n_items = len(judge_1)
        if n_items == 0:
            return 1.0  # Convention: return 1 if nothing to disagree on
        # Observed agreement
        observed_agreement = sum(1 for i in range(n_items) if judge_1[i] == judge_2[i])
        po = observed_agreement / n_items
        # Expected agreement
        labels = set(judge_1) | set(judge_2)
        count1 = Counter(judge_1)
        count2 = Counter(judge_2)
        pe = sum((count1.get(label, 0) / n_items) * (count2.get(label, 0) / n_items) for label in labels)
        # Handle edge case
        if pe == 1.0:
            return 1.0 if po == 1.0 else 0.0
        else:
            return (po - pe) / (1 - pe)
        
    @staticmethod
    def format_data_for_fleiss_kappa(eval_df: pd.DataFrame) -> np.ndarray:
        """
        Format data for Fleiss' Kappa calculation.
        
        Args:
            - eval_df: A pandas DataFrame with columns 'judge' and 'pass'.
        
        Returns:
            - ratings_array: A 2D numpy array of shape (n_items, n_categories)
        """
        evaluator_names = eval_df["judge"].unique().tolist()
        evaluator_result_lol = []
        for evaluator_name in evaluator_names:
            evaluator_result_list_i = eval_df[eval_df["judge"] == evaluator_name]["answer"].tolist()
            evaluator_result_lol.append(evaluator_result_list_i)
        ratings_per_item = list(zip(*evaluator_result_lol))
        # Extract unique categories dynamically from all raters
        unique_categories = sorted(set(val for sublist in evaluator_result_lol for val in sublist))

        # Prepare the 2D ratings array
        ratings_array = np.zeros((len(ratings_per_item), len(unique_categories)), dtype=int)

        # Populate the array by counting ratings for each item
        for i, ratings in enumerate(ratings_per_item):
            counts = Counter(ratings)  # Count occurrences of each category for the item
            for j, category in enumerate(unique_categories):
                ratings_array[i, j] = counts.get(category, 0)  # Fill in counts for the category
        return ratings_array

    @staticmethod
    def calculate_fleiss_kappa(ratings: np.ndarray) -> float:
        """
        Calculate Fleiss' Kappa for multiple raters.
        
        Args:
            - ratings: A 2D numpy array of shape (n_items, n_categories), where:
                - n_items is the number of items being rated.
                - n_categories is the number of possible categories.
                Each entry in the array represents the count of raters assigning that category to the item.
            
        Returns:
            - kappa: Fleiss' Kappa score (float).
        """
        n_items, n_categories = ratings.shape
        n_raters = np.sum(ratings[0])  # Total number of raters per item (assumed constant)
        # Calculate the proportion of all assignments to a particular category
        p = np.sum(ratings, axis=0) / (n_items * n_raters)
        # Calculate the agreement for each item
        P = (np.sum(ratings ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
        # Mean agreement over all items
        P_bar = np.mean(P)
        # Expected agreement (by chance)
        P_e_bar = np.sum(p ** 2)
        # Fleiss' Kappa
        if P_e_bar == 1:  # Handle edge case where expected agreement is perfect
            return 1.0
        kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
        return kappa
    
    def print_iaa_interpretation_guidlines(self):
        """
        Prints the IAA guidelines.
        """
        print("""
        • < 0: Poor
        • 0.00−0.20: Slight
        • 0.21−0.40: Fair
        • 0.41−0.60: Moderate
        • 0.61−0.80: Substantial 
        • 0.81−1.00: Almost perfect
        
        In practice, we aim for κ ≥ 0.6 to ensure labeling reliability.""")
    
    def calculate_iaa(self, eval_df: pd.DataFrame) -> float:
        """
        Calculates the inter-annotator agreement (IAA) for a given evaluation DataFrame.

        Args:
            eval_df (pd.DataFrame): The evaluation DataFrame containing 'judge' and 'answer' columns.

        Returns:
            float: The IAA value.
        """
        assert "judge" in eval_df.columns and eval_df["judge"].dtype == "object", "Column 'judge' must be present and of type str"
        assert "answer" in eval_df.columns and eval_df["answer"].dtype == "int", "Column 'answer' must be present and of type int"
        eval_df = eval_df.copy()
        eval_df.sort_values(["conversation_id", "message_idx", "judge"], inplace=True)
        evaluator_names = eval_df["judge"].unique().tolist()
        evaluator_n = len(evaluator_names)
        if evaluator_n == 1:
            return 1.0
        elif evaluator_n == 2:
            print(f"Calculating Cohen's Kappa for {evaluator_names[0]} and {evaluator_names[1]}")
            judge_1_result_list = eval_df[eval_df["judge"] == evaluator_names[0]]["answer"].tolist()
            judge_2_result_list = eval_df[eval_df["judge"] == evaluator_names[1]]["answer"].tolist()
            return self.calculate_cohens_kappa(judge_1_result_list, judge_2_result_list)
        else:
            print(f"Calculating Fleiss' Kappa for {evaluator_names}")
            ratings = self.format_data_for_fleiss_kappa(eval_df)
            return self.calculate_fleiss_kappa(ratings)
        
    def calculate_iaa_by_eval_name(self, eval_df: pd.DataFrame) -> dict:
        """
        Calculates the IAA for each evaluation name.

        Args:
            eval_df (pd.DataFrame): The evaluation DataFrame containing 'eval_name' and 'answer' columns.

        Returns:
            dict: A dictionary of IAA values for each evaluation name.
        """
        eval_names = eval_df["eval_name"].unique().tolist()
        iaa_dict = {}
        for eval_name in eval_names:    
            eval_df_i = eval_df[eval_df["eval_name"] == eval_name]
            iaa_dict[eval_name] = self.calculate_iaa(eval_df_i)
        self.print_iaa_interpretation_guidlines()
        return iaa_dict
    
    @staticmethod
    def calculate_agreement_percentage(list1: list, list2: list) -> float:
        """
        Calculates the agreement percentage between two lists.

        Args:
            list1 (list): The first list.
            list2 (list): The second list.
        
        Returns:
            float: The agreement percentage.
        """
        # Ensure both lists are of the same length
        assert len(list1) == len(list2), "Both lists must have the same length"
        # Count the number of matches (elements that are the same)
        matches = sum(1 for a, b in zip(list1, list2) if a == b)
        # Calculate the percentage of agreement
        total = len(list1)
        percentage = round((matches / total), 2)
        return percentage

    @staticmethod
    def map_eval_score(df: pd.DataFrame) -> list:
        """
        Maps the evaluation score to a numerical value.

        Args:
            df (pd.DataFrame): The input DataFrame
        
        Returns:
            list: A list of mapped evaluation scores
        """
        eval_map = {"pass": 1, "fail": 0}
        return df["answer"].map(eval_map)
        
    def create_graph_df(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a DataFrame for graphing the evaluation results.

        Args:
            df (pd.DataFrame): The input DataFrame

        Returns:
            pd.DataFrame: The graph DataFrame
        """
        graph_df = eval_df.copy()
        graph_df["eval_score"] = self.map_eval_score(graph_df)
        agg_graph_df = graph_df.groupby(["eval_name", "data_source"])["eval_score"].sum().reset_index().rename(columns={"eval_score": "pass_count"})
        agg_graph_df["pass_pct"] = np.round(agg_graph_df["pass_count"] / 73, 2) * 100
        return agg_graph_df

    @staticmethod
    def create_bar_graph(graph_df: pd.DataFrame):
        """
        Creates a bar graph of the evaluation results.

        Args:
            graph_df (pandas.DataFrame): DataFrame containing evaluation results

        Returns:
            None
        """
        ax = sns.barplot(
                data=graph_df,
                x="eval_name",
                y="pass_pct",
                hue="data_source",
                edgecolor="k"
            )
        plt.xticks(rotation=45)
        ax.set_xlabel("Evaluation Name", fontsize=12)  
        ax.set_ylabel("Pass Percentage (%)", fontsize=12)  
        if graph_df["data_source"].unique().shape[0] > 1:
            ax.legend(title="Eval Data", title_fontsize=10)
        else:
            ax.legend_.remove()
        # Display the plot
        plt.show()

    def analyze(self, eval_df: pd.DataFrame):
        """
        Analyzes the evaluation results.

        Args:
            eval_df (pd.DataFrame): The input DataFrame

        Returns:
            None
        """
        graph_df = self.create_graph_df(eval_df)
        self.create_bar_graph(graph_df)

    @staticmethod
    def create_failure_funnel_df(failure_funnel_dict: dict) -> pd.DataFrame:
        """
        Creates a failure funnel DataFrame from a dictionary of failure stages.

        Args:
            failure_funnel_dict (dict): A dictionary mapping conversation IDs to failure stages.
        """
        return pd.DataFrame(
            list(failure_funnel_dict.items()), columns=["conversation_id", "fail_stage"])
    
    def create_failure_funnel_transition_df(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a failure funnel transition DataFrame from a given evaluation DataFrame.

        Args:
            eval_df (pd.DataFrame): The input DataFrame
        """
        baseline_eval_df = eval_df[eval_df["data_source"] == "baseline"].copy()
        inference_eval_df = eval_df[eval_df["data_source"] != "baseline"].copy()
        baseline_failure_funnel_dict = self.spoof_min_fail_dict(baseline_eval_df)
        inference_failure_funnel_dict = self.spoof_min_fail_dict(inference_eval_df)
        baseline_failure_funnel_df = self.create_failure_funnel_df(baseline_failure_funnel_dict)
        baseline_failure_funnel_df.rename(columns={"fail_stage": "baseline_fail_stage"}, inplace=True)
        inference_failure_funnel_df = self.create_failure_funnel_df(inference_failure_funnel_dict)
        inference_failure_funnel_df.rename(columns={"fail_stage": "inference_fail_stage"}, inplace=True)
        failure_funnel_transition_df = pd.merge(baseline_failure_funnel_df, inference_failure_funnel_df, on=["conversation_id"], how="inner")
        return failure_funnel_transition_df

    @staticmethod
    def create_transition_matrix_df(failure_funnel_transition_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a transition matrix DataFrame from a failure funnel transition DataFrame.

        Args:
            failure_funnel_transition_df (pd.DataFrame): The input DataFrame
        """
        failure_tuples = list(zip(
            failure_funnel_transition_df['baseline_fail_stage'], 
            failure_funnel_transition_df['inference_fail_stage']
            ))
        # Initialize a 4x4 matrix with zeros
        matrix = [[0 for _ in range(4)] for _ in range(4)]
        # Populate the matrix
        for baseline, inference in failure_tuples:
            matrix[inference - 1][baseline - 1] += 1
        data = np.array(matrix)
        baseline_stages = ['Baseline 1', 'Baseline 2', 'Baseline 3', 'Baseline Pass']
        inference_stages = ['Inference 1', 'Inference 2', 'Inference 3', 'Inference Pass']
        transition_graph_df = pd.DataFrame(data, columns=baseline_stages, index=inference_stages)
        return transition_graph_df

    def create_transition_graph(self, eval_df):
        """
        Creates a transition graph from a given evaluation DataFrame.

        Args:
            eval_df (pd.DataFrame): The input DataFrame

        Returns:
            None
        """
        failure_funnel_transition_df = self.create_failure_funnel_transition_df(eval_df)
        transition_graph_df = self.create_transition_matrix_df(failure_funnel_transition_df)
        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(transition_graph_df, annot=True, fmt="d", cmap="Blues", cbar=True)

        # Add labels and title
        plt.title("Baseline to Inference Transition Matrix")
        plt.xlabel("Baseline Fail Stage")
        plt.ylabel("Inference Fail Stage")
        # Show the plot
        plt.show()

    @staticmethod
    def extract_failure_reasons(eval_df: pd.DataFrame, k: int = 30) -> list:
        """
        Extracts the failure answers from a given evaluation DataFrame.

        Args:
            eval_df (pd.DataFrame): The input DataFrame
            k (int): The number of failure reasons to extract

        Returns:
            list: A list of failure reasons
        """
        return eval_df.sample(k)[eval_df["answer"] == "fail"]["reason"].tolist()
    
    def identify_failure_modes(self, eval_df: pd.DataFrame) -> dict:
        """
        Identifies the failure modes from a list of failure answers.

        Args:
            failure_answers (list): A list of failure answers.
        """
        failure_answers = self.extract_failure_reasons(eval_df)
        failure_mode_categorizer_prompt = self.create_failure_mode_categorizer_prompt(failure_answers)
        messages = [SystemMessage(content=failure_mode_categorizer_prompt)]
        failure_mode_categorizer_answer = self.invoke_llm(messages)    
        return failure_mode_categorizer_answer

    @staticmethod
    def create_failure_mode_categorizer_prompt(failure_answers: list) -> dict:
        """
        Identifies the failure modes from a list of failure answers.

        Args:
            failure_answers (list): A list of failure answers.

        Returns:
            dict: A dictionary of failure modes.
        """
        failure_mode_categorizer_prompt = f"""
        You are an expert at clustering information into sensible groups. You will be given a list of sentences consiting of feedback that descirbe issues with a answer that was evaluated poor. You will not be given the answers that the feedback refers to. 

        Your job is to create 3-5 groups that categorize the feedback. Each group name should not be longer than 10 words. For each group, please provide 2-3 examples of feedback that fit into the group. 

        Your output should be in json format. The json should follow this format:
            {{
                "group_name_1": [
                    "example_feedback_1", "example_feedback_2","example_feedback_3"
                    ],
                "group_name_2": [
                    "example_feedback_1", "example_feedback_2","example_feedback_3"
                ],
                ...
        }}

        BEGIN FEEDBACK DATA TO EVALUATE
        ###############################
        {failure_answers}
        ###############################
        """
        return failure_mode_categorizer_prompt

    @staticmethod
    def calibrate_success_rate(test_labels, test_preds, unlabeled_preds, B=20000):
        """
        Estimates the true success rate and calculates a 95% bootstrap confidence interval.

        Args:
            test_labels: array-like of 0/1, human labels on test set (1 = Pass).
            test_preds: array-like of 0/1, judge predictions on test set (1 = Pass).
            unlabeled_preds: array-like of 0/1, judge predictions on unlabeled data (1 = Pass).
            B: number of bootstrap iterations.

        Returns:
            tuple:
                theta_hat: point estimate of true success rate.
                L: lower bound of the 95% bootstrap confidence interval.
                U: upper bound of the 95% bootstrap confidence interval.
        """
        # Convert inputs to NumPy arrays
        test_labels = np.asarray(test_labels, dtype=int)
        test_preds = np.asarray(test_preds, dtype=int)
        unlabeled_preds = np.asarray(unlabeled_preds, dtype=int)

        # Step 1: Judge accuracy on test set
        P = test_labels.sum()  # Number of positive labels
        F = len(test_labels) - P  # Number of negative labels
        TPR = ((test_labels == 1) & (test_preds == 1)).sum() / P  # True Positive Rate
        TNR = ((test_labels == 0) & (test_preds == 0)).sum() / F  # True Negative Rate

        # Step 2: Raw observed success rate
        p_obs = unlabeled_preds.sum() / len(unlabeled_preds)

        # Step 3: Correct estimate
        denom = TPR + TNR - 1
        if denom <= 0:
            raise ValueError("Judge accuracy too low for correction")
        theta_hat = (p_obs + TNR - 1) / denom
        theta_hat = np.clip(theta_hat, 0, 1)  # Ensure theta_hat is between 0 and 1

        # Step 4: Bootstrap confidence interval
        N = len(test_labels)
        idx = np.arange(N)  # Indices for sampling
        samples = []

        for _ in range(B):
            boot_idx = np.random.choice(idx, size=N, replace=True)  # Bootstrap sample
            lbl_boot = test_labels[boot_idx]
            pred_boot = test_preds[boot_idx]

            P_boot = lbl_boot.sum()
            F_boot = N - P_boot

            # Skip invalid bootstrap samples
            if P_boot == 0 or F_boot == 0:
                continue

            TPR_star = ((lbl_boot == 1) & (pred_boot == 1)).sum() / P_boot
            TNR_star = ((lbl_boot == 0) & (pred_boot == 0)).sum() / F_boot
            denom_star = TPR_star + TNR_star - 1

            if denom_star <= 0:
                continue

            theta_star = (p_obs + TNR_star - 1) / denom_star
            samples.append(np.clip(theta_star, 0, 1))  # Ensure theta_star is between 0 and 1

        if not samples:
            raise RuntimeError("No valid bootstrap samples; check inputs")

        # Calculate 95% confidence interval
        L, U = np.percentile(samples, [2.5, 97.5])
        return {
            "theta_hat": theta_hat,
            "low_ci_bound": L,
            "high_ci_bound": U
            }
    
    def extract_labels_and_preds(self, calibrate_eval_df: pd.DataFrame) -> dict:
        """
        Extracts the labels and predictions from a given evaluation DataFrame.

        Args:
            calibrate_eval_df (pd.DataFrame): The evaluation DataFrame containing 'eval_name' and 'answer' columns.
       
        Returns:
            dict: A dictionary of labels and predictions.
        """
        labeled_df = calibrate_eval_df[~calibrate_eval_df["golden_answer"].isna()].copy()
        unlabeled_df = calibrate_eval_df[calibrate_eval_df["golden_answer"].isna()].copy()
        test_labels = labeled_df["golden_answer"].tolist()
        test_preds = labeled_df["answer"].tolist()
        unlabeled_preds = unlabeled_df["answer"].tolist()
        return {"test_labels": test_labels, "test_preds": test_preds, "unlabeled_preds": unlabeled_preds}
    
    def convert_answer_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Checks if DataFrame has 'answer' column and converts string values to integers.

        Args:
            df (pd.DataFrame): Input DataFrame that may contain 'answer' column
            
        Returns:
            pd.DataFrame: DataFrame with 'answer' column converted to integers if applicable
        """
        if 'answer' in df.columns and df['answer'].dtype == 'object':
            df['answer'] = df['answer'].map({'pass': 1, 'fail': 0})
        return df

    def calibrate_success_rate_by_eval_name(self, eval_df: pd.DataFrame, golden_df: pd.DataFrame) -> dict:
        """
        Calibrates the success rate for each evaluation name.   

        Args:
            eval_df (pd.DataFrame): The evaluation DataFrame containing 'eval_name' and 'answer' columns.
            golden_df (pd.DataFrame): The golden DataFrame containing 'chat_id', 'conversation_id', 'message_idx', and 'golden_answer' columns.
        
        Returns:
            dict: A dictionary of success rates for each evaluation name.
        """
        eval_df = self.convert_answer_to_int(eval_df)
        golden_df = self.convert_answer_to_int(golden_df)
        golden_df.rename(columns={"answer": "golden_answer"}, inplace=True)
        calibrate_eval_df = eval_df.merge(golden_df, on=["chat_id", "conversation_id", "message_idx", "eval_name"], how="left")
        eval_names = calibrate_eval_df["eval_name"].unique().tolist()
        success_rate_dict = {}
        for eval_name in eval_names:    
            calibrate_eval_df_i = calibrate_eval_df[calibrate_eval_df["eval_name"] == eval_name]
            label_pred_dict = self.extract_labels_and_preds(calibrate_eval_df_i)
            success_rate_dict[eval_name] = self.calibrate_success_rate(test_labels=label_pred_dict["test_labels"], 
                                                                      test_preds=label_pred_dict["test_preds"], 
                                                                      unlabeled_preds=label_pred_dict["unlabeled_preds"])
        return success_rate_dict
    
    def init_pinecone(self, pinecone_namespace):  
        """
        Initializes a Pinecone index.

        Args:
            pinecone_index_name (str): Name of the Pinecone index to initialize

        Returns:
            None
        """
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        pinecone_client = PineconeVectorStore(
            index_name=self.pinecone_index_name,
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            embedding=embeddings,
            namespace=pinecone_namespace
        )
        self.pinecone_client = pinecone_client
    
    def similarity_search(self, query: str, pinecone_namespace: str, k: int=10) -> List[str]:
        """
        Retrieves documents from Pinecone.

        Args:
            query (str): The query to retrieve documents for
            pinecone_index_name (str): Name of the Pinecone index to search in
            k (int): Number of documents to retrieve
        """
        if self.pinecone_client is None:
            self.init_pinecone(pinecone_namespace)
        return self.pinecone_client.similarity_search(query, k=k)

    @staticmethod
    def calculate_mrr(index_list: List[int], k: int) -> float:
        """
        Calculates the Mean Reciprocal Rank (MRR) for a list of indices.

        Args:
            index_list (List[int]): A list of indices
            k (int): The number of documents to retrieve

        Returns:
            float: The Mean Reciprocal Rank (MRR)
        """
        # Compute reciprocal ranks
        reciprocal_ranks = []
        for index in index_list:
            if index <= k:  # Relevant item found within the top-k results
                reciprocal_ranks.append(1 / index)
            else:  # Relevant item not found
                reciprocal_ranks.append(0)
        # Calculate Mean Reciprocal Rank
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
        return mrr