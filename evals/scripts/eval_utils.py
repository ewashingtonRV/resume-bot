import requests
import json
import pandas as pd
from eval_prompts import *

def ask_single_question(question: str) -> str:
    api_base_url = "http://localhost:8000"

    messages = [
        {"role": "user", "content": question},
    ]
    # Prepare request
    payload = {
        "messages": messages
    }
    response = requests.post(f"{api_base_url}/chat", json=payload)
    response_json = convert_response_string_to_json(response)
    answer = get_answer_from_response_json(response_json)
    return answer

def convert_response_string_to_json(response: requests.Response) -> dict:
    response_str = response.content.decode("utf-8")
    # Load the string into a Python dictionary
    response_json = json.loads(response_str)
    return response_json

def get_answer_from_response_json(response_json: dict) -> str:
    response_lod = response_json["messages"]
    answer_list = [response for response in response_lod if response['role'] == 'assistant']
    return answer_list[0]["content"]

def get_golden_dataset_path() -> str:
    golden_dataset_path = "/Users/ewashington/Desktop/github/resume-bot/data/evals/golden_df.csv"
    return golden_dataset_path

def get_golden_dataset() -> pd.DataFrame:
    golden_dataset_path = get_golden_dataset_path()

    # Try these encodings (most likely to work):
    encodings_to_try = ['cp1252', 'latin-1', 'iso-8859-1', 'utf-8-sig']

    for encoding in encodings_to_try:
        try:
            golden_df = pd.read_csv(golden_dataset_path, encoding=encoding)
            print(f"✅ Success with encoding: {encoding}")
            break
        except UnicodeDecodeError as e:
            print(f"❌ Failed with {encoding}: {e}")
    return golden_df.dropna()

def save_golden_dataset(df: pd.DataFrame) -> None:
    golden_dataset_path = get_golden_dataset_path()
    df.to_csv(golden_dataset_path, index=False)
    print(f"Saved golden dataset to {golden_dataset_path}")

def create_lookup_dict(df: pd.DataFrame, key_column: str, value_column: str) -> dict:
    return {row[key_column]: row[value_column] for _, row in df.iterrows()}

def get_auto_annotations(golden_lod: list[dict], evaluator) -> tuple[list[dict], list[dict]]:
    factual_accuracy_response_lod = []
    successful_response_response_lod = []

    for i, dict in enumerate(golden_lod):
        if i % 10 == 0:
            print(round(i / len(golden_lod), 2))
        human_question = dict["human_question"]
        ai_answer = dict["ai_answer"]
        reference_text = dict["system_prompt"]
        tool_used = dict["tool_used"]
        thread_id = dict["thread_id"]

        factual_accuracy_system_prompt = create_factual_accuracy_system_prompt(human_question, reference_text, ai_answer)
        successful_response_system_prompt = create_successful_response_system_prompt(human_question, ai_answer, tool_used)

        factual_accuracy_response = evaluator.auto_annotate(factual_accuracy_system_prompt)
        factual_accuracy_response["eval_name"] = "factual_accuracy"
        factual_accuracy_response["thread_id"] = thread_id
        factual_accuracy_response_lod.append(factual_accuracy_response)

        successful_response_response = evaluator.auto_annotate(successful_response_system_prompt)
        successful_response_response["eval_name"] = "complete_response"
        successful_response_response["thread_id"] = thread_id
        successful_response_response_lod.append(successful_response_response)
        
    return factual_accuracy_response_lod, successful_response_response_lod

def create_eval_results_df(response_lods: list[list[dict]], golden_df: pd.DataFrame=None) -> pd.DataFrame:
    # Create empty list to store individual dataframes
    eval_dfs = []
    
    # Process each response lod
    for response_lod in response_lods:
        # Get eval name from first item in lod
        eval_name = response_lod[0]["eval_name"]
        
        # Create dataframe from response lod
        eval_df = pd.DataFrame(response_lod)
        
        # Map to corresponding golden response column
        if golden_df is not None:
            golden_col = f"eval_{eval_name}" if eval_name != "complete_response" else "eval_answer_user_question"
            thread_id_to_golden = create_lookup_dict(golden_df, "thread_id", golden_col)
            eval_df["golden_response"] = eval_df["thread_id"].map(thread_id_to_golden)
        eval_dfs.append(eval_df)
    
    # Combine all evaluation dataframes
    eval_results_df = pd.concat(eval_dfs)
    return eval_results_df