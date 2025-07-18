import requests
import json
import pandas as pd

def ask_single_question(question):
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

def convert_response_string_to_json(response):
    response_str = response.content.decode("utf-8")
    # Load the string into a Python dictionary
    response_json = json.loads(response_str)
    return response_json

def get_answer_from_response_json(response_json):
    response_lod = response_json["messages"]
    answer_list = [response for response in response_lod if response['role'] == 'assistant']
    return answer_list[0]["content"]

def get_golden_dataset_path():
    golden_dataset_path = "/Users/ewashington/Desktop/github/resume-bot/data/evals/golden_df.csv"
    return golden_dataset_path

def get_golden_dataset():
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

def save_golden_dataset(df):
    golden_dataset_path = get_golden_dataset_path()
    df.to_csv(golden_dataset_path, index=False)
    print(f"Saved golden dataset to {golden_dataset_path}")

