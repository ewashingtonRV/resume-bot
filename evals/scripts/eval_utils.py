import requests
import json


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