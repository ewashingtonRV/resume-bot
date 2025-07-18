

import os
from datetime import datetime, timedelta
from langsmith import Client
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class TracesETL:
    def __init__(self):
       pass

    @staticmethod
    def get_runs_list(lookback_days: int = 1):
        client = Client()
        langsmith_project_name = os.getenv("LANGSMITH_PROJECT")
        start_time = datetime.now() - timedelta(days=lookback_days)
        project_runs = client.list_runs(
            project_name=langsmith_project_name,
            start_time=start_time,
            run_type="llm")
        runs_list = list(project_runs)
        return runs_list

    @staticmethod
    def extract_question(trace_lod: list[dict]) -> str:
        input_dict = trace_lod[-1]["inputs"]["messages"][0][-1]["kwargs"]
        if input_dict["type"] == "human":
            question = input_dict["content"]
            return question
        else:
            raise ValueError("Input is not a human message")
        
    @staticmethod
    def extract_system_prompt(trace_lod: list[dict]) -> str:
        input_dict = trace_lod[-1]["inputs"]["messages"][0][0]["kwargs"]
        if input_dict["type"] == "system":
            system_prompt = input_dict["content"]
            return system_prompt
        else:
            raise ValueError("Input is not a system_prompt")
        
    @staticmethod
    def extract_ai_answer(trace_lod):
        return trace_lod[0]["outputs"]["generations"][0][0]["text"]

    @staticmethod
    def extract_classifications(trace_lod):
        classification_dict = {}
        for trace_dict in trace_lod:
            langgraph_node = trace_dict["extra"]["metadata"]["langgraph_node"]
            if "classification" in langgraph_node:
                classification = trace_dict["outputs"]["generations"][0][0]["text"]
                classification_dict[langgraph_node] = classification
        return classification_dict

    @staticmethod
    def get_filtered_traces_and_thread_id_list(is_local_testing: bool, runs_list):
        filtered_traces = [trace for trace in runs_list if trace.metadata['is_local_testing'] == is_local_testing]
        thread_id_list = list(set([trace.metadata['thread_id'] for trace in filtered_traces]))
        return filtered_traces, thread_id_list

    def create_eval_lod(self, is_local_testing: bool = True, lookback_days: int = 1):
        runs_list = self.get_runs_list(lookback_days=lookback_days)
        filtered_traces, thread_id_list = self.get_filtered_traces_and_thread_id_list(is_local_testing=is_local_testing, runs_list=runs_list)
        output_lod = []
        for thread_id in thread_id_list:
            output_dict_i = {}
            trace_lod_i = [trace.dict() for trace in filtered_traces if trace.metadata['thread_id'] == thread_id]
            human_question = self.extract_question(trace_lod_i)
            system_prompt = self.extract_system_prompt(trace_lod_i)
            ai_answer = self.extract_ai_answer(trace_lod_i)
            classifications = self.extract_classifications(trace_lod_i)
            output_dict_i["thread_id"] = thread_id
            output_dict_i["human_question"] = human_question
            output_dict_i["system_prompt"] = system_prompt
            output_dict_i["ai_answer"] = ai_answer
            output_dict_i["tool_used"] = "tools" in trace_lod_i[0]["extra"]["invocation_params"].keys()
            output_dict_i.update(classifications)
            output_lod.append(output_dict_i)
        return output_lod
    
    @staticmethod
    def create_messages(trace_dict: dict) -> list:
        messages = [
                SystemMessage(content=trace_dict["system_prompt"]),
                HumanMessage(content=trace_dict["human_question"]),
                AIMessage(content=trace_dict["ai_answer"])
            ]
        return messages