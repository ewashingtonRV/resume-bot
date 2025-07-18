def create_factual_accuracy_system_prompt(user_question: str, reference_text: str, ai_answer: str) -> str:
    factual_accuracy_system_prompt_template = """You are a helpful assistant whose objective is to evaluate the factual accuracy of an AI Agent's responses. The AI Agent's name is Remy.

    Your task is to determine if Remy's response (ai_answer) contains only information that can be found in the reference text, contains github statistics, or if it provides a disclaimer that some of the response information is not in the reference text.

    Here is the context you need to evaluate:
    User Question: {user_question}
    Reference Text: {reference_text} 
    Remy's Response: {ai_answer}

    Guidelines for evaluation:
    1. Compare Remy's response against the reference text
    2. Check if all factual claims in the response are supported by the reference text
    3. If Remy explicitly disclaims that some information is not in the reference text, this is acceptable
    4. If Remy's response contains github statistics, this is acceptable

    Return the following json:
        {{
            "response": "pass" or "fail",
            "response_reasoning": "if factual_accuracy is 'fail', provide reasoning for the answer. Only provide reasoning if factual_accuracy is 'fail'." 
        }}
    """
    return factual_accuracy_system_prompt_template.format(user_question=user_question, reference_text=reference_text, ai_answer=ai_answer)

def create_successful_response_system_prompt(user_question: str, ai_answer: str, tool_used: bool) -> str:
    successful_response_system_prompt_template = """You are a helpful assistant whose objective is to evaluate if the AI Agent's response successfully answers the user's question. The AI Agent's name is Remy.
    Your task is to determine if Remy's response (ai_answer) successfully answers the user's question (user_question).
    If tool_used is True, then an acceptable response provides github statistics.

    ### Input
    Here is the context you need to evaluate:
    User Question: {user_question}
    Remy's Response: {ai_answer}
    Tool Used: {tool_used}

    ### Examples:
    1. User Question: 
            "Can you provide more information about the propensity model?"
        Remy's Response: 
            "The propensity model mentioned in relation to the recommendation service Eric Washington developed involves predicting user behavior based on their interaction history. Here's how it works:

                1. **User Embeddings**: The model begins by tokenizing the events from a user's history. This tokenized data is input into a Long Short-Term Memory (LSTM) network with an attention layer, which is trained to predict the next event a user might take.

                2. **Output**: The trained LSTM generates user embeddings, which represent the likelihood of various actions a user might take, such as clicking on a product, reading a newsletter, or viewing another page.

                3. **Integration with Supervised Models**: These user embeddings are then fed into supervised classification models that output propensity scores. These scores help determine the probability that a user will engage with specific content or actions based on their historical behavior and context.

                4. **Contextual Features**: The efficacy of the propensity model is enhanced by including contextual features like geo-location, time of visit, and specific content tags (e.g., hierarchical clusters and named entity recognition) to better inform the recommendations made.

                5. **Dimensionality Reduction**: To handle the high volume of content tags, techniques such as target encoding are utilized to reduce dimensionality and enhance the model's ability to capture interactions between content types and recommendations.

            Overall, the propensity model plays a critical role in predicting user engagement effectively, enabling personalized recommendations that are likely to resonate with individual users."
        Tool Used: False
        Response: "pass"
        Response Reasoning: "Remy's response successfully answers the user's question."

    2. User Question: 
            "Can you describe what github activity Eric has with AI agent projects?"
        Remy's Response: 
            "It seems that there is no specific GitHub activity recorded for AI agent projects under Eric Washington's repositories. If you have any other questions about his experiences or contributions, feel free to ask!"
        Tool Used: True
        Response: "fail"
        Response Reasoning: "Remy's response does not provide github statistics."

    3. User Question: 
        "Could you describe a specific challenge you faced during integration with UMLS and how you resolved it?"
        Remy's Response: 
            "One specific challenge faced during the integration with UMLS was dealing with excessive and redundant data output from AWS Comprehend Medical. The API often returned too much information, including multiple codes with low probabilities or repeated entities with slight variations, which made it difficult to use the raw results effectively.
            To resolve this, a custom post-processing pipeline was developed. This pipeline filters out irrelevant codes, aggregates similar entities, and reformats the returned data into a structured format. This process ensures that the enriched data is actionable and relevant, meeting the specific needs of the business teams. This approach helped streamline the tagging process and improved the overall utility of the clinical tags applied to assets."
        Tool Used: True
        Response: "fail"
        Response Reasoning: "Remy's response provides information on how it solved challenges with AWS Medical Comprehend, but does not state any specific UMLS challenges."

    ### Output
    Return the following json:
        {{
            "response": "pass" or "fail",
            "response_reasoning": "if successful_response is 'fail', provide reasoning for the answer. Only provide reasoning if successful_response is 'fail'." 
        }}
    """
    return successful_response_system_prompt_template.format(user_question=user_question, ai_answer=ai_answer, tool_used=tool_used)