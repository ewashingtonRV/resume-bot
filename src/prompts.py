from src.utils import MarkdownReader
mdr = MarkdownReader()

resume_reference_text = mdr.read_markdown_files('./data/resume.md')
category_dict = [
    {
        "category_name": "data science leader",
        "category_description": "Lead a team of data scientists focused on building, deploying, and evaluating AI agents. Heavily involved in product road mapping and system architecture decisions."
    },
    {
        "category_name": "raas",
        "category_description": "Collaborate with engineers to create and maintain our Relevance-as-a-Service (RaaS), which leverages Vespa, internal APIs, and GraphQL to recommend a wide range of assets including products, providers, drugs, medical information, and more."
    },
    {
        "category_name": "medical taxonomy",
        "category_description": "Created, deployed, and maintain our Medical Enrichment Service that applies clinical tags (ICD-10, SNOMED, RxNorms, and CUIs) to assets using AWS Comprehend Medical, UMLS, as well as external APIs."
    },
    {
        "category_name": "recommendation model",
        "category_description": "Created a recommendation service using contextual bandits that led to nearly a 30 percent increase in user engagement on Healthline.com."
    },
    {
        "category_name": "smart links",
        "category_description": "Developed a model to recommend in-text links as well as a process for automatically injecting said links on site at scale. The service led to a 4 percent increase in traffic, worth several million dollars."
    },
    {
        "category_name": "kmeta tagging",
        "category_description": "Deployed an end-to-end tagging product that enabled analysts to create and apply more granular tags to content, leading to roughly $10 million in increased ad sales."
    },
    {
        "category_name": "article recommendation",
        "category_description": "Deployed an article recommendation system that led to a 20 percent increase in user engagement, worth $2 million dollars."
    },
    {
        "category_name": "tagging",
        "category_description": "Deployed an end-to-end tagging product that enabled analysts to create and apply more granular tags to content, leading to roughly $10 million in increased ad sales."
    }
]

intent_classification_system_prompt = """Your objective is to classify a user's question to one of the categories below. 
    The format for the categories are: 
    - <Category Name> 
        - <Category Description>

    Classify category by matching a users query to the most similar Category Name and Category Description.
    ************
    {category_dict}
    ************

    Only respond with the category name. Do not provide any additional information.
    """.format(category_dict=category_dict)

base_system_prompt_template = """Your objective is to help the user understand Eric Washington's resume better. 
    Users questions are specifically about this {bullet_point} in Eric Washington's resume. 
    Use the reference text and chat history to guide your response.
    Restrict your response to information in the reference text. 
    Please read the query, reference text, and response history carefully before determining your response.
    The user's question is provided below:
    The reference text is provided below:
    [BEGIN DATA]
    ************
    [Reference text]: {reference_text}
    ************
    [END DATA]
    """

def create_system_prompt(category_name, reference_text_path):
    reference_text = mdr.read_markdown_files(reference_text_path)
    bullet_point = [d["category_description"] for d in category_dict if d["category_name"] == category_name][0]
    return base_system_prompt_template.format(bullet_point=bullet_point, reference_text=reference_text)

recommendation_system_prompt = create_system_prompt("recommendation model", "./data/projects/decision-api")
medical_taxonomy_system_prompt = create_system_prompt("medical taxonomy", "./data/projects/medical-taxonomy-enrichment")
smart_links_system_prompt = create_system_prompt("smart links", "./data/projects/smart-links")