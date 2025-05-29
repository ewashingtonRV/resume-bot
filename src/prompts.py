from src.utils import MarkdownReader
mdr = MarkdownReader()

base_system_prompt_template = """When responding to a user, you should use the reference text and chat history to guide your response.
    Please tell the user when you base your answer on information that is not in the reference text.
    Please read the query, reference text, and response history carefully before determining your response.
    The reference text is provided below:
    [BEGIN DATA]
    ************
    [Reference text]: {reference_text}
    ************
    [END DATA]
    """

recommendation_reference_text = mdr.read_markdown_files('./data/projects/decision-api')
medical_taxonomy_reference_text = mdr.read_markdown_files('./data/projects/medical-taxonomy-enrichment')
smart_links_reference_text = mdr.read_markdown_files("./data/projects/smart-links")

recommendation_system_prompt= """You are an expert at building recommendation engines using contextual bandit models.
    {base_system_prompt_template}
    """.format(base_system_prompt_template=base_system_prompt_template.format(reference_text=recommendation_reference_text))

medical_taxonomy_system_prompt = """You are an expert at building medical taxonomy enrichment models.
    {base_system_prompt_template}
    """.format(base_system_prompt_template=base_system_prompt_template.format(reference_text=medical_taxonomy_reference_text))

smart_links_system_prompt = """You are an expert at building in-text link recommendation models geared towards SEO optimization.
    {base_system_prompt_template}
    """.format(base_system_prompt_template=base_system_prompt_template.format(reference_text=smart_links_reference_text))

resume_reference_text = mdr.read_markdown_files('./data/resume.md')
intent_classification_system_prompt = """Your objective is to classify a user's question to one of the categories below. 
The format for the categories are: 
- <Category Name> 
    - <Category Description>

Classify category by matching a users query to the most similar Category Name and Category Description.
************
- Category Name: Data science leader
    - Category Description: Lead a team of data scientists focused on building, deploying, and evaluating AI agents. Heavily involved in product road mapping and system architecture decisions. 
- Category Name: RaaS
    - Category Description: Collaborate with engineers to create and maintain our Relevance-as-a-Service (RaaS), which leverages Vespa, internal APIs, and GraphQL to recommend a wide range of assets including products, providers, drugs, medical information, and more. 
- Category Name: Medical taxonomy
    - Category Description: Created, deployed, and maintain our Medical Enrichment Service that applies clinical tags (ICD-10, SNOMED, RxNorms, and CUIs) to assets using AWS Comprehend Medical, UMLS, as well as external APIs.
- Category Name: Recommendation model
    - Category Description: Created a recommendation service using contextual bandits that led to nearly a 30 percent increase in user engagement on Healthline.com.  
- Category Name: Smart links
    - Category Description: Developed a model to recommend in-text links as well as a process for automatically injecting said links on site at scale. The service led to a 4 percent increase in traffic, worth several million dollars.  
- Category Name: Kmeta tagging
    - Category Description: Deployed an end-to-end tagging product that enabled analysts to create and apply more granular tags to content, leading to roughly $10 million in increased ad sales.
- Category Name: Article recommendation
    - Category Description: Deployed an article recommendation system that led to a 20 percent increase in user engagement, worth $2 million dollars. 
- Category Name: Tagging
    - Category Description: Deployed an end-to-end tagging product that enabled analysts to create and apply more granular tags to content, leading to roughly $10 million in increased ad sales.
- Category Name: Other
    - Category Description: The user is asking questions about the resume
************

 Only respond with the category name. Do not provide any additional information.
"""