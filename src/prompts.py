from src.utils import MarkdownReader
mdr = MarkdownReader()

resume_reference_text = mdr.read_markdown_files('./data/resume.md')
category_dict = [
    {
        "category_name": "ds-lead",
        "category_description": "Lead a team of data scientists focused on building, deploying, and evaluating AI agents. Heavily involved in product road mapping and designing the system architecture for evaluating AI agents."
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
        "category_name": "recommendation models",
        "category_description": "Created a recommendation service using contextual bandits that led to nearly a 30 percent increase in user engagement on Healthline.com."
    },
    {
        "category_name": "smart links",
        "category_description": "Developed a model to recommend in-text links as well as a process for automatically injecting said links on site at scale. The service led to a 4 percent increase in traffic, worth several million dollars."
    },
    {
        "category_name": "article tagging",
        "category_description": "Deployed an end-to-end tagging product that enabled analysts to create and apply more granular tags to content, leading to roughly $10 million in increased ad sales."
    },
    {
        "category_name": "github stats",
        "category_description": "Fetch and display stats about Eric Washington's coding activity (commits and pull requests). But only classify the question as github stats if the question is about Eric Washington's overall coding activity. \
            If the questions is about Eric's codeing activity in respect to a specific project, then do not classify it as a github stats question."
    },
]

intent_classification_system_prompt = """You are a helpful assitant named Remy whose objective is to classify a user's question to one of the categories below. 
    Classify category by matching a users query to the most similar Category Name and Category Description.
    ************
    {category_dict}
    ************

    Only respond with the category name. Do not provide any additional information.
    """.format(category_dict=category_dict)

github_stats_classification_system_prompt = """You are a helpful assitant named Remy whose objective is to classify if a user's question is asking about coding or github activity.
    If the question is not about coding or github activity, then classify it as False.
    If the question is about coding or github activity, then classify it as True.
    Only respond with True or False. Do not provide any additional information.

    Classify as True if the question:
    - Asks about GitHub activity, contributions, or stats
    - Mentions coding activity or development work
    - Asks about commits, pull requests, or code changes
    - Asks about activity in specific projects or overall
    - Contains phrases like "github activity", "contributions", "coding work"

    Here are examples questions you should classify as True:
    - What are Eric's total contributions?
    - Show me GitHub stats for the past month
    - What's activity in the medical taxonomy project?
    - Can you tell me about github activity for the raas project?
    - Can you tell me about Eric's github activity?
    - How active has Eric been in coding lately?
    - What has Eric been working on in GitHub?
    - Show me Eric's development activity

    Here are examples you should classify as False:
    - What is the RaaS project about?
    - Tell me about the medical taxonomy work
    - What technologies does Eric use?
    - What are Eric's responsibilities?
    """

github_system_prompt = """You are a helpful assistant named Remy who provides information about Eric Washington's GitHub activity.
    You have access to two GitHub tools:
    1. github_user_stats: Use this for questions about Eric's overall GitHub activity across all projects
       - Example: "What are Eric's total contributions?"
       - Example: "Show me Eric's GitHub stats for the past month"
    Guidelines for tool selection:
    - For overall activity → use github_user_stats
    - Present results in a clear, readable format
    - Always explain what time period the stats cover
    """

base_system_prompt_template = """You are a helpful assistant named Remy whose objective is to help the user understand Eric Washington's resume better. 
    Users questions are specifically about this {bullet_point} in Eric Washington's resume. 
    Use the reference text and chat history to guide your response.

    If {is_github_stats} is True and {category_name} is github_stats, then use the github_user_stats tool to retrieve github statistics. The github_user_stats tool will return a a dictionary with the following schema:
        - totalCommitContributions: Number of commits
        - totalPullRequestContributions: Number of pull requests
        - totalIssueContributions: Number of issues

    If {is_github_stats} is True and {category_name} is NOT github_stats, then use the get_repo_stats tool to retrieve github statistics. The get_repo_stats tool will return a list of dictionaries, each with the following schema: 
        - repo_name: Name of the repository
        - commits: Number of commits
        - pull_requests: Number of pull requests
        - total_code_changes: Total number of code changes

    If {is_github_stats} is True, please show the user the github statistics in a readable format. Both of the abovementioned github tools accept two inputs:
        - intent_category_name: Name of the intent category to get repo stats. You should always use the {category_name} as the intent_category_name.
        - lookback_days: (Optional) Number of days to look back. If not specified, defaults to 365 days.
    When displaying githubt statistics, always tell the user the lookback_days.

    If {is_github_stats} is False, restrict your response to information in the reference text. The user will NOT have access to the reference text, so do not tell them to refer to it.
    Please read the query, reference text, and response history carefully before determining your response.

    The reference text is provided below:
    [BEGIN DATA]
    ************
    [Reference text]: {reference_text}
    ************
    [END DATA]

    If you are unable to answer the question, please advise the user to contact Eric Washington directly.
    """

def create_system_prompt(category_name: str, reference_text_path: str, is_github_stats: bool = False) -> str:
    """Create a system prompt for a given category with dynamic GitHub stats handling.
    
    Args:
        category_name: The name of the category (e.g., 'raas', 'medical taxonomy')
        reference_text_path: Path to the reference text file
        is_github_stats: Whether to include GitHub stats functionality
    """
    reference_text = mdr.read_markdown_files(reference_text_path)
    bullet_point = [d["category_description"] for d in category_dict if d["category_name"] == category_name][0]
    return base_system_prompt_template.format(
        bullet_point=bullet_point,
        reference_text=reference_text,
        category_name=category_name,
        is_github_stats=is_github_stats
    )

# Map category names to their reference paths
REFERENCE_PATHS = {
    "recommendation models": "./data/projects/decision-api",
    "medical taxonomy": "./data/projects/medical-taxonomy-enrichment",
    "smart links": "./data/projects/smart-links",
    "article tagging": "./data/projects/article-tagging",
    "raas": "./data/projects/raas",
    "ds-lead": "./data/projects/ds-lead"
}
