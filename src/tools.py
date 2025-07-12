import requests
from datetime import datetime, timedelta
import os
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
import logging
load_dotenv()

class GitHubStats:
    def __init__(self):
        self.token = os.getenv('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("GitHub token not found. Please set GITHUB_TOKEN in your .env file")
        
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json'
        }
        self.base_url = 'https://api.github.com'
        self.username = "ewashingtonRV"
        self.org_name = "rvohealth"
        self.org_id = "O_kgDOBtImFw"
        self.graphql_url = "https://api.github.com/graphql"

    def get_repos(self) -> list[dict]:
        """
        Get all repositories for the organization.
        """
        logging.info("Getting repository mappings")
        repos_lod = [
            {"category_name": "ds-lead", "repos": ["dmd-evals-dbx-scratch", "health-assistant-batch-evals", "optum-now-health-assistant-ai-api", 
                                                   "optum-now-health-assistant-be", "optum-now-health-assistant-ds", "rvo-eval-sdk"]},
            {"category_name": "raas", "repos": ["raas-admin-console-ds-sidecar",  "optum-now-core-graphql-api", "optum-now-traveler-api-ds-sidecar", "optum-now-traveler-ds-api", "traverler-v3-scratch", "unified-search-vespa-application"]},
            {"category_name": "medical taxonomy", "repos": ["medical-taxonomy-enrichment-api", "guides-symptoms-preprocess"]},
            {"category_name": "recommendation models", "repos": ["rvo-sdapi-models", "tfe_databricks_datascience"]},
            {"category_name": "smart links", "repos": ["healthline-smart-links",  "end_of_section_links"]},
            {"category_name": "article tagging", "repos": ["healthline-k1-tagging", "healthline-kmeta-tagging", "healthline-smart-links", "healthline-update-text-data", "healthline-write-content-app-data"]}
        ]
        logging.info(f"Found {len(repos_lod)} category mappings")
        return repos_lod
        
    def get_repo_user_commits(self, repo_name: str, lookback_days: int) -> int:
        """
        Get the number of commits made by the user to a specific repository.
        """
        logging.info(f"Getting commits for repo: {repo_name}")
        start_date, end_date = self.calculate_date_range(lookback_days)
        commits_url = f'{self.base_url}/repos/{self.org_name}/{repo_name}/commits'
        params = {
            'author': self.username,
            'since': start_date.isoformat(),
            'until': end_date.isoformat()
        }
        try:
            commits_response = requests.get(commits_url, headers=self.headers, params=params)
            commits_response.raise_for_status()
            return commits_response
        except requests.exceptions.RequestException as e:
            logging.error(f"Error getting commits for repo {repo_name}: {str(e)}")
            return 0
    
    def get_repo_user_prs(self, repo_name: str, lookback_days: int) -> int:
        """
        Get the number of pull requests made by the user to a specific repository.
        """
        logging.info(f"Getting PRs for repo: {repo_name}")
        start_date, end_date = self.calculate_date_range(lookback_days)
        pulls_url = f'{self.base_url}/repos/{self.org_name}/{repo_name}/pulls'
        params = {
            'state': 'all',
            'creator': self.username,
            'sort': 'created',
            'direction': 'desc',
            'since': start_date.isoformat(),
            'until': end_date.isoformat()
        }
        try:
            pulls_response = requests.get(pulls_url, headers=self.headers, params=params)
            pulls_response.raise_for_status()
            return pulls_response
        except requests.exceptions.RequestException as e:
            logging.error(f"Error getting PRs for repo {repo_name}: {str(e)}")
            return 0
    
    @staticmethod
    def calculate_date_range(lookback_days: int) -> Tuple[datetime, datetime]:
        """
        Calculate the date range for the lookback period.
        """
        end_date = datetime.now()
        if isinstance(lookback_days, str):
            lookback_days = int(lookback_days)
        start_date = end_date - timedelta(days=lookback_days)
        return start_date, end_date
    
    def get_user_stats(self, lookback_days: int = 365) -> Dict[str, Any]:
        """
        Get GitHub statistics for the user.

        Args:
            lookback_days: Number of days to look back for statistics

        Returns:
            Dict containing various GitHub statistics for the user
        """
        logging.info(f"Getting user stats for past {lookback_days} days")
        start_date, end_date = self.calculate_date_range(lookback_days)
        query = f"""
            query {{
            user(login: "{self.username}") {{
                contributionsCollection(
                    organizationID: "{self.org_id}"
                    from: "{start_date.isoformat()}"
                    to: "{end_date.isoformat()}"
                ) {{
                    totalCommitContributions
                    totalPullRequestContributions
                    totalIssueContributions
                    }}
                }}
            }}
            """
        try:
            response = requests.post(self.graphql_url, json={"query": query}, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error getting user stats: {str(e)}")
            return {}

    def get_repo_stats(self, lookback_days: int = 365, intent_category_name: str = None) -> Dict[str, Any]:
        """
        Get GitHub statistics for contributions to repositories.

        Args:
            lookback_days: Number of days to look back for statistics
            intent_category_name: Category name to filter repositories (e.g., 'raas', 'medical taxonomy')

        Returns:
            Dict containing repository statistics including PRs and commits per repo
        """
        logging.info(f"Getting repo stats for category '{intent_category_name}' over past {lookback_days} days")
        
        # Initialize stats dictionary
        stats_lod = []
        
        try:
            repo_lod = self.get_repos()
            logging.info(f"Looking for repos in category: {intent_category_name}")
            
            # Find the category's repositories
            category_repo_lod = [repo for repo in repo_lod if repo["category_name"] == intent_category_name]
            category_repos = category_repo_lod[0]["repos"]
            logging.info(f"Found {len(category_repos)} repos for category {intent_category_name}: {category_repos}")
            
            for repo_name in category_repos:
                # Get commit and PR counts for this repo
                commit_response = self.get_repo_user_commits(repo_name, lookback_days)
                commit_count = len(commit_response.json())
                pr_response = self.get_repo_user_prs(repo_name, lookback_days)
                pr_count = len(pr_response.json())
                
                # Only include repos where user has contributed
                if commit_count > 0 or pr_count > 0:
                    stats_dict = {
                        'repo_name': repo_name,
                        'commits': commit_count,
                        'pull_requests': pr_count
                    }
                    total_code_changes = self.get_commit_stats_total_code_changes(commit_response.json(), repo_name)
                    stats_dict['total_code_changes'] = total_code_changes
                    stats_lod.append(stats_dict)
            logging.info(f"Final stats for {intent_category_name}: {stats_lod}")
            return stats_lod
        except Exception as e:
            logging.error(f"Error in get_repo_stats: {str(e)}", exc_info=True)
            raise
        
    def get_sha_list(self, commits_response: list[dict]) -> list[str]:
        sha_list = []
        for commit_dict in commits_response:
            commit_author = commit_dict.get("author", None)
            if commit_author:
                commit_author_login = commit_dict["author"].get("login", None)
                if commit_author_login == self.username:
                    sha = commit_dict.get("sha", None)
                    if sha:
                        sha_list.append(sha)
        return sha_list

    def get_commit_stats(self, sha: str, repo_name: str) -> dict:
        commits_sha_url = f'{self.base_url}/repos/{self.org_name}/{repo_name}/commits/{sha}'
        commits_sha_response = requests.get(commits_sha_url, headers=self.headers)
        commits_sha_response = commits_sha_response.json()
        commit_stats_dict = commits_sha_response["stats"]
        return commit_stats_dict
    
    def get_commit_stats_total_code_changes(self, commits_response: list[dict], repo_name: str) -> int:
        sha_list = self.get_sha_list(commits_response)
        commit_stats_lod = []
        for sha in sha_list:
            commit_stats_dict = self.get_commit_stats(sha, repo_name)
            commit_stats_lod.append(commit_stats_dict)
        total_code_changes = 0
        for commit_stats_dict in commit_stats_lod:
            total_code_changes += commit_stats_dict["total"]
        return total_code_changes
    

    
