import requests
from datetime import datetime, timedelta
import os
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
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

    def get_repos(self) -> list[str]:
        """
        Get all repositories for the organization.
        """
        return [
            "dmd-evals-dbx-scratch", 
            "end_of_section_links",
            "guides-symptoms-preprocess",
            "health-assistant-batch-evals",
            "healthline-k1-tagging",
            "healthline-kmeta-tagging",
            "healthline-smart-links",
            "healthline-update-text-data",
            "healthline-write-content-app-data",
            "medical-taxonomy-enrichment-api",
            "optum-now-core-graphql-api",
            "optum-now-health-assistant-ai-api",
            "optum-now-health-assistant-be",
            "optum-now-health-assistant-ds",
            "optum-now-health-assistant-fe",
            "optum-now-raas-api-ds-sidecar",
            "optum-now-traveler-api-ds-sidecar",
            "optum-now-traveler-ds-api",
            "raas-admin-console",
            "raas-admin-console-ds-sidecar",
            "raas-py-sdk",
            "raas-py-sdk-classes",
            "raas-sdk",
            "rvo-appraiser",
            "rvo-eval-sdk",
            "rvo-sdapi-models",
            "tfe_databricks_datascience",
            "traveler-ds-api-classes",
            "traveler-ds-sdk",
            "traverler-v3-scratch",
            "unified-search-graphql-api",
            "unified-search-vespa-application"
        ]
        
    
    def get_repo_user_commits(self, repo_name: str, lookback_days: int) -> int:
        """
        Get the number of commits made by the user to a specific repository.
        """
        start_date, end_date = self.calculate_date_range(lookback_days)
        commits_url = f'{self.base_url}/repos/{self.org_name}/{repo_name}/commits'
        params = {
            'author': self.username,
            'since': start_date.isoformat(),
            'until': end_date.isoformat()
        }
        commits_response = requests.get(commits_url, headers=self.headers, params=params)
        commits_response.raise_for_status()
        return len(commits_response.json()) 
    
    def get_repo_user_prs(self, repo_name: str, lookback_days: int) -> int:
        """
        Get the number of pull requests made by the user to a specific repository.
        """
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
        pulls_response = requests.get(pulls_url, headers=self.headers, params=params)   
        pulls_response.raise_for_status()
        return len(pulls_response.json())
    
    @staticmethod
    def calculate_date_range(lookback_days: int) -> Tuple[datetime, datetime]:
        """
        Calculate the date range for the lookback period.
        """
        end_date = datetime.now()
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
        response = requests.post(self.graphql_url, json={"query": query}, headers=self.headers)
        return response.json()

    def get_repo_stats(self, lookback_days: int = 365) -> Dict[str, Any]:
        """
        Get GitHub statistics for contributions to repositories.

        Args:
            lookback_days: Number of days to look back for statistics

        Returns:
            Dict containing repository statistics including PRs and commits per repo
        """
        # Calculate date range
        start_date, end_date = self.calculate_date_range(lookback_days)
        
        # Initialize stats dictionary
        stats = {
            'repositories': {}
        }
        
        try:
            repositories = self.get_repos()
            for repo in repositories:
                repo_name = repo['name']
                
                # Get commit and PR counts for this repo
                commit_count = self.get_user_commits(repo_name, lookback_days)
                pr_count = self.get_user_pulls(repo_name, lookback_days)
                
                # Only include repos where user has contributed
                if commit_count > 0 or pr_count > 0:
                    stats['repositories'][repo_name] = {
                        'commits': commit_count,
                        'pull_requests': pr_count
                    }
            return stats
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching GitHub org stats: {str(e)}")
        
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
    

    
