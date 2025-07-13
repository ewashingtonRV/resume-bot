import aiohttp
import asyncio
from datetime import datetime, timedelta
import os
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
import logging
import ssl
import nest_asyncio
load_dotenv()

# Allow nested event loops (needed for FastAPI)
nest_asyncio.apply()

def run_async_safe(coro):
    """Safely run async code in both sync and async contexts."""
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're in an async context, we need to handle it differently
        if loop.is_running():
            # We're in a running event loop, so we can't use asyncio.run()
            # Instead, we'll use nest_asyncio to allow nested loops
            return asyncio.run(coro)
        else:
            # No running loop, safe to use asyncio.run()
            return asyncio.run(coro)
    except RuntimeError:
        # No event loop exists, safe to use asyncio.run()
        return asyncio.run(coro)

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

    def _create_ssl_session(self) -> aiohttp.ClientSession:
        """Create an aiohttp session with SSL configuration that works on macOS."""
        # Create SSL context that's more permissive for macOS
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        return aiohttp.ClientSession(connector=connector)

    def get_repos(self) -> list[dict]:
        """Get repositories dictionary."""
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

    @staticmethod
    def calculate_date_range(lookback_days: int) -> Tuple[datetime, datetime]:
        """Calculate the date range for the lookback period."""
        end_date = datetime.now()
        if isinstance(lookback_days, str):
            lookback_days = int(lookback_days)
        start_date = end_date - timedelta(days=lookback_days)
        return start_date, end_date

    async def get_repos_commits_batch(self, repo_names: List[str], lookback_days: int) -> Dict[str, List[Dict]]:
        """Get commits for multiple repositories in parallel."""
        start_date, end_date = self.calculate_date_range(lookback_days)
        
        async with self._create_ssl_session() as session:
            tasks = []
            for repo_name in repo_names:
                commits_url = f'{self.base_url}/repos/{self.org_name}/{repo_name}/commits'
                params = {
                    'author': self.username,
                    'since': start_date.isoformat(),
                    'until': end_date.isoformat()
                }
                tasks.append(self._fetch_commits(session, repo_name, commits_url, params))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            commits_dict = {}
            for i, result in enumerate(results):
                repo_name = repo_names[i]
                if isinstance(result, Exception):
                    logging.error(f"Error getting commits for repo {repo_name}: {str(result)}")
                    commits_dict[repo_name] = []
                else:
                    commits_dict[repo_name] = result
                    logging.info(f"Found {len(result)} commits for repo {repo_name}")
            
            return commits_dict

    async def get_repos_prs_batch(self, repo_names: List[str], lookback_days: int) -> Dict[str, List[Dict]]:
        """Get PRs for multiple repositories in parallel."""
        start_date, end_date = self.calculate_date_range(lookback_days)
        
        async with self._create_ssl_session() as session:
            tasks = []
            for repo_name in repo_names:
                pulls_url = f'{self.base_url}/repos/{self.org_name}/{repo_name}/pulls'
                params = {
                    'state': 'all',
                    'creator': self.username,
                    'sort': 'created',
                    'direction': 'desc',
                    'since': start_date.isoformat(),
                    'until': end_date.isoformat()
                }
                tasks.append(self._fetch_prs(session, repo_name, pulls_url, params))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            prs_dict = {}
            for i, result in enumerate(results):
                repo_name = repo_names[i]
                if isinstance(result, Exception):
                    logging.error(f"Error getting PRs for repo {repo_name}: {str(result)}")
                    prs_dict[repo_name] = []
                else:
                    prs_dict[repo_name] = result
                    logging.info(f"Found {len(result)} PRs for repo {repo_name}")
            
            return prs_dict

    async def _fetch_commits(self, session: aiohttp.ClientSession, repo_name: str, url: str, params: dict) -> List[Dict]:
        """Fetch commits for a single repository."""
        try:
            async with session.get(url, headers=self.headers, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            raise Exception(f"Failed to fetch commits for {repo_name}: {str(e)}")

    async def _fetch_prs(self, session: aiohttp.ClientSession, repo_name: str, url: str, params: dict) -> List[Dict]:
        """Fetch PRs for a single repository."""
        try:
            async with session.get(url, headers=self.headers, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            raise Exception(f"Failed to fetch PRs for {repo_name}: {str(e)}")

    async def get_commit_stats_batch(self, repo_name: str, commit_shas: List[str]) -> List[Dict]:
        """Get commit stats for multiple commits in parallel."""
        async with self._create_ssl_session() as session:
            tasks = []
            for sha in commit_shas:
                commits_sha_url = f'{self.base_url}/repos/{self.org_name}/{repo_name}/commits/{sha}'
                tasks.append(self._fetch_commit_stats(session, sha, commits_sha_url))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            stats_list = []
            for i, result in enumerate(results):
                sha = commit_shas[i]
                if isinstance(result, Exception):
                    logging.error(f"Error getting commit stats for {repo_name}/{sha}: {str(result)}")
                    stats_list.append({"total": 0})
                else:
                    stats_list.append(result)
            
            return stats_list

    async def _fetch_commit_stats(self, session: aiohttp.ClientSession, sha: str, url: str) -> Dict:
        """Fetch commit stats for a single commit."""
        try:
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                commit_data = await response.json()
                return commit_data.get("stats", {"total": 0})
        except Exception as e:
            raise Exception(f"Failed to fetch commit stats for {sha}: {str(e)}")

    def get_repo_stats(self, lookback_days: int = 365, intent_category_name: str = None) -> List[Dict[str, Any]]:
        """Get GitHub statistics for contributions to repositories (sync wrapper)."""
        return run_async_safe(self.get_repo_stats_async(lookback_days, intent_category_name))

    async def get_repo_stats_async(self, lookback_days: int = 365, intent_category_name: str = None) -> List[Dict[str, Any]]:
        """Get GitHub statistics for contributions to repositories asynchronously."""
        logging.info(f"Getting repo stats for category '{intent_category_name}' over past {lookback_days} days")
        stats_lod = []
        
        try:
            repo_lod = self.get_repos()
            category_repo_lod = [repo for repo in repo_lod if repo["category_name"] == intent_category_name]
            if not category_repo_lod:
                logging.warning(f"No category found matching '{intent_category_name}'")
                return stats_lod
                
            category_repos = category_repo_lod[0]["repos"]
            logging.info(f"Found {len(category_repos)} repos for category {intent_category_name}: {category_repos}")

            # Fetch commits and PRs in parallel for all repos
            commits_dict, prs_dict = await asyncio.gather(
                self.get_repos_commits_batch(category_repos, lookback_days),
                self.get_repos_prs_batch(category_repos, lookback_days)
            )

            # Process results for each repo
            for repo_name in category_repos:
                commits = commits_dict.get(repo_name, [])
                prs = prs_dict.get(repo_name, [])
                
                if commits or prs:
                    # Get commit stats in parallel
                    sha_list = [commit["sha"] for commit in commits if commit.get("sha")]
                    if sha_list:
                        commit_stats = await self.get_commit_stats_batch(repo_name, sha_list)
                        total_code_changes = sum(stat.get("total", 0) for stat in commit_stats)
                    else:
                        total_code_changes = 0

                    stats_dict = {
                        'repo_name': repo_name,
                        'commits': len(commits),
                        'pull_requests': len(prs),
                        'total_code_changes': total_code_changes
                    }
                    stats_lod.append(stats_dict)

            logging.info(f"Final stats for {intent_category_name}: {stats_lod}")
            return stats_lod

        except Exception as e:
            logging.error(f"Error in get_repo_stats: {str(e)}", exc_info=True)
            raise

    def get_user_stats(self, lookback_days: int = 365) -> Dict[str, Any]:
        """Get GitHub statistics for the user (sync wrapper)."""
        return run_async_safe(self.get_user_stats_async(lookback_days))

    async def get_user_stats_async(self, lookback_days: int = 365) -> Dict[str, Any]:
        """Get GitHub statistics for the user asynchronously."""
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
            async with self._create_ssl_session() as session:
                async with session.post(self.graphql_url, json={"query": query}, headers=self.headers) as response:
                    response.raise_for_status()
                    return await response.json()
        except Exception as e:
            logging.error(f"Error getting user stats: {str(e)}")
            return {}
    

    
