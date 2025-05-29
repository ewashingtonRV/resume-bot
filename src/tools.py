import requests
from datetime import datetime, timedelta
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class GitHubStats:
    def __init__(self):
        self.token = os.getenv('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("GitHub token not found. Please set GITHUB_TOKEN in your .env file")
        
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'

    def get_yearly_stats(self, username: str) -> Dict[str, Any]:
        """
        Get GitHub statistics for the past year for a specific user.
        
        Args:
            username (str): GitHub username
            
        Returns:
            Dict containing various GitHub statistics
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Initialize stats dictionary
        stats = {
            'total_contributions': 0,
            'repositories_created': 0,
            'pull_requests': 0,
            'issues': 0,
            'commits': 0,
            'repositories': []
        }
        
        try:
            # Get user's repositories
            repos_url = f'{self.base_url}/users/{username}/repos'
            repos_response = requests.get(repos_url, headers=self.headers)
            repos_response.raise_for_status()
            repositories = repos_response.json()
            
            # Count repositories created in the last year
            for repo in repositories:
                created_at = datetime.strptime(repo['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                if created_at >= start_date:
                    stats['repositories_created'] += 1
                
                # Get commit activity for each repository
                commits_url = f'{self.base_url}/repos/{username}/{repo["name"]}/commits'
                params = {'since': start_date.isoformat(), 'until': end_date.isoformat()}
                commits_response = requests.get(commits_url, headers=self.headers, params=params)
                
                if commits_response.status_code == 200:
                    commits = commits_response.json()
                    stats['commits'] += len(commits)
                
                # Get pull requests
                pulls_url = f'{self.base_url}/repos/{username}/{repo["name"]}/pulls'
                params = {'state': 'all', 'sort': 'created', 'direction': 'desc'}
                pulls_response = requests.get(pulls_url, headers=self.headers, params=params)
                
                if pulls_response.status_code == 200:
                    pulls = pulls_response.json()
                    stats['pull_requests'] += len([pr for pr in pulls if datetime.strptime(pr['created_at'], '%Y-%m-%dT%H:%M:%SZ') >= start_date])
                
                # Get issues
                issues_url = f'{self.base_url}/repos/{username}/{repo["name"]}/issues'
                params = {'state': 'all', 'sort': 'created', 'direction': 'desc'}
                issues_response = requests.get(issues_url, headers=self.headers, params=params)
                
                if issues_response.status_code == 200:
                    issues = issues_response.json()
                    stats['issues'] += len([issue for issue in issues if datetime.strptime(issue['created_at'], '%Y-%m-%dT%H:%M:%SZ') >= start_date])
                
                # Add repository info
                stats['repositories'].append({
                    'name': repo['name'],
                    'stars': repo['stargazers_count'],
                    'forks': repo['forks_count'],
                    'created_at': repo['created_at']
                })
            
            return stats
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching GitHub stats: {str(e)}")

def get_github_yearly_stats(username: str) -> Dict[str, Any]:
    """
    Convenience function to get GitHub stats for a user.
    
    Args:
        username (str): GitHub username
        
    Returns:
        Dict containing various GitHub statistics
    """
    github_stats = GitHubStats()
    return github_stats.get_yearly_stats(username)

# Example usage:
if __name__ == "__main__":
    # Replace with your GitHub username
    username = "your-username"
    try:
        stats = get_github_yearly_stats(username)
        print(f"\nGitHub Statistics for {username} (Last 365 days):")
        print(f"Total Commits: {stats['commits']}")
        print(f"Repositories Created: {stats['repositories_created']}")
        print(f"Pull Requests: {stats['pull_requests']}")
        print(f"Issues: {stats['issues']}")
        print("\nRepositories:")
        for repo in stats['repositories']:
            print(f"- {repo['name']}: ⭐ {repo['stars']} | 🍴 {repo['forks']}")
    except Exception as e:
        print(f"Error: {str(e)}") 