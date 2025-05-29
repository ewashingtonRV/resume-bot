import os
from pathlib import Path


class MarkdownReader:
    def __init__(self):
        pass

    def validate_directory_path(self, directory_path: str) -> bool:
        """
        Validates that the directory path exists.
        
        Args:
            directory_path (str): The path to the directory to validate
            
        Returns: 
            bool: True if the directory path exists, False otherwise
        """
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory {directory_path} does not exist")
        return True

    @staticmethod
    def read_markdown_files(directory_path: str) -> str:
        """Read and concatenate all markdown files in the directory."""
        path = Path(directory_path)
        markdown_files = list(path.glob("**/*.md"))
        content = []
        
        for file_path in markdown_files:
            with open(file_path, "r") as f:
                content.append(f.read())
                
        return "\n\n".join(content)