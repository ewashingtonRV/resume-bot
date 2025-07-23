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
    def read_markdown_files(path_input: str) -> str:
        """Read and concatenate markdown files from a file or directory path."""
        path = Path(path_input)
        content = []
        
        if path.is_file() and path.suffix.lower() in ['.md', '.markdown']:
            # Single markdown file
            try:
                with open(path, "r", encoding='utf-8') as f:
                    content.append(f.read())
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                with open(path, "r", encoding='latin-1') as f:
                    content.append(f.read())
        elif path.is_dir():
            # Directory - find all markdown files
            markdown_files = list(path.glob("**/*.md"))
            for file_path in markdown_files:
                try:
                    with open(file_path, "r", encoding='utf-8') as f:
                        content.append(f.read())
                except UnicodeDecodeError:
                    # Try with different encoding if UTF-8 fails
                    with open(file_path, "r", encoding='latin-1') as f:
                        content.append(f.read())
        else:
            raise ValueError(f"Path {path_input} is not a valid markdown file or directory")
                
        return "\n\n".join(content)