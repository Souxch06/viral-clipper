import re

class FileStringUtils:
    @staticmethod
    def sanitize_filename(title: str) -> str:
        if title is None:
            title = 'default'
        """Clean title for use as filename"""
        # Remove emojis and special characters
        title = re.sub(r'[^\w\s-]', '', title)
        # Replace spaces with underscores
        title = re.sub(r'[\s\-]+', '_', title)
        # Remove multiple underscores
        title = re.sub(r'_+', '_', title)
        # Trim underscores
        title = title.strip('_')
        if len(title) == 0:
            title = 'default'
        return title
