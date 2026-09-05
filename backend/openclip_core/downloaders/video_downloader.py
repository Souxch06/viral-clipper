#!/usr/bin/env python3
"""
Unified Video Downloader
Supports both Bilibili and YouTube platforms
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Callable, Any

from .bilibili_downloader import ImprovedBilibiliDownloader
from .download_error_utils import enrich_download_error_message
from .youtube_downloader import YouTubeDownloader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoDownloader:
    """
    Unified video downloader that automatically detects platform and uses appropriate downloader
    """
    
    def __init__(
        self,
        output_dir: str = "downloads",
        quality: str = "best",
        browser: Optional[str] = None,
        cookies: Optional[str] = None,
        js_runtime: Optional[str] = "auto",
        js_runtime_path: Optional[str] = None,
    ):
        """
        Initialize the unified video downloader
        
        Args:
            output_dir: Base directory to save downloaded videos
            quality: Video quality preference (best, worst, or specific format)
            browser: Browser for cookie extraction when explicitly provided and no cookie file is provided
            cookies: Optional path to a Netscape-format cookies.txt file
            js_runtime: JavaScript runtime strategy for YouTube ('auto', 'deno', 'node', 'none')
            js_runtime_path: Optional explicit path to the JS runtime executable
        """
        self.output_dir = output_dir
        self.quality = quality
        self.browser = browser
        self.cookies = cookies
        self.js_runtime = js_runtime
        self.js_runtime_path = js_runtime_path
        
        # Initialize platform-specific downloaders
        self.bilibili_downloader = ImprovedBilibiliDownloader(
            output_dir=output_dir,
            quality=quality,
            browser=browser,
            cookies=cookies,
        )
        
        self.youtube_downloader = YouTubeDownloader(
            output_dir=output_dir,
            quality=quality,
            browser=browser,
            cookies=cookies,
            js_runtime=js_runtime,
            js_runtime_path=js_runtime_path,
        )

    def build_user_facing_error_message(self, source: str, error_text: str) -> str:
        """Build a user-facing error message with platform-specific guidance."""
        platform = self.detect_platform(source)
        youtube_downloader = getattr(self, 'youtube_downloader', None)
        has_cookie_auth = bool(
            getattr(youtube_downloader, 'cookies', None) or
            getattr(youtube_downloader, 'browser', None)
        )
        return enrich_download_error_message(
            source=source,
            error_text=error_text,
            platform=platform,
            has_cookie_auth=has_cookie_auth,
        )
    
    def detect_platform(self, url: str) -> str:
        """
        Detect video platform from URL
        
        Args:
            url: Video URL
            
        Returns:
            Platform name: 'bilibili', 'youtube', or 'unknown'
        """
        # Bilibili patterns
        bilibili_patterns = [
            r'https?://(?:www\.)?bilibili\.com/video/[Bb][Vv][0-9A-Za-z]+',
            r'https?://(?:www\.)?bilibili\.com/bangumi/',
            r'https?://(?:www\.)?b23\.tv/',
            r'https?://(?:m\.)?bilibili\.com/video/',
        ]
        
        # YouTube patterns
        youtube_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'https?://(?:www\.)?youtube\.com/shorts/[\w-]+',
            r'https?://youtu\.be/[\w-]+',
            r'https?://(?:www\.)?youtube\.com/embed/[\w-]+',
        ]
        
        if any(re.match(pattern, url) for pattern in bilibili_patterns):
            return 'bilibili'
        elif any(re.match(pattern, url) for pattern in youtube_patterns):
            return 'youtube'
        else:
            return 'unknown'
    
    async def get_video_info(self, url: str) -> Dict[str, Any]:
        """
        Get video information without downloading
        
        Args:
            url: Video URL
            
        Returns:
            Video information dictionary
        """
        platform = self.detect_platform(url)
        
        if platform == 'bilibili':
            logger.info("🎬 Detected platform: Bilibili")
            video_info = await self.bilibili_downloader.get_video_info(url)
            return video_info.to_dict()
        elif platform == 'youtube':
            logger.info("🎬 Detected platform: YouTube")
            video_info = await self.youtube_downloader.get_video_info(url)
            return video_info.to_dict()
        else:
            raise ValueError(f"Unsupported platform or invalid URL: {url}")
    
    async def download_video(
        self,
        url: str,
        custom_filename: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, str]:
        """
        Download video and subtitles from any supported platform
        
        Args:
            url: Video URL (Bilibili or YouTube)
            custom_filename: Custom filename template
            progress_callback: Progress callback function
            
        Returns:
            Dictionary containing video_path, subtitle_path, and video_info
        """
        platform = self.detect_platform(url)
        
        if platform == 'bilibili':
            logger.info("🎬 Downloading from Bilibili...")
            return await self.bilibili_downloader.download_video(
                url, custom_filename, progress_callback
            )
        elif platform == 'youtube':
            logger.info("🎬 Downloading from YouTube...")
            return await self.youtube_downloader.download_video(
                url, custom_filename, progress_callback
            )
        else:
            raise ValueError(f"Unsupported platform or invalid URL: {url}")


class DownloadProcessor:
    """Handles video downloading operations with unified downloader"""
    
    def __init__(self, downloader: VideoDownloader):
        self.downloader = downloader
    
    async def download_video(self, 
                           url: str, 
                           custom_filename: Optional[str],
                           progress_callback: Optional[Callable[[str, float], None]]) -> Dict[str, Any]:
        """Download video and subtitles with progress tracking"""
        
        # Create progress callback for download phase
        from openclip_core.video_utils import ProgressCallbackManager
        download_progress = ProgressCallbackManager.create_download_progress_callback(
            progress_callback, 0, 25
        )
        
        try:
            return await self.downloader.download_video(
                url,
                custom_filename,
                download_progress
            )
        except Exception as e:
            raise RuntimeError(
                self.downloader.build_user_facing_error_message(url, str(e))
            ) from e
