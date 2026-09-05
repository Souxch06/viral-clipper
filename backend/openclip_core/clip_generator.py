#!/usr/bin/env python3
"""
Clip Generator - Extract engaging video clips from analyzed moments
"""
import json
import subprocess
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from core.file_string_utils import FileStringUtils

logger = logging.getLogger(__name__)


class ClipGenerator:
    """Generate video clips from engaging moments analysis"""

    # Sentence-ending punctuation for boundary snapping
    SENTENCE_ENDINGS = set('.!?。？！')

    def __init__(self, output_dir: str = "engaging_clips",
                 normalize_boundaries: bool = False,
                 normalize_start_max_shift: float = 5.0,
                 normalize_end_max_shift: float = 8.0,
                 subtitle_gap_boundary_threshold: float = 0.6):
        """
        Initialize clip generator

        Args:
            output_dir: Directory to save generated clips
            normalize_boundaries: When True, normalize both start and end times
                to nearby SRT boundaries before clip extraction
            normalize_start_max_shift: Maximum seconds to move clip start earlier
                when normalizing to a subtitle boundary
            normalize_end_max_shift: Maximum seconds to move clip end later when
                normalizing to a subtitle boundary
            subtitle_gap_boundary_threshold: Minimum gap between subtitle blocks
                to treat as a stronger thought boundary when punctuation is weak
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.normalize_boundaries = normalize_boundaries
        self.normalize_start_max_shift = normalize_start_max_shift
        self.normalize_end_max_shift = normalize_end_max_shift
        self.subtitle_gap_boundary_threshold = subtitle_gap_boundary_threshold
        logger.info(f"📁 Clip output directory: {self.output_dir}")
        if self.normalize_boundaries:
            logger.info(
                "🧭 Boundary normalization: enabled "
                f"(start <= {normalize_start_max_shift}s back, end <= {normalize_end_max_shift}s forward, "
                f"subtitle gap >= {subtitle_gap_boundary_threshold}s)"
            )
    
    def generate_clips_from_analysis(self, 
                                    analysis_file: str,
                                    video_dir: str,
                                    subtitle_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate clips from engaging moments analysis
        
        Args:
            analysis_file: Path to top_engaging_moments.json
            video_dir: Directory containing source video files
            subtitle_dir: Directory containing subtitle files (optional)
            
        Returns:
            Dictionary with generation results
        """
        try:
            # Load analysis data
            with open(analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            video_dir = Path(video_dir)
            subtitle_dir = Path(subtitle_dir) if subtitle_dir else video_dir
            
            logger.info("🎬 Generating clips from Top Engaging Moments")
            logger.info(f"📁 Output: {self.output_dir}")
            logger.info(f"📝 Subtitle directory: {subtitle_dir}")
            
            clips_info = []
            successful_clips = 0
            
            # Process each engaging moment
            for moment in data['top_engaging_moments']:
                rank = moment['rank']
                title = moment['title']
                video_part = moment['timing']['video_part']
                start_time = moment['timing']['start_time']
                end_time = moment['timing']['end_time']
                duration = moment['timing']['duration']
                
                logger.info(f"[Rank {rank}] Processing: {title}")
                
                # Get source video file
                input_video = self._find_video_file(video_part, video_dir)
                if not input_video:
                    logger.warning(f"✗ Skipping rank {rank}: Video file not found")
                    continue
                
                # Create output filename
                safe_title = FileStringUtils.sanitize_filename(title)
                output_filename = f"rank_{rank:02d}_{safe_title}.mp4"
                output_path = self.output_dir / output_filename
                
                effective_start_time = start_time
                effective_end_time = end_time
                normalization_details = {
                    'start': 'disabled',
                    'end': 'disabled',
                }
                srt_segments = None
                subtitle_file = self._find_subtitle_file(video_part, subtitle_dir)
                normalization_source = "whisper" if moment.get("whisper_subtitle_source") else "original"
                normalization_subtitle_file = moment.get("whisper_subtitle_source") or subtitle_file
                logger.info(f"  📝 Boundary transcript source: {normalization_source}")
                if normalization_subtitle_file and self.normalize_boundaries:
                    srt_segments = self._parse_srt_file(normalization_subtitle_file)

                if srt_segments:
                    effective_start_time, effective_end_time, normalization_details = self._normalize_clip_boundaries(
                        start_time,
                        end_time,
                        srt_segments
                    )

                # Create the clip
                success = self._create_clip(
                    input_video,
                    effective_start_time,
                    effective_end_time,
                    str(output_path),
                    title
                )

                if success:
                    # Generate subtitle file for the clip
                    subtitle_filename = f"rank_{rank:02d}_{safe_title}.srt"
                    subtitle_path = self.output_dir / subtitle_filename
                    subtitle_generated = self._extract_subtitle_for_clip(
                        video_part,
                        effective_start_time,
                        effective_end_time,
                        str(subtitle_path),
                        subtitle_dir
                    )
                    whisper_subtitle_filename = None
                    whisper_subtitle_source = moment.get("whisper_subtitle_source")
                    if whisper_subtitle_source:
                        whisper_subtitle_filename = f"rank_{rank:02d}_{safe_title}.whisper.srt"
                        whisper_subtitle_path = self.output_dir / whisper_subtitle_filename
                        whisper_subtitle_generated = self._extract_subtitle_from_file(
                            whisper_subtitle_source,
                            effective_start_time,
                            effective_end_time,
                            str(whisper_subtitle_path),
                        )
                        if not whisper_subtitle_generated:
                            whisper_subtitle_filename = None
                    
                    effective_duration = max(
                        0.0,
                        self._parse_time_flexible(effective_end_time)
                        - self._parse_time_flexible(effective_start_time)
                    )
                    successful_clips += 1
                    clips_info.append({
                        'rank': rank,
                        'title': title,
                        'filename': output_filename,
                        'subtitle_filename': subtitle_filename if subtitle_generated else None,
                        'whisper_subtitle_filename': whisper_subtitle_filename,
                        'duration': round(effective_duration or duration, 3),
                        'video_part': video_part,
                        'time_range': f"{effective_start_time} - {effective_end_time}",
                        'original_time_range': f"{start_time} - {end_time}",
                        'normalization_details': normalization_details,
                        'engagement_level': moment['engagement_details'].get('engagement_level', 'N/A'),
                        'why_engaging': moment['why_engaging']
                    })
                    logger.info(f"✓ Saved: {output_filename}")
                    if subtitle_generated:
                        logger.info(f"✓ Subtitle: {subtitle_filename}")
                    else:
                        logger.info(f"⚠ No subtitle generated for this clip")
                    if whisper_subtitle_filename:
                        logger.info(f"✓ Whisper subtitle: {whisper_subtitle_filename}")
                else:
                    logger.error(f"✗ Failed: {output_filename}")
            
            # Create summary
            if clips_info:
                self._create_summary(clips_info, data)
            
            result = {
                'success': successful_clips > 0,
                'total_clips': len(data['top_engaging_moments']),
                'successful_clips': successful_clips,
                'clips_info': clips_info,
                'output_dir': str(self.output_dir)
            }
            
            logger.info(f"🎯 Generated {successful_clips}/{len(data['top_engaging_moments'])} clips")
            return result
            
        except Exception as e:
            logger.error(f"Error generating clips: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_clips': 0,
                'successful_clips': 0,
                'clips_info': []
            }
    
    def _find_video_file(self, video_part: str, video_dir: Path) -> Optional[str]:
        """Find video file for a given part"""
        # Try common patterns
        patterns = [
            f"*_{video_part}.mp4",
            f"{video_part}.mp4",
            "*.mp4"  # Fallback for single video
        ]
        
        for pattern in patterns:
            matches = list(video_dir.glob(pattern))
            if matches:
                return str(matches[0])
        
        return None
    
    def _find_subtitle_file(self, video_part: str, subtitle_dir: Path) -> Optional[str]:
        """Find subtitle file for a given part"""
        # Try common patterns
        patterns = [
            f"*_{video_part}.srt",
            f"{video_part}.srt",
            "*.srt"  # Fallback for single subtitle
        ]
        
        for pattern in patterns:
            matches = list(subtitle_dir.glob(pattern))
            if matches:
                return str(matches[0])
        
        return None
    
    def _parse_srt_file(self, srt_path: str) -> List[Dict]:
        """Parse SRT file and extract subtitle segments"""
        segments = []
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Split by double newlines to get individual subtitle blocks
            blocks = re.split(r'\n\s*\n', content)
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    index = int(lines[0])
                    time_line = lines[1]
                    text = '\n'.join(lines[2:])
                    
                    # Parse time line "00:00:00,000 --> 00:00:00,800"
                    time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', time_line)
                    if time_match:
                        start_time = time_match.group(1)
                        end_time = time_match.group(2)
                        segments.append({
                            'index': index,
                            'start_time': start_time,
                            'end_time': end_time,
                            'text': text
                        })
            
            logger.info(f"📝 Parsed {len(segments)} subtitle segments from {srt_path}")
            return segments
            
        except Exception as e:
            logger.warning(f"⚠ Error parsing SRT file: {e}")
            return []
    
    def _time_to_seconds_srt(self, time_str: str) -> float:
        """Convert SRT time format (HH:MM:SS,mmm) to seconds"""
        time_part, ms_part = time_str.split(',')
        h, m, s = map(int, time_part.split(':'))
        ms = int(ms_part)
        return h * 3600 + m * 60 + s + ms / 1000.0
    
    def _seconds_to_time_srt(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        ms = int((seconds % 1) * 1000)
        seconds = int(seconds)
        
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    
    def _extract_subtitle_for_clip(self, video_part: str, start_time: str, 
                                    end_time: str, output_path: str, 
                                    subtitle_dir: Path) -> bool:
        """Extract subtitle segments for a clip's time range and save to file"""
        try:
            # Find subtitle file
            subtitle_file = self._find_subtitle_file(video_part, subtitle_dir)
            if not subtitle_file:
                logger.info(f"⚠ No subtitle file found for {video_part}")
                return False
            return self._extract_subtitle_from_file(
                subtitle_file,
                start_time,
                end_time,
                output_path,
            )
            
        except Exception as e:
            logger.warning(f"⚠ Error extracting subtitle for clip: {e}")
            return False

    def _extract_subtitle_from_file(
        self,
        subtitle_file: str,
        start_time: str,
        end_time: str,
        output_path: str,
        speed: float = 1.0,
    ) -> bool:
        """Extract subtitle segments from an explicit SRT file for a clip's time range."""
        segments = self._parse_srt_file(subtitle_file)
        if not segments:
            logger.info(f"⚠ No subtitle segments found in {subtitle_file}")
            return False

        clip_start = self._parse_time_flexible(start_time)
        clip_end = self._parse_time_flexible(end_time)

        clip_segments = []
        for seg in segments:
            seg_start = self._time_to_seconds_srt(seg['start_time'])
            seg_end = self._time_to_seconds_srt(seg['end_time'])

            if seg_end > clip_start and seg_start < clip_end:
                new_start = max(0.0, seg_start - clip_start) / max(speed, 0.001)
                new_end = max(new_start + 0.1, (seg_end - clip_start) / max(speed, 0.001))

                clip_segments.append({
                    'index': len(clip_segments) + 1,
                    'start_time': self._seconds_to_time_srt(new_start),
                    'end_time': self._seconds_to_time_srt(new_end),
                    'text': seg['text']
                })

        if not clip_segments:
            logger.info(f"⚠ No subtitle segments overlap with clip time range")
            return False

        with open(output_path, 'w', encoding='utf-8') as f:
            for seg in clip_segments:
                f.write(f"{seg['index']}\n")
                f.write(f"{seg['start_time']} --> {seg['end_time']}\n")
                f.write(f"{seg['text']}\n\n")

        logger.info(f"✓ Generated subtitle file with {len(clip_segments)} segments")
        return True

    @staticmethod
    def _build_audio_tempo_filter(speed: float) -> str:
        if speed <= 0:
            raise ValueError("speed must be positive")
        remaining = speed
        factors: list[float] = []
        while remaining > 2.0:
            factors.append(2.0)
            remaining /= 2.0
        factors.append(remaining)
        return ",".join(f"atempo={factor:.3f}" for factor in factors)
    
    def _time_to_seconds(self, time_str: str) -> int:
        """Convert MM:SS or HH:MM:SS to seconds"""
        parts = time_str.split(':')
        if len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return 0

    def _parse_time_flexible(self, time_str: str) -> float:
        """Parse time string in HH:MM:SS, MM:SS, or HH:MM:SS.mmm format to seconds."""
        # Handle HH:MM:SS.mmm (ffmpeg-style with dot)
        if '.' in time_str:
            main, ms = time_str.rsplit('.', 1)
            parts = main.split(':')
            base = sum(int(p) * m for p, m in zip(parts, [3600, 60, 1][-len(parts):]))
            return base + int(ms) / 1000.0
        # Handle HH:MM:SS or MM:SS
        parts = time_str.split(':')
        return float(sum(int(p) * m for p, m in zip(parts, [3600, 60, 1][-len(parts):])))

    def _seconds_to_ffmpeg_time(self, seconds: float) -> str:
        """Convert seconds (float) to ffmpeg-compatible time string HH:MM:SS.mmm"""
        ms = int(round((seconds % 1) * 1000))
        total_s = int(seconds)
        h = total_s // 3600
        m = (total_s % 3600) // 60
        s = total_s % 60
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def _is_sentence_boundary_segment(self, seg: Dict) -> bool:
        """Return True when this subtitle segment appears to end a sentence."""
        text = seg.get('text', '').rstrip()
        return bool(text and text[-1] in self.SENTENCE_ENDINGS)

    def _gap_before_segment(self, srt_segments: List[Dict], index: int) -> float:
        """Return the gap in seconds before the segment at index."""
        if index <= 0:
            return 0.0
        prev_end = self._time_to_seconds_srt(srt_segments[index - 1]['end_time'])
        cur_start = self._time_to_seconds_srt(srt_segments[index]['start_time'])
        return max(0.0, cur_start - prev_end)

    def _gap_after_segment(self, srt_segments: List[Dict], index: int) -> float:
        """Return the gap in seconds after the segment at index."""
        if index >= len(srt_segments) - 1:
            return 0.0
        cur_end = self._time_to_seconds_srt(srt_segments[index]['end_time'])
        next_start = self._time_to_seconds_srt(srt_segments[index + 1]['start_time'])
        return max(0.0, next_start - cur_end)

    def _is_start_boundary_segment(self, srt_segments: List[Dict], index: int) -> bool:
        """
        Return True when this subtitle segment appears to start a fresh thought.
        We treat a segment as a stronger start candidate when the previous
        segment ends a sentence, when a larger subtitle gap precedes it,
        or when it is the first subtitle block.
        """
        if index <= 0:
            return True
        return (
            self._is_sentence_boundary_segment(srt_segments[index - 1])
            or self._gap_before_segment(srt_segments, index) >= self.subtitle_gap_boundary_threshold
        )

    def _find_start_boundary(self, start_time_seconds: float, srt_segments: List[Dict]) -> tuple[float, str]:
        """Snap start backward to a nearby subtitle boundary, preferring sentence starts."""
        lower_limit = max(0.0, start_time_seconds - self.normalize_start_max_shift)
        fallback = None
        strong = None
        strong_idx = None

        for idx, seg in enumerate(srt_segments):
            seg_start = self._time_to_seconds_srt(seg['start_time'])
            if seg_start > start_time_seconds:
                break
            if seg_start < lower_limit:
                continue

            fallback = seg_start
            if self._is_start_boundary_segment(srt_segments, idx):
                strong = seg_start
                strong_idx = idx

        if strong is not None and strong != start_time_seconds:
            gap_before = self._gap_before_segment(srt_segments, strong_idx or 0)
            reason = (
                f"subtitle gap {gap_before:.2f}s"
                if gap_before >= self.subtitle_gap_boundary_threshold
                else "sentence boundary"
            )
            logger.info(
                f"  🔧 Normalized start_time: {self._seconds_to_ffmpeg_time(start_time_seconds)}"
                f" → {self._seconds_to_ffmpeg_time(strong)}"
                f" (-{start_time_seconds - strong:.1f}s)"
                f" | reason: {reason}"
            )
            return strong, reason

        if fallback is not None and fallback != start_time_seconds:
            logger.info(
                f"  ⚠ No strong start boundary within {self.normalize_start_max_shift}s,"
                f" snapping to nearest subtitle start:"
                f" {self._seconds_to_ffmpeg_time(start_time_seconds)}"
                f" → {self._seconds_to_ffmpeg_time(fallback)}"
            )
            return fallback, "subtitle boundary fallback"

        return start_time_seconds, "unchanged"

    def _snap_end_time(self, end_time_seconds: float,
                       srt_segments: List[Dict],
                       max_extension: Optional[float] = None) -> tuple[float, str]:
        """
        Snap end_time forward to the nearest SRT segment that ends with
        sentence-ending punctuation.

        Args:
            end_time_seconds: Current end time in seconds
            srt_segments: Parsed SRT segments with 'end_time' and 'text'

        Returns:
            Adjusted end time in seconds (float, ms precision).
            Returns original if no suitable boundary found within max_extension.
        """
        extension = self.normalize_end_max_shift if max_extension is None else max_extension
        limit = end_time_seconds + extension
        nearest_seg_end = None
        gap_boundary = None

        for idx, seg in enumerate(srt_segments):
            seg_end = self._time_to_seconds_srt(seg['end_time'])
            if seg_end < end_time_seconds:
                continue
            if seg_end > limit:
                break
            # Track the nearest SRT segment for ms precision fallback
            if nearest_seg_end is None:
                nearest_seg_end = seg_end
            if self._is_sentence_boundary_segment(seg):
                text = seg['text'].rstrip()
                if seg_end != end_time_seconds:
                    logger.info(
                        f"  🔧 Snapped end_time: {self._seconds_to_ffmpeg_time(end_time_seconds)}"
                        f" → {self._seconds_to_ffmpeg_time(seg_end)}"
                        f" (+{seg_end - end_time_seconds:.1f}s)"
                        f" | ends with: \"{text[-40:]}\""
                    )
                return seg_end, "sentence boundary"
            if gap_boundary is None:
                gap_after = self._gap_after_segment(srt_segments, idx)
                if gap_after >= self.subtitle_gap_boundary_threshold:
                    gap_boundary = (seg_end, gap_after)

        if gap_boundary is not None:
            seg_end, gap_after = gap_boundary
            if seg_end != end_time_seconds:
                logger.info(
                    f"  🔧 Snapped end_time: {self._seconds_to_ffmpeg_time(end_time_seconds)}"
                    f" → {self._seconds_to_ffmpeg_time(seg_end)}"
                    f" (+{seg_end - end_time_seconds:.1f}s)"
                    f" | reason: subtitle gap {gap_after:.2f}s"
                )
            return seg_end, "subtitle gap"

        # No sentence boundary found — still snap to nearest SRT segment for ms precision
        if nearest_seg_end is not None and nearest_seg_end != end_time_seconds:
            logger.info(
                f"  ⚠ No sentence or gap boundary within {extension}s,"
                f" snapping to nearest SRT segment for ms precision:"
                f" {self._seconds_to_ffmpeg_time(end_time_seconds)}"
                f" → {self._seconds_to_ffmpeg_time(nearest_seg_end)}"
            )
            return nearest_seg_end, "subtitle boundary fallback"

        return end_time_seconds, "unchanged"

    def _normalize_clip_boundaries(
        self,
        start_time: str,
        end_time: str,
        srt_segments: List[Dict],
    ) -> tuple[str, str, Dict[str, str]]:
        """
        Normalize clip boundaries to nearby subtitle boundaries.

        Rules:
        - Move start backward up to normalize_start_max_shift
        - Move end forward up to normalize_end_max_shift
        - Prefer sentence boundaries when available
        - Fall back to nearest subtitle boundaries when that still improves alignment
        """
        start_seconds = self._parse_time_flexible(start_time)
        end_seconds = self._parse_time_flexible(end_time)

        normalized_start, start_reason = self._find_start_boundary(start_seconds, srt_segments)
        normalized_end, end_reason = self._snap_end_time(
            end_seconds,
            srt_segments,
            max_extension=self.normalize_end_max_shift,
        )

        # Guard against pathological overlap after normalization.
        if normalized_start >= normalized_end:
            logger.warning(
                "⚠ Boundary normalization produced an invalid range; "
                "falling back to original timestamps"
            )
            return start_time, end_time, {
                "start": "invalid normalization fallback",
                "end": "invalid normalization fallback",
            }

        return (
            self._seconds_to_ffmpeg_time(normalized_start),
            self._seconds_to_ffmpeg_time(normalized_end),
            {
                "start": start_reason,
                "end": end_reason,
            },
        )

    def _create_clip(self, input_video: str, start_time: str,
                    end_time: str, output_path: str, title: str, speed: float = 1.0) -> bool:
        """Create a video clip using ffmpeg.

        start_time/end_time may be HH:MM:SS or HH:MM:SS.mmm (when snapped).
        """
        try:
            # Support both HH:MM:SS and HH:MM:SS.mmm formats
            start_seconds = self._parse_time_flexible(start_time)
            end_seconds = self._parse_time_flexible(end_time)
            duration = end_seconds - start_seconds

            # Use ffmpeg to extract clip
            cmd = [
                'ffmpeg',
                '-ss', start_time,
                '-t', f"{duration:.3f}",
                '-i', input_video,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-vf', f'setpts=(PTS-STARTPTS)/{speed:.3f}',
                '-af', self._build_audio_tempo_filter(speed),
                '-movflags', '+faststart',
                '-y',
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                check=True
            )
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error creating clip: {e}")
            return False
    
    def _create_summary(self, clips_info: List[Dict], data: Dict):
        """Create markdown summary of generated clips"""
        summary_path = self.output_dir / "engaging_moments_summary.md"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# 🔥 Top Engaging Moments - Video Clips\n\n")
            f.write(f"**Total Clips**: {len(clips_info)}\n\n")
            
            analysis_summary = data.get('analysis_summary')
            if isinstance(analysis_summary, dict):
                f.write("## 📊 Analysis Summary\n")
                themes = analysis_summary.get('highest_engagement_themes') or []
                total_time = analysis_summary.get('total_engaging_content_time')
                recommendation = analysis_summary.get('recommendation')

                if themes:
                    f.write(f"**Highest Engagement Themes**: {', '.join(themes)}\n")
                if total_time:
                    f.write(f"**Total Engaging Content Time**: {total_time}\n")
                if recommendation:
                    f.write(f"**Recommendation**: {recommendation}\n")
                if themes or total_time or recommendation:
                    f.write("\n")
            
            f.write("## 🎬 Generated Clips\n\n")
            f.write("| Rank | Title | Video File | Subtitle File | Duration | Engagement |\n")
            f.write("|------|-------|------------|----------------|----------|------------|\n")
            
            for clip in clips_info:
                subtitle_file = clip.get('subtitle_filename', 'N/A')
                if subtitle_file:
                    subtitle_file = f"`{subtitle_file}`"
                else:
                    subtitle_file = "N/A"
                f.write(f"| {clip['rank']} | {clip['title']} | `{clip['filename']}` | "
                       f"{subtitle_file} | {clip['duration']} | {clip['engagement_level']} |\n")
            
            f.write("\n## 📝 Detailed Descriptions\n\n")
            for clip in clips_info:
                f.write(f"### Rank {clip['rank']}: {clip['title']}\n")
                f.write(f"**Time Range**: {clip['time_range']}\n")
                f.write(f"**Duration**: {clip['duration']}\n")
                f.write(f"**Video File**: `{clip['filename']}`\n")
                if clip.get('subtitle_filename'):
                    f.write(f"**Subtitle File**: `{clip['subtitle_filename']}`\n")
                f.write(f"**Why Engaging**: {clip['why_engaging']}\n\n")
        
        logger.info(f"📄 Summary created: {summary_path}")
