#!/usr/bin/env python3
"""
Whisper Transcript Generation
Complete transcript processing including CLI usage and orchestration functionality
"""

import re
import subprocess
import sys
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any
import whisper
from openclip_core.config import WHISPER_MODEL, TRANSCRIPT_LANGUAGE_DETECT_MODEL
from openclip_core.transcript_generation_paraformer import ParaformerTranscriptProcessor

logger = logging.getLogger(__name__)

try:
    from openclip_core.transcript_generation_whisperx import TranscriptProcessorWhisperX, WHISPERX_AVAILABLE
except ImportError:
    WHISPERX_AVAILABLE = False


def select_transcript_backend(
    detected_language: str,
    paraformer_available: bool,
    use_whisperx: bool,
) -> str:
    """Choose the transcript backend for a detected language."""
    language = (detected_language or "").lower()
    if language.startswith("zh") and paraformer_available:
        return "paraformer"
    return "whisperx" if use_whisperx else "whisper"


def summarize_transcript_sources(sources: List[str]) -> str:
    """Summarize one or more transcript source names into a display value."""
    unique_sources = []
    for source in sources:
        if source and source not in unique_sources:
            unique_sources.append(source)
    if not unique_sources:
        return "unknown"
    if len(unique_sources) == 1:
        return unique_sources[0]
    return "mixed:" + ",".join(unique_sources)


def build_whisper_initial_prompt(language: Optional[str]) -> Optional[str]:
    """Return a style prompt for Whisper when a language benefits from steering.

    Whisper uses a single `zh` language code for Chinese, and OpenAI maintainers
    recommend `initial_prompt` to bias the transcript style toward simplified
    or traditional script. We prefer Simplified Chinese for Chinese transcripts.
    """
    normalized = (language or "").strip().lower()
    if normalized.startswith("zh") or normalized == "chinese":
        return "以下是普通话的简体中文字幕。"
    return None

def run_whisper_cli(file_path, model_name=WHISPER_MODEL, language=None, output_format="srt", output_dir=None):
    """
    Transcribe audio/video file using OpenAI Whisper CLI

    Args:
        file_path (str): Path to audio/video file
        model_name (str): Whisper model to use (tiny, base, small, medium, large, turbo)
        language (str): Language code (e.g., 'en', 'zh', 'ja') or None for auto-detection
        output_format (str): Output format (txt, vtt, srt, tsv, json, all)
        output_dir (str): Directory to write output files to (defaults to current directory)

    Returns:
        bool: True if successful, False if failed
    """
    print(f"🎵 Transcribing: {file_path}")
    print(f"📊 Model: {model_name}")
    print(f"📝 Output format: {output_format}")

    # Build the whisper command
    cmd = [sys.executable, "-m", "whisper", file_path, "--model", model_name, "--output_format", output_format]

    if output_dir:
        cmd.extend(["--output_dir", str(output_dir)])

    if language:
        cmd.extend(["--language", language])
        print(f"🌍 Language: {language}")
    else:
        print("🔍 Language: Auto-detection")

    initial_prompt = build_whisper_initial_prompt(language)
    if initial_prompt:
        cmd.extend(["--initial_prompt", initial_prompt])
        print("🈶 Script preference: Simplified Chinese")
    
    try:
        print("\n⏳ Running Whisper...")
        print("📋 Progress will be shown below:")
        print("-" * 50)
        
        # Run without capturing output to show real-time progress
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("-" * 50)
            print("✅ Transcription completed successfully!")
            return True
        else:
            print("-" * 50)
            print(f"❌ Transcription failed with return code: {result.returncode}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ Whisper module not found in the current Python environment.")
        return False

def demonstrate_whisper():
    """Demonstrate different Whisper usage examples"""
    
    print("=== OpenAI Whisper CLI Demo ===\n")
    
    # Check if we have a sample file
    sample_file = "../video_sample.mp4"
    
    if os.path.exists(sample_file):
        print("📁 Found sample video file!")
        
        print("\n--- Example 1: Basic transcription (tiny model, fast) ---")
        success = run_whisper_cli(sample_file, model_name="tiny")
        
        if success:
            # Look for output files
            base_name = os.path.splitext(os.path.basename(sample_file))[0]
            txt_file = f"{base_name}.txt"
            
            if os.path.exists(txt_file):
                print(f"\n📄 Transcript saved to: {txt_file}")
                # Show first few lines
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        preview = content[:200] + "..." if len(content) > 200 else content
                        print(f"Preview: {preview}")
                except Exception as e:
                    print(f"Could not read transcript: {e}")
        
        print("\n--- Example 2: Different formats ---")
        print("💡 You can also generate different output formats:")
        
    else:
        print("📂 No sample file found. Here are usage examples:")
    
    print("\n🎯 Usage Examples:")
    print("1. Basic transcription:")
    print("   whisper audio.mp3")
    
    print("\n2. Specify model size:")
    print("   whisper audio.mp3 --model small")
    
    print("\n3. Specify language:")
    print("   whisper audio.mp3 --language en")
    
    print("\n4. Multiple output formats:")
    print("   whisper audio.mp3 --output_format all")
    
    print("\n5. Subtitle format:")
    print("   whisper video.mp4 --output_format srt")
    
    print("\n📏 Available Models (speed vs accuracy):")
    models = [
        ("tiny", "Fastest, least accurate"),
        ("base", "Good balance"),
        ("small", "Better accuracy"),
        ("medium", "High accuracy"),
        ("large", "Best accuracy, slowest"),
        ("turbo", "Fast and accurate")
    ]
    
    for model, desc in models:
        print(f"   • {model}: {desc}")
    
    print("\n📋 Output Formats:")
    formats = ["txt", "vtt", "srt", "tsv", "json", "all"]
    for fmt in formats:
        print(f"   • {fmt}")

def simple_transcribe(audio_file, model="base"):
    """Simple function to transcribe an audio file"""
    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        return False
    
    return run_whisper_cli(audio_file, model_name=model)


class TranscriptProcessor:
    """Handles all transcript-related operations"""

    def __init__(
        self,
        whisper_model: str = WHISPER_MODEL,
        language: Optional[str] = None,
        enable_diarization: bool = False,
        speaker_references_dir: Optional[str] = None,
    ):
        self.whisper_model = whisper_model
        self.language = language  # None = auto-detect
        self.enable_diarization = enable_diarization
        self.language_detection_model = TRANSCRIPT_LANGUAGE_DETECT_MODEL
        # WhisperX is required for diarization; enable it automatically when requested.
        self.use_whisperx = enable_diarization and WHISPERX_AVAILABLE
        self.paraformer_processor = ParaformerTranscriptProcessor()
        self._language_detector = None

        if enable_diarization and not WHISPERX_AVAILABLE:
            logger.warning("⚠️  Speaker diarization requested but WhisperX is not installed. Falling back to openai-whisper (no speaker labels). Run: uv sync --extra speakers")

        self.whisperx_processor = None
        if self.use_whisperx:
            self.whisperx_processor = TranscriptProcessorWhisperX(
                whisper_model,
                enable_diarization=enable_diarization,
                speaker_references_dir=speaker_references_dir,
            )

        if self.paraformer_processor.is_available():
            logger.info(f"🈶 Chinese ASR backend: Paraformer ({self.paraformer_processor.project_dir})")
        else:
            logger.warning(
                "⚠️  Paraformer is unavailable; Chinese audio will fall back to Whisper. "
                f"Reason: {self.paraformer_processor.availability_error()}"
            )

    async def process_transcripts(self,
                                subtitle_path: str,
                                video_files: List[str] or str,
                                force_whisper: bool,
                                progress_callback: Optional[Callable[[str, float], None]]) -> Dict[str, Any]:
        """Process transcripts - either use existing subtitles or generate with whisper/whisperx"""

        has_existing = subtitle_path and os.path.exists(subtitle_path)

        if force_whisper or not has_existing:
            logger.info("📝 Generating transcripts locally with automatic language routing")
            return await self._generate_routed_transcripts(video_files, progress_callback)
        else:
            # Scenario 2: Use existing transcript
            if self.whisperx_processor and self.enable_diarization:
                if self._has_speaker_labels(subtitle_path):
                    logger.info("📥 Source transcript already has speaker labels, skipping diarization")
                    return {
                        'source': 'existing_diarized',
                        'transcript_path': subtitle_path if isinstance(video_files, str) else '',
                        'transcript_parts': [] if isinstance(video_files, str) else self._get_existing_transcript_parts(video_files)
                    }
                else:
                    logger.info("⚡ Using WhisperX diarization on existing transcript")
                    return await self._add_speakers_to_existing(video_files, progress_callback)
            else:
                logger.info("📥 Using existing subtitles")
                return {
                    'source': 'bilibili' if 'bilibili' in subtitle_path else 'existing',
                    'transcript_path': subtitle_path if isinstance(video_files, str) else '',
                    'transcript_parts': [] if isinstance(video_files, str) else self._get_existing_transcript_parts(video_files)
                }

    def _get_language_detector(self):
        if self._language_detector is None:
            self._language_detector = whisper.load_model(self.language_detection_model)
        return self._language_detector

    def _detect_transcript_language(self, media_path: str) -> str:
        media_path = str(media_path)
        try:
            detector = self._get_language_detector()
            audio = whisper.load_audio(media_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(detector.device)
            _, probs = detector.detect_language(mel)
            detected_language, confidence = max(probs.items(), key=lambda item: item[1])
            logger.info(
                f"🔎 Transcript language for {Path(media_path).name}: "
                f"{detected_language} ({confidence:.1%})"
            )
            return detected_language
        except Exception as e:
            logger.warning(
                f"⚠️  Transcript language detection failed for {Path(media_path).name} "
                f"({e}). Falling back to English/Whisper."
            )
            return "en"

    async def _generate_routed_transcripts(
        self,
        video_files: List[str] or str,
        progress_callback: Optional[Callable[[str, float], None]],
    ) -> Dict[str, Any]:
        """Generate transcripts with Whisper for English and Paraformer for Chinese."""
        if isinstance(video_files, str):
            video_files = [video_files]

        transcript_parts = []
        transcript_sources = []
        total_files = len(video_files)

        for i, video_file in enumerate(video_files):
            video_path = Path(video_file)
            video_dir = video_path.parent
            base_progress = 35 + (i / total_files) * 13 if total_files else 35

            detected_language = self._detect_transcript_language(str(video_path))
            backend = select_transcript_backend(
                detected_language=detected_language,
                paraformer_available=self.paraformer_processor.is_available(),
                use_whisperx=self.use_whisperx,
            )

            if progress_callback:
                progress_callback(
                    f"Generating transcript {i+1}/{total_files} with {backend}...",
                    base_progress,
                )

            logger.info(
                f"🔀 Transcript backend for {video_path.name}: {backend} "
                f"(detected language: {detected_language})"
            )

            srt_path = ""
            source = backend

            try:
                if backend == "paraformer":
                    srt_path, _ = self.paraformer_processor.transcribe_chinese_to_srt(
                        str(video_path),
                        video_dir,
                    )
                    logger.info(f"✅ Paraformer generated: {Path(srt_path).name}")
                    if self.whisperx_processor and self.enable_diarization:
                        logger.info("⚡ Running WhisperX diarization on Paraformer transcript")
                        srt_path = await self.whisperx_processor.add_speakers_to_existing_transcript(
                            srt_path,
                            str(video_path),
                            progress_callback,
                        )
                        source = "paraformer_diarized"
                elif backend == "whisperx":
                    srt_path = await self.whisperx_processor.transcribe_with_whisperx(
                        str(video_path),
                        progress_callback,
                    )
                    if srt_path:
                        logger.info(f"✅ WhisperX generated: {Path(srt_path).name}")
                else:
                    success = run_whisper_cli(
                        str(video_path),
                        model_name=self.whisper_model,
                        language=detected_language,
                        output_format="srt",
                        output_dir=str(video_dir),
                    )
                    if success:
                        srt_path = str(video_dir / f"{video_path.stem}.srt")
                        logger.info(f"✅ Whisper generated: {Path(srt_path).name}")
            except Exception as e:
                if backend == "paraformer":
                    logger.warning(
                        f"⚠️  Paraformer failed for {video_path.name} ({e}). Falling back to Whisper."
                    )
                    source = "whisper_fallback"
                    success = run_whisper_cli(
                        str(video_path),
                        model_name=self.whisper_model,
                        language=detected_language,
                        output_format="srt",
                        output_dir=str(video_dir),
                    )
                    if success:
                        srt_path = str(video_dir / f"{video_path.stem}.srt")
                        logger.info(f"✅ Whisper fallback generated: {Path(srt_path).name}")
                else:
                    logger.error(f"❌ {backend} failed for {video_path.name}: {e}")

            if srt_path and Path(srt_path).exists():
                transcript_parts.append(str(srt_path))
                transcript_sources.append(source)
            else:
                logger.error(f"❌ Transcript generation failed for {video_path.name}")

        return {
            'source': summarize_transcript_sources(transcript_sources),
            'transcript_path': transcript_parts[0] if len(transcript_parts) == 1 else '',
            'transcript_parts': transcript_parts,
        }
    
    async def _generate_whisper_transcripts(self, 
                                          video_files: List[str] or str,
                                          progress_callback: Optional[Callable[[str, float], None]]) -> Dict[str, Any]:
        """Generate transcripts using Whisper"""
        
        if isinstance(video_files, str):
            video_files = [video_files]
        
        transcript_parts = []
        total_files = len(video_files)
        
        for i, video_file in enumerate(video_files):
            # Update progress
            if progress_callback:
                base_progress = 35 + (i / total_files) * 13  # 35-48% range
                progress_callback(f"Generating transcript {i+1}/{total_files}...", base_progress)
            
            logger.info(f"🎙️  Generating transcript for: {Path(video_file).name}")
            
            video_path = Path(video_file)
            video_dir = video_path.parent

            success = run_whisper_cli(
                str(video_path),
                model_name=self.whisper_model,
                language=self.language,
                output_format="srt",
                output_dir=str(video_dir)
            )

            if success:
                srt_path = video_dir / f"{video_path.stem}.srt"
                if srt_path.exists():
                    transcript_parts.append(str(srt_path))
                    logger.info(f"✅ Generated: {srt_path.name}")
                else:
                    logger.warning(f"⚠️  SRT file not found for {video_path.name}")
            else:
                logger.error(f"❌ Whisper failed for {video_path.name}")
        
        return {
            'source': 'whisper',
            'transcript_path': transcript_parts[0] if len(transcript_parts) == 1 else '',
            'transcript_parts': transcript_parts
        }
    
    async def _generate_whisperx_transcripts(self,
                                             video_files: List[str] or str,
                                             progress_callback: Optional[Callable[[str, float], None]]) -> Dict[str, Any]:
        """Generate transcripts using WhisperX (Scenario 1)."""
        if isinstance(video_files, str):
            video_files = [video_files]

        transcript_parts = []
        total_files = len(video_files)

        for i, video_file in enumerate(video_files):
            if progress_callback:
                base_progress = 35 + (i / total_files) * 13
                progress_callback(f"Transcribing {i+1}/{total_files} with WhisperX...", base_progress)

            logger.info(f"⚡ WhisperX transcribing: {Path(video_file).name}")
            srt_path = await self.whisperx_processor.transcribe_with_whisperx(video_file, progress_callback)

            if srt_path and Path(srt_path).exists():
                transcript_parts.append(srt_path)
                logger.info(f"✅ Generated: {Path(srt_path).name}")
            else:
                logger.error(f"❌ WhisperX failed for {Path(video_file).name}")

        return {
            'source': 'whisperx',
            'transcript_path': transcript_parts[0] if len(transcript_parts) == 1 else '',
            'transcript_parts': transcript_parts,
        }

    async def _add_speakers_to_existing(self,
                                        video_files: List[str] or str,
                                        progress_callback: Optional[Callable[[str, float], None]]) -> Dict[str, Any]:
        """Add speaker labels to existing SRT files via diarization (Scenario 2)."""
        if isinstance(video_files, str):
            video_files = [video_files]

        transcript_parts = []
        total_files = len(video_files)

        for i, video_file in enumerate(video_files):
            video_path = Path(video_file)
            srt_path = video_path.parent / f"{video_path.stem}.srt"

            if not srt_path.exists():
                logger.warning(f"⚠️  No subtitle found next to {video_path.name}, skipping diarization")
                continue

            if progress_callback:
                base_progress = 35 + (i / total_files) * 13
                progress_callback(f"Diarizing {i+1}/{total_files}...", base_progress)

            logger.info(f"⚡ WhisperX diarizing: {video_path.name}")
            updated_srt = await self.whisperx_processor.add_speakers_to_existing_transcript(
                str(srt_path), video_file, progress_callback
            )
            transcript_parts.append(updated_srt)

        return {
            'source': 'whisperx_diarized',
            'transcript_path': transcript_parts[0] if len(transcript_parts) == 1 else '',
            'transcript_parts': transcript_parts,
        }

    def _has_speaker_labels(self, srt_path: str) -> bool:
        """Return True if the SRT file already contains [SpeakerName] prefixes.

        WhisperX writes speaker labels as '[SPEAKER_00] text' or '[Sam Altman] text' —
        the bracket content always starts with an uppercase letter.
        YouTube sound annotations like '[laughter]' or '[applause]' are lowercase
        and must not be treated as speaker labels.
        """
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if re.match(r'^\[[A-Z]', line.strip()):
                        return True
        except (OSError, IOError):
            pass
        return False

    def _get_existing_transcript_parts(self, video_files: List[str]) -> List[str]:
        """Get existing transcript parts (they should already exist from splitting)"""
        transcript_parts = []
        
        for video_file in video_files:
            video_path = Path(video_file)
            srt_path = video_path.parent / f"{video_path.stem}.srt"
            
            if srt_path.exists():
                transcript_parts.append(str(srt_path))
            else:
                logger.warning(f"⚠️  Expected transcript not found: {srt_path}")
        
        return transcript_parts


def main():
    """Main function"""
    
    # Check command line arguments
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        model = sys.argv[2] if len(sys.argv) > 2 else "base"
        
        print(f"🎵 Transcribing file: {audio_file}")
        simple_transcribe(audio_file, model)
    else:
        # Run demonstration
        demonstrate_whisper()
    
    print("\n🚀 To transcribe your own file:")
    print("   python main.py your_audio_file.mp3 [model]")
    print("   Example: python main.py speech.wav tiny")

if __name__ == "__main__":
    main()
