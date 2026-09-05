#!/usr/bin/env python3
"""
Minimal agentic analysis coordinator for engaging-moments mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
from time import perf_counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from core.config import WHISPER_MODEL
from core.transcript_generation_whisper import run_whisper_cli

logger = logging.getLogger(__name__)


class AnalysisCoordinator:
    """Runs a bounded analysis loop on top of the existing analyzer."""

    def __init__(
        self,
        analyzer,
        overlap_threshold_seconds: float = 45.0,
        overlap_threshold_ratio: float = 0.6,
        repair_max_context_segments: int = 4,
        judge_batch_size: int = 4,
        rejudge_batch_size: int = 2,
        verification_context_before_segments: int = 10,
        verification_context_after_segments: int = 8,
        verification_context_before_seconds: float = 60.0,
        verification_context_after_seconds: float = 60.0,
        verification_whisper_model: str = WHISPER_MODEL,
        max_parallel_judge_batches: int = 2,
        judge_batch_launch_stagger_seconds: float = 0.25,
        max_parallel_repairs: int = 2,
        repair_launch_stagger_seconds: float = 0.25,
    ):
        self.analyzer = analyzer
        self.overlap_threshold_seconds = overlap_threshold_seconds
        self.overlap_threshold_ratio = overlap_threshold_ratio
        self.repair_max_context_segments = repair_max_context_segments
        self.judge_batch_size = max(1, int(judge_batch_size))
        self.rejudge_batch_size = max(1, int(rejudge_batch_size))
        self.verification_context_before_segments = verification_context_before_segments
        self.verification_context_after_segments = verification_context_after_segments
        self.verification_context_before_seconds = verification_context_before_seconds
        self.verification_context_after_seconds = verification_context_after_seconds
        self.verification_whisper_model = verification_whisper_model
        self.max_parallel_judge_batches = max(1, int(max_parallel_judge_batches))
        self.judge_batch_launch_stagger_seconds = max(0.0, float(judge_batch_launch_stagger_seconds))
        self.max_parallel_repairs = max(1, int(max_parallel_repairs))
        self.repair_launch_stagger_seconds = max(0.0, float(repair_launch_stagger_seconds))

    async def run(
        self,
        transcript_parts: List[str],
        progress_callback=None,
    ) -> Dict[str, Any]:
        if not transcript_parts:
            return {
                "error": "No transcript parts available for agentic analysis",
                "highlights_files": [],
                "aggregated_file": None,
                "top_moments": None,
                "total_parts_analyzed": 0,
                "agentic_analysis": True,
            }

        transcript_dir = Path(transcript_parts[0]).parent
        run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        workflow = self._create_workflow(run_id, transcript_parts)
        analysis_plan = self._build_analysis_plan(transcript_parts)
        analysis_plan["run_id"] = run_id
        analysis_plan_file = transcript_dir / "analysis_plan.json"
        await self.analyzer.save_highlights_to_file(analysis_plan, str(analysis_plan_file))
        self._add_stage(
            workflow,
            stage="plan",
            status="completed",
            artifact=analysis_plan_file.name,
        )

        highlights_files = []
        logger.info("🧠 Agentic analysis enabled: running bounded analysis loop")

        if progress_callback:
            progress_callback("Planning agentic analysis...", 50)

        for i, transcript_path in enumerate(transcript_parts):
            part_name = f"part{i+1:02d}"
            highlights = await self.analyzer.analyze_part_for_engaging_moments(
                transcript_path, part_name
            )
            highlights_file = Path(transcript_path).parent / f"highlights_{part_name}.json"
            await self.analyzer.save_highlights_to_file(highlights, str(highlights_file))
            highlights_files.append(str(highlights_file))

            if progress_callback:
                progress = 50 + (i + 1) * 8 / len(transcript_parts)
                progress_callback(
                    f"Generated agentic candidates for part {i+1}/{len(transcript_parts)}",
                    progress,
                )

        self._add_stage(
            workflow,
            stage="generate_candidates",
            status="completed",
            input_candidates=len(transcript_parts),
            output_candidates=len(highlights_files),
            artifacts=[Path(p).name for p in highlights_files],
        )

        logger.info("🔄 Aggregating pre-verification shortlist...")
        pre_verify_pool_size = self._compute_pre_verify_pool_size()
        top_moments = self.analyzer.build_pre_verify_pool(
            highlights_files,
            pre_verify_pool_size,
        )
        pre_verify_file = transcript_dir / "top_engaging_moments.pre_verify.json"
        await self.analyzer.save_highlights_to_file(top_moments, str(pre_verify_file))
        transcript_map = {
            f"part{i+1:02d}": path for i, path in enumerate(transcript_parts)
        }
        self._attach_whisper_context_transcripts(
            top_moments.get("top_engaging_moments", []),
            transcript_map,
            transcript_dir / "whisper_pre_verify_contexts",
        )
        self._add_stage(
            workflow,
            stage="aggregate_round_1",
            status="completed",
            input_candidates=self._count_raw_candidates(highlights_files),
            output_candidates=len(top_moments.get("top_engaging_moments", [])),
            artifact=pre_verify_file.name,
        )

        if progress_callback:
            progress_callback("Verifying standalone quality...", 60)

        final_top_moments, repair_pass_used, verification_report = await self._verify_and_finalize(
            top_moments,
            transcript_parts,
            progress_callback=progress_callback,
        )
        verification_report["run_id"] = run_id
        verification_report_file = transcript_dir / "verification_report.json"
        await self.analyzer.save_highlights_to_file(
            verification_report,
            str(verification_report_file),
        )
        self._add_stage(
            workflow,
            stage="verify_round_1",
            status="completed",
            input_candidates=len(top_moments.get("top_engaging_moments", [])),
            output_candidates=len(final_top_moments.get("top_engaging_moments", [])),
            repaired=sum(1 for clip in verification_report["clips"] if clip["decision"] == "repaired"),
            rejected=sum(1 for clip in verification_report["clips"] if clip["decision"] == "rejected"),
            kept=sum(1 for clip in verification_report["clips"] if clip["decision"] == "kept"),
            artifact=verification_report_file.name,
        )

        aggregated_file = transcript_dir / "top_engaging_moments.json"
        await self.analyzer.save_highlights_to_file(final_top_moments, str(aggregated_file))
        self._add_stage(
            workflow,
            stage="finalize",
            status="completed",
            output_candidates=len(final_top_moments.get("top_engaging_moments", [])),
            artifact=aggregated_file.name,
        )
        workflow_file = transcript_dir / "analysis_workflow.json"
        await self.analyzer.save_highlights_to_file(workflow, str(workflow_file))

        return {
            "highlights_files": highlights_files,
            "aggregated_file": str(aggregated_file),
            "pre_verify_file": str(pre_verify_file),
            "analysis_plan_file": str(analysis_plan_file),
            "workflow_file": str(workflow_file),
            "verification_report_file": str(verification_report_file),
            "top_moments": final_top_moments,
            "total_parts_analyzed": len(transcript_parts),
            "repair_pass_used": repair_pass_used,
            "agentic_analysis": True,
        }

    def _build_analysis_plan(self, transcript_parts: List[str]) -> Dict[str, Any]:
        return {
            "agentic_analysis_enabled": True,
            "mode": "engaging_moments",
            "language": getattr(self.analyzer, "language", "zh"),
            "user_intent": getattr(self.analyzer, "user_intent", None),
            "target_clip_count": getattr(self.analyzer, "max_clips", 0),
            "pre_verify_pool_size": self._compute_pre_verify_pool_size(),
            "verification_mode": "llm_standalone_first",
            "retry_budget": 1,
            "total_parts": len(transcript_parts),
            "analysis_timestamp": datetime.now().isoformat() + "Z",
        }

    def _compute_pre_verify_pool_size(self) -> int:
        target = max(int(getattr(self.analyzer, "max_clips", 0) or 0), 0)
        if target <= 0:
            return 0
        return max(target * 2, target + 3)

    async def _verify_and_finalize(
        self,
        aggregated: Dict[str, Any],
        transcript_parts: List[str],
        progress_callback=None,
    ) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        transcript_map = {
            f"part{i+1:02d}": path for i, path in enumerate(transcript_parts)
        }
        raw_candidates = aggregated.get("top_engaging_moments", []) or []
        reviewed = [
            self._prepare_candidate_for_review(deepcopy(moment), transcript_map)
            for moment in raw_candidates
        ]
        judge_batches = max(1, (len(reviewed) + self.judge_batch_size - 1) // self.judge_batch_size)
        logger.info(
            "🔍 verify_round_1: judging %s candidates in %s batches (batch_size=%s)",
            len(reviewed),
            judge_batches,
            self.judge_batch_size,
        )
        if progress_callback:
            progress_callback("Verifying standalone quality: judge batches...", 60)
        await self._apply_parallel_verification_batches(
            reviewed,
            batch_size=self.judge_batch_size,
            mode="judge",
            progress_callback=progress_callback,
            progress_start=60,
            progress_end=64,
        )

        selected: List[Dict[str, Any]] = []
        selected_source_keys: set[Tuple[Any, Any, Any]] = set()
        rejected_for_repair: List[Dict[str, Any]] = []
        verification_clips: List[Dict[str, Any]] = []
        for candidate in reviewed:
            if candidate["_passes_llm"] and candidate["_passes_deterministic"]:
                if self._has_excessive_overlap(selected, candidate):
                    candidate["verification_status"] = "rejected_overlap"
                    candidate["verification_notes"] = self._append_note(
                        candidate.get("verification_notes", ""),
                        "Rejected after verification due to excessive overlap with a higher-ranked clip.",
                    )
                    rejected_for_repair.append(candidate)
                    verification_clips.append(self._build_verification_clip_entry(candidate, "rejected"))
                    continue
                selected.append(candidate)
                selected_source_keys.add(self._candidate_source_key(candidate))
                verification_clips.append(self._build_verification_clip_entry(candidate, "kept"))
            else:
                rejected_for_repair.append(candidate)
                verification_clips.append(self._build_verification_clip_entry(candidate, "rejected"))

        target = min(getattr(self.analyzer, "max_clips", len(raw_candidates)), len(raw_candidates))
        repair_pass_used = bool(rejected_for_repair)
        repaired_candidates, failed_repairs = await self._apply_parallel_repairs(
            rejected_for_repair,
            transcript_map,
            progress_callback=progress_callback,
            progress_start=64,
            progress_end=68,
        )
        for candidate in failed_repairs:
            if candidate.get("_repair_planner_attempted"):
                candidate["verification_notes"] = (
                    candidate.get("_planner_reason")
                    or candidate.get("verification_notes", "")
                )
                verification_clips.append(
                    self._build_verification_clip_entry(candidate, "rejected")
                )

        rejudge_batches = (
            max(1, (len(repaired_candidates) + self.rejudge_batch_size - 1) // self.rejudge_batch_size)
            if repaired_candidates
            else 0
        )
        if repaired_candidates:
            logger.info(
                "🔁 verify_round_1: rejudging %s repaired candidates in %s batches (batch_size=%s)",
                len(repaired_candidates),
                rejudge_batches,
                self.rejudge_batch_size,
            )
            if progress_callback:
                progress_callback("Verifying standalone quality: rejudge batches...", 68)
        await self._apply_parallel_verification_batches(
            repaired_candidates,
            batch_size=self.rejudge_batch_size,
            mode="rejudge",
            progress_callback=progress_callback,
            progress_start=68,
            progress_end=69.5,
        )
        for repaired in repaired_candidates:
            if repaired["_passes_llm"] and repaired["_passes_deterministic"]:
                if self._has_excessive_overlap(selected, repaired, allow_repaired_overlap=True):
                    repaired["verification_status"] = "rejected_overlap"
                    repaired["verification_notes"] = self._append_note(
                        repaired.get("verification_notes", ""),
                        "Rejected after repair due to excessive overlap with a higher-ranked clip.",
                    )
                    verification_clips.append(self._build_verification_clip_entry(repaired, "rejected"))
                    continue
                selected.append(repaired)
                selected_source_keys.add(self._candidate_source_key(repaired))
                verification_clips.append(self._build_verification_clip_entry(repaired, "repaired"))
            else:
                verification_clips.append(self._build_verification_clip_entry(repaired, "rejected"))

        final_selected = selected[:target]
        final_selected_keys = {
            self._candidate_source_key(candidate)
            for candidate in final_selected
        }
        finalized_moments = []
        for rank, candidate in enumerate(final_selected, start=1):
            cleaned = {
                key: value
                for key, value in candidate.items()
                if not key.startswith("_")
            }
            whisper_source = candidate.get("_whisper_transcript_path")
            if whisper_source:
                cleaned["whisper_subtitle_source"] = whisper_source
            cleaned["rank"] = rank
            finalized_moments.append(cleaned)

        kept_count = sum(1 for clip in verification_clips if clip.get("decision") == "kept")
        repaired_count = sum(1 for clip in verification_clips if clip.get("decision") == "repaired")
        rejected_count = sum(1 for clip in verification_clips if clip.get("decision") == "rejected")
        logger.info(
            "📊 verify_round_1 complete: kept=%s repaired=%s rejected=%s final_selected=%s",
            kept_count,
            repaired_count,
            rejected_count,
            len(finalized_moments),
        )
        if progress_callback:
            progress_callback("Verifying standalone quality complete", 70)

        final_result = {
            "top_engaging_moments": finalized_moments,
            "total_moments": len(finalized_moments),
            "analysis_timestamp": datetime.now().isoformat() + "Z",
            "aggregation_criteria": "Agentic verification with standalone-first LLM review",
            "analysis_summary": {
                "verification_mode": "llm_standalone_first",
                "repair_pass_used": repair_pass_used,
                "selected_after_verification": len(finalized_moments),
            },
            "honorable_mentions": [],
        }
        verification_report = {
            "round": 1,
            "verification_mode": "llm_standalone_first",
            "repair_pass_used": repair_pass_used,
            "clips": self._annotate_selected_entries(
                self._dedupe_verification_entries(verification_clips),
                final_selected_keys,
            ),
            "analysis_timestamp": datetime.now().isoformat() + "Z",
        }
        return final_result, repair_pass_used, verification_report

    def _prepare_candidate_for_review(
        self,
        candidate: Dict[str, Any],
        transcript_map: Dict[str, str],
    ) -> Dict[str, Any]:
        candidate = self._normalize_candidate(candidate)
        part_name = candidate["timing"]["video_part"]
        transcript_path = candidate.get("_whisper_transcript_path") or transcript_map.get(part_name)
        transcript_entries = candidate.get("_whisper_transcript_entries")
        if transcript_entries is None:
            transcript_entries = self.analyzer.parse_srt_file(transcript_path) if transcript_path else []
        coverage_entries = []
        verification_context: Dict[str, str] = {
            "actual_clip_excerpt": "",
            "context_before": "",
            "context_after": "",
            "context_before_start_time": "",
            "context_after_end_time": "",
        }
        evidence_excerpt = ""
        if transcript_path:
            (
                coverage_entries,
                verification_context,
                evidence_excerpt,
            ) = self._build_clip_context(
                transcript_entries,
                candidate["timing"]["start_time"],
                candidate["timing"]["end_time"],
            )

        duration_ok, duration_seconds = self._check_duration(candidate)
        transcript_ok = bool(coverage_entries and evidence_excerpt.strip())
        candidate["timing"]["duration"] = f"{duration_seconds}s"
        candidate["duration_seconds"] = duration_seconds
        candidate["evidence_excerpt"] = evidence_excerpt
        if getattr(self.analyzer, "user_intent", None):
            candidate["intent_alignment_score"] = 0.5
        candidate["suggested_start_time"] = None
        candidate["suggested_end_time"] = None
        candidate["repair_strategy"] = "none"
        candidate["repair_diagnosis"] = "none"
        candidate["_repair_planner_attempted"] = False
        candidate["_repairable"] = None
        candidate["_planner_reason"] = None
        candidate["_rejudge_keep"] = None
        candidate["_rejudge_reason"] = None
        candidate["verification_notes"] = ""
        candidate["_judge_keep"] = None
        candidate["_judge_reason"] = None
        candidate["_passes_llm"] = False
        candidate["_passes_deterministic"] = duration_ok and transcript_ok
        candidate["_original_rank"] = candidate.get("rank", 9999)
        candidate["_transcript_path"] = transcript_path
        candidate["_transcript_entries"] = transcript_entries
        candidate["_verification_transcript_source"] = candidate.get(
            "_verification_transcript_source",
            "whisper" if candidate.get("_whisper_transcript_entries") else "original",
        )
        candidate["_verification_transcript_reason"] = candidate.get(
            "_verification_transcript_reason",
            "Whisper context transcript attached for this candidate"
            if candidate.get("_whisper_transcript_entries")
            else "Original transcript context retained for this candidate",
        )
        candidate["_verification_context"] = verification_context
        candidate["_coverage_entries"] = coverage_entries
        return candidate

    def _attach_whisper_context_transcripts(
        self,
        candidates: List[Dict[str, Any]],
        transcript_map: Dict[str, str],
        cache_dir: Path,
    ) -> None:
        if not candidates:
            return

        cache_dir.mkdir(parents=True, exist_ok=True)
        for candidate in candidates:
            normalized = self._normalize_candidate(candidate)
            part_name = normalized["timing"]["video_part"]
            transcript_path = transcript_map.get(part_name)
            if not transcript_path:
                continue
            transcript_entries = self.analyzer.parse_srt_file(transcript_path)
            should_generate, reason = self._should_generate_whisper_context(
                normalized,
                transcript_entries,
            )
            if not should_generate:
                logger.info(
                    "📝 Candidate '%s': using original transcript context (%s)",
                    normalized.get("title", ""),
                    reason,
                )
                candidate["_verification_transcript_source"] = "original"
                candidate["_verification_transcript_reason"] = reason
                continue
            whisper_entries, whisper_srt_path = self._generate_whisper_context_entries(
                normalized,
                transcript_path,
                cache_dir,
            )
            if not whisper_entries:
                continue
            logger.info(
                "📝 Candidate '%s': using Whisper context transcript (%s)",
                normalized.get("title", ""),
                reason,
            )
            candidate["_verification_transcript_source"] = "whisper"
            candidate["_verification_transcript_reason"] = reason
            candidate["_whisper_transcript_entries"] = whisper_entries
            candidate["_whisper_transcript_path"] = whisper_srt_path

    def _should_generate_whisper_context(
        self,
        candidate: Dict[str, Any],
        transcript_entries: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        if len(transcript_entries) < 2:
            return False, "not enough subtitle segments"

        clip_start = self.analyzer.time_to_seconds(candidate["timing"]["start_time"])
        clip_end = self.analyzer.time_to_seconds(candidate["timing"]["end_time"])
        window_start = max(0.0, clip_start - self.verification_context_before_seconds)
        window_end = clip_end + self.verification_context_after_seconds

        window_entries = []
        for entry in transcript_entries:
            start_seconds = self.analyzer.time_to_seconds(entry["start_time"])
            end_seconds = self.analyzer.time_to_seconds(entry["end_time"])
            if end_seconds > window_start and start_seconds < window_end:
                window_entries.append(entry)

        if len(window_entries) < 2:
            return False, "not enough subtitle segments in review window"

        overlap_pairs = 0
        for current, nxt in zip(window_entries, window_entries[1:]):
            current_end = self.analyzer.time_to_seconds(current["end_time"])
            next_start = self.analyzer.time_to_seconds(nxt["start_time"])
            if next_start < current_end:
                overlap_pairs += 1

        overlap_ratio = overlap_pairs / max(1, len(window_entries) - 1)
        if overlap_pairs >= 2 and overlap_ratio >= 0.2:
            return True, f"overlap ratio {overlap_ratio:.2f} across {overlap_pairs} adjacent subtitle pairs"
        return False, f"overlap ratio {overlap_ratio:.2f} across {overlap_pairs} adjacent subtitle pairs"

    def _generate_whisper_context_entries(
        self,
        candidate: Dict[str, Any],
        transcript_path: str,
        cache_dir: Path,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        media_path = self._find_media_for_transcript(transcript_path)
        if not media_path:
            return [], None

        clip_start = self.analyzer.time_to_seconds(candidate["timing"]["start_time"])
        clip_end = self.analyzer.time_to_seconds(candidate["timing"]["end_time"])
        window_start = max(0.0, clip_start - self.verification_context_before_seconds)
        window_end = clip_end + self.verification_context_after_seconds
        if window_end <= window_start:
            return [], None

        cache_stem = self._candidate_cache_stem(candidate)
        window_video = cache_dir / f"{cache_stem}.mp4"
        window_srt = cache_dir / f"{cache_stem}.srt"
        absolute_srt = cache_dir / f"{cache_stem}.absolute.srt"

        if not absolute_srt.exists():
            if not window_video.exists():
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-ss",
                            self._seconds_to_cli_time(window_start),
                            "-to",
                            self._seconds_to_cli_time(window_end),
                            "-i",
                            str(media_path),
                            "-c",
                            "copy",
                            str(window_video),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except Exception as exc:
                    logger.warning(f"Failed to extract Whisper context window for {candidate.get('title', '')}: {exc}")
                    return [], None

            if not window_srt.exists():
                success = run_whisper_cli(
                    str(window_video),
                    model_name=self.verification_whisper_model,
                    language=None,
                    output_format="srt",
                    output_dir=str(cache_dir),
                )
                if not success or not window_srt.exists():
                    return [], None

            relative_entries = self.analyzer.parse_srt_file(str(window_srt))
            absolute_entries = []
            for entry in relative_entries:
                start_seconds = window_start + self.analyzer.time_to_seconds(entry["start_time"])
                end_seconds = window_start + self.analyzer.time_to_seconds(entry["end_time"])
                absolute_entries.append(
                    {
                        "start_time": self._seconds_to_srt_time(start_seconds),
                        "end_time": self._seconds_to_srt_time(end_seconds),
                        "text": entry["text"],
                    }
                )
            if not absolute_entries:
                return [], None
            self._write_srt_entries(absolute_srt, absolute_entries)

        return self.analyzer.parse_srt_file(str(absolute_srt)), str(absolute_srt)

    def _candidate_cache_stem(self, candidate: Dict[str, Any]) -> str:
        part_name = candidate["timing"]["video_part"]
        start_time = candidate["timing"]["start_time"].replace(":", "-").replace(",", "-")
        end_time = candidate["timing"]["end_time"].replace(":", "-").replace(",", "-")
        return f"{part_name}_{start_time}_{end_time}"

    def _find_media_for_transcript(self, transcript_path: str) -> Optional[Path]:
        transcript = Path(transcript_path)
        for suffix in (".mp4", ".mkv", ".mov", ".webm", ".avi"):
            candidate = transcript.with_suffix(suffix)
            if candidate.exists():
                return candidate
        return None

    def _seconds_to_cli_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    def _seconds_to_srt_time(self, seconds: float) -> str:
        total_ms = int(round(max(0.0, seconds) * 1000))
        hours = total_ms // 3_600_000
        minutes = (total_ms % 3_600_000) // 60_000
        secs = (total_ms % 60_000) // 1000
        ms = total_ms % 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    def _write_srt_entries(self, output_path: Path, entries: List[Dict[str, Any]]) -> None:
        lines: List[str] = []
        for idx, entry in enumerate(entries, start=1):
            lines.extend(
                [
                    str(idx),
                    f"{entry['start_time']} --> {entry['end_time']}",
                    entry["text"],
                    "",
                ]
            )
        output_path.write_text("\n".join(lines), encoding="utf-8")

    async def _apply_parallel_verification_batches(
        self,
        candidates: List[Dict[str, Any]],
        batch_size: int,
        mode: str,
        progress_callback=None,
        progress_start: Optional[float] = None,
        progress_end: Optional[float] = None,
    ) -> None:
        if not candidates:
            return

        total_batches = max(1, (len(candidates) + batch_size - 1) // batch_size)
        semaphore = asyncio.Semaphore(self.max_parallel_judge_batches)
        completion_lock = asyncio.Lock()
        completed_batches = 0
        batch_results: List[Optional[List[Dict[str, Any]]]] = [None] * total_batches
        mode_label = "Judge" if mode == "judge" else "Rejudge"

        async def run_batch(batch_index: int, batch_candidates: List[Dict[str, Any]]) -> None:
            nonlocal completed_batches
            start = batch_index * batch_size
            end = start + len(batch_candidates)
            logger.info(
                "🧠 %s batch %s/%s launched: candidates %s-%s",
                mode_label,
                batch_index + 1,
                total_batches,
                start + 1,
                end,
            )
            async with semaphore:
                batch_started_at = perf_counter()
                results = await asyncio.to_thread(
                    self._run_single_verification_batch,
                    batch_candidates,
                    mode,
                )
                elapsed = perf_counter() - batch_started_at
            logger.info(
                "✅ %s batch %s/%s completed in %.1fs",
                mode_label,
                batch_index + 1,
                total_batches,
                elapsed,
            )
            batch_results[batch_index] = results
            if (
                progress_callback
                and progress_start is not None
                and progress_end is not None
                and total_batches > 0
            ):
                async with completion_lock:
                    completed_batches += 1
                    progress = progress_start + completed_batches * (progress_end - progress_start) / total_batches
                    completed_now = completed_batches
                progress_callback(
                    f"Verifying standalone quality: {mode_label.lower()} batch {completed_now}/{total_batches}...",
                    progress,
                )

        tasks = []
        for batch_index in range(total_batches):
            start = batch_index * batch_size
            end = min(start + batch_size, len(candidates))
            tasks.append(asyncio.create_task(run_batch(batch_index, candidates[start:end])))
            if batch_index < total_batches - 1 and self.judge_batch_launch_stagger_seconds > 0:
                await asyncio.sleep(self.judge_batch_launch_stagger_seconds)

        await asyncio.gather(*tasks)

        verification_results: List[Dict[str, Any]] = []
        for batch_index, result in enumerate(batch_results):
            if result is None:
                raise RuntimeError(f"{mode_label} batch {batch_index + 1} did not return results")
            verification_results.extend(result)
        for candidate, llm_verification in zip(candidates, verification_results):
            self._apply_llm_verification_result(candidate, llm_verification, mode=mode)

    async def _apply_parallel_repairs(
        self,
        candidates: List[Dict[str, Any]],
        transcript_map: Dict[str, str],
        progress_callback=None,
        progress_start: Optional[float] = None,
        progress_end: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not candidates:
            return [], []

        total_repairs = len(candidates)
        semaphore = asyncio.Semaphore(self.max_parallel_repairs)
        completion_lock = asyncio.Lock()
        completed_repairs = 0
        repair_results: List[Optional[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]] = [None] * total_repairs

        async def run_repair(repair_index: int, candidate: Dict[str, Any]) -> None:
            nonlocal completed_repairs
            logger.info(
                "🛠️ Repair task %s/%s launched for '%s'",
                repair_index + 1,
                total_repairs,
                candidate.get("title", "Untitled clip"),
            )
            async with semaphore:
                repair_started_at = perf_counter()
                repaired = await asyncio.to_thread(
                    self._attempt_boundary_repair,
                    candidate,
                    transcript_map,
                )
                elapsed = perf_counter() - repair_started_at
            logger.info(
                "✅ Repair task %s/%s completed in %.1fs for '%s'",
                repair_index + 1,
                total_repairs,
                elapsed,
                candidate.get("title", "Untitled clip"),
            )
            repair_results[repair_index] = (candidate, repaired)
            if (
                progress_callback
                and progress_start is not None
                and progress_end is not None
                and total_repairs > 0
            ):
                async with completion_lock:
                    completed_repairs += 1
                    progress = progress_start + completed_repairs * (progress_end - progress_start) / total_repairs
                    completed_now = completed_repairs
                progress_callback(
                    f"Verifying standalone quality: repair {completed_now}/{total_repairs}...",
                    progress,
                )

        tasks = []
        for repair_index, candidate in enumerate(candidates):
            tasks.append(asyncio.create_task(run_repair(repair_index, candidate)))
            if repair_index < total_repairs - 1 and self.repair_launch_stagger_seconds > 0:
                await asyncio.sleep(self.repair_launch_stagger_seconds)

        await asyncio.gather(*tasks)

        repaired_candidates: List[Dict[str, Any]] = []
        failed_candidates: List[Dict[str, Any]] = []
        for repair_index, result in enumerate(repair_results):
            if result is None:
                raise RuntimeError(f"Repair task {repair_index + 1} did not return a result")
            candidate, repaired = result
            if repaired is None:
                failed_candidates.append(candidate)
            else:
                repaired_candidates.append(repaired)
        return repaired_candidates, failed_candidates

    def _apply_llm_verification_result(
        self,
        candidate: Dict[str, Any],
        llm_verification: Dict[str, Any],
        mode: str,
    ) -> None:
        duration_seconds = candidate.get("duration_seconds", 0)
        passes_deterministic = bool(candidate.get("_passes_deterministic"))
        if mode == "judge":
            candidate["verification_status"] = (
                "verified"
                if llm_verification["keep"] and passes_deterministic
                else "rejected"
            )
            candidate["verification_notes"] = llm_verification["reason"]
            candidate["_judge_keep"] = llm_verification["keep"]
            candidate["_judge_reason"] = llm_verification["reason"]
            candidate["selection_confidence"] = llm_verification["standalone_score"]
            if getattr(self.analyzer, "user_intent", None):
                candidate["intent_alignment_score"] = llm_verification["intent_alignment_score"]
            candidate["repair_diagnosis"] = llm_verification.get("repair_diagnosis", "none")

            if not self._check_duration_deterministic(candidate):
                candidate["verification_notes"] = self._append_note(
                    candidate["verification_notes"],
                    f"Rejected by deterministic validation: duration {duration_seconds}s is out of range.",
                )
            if not candidate.get("_coverage_entries") or not candidate.get("evidence_excerpt", "").strip():
                candidate["verification_notes"] = self._append_note(
                    candidate["verification_notes"],
                    "Rejected by deterministic validation: transcript coverage is insufficient for this time range.",
                )
            candidate["_passes_llm"] = llm_verification["keep"]
        else:
            candidate["verification_status"] = (
                "repaired_verified"
                if llm_verification["keep"] and passes_deterministic
                else "repaired_rejected"
            )
            candidate["verification_notes"] = self._append_note(
                llm_verification["reason"],
                (
                    f"Boundary repair adjusted the clip to "
                    f"{candidate['timing']['start_time']} -> {candidate['timing']['end_time']}."
                ),
            )
            candidate["selection_confidence"] = llm_verification["standalone_score"]
            candidate["_rejudge_keep"] = llm_verification["keep"]
            candidate["_rejudge_reason"] = llm_verification["reason"]
            if getattr(self.analyzer, "user_intent", None):
                candidate["intent_alignment_score"] = llm_verification["intent_alignment_score"]
            candidate["_passes_llm"] = llm_verification["keep"]
            if not self._check_duration_deterministic(candidate):
                candidate["verification_notes"] = self._append_note(
                    candidate["verification_notes"],
                    f"Rejected by deterministic validation: duration {duration_seconds}s is out of range.",
                )
            if not candidate.get("_coverage_entries") or not candidate.get("evidence_excerpt", "").strip():
                candidate["verification_notes"] = self._append_note(
                    candidate["verification_notes"],
                    "Rejected by deterministic validation: transcript coverage is insufficient for this time range.",
                )

    def _check_duration_deterministic(self, candidate: Dict[str, Any]) -> bool:
        duration_ok, _ = self._check_duration(candidate)
        return duration_ok

    def _normalize_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        timing = candidate.get("timing") or {}
        video_part = (
            timing.get("video_part")
            or candidate.get("_source_video_part")
            or candidate.get("source_part")
            or "unknown"
        )
        start_time = timing.get("start_time") or candidate.get("start_time") or "00:00:00"
        end_time = timing.get("end_time") or candidate.get("end_time") or "00:00:00"
        duration_seconds = candidate.get("duration_seconds")
        if duration_seconds is None:
            try:
                duration_seconds = int(
                    self.analyzer.time_to_seconds(end_time)
                    - self.analyzer.time_to_seconds(start_time)
                )
            except Exception:
                duration_seconds = 0
        candidate["timing"] = {
            "video_part": video_part,
            "start_time": start_time,
            "end_time": end_time,
            "duration": f"{duration_seconds}s",
        }
        candidate.setdefault("summary", "")
        candidate.setdefault("engagement_details", {"engagement_level": "medium"})
        candidate.setdefault("why_engaging", "")
        candidate.setdefault("tags", [])
        return candidate

    def _build_clip_context(
        self,
        transcript_entries: List[Dict[str, Any]],
        start_time: str,
        end_time: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str], str]:
        start_seconds = self.analyzer.time_to_seconds(start_time)
        end_seconds = self.analyzer.time_to_seconds(end_time)
        overlap_indexes = []
        for idx, entry in enumerate(transcript_entries):
            entry_start = self.analyzer.time_to_seconds(entry["start_time"])
            entry_end = self.analyzer.time_to_seconds(entry["end_time"])
            if entry_end >= start_seconds and entry_start <= end_seconds:
                overlap_indexes.append(idx)

        if not overlap_indexes:
            return [], {
                "actual_clip_excerpt": "",
                "context_before": "",
                "context_after": "",
                "context_before_start_time": "",
                "context_after_end_time": "",
            }, ""

        overlap_entries = [transcript_entries[idx] for idx in overlap_indexes]
        context_before_entries = self._collect_context_entries(
            transcript_entries=transcript_entries,
            anchor_index=overlap_indexes[0],
            direction="before",
            max_segments=self.verification_context_before_segments,
            max_seconds=self.verification_context_before_seconds,
        )
        context_after_entries = self._collect_context_entries(
            transcript_entries=transcript_entries,
            anchor_index=overlap_indexes[-1],
            direction="after",
            max_segments=self.verification_context_after_segments,
            max_seconds=self.verification_context_after_seconds,
        )

        actual_clip_excerpt = "\n".join(
            f"[{entry['start_time']} --> {entry['end_time']}] {entry['text']}"
            for entry in overlap_entries
        )
        context_before = "\n".join(
            f"[{entry['start_time']} --> {entry['end_time']}] {entry['text']}"
            for entry in context_before_entries
        )
        context_after = "\n".join(
            f"[{entry['start_time']} --> {entry['end_time']}] {entry['text']}"
            for entry in context_after_entries
        )
        evidence_excerpt = " ".join(entry["text"] for entry in overlap_entries).strip()
        evidence_excerpt = re.sub(r"\s+", " ", evidence_excerpt)[:280]
        verification_context = {
            "actual_clip_excerpt": actual_clip_excerpt,
            "context_before": context_before,
            "context_after": context_after,
            "context_before_start_time": (
                self._normalize_time(context_before_entries[0]["start_time"])
                if context_before_entries
                else ""
            ),
            "context_after_end_time": (
                self._normalize_time(context_after_entries[-1]["end_time"])
                if context_after_entries
                else ""
            ),
        }
        return overlap_entries, verification_context, evidence_excerpt

    def _collect_context_entries(
        self,
        transcript_entries: List[Dict[str, Any]],
        anchor_index: int,
        direction: str,
        max_segments: int,
        max_seconds: float,
    ) -> List[Dict[str, Any]]:
        if max_segments <= 0 or max_seconds <= 0:
            return []

        if direction == "before":
            indexes = range(anchor_index - 1, -1, -1)
        else:
            indexes = range(anchor_index + 1, len(transcript_entries))

        collected: List[Dict[str, Any]] = []
        collected_seconds = 0.0
        for idx in indexes:
            entry = transcript_entries[idx]
            entry_start = self.analyzer.time_to_seconds(entry["start_time"])
            entry_end = self.analyzer.time_to_seconds(entry["end_time"])
            entry_duration = max(0.0, entry_end - entry_start)
            if collected and collected_seconds + entry_duration > max_seconds:
                break
            collected.append(entry)
            collected_seconds += entry_duration
            if len(collected) >= max_segments:
                break

        if direction == "before":
            collected.reverse()
        return collected

    def _build_verification_prompt(
        self,
        candidate: Dict[str, Any],
        verification_context: Dict[str, str],
    ) -> str:
        user_intent = getattr(self.analyzer, "user_intent", None)
        actual_clip_excerpt = verification_context.get("actual_clip_excerpt", "").strip()
        context_before = verification_context.get("context_before", "").strip()
        context_after = verification_context.get("context_after", "").strip()
        return f"""
Candidate:
- Title: {candidate.get('title', '')}
- Summary: {candidate.get('summary', '')}
- Time Range: {candidate['timing']['start_time']} -> {candidate['timing']['end_time']}
- Why engaging: {candidate.get('why_engaging', '')}
- User intent: {user_intent or "none"}

Actual clip transcript:
{actual_clip_excerpt}

Context before:
{context_before or "[none]"}

Context after:
{context_after or "[none]"}
"""

    def _run_llm_verification_single(
        self,
        candidate: Dict[str, Any],
        verification_context: Dict[str, str],
        mode: str = "judge",
    ) -> Dict[str, Any]:
        user_intent = getattr(self.analyzer, "user_intent", None)
        actual_clip_excerpt = verification_context.get("actual_clip_excerpt", "").strip()
        context_before = verification_context.get("context_before", "").strip()
        context_after = verification_context.get("context_after", "").strip()
        if not actual_clip_excerpt:
            return {
                "keep": False,
                "standalone_score": 0.0,
                "intent_alignment_score": 0.5,
                "reason": "No clip transcript available for standalone verification.",
                "repair_diagnosis": "missing_clip_transcript",
            }

        mode_guidance = self._verification_mode_guidance(mode)

        verification_prompt = f"""
You are verifying whether a candidate highlight clip is standalone.

A standalone clip must pass a strict user-experience bar.

The clip should feel satisfying to a new viewer who has not seen the rest of the video.
Reject clips that feel like incomplete excerpts, even if they contain an interesting moment.

Hard fail conditions:
- the clip starts in the middle of a topic, answer, argument, joke, or story
- the clip is missing the setup, question, trigger, or premise needed to understand it
- the clip ends before the answer, punchline, payoff, reaction, or conclusion fully lands
- the clip changes topic or cuts off while the speaker is still developing the same point
- the clip ends on the setup for the next question, example, or sub-argument instead of finishing the current one
- the clip ends with a line that clearly signals continuation, transition, or "more to come immediately"
- the viewer would likely feel "this started too late" or "this ended too early"

A strong standalone clip:
- is understandable without watching the rest of the source video
- includes enough setup, question, or trigger context when needed
- contains a complete mini-arc: setup -> development -> payoff/conclusion
- does not feel cut off at the start or end
- works as a shareable short-form segment on its own

Judge standalone quality based on the Actual clip transcript only.
Use Context before and Context after only as reference material to determine:
- whether the current clip starts too late
- whether the current clip ends too early
- whether expanding the boundaries could make it standalone

{mode_guidance}

Be conservative. If the clip is only partially understandable but still feels like it begins mid-topic
or ends before finishing the topic, set "keep" to false.
It is not enough that the core point is mostly understandable. The clip should also feel finished.

Return ONLY valid JSON:
{{
  "keep": true,
  "standalone_score": 0.82,
  "intent_alignment_score": 0.75,
  "reason": "Short explanation.",
  "repair_diagnosis": "none"
}}

Scoring guidance:
- 0.90-1.00: fully standalone, complete thought arc, clearly satisfying
- 0.70-0.89: mostly standalone, minor roughness but still feels complete
- 0.40-0.69: partially understandable but UX is weak or incomplete; usually reject
- 0.00-0.39: clearly not standalone; missing setup or missing payoff

When deciding "keep":
- prefer false over true when uncertain
- if the clip starts mid-topic, keep must be false
- if the clip ends before the topic/payoff finishes, keep must be false
- set repair_diagnosis to one of:
  - "bad_start"
  - "bad_end"
  - "bad_start_and_end"
  - "missing_premise"
  - "missing_payoff"
  - "not_fixable_content"
  - "none"

Candidate:
{self._build_verification_prompt(candidate, verification_context)}
"""
        try:
            response = self.analyzer.llm_client.simple_chat(
                verification_prompt,
                model=getattr(self.analyzer, "model", None),
            )
            parsed = self._extract_json_object(response)
            keep = bool(parsed.get("keep", True))
            standalone_score = self._clamp_score(parsed.get("standalone_score", 0.5))
            intent_alignment_score = self._clamp_score(
                parsed.get("intent_alignment_score", 0.5)
            )
            reason = str(parsed.get("reason", "LLM verification completed.")).strip()
            repair_diagnosis = str(parsed.get("repair_diagnosis", "none")).strip() or "none"
            return {
                "keep": keep,
                "standalone_score": standalone_score,
                "intent_alignment_score": intent_alignment_score,
                "reason": reason,
                "repair_diagnosis": repair_diagnosis,
            }
        except Exception as e:
            logger.warning(f"LLM verification failed, using safe fallback: {e}")
            return {
                "keep": True,
                "standalone_score": 0.5,
                "intent_alignment_score": 0.5,
                "reason": "LLM verification failed; kept candidate with neutral confidence fallback.",
                "repair_diagnosis": "none",
            }

    def _run_llm_verification_batch(
        self,
        candidates: List[Dict[str, Any]],
        batch_size: int,
        mode: str,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        results: List[Dict[str, Any]] = []
        for start in range(0, len(candidates), max(1, batch_size)):
            batch = candidates[start : start + max(1, batch_size)]
            results.extend(self._run_single_verification_batch(batch, mode=mode))
        return results

    def _run_single_verification_batch(
        self,
        batch: List[Dict[str, Any]],
        mode: str,
    ) -> List[Dict[str, Any]]:
        if not batch:
            return []
        if len(batch) == 1:
            candidate = batch[0]
            return [
                self._run_llm_verification_single(
                    candidate,
                    candidate.get("_verification_context", {}),
                    mode=mode,
                )
            ]

        batch_prompt = self._build_batched_verification_prompt(batch, mode=mode)
        try:
            response = self.analyzer.llm_client.simple_chat(
                batch_prompt,
                model=getattr(self.analyzer, "model", None),
            )
            parsed = self._extract_json_object(response)
            parsed_results = parsed.get("results") if isinstance(parsed, dict) else parsed
            if not isinstance(parsed_results, list) or len(parsed_results) != len(batch):
                raise ValueError("Batched verification response shape mismatch")
            results: List[Dict[str, Any]] = []
            for item in parsed_results:
                results.append(
                    {
                        "keep": bool(item.get("keep", True)),
                        "standalone_score": self._clamp_score(item.get("standalone_score", 0.5)),
                        "intent_alignment_score": self._clamp_score(
                            item.get("intent_alignment_score", 0.5)
                        ),
                        "reason": str(item.get("reason", "LLM verification completed.")).strip(),
                        "repair_diagnosis": str(item.get("repair_diagnosis", "none")).strip() or "none",
                    }
                )
            return results
        except Exception as e:
            logger.warning(f"Batched LLM verification failed, falling back to single verification: {e}")
            return [
                self._run_llm_verification_single(
                    candidate,
                    candidate.get("_verification_context", {}),
                    mode=mode,
                )
                for candidate in batch
            ]

    def _build_batched_verification_prompt(
        self,
        candidates: List[Dict[str, Any]],
        mode: str,
    ) -> str:
        prompt_parts = []
        for idx, candidate in enumerate(candidates, start=1):
            prompt_parts.append(
                f"Candidate #{idx}\n{self._build_verification_prompt(candidate, candidate.get('_verification_context', {}))}"
            )
        mode_guidance = self._verification_mode_guidance(mode)
        return f"""
You are verifying whether candidate highlight clips are standalone.

Apply the same strict standalone bar to every candidate independently.
Judge standalone quality based on the Actual clip transcript only.
Use Context before and Context after only as reference material to decide whether the current clip starts too late or ends too early.
{mode_guidance}

Return ONLY valid JSON in this shape:
{{
  "results": [
    {{
      "keep": true,
      "standalone_score": 0.82,
      "intent_alignment_score": 0.75,
      "reason": "Short explanation.",
      "repair_diagnosis": "none"
    }}
  ]
}}

The results array must have exactly {len(candidates)} items and preserve the same order as the candidates below.
For {mode} mode, each result should evaluate the current clip version exactly as shown.

{chr(10).join(prompt_parts)}
"""

    def _verification_mode_guidance(self, mode: str) -> str:
        if mode == "judge":
            return """
Mode: judge
- this is the first-pass diagnostic review
- be conservative
- identify whether boundaries are the problem
- if the clip has a real idea but the boundaries are wrong, reject it and classify the failure so repair can try to help
"""
        return """
Mode: rejudge
- this is the final shipping gate for repaired clips
- do not imagine further edits or possible future repairs
- decide whether this exact repaired clip is good enough to publish as-is
- if uncertain, reject
- reject if the clip still feels unfinished
- reject if the ending introduces the next question, example, or sub-argument rather than landing the current point
"""

    def _run_llm_repair_planner(
        self,
        candidate: Dict[str, Any],
        verification_context: Dict[str, str],
    ) -> Dict[str, Any]:
        actual_clip_excerpt = verification_context.get("actual_clip_excerpt", "").strip()
        context_before = verification_context.get("context_before", "").strip()
        context_after = verification_context.get("context_after", "").strip()
        context_before_start_time = verification_context.get("context_before_start_time", "").strip()
        context_after_end_time = verification_context.get("context_after_end_time", "").strip()
        if not actual_clip_excerpt:
            return {
                "repairable": False,
                "repair_strategy": "none",
                "suggested_start_time": None,
                "suggested_end_time": None,
                "reason": "No clip transcript available for repair planning.",
            }

        repair_prompt = f"""
You are repairing a candidate highlight clip that already failed standalone verification.

Your job is only to plan better boundaries.
Do not decide final keep/reject here.

Judge failed because:
- reason: {candidate.get('verification_notes', '')}
- diagnosis: {candidate.get('repair_diagnosis', 'none')}

Repair goal:
- include the missing setup if the clip starts too late
- include the missing ending if the clip ends too early
- preserve the core moment while making the clip feel complete to a new viewer
- prefer the smallest boundary changes that create a satisfying mini-arc
- choose an ending that lands the current point, not one that opens the next question, example, or sub-argument
- prefer a true stopping point over the first merely grammatical clause
- choose suggested_start_time and suggested_end_time from the timestamps shown in Actual clip transcript / Context before / Context after
- do not invent timestamps from the middle of a subtitle line

Important rules:
- if diagnosis is "bad_start", "bad_start_and_end", or "missing_premise", you must provide a non-null suggested_start_time
- if diagnosis is "bad_end", "bad_start_and_end", or "missing_payoff", you must provide a non-null suggested_end_time
- if Context before contains the missing setup, choose suggested_start_time from that context
- if Context after contains the missing payoff, choose suggested_end_time from that context
- do not stop at a line that obviously sets up the next thought if a better stopping point appears shortly after in the provided context
- if no nearby stopping point clearly makes the clip feel complete, set repairable to false

Available context boundary anchors:
- earliest Context before start_time: {context_before_start_time or "none"}
- latest Context after end_time: {context_after_end_time or "none"}

Return ONLY valid JSON:
{{
  "repairable": true,
  "repair_strategy": "expand_start",
  "suggested_start_time": "00:07:02",
  "suggested_end_time": "00:08:29",
  "reason": "Needs the setup question before the answer."
}}

Candidate:
- Title: {candidate.get('title', '')}
- Summary: {candidate.get('summary', '')}
- Current Time Range: {candidate['timing']['start_time']} -> {candidate['timing']['end_time']}

Actual clip transcript:
{actual_clip_excerpt}

Context before:
{context_before or "[none]"}

Context after:
{context_after or "[none]"}
"""
        try:
            response = self.analyzer.llm_client.simple_chat(
                repair_prompt,
                model=getattr(self.analyzer, "model", None),
            )
            parsed = self._extract_json_object(response)
            diagnosis = str(candidate.get("repair_diagnosis", "none")).strip() or "none"
            suggested_start_time = parsed.get("suggested_start_time")
            suggested_end_time = parsed.get("suggested_end_time")
            if (
                diagnosis in {"bad_start", "bad_start_and_end", "missing_premise"}
                and not suggested_start_time
            ):
                suggested_start_time = context_before_start_time or candidate["timing"]["start_time"]
            if (
                diagnosis in {"bad_end", "bad_start_and_end", "missing_payoff"}
                and not suggested_end_time
            ):
                suggested_end_time = context_after_end_time or candidate["timing"]["end_time"]
            return {
                "repairable": bool(parsed.get("repairable", False)),
                "repair_strategy": str(parsed.get("repair_strategy", "none")).strip() or "none",
                "suggested_start_time": suggested_start_time,
                "suggested_end_time": suggested_end_time,
                "reason": str(parsed.get("reason", "Boundary repair planned.")).strip(),
            }
        except Exception as e:
            logger.warning(f"LLM repair planning failed, using heuristic fallback: {e}")
            return {
                "repairable": False,
                "repair_strategy": "none",
                "suggested_start_time": None,
                "suggested_end_time": None,
                "reason": "LLM repair planner failed.",
            }

    def _attempt_boundary_repair(
        self,
        candidate: Dict[str, Any],
        transcript_map: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        if candidate.get("_passes_deterministic") is False and not candidate.get("_transcript_entries"):
            return None

        transcript_entries = candidate.get("_transcript_entries") or []
        if not transcript_entries:
            transcript_path = transcript_map.get(candidate["timing"]["video_part"])
            if not transcript_path:
                return None
            transcript_entries = self.analyzer.parse_srt_file(transcript_path)
        verification_notes = candidate.get("verification_notes", "")
        diagnosis = str(candidate.get("repair_diagnosis", "none")).strip() or "none"
        if diagnosis in {"none", "not_fixable_content", "missing_clip_transcript"}:
            return None

        logger.info(
            "🛠️ Repair planner: attempting boundary repair for '%s' (diagnosis=%s)",
            candidate.get("title", "Untitled clip"),
            diagnosis,
        )

        verification_context = {
            "actual_clip_excerpt": "",
            "context_before": "",
            "context_after": "",
            "context_before_start_time": "",
            "context_after_end_time": "",
        }
        if candidate["timing"]["start_time"] and candidate["timing"]["end_time"]:
            _, verification_context, _ = self._build_clip_context(
                transcript_entries,
                candidate["timing"]["start_time"],
                candidate["timing"]["end_time"],
            )

        repair_plan = self._run_llm_repair_planner(candidate, verification_context)
        candidate["_repair_planner_attempted"] = True
        candidate["_repairable"] = repair_plan.get("repairable")
        candidate["_planner_reason"] = repair_plan.get("reason")
        if not repair_plan.get("repairable"):
            logger.info(
                "❌ Repair planner: no usable repair for '%s' (%s)",
                candidate.get("title", "Untitled clip"),
                candidate.get("_planner_reason") or "planner marked clip as not repairable",
            )
            return None

        suggested_start = repair_plan.get("suggested_start_time")
        suggested_end = repair_plan.get("suggested_end_time")
        snapped_start, snapped_end = self._derive_repaired_window(
            candidate,
            transcript_entries,
            suggested_start,
            suggested_end,
        )
        if not snapped_start or not snapped_end:
            logger.info(
                "❌ Repair planner: failed to derive repaired window for '%s'",
                candidate.get("title", "Untitled clip"),
            )
            return None
        if (
            snapped_start == candidate["timing"]["start_time"]
            and snapped_end == candidate["timing"]["end_time"]
        ):
            logger.info(
                "❌ Repair planner: suggested window unchanged for '%s'",
                candidate.get("title", "Untitled clip"),
            )
            return None

        repaired = deepcopy(candidate)
        repaired["timing"]["start_time"] = snapped_start
        repaired["timing"]["end_time"] = snapped_end
        repaired["start_time"] = snapped_start
        repaired["end_time"] = snapped_end
        coverage_entries, verification_context, evidence_excerpt = self._build_clip_context(
            transcript_entries,
            snapped_start,
            snapped_end,
        )
        if not coverage_entries:
            logger.info(
                "❌ Repair planner: repaired window had no transcript coverage for '%s'",
                candidate.get("title", "Untitled clip"),
            )
            return None

        duration_ok, duration_seconds = self._check_duration(repaired)
        transcript_ok = bool(evidence_excerpt.strip())
        repaired["timing"]["duration"] = f"{duration_seconds}s"
        repaired["duration_seconds"] = duration_seconds
        repaired["evidence_excerpt"] = evidence_excerpt
        repaired["suggested_start_time"] = suggested_start
        repaired["suggested_end_time"] = suggested_end
        repaired["repair_strategy"] = repair_plan.get("repair_strategy", "none")
        repaired["repair_diagnosis"] = candidate.get("repair_diagnosis", "none")
        repaired["_repair_planner_attempted"] = True
        repaired["_repairable"] = repair_plan.get("repairable")
        repaired["_planner_reason"] = repair_plan.get("reason")
        repaired["_passes_deterministic"] = duration_ok and transcript_ok
        repaired["_repair_attempted"] = True
        repaired["_transcript_entries"] = transcript_entries
        repaired["_repair_source_rank"] = candidate.get("_original_rank", candidate.get("rank"))
        repaired["_old_start_time"] = candidate["timing"]["start_time"]
        repaired["_old_end_time"] = candidate["timing"]["end_time"]
        repaired["_verification_context"] = verification_context
        repaired["_coverage_entries"] = coverage_entries
        logger.info(
            "✅ Repair planner: produced repaired window %s -> %s for '%s'",
            snapped_start,
            snapped_end,
            candidate.get("title", "Untitled clip"),
        )
        return repaired

    def _derive_repaired_window(
        self,
        candidate: Dict[str, Any],
        transcript_entries: List[Dict[str, Any]],
        suggested_start: Any,
        suggested_end: Any,
    ) -> Tuple[Optional[str], Optional[str]]:
        try:
            current_start = self.analyzer.time_to_seconds(candidate["timing"]["start_time"])
            current_end = self.analyzer.time_to_seconds(candidate["timing"]["end_time"])
        except Exception:
            return None, None

        overlap_indexes = []
        for idx, entry in enumerate(transcript_entries):
            entry_start = self.analyzer.time_to_seconds(entry["start_time"])
            entry_end = self.analyzer.time_to_seconds(entry["end_time"])
            if entry_end >= current_start and entry_start <= current_end:
                overlap_indexes.append(idx)
        if not overlap_indexes:
            return None, None

        start_idx = overlap_indexes[0]
        end_idx = overlap_indexes[-1]

        snapped_start = self._snap_time_to_entry_start(transcript_entries, suggested_start)
        snapped_end = self._snap_time_to_entry_end(transcript_entries, suggested_end)

        if snapped_start is None:
            repair_start_idx = max(0, start_idx - self.repair_max_context_segments)
            snapped_start = transcript_entries[repair_start_idx]["start_time"]
        if snapped_end is None:
            repair_end_idx = min(
                len(transcript_entries) - 1,
                end_idx + self.repair_max_context_segments,
            )
            snapped_end = transcript_entries[repair_end_idx]["end_time"]

        snapped_start = self._normalize_time(snapped_start)
        snapped_end = self._normalize_time(snapped_end)

        return snapped_start, snapped_end

    def _snap_time_to_entry_start(
        self,
        transcript_entries: List[Dict[str, Any]],
        suggested_time: Any,
    ) -> Optional[str]:
        if not suggested_time:
            return None
        try:
            target = self.analyzer.time_to_seconds(self._normalize_time(str(suggested_time)))
        except Exception:
            return None
        candidates = []
        for entry in transcript_entries:
            try:
                entry_seconds = self.analyzer.time_to_seconds(entry["start_time"])
            except Exception:
                continue
            if entry_seconds <= target:
                candidates.append((target - entry_seconds, entry["start_time"]))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _snap_time_to_entry_end(
        self,
        transcript_entries: List[Dict[str, Any]],
        suggested_time: Any,
    ) -> Optional[str]:
        if not suggested_time:
            return None
        try:
            target = self.analyzer.time_to_seconds(self._normalize_time(str(suggested_time)))
        except Exception:
            return None
        candidates = []
        for entry in transcript_entries:
            try:
                entry_seconds = self.analyzer.time_to_seconds(entry["end_time"])
            except Exception:
                continue
            if entry_seconds >= target:
                candidates.append((entry_seconds - target, entry["end_time"]))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _normalize_time(self, value: str) -> str:
        return value.replace(".", ",").split(",")[0]

    def _extract_json_object(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        loose = re.search(r"\{.*\}", response, re.DOTALL)
        if loose:
            return json.loads(loose.group(0))

        raise json.JSONDecodeError("No JSON object found in response", response, 0)

    def _check_duration(self, candidate: Dict[str, Any]) -> Tuple[bool, int]:
        try:
            start_seconds = self.analyzer.time_to_seconds(candidate["timing"]["start_time"])
            end_seconds = self.analyzer.time_to_seconds(candidate["timing"]["end_time"])
            duration_seconds = int(max(0.0, end_seconds - start_seconds))
        except Exception:
            return False, 0
        return 30 <= duration_seconds <= 240, duration_seconds

    def _has_excessive_overlap(
        self,
        selected: List[Dict[str, Any]],
        candidate: Dict[str, Any],
        allow_repaired_overlap: bool = False,
    ) -> bool:
        candidate_start = self.analyzer.time_to_seconds(candidate["timing"]["start_time"])
        candidate_end = self.analyzer.time_to_seconds(candidate["timing"]["end_time"])
        candidate_duration = max(0.0, candidate_end - candidate_start)
        for existing in selected:
            existing_start = self.analyzer.time_to_seconds(existing["timing"]["start_time"])
            existing_end = self.analyzer.time_to_seconds(existing["timing"]["end_time"])
            existing_duration = max(0.0, existing_end - existing_start)
            overlap = max(0.0, min(candidate_end, existing_end) - max(candidate_start, existing_start))
            shorter_duration = max(1.0, min(candidate_duration, existing_duration))
            overlap_ratio = overlap / shorter_duration
            if not (
                overlap > self.overlap_threshold_seconds
                or overlap_ratio > self.overlap_threshold_ratio
            ):
                continue

            if allow_repaired_overlap and self._adds_substantial_new_material(
                existing_start,
                existing_end,
                candidate_start,
                candidate_end,
            ):
                continue

            if not self._is_editorially_redundant(existing, candidate, overlap, shorter_duration):
                continue

            if allow_repaired_overlap and overlap <= 60.0:
                continue

            return True
        return False

    def _adds_substantial_new_material(
        self,
        existing_start: float,
        existing_end: float,
        candidate_start: float,
        candidate_end: float,
    ) -> bool:
        unique_before = max(0.0, existing_start - candidate_start)
        unique_after = max(0.0, candidate_end - existing_end)
        total_unique = unique_before + unique_after
        return total_unique >= 30.0 or unique_before >= 15.0 or unique_after >= 15.0

    def _is_editorially_redundant(
        self,
        existing: Dict[str, Any],
        candidate: Dict[str, Any],
        overlap_seconds: float,
        shorter_duration: float,
    ) -> bool:
        if existing.get("timing", {}).get("video_part") != candidate.get("timing", {}).get("video_part"):
            return False

        existing_text = self._editorial_text(existing)
        candidate_text = self._editorial_text(candidate)
        similarity = self._token_similarity(existing_text, candidate_text)
        if similarity >= 0.33:
            return True

        return overlap_seconds / max(1.0, shorter_duration) >= 0.85

    def _editorial_text(self, candidate: Dict[str, Any]) -> str:
        return " ".join(
            str(candidate.get(key, ""))
            for key in ["title", "summary", "why_engaging"]
        ).lower()

    def _token_similarity(self, left: str, right: str) -> float:
        left_tokens = self._tokenize_editorial_text(left)
        right_tokens = self._tokenize_editorial_text(right)
        if not left_tokens or not right_tokens:
            return 0.0
        intersection = left_tokens & right_tokens
        union = left_tokens | right_tokens
        return len(intersection) / max(1, len(union))

    def _tokenize_editorial_text(self, text: str) -> set[str]:
        normalized = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", text)
        chunks = [chunk for chunk in normalized.split() if chunk]
        tokens: set[str] = set()
        for chunk in chunks:
            if re.search(r"[\u4e00-\u9fff]", chunk):
                tokens.update(self._char_ngrams(chunk))
            else:
                if len(chunk) >= 3:
                    tokens.add(chunk)
        return tokens

    def _char_ngrams(self, text: str, size: int = 2) -> set[str]:
        if len(text) <= size:
            return {text}
        return {text[i : i + size] for i in range(len(text) - size + 1)}

    def _clamp_score(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.5

    def _append_note(self, base: str, extra: str) -> str:
        if not base:
            return extra
        return f"{base} {extra}"

    def _create_workflow(self, run_id: str, transcript_parts: List[str]) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "mode": "engaging_moments",
            "agentic_analysis": True,
            "provider": getattr(self.analyzer, "provider", "unknown"),
            "model": getattr(self.analyzer, "model", None),
            "language": getattr(self.analyzer, "language", "zh"),
            "user_intent": getattr(self.analyzer, "user_intent", None),
            "total_parts": len(transcript_parts),
            "started_at": datetime.now().isoformat() + "Z",
            "stages": [],
        }

    def _add_stage(self, workflow: Dict[str, Any], stage: str, status: str, **extra: Any) -> None:
        entry = {
            "stage": stage,
            "status": status,
            "timestamp": datetime.now().isoformat() + "Z",
        }
        entry.update({k: v for k, v in extra.items() if v is not None})
        workflow["stages"].append(entry)

    def _count_raw_candidates(self, highlights_files: List[str]) -> int:
        total = 0
        for file_path in highlights_files:
            try:
                data = json.loads(Path(file_path).read_text(encoding="utf-8"))
            except Exception:
                continue
            total += len(data.get("engaging_moments", []))
        return total

    def _candidate_source_key(self, candidate: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        return (
            candidate.get("timing", {}).get("video_part"),
            candidate.get("_old_start_time") or candidate.get("timing", {}).get("start_time"),
            candidate.get("_old_end_time") or candidate.get("timing", {}).get("end_time"),
        )

    def _build_verification_clip_entry(self, candidate: Dict[str, Any], decision: str) -> Dict[str, Any]:
        entry = {
            "title": candidate.get("title", ""),
            "original_rank": candidate.get("_original_rank", candidate.get("rank")),
            "decision": decision,
            "reason": candidate.get("verification_notes", ""),
            "verification_transcript_source": candidate.get("_verification_transcript_source"),
            "verification_transcript_reason": candidate.get("_verification_transcript_reason"),
            "verification_transcript_path": candidate.get("_whisper_transcript_path"),
            "judge_keep": candidate.get("_judge_keep"),
            "judge_reason": candidate.get("_judge_reason"),
            "standalone_score": candidate.get("selection_confidence"),
            "verification_status": candidate.get("verification_status"),
            "start_time": candidate.get("timing", {}).get("start_time"),
            "end_time": candidate.get("timing", {}).get("end_time"),
            "video_part": candidate.get("timing", {}).get("video_part"),
            "evidence_excerpt": candidate.get("evidence_excerpt", ""),
            "repair_diagnosis": candidate.get("repair_diagnosis"),
            "repair_planner_attempted": candidate.get("_repair_planner_attempted"),
            "repairable": candidate.get("_repairable"),
            "planner_reason": candidate.get("_planner_reason"),
            "rejudge_keep": candidate.get("_rejudge_keep"),
            "rejudge_reason": candidate.get("_rejudge_reason"),
            "selected_in_final_top_n": False,
        }
        if candidate.get("_old_start_time") or candidate.get("_old_end_time"):
            entry["old_start_time"] = candidate.get("_old_start_time")
            entry["old_end_time"] = candidate.get("_old_end_time")
        return entry

    def _dedupe_verification_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
        priority = {"kept": 3, "repaired": 2, "rejected": 1}
        for entry in entries:
            key = (
                entry.get("video_part"),
                entry.get("old_start_time") or entry.get("start_time"),
                entry.get("old_end_time") or entry.get("end_time"),
            )
            existing = deduped.get(key)
            if existing is None or priority.get(entry["decision"], 0) >= priority.get(existing["decision"], 0):
                deduped[key] = entry
        return list(deduped.values())

    def _annotate_selected_entries(
        self,
        entries: List[Dict[str, Any]],
        final_selected_keys: set[Tuple[Any, Any, Any]],
    ) -> List[Dict[str, Any]]:
        for entry in entries:
            key = (
                entry.get("video_part"),
                entry.get("old_start_time") or entry.get("start_time"),
                entry.get("old_end_time") or entry.get("end_time"),
            )
            entry["selected_in_final_top_n"] = key in final_selected_keys
        return entries
