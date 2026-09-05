#!/usr/bin/env python3
"""
Speaker Identification via Embedding Similarity

Maps diarization SPEAKER_XX labels to real names by comparing speaker
embeddings against short reference audio clips.

Usage:
    Place reference WAV/MP3 files in a directory:
        references/
            Host.wav
            Guest_Alice.wav
    Then pass --speaker-references references/ to the CLI.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".m4a", ".ogg"})


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SpeakerIdentifier:
    """
    Maps SPEAKER_XX labels from diarization to real names.

    Reference audio:
    - 5–30 seconds of clean, single-speaker speech
    - Filename stem becomes the speaker name (e.g. Host.wav → "Host")
    - Any format whisperx.load_audio supports (.wav, .mp3, .flac, .m4a, .ogg)

    Matching:
    - Embeds each reference clip via the diarization pipeline
    - Compares against per-speaker embeddings returned by diarization
    - Assigns a real name only if cosine similarity ≥ threshold
    """

    def __init__(self, references_dir: str, threshold: float = 0.7):
        self.references_dir = Path(references_dir)
        self.threshold = threshold
        # name → embedding vector
        self.reference_embeddings: Dict[str, np.ndarray] = {}

    def load_references(self, diarize_pipeline) -> None:
        """
        Extract embeddings from all audio files in references_dir.

        Reuses the already-loaded diarize_pipeline so we don't load a second
        model.  Assumes each file contains a single speaker.
        """
        import whisperx  # available whenever diarize_pipeline exists

        ref_files = [
            f for f in sorted(self.references_dir.iterdir())
            if f.suffix.lower() in _AUDIO_EXTENSIONS
        ]

        if not ref_files:
            logger.warning(f"⚠️  No audio files found in {self.references_dir} — name mapping disabled")
            return

        logger.info(f"🎙️  Loading {len(ref_files)} reference speaker(s) from {self.references_dir.name}/")

        for ref_file in ref_files:
            name = ref_file.stem
            try:
                audio = whisperx.load_audio(str(ref_file))
                result = diarize_pipeline(audio, return_embeddings=True)

                if not isinstance(result, tuple):
                    logger.warning(f"⚠️  No embeddings returned for {ref_file.name}, skipping")
                    continue

                _, ref_embeddings = result
                if not ref_embeddings:
                    logger.warning(f"⚠️  Empty embeddings for {ref_file.name}, skipping")
                    continue

                # Reference clips should be single-speaker; take the first
                first_speaker = next(iter(ref_embeddings))
                self.reference_embeddings[name] = np.array(ref_embeddings[first_speaker])
                logger.info(f"  ✅ {name}")

            except Exception as e:
                logger.warning(f"⚠️  Failed to load reference {ref_file.name}: {e}")

        logger.info(f"🎙️  {len(self.reference_embeddings)} reference speaker(s) ready")

    def map_speakers(self, speaker_embeddings: Dict[str, list]) -> Dict[str, str]:
        """
        Match SPEAKER_XX → real name via cosine similarity.

        Returns a partial mapping — only speakers whose best match meets
        the similarity threshold are included.  Unmatched speakers keep
        their SPEAKER_XX label (caller handles the fallback).
        """
        if not self.reference_embeddings or not speaker_embeddings:
            return {}

        mapping: Dict[str, str] = {}

        for speaker_id, embedding in speaker_embeddings.items():
            emb = np.array(embedding)
            best_name: Optional[str] = None
            best_sim = -1.0

            for ref_name, ref_emb in self.reference_embeddings.items():
                sim = _cosine_similarity(emb, ref_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_name = ref_name

            if best_name and best_sim >= self.threshold:
                mapping[speaker_id] = best_name
                logger.info(f"  🏷️  {speaker_id} → {best_name} (sim={best_sim:.3f})")
            else:
                logger.info(f"  ❓  {speaker_id} unmatched (best: {best_name}, sim={best_sim:.3f})")

        return mapping