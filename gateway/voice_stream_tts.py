"""
Voice Stream TTS — Early acknowledgment + chunked TTS streaming for Discord voice channels.

Solves the 45-90s silence gap between user speech and TTS output by:
  Part A: Playing a short acknowledgment within ~3s of voice input receipt
  Part B: Streaming TTS sentence-by-sentence as the agent generates response text

This module is self-contained so it can be patched into the gateway with minimal edits to run.py.
"""

import asyncio
import json
import logging
import os
import random
import re
import tempfile
import threading
import uuid
from typing import Optional, Callable

logger = logging.getLogger("gateway.voice_stream_tts")

# ─── Early Acknowledgment Phrases ───────────────────────────────────────────

_ACK_PHRASES = [
    "on it",
    "let me check",
    "looking into that",
    "one moment",
    "got it",
    "sure thing",
    "thinking...",
    "right away",
]

# ─── Sentence Splitting ─────────────────────────────────────────────────────

# Split on sentence-ending punctuation followed by whitespace, or double newlines.
_SENTENCE_BOUNDARY_RE = re.compile(
    r'(?<=[.!?])\s+|(?<=\n)\s*\n'
)

# Minimum chars before we flush a fragment as a "sentence"
_MIN_SENTENCE_LEN = 30

# If buffer exceeds this on a timeout, flush it even without a sentence boundary
_FLUSH_THRESHOLD = 150

# ─── Markdown stripping (lightweight version) ───────────────────────────────

_MD_CODE_BLOCK = re.compile(r'```[\s\S]*?```')
_MD_LINK = re.compile(r'\[([^\]]*)\]\([^)]*\)')
_MD_URL = re.compile(r'https?://\S+')
_MD_BOLD = re.compile(r'\*\*([^*]+)\*\*')
_MD_ITALIC = re.compile(r'\*([^*]+)\*')
_MD_INLINE_CODE = re.compile(r'`([^`]*)`')
_MD_HEADER = re.compile(r'^#{1,6}\s+', re.MULTILINE)
_MD_LIST_ITEM = re.compile(r'^\s*[-*+]\s+', re.MULTILINE)
_MD_HR = re.compile(r'^---+\s*$', re.MULTILINE)
_MD_EXCESS_NL = re.compile(r'\n{3,}')
_THINK_BLOCK = re.compile(r'</?think[^>]*>', re.DOTALL)


def _strip_markdown_for_tts(text: str) -> str:
    """Remove markdown formatting before speech synthesis."""
    text = _THINK_BLOCK.sub('', text)
    text = _MD_CODE_BLOCK.sub(' ', text)
    text = _MD_LINK.sub(r'\1', text)
    text = _MD_URL.sub('', text)
    text = _MD_BOLD.sub(r'\1', text)
    text = _MD_ITALIC.sub(r'\1', text)
    text = _MD_INLINE_CODE.sub(r'\1', text)
    text = _MD_HEADER.sub('', text)
    text = _MD_LIST_ITEM.sub('', text)
    text = _MD_HR.sub('', text)
    text = _MD_EXCESS_NL.sub('\n\n', text)
    return text.strip()


def _split_sentences(text: str) -> list:
    """Split text into sentences, respecting minimum length."""
    if not text or not text.strip():
        return []

    raw_parts = _SENTENCE_BOUNDARY_RE.split(text)
    sentences = []
    buffer = ""

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        buffer = (buffer + " " + part).strip() if buffer else part
        if len(buffer) >= _MIN_SENTENCE_LEN:
            sentences.append(buffer)
            buffer = ""

    # Remaining buffer becomes its own sentence if non-trivial
    if buffer and len(buffer.strip()) >= 10:
        sentences.append(buffer.strip())

    return sentences


# ─── Part A: Early Acknowledgment ───────────────────────────────────────────

async def send_early_ack(
    adapter,
    guild_id: int,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Generate and play a short TTS acknowledgment in the voice channel.

    Fire-and-forget: errors are logged but never propagated.
    """
    try:
        if not adapter or not hasattr(adapter, 'is_in_voice_channel'):
            return
        if not adapter.is_in_voice_channel(guild_id):
            return

        phrase = random.choice(_ACK_PHRASES)
        audio_path = None
        actual_path = None

        try:
            from tools.tts_tool import text_to_speech_tool

            audio_path = os.path.join(
                tempfile.gettempdir(), "hermes_voice",
                f"tts_ack_{uuid.uuid4().hex[:8]}.mp3",
            )
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)

            result_json = await asyncio.to_thread(
                text_to_speech_tool, text=phrase, output_path=audio_path
            )
            result = json.loads(result_json)
            actual_path = result.get("file_path", audio_path)

            if result.get("success") and os.path.isfile(actual_path):
                await adapter.play_in_voice_channel(guild_id, actual_path)
            else:
                logger.debug("Early ack TTS failed: %s", result.get("error"))
        finally:
            for p in {audio_path, actual_path} - {None}:
                try:
                    os.unlink(p)
                except OSError:
                    pass
    except Exception as e:
        logger.debug("Early ack error (non-fatal): %s", e)


# ─── Part B: VoiceStreamTTS ─────────────────────────────────────────────────

class VoiceStreamTTS:
    """Buffers streaming text, splits into sentences, and TTS-plays each in a Discord VC.

    Thread-safe: text arrives from the synchronous agent loop (via feed_text()),
    TTS generation + playback happens on the async event loop.

    Usage:
        streamer = VoiceStreamTTS(adapter, guild_id, loop)
        # From agent callback (sync thread):
        streamer.feed_text("Here is the first part of my response.")
        streamer.feed_text(" And here is the second sentence. Done!")
        streamer.finish()
        # On the async loop, the streamer auto-plays sentences as they arrive.
    """

    def __init__(
        self,
        adapter,
        guild_id: int,
        loop: asyncio.AbstractEventLoop,
        max_text_length: int = 4000,
    ):
        self._adapter = adapter
        self._guild_id = guild_id
        self._loop = loop
        self._max_text = max_text_length

        self._text_buffer = ""
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._total_chars_sent = 0
        self._spoken_sentences = set()

        # Track audio paths for cleanup
        self._audio_paths = []
        self._audio_paths_lock = threading.Lock()

        # Start the consumer task on the event loop
        self._sentence_queue = asyncio.Queue()
        self._consumer_task = asyncio.ensure_future(
            self._consume_sentences(), loop=loop
        )

    def feed_text(self, text: str) -> None:
        """Add streaming text delta. Called from the synchronous agent loop thread."""
        if self._done.is_set() or not text:
            return

        with self._lock:
            self._text_buffer += text

            # Check if we have enough for at least one sentence
            sentences = _split_sentences(self._text_buffer)
            if len(sentences) > 1:
                # Send all but the last (which may be incomplete)
                for s in sentences[:-1]:
                    self._enqueue_sentence(s)
                self._text_buffer = sentences[-1]
            elif len(sentences) == 1 and len(self._text_buffer) >= _FLUSH_THRESHOLD:
                # Buffer is long enough to flush even without clean boundary
                self._enqueue_sentence(sentences[0])
                self._text_buffer = ""

    def feed_segment_break(self) -> None:
        """Flush whatever is in the buffer as a sentence. Called on segment breaks."""
        with self._lock:
            if self._text_buffer.strip():
                cleaned = _strip_markdown_for_tts(self._text_buffer.strip())
                if cleaned:
                    self._enqueue_sentence(cleaned)
                self._text_buffer = ""

    def finish(self) -> None:
        """Signal that no more text is coming. Flushes remaining buffer."""
        with self._lock:
            if self._text_buffer.strip():
                cleaned = _strip_markdown_for_tts(self._text_buffer.strip())
                if cleaned and len(cleaned) >= 10:
                    self._enqueue_sentence(cleaned)
                self._text_buffer = ""
        self._done.set()
        # Sentinel None to wake the consumer
        asyncio.run_coroutine_threadsafe(self._sentence_queue.put(None), self._loop)

    def cancel(self) -> None:
        """Cancel immediately without flushing."""
        self._done.set()
        asyncio.run_coroutine_threadsafe(self._sentence_queue.put(None), self._loop)

    @property
    def chars_streamed(self) -> int:
        return self._total_chars_sent

    def _enqueue_sentence(self, sentence: str) -> None:
        """Add a sentence to the async queue for TTS processing."""
        if not sentence or len(sentence.strip()) < 5:
            return

        # Dedup check (skip near-identical sentences)
        normalized = sentence.strip().lower()
        for existing in self._spoken_sentences:
            if normalized in existing or existing in normalized:
                return
        self._spoken_sentences.add(normalized)

        self._total_chars_sent += len(sentence)
        # run_coroutine_threadsafe is safe from sync threads
        try:
            asyncio.run_coroutine_threadsafe(
                self._sentence_queue.put(sentence), self._loop
            )
        except RuntimeError:
            pass  # loop may be closed during shutdown

    async def _consume_sentences(self) -> None:
        """Async consumer: takes sentences from queue, generates TTS, plays in VC."""
        try:
            while True:
                sentence = await self._sentence_queue.get()
                if sentence is None:
                    # Sentinel — we're done
                    break

                if not sentence.strip():
                    continue

                await self._tts_and_play(sentence)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("VoiceStreamTTS consumer error: %s", e, exc_info=True)
        finally:
            await self._cleanup_audio_files()

    async def _tts_and_play(self, sentence: str) -> None:
        """Generate TTS for a sentence and play it in the voice channel."""
        audio_path = None
        actual_path = None

        try:
            if not self._adapter or not self._adapter.is_in_voice_channel(self._guild_id):
                return

            from tools.tts_tool import text_to_speech_tool

            audio_path = os.path.join(
                tempfile.gettempdir(), "hermes_voice",
                f"tts_stream_{uuid.uuid4().hex[:8]}.mp3",
            )
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)

            result_json = await asyncio.to_thread(
                text_to_speech_tool, text=sentence[:1000], output_path=audio_path
            )
            result = json.loads(result_json)
            actual_path = result.get("file_path", audio_path)

            if result.get("success") and os.path.isfile(actual_path):
                # play_in_voice_channel already queues playback (waits for current)
                await self._adapter.play_in_voice_channel(self._guild_id, actual_path)
            else:
                logger.debug("Stream TTS failed for sentence: %s", result.get("error"))

        except Exception as e:
            logger.debug("Stream TTS+play error (non-fatal): %s", e)
        finally:
            # Clean up audio files after playback
            for p in {audio_path, actual_path} - {None}:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    async def _cleanup_audio_files(self) -> None:
        """Ensure any remaining temp files are cleaned up."""
        with self._audio_paths_lock:
            for p in self._audio_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
            self._audio_paths.clear()

    async def wait_finished(self, timeout: float = 5.0) -> None:
        """Wait for the consumer task to finish (with timeout)."""
        if self._consumer_task and not self._consumer_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._consumer_task), timeout=timeout
                )
            except asyncio.TimeoutError:
                self._consumer_task.cancel()
