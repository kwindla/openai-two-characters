from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    EndFrame,
    SystemFrame,
)
from loguru import logger
from dataclasses import dataclass
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
import re


@dataclass
class CharacterTagFrame(Frame):
    """Frame that indicates the character of the following text"""

    character: str


@dataclass
class NextCharacterSequenceFrame(Frame):
    pass


class CharacterTagger(FrameProcessor):
    """Frame processor to remove single-token character tags from the LLM
    output stream, buffer text segments, and emit text segments with character tags."""

    @dataclass
    class Segment:
        character: str
        text: str
        buffered: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_character = "AA"
        self.segments: list[CharacterTagger.Segment] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            # Don't automatically push this frame
            self.segments = []
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            # Don't automatically push this frame
            await self.flush_character_segment()
            return

        if isinstance(frame, NextCharacterSequenceFrame):
            # We expect this frame to come upstream to us from the TTSSegmentSequencer
            await self.flush_character_segment()

        if isinstance(frame, LLMTextFrame):
            match = re.match(r"(.*)(AA|BB)(.*)", frame.text)
            if match:
                pre_text = match.group(1)
                if pre_text:
                    await self.push_text(pre_text)
                character = match.group(2)
                if character and character != self.current_character:
                    self.current_character = character
                    await self.create_segment(character)
                post_text = match.group(3)
                if post_text:
                    await self.push_text(post_text)
            else:
                await self.push_text(frame)
            return

        await self.push_frame(frame, direction)

    async def create_segment(self, character: str):
        should_buffer = len(self.segments) > 0
        logger.debug(f"Creating segment: {character}, should_buffer: {should_buffer}")
        self.segments.append(
            CharacterTagger.Segment(character=character, text="", buffered=should_buffer)
        )
        if not should_buffer:
            await self.push_frame(CharacterTagFrame(character=character))
            await self.push_frame(LLMFullResponseStartFrame())

    async def push_text(self, text_or_frame: str | LLMTextFrame):
        # We expect to always get a character tag at the start of a response. We prompt
        # the LLM to try to make that happen. But, of course, it might not. So if there was
        # no initial character tag, we might need to create a segment here.
        if not self.segments:
            await self.create_segment("AA")
        frame = (
            text_or_frame
            if isinstance(text_or_frame, LLMTextFrame)
            else LLMTextFrame(text=text_or_frame)
        )
        if not self.segments[-1].buffered:
            await self.push_frame(frame)
        else:
            self.segments[-1].text += frame.text

    async def flush_character_segment(self):
        if not self.segments:
            return
        segment = self.segments.pop(0)
        if not segment.buffered:
            await self.push_frame(LLMFullResponseEndFrame())
            return
        await self.push_frame(CharacterTagFrame(character=segment.character))
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(text=segment.text))
        await self.push_frame(LLMFullResponseEndFrame())


class CharacterGate(FrameProcessor):
    """Frame processor that opens and closes a pipeline based on character tags.
    This directs the character-specific LLM response segments to the correct TTS element."""

    def __init__(self, character, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.character = character
        self.open = False

    def _should_passthrough_frame(self, frame):
        if self.open:
            return True
        return isinstance(frame, (EndFrame, SystemFrame))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, CharacterTagFrame):
            if frame.character == self.character:
                self.open = True
            else:
                self.open = False

        if self._should_passthrough_frame(frame):
            await self.push_frame(frame, direction)


class CharacterRetagger(FrameProcessor):
    def __init__(self, character, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.character = character

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            await self.push_frame(frame)
            await self.push_frame(LLMTextFrame(text=f"{self.character}\n"))
        else:
            await self.push_frame(frame, direction)


class TTSSegmentSequencer(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseEndFrame):
            await self.push_frame(NextCharacterSequenceFrame(), direction=FrameDirection.UPSTREAM)
        await self.push_frame(frame, direction)
