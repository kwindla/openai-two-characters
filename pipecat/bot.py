#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Dict
import json
import os
import sys
from dotenv import load_dotenv
from loguru import logger

import uvicorn


from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecatcloud.agent import (
    DailySessionArguments,
    SessionArguments,
    WebSocketSessionArguments,
)
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

from character_processor import CharacterTagger, CharacterGate, TTSSegmentSequencer

load_dotenv(override=True)
logger.remove()
logger.add(sys.stderr, level="DEBUG")


async def main(transport: BaseTransport):
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    tts_narrator = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini-tts",
        voice="ballad",
        instructions="British voice - tone: warm, formal; pacing: medium-fast, clear pronunciation; style: posh, public school, RP, received pronunciation; emotion: friendly.",
    )

    tts_character = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini-tts",
        voice="sage",
        instructions="American voice - tone: high-pitched, sweet; pacing: gentle, curious; emotion: softly excited.",
    )

    context = OpenAILLMContext(
        [
            {
                "role": "system",
                "content": """We are going to make up adventure stories together!

The stories will be read aloud. Keep sentences short. Use only plain text.

You are a more experienced story teller than I am. You will help me with ideas and do most of the storytelling. 

There is always a main character in the story. The main character is an eleven year old girl named Rosamund. Rosamund is very precocious and friendly, but she is also slightly grumpy!

Our stories should be as imaginative as possible, full of magical elements, and slightly dark.

Examples:

- In a fog‑shrouded harbor town where every public clock stops at midnight, a curious girl discovers that her grandfather the clockmaker has built a clock that can rewind a single hour of time.

- One very hot summer day, the town reservoir dries up and reveals the spires of a formerly underwater library. The library is filled with books in a mysterious language.

- In a seaside village, fishermen haul up metal kraken parts stamped with mysterious runes. A mechanically gifted girl rebuilds the creature to defend her home—but discovers it was originally designed to protect something on the ocean floor.

Lean into elements that are unexpected, quirky, and unusual. For example, animal friends should not only be able to talk, but should wear strange costumes and have interesting personalities. A chipmunk who wears boxing gloves and a top hat. A tiny, tiny cat with jade earings and a monocle. 

Any time you come to a pause in the story, prompt the user to help you continue the story. Be specific in your prompts. Ask questions like, "What do you think should happen next to these spurious and questionable characters?" or "If you were in this situation, which of course it's impossible that you ever would have gotten yourself into, how would you extricate yourself?"

Any time you describe a scene, paint it in vivid detail. Use sensory details to help the user visualize the scene. Always include unexpected elements like surreal objects, magical creatures with a punk rock aesthetic, or fish-out-of-water tertiary characters. 

When you tell the story, perform as two different voices.

Voice AA is the narrator and all secondary characters. The narrator is British. He uses posh, boarding-school, RP Britishisms and revels in throwing creative wording and mildly archaic constructions into his narration.

Voice BB is the main character, Rosamund. Rosamund is American and speaks in a contemporary American idiom.

Format the story to separate the parts for the two voices. Use two tags:
  - AA for the narrator and all secondary characters
  - BB for the main character, Rosamund

For example:

AA
Rosamund work up early because it was Saturday. She thought to herself

BB
I hope that the friendly owl comes back to visit today.

AA
As you might have guessed, Rosamund recently met a friendly owl. The owl's name was Hoot. Well, actually the owl is an ancient, wise, and magical owl with a name that no human being could ever possibly pronounce correctly. When Rosamund asked the owl her name, she said Hoo-ooo-oot. Rosamund, of course, being a very polite girl, said

BB
I'm very pleased to meet you, Hoo-oot.
""",
            },
            {
                "role": "user",
                "content": "Introduce the narrator, then have Rosamund introduce herself. Keep the introductions short - just two sentences. Then ask what kind of adventure the user wants to make up.",
            },
        ],
    )

    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            CharacterTagger(),
            ParallelPipeline(
                [CharacterGate("AA"), tts_narrator],
                [CharacterGate("BB"), tts_character],
            ),
            TTSSegmentSequencer(),
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        idle_timeout_secs=20,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected: {client}")
        # Kick off the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected: {client}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


#
# ---- Functions to run the bot. ----
#
# In a production application the logic here could be separated
# out into utility modules.
#


# Run the bot in the cloud. Pipecat Cloud or your hosting infrastructure calls this
# function with either Twilio or Daily session arguments.
async def bot(args: SessionArguments):
    try:
        if isinstance(args, WebSocketSessionArguments):
            logger.info("Starting WebSocket bot")

            start_data = args.websocket.iter_text()
            await start_data.__anext__()
            call_data = json.loads(await start_data.__anext__())
            stream_sid = call_data["start"]["streamSid"]
            transport = FastAPIWebsocketTransport(
                websocket=args.websocket,
                params=FastAPIWebsocketParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    add_wav_header=False,
                    vad_enabled=True,
                    vad_analyzer=SileroVADAnalyzer(),
                    vad_audio_passthrough=True,
                    serializer=TwilioFrameSerializer(stream_sid),
                ),
            )
        elif isinstance(args, DailySessionArguments):
            logger.info("Starting Daily bot")
            transport = DailyTransport(
                args.room_url,
                args.token,
                "Respond bot",
                DailyParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    transcription_enabled=False,
                    vad_enabled=True,
                    vad_analyzer=SileroVADAnalyzer(),
                    vad_audio_passthrough=True,
                ),
            )

        await main(transport)
        logger.info("Bot process completed")
    except Exception as e:
        logger.exception(f"Error in bot process: {str(e)}")
        raise


# Run the bot locally. This is useful for testing and development.
def local():
    try:
        app = FastAPI()

        # Store connections by pc_id
        pcs_map: Dict[str, SmallWebRTCConnection] = {}

        ice_servers = ["stun:stun.l.google.com:19302"]
        app.mount("/client", SmallWebRTCPrebuiltUI)

        @app.get("/", include_in_schema=False)
        async def root_redirect():
            return RedirectResponse(url="/client/")

        @app.post("/api/offer")
        async def offer(request: dict, background_tasks: BackgroundTasks):
            pc_id = request.get("pc_id")

            if pc_id and pc_id in pcs_map:
                pipecat_connection = pcs_map[pc_id]
                logger.info(f"Reusing existing connection for pc_id: {pc_id}")
                await pipecat_connection.renegotiate(
                    sdp=request["sdp"],
                    type=request["type"],
                    restart_pc=request.get("restart_pc", False),
                )
                return pipecat_connection.get_answer()
            else:
                pipecat_connection = SmallWebRTCConnection(ice_servers)
                await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

                @pipecat_connection.event_handler("closed")
                async def handle_disconnected(
                    webrtc_connection: SmallWebRTCConnection,
                ):
                    logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
                    pcs_map.pop(webrtc_connection.pc_id, None)

            transport = SmallWebRTCTransport(
                webrtc_connection=pipecat_connection,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    vad_analyzer=SileroVADAnalyzer(),
                ),
            )
            background_tasks.add_task(main, transport)

            answer = pipecat_connection.get_answer()
            # Updating the peer connection inside the map
            pcs_map[answer["pc_id"]] = pipecat_connection

            return answer

        uvicorn.run(app, host="0.0.0.0", port=7860)

    except Exception as e:
        logger.exception(f"Error in local bot process: {str(e)}")
        raise


if __name__ == "__main__":
    local()
