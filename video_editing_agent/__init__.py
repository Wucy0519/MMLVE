"""
Multi-shot real-scene long video editing agent.

Five pipeline modules orchestrated by :class:`VideoEditingAgent`:

1. :class:`~video_editing_agent.modules.instruction_parser.InstructionParser`
2. :class:`~video_editing_agent.modules.video_grounder.VideoGrounder`
3. :class:`~video_editing_agent.modules.keyframe_editor.KeyframeEditor`
4. :class:`~video_editing_agent.modules.video_propagator.VideoPropagator`
5. :class:`~video_editing_agent.modules.video_assembler.VideoAssembler`
"""

from video_editing_agent.agent import VideoEditingAgent

__all__ = ["VideoEditingAgent"]
__version__ = "0.1.0"
