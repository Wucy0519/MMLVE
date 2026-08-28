"""
Module 1 — Instruction parsing and asset preparation.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from video_editing_agent.clients.base import ModelApiClientBase
from video_editing_agent.config import AgentConfig
from video_editing_agent.schemas.instructions import (
    EntityInstruction,
    EntityInstructionSet,
    TimeCondition,
)

from video_editing_agent.utils.json_utils import (
    merge_instructions_one_per_entity,
    normalize_instruction_instance_scope,
)
from video_editing_agent.utils.temporal_utils import (
    enrich_all_instruction_appearance_cues,
    normalize_all_referential_time_conditions,
)

logger = logging.getLogger(__name__)

class InstructionParser:
    """Instruction & Asset Agent (Module 1)."""

    def __init__(
        self,
        config: AgentConfig,
        api_client: ModelApiClientBase,
    ) -> None:
        self.config = config
        self.api_client = api_client

    async def run(self, user_prompt: str) -> EntityInstructionSet:
        """Execute the full instruction parsing pipeline."""
        try:
            logger.info("InstructionParser.run started")

            rewrite = await self.api_client.rewrite_user_prompt(user_prompt)
            rewritten_prompt = rewrite.get("rewritten_prompt") or user_prompt
            clarifications = rewrite.get("clarifications") or []
            rewrite_success_criteria = rewrite.get("success_criteria_prompts") or []
            if clarifications:
                for note in clarifications:
                    logger.info("Clarification: %s", note)

            raw = await self._parse_with_llm(rewritten_prompt)
            raw = normalize_all_referential_time_conditions(
                raw,
                source_prompt=f"{user_prompt}\n{rewritten_prompt}",
            )
            raw = enrich_all_instruction_appearance_cues(
                raw,
                source_prompt=f"{user_prompt}\n{rewritten_prompt}",
            )
            raw = merge_instructions_one_per_entity(raw)
            raw = normalize_instruction_instance_scope(raw)
            instructions = self._dicts_to_entities(raw)
            instructions = self._attach_success_criteria(
                instructions,
                rewrite_success_criteria,
            )
            instructions = [self._clear_ref_image_fields(i) for i in instructions]
            resolved = await self._resolve_conflicts(instructions)
            resolved = [self._clear_ref_image_fields(i) for i in resolved]

            result = EntityInstructionSet(
                source_prompt=user_prompt,
                rewritten_prompt=rewritten_prompt,
                clarifications=clarifications,
                instructions=resolved,
            )
            result.save(self.config.entity_instru_path)
            logger.info(
                "InstructionParser.run done — %d instructions → %s",
                len(resolved),
                self.config.entity_instru_path,
            )
            return result

        except Exception as exc:
            logger.error("InstructionParser.run failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Instruction parsing failed: {exc}") from exc

    async def _parse_with_llm(self, user_prompt: str) -> List[Dict]:
        """Call LLM to produce structured instruction JSON."""
        response = await self.api_client.parse_instructions(user_prompt)
        return response.get("instructions", [])

    def _dicts_to_entities(self, raw: List[Dict]) -> List[EntityInstruction]:
        """Convert API dicts to typed EntityInstruction objects (one per entity)."""
        entities: List[EntityInstruction] = []
        for i, item in enumerate(raw):
            iid = item.get("instruction_id") or f"instr_{i + 1:03d}"
            eid = item.get("entity_id") or f"entity_{i + 1:02d}"
            entities.append(
                EntityInstruction(
                    instruction_id=iid,
                    entity_id=eid,
                    subject_features=item.get("subject_features", ""),
                    edit_prompt=item.get("edit_prompt", ""),
                    time_condition=TimeCondition.from_dict(
                        item.get("time_condition", {"condition_type": "event"})
                    ),
                    appearance_time_hint=item.get("appearance_time_hint", "") or "",
                    success_criteria_prompt=item.get("success_criteria_prompt", ""),
                    target_instance_scope=str(
                        item.get("target_instance_scope", "single") or "single"
                    ),
                    needs_ref_image=False,
                    ref_subject=None,
                    ref_image_path=None,
                )
            )
        return entities

    @staticmethod
    def _attach_success_criteria(
        instructions: List[EntityInstruction],
        rewrite_success_criteria: List[Dict],
    ) -> List[EntityInstruction]:
        """Attach rewrite-stage QA prompts to parsed instructions by order."""
        for idx, instr in enumerate(instructions):
            if (instr.success_criteria_prompt or "").strip():
                continue
            if idx >= len(rewrite_success_criteria):
                instr.success_criteria_prompt = InstructionParser._default_success_criteria(instr)
                continue
            item = rewrite_success_criteria[idx] or {}
            prompt = str(item.get("success_criteria_prompt", "")).strip()
            instr.success_criteria_prompt = prompt or InstructionParser._default_success_criteria(instr)
        return instructions

    @staticmethod
    def _default_success_criteria(instr: EntityInstruction) -> str:
        """Fallback keyframe QA prompt when rewrite output is incomplete."""
        return (
            f"Judge whether the edited keyframe successfully applies the requested edit "
            f"to the target \"{instr.subject_features}\": {instr.edit_prompt}. "
            "The target edit must be clearly visible, natural, and photorealistic. "
            "Unrelated background, camera framing, and other subjects should remain as close as possible "
            "to the original frame. Fail if the target is unchanged, the wrong subject is edited, "
            "the frame is cropped/recomposed, or major artifacts are introduced."
        )

    @staticmethod
    def _clear_ref_image_fields(instr: EntityInstruction) -> EntityInstruction:
        """Disable workspace/ref_images T2I assets — entity_refs only."""
        instr.needs_ref_image = False
        instr.ref_subject = None
        instr.ref_image_path = None
        return instr

    async def _resolve_conflicts(
        self,
        instructions: List[EntityInstruction],
    ) -> List[EntityInstruction]:
        """Temporal conflict resolution."""
        raw = [i.to_dict() for i in instructions]
        resolved_raw = await self.api_client.resolve_temporal_conflicts(raw)
        return [EntityInstruction.from_dict(d) for d in resolved_raw]
