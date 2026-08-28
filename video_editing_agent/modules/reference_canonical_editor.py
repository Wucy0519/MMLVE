"""
Module 2.5 — Canonical reference pre-editing for cross-scene consistency.

Before keyframe anchor editing, each instruction's saved source reference frame
is edited once according to that instruction. The result is reused as a
consistency guide when editing first frames across scenes.
"""

from __future__ import annotations

import logging
import os

from video_editing_agent.clients.base import ModelApiClientBase
from video_editing_agent.config import AgentConfig
from video_editing_agent.schemas.instructions import EditAction, EntityInstructionSet
from video_editing_agent.utils.mask_utils import (
    build_inpaint_edit_directives,
    entity_ref_canonical_path,
    entity_ref_mask_path,
    entity_ref_multiview_path,
    entity_ref_src_path,
    mask_has_content,
    save_before_after_entity_reference,
)

logger = logging.getLogger(__name__)


class ReferenceCanonicalEditor:
    """Pre-edit entity reference source frames into canonical consistency guides."""

    def __init__(
        self,
        config: AgentConfig,
        api_client: ModelApiClientBase,
    ) -> None:
        self.config = config
        self.api_client = api_client

    async def run(self, entity_instru: EntityInstructionSet) -> dict[str, str]:
        """Edit each instruction's saved source reference frame once."""
        ref_dir = os.path.join(self.config.workspace_dir, "entity_refs")
        os.makedirs(ref_dir, exist_ok=True)
        canonical_paths: dict[str, str] = {}

        logger.info("ReferenceCanonicalEditor.run started — %d instructions", len(entity_instru.instructions))
        for instr in entity_instru.instructions:
            multiview_path = entity_ref_multiview_path(ref_dir, instr.instruction_id)
            out_path = entity_ref_canonical_path(ref_dir, instr.instruction_id)
            if os.path.exists(multiview_path) and os.path.exists(out_path):
                canonical_paths[instr.instruction_id] = out_path
                logger.debug(
                    "Front-view canonical already present for %s — skip legacy edit",
                    instr.instruction_id,
                )
                continue

            src_path = entity_ref_src_path(ref_dir, instr.instruction_id)
            mask_path = entity_ref_mask_path(ref_dir, instr.instruction_id)

            if not os.path.exists(src_path):
                logger.debug("No source ref for %s — skip canonical edit", instr.instruction_id)
                continue
            if not mask_has_content(mask_path):
                logger.debug("No reference mask for %s — skip canonical edit", instr.instruction_id)
                continue

            try:
                if instr.action == EditAction.DELETE:
                    self._build_delete_identification(instr, src_path, mask_path, out_path)
                else:
                    await self._edit_one(instr, src_path, mask_path, out_path)
                canonical_paths[instr.instruction_id] = out_path
            except Exception as exc:
                logger.warning(
                    "Canonical reference edit failed for %s: %s",
                    instr.instruction_id,
                    exc,
                )

        logger.info(
            "ReferenceCanonicalEditor.run done — %d canonical refs",
            len(canonical_paths),
        )
        return canonical_paths

    def _build_delete_identification(
        self,
        instr,
        src_path: str,
        mask_path: str,
        out_path: str,
    ) -> str:
        """Build left-right card: original entity vs empty removed panel."""
        save_before_after_entity_reference(
            src_path,
            mask_path,
            out_path,
            edited_frame_path=None,
            instruction_id=instr.instruction_id,
            entity_id=instr.entity_id,
            action=instr.action.value,
            subject_features=instr.subject_features,
        )
        logger.info("Delete-target comparison ref saved: %s", out_path)
        return out_path

    async def _edit_one(
        self,
        instr,
        src_path: str,
        mask_path: str,
        out_path: str,
    ) -> str:
        """Inpaint reference frame, then save original-vs-edited left-right card."""
        edit_directives = build_inpaint_edit_directives([
            {
                "color_name": "colored",
                "subject_features": instr.subject_features,
                "edit_prompt": instr.edit_prompt,
            },
        ])

        temp_path = out_path + ".fullframe.tmp.png"
        try:
            await self.api_client.masked_inpaint(
                image_path=src_path,
                mask_path=mask_path,
                edit_directives=edit_directives,
                output_path=temp_path,
                ref_image_path=None,
                consistency_ref_paths=None,
                preserve_frame_structure=False,
            )
            save_before_after_entity_reference(
                src_path,
                mask_path,
                out_path,
                edited_frame_path=temp_path,
                instruction_id=instr.instruction_id,
                entity_id=instr.entity_id,
                action=instr.action.value,
                subject_features=instr.subject_features,
            )
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as exc:
                    logger.warning("Failed to remove temp canonical frame %s: %s", temp_path, exc)

        logger.info("Canonical before/after reference saved: %s", out_path)
        return out_path
