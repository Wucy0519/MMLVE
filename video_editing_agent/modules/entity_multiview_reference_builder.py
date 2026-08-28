"""
Module 2.25 — Front-view entity reference builder.

After ``entity_keyframe_appearances.json``, for each instruction:
  1. VLM-select reference keyframes (quality, orientation, temporal coverage).
  2. Synthesize one front-view reference (matching source art style) with edit → QA → retry loop.
  3. For modify/add: edit the front-view reference per instruction with edit → QA → retry loop.
  4. Persist bundle under ``entity_refs/``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

from video_editing_agent.clients.base import ModelApiClientBase
from video_editing_agent.config import AgentConfig
from video_editing_agent.schemas.entity_keyframe_appearances import (
    EntityKeyframeAppearanceSet,
    EntityKeyframeRecord,
    KeyframeEntityAppearance,
)
from video_editing_agent.schemas.instructions import EditAction, EntityInstruction, EntityInstructionSet
from video_editing_agent.utils.ffmpeg_utils import probe_duration
from video_editing_agent.utils.mask_utils import entity_ref_canonical_path, entity_ref_multiview_path
from video_editing_agent.utils.multiview_qa_utils import build_multiview_qa_avoid_operations
from video_editing_agent.utils.multiview_ref_utils import (
    REFERENCE_GRID_COLS,
    appearances_catalog_for_selection,
    format_keyframe_notes,
    resolve_selected_appearances,
    save_keyframe_grid,
    save_multiview_entity_refs,
)
from video_editing_agent.utils.workspace_checkpoints import is_entity_multiview_refs_complete

logger = logging.getLogger(__name__)

MAX_MULTIVIEW_QA_RETRIES = 1


class EntityMultiviewReferenceBuilder:
    """Build front-view entity references from keyframe sightings."""

    def __init__(self, config: AgentConfig, api_client: ModelApiClientBase) -> None:
        self.config = config
        self.api_client = api_client

    async def run(
        self,
        entity_instru: EntityInstructionSet,
        appearance_set: EntityKeyframeAppearanceSet,
    ) -> Dict[str, str]:
        """Generate front-view entity refs for all instructions with sightings."""
        if is_entity_multiview_refs_complete(self.config, entity_instru):
            logger.info("Checkpoint: skip entity front-view refs — already complete")
            return self._load_existing_paths(self.config, entity_instru)

        ref_dir = os.path.join(self.config.workspace_dir, "entity_refs")
        os.makedirs(ref_dir, exist_ok=True)
        canonical_paths: Dict[str, str] = {}
        video_duration_sec = self._video_duration_sec()

        for instr in entity_instru.instructions:
            record = appearance_set.record_for_instruction(instr.instruction_id)
            if record is None or not record.appearances:
                logger.warning(
                    "No keyframe appearances for %s — skip front-view refs",
                    instr.instruction_id,
                )
                continue

            try:
                out = await self._build_one(
                    ref_dir,
                    instr,
                    record,
                    video_duration_sec=video_duration_sec,
                )
                if out:
                    canonical_paths[instr.instruction_id] = out
            except Exception as exc:
                logger.warning(
                    "Front-view ref build failed for %s: %s — "
                    "falling back to best keyframe as before-ref",
                    instr.instruction_id,
                    exc,
                    exc_info=True,
                )
                # Fallback: use the best keyframe image directly as the
                # before-reference (canonical) path so downstream video
                # editing can still proceed with at least a before-ref.
                fallback_path = self._save_fallback_ref(ref_dir, instr, record)
                if fallback_path:
                    canonical_paths[instr.instruction_id] = fallback_path

        # ── Update subject_features with VLM-described authoritative
        # descriptions from synth_sheet.png ──────────────────────────────
        await self._update_subject_features_from_synth_sheets(
            ref_dir, entity_instru,
        )

        logger.info(
            "EntityMultiviewReferenceBuilder done — %d front-view refs in %s",
            len(canonical_paths),
            ref_dir,
        )
        return canonical_paths

    def _video_duration_sec(self) -> float:
        try:
            if os.path.exists(self.config.source_video_path):
                return float(probe_duration(self.config.source_video_path))
        except Exception as exc:
            logger.warning("Could not probe video duration: %s", exc)
        return 0.0

    def _save_fallback_ref(
        self,
        ref_dir: str,
        instr: EntityInstruction,
        record: EntityKeyframeRecord,
    ) -> Optional[str]:
        """Save the best keyframe as a fallback before-ref when synthesis fails."""
        try:
            from video_editing_agent.utils.multiview_ref_utils import (
                ensure_square_reference_file,
                save_keyframe_grid,
            )
            # Pick the highest-confidence appearance.
            appearances = sorted(
                record.appearances,
                key=lambda a: float(a.confidence or 0.0),
                reverse=True,
            )
            if not appearances:
                return None
            best = appearances[0]
            if not best.keyframe_path or not os.path.exists(best.keyframe_path):
                return None
            fallback_path = entity_ref_canonical_path(ref_dir, instr.instruction_id)
            shutil.copy2(best.keyframe_path, fallback_path)
            ensure_square_reference_file(fallback_path)
            logger.info(
                "Saved fallback before-ref for %s from keyframe %s",
                instr.instruction_id,
                best.keyframe_path,
            )
            return fallback_path
        except Exception as exc:
            logger.warning(
                "Fallback ref save failed for %s: %s",
                instr.instruction_id,
                exc,
            )
            return None

    async def _update_subject_features_from_synth_sheets(
        self,
        ref_dir: str,
        entity_instru: EntityInstructionSet,
    ) -> None:
        """Use VLM to describe each entity from its synth_sheet.png, then
        replace the original (coarse) subject_features with the detailed
        description as the authoritative identity description.

        The synth_sheet.png is the front-view synthesis sheet produced by
        _build_one(). After all entities have their sheets generated, this
        method iterates over them and asks the VLM for a detailed visual
        description. The updated entity_instru is then saved back to disk
        so downstream modules (keyframe detection, video editing) use the
        authoritative description.
        """
        updated_count = 0
        descriptions: Dict[str, str] = {}

        for instr in entity_instru.instructions:
            synth_path = os.path.join(
                ref_dir, f"{instr.instruction_id}_front_work", "synth_sheet.png",
            )
            if not os.path.exists(synth_path):
                logger.warning(
                    "%s: synth_sheet.png not found — skip description update",
                    instr.instruction_id,
                )
                continue

            try:
                description = await self.api_client.describe_entity_from_synthesis_sheet(
                    synth_sheet_path=synth_path,
                    instruction_id=instr.instruction_id,
                    entity_id=instr.entity_id,
                    original_subject_features=instr.subject_features,
                )
            except Exception as exc:
                logger.warning(
                    "%s: VLM description failed — keeping original subject_features: %s",
                    instr.instruction_id,
                    exc,
                )
                continue

            if not description or len(description.strip()) < 20:
                logger.warning(
                    "%s: VLM description too short — keeping original subject_features",
                    instr.instruction_id,
                )
                continue

            original = instr.subject_features
            instr.subject_features = description.strip()
            descriptions[instr.instruction_id] = {
                "original_subject_features": original,
                "authoritative_subject_features": instr.subject_features,
                "synth_sheet_path": synth_path,
            }
            updated_count += 1
            logger.info(
                "%s: subject_features updated from synth_sheet description (%d → %d chars)",
                instr.instruction_id,
                len(original),
                len(instr.subject_features),
            )

        if updated_count > 0:
            # Save the updated entity_instru back to disk
            entity_instru.save(self.config.entity_instru_path)
            logger.info(
                "Updated %d/%d entity subject_features from synth_sheet descriptions → %s",
                updated_count,
                len(entity_instru.instructions),
                self.config.entity_instru_path,
            )
            # Also save a sidecar with before/after for auditing
            sidecar_path = os.path.join(ref_dir, "subject_features_updates.json")
            try:
                with open(sidecar_path, "w", encoding="utf-8") as fh:
                    json.dump(descriptions, fh, indent=2, ensure_ascii=False)
            except OSError:
                pass

    async def _build_one(
        self,
        ref_dir: str,
        instr: EntityInstruction,
        record: EntityKeyframeRecord,
        *,
        video_duration_sec: float,
    ) -> Optional[str]:
        existing = entity_ref_canonical_path(ref_dir, instr.instruction_id)
        if os.path.exists(existing) and os.path.exists(
            entity_ref_multiview_path(ref_dir, instr.instruction_id)
        ):
            logger.info("Front-view refs already exist for %s — skip", instr.instruction_id)
            return existing

        select_count = min(6, max(1, self.config.entity_multiview_top_k))

        valid_appearances = self._valid_appearances(record.appearances)
        if not valid_appearances:
            return None

        catalog = appearances_catalog_for_selection(valid_appearances)
        selected_indices = await self.api_client.select_multiview_reference_keyframes(
            entity_id=instr.entity_id,
            instruction_id=instr.instruction_id,
            subject_features=instr.subject_features,
            appearance_time_hint=instr.appearance_time_hint or record.appearance_time_hint,
            appearances_catalog=catalog,
            select_count=min(select_count, len(valid_appearances)),
            video_duration_sec=video_duration_sec,
        )
        appearances = resolve_selected_appearances(valid_appearances, selected_indices)
        if len(appearances) < select_count:
            selected_paths = {a.keyframe_path for a in appearances}
            for app in valid_appearances:
                if len(appearances) >= select_count:
                    break
                if app.keyframe_path in selected_paths:
                    continue
                appearances.append(app)
                selected_paths.add(app.keyframe_path)
        if not appearances:
            appearances = valid_appearances[:select_count]

        work_dir = os.path.join(ref_dir, f"{instr.instruction_id}_front_work")
        os.makedirs(work_dir, exist_ok=True)

        keyframe_paths = [a.keyframe_path for a in appearances]
        grid_path = os.path.join(work_dir, "input_keyframe_grid.png")
        save_keyframe_grid(keyframe_paths, grid_path, cols=REFERENCE_GRID_COLS)
        keyframe_notes = format_keyframe_notes(appearances)
        input_manifest: Dict[str, Any] = {
            "instruction_id": instr.instruction_id,
            "entity_id": instr.entity_id,
            "subject_features": instr.subject_features,
            "edit_prompt": instr.edit_prompt,
            "action": instr.action.value,
            "selection_method": "vlm_best_entity_related_front_view_set",
            "selected_indices": selected_indices,
            "video_duration_sec": video_duration_sec,
            "appearances_catalog": catalog,
            "keyframe_grid_path": grid_path,
            "keyframe_notes": keyframe_notes,
            "keyframes": [a.to_dict() for a in appearances],
            "reference_qa_mode": "front_view_edit_qa_retry",
            "max_attempts": MAX_MULTIVIEW_QA_RETRIES + 1,
        }

        best_source, synth_qa = await self._synthesize_with_qa(
            work_dir=work_dir,
            grid_path=grid_path,
            keyframe_notes=keyframe_notes,
            instr=instr,
            record=record,
        )
        input_manifest["synthesis_qa"] = synth_qa
        input_manifest["synthesis_front_path"] = best_source

        edited_path: Optional[str] = None
        if instr.action != EditAction.DELETE:
            edited_path, edit_qa = await self._edit_with_qa(
                work_dir=work_dir,
                source_path=best_source,
                grid_path=grid_path,
                instr=instr,
            )
            input_manifest["edit_qa"] = edit_qa
            input_manifest["edited_front_path"] = edited_path
        else:
            logger.info(
                "Delete instruction %s — skip edited front-view overview",
                instr.instruction_id,
            )

        best_ref_keyframe = max(
            appearances,
            key=lambda a: (float(a.quality_score), float(a.confidence)),
        )
        paths = save_multiview_entity_refs(
            ref_dir,
            instr.instruction_id,
            entity_id=instr.entity_id,
            action=instr.action.value,
            subject_features=instr.subject_features,
            multiview_source_path=best_source,
            multiview_edited_path=edited_path,
            top_keyframe_path=best_ref_keyframe.keyframe_path,
            input_manifest=input_manifest,
        )
        logger.info(
            "Saved front-view refs for %s → %s",
            instr.instruction_id,
            paths.get("canonical", ""),
        )
        return paths.get("canonical")

    async def _synthesize_with_qa(
        self,
        *,
        work_dir: str,
        grid_path: str,
        keyframe_notes: str,
        instr: EntityInstruction,
        record: EntityKeyframeRecord,
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Synthesize front-view reference with QA-guided retries."""
        max_attempts = MAX_MULTIVIEW_QA_RETRIES + 1
        final_path = os.path.join(work_dir, "synth_sheet.png")
        qa_history: List[Dict[str, Any]] = []
        avoid_ops = ""
        positive_ops = ""

        current_path = os.path.join(work_dir, "synth_attempt_01.png")
        await self.api_client.generate_entity_multiview_sheet(
            grid_path,
            entity_id=instr.entity_id,
            instruction_id=instr.instruction_id,
            subject_features=instr.subject_features,
            appearance_time_hint=instr.appearance_time_hint or record.appearance_time_hint,
            keyframe_notes=keyframe_notes,
            output_path=current_path,
            avoid_operations=avoid_ops,
            positive_prompt=positive_ops,
        )

        qa_passed = False
        for attempt_num in range(1, max_attempts + 1):
            qa = await self.api_client.validate_entity_multiview_synthesis(
                keyframe_grid_path=grid_path,
                multiview_sheet_path=current_path,
                entity_id=instr.entity_id,
                instruction_id=instr.instruction_id,
                subject_features=instr.subject_features,
                keyframe_notes=keyframe_notes,
            )
            qa["attempt"] = attempt_num
            qa_history.append(qa)

            if qa.get("passed"):
                qa_passed = True
                logger.info(
                    "%s front-view synthesis QA passed on attempt %d/%d",
                    instr.instruction_id,
                    attempt_num,
                    max_attempts,
                )
                break

            if attempt_num >= max_attempts:
                logger.warning(
                    "%s front-view synthesis QA still failed on attempt %d/%d: %s",
                    instr.instruction_id,
                    attempt_num,
                    max_attempts,
                    qa.get("feedback", ""),
                )
                break

            avoid_ops = build_multiview_qa_avoid_operations(qa)
            positive_ops = str(qa.get("positive_prompt", "")).strip()
            logger.warning(
                "%s front-view synthesis QA failed attempt %d/%d — retrying: %s",
                instr.instruction_id,
                attempt_num,
                max_attempts,
                qa.get("feedback", ""),
            )
            next_path = (
                final_path
                if attempt_num == max_attempts - 1
                else os.path.join(work_dir, f"synth_attempt_{attempt_num + 1:02d}.png")
            )
            await self.api_client.generate_entity_multiview_sheet(
                grid_path,
                entity_id=instr.entity_id,
                instruction_id=instr.instruction_id,
                subject_features=instr.subject_features,
                appearance_time_hint=instr.appearance_time_hint or record.appearance_time_hint,
                keyframe_notes=keyframe_notes,
                output_path=next_path,
                avoid_operations=avoid_ops,
                positive_prompt=positive_ops,
            )
            current_path = next_path

        if current_path != final_path and os.path.exists(current_path):
            shutil.copy2(current_path, final_path)

        qa_sidecar = os.path.join(work_dir, "synthesis.qa.json")
        with open(qa_sidecar, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "instruction_id": instr.instruction_id,
                    "entity_id": instr.entity_id,
                    "attempts": qa_history,
                    "qa_passed": qa_passed,
                    "max_attempts": max_attempts,
                    "final_avoid_operations": avoid_ops,
                    "final_positive_operations": positive_ops,
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )

        return (final_path if os.path.exists(final_path) else current_path, qa_history)

    async def _edit_with_qa(
        self,
        *,
        work_dir: str,
        source_path: str,
        grid_path: str,
        instr: EntityInstruction,
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Edit front-view reference with QA-guided retries."""
        max_attempts = MAX_MULTIVIEW_QA_RETRIES + 1
        final_path = os.path.join(work_dir, "edit_sheet.png")
        qa_history: List[Dict[str, Any]] = []
        avoid_ops = ""
        positive_ops = ""

        current_path = os.path.join(work_dir, "edit_attempt_01.png")
        await self.api_client.edit_entity_multiview_sheet(
            source_path,
            entity_id=instr.entity_id,
            instruction_id=instr.instruction_id,
            subject_features=instr.subject_features,
            edit_prompt=instr.edit_prompt,
            output_path=current_path,
            avoid_operations=avoid_ops,
            positive_prompt=positive_ops,
        )

        qa_passed = False
        for attempt_num in range(1, max_attempts + 1):
            qa = await self.api_client.validate_entity_multiview_edit(
                multiview_source_path=source_path,
                multiview_edited_path=current_path,
                keyframe_grid_path=grid_path,
                entity_id=instr.entity_id,
                instruction_id=instr.instruction_id,
                subject_features=instr.subject_features,
                edit_prompt=instr.edit_prompt,
            )
            qa["attempt"] = attempt_num
            qa_history.append(qa)

            if qa.get("passed"):
                qa_passed = True
                logger.info(
                    "%s front-view edit QA passed on attempt %d/%d",
                    instr.instruction_id,
                    attempt_num,
                    max_attempts,
                )
                break

            if attempt_num >= max_attempts:
                logger.warning(
                    "%s front-view edit QA still failed on attempt %d/%d: %s",
                    instr.instruction_id,
                    attempt_num,
                    max_attempts,
                    qa.get("feedback", ""),
                )
                break

            avoid_ops = build_multiview_qa_avoid_operations(qa)
            positive_ops = str(qa.get("positive_prompt", "")).strip()
            logger.warning(
                "%s front-view edit QA failed attempt %d/%d — retrying: %s",
                instr.instruction_id,
                attempt_num,
                max_attempts,
                qa.get("feedback", ""),
            )
            next_path = (
                final_path
                if attempt_num == max_attempts - 1
                else os.path.join(work_dir, f"edit_attempt_{attempt_num + 1:02d}.png")
            )
            await self.api_client.edit_entity_multiview_sheet(
                source_path,
                entity_id=instr.entity_id,
                instruction_id=instr.instruction_id,
                subject_features=instr.subject_features,
                edit_prompt=instr.edit_prompt,
                output_path=next_path,
                avoid_operations=avoid_ops,
                positive_prompt=positive_ops,
            )
            current_path = next_path

        if current_path != final_path and os.path.exists(current_path):
            shutil.copy2(current_path, final_path)

        qa_sidecar = os.path.join(work_dir, "edit.qa.json")
        with open(qa_sidecar, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "instruction_id": instr.instruction_id,
                    "entity_id": instr.entity_id,
                    "attempts": qa_history,
                    "qa_passed": qa_passed,
                    "max_attempts": max_attempts,
                    "final_avoid_operations": avoid_ops,
                    "final_positive_operations": positive_ops,
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )

        return (final_path if os.path.exists(final_path) else current_path, qa_history)

    @staticmethod
    def _valid_appearances(
        appearances: List[KeyframeEntityAppearance],
    ) -> List[KeyframeEntityAppearance]:
        return [
            a for a in appearances
            if a.keyframe_path and os.path.exists(a.keyframe_path)
        ]

    @staticmethod
    def _load_existing_paths(
        config: AgentConfig,
        entity_instru: EntityInstructionSet,
    ) -> Dict[str, str]:
        ref_dir = os.path.join(config.workspace_dir, "entity_refs")
        paths: Dict[str, str] = {}
        for instr in entity_instru.instructions:
            canonical = entity_ref_canonical_path(ref_dir, instr.instruction_id)
            if os.path.exists(canonical):
                paths[instr.instruction_id] = canonical
        return paths
