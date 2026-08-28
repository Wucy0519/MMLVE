"""
Module 3 — Keyframe anchor high-fidelity editing (first frame only).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from video_editing_agent.clients.base import ModelApiClientBase
from video_editing_agent.config import AgentConfig
from video_editing_agent.schemas.instructions import EditAction, EntityInstruction, EntityInstructionSet
from video_editing_agent.schemas.scenes import SpatialMaskAsset, TimeInstruction, TimeInstructionSet
from video_editing_agent.utils.edit_qa_utils import (
    build_edit_retry_guidance_section,
    build_keyframe_qa_avoid_operations,
    merge_vlm_prompts_into_planned_edits,
    normalize_qa_error_list,
    overwrite_planned_edits_sidecar,
)
from video_editing_agent.utils.mask_utils import (
    ENTITY_COLOR_REGISTRY_FILENAME,
    EntityColorRegistry,
    assess_keyframe_entity_recognition,
    build_keyframe_location_edit_directives,
    collect_keyframe_entity_ref_paths,
    entity_color_hex,
    entity_color_name,
    color_name_from_hex,
    entity_ref_overlay_path,
    find_keyframe_location_conflicts,
)
from video_editing_agent.utils.workspace_checkpoints import (
    KeyframeAnchors,
    load_module3_checkpoint,
    module3_scene_is_done,
    persist_module3_scene_skip,
    reconcile_time_instruction_requires_edit,
    save_module3_manifest,
)

logger = logging.getLogger(__name__)


@dataclass
class SceneEditOutcome:
    """Result of editing one scene's begin keyframe."""

    anchors: Optional[KeyframeAnchors] = None
    skip_reason: Optional[str] = None


class KeyframeEditor:
    """Keyframe Anchor Editing Agent (Module 3)."""

    def __init__(
        self,
        config: AgentConfig,
        api_client: ModelApiClientBase,
    ) -> None:
        self.config = config
        self.api_client = api_client

    async def run(
        self,
        time_instru: TimeInstructionSet,
        entity_instru: EntityInstructionSet,
    ) -> Tuple[Dict[str, KeyframeAnchors], Dict[str, Dict[str, object]]]:
        """Edit begin anchor keyframes; resume per-scene from workspace checkpoint."""
        logger.info("KeyframeEditor.run started")
        instr_by_id = {i.instruction_id: i for i in entity_instru.instructions}
        scene_by_id = {s.scene_id: s for s in time_instru.scenes}

        if reconcile_time_instruction_requires_edit(
            time_instru,
            entity_instru,
            self.config.scenes_dir,
        ):
            time_instru.save(self.config.time_instru_path)
            logger.info("Reconciled requires_edit flags from bound instructions")

        if self.config.resume_from_checkpoints:
            edited_paths, skipped_scenes = load_module3_checkpoint(self.config)
        else:
            edited_paths, skipped_scenes = {}, {}

        for scene_id in list(skipped_scenes.keys()):
            binding = next(
                (b for b in time_instru.time_instructions if b.scene_id == scene_id),
                None,
            )
            if binding is not None and binding.requires_edit:
                skipped_scenes.pop(scene_id, None)
                logger.info(
                    "Cleared stale module-3 skip for %s — scene requires edit again",
                    scene_id,
                )

        for binding in time_instru.time_instructions:
            if not binding.requires_edit:
                continue

            scene_id = binding.scene_id
            if module3_scene_is_done(scene_id, edited_paths, skipped_scenes):
                if scene_id in skipped_scenes:
                    logger.info(
                        "Checkpoint: skip %s — previously skipped (%s)",
                        scene_id,
                        skipped_scenes[scene_id].get("reason", "unknown"),
                    )
                else:
                    logger.info(
                        "Checkpoint: skip %s — begin keyframe already exists",
                        scene_id,
                    )
                continue

            try:
                outcome = await self._edit_scene(
                    binding,
                    instr_by_id,
                    scene_by_id,
                    existing_anchors=edited_paths.get(scene_id, {}),
                )
            except Exception as exc:
                logger.error(
                    "Scene %s keyframe edit failed (%s) — will retry on next run",
                    scene_id, exc,
                )
                continue

            if outcome.anchors:
                prev = edited_paths.get(scene_id, {})
                edited_paths[scene_id] = {**prev, **outcome.anchors}
            elif outcome.skip_reason:
                persist_module3_scene_skip(
                    self.config,
                    time_instru,
                    scene_id,
                    outcome.skip_reason,
                    skipped_scenes,
                )

            if self.config.resume_from_checkpoints:
                save_module3_manifest(
                    self.config,
                    {scene_id: edited_paths.get(scene_id, {})},
                    skipped_scenes,
                    time_instru=time_instru,
                )

        logger.info(
            "KeyframeEditor.run done — %d scenes with anchors, %d skipped",
            len(edited_paths),
            len(skipped_scenes),
        )
        return edited_paths, skipped_scenes

    async def _edit_scene(
        self,
        binding: TimeInstruction,
        instr_by_id: Dict[str, EntityInstruction],
        scene_by_id: dict,
        *,
        existing_anchors: KeyframeAnchors,
    ) -> SceneEditOutcome:
        """Apply all applicable instructions to the begin anchor frame."""
        scene = scene_by_id.get(binding.scene_id)
        image_begin, mask_path = self._load_scene_assets(binding, scene_by_id)
        if not image_begin or not os.path.exists(image_begin):
            logger.warning("Missing image_begin for %s — skip", binding.scene_id)
            return SceneEditOutcome()

        begin_out = os.path.join(
            self.config.keyframes_dir,
            f"{binding.scene_id}_edited_begin.png",
        )

        entity_ids = [
            instr_by_id[iid].entity_id
            for iid in binding.instruction_ids
            if iid in instr_by_id
        ]
        mask_path = self._ensure_mask_file(binding, image_begin, mask_path)
        self._ensure_mask_asset(binding, entity_ids)

        pending: List[Tuple[EntityInstruction, str]] = []
        for iid in binding.instruction_ids:
            instr = instr_by_id.get(iid)
            if instr is None:
                continue

            color_name = self._color_name_for_entity(binding, instr.entity_id)
            if not color_name:
                continue

            ref_path = self._entity_ref_path(instr.instruction_id)
            visible = await self.api_client.check_entity_in_frame(
                image_begin,
                instr.subject_features,
                action=instr.action.value,
                reference_image_path=ref_path or None,
            )
            if not visible:
                if instr.action == EditAction.DELETE:
                    logger.info(
                        "Skip delete %s in %s — target already absent",
                        iid, binding.scene_id,
                    )
                else:
                    logger.info(
                        "Skip %s in %s — VLM: entity not detected in frame",
                        iid, binding.scene_id,
                    )
                continue

            pending.append((instr, color_name))

        if not pending:
            logger.info(
                "Scene %s: VLM detected no target entities — skip keyframe edit",
                binding.scene_id,
            )
            return SceneEditOutcome(skip_reason="no_entities_detected")

        edit_items = [
            {
                "color_name": color_name,
                "instruction_id": instr.instruction_id,
                "edit_prompt": instr.edit_prompt,
                "entity_id": instr.entity_id,
                "subject_features": instr.subject_features,
                "color_hex": self._color_hex_for_entity(binding, instr.entity_id),
                "reference_overlay_path": self._entity_ref_path(instr.instruction_id),
            }
            for instr, color_name in pending
        ]

        anchors: KeyframeAnchors = {}
        if existing_anchors.get("begin") and os.path.exists(existing_anchors["begin"]):
            anchors["begin"] = existing_anchors["begin"]
            logger.info(
                "Scene %s: reuse existing begin keyframe %s",
                binding.scene_id,
                existing_anchors["begin"],
            )
        else:
            ok = await self._edit_anchor_frame(
                binding=binding,
                image_path=image_begin,
                mask_path=mask_path,
                edit_items=edit_items,
                pending=pending,
                out_path=begin_out,
            )
            if not ok:
                logger.warning(
                    "Scene %s begin-frame inpaint failed — skip scene",
                    binding.scene_id,
                )
                return SceneEditOutcome(skip_reason="inpaint_failed")
            anchors["begin"] = begin_out

        return SceneEditOutcome(anchors=anchors)

    async def _edit_anchor_frame(
        self,
        *,
        binding: TimeInstruction,
        image_path: str,
        mask_path: str,
        edit_items: List[dict],
        pending: List[Tuple[EntityInstruction, str]],
        out_path: str,
    ) -> bool:
        """Run location derivation + inpaint for one anchor frame."""
        location_prompts, location_records = await self.api_client.derive_keyframe_edit_entity_locations(
            image_path,
            mask_path,
            edit_items,
        )
        entity_id_by_iid = {
            item["instruction_id"]: item["entity_id"]
            for item in edit_items
        }
        records_for_mask: List[Dict[str, object]] = []
        for record in location_records:
            adapted = dict(record)
            iid = str(adapted.get("instruction_id", "")).strip()
            if not str(adapted.get("entity_id", "")).strip() and iid:
                adapted["entity_id"] = entity_id_by_iid.get(iid, "")
            records_for_mask.append(adapted)
        entity_descs = [
            {
                "entity_id": item["entity_id"],
                "description": item["subject_features"],
                "instruction_id": item["instruction_id"],
            }
            for item in edit_items
        ]
        color_map = {item["entity_id"]: item["color_hex"] for item in edit_items}
        entity_refs = {
            item["entity_id"]: item["reference_overlay_path"]
            for item in edit_items
            if item.get("reference_overlay_path")
        }
        instr_labels = {item["entity_id"]: item["instruction_id"] for item in edit_items}
        supplement = getattr(self.api_client, "supplement_mask_for_located_entities", None)
        if supplement is not None and records_for_mask:
            await supplement(
                image_path,
                mask_path,
                entity_descs,
                color_map,
                records_for_mask,
                entity_references=entity_refs,
                instruction_labels=instr_labels,
                min_confidence=self.config.keyframe_entity_min_confidence,
            )
        records_by_iid = {
            str(record.get("instruction_id", "")).strip(): record
            for record in location_records
            if str(record.get("instruction_id", "")).strip()
        }
        subject_by_iid = {
            item["instruction_id"]: item["subject_features"]
            for item in edit_items
        }
        location_conflicts = find_keyframe_location_conflicts(
            records_by_iid,
            subject_by_instruction=subject_by_iid,
            entity_id_by_instruction=entity_id_by_iid,
        )
        rejected_recognition: List[Dict[str, object]] = []
        if self.config.dev_mode:
            accepted_pending = list(pending)
        else:
            accepted_pending = []
            for instr, color_name in pending:
                record = records_by_iid.get(instr.instruction_id)
                ok, reason = assess_keyframe_entity_recognition(
                    record,
                    min_confidence=self.config.keyframe_entity_min_confidence,
                    subject_features=instr.subject_features,
                    mask_path=mask_path,
                    color_hex=self._color_hex_for_entity(binding, instr.entity_id),
                    action=instr.action.value,
                )
                if ok:
                    accepted_pending.append((instr, color_name))
                    continue
                logger.warning(
                    "Skip %s in %s — %s",
                    instr.instruction_id,
                    binding.scene_id,
                    reason,
                )
                rejected_recognition.append({
                    "instruction_id": instr.instruction_id,
                    "entity_id": instr.entity_id,
                    "reason": reason,
                    "confidence": record.get("confidence") if record else None,
                    "present_in_frame": (
                        record.get("present_in_frame") if record else None
                    ),
                })

            if rejected_recognition:
                logger.info(
                    "Scene %s: skipped %d instruction(s) due to low entity recognition confidence",
                    binding.scene_id,
                    len(rejected_recognition),
                )

        if not accepted_pending:
            logger.info(
                "Scene %s: no instructions passed entity recognition — skip inpaint",
                binding.scene_id,
            )
            return False

        planned_edits = [
            {
                "instruction_id": instr.instruction_id,
                "entity_id": instr.entity_id,
                "action": instr.action.value,
                "edit_prompt": instr.edit_prompt,
                "subject_features": instr.subject_features,
                "location_prompt": location_prompts.get(instr.instruction_id, ""),
            }
            for instr, _ in accepted_pending
        ]

        location_sidecar = f"{out_path}.location_prompts.json"
        with open(location_sidecar, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "prompts": location_prompts,
                    "records": location_records,
                    "planned_edits": planned_edits,
                    "location_conflicts": location_conflicts,
                    "rejected_recognition": rejected_recognition,
                    "min_confidence": self.config.keyframe_entity_min_confidence,
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )

        accepted_iids = {instr.instruction_id for instr, _ in accepted_pending}
        action_by_iid = {
            instr.instruction_id: instr.action.value
            for instr, _ in accepted_pending
        }
        frame_edit_items = [
            item for item in edit_items if item["instruction_id"] in accepted_iids
        ]
        rewrite_items = []
        for item in frame_edit_items:
            rewrite_items.append({
                "instruction_id": item["instruction_id"],
                "edit_prompt": item["edit_prompt"],
                "location_prompt": location_prompts.get(item["instruction_id"], ""),
                "color_name": item["color_name"],
                "subject_features": item["subject_features"],
                "action": action_by_iid.get(item["instruction_id"], ""),
            })
        edit_directives = build_keyframe_location_edit_directives(rewrite_items)
        qa_edit_prompt = self._build_qa_edit_prompt(accepted_pending)
        qa_success_criteria = self._build_success_criteria_prompt(accepted_pending)
        qa_action = (
            "delete"
            if all(instr.action == EditAction.DELETE for instr, _ in accepted_pending)
            else "modify"
        )
        qa_subject_features = self._build_qa_subject_features(accepted_pending)

        try:
            entity_ref_guides = self._build_entity_ref_guides(accepted_pending)
            await self._inpaint_with_qa(
                image_begin=image_path,
                mask_begin=mask_path,
                edit_directives=edit_directives,
                ref_image=None,
                output_path=out_path,
                qa_edit_prompt=qa_edit_prompt,
                qa_action=qa_action,
                qa_success_criteria=qa_success_criteria,
                qa_instruction_ids=qa_subject_features,
                inpaint_guidance="location",
                entity_ref_guides=entity_ref_guides,
            )
        except Exception as exc:
            logger.warning(
                "Scene %s inpaint failed for %s: %s",
                binding.scene_id,
                out_path,
                exc,
            )
            return False

        success_criteria_by_iid = {
            instr.instruction_id: (
                instr.success_criteria_prompt or instr.edit_prompt or ""
            ).strip()
            for instr, _ in accepted_pending
        }
        records_by_iid = {
            str(record.get("instruction_id", "")).strip(): record
            for record in location_records
            if str(record.get("instruction_id", "")).strip()
        }
        for item in planned_edits:
            iid = str(item.get("instruction_id", "")).strip()
            record = records_by_iid.get(iid)
            if record:
                loc_edit = str(record.get("location_edit_prompt", "") or "").strip()
                if loc_edit:
                    item["location_edit_prompt"] = loc_edit
        merged_planned_edits = merge_vlm_prompts_into_planned_edits(
            planned_edits,
            success_criteria_by_instruction=success_criteria_by_iid,
        )
        overwrite_planned_edits_sidecar(location_sidecar, merged_planned_edits)
        return True

    def _entity_ref_path(self, instruction_id: str) -> str:
        ref_path = entity_ref_overlay_path(
            os.path.join(self.config.workspace_dir, "entity_refs"),
            instruction_id,
        )
        return ref_path if os.path.exists(ref_path) else ""

    def _build_entity_ref_guides(
        self,
        pending: List[Tuple[EntityInstruction, str]],
    ) -> List[Dict[str, object]]:
        """Collect per-instruction entity_refs bundles for inpaint API attachment."""
        ref_dir = os.path.join(self.config.workspace_dir, "entity_refs")
        guides: List[Dict[str, object]] = []
        for instr, color_name in pending:
            paths = collect_keyframe_entity_ref_paths(
                ref_dir,
                instr.instruction_id,
                action=instr.action.value,
            )
            if not paths:
                continue
            guides.append({
                "instruction_id": instr.instruction_id,
                "entity_id": instr.entity_id,
                "action": instr.action.value,
                "color_name": color_name,
                "paths": paths,
            })
        return guides

    @staticmethod
    def _merge_qa_retry_directives(base_directives: str, qa: Dict[str, Any]) -> str:
        """Append VLM QA retry guidance (positive first, then avoid) to the next inpaint prompt."""
        positive_ops = str(qa.get("positive_prompt", "")).strip()
        avoid_ops = build_keyframe_qa_avoid_operations(qa)
        retry_guidance = build_edit_retry_guidance_section(
            positive_prompt=positive_ops,
            avoid_operations=avoid_ops,
        )
        if not retry_guidance:
            return base_directives
        return f"{base_directives.rstrip()}{retry_guidance}"

    @staticmethod
    def _color_hex_for_entity(binding: TimeInstruction, entity_id: str) -> str:
        if binding.mask_asset:
            hex_color = binding.mask_asset.entity_color_map.get(entity_id)
            if hex_color:
                return hex_color
        return "#FF0000"

    @staticmethod
    def _color_name_for_entity(binding: TimeInstruction, entity_id: str) -> str:
        if binding.mask_asset:
            name = binding.mask_asset.entity_color_name_map.get(entity_id)
            if name:
                return name
            hex_color = (binding.mask_asset.entity_color_map.get(entity_id) or "").upper()
            if hex_color:
                resolved = color_name_from_hex(hex_color)
                if resolved:
                    return resolved
                for i in range(6):
                    if entity_color_hex(i).upper() == hex_color:
                        return entity_color_name(i)
        return ""

    def _ensure_mask_file(
        self,
        binding: TimeInstruction,
        image_begin: str,
        mask_path: str,
    ) -> str:
        """Ensure ``mask_0000.png`` exists (empty black mask when Module 2 missed)."""
        if mask_path and os.path.exists(mask_path):
            return mask_path

        from PIL import Image

        mask_dir = os.path.join(self.config.scenes_dir, binding.scene_id, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, "mask_0000.png")
        if not os.path.exists(mask_path):
            frame = Image.open(image_begin).convert("RGB")
            Image.new("RGB", frame.size, (0, 0, 0)).save(mask_path)
            logger.info(
                "Scene %s: created empty mask_0000 — Module 3 will rely on VLM location",
                binding.scene_id,
            )
        return mask_path

    def _ensure_mask_asset(
        self,
        binding: TimeInstruction,
        entity_ids: List[str],
    ) -> None:
        """Assign stable palette colors when ``mask_asset`` is missing or incomplete."""
        registry = EntityColorRegistry(
            os.path.join(self.config.workspace_dir, ENTITY_COLOR_REGISTRY_FILENAME),
        )
        color_map, name_map = registry.build_color_maps(
            [eid for eid in entity_ids if eid],
        )
        if not color_map:
            return

        mask_dir = os.path.join(self.config.scenes_dir, binding.scene_id, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        if binding.mask_asset is None:
            binding.mask_asset = SpatialMaskAsset(
                scene_id=binding.scene_id,
                mask_dir=os.path.abspath(mask_dir),
                entity_color_map=color_map,
                entity_color_name_map=name_map,
            )
            return

        binding.mask_asset.entity_color_map.update(color_map)
        binding.mask_asset.entity_color_name_map.update(name_map)

    def _load_scene_assets(
        self,
        binding: TimeInstruction,
        scene_by_id: dict,
    ) -> Tuple[str, str]:
        """Resolve first-frame image and shared multi-color mask paths."""
        scene = scene_by_id.get(binding.scene_id)
        image_begin = (scene.first_frame_path if scene else "") or ""
        mask_begin = ""
        if binding.mask_asset:
            mask_begin = os.path.join(binding.mask_asset.mask_dir, "mask_0000.png")
        return image_begin, mask_begin

    @staticmethod
    def _build_qa_edit_prompt(
        pending: List[Tuple[EntityInstruction, str]],
    ) -> str:
        """Original edit_prompt lines for VLM keyframe QA (not location directives)."""
        lines = []
        for instr, _color_name in pending:
            edit = (instr.edit_prompt or "").strip()
            if edit:
                lines.append(f"{instr.instruction_id}: {edit}")
        return "\n".join(lines) or "No edit instructions."

    @staticmethod
    def _build_qa_subject_features(
        pending: List[Tuple[EntityInstruction, str]],
    ) -> str:
        """Subject descriptions for delete QA prompts."""
        lines = []
        for instr, _color_name in pending:
            subject = (instr.subject_features or "").strip()
            if subject:
                lines.append(f"{instr.instruction_id}: {subject}")
        return "\n".join(lines)

    @staticmethod
    def _build_success_criteria_prompt(
        pending: List[Tuple[EntityInstruction, str]],
    ) -> str:
        """Combine per-instruction QA prompts for batched keyframe validation."""
        criteria = []
        for idx, (instr, color_name) in enumerate(pending, start=1):
            criteria.append(
                f"Edit {idx} ({instr.instruction_id}, {color_name}): "
                f"{instr.success_criteria_prompt or instr.edit_prompt}"
            )
        return "\n".join(criteria)

    async def _inpaint_with_qa(
        self,
        *,
        image_begin: str,
        mask_begin: str,
        edit_directives: str,
        ref_image: Optional[str],
        output_path: str,
        qa_edit_prompt: str,
        qa_action: str,
        qa_success_criteria: str,
        qa_instruction_ids: str,
        inpaint_guidance: str,
        entity_ref_guides: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        """Location-guided inpaint with a single VLM QA pass and one retry on failure."""
        strength = 1.0

        async def _run_inpaint(directives: str, *, label: str) -> None:
            logger.info("Keyframe inpaint (%s) → %s", label, output_path)
            await self.api_client.masked_inpaint(
                image_begin,
                mask_begin,
                directives,
                output_path,
                ref_image_path=None,
                entity_ref_guides=entity_ref_guides,
                preserve_frame_structure=False,
                inpaint_guidance=inpaint_guidance,
                strength=strength,
            )

        if not self.config.keyframe_edit_qa:
            await _run_inpaint(edit_directives, label="VLM QA disabled")
            return

        await _run_inpaint(edit_directives, label="initial")
        qa = await self.api_client.validate_edit_quality(
            image_begin,
            output_path,
            qa_edit_prompt,
            subject_features=qa_instruction_ids,
            action=qa_action,
            success_criteria_prompt=qa_success_criteria,
            keyframe=True,
        )
        qa_history: List[Dict[str, Any]] = [{"attempt": 1, "phase": "vlm_qa", **qa}]
        merged_directives = edit_directives
        retried = False
        avoid_ops = ""

        if qa.get("passed"):
            logger.info(
                "Keyframe edit QA passed (score=%.2f) — keeping initial result",
                float(qa.get("score", 0)),
            )
        else:
            logger.warning(
                "Keyframe edit QA failed — retrying inpaint once | failed=%s | %s",
                qa.get("failed_aspects", []),
                qa.get("feedback", ""),
            )
            retried = True
            avoid_ops = build_keyframe_qa_avoid_operations(qa)
            merged_directives = self._merge_qa_retry_directives(edit_directives, qa)
            await _run_inpaint(merged_directives, label="qa-retry")

        qa_sidecar = f"{output_path}.qa.json"
        try:
            with open(qa_sidecar, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "qa_enabled": True,
                        "qa_checks": 1,
                        "retried_inpaint": retried,
                        "attempts": qa_history,
                        "final_passed": bool(qa.get("passed")),
                        "edit_directives_base": edit_directives,
                        "edit_directives_final": merged_directives,
                        "qa_edit_errors": normalize_qa_error_list(qa.get("edit_errors"))
                        or normalize_qa_error_list(qa.get("failed_aspects")),
                        "qa_avoid_edit_operations": avoid_ops if retried else "",
                        "qa_positive_edit_operations": str(qa.get("positive_prompt", "")).strip() if retried else "",
                    },
                    fh,
                    indent=2,
                    ensure_ascii=False,
                )
        except OSError as exc:
            logger.warning("Could not write keyframe QA sidecar %s: %s", qa_sidecar, exc)