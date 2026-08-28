"""English prompt templates for LLM / vision calls."""

INSTRUCTION_REWRITE_PROMPT = """You are a video editing instruction clarifier.

The user provides a natural-language request to edit a real-world video. The request may be vague,
incomplete, ambiguous, or mix multiple languages (e.g. Chinese + English).

Your job is to REWRITE the request into a clear, explicit editing brief BEFORE structured parsing.

Focus on clarifying:
- WHO / WHAT is the target (precise visual identity for each person or object)
- If the user cites a video moment to identify WHO (e.g. "约第30秒出现的女子", "视频前几帧里的男子",
  "the woman in the first few frames"), preserve that wording as a natural-language identification cue
  in the rewritten brief — do NOT convert it to a numeric timestamp or edit start time
- WHAT visual change to apply for each target (describe the desired outcome in plain language)
- WHEN the edit applies — distinguish:
  * Entity identification time cues (e.g. "around 30 seconds", "约第30秒出现") → only help identify WHO, NOT edit start time
  * True edit time windows (e.g. "only in the first 10 seconds") → keep as explicit scope
- Resolve pronouns and coreference (he/she/that man/the girl → explicit description)
- Split multiple distinct edits into numbered, separate sentences
- Remove noise and repetition; keep all user intent

Rules:
- Output in English
- Be specific and visually grounded (clothing, hair, accessories, colors, approximate age)
- Do NOT invent edits the user did not ask for
- Do NOT drop any requested edit
- Return ONLY valid JSON, no markdown fences

Return format:
{{
  "rewritten_prompt": "Single coherent editing brief with one clear sentence per distinct edit.",
  "success_criteria_prompts": [
    {{
      "rewrite_id": "rewrite_001",
      "target_subject": "Precise target subject for this edit",
      "edit_goal": "What must be visibly changed",
      "success_criteria_prompt": "A self-contained QA prompt/checklist for deciding whether the keyframe edit succeeded. It must mention the target, expected visual result, what must remain preserved, and clear failure cases."
    }}
  ],
  "clarifications": [
    "Brief note on each ambiguity you resolved"
  ]
}}

Original user request:
{user_prompt}
"""

INSTRUCTION_PARSE_PROMPT = """You are a video editing instruction parser.

Parse the user's natural-language editing request into a JSON object with an "instructions" array.

**One entity → one instruction entry.** Each distinct person or object that must be edited gets exactly
one object in the array. The same entity_id must NOT appear twice. If the user mentions multiple
changes to the same person/object, merge them into a single instruction for that entity.

Do NOT classify edits as add / delete / modify. Describe every edit only through subject_features,
edit_prompt, and success_criteria_prompt.

Each instruction must include:
- instruction_id: unique id like "instr_001", "instr_002"
- entity_id: stable entity id for coreference across shots, like "entity_01" (one per distinct target)
- subject_features: visual description of the target subject/object IN THE VIDEO (for grounding/tracking)
- appearance_time_hint: optional English phrase describing WHEN/WHERE in the video the user
  cites the target for identification only — keep the user's vagueness (e.g. "appears in the first
  few frames of the video", "appears around second 30 in the video"). Do NOT output a numeric
  second field; use natural language only. NOT an edit window.
- edit_prompt: detailed inpainting/generation description of the desired visual outcome
- success_criteria_prompt: self-contained prompt/checklist for judging whether the first-frame keyframe edit succeeded
- target_instance_scope: "single" or "multiple" — whether the edit applies to ONE specific tracked
  person/object instance, or to EVERY instance in the video that matches the same features
- time_condition: object with:
    - condition_type: "absolute" or "event"
    - start_sec / end_sec: floats (required ONLY if user explicitly limits WHEN to edit)
    - event_description: string (required if event), e.g. "wherever the entity appears in the video"

Temporal reference disambiguation (CRITICAL):
- Timestamps or vague time phrases in the user prompt (e.g. "around 30 seconds", "at about the 30s mark",
  "the man who appears at 30 seconds", "约第30秒出现的") are usually **entity identification cues** —
  they help pick WHICH person/object is meant, NOT when to start/stop editing.
- The identified subject may appear earlier, later, or throughout the video. Do NOT convert such
  references into condition_type "absolute" or an edit window starting at that second.
- Put visual identity in subject_features (e.g. "woman in red dress; appears in the first few frames of the video").
  When the user uses a referential time/position phrase to identify the target, also set appearance_time_hint
  as a natural-language description — preserve fuzzy wording like "first few frames", NOT a parsed float.
- Default time_condition:
  condition_type "event", event_description "wherever the entity appears in the video"
  (or a visual event like "while dancing" ONLY if the user clearly ties edit scope to that action).
- Use condition_type "absolute" ONLY when the user explicitly constrains the edit TIME WINDOW, e.g.
  "only between 10s and 20s", "edit just the first minute", "change shirt only in this 5-second clip".
  In that case set both start_sec and end_sec as explicit bounds — never use 9999 as open-ended end
  for a referential timestamp.

Rules:
- Resolve coreference: same person/object across sentences share entity_id.
- Split multiple distinct **entities** into separate instructions; never split one entity into multiple rows.
- **Target instance scope (CRITICAL):**
  - Default target_instance_scope="single" for every instruction unless the user EXPLICITLY asks to edit
    ALL / EVERY / EACH person or object sharing the same features (e.g. "all men wearing flat caps",
    "every person in red shirts", "remove all the ducks").
  - Use target_instance_scope="single" when the user names one specific person/object, even if similar
    people appear elsewhere in the video (e.g. "the man leaning on the railing" = one man, not all men
    with similar clothes).
  - Use target_instance_scope="multiple" ONLY when plural/all/every language clearly applies to every
    matching instance, not merely because multiple similar extras exist in the scene.
- Use English for all string values.
- Return ONLY valid JSON, no markdown fences.
- Preserve or adapt any success criteria from the rewrite step into success_criteria_prompt.
- success_criteria_prompt MUST be specific enough for a VLM judge comparing original and edited frames.

Do NOT output needs_ref_image, ref_subject, or ref_image_path — isolated T2I ref_images are not used.

Example — user: "Change the red dress of the woman who appears in the first few frames to blue"
CORRECT:
  subject_features: "woman in red dress; appears in the first few frames of the video"
  appearance_time_hint: "appears in the first few frames of the video"
  edit_prompt: "replace the red dress with a solid blue dress"
  target_instance_scope: "single"
  time_condition: {{"condition_type": "event", "event_description": "wherever the woman appears in the video"}}

Example — user: "Change the hair color of all girls wearing yellow dresses to blue"
CORRECT:
  subject_features: "girls wearing yellow dresses"
  edit_prompt: "change hair color to blue for every girl wearing a yellow dress"
  target_instance_scope: "multiple"
  time_condition: {{"condition_type": "event", "event_description": "wherever matching girls appear in the video"}}

Example — user: "Place a yellow duck on the blond blue-eyed male lead who appears around 30 seconds"
CORRECT:
  subject_features: "blond blue-eyed male lead; appears around second 30 in the video"
  appearance_time_hint: "appears around second 30 in the video"
  edit_prompt: "place a small yellow duck on his shoulder"
  target_instance_scope: "single"
  time_condition: {{"condition_type": "event", "event_description": "wherever the male lead appears in the video"}}
WRONG:
  time_condition: {{"condition_type": "absolute", "start_sec": 30.0, "end_sec": 9999.0}}
  multiple instructions with the same entity_id

User request (already clarified by a rewrite step — parse this text):
{user_prompt}
"""

EVENT_GROUNDING_PROMPT = """You are an event grounding assistant for video editing.

Given scene keyframe images (in order) and a time condition, decide which scenes match.

Time condition (JSON):
{time_condition}

Scene list:
{scene_list}

Return ONLY valid JSON:
{{
  "matched_scene_ids": ["scene_01", ...],
  "reasoning": "brief explanation"
}}

A scene matches if its keyframe visually shows the subject described in the event (appearance, action, context).

IMPORTANT — referential timestamps:
If event_description or the user intent mentions a time like "around 30 seconds" or "who appears at 30s",
that timestamp is ONLY for identifying which person/object to look for. Do NOT restrict matching to
scenes near that second. Match every scene where the described subject is visually present.
Only use absolute time overlap when condition_type is "absolute" with explicit start_sec and end_sec bounds.
"""

MASK_ANTI_STITCH_FAILURE_CLAUSE = """
COMMON FAILURE — DO NOT DO THIS:
- Using a reference image (image 2+) as the output base or output size template.
- Returning a tinted/scene reproduction instead of a black-background mask.
- Copying the semi-transparent colored overlay from a reference image into the output.
- Colored mask pixels appearing ONLY on the left half OR ONLY on the right half of the output.
- Output width different from image 1 width.
If you catch yourself segmenting image 2+ or reproducing a reference overlay on a scene background, STOP and re-segment image 1 on a full single-frame black canvas with image 1's exact dimensions.
"""

MASK_TARGET_FRAME_CORRESPONDENCE_CLAUSE = """
TARGET-FRAME ONE-TO-ONE CORRESPONDENCE (HIGHEST PRIORITY):
- Image 1 is the detection frame. The output mask must correspond to image 1 ONLY — pixel-for-pixel, same canvas, same coordinates.
- Every colored mask pixel at (x, y) must mark the subject at the SAME (x, y) location in image 1.
- Location guidance text (if provided per entity) describes WHERE to look in image 1 — use it only as a spatial hint for image 1, never as a mask template from another frame.
- Correct: one black-background mask image with exactly image 1's width and height, marking only what is visible in image 1.
"""

MASK_REFERENCE_LOCATION_PROMPT = """You are locating tracked video-edit entities across frames for video inpainting.

Image 1 is the CURRENT detection frame (the frame that will be segmented).
Image 2+ are REFERENCE frames from earlier moments. Each reference shows the target entity under a
semi-transparent colored indicative mask overlay (the colored film marks WHO was selected before).

Your job: for each entity below, decide whether the SAME subject/object appears in image 1 and,
if so, write structured localization cues for image 1.

Entities to locate:
{entity_list}

CRITICAL — VIEWPOINT & POSE INVARIANCE:
- The SAME person/object often reappears with a DIFFERENT viewpoint: back view, side profile,
  3/4 angle, head turned away, sitting vs standing, partial occlusion, or motion blur.
- Do NOT require the subject in image 1 to match the reference pose, facing direction, or visibility.
- Use the reference overlay to learn WHO (identity), not WHERE or WHICH POSE to expect in image 1.
- Identity matching should rely on stable cues: clothing colors/style, hair, body build, gender presentation,
  accessories, object shape/material, skin tone, and scene role — NOT on identical head orientation.
- If only the back, side, hair, shoulders, or partial body is visible in image 1, that STILL counts as present.
- Only set present_in_frame=false when the subject is genuinely not in image 1 or cannot be distinguished.

Rules:
- Image 1 is the only frame whose coordinates matter for spatial fields.
- Do NOT copy reference overlay position into image 1.
- spatial_region / landmark_relations must describe image 1 only.
- visible_body_parts: list what is actually visible in image 1 (e.g. "back", "left profile", "torso", "hair").
- viewpoint_in_detection: describe orientation in image 1 even if different from reference.
- confidence: 0.0-1.0 for identity match confidence in image 1.
- Do NOT describe edit operations.

Return ONLY valid JSON:
{{
  "locations": [
    {{
      "entity_id": "entity_01",
      "present_in_frame": true,
      "confidence": 0.85,
      "viewpoint_change": true,
      "identity_cues": ["blue denim jacket", "short dark hair", "medium build male"],
      "viewpoint_in_reference": "front-facing upper body",
      "viewpoint_in_detection": "back view, head turned slightly left, upper body visible",
      "visible_body_parts": ["back", "shoulders", "arms"],
      "spatial_region": "center-left midground",
      "landmark_relations": ["standing in front of the glass door", "left of the crowd"],
      "location_prompt": "optional short freeform summary for image 1"
    }}
  ]
}}
"""

MASK_REFERENCE_LOCATION_REID_PROMPT = """You are re-identifying ONE tracked entity across a viewpoint change.

Image 1 is the CURRENT detection frame.
Image 2 is a REFERENCE frame where the entity was previously marked with a colored mask overlay.

Target entity:
{entity_block}

The subject may appear in image 1 with a VERY DIFFERENT viewpoint than the reference:
back turned, side profile, partial body, occlusion, different pose, or smaller/larger scale.

Task:
1. Decide if the SAME subject from the reference is visible anywhere in image 1.
2. If yes, provide structured localization for image 1 that helps segment the visible region
   even when the face is not visible or the pose differs.

Be generous on presence when clothing/body/object identity matches but orientation differs.
Only answer present_in_frame=false if the subject truly cannot be found.

Return ONLY valid JSON:
{{
  "entity_id": "{entity_id}",
  "present_in_frame": true,
  "confidence": 0.0,
  "viewpoint_change": true,
  "identity_cues": ["..."],
  "viewpoint_in_reference": "...",
  "viewpoint_in_detection": "...",
  "visible_body_parts": ["..."],
  "spatial_region": "...",
  "landmark_relations": ["..."],
  "location_prompt": "..."
}}
"""

MASK_SEGMENTATION_IMAGE_PROMPT = """Generate one multi-entity edit-region mask for video inpainting.

Canvas: image 1 only — output MUST match image 1 width, height, and pixel grid.
Style: solid black (#000000) background; solid assigned-color regions on real contours only.

{anti_stitch_failure_clause}

Per-entity instructions (segment every listed target that appears in image 1):
{entity_queries}

Output:
- One RGB mask aligned to image 1. No scene reproduction, reference paste, or tinted overlay.
- EXACT assigned hex per entity_id. Contour segmentation only — no boxes or ellipses.
{anti_copy_clause}
"""

ANTI_COPY_MASK_CLAUSE = """
NON-NEGOTIABLE RETRY MASK CONSTRAINTS:
The previous attempt may have segmented a reference image instead of image 1, reproduced a tinted overlay scene, or reused an old mask layout. All are wrong.
Re-segment ONLY from image 1. The output must stay in strict one-to-one correspondence with image 1's pixel grid.
Use reference overlay images for identity only, never for geometry, coordinates, canvas size, or output style.
Ignore any old mask files if they exist on disk; they are not inputs and must not influence this retry.
The new mask must be a single-frame black-background mask for image 1 only — never a reference frame with colored film and never a side-by-side stitched image.
The new mask must NOT resemble any old mask layout or reference-overlay layout in silhouette, size, position, pose, or contour.
Similarity to a reference overlay or previous mask layout is a failure even if the target identity is correct.
"""

MASK_VALIDATION_PROMPT = """You are validating a candidate multi-entity segmentation mask for a video editing pipeline.

Image 1 is the current source video frame.
Image 2 is the candidate colored mask generated for image 1.

Targets and assigned colors:
{entity_color_list}

Previous query prompt used to generate image 2:
{query_prompt}

Return ONLY valid JSON:
{{
  "valid": true,
  "present_entity_ids": ["entity_01"],
  "rejected_entity_ids": ["entity_02"],
  "feedback": "brief explanation",
  "revised_query_prompt": "If valid=false or important targets are missed, rewrite the complete segmentation query prompt to better find the target entities in image 1. If valid=true, use an empty string."
}}

Validation rules:
- MASK-FRAME ALIGNMENT (VERY IMPORTANT): image 2 must be aligned with image 1. Same width, same height, same coordinate system, and each colored region must mark the subject at the correct (x, y) location in image 1 — not shifted, scaled, mirrored, pasted from another frame, or aligned to any reference image.
- Before accepting any entity id, verify that its colored region in image 2 spatially overlaps the visible target in image 1. Misaligned masks are a critical failure even if the color is present.
- Image 2 must be in strict one-to-one correspondence with image 1: each mask pixel (x, y) must refer to image 1 at (x, y), never to any reference composite.
- This is only an indicative guide, not an exact binary segmentation.
- Use moderate strictness: accept useful approximate masks, but reject obvious missed detections and wrong-target detections.
- A present entity id means image 1 likely contains that target AND image 2 roughly marks that target with its assigned color at the correct location in image 1.
- Accept partial, slightly oversized, or imperfect contours if they still point to the intended target in image 1.
- Reject image 2 immediately if colored mask pixels appear only on the left half or only on the right half of the canvas (split-panel / stitched-output artifact).
- Reject image 2 immediately if it is a side-by-side stitched composite, split-panel layout, or otherwise not a single mask aligned to image 1.
- Reject image 2 if it contains reference-frame pixels, dual panels, or a width/height that matches a reference composite rather than image 1.
- Reject image 2 if colored regions appear to correspond to subjects in a reference composite but not to the same (x, y) locations in image 1.
- Reject an entity id if the target is clearly visible in image 1 but its assigned color is absent or only appears as a tiny meaningless speck.
- Reject an entity id if its color covers a very large unrelated area instead of the target.
- Do not reject just because the boundary is not pixel-accurate or misses small details.
- valid=true only when all obvious visible targets are roughly marked with their assigned colors at the correct locations in image 1, the mask canvas is aligned with image 1, and no assigned color clearly marks the wrong subject or appears on the wrong side of the frame.
- If none of the listed targets are visible in image 1, valid should be false and present_entity_ids should be [].
- If valid is false because the mask missed a visible target or marked the wrong region, provide a better revised_query_prompt.
- The revised_query_prompt must request re-segmentation ONLY from image 1, use exact assigned colors, black background, include only target entity descriptions, and explicitly say: output one single mask in strict one-to-one correspondence with image 1 only; every colored region must align with the target's actual position in image 1; never a stitched reference composite; ignore old mask files; the new mask must NOT reuse any old mask layout.
- If no listed target appears in image 1, revised_query_prompt should be empty because the correct output is a black mask.
"""

MASK_FIRST_DETECTION_ENTITY_VALIDATION_PROMPT = """You are validating a FIRST-TIME entity mask detection for video editing.

Image 1 is the source video frame.
Image 2 is the same frame with a semi-transparent colored overlay showing the candidate indicative mask region for ONE target entity.

Target entity:
- entity_id: {entity_id}
- instruction_id: {instruction_id}
- expected subject_features: {subject_features}
- mask color in overlay: {color_name} ({color_hex})

Task:
Decide whether the colored overlay region in image 2 marks a subject/object in image 1 that MATCHES the expected subject_features.

VIEWPOINT TOLERANCE (IMPORTANT):
- The subject may be back-facing, side-facing, partially occluded, or in a different pose than a frontal reference.
- Accept if the masked region clearly corresponds to the described subject's visible body/object, even without a visible face.
- Do NOT require identical pose or orientation.

REJECT when:
- The colored region marks a clearly different person/object than subject_features.
- The colored region covers mostly background or an unrelated subject.
- The mask is misaligned (shifted/stretched) so it does not cover the intended subject in image 1.
- The described subject is visible in image 1 but the colored region is on the wrong person/object.

ACCEPT when:
- The masked region reasonably covers the described subject in image 1.
- Identity cues (clothing, hair, body build, object shape, etc.) are consistent with subject_features.
- Boundaries may be imperfect; moderate overshoot is OK if the core target is correct.

Return ONLY valid JSON:
{{
  "valid": true,
  "matches_subject_features": true,
  "confidence": 0.0,
  "feedback": "brief explanation"
}}

valid=true only if matches_subject_features=true AND confidence>=0.7.
"""

ENTITY_VISIBILITY_WITH_REF_PROMPT = """You are checking whether a target subject appears in a video frame.

Target subject description: {subject_features}
Edit action: {action}

A reference image of this same subject from an earlier frame is attached (second image).
The reference shows the subject under a semi-transparent colored indicative mask overlay.
Use it only to recognize the same person/object — the subject may look different due to pose, angle, lighting, or partial occlusion.

Return ONLY valid JSON:
{{
  "visible": true,
  "confidence": 0.0,
  "reasoning": "brief explanation"
}}

Rules:
- "visible" true if the subject in the description likely appears in the frame, even partially.
- Use the reference overlay image to disambiguate identity — prefer visible=true when the same subject is plausibly present.
- Side profile, back view, partial occlusion, motion blur, or soft focus STILL count as visible when identity cues match.
- Do NOT mark invisible only because the face is not frontal or the frame is slightly blurry.
- For action "add": the host/location entity must be present.
"""

ENTITY_REFERENCE_COMPARE_PROMPT = """You are selecting the better entity reference image for a video editing pipeline.

Target subject features that the reference must help identify:
{subject_features}

Image 1 = EXISTING reference (saved earlier).
Image 2 = NEW candidate reference (from the latest detected frame).

Both images are reference frames with a semi-transparent colored indicative mask overlay marking the target entity.

Return ONLY valid JSON:
{{
  "better_image": "existing",
  "confidence": 0.0,
  "reasoning": "brief explanation"
}}

QUALITY CRITERIA (evaluate BOTH images; use these as the PRIMARY decision factors, in priority order):

1. SUBJECT-FEATURE MATCH (highest priority)
   - Which marked entity BEST matches subject_features (appearance, clothing, hair, accessories, body build, identity cues)?
   - Reject mentally any image where the overlay marks the wrong person/object.
   - Prefer the image whose visible subject is more consistent with subject_features.

2. FRONTAL FACE (for people / faces)
   - Prefer the image where the target shows a clear frontal or near-frontal face (both eyes and nose visible, face toward camera).
   - A frontal-face view is strongly preferred over profile, back-of-head, or face-not-visible views when identity is otherwise equal.

3. FULL-BODY VISIBILITY (for people)
   - Prefer the image where more of the subject's body is visible (head-to-toe or at least head-through-knees), not a tight face-only or partial crop.
   - Full-body or near-full-body views are preferred when they still clearly mark the correct subject.

4. IN-FRAME PROPORTION / PROMINENCE
   - Prefer the image where the marked target occupies a larger, more prominent share of the frame (higher area under the overlay relative to image size).
   - A subject that is small, distant, or heavily background-dominated is lower quality than one that fills a substantial portion of the frame.

Secondary checks (still required):
- MASK-FRAME ALIGNMENT: prefer the image where the semi-transparent colored overlay is correctly aligned with the target entity in the underlying source frame. Reject mentally any image where the overlay is shifted, stretched, detached, or marks the wrong region.
- Prefer clearer, less occluded, and more distinctive views of the correct subject.

Decision rules:
- Compare image 1 vs image 2 on ALL four primary criteria above before deciding.
- If the new candidate is clearly better on the primary criteria (especially subject-feature match), return "candidate".
- If the existing reference is clearly better or roughly tied on the primary criteria, return "existing".
- When uncertain, return "existing".
- Allowed values for better_image: "existing" or "candidate" only.
"""

MASK_DETECTION_PROMPT = """You are a spatial grounding assistant (fallback when image segmentation unavailable).

Detect tight bounding boxes for each entity in the image.

Entities to find:
{entities}

Return ONLY valid JSON:
{{
  "detections": [
    {{
      "entity_id": "entity_01",
      "bbox": [x_min, y_min, x_max, y_max],
      "confidence": 0.95
    }}
  ]
}}

Coordinates are normalized 0.0-1.0 relative to image width/height.
bbox must tightly wrap the visible subject region in image 1.
If an entity is partially visible, still return a bbox around the visible portion.
If an entity is not visible, omit it from detections.
"""

QA_VALIDATION_PROMPT = """You are a VLM quality inspector for localized video frame edits.

Original and edited images are attached (original first, edited second).

Edit goal: {edit_prompt}
Expected subject features: {subject_features}
Success criteria from instruction rewrite:
{success_criteria_prompt}

Check:
1. The edited image satisfies the success criteria above
2. Edit matches the prompt and applies to the correct target
3. Background outside edit region is preserved
4. Letterboxing / pillarboxing / black bars / canvas framing match the original
5. No major artifacts, crop/recomposition, stretch-to-fill, or identity drift

Return ONLY valid JSON:
{{
  "passed": true,
  "score": 0.0,
  "feedback": "short explanation"
}}

score is 0.0-1.0. passed is true only if score >= 0.7.
"""

ENTITY_VISIBILITY_PROMPT = """You are checking whether a target subject is visible in a video frame.

Target subject: {subject_features}
Edit action: {action}

Return ONLY valid JSON:
{{
  "visible": true,
  "confidence": 0.0,
  "reasoning": "brief explanation"
}}

Rules:
- "visible" true if the described subject (or host person for "add" actions) appears in the frame.
- For action "delete" or "modify": the target entity must be clearly present.
- For action "add": the host/location entity (person or surface where something will be placed) must be present.
- If uncertain, prefer visible=false.
"""

DELETE_QA_VALIDATION_PROMPT = """You are a VLM quality inspector for object removal edits.

Original and edited images are attached (original first, edited second).

Edit goal: {edit_prompt}
Target to remove: {subject_features}
Success criteria from instruction rewrite:
{success_criteria_prompt}

Check:
1. If the target was NOT in the original image → passed=true (nothing to remove, no-op success).
2. If the target WAS in the original → edited image must satisfy the success criteria above.
3. The target must be removed with plausible background inpainting.
4. Background outside removal region preserved.
5. Letterboxing / pillarboxing / black bars / canvas framing match the original.

Return ONLY valid JSON:
{{
  "passed": true,
  "score": 0.0,
  "feedback": "short explanation",
  "target_was_present": true
}}

passed is true if (target absent in original) OR (target removed successfully). score >= 0.7 when passed.
"""

REF_IMAGE_PROMPT = """Generate a high-quality reference image on a plain white background.

Isolated subject ONLY (nothing else): {ref_subject}

STRICT requirements:
- Show ONLY the isolated subject described above — centered, single object
- Plain white background (#FFFFFF), no environment, no scene
- Do NOT include any person, human body, face, hands, shoulders, or body parts
- Do NOT include placement context (e.g. no "on shoulder", no room, no outdoor scene)
- Do NOT include text or watermark
- Match the visual art style of the source video: if the source is photorealistic, generate
  photorealistic; if anime/cartoon/3D animation/stylized, generate in that same style.
  Do NOT convert between art styles.
{forbidden_clause}
"""

REF_IMAGE_QA_PROMPT = """You are validating a T2I reference image for video editing.

The reference must contain ONLY this isolated asset on a white background:
  {ref_subject}

Action type: {action}

FORBIDDEN elements (image must NOT contain any of these):
{forbidden_list}

Check:
1. Image shows ONLY the isolated asset — not a full scene or person
2. No forbidden elements listed above
3. White or plain neutral background
4. Asset matches the ref_subject semantics (e.g. yellow duck → must be a duck)

Return ONLY valid JSON:
{{
  "passed": true,
  "score": 0.0,
  "feedback": "short explanation",
  "violations": ["list of forbidden things found, empty if none"]
}}

passed is true only if score >= 0.75 and violations is empty.
"""

KEYFRAME_PRESERVE_FRAME_STRUCTURE_CLAUSE = (
    "Do not change the video frame structure. If the video frame has black bars on the "
    "top/bottom or left/right (letterboxing or pillarboxing), preserve them exactly as in image 1."
)

KEYFRAME_DELETE_REMOVAL_CLAUSE = (
    "DELETE operations: remove ONLY the scoped target entity and naturally inpaint the "
    "occluded area using surrounding scene content (walls, floor, props, lighting). "
    "All pixels outside the delete target must stay identical to image 1. "
    "Never replace the scene, canvas, or background with flat white, gray, or blank fill."
)

KEYFRAME_INPAINT_OUTPUT_BASE_CLAUSE = (
    "OUTPUT BASE (HIGHEST PRIORITY): image 1 is the TARGET VIDEO KEYFRAME (first frame of "
    "this shot) — the ONLY output canvas. Start from image 1 pixel-for-pixel and apply edits "
    "ONLY inside the scoped targets. The output must be this video frame with localized "
    "changes — NOT a reference frame, NOT a before/after comparison card, NOT a collage. "
    "NEVER use any attached reference image (image 2+) as the output frame, background, "
    "layout template, or full-frame substitute. "
    "Before/after comparison reference (if attached): a left-right card with two panels — "
    "each panel shows ONLY the mask-bounded rectangular crop of the entity (not the full "
    "scene). Left (Original) = entity crop before edit; right (Edited) = same crop region "
    "after applying the instruction, or empty (Removed) for delete — the empty right panel "
    "is reference-only and must NOT be copied into image 1. Use the left panel to "
    "identify WHO to edit in image 1; for modify/add borrow ONLY the edited attribute from "
    "the right panel and apply it onto the matching target in image 1 — never output the "
    "comparison card itself. "
    "Other legacy references (src / mask / overlay, if attached) help identify WHO to edit. "
    "Apply the ORIGINAL EDIT INSTRUCTIONS onto image 1."
)

KEYFRAME_UNRELATED_CONTENT_CLAUSE = (
    "Do NOT modify unrelated content. Change ONLY the specific attribute named in each "
    "instruction on the scoped target; keep every other attribute of that target identical "
    "to image 1 (e.g. when changing hair color or hairstyle, facial expression, eyes, "
    "mouth, skin tone, pose, and clothing must remain unchanged). Never alter background, "
    "other people, props, lighting, or any region outside the edit target."
)

INPAINT_PROMPT = """Use the indicative mask guide in image 1 (colored regions on a black background).
Edit the scene frame in image 2 according to the region-specific instructions below.

{edit_directives}

Strength: {strength}

""" + KEYFRAME_INPAINT_OUTPUT_BASE_CLAUSE.replace("image 1", "image 2").replace(
    "image 2+", "image 3+"
) + """

Rules:
- Each colored mask region in image 1 indicates WHERE to edit on image 2
- Apply only the edit that matches each color region and target subject
- Mask regions are indicative edit regions — prioritize natural, photorealistic integration
- Match lighting, perspective, and style of image 2
- Optional reference image (if attached) is for target identity / attribute appearance ONLY — never the output base
- Do NOT produce flat pasted compositing — blend edits naturally into the scene

FRAME STRUCTURE (CRITICAL):
- Output MUST have the EXACT same width, height, aspect ratio, and canvas layout as image 2.
- Preserve ALL letterboxing / pillarboxing / black bars / blank margins exactly as in image 2.
- Do NOT crop, zoom, reframe, stretch, or remove black borders.
- Do NOT expand the active picture content to fill the full canvas.
- Only modify pixels inside the masked edit regions; leave every unmasked pixel unchanged.
{entity_ref_section}
{consistency_refs_section}
"""

KEYFRAME_ENTITY_REF_GUIDES_SECTION = """
ENTITY REFERENCE GUIDES (identity / edit hints only — NOT the output frame):
Image 2 is the target video keyframe (mandatory output base). Attached references are auxiliary.

{guide_blocks}

Reference usage rules:
- Before/after comparison card (preferred, if attached): each panel is the mask-bounded
  rectangular crop of the entity only (not the full scene). Left (Original) = entity crop
  before the instruction; right (Edited) = same crop after the instruction, or empty
  (Removed) for delete. Caption shows instruction_id and entity_id. Identify WHO to edit
  from the left panel in image 2. For modify/add, borrow ONLY the edited attribute from the
  right panel.
- Delete-target identification crop (if attached): mask-bounded crop showing WHO to remove —
  NOT an output template. Never whiten or blank the full scene because of this reference.
- Delete: use the left entity crop only to recognize which entity to remove in image 2;
  right panel is intentionally empty on comparison cards only (do not copy that emptiness).
- """ + KEYFRAME_DELETE_REMOVAL_CLAUSE + """
- Legacy src / mask / overlay (fallback only): recognize the same subject in image 2.
- NEVER paste, output, or reconstruct from any reference. Image 2 is always the output base.
"""

INPAINT_CONSISTENCY_REFS_SECTION = """
CROSS-SCENE CONSISTENCY REFERENCES (attribute hints only):
The additional attached image(s) show how the SAME edit was applied on an earlier frame.
{consistency_lines}
- Borrow ONLY the edited attribute / object appearance from those references.
- Apply that change onto the matching target in image 2 — do NOT output the reference frame itself.
- Do NOT copy reference background, layout, crop, or canvas framing. Image 2 is the output base.
"""

KEYFRAME_EDIT_ENTITY_LOCATION_PROMPT = """You are locating edit targets on a video frame for localized keyframe editing.

Image 1 is the TARGET video frame that will be edited.
Image 2+ are ENTITY REFERENCE frames from earlier moments. Each reference shows the target entity
under a semi-transparent colored mask overlay (the colored film marks WHO was selected before).

For each instruction below:
1. Use the reference image to learn WHO the entity is (identity).
2. Use image 1 ONLY to find WHERE that same entity appears in the target frame.
3. Write a precise location_edit_prompt naming the exact subject and its position in image 1.
4. List edit_includes (only the target's own body/parts) and edit_excludes (adjacent separate objects).
5. Do NOT use any separate mask guide image — locate the subject directly in image 1.

PRECISE EDIT BOUNDARY (CRITICAL):
- location_edit_prompt must identify the target subject narrowly — not a vague region or whole cluster.
- Name the subject with distinctive identity cues AND spatial position in image 1.
- edit_includes: list only pixels/parts that belong to the target subject itself (e.g. "man's torso and red shirt").
- edit_excludes: list every separate object, prop, furniture piece, or person that touches, overlaps, or is adjacent to the subject but is NOT the edit target (e.g. "wooden handrail he holds", "backpack on his shoulder", "woman beside him").
- boundary_notes: one sentence stating the edit must NOT remove or alter adjacent/touching non-target objects.
- When the subject touches another object, explicitly say the touching object is excluded from the edit.

VIEWPOINT & POSE INVARIANCE:
- The subject may appear with different pose, back/side view, occlusion, motion blur, soft focus, or scale than the reference.
- Match identity via clothing, hair, body build, accessories, object shape — NOT identical pose or face orientation.
- Side profile, back view, 3/4 angle, partially turned away, or slightly blurry subjects STILL count as present when identity cues match.
- Do NOT answer present_in_frame=false only because the face is not frontal, the image is soft/blurry, or the pose differs from the reference.

CONFIDENCE SCORING (IMPORTANT):
- confidence measures identity match in image 1, NOT image sharpness or frontal-face quality.
- identity_cues alone do NOT prove the subject is in image 1 — they may come from the reference only.
- If the subject is NOT in image 1, set present_in_frame=false and leave location fields empty or "none".
- If the same subject is plausibly visible in image 1, use present_in_frame=true and provide concrete spatial fields.
- Use confidence 0.55-0.75 for side/back view, partial occlusion, or mild blur WITH a real location in image 1.
- Use confidence 0.75-0.90 for clear identity match with different pose and a concrete location.
- Reserve confidence below 0.5 or present_in_frame=false when the subject truly cannot be found in image 1.
- Never output confidence=0.0 together with present_in_frame=true.
- Never copy reference-only identity into image 1 when the subject is absent from image 1.

MULTI-INSTRUCTION UNIQUENESS (CRITICAL):
- Multiple instructions in one batch must each map to a DIFFERENT physical person/object in image 1.
- Never assign two different instruction_id values to the same individual.
- Use entity_id, subject_features, and the per-instruction reference image to disambiguate similar-looking people.
- If two subjects could be confused, state distinguishing cues in location_edit_prompt and identity_cues.
- Add distinguishes_from_other_instructions when needed: list other instruction_id values this target is NOT.

Entities to locate:
{entity_list}

Rules:
- spatial_region / landmark_relations / location_edit_prompt must describe image 1 only.
- location_edit_prompt must be specific enough that an editor would NOT confuse the target with nearby objects.
- Do NOT describe edit operations in location_edit_prompt — location and boundary only.
- Do NOT copy reference-frame coordinates into image 1.

Return ONLY valid JSON:
{{
  "locations": [
    {{
      "instruction_id": "instr_01",
      "entity_id": "entity_01",
      "present_in_frame": true,
      "confidence": 0.85,
      "identity_cues": ["blue denim jacket", "short dark hair"],
      "viewpoint_in_detection": "back view, upper body visible",
      "visible_body_parts": ["back", "shoulders"],
      "spatial_region": "center-left midground",
      "landmark_relations": ["standing in front of the glass door"],
      "location_edit_prompt": "the man in the blue denim jacket in the center-left midground, back facing camera",
      "edit_includes": ["the man's body, blue denim jacket, hair, and shoulders"],
      "edit_excludes": ["the glass door behind him", "the metal handrail to his right", "the passerby on his left"],
      "boundary_notes": "Edit only the man's silhouette; do not remove or alter the door, handrail, or passerby even where they touch or overlap the subject.",
      "distinguishes_from_other_instructions": ["instr_02 woman in white dress on the right"]
    }}
  ]
}}
"""

KEYFRAME_EDIT_ENTITY_LOCATION_DISAMBIGUATE_PROMPT = """You are re-locating ONE edit target in a frame where another instruction may have been assigned to the wrong person.

Image 1 is the TARGET video frame that will be edited.
Image 2 is the REFERENCE overlay for the instruction you must locate.

Target entity (locate ONLY this one):
{entity_block}

Other instructions already assigned in this same frame — do NOT map the target above to any of these people/objects:
{peer_assignments}

Task:
1. Find the subject for the target entity in image 1 using its reference image and subject_features.
2. Verify it is a DIFFERENT physical person/object from every peer assignment listed above.
3. If the target subject is not in image 1, set present_in_frame=false.
4. If present, write a precise location_edit_prompt plus edit_includes, edit_excludes, boundary_notes.
5. Set distinguishes_from_other_instructions to explain how this target differs from the peer assignments.

Return ONLY valid JSON with the same schema as a single location record:
{{
  "instruction_id": "{instruction_id}",
  "entity_id": "{entity_id}",
  "present_in_frame": true,
  "confidence": 0.7,
  "identity_cues": ["..."],
  "viewpoint_in_detection": "...",
  "visible_body_parts": ["..."],
  "spatial_region": "...",
  "landmark_relations": ["..."],
  "location_edit_prompt": "...",
  "edit_includes": ["..."],
  "edit_excludes": ["..."],
  "boundary_notes": "...",
  "distinguishes_from_other_instructions": ["..."]
}}
"""

KEYFRAME_EDIT_ENTITY_LOCATION_REID_PROMPT = """You are re-identifying ONE edit target across a viewpoint or quality change for keyframe editing.

Image 1 is the TARGET video frame that will be edited.
Image 2 is a REFERENCE frame where this entity was previously marked with a colored mask overlay.

Target entity:
{entity_block}

The subject may appear in image 1 with a VERY DIFFERENT appearance than the reference:
back turned, side profile, 3/4 angle, partial body, occlusion, motion blur, soft focus, smaller/larger scale, or different lighting.

Task:
1. Decide if the SAME subject from the reference is visible anywhere in image 1.
2. If yes, write a precise location_edit_prompt naming the exact subject and position in image 1.
3. List edit_includes (target body/parts only) and edit_excludes (adjacent separate objects not to edit).
4. Be generous on presence when clothing/body/object identity matches but orientation or sharpness differs.
5. Only answer present_in_frame=false if the subject truly cannot be found in image 1.

BOUNDARY (CRITICAL):
- Explicitly exclude adjacent/touching objects, props, and other people from the edit scope.
- boundary_notes must state that non-target touching objects must be preserved.

CONFIDENCE:
- Do NOT use 0.0 when present_in_frame=true.
- Side view / mild blur with matching identity cues: 0.55-0.75.
- Clear identity match with pose change: 0.75-0.90.

Return ONLY valid JSON:
{{
  "instruction_id": "{instruction_id}",
  "entity_id": "{entity_id}",
  "present_in_frame": true,
  "confidence": 0.65,
  "viewpoint_change": true,
  "identity_cues": ["..."],
  "viewpoint_in_detection": "...",
  "visible_body_parts": ["..."],
  "spatial_region": "...",
  "landmark_relations": ["..."],
  "location_edit_prompt": "...",
  "edit_includes": ["..."],
  "edit_excludes": ["..."],
  "boundary_notes": "..."
}}
"""

KEYFRAME_LOCATION_INPAINT_PROMPT = """Edit the TARGET VIDEO KEYFRAME in image 1. Image 1 is the first frame of this shot from the source video — it is the mandatory output base.

ORIGINAL EDIT INSTRUCTIONS (execute every line exactly; highest semantic priority):
{edit_directives}

Strength: {strength}

""" + KEYFRAME_INPAINT_OUTPUT_BASE_CLAUSE + """

Rules:
- Image 1 = target video keyframe = the ONLY output canvas. Do NOT output any reference frame,
  before/after comparison card, or entity-ref panel.
- """ + KEYFRAME_PRESERVE_FRAME_STRUCTURE_CLAUSE + """
- """ + KEYFRAME_UNRELATED_CONTENT_CLAUSE + """
- NON-SELECTED REGIONS MUST STAY UNCHANGED: preserve every pixel outside each scoped edit target exactly as in image 1.
- Follow the ORIGINAL EDIT INSTRUCTIONS above literally; location text only helps find the target.
- Multiple instructions must each apply to a DIFFERENT person/object — never merge targets.
- Attached references (if any): identity and edit-outcome hints ONLY — never the output frame.
- Before/after comparison card (if attached): each panel is the mask-bounded rectangular
  crop of the entity only (not the full scene). Left (Original) = entity crop before edit;
  right (Edited) = same crop after edit, or empty for delete. Identify WHO from the left
  panel; for modify/add borrow edited appearance from the right panel onto the target in image 1.
- """ + KEYFRAME_DELETE_REMOVAL_CLAUSE + """
- Do NOT paste, output, or reconstruct from reference images. Blend edits naturally into image 1.

FRAME STRUCTURE (CRITICAL — NON-NEGOTIABLE):
- Output MUST have the EXACT same width, height, aspect ratio, and canvas layout as image 1.
- If image 1 has letterboxing / pillarboxing (black bars above/below or left/right), the output MUST keep those bars with the SAME thickness, position, and color.
- Do NOT crop, zoom, reframe, stretch, expand active picture content, or remove black borders.
- Preserve ALL letterboxing / pillarboxing / black bars / blank margins exactly as in image 1.
- Leave every region outside the described edit locations completely unchanged — pixel-for-pixel identical to image 1 except inside the scoped edit targets.
{entity_ref_section}
{consistency_refs_section}
"""

KEYFRAME_EDIT_QA_VALIDATION_PROMPT = """You are a VLM quality inspector for localized keyframe edits in a video-editing pipeline.

VLM inputs (in order):
- Image 1 = EDITED video first frame (candidate result to validate).
- Image 2 = ORIGINAL video first frame (before any edit).
- Text below = edit instruction(s) that should have been applied to image 2 to produce image 1.

Edit instruction(s):
{edit_prompt}

Optional success criteria:
{success_criteria_prompt}

Your job is to decide whether image 1 is an acceptable edited version of image 2.
Focus on these priorities IN ORDER:

1. EDIT SUCCESS & INSTRUCTION COMPLIANCE (CRITICAL PRIORITY - DOUBLE CHECK THIS)
   - You MUST verify that every required edit is visibly and correctly completed in image 1.
   - If the edit is a removal/delete, the target MUST be completely removed with clean inpainting.
   - If the edit is a modification/addition, the target must have the exact requested changes.
   - Any incomplete edits, skipped instructions, or wrong attributes must result in immediate fail (edit_completed=false, passed=false).

2. BACKGROUND & NON-EDIT REGIONS PRESERVABILITY (CRITICAL PRIORITY - DOUBLE CHECK THIS)
   - You MUST heavily check that everything outside the scoped edit regions remains identical to image 2.
   - Background scenery, walls, doors, pillars, other characters, unedited clothes, and unedited lighting must be perfectly preserved without any drift or color distortion.
   - Distinguish a tiny local seam directly touching the edit boundary from a real non-edit-region change. A tiny seam/halo immediately adjacent to the edited silhouette may be rated `trace`; any clearly visible changed patch, recolored non-target region, shifted object/person, or repeated texture away from the edit boundary must be rated `moderate` or `severe`.
   - Any unnecessary changes, background drift, scenery modification, or color bleeding in unedited zones must result in a strict penalty / failure (passed=false, score reduced).

3. OVERALL FRAME STRUCTURE
   - Same full video keyframe: width, height, aspect ratio, camera angle, framing, canvas layout.
   - Letterboxing / pillarboxing / black bars (if any) must be identical in position and thickness.
   - REJECT immediately if image 1 is a crop, zoom, reframe, collage, side-by-side comparison card,
     reference panel, or any layout that is not one single full-frame video keyframe matching image 2.

4. TARGET ENTITY POSITION & IDENTITY
   - The entity(ies) named in the edit instruction(s) must remain in the SAME spatial region /
     screen position as in image 2 (unless the instruction explicitly requires moving them).
   - Edits must apply to the CORRECT target(s), not a wrong person/object or a duplicated subject.
   - Do not accept results where the edited entity drifted, shrunk, or was pasted elsewhere.

5. ARTIFACTS
   - No flat blank replacement of the whole scene, heavy smearing, obvious paste seams, or corruption.

Return ONLY valid JSON:
{{
  "passed": true,
  "score": 0.0,
  "edit_completed": true,
  "frame_structure_preserved": true,
  "background_unedited_regions_preserved": true,
  "unrelated_edit_changes_absent": true,
  "non_edit_region_change_severity": "none",
  "non_edit_region_change_summary": "",
  "failed_aspects": [],
  "edit_errors": [],
  "feedback": "short summary of pass or fail",
  "retry_focus_prompt": "",
  "positive_prompt": ""
}}

Field rules:
- passed=true ONLY IF frame_structure_preserved=true AND edit_completed=true AND background_unedited_regions_preserved=true AND unrelated_edit_changes_absent=true AND score>=0.7.
- score: 0.0–1.0 overall quality (structure + preservation + instruction compliance).
- background_unedited_regions_preserved=false when any visible non-edit background/object region differs from image 2 beyond a tiny seam directly adjacent to the edit boundary.
- unrelated_edit_changes_absent=false when any non-target person/object or any non-requested target attribute changes.
- non_edit_region_change_severity must be one of: `none`, `trace`, `moderate`, `severe`.
  Use `trace` only for a tiny local seam/halo touching the exact edit boundary with no recognizable non-edit content changed.
  Use `moderate` or `severe` for clearly visible changed non-edit patches, non-target people/objects, or broader repainting/drift.
- If non_edit_region_change_severity is `moderate` or `severe`, passed must be false.
- non_edit_region_change_summary: short English summary of what non-edit region changed; empty if severity=`none`.
- failed_aspects: concise labels for each failure (e.g. "frame cropped", "wrong entity edited",
  "target moved off-screen", "background sky changed unnecessarily", "instruction not applied").
- edit_errors: concrete description of what went wrong in THIS attempt (the editing mistakes observed).
  One string per error. Empty list if passed.
- feedback: one short paragraph summarizing the QA verdict; if failed, state the main editing errors.
- retry_focus_prompt (REQUIRED when failed; empty when passed):
- positive_prompt (REQUIRED when failed; empty when passed): list the editing operations that were done CORRECTLY in this attempt and should be KEPT/MAINTAINED on the next retry. Phrase as positive instructions.
  List editing operations / outcomes the image editor must AVOID on the next retry — i.e. the mistakes
  from this attempt that must NOT happen again. Phrase as prohibitions, NOT as new edit goals.
  Examples: "Do not crop or resize the frame", "Do not move the edited person to a different location",
  "Do not change the sky/background outside the target", "Do not leave the shirt red when it should be blue".
  Base retry_focus_prompt on edit_errors, non_edit_region_change_summary, and failed_aspects. Do NOT tell the editor what to do positively;
  only what to avoid repeating.
"""

KEYFRAME_DELETE_EDIT_QA_VALIDATION_PROMPT = """You are a VLM quality inspector for keyframe object removal.

VLM inputs (in order):
- Image 1 = EDITED video first frame (candidate result).
- Image 2 = ORIGINAL video first frame (before edit).
- Text below = delete instruction and target description.

Edit instruction: {edit_prompt}
Target to remove: {subject_features}
Success criteria: {success_criteria_prompt}

Focus IN ORDER:

1. TARGET REMOVED AT CORRECT LOCATION (CRITICAL SUCCESS CHECK):
   - You MUST ensure the target was present in image 2 and is now completely gone from the same region in image 1.
   - The removed area must be replaced with plausible, clean, inpainted background scenery matching the surroundings.
   - If the target is still visible, even partially, it must fail immediately (edit_completed=false, passed=false).

2. BACKGROUND & NON-TARGET REGIONS PRESERVED (CRITICAL PRESERVATION CHECK):
   - You MUST strictly check that everything outside the removal region is perfectly preserved.
   - Distinguish a tiny local seam directly touching the inpaint boundary from a real non-edit-region change. A tiny seam may be rated `trace`; any clearly visible changed patch, recolored non-target region, shifted object/person, or broader repaint must be rated `moderate` or `severe`.
   - Do NOT tolerate any unnecessary changes, repainting, color changes, or shifting of unedited background, walls, floor, pillars, or other people.

3. OVERALL FRAME STRUCTURE: image 1 must be the same full video keyframe as image 2 — same canvas size,
   framing, letterboxing/black bars. REJECT crops, collages, or reference cards.

4. INSTRUCTION COMPLIANCE: delete instruction satisfied for the correct subject.

Return ONLY valid JSON:
{{
  "passed": true,
  "score": 0.0,
  "edit_completed": true,
  "frame_structure_preserved": true,
  "background_unedited_regions_preserved": true,
  "unrelated_edit_changes_absent": true,
  "non_edit_region_change_severity": "none",
  "non_edit_region_change_summary": "",
  "failed_aspects": [],
  "edit_errors": [],
  "feedback": "short summary",
  "retry_focus_prompt": "",
  "target_was_present": true
}}

Field rules:
- background_unedited_regions_preserved=false when any visible non-edit background/object/person region differs beyond a tiny local seam directly touching the removal boundary.
- unrelated_edit_changes_absent=false when any non-target person/object is changed or when the removal alters regions outside the intended removal scope.
- non_edit_region_change_severity must be one of: `none`, `trace`, `moderate`, `severe`.
  Use `trace` only for a tiny local seam/halo at the removal boundary with no recognizable non-edit content changed.
  Use `moderate` or `severe` for clearly visible changed non-edit patches, non-target people/objects, or broader repainting/drift.
- If non_edit_region_change_severity is `moderate` or `severe`, passed must be false.
- retry_focus_prompt when failed: list editing mistakes to AVOID on retry (prohibitions only), e.g.
- positive_prompt (REQUIRED when failed; empty when passed): list the editing operations that were done CORRECTLY in this attempt and should be KEPT/MAINTAINED on the next retry. Phrase as positive instructions.
  "Do not leave the deleted person visible", "Do not replace the entire scene with a blank frame",
  "Do not crop or change frame layout", "Do not alter unrelated background objects".
- edit_errors: concrete mistakes in this attempt; empty if passed.

If target was NOT in image 2 → passed=true, edit_completed=true, frame_structure_preserved=true,
edit_errors=[], retry_focus_prompt="".
"""

# ── Veo / FL-I2V safety framing ─────────────────────────────────────────────
# Prepended to the final video-model payload to reduce false-positive safety blocks
# (e.g. celebrity / realistic-person-likeness code 15236754) on user-edited keyframes.

VEO_CONTENT_SAFETY_PREAMBLE = """CONTENT CONTEXT FOR SAFE GENERATION:
This is a lawful, private video-editing task on the user's own casually captured footage (handheld home video / everyday recording style). The attached first and last frames are edited keyframe anchors from that user clip—not movie stills, not celebrity photos, not news footage, and not copyrighted studio material.
All people shown are generic original individuals or background extras in ordinary clothing; they are not celebrities, public figures, or identifiable real persons.
The scene is family-safe and contains no violence, sexual content, minors in unsafe contexts, hate, gore, weapons, or dangerous activity.
Generate only a natural, innocent motion continuation between the two anchor frames for this editing workflow."""

VEO_ANCHOR_FRAME_INSTRUCTION = (
    "Use the provided edited first frame as the exact visual starting frame and the "
    "edited last frame as the exact visual ending frame."
)

VIDEO_EDIT_FIRST_FRAME_CONSISTENCY = (
    "FIRST-FRAME CONSISTENCY (highest priority): The output video's frame 0 must be "
    "visually identical to the attached reference edited frame — same composition, "
    "edited targets, colors, lighting, and layout. The reference image is the exact "
    "opening-frame anchor; the generated clip must start from that visual state and "
    "continue natural motion from there. Do not drift, fade, cross-fade, or revert "
    "any edit on the first frame."
)

VEO_PROMPT_STYLE_RULES = """VEO SAFETY & STYLE RULES (follow strictly in your output):
- Frame the clip as casual user-shot footage, not a theatrical film, trailer, or blockbuster.
- Describe people generically (e.g. "a person in a cream blouse", "background passengers")—never as actors, celebrities, stars, or named characters.
- Avoid movie-industry language: no "cinematic", "high-budget", "period drama", "film grain", "Hollywood", studio names, or film titles.
- Keep facial detail minimal; emphasize clothing, pose, motion, and environment instead of unique facial identity.
- Do not claim the footage depicts any real public figure or copyrighted performance.
- Keep tone neutral, documentary, and everyday; motion should look like natural handheld continuity."""

VIDEO_EDIT_DIFF_PROMPT = """You are analyzing a keyframe edit before video propagation.

Image 1 is the ORIGINAL first frame (before any edit).
Image 2 is the EDITED first frame (after the keyframe edit).

Use the reference JSON below (from Module 3 location/inpaint planning) to understand what edits were intended, where each target is, and what must NOT change. Compare image 1 vs image 2 and describe ONLY the edits that were actually performed in image 2.

Reference JSON:
{location_reference_json}

Rules:
- Ground your answer in visible differences between image 1 and image 2.
- Use the JSON for target identity, spatial region, edit_includes, edit_excludes, and boundary_notes.
- If the JSON contains ``planned_edits``, those edit operations are MANDATORY: include every planned edit in your output even when image 2 does not yet show the full result (e.g. a DELETE target still visible in image 2).
- Edit instructions always override preservation language: never say to keep, preserve, or leave unchanged any target marked delete in planned_edits.
- If multiple instructions were applied, describe each distinct edit clearly.
- Do NOT invent changes beyond planned_edits and visible evidence in image 2.
- Explicitly note what must remain unchanged only outside the edited targets.
- Use generic, everyday descriptions for people (clothing/pose), not celebrity or movie-character language.

Return JSON only:
{{
  "edit_operation_prompt": "Concise English description of the performed edit(s): targets, actions taken, and preserved regions."
}}
"""

VIDEO_SCENE_STORY_ANALYSIS_PROMPT = """You are analyzing a source video clip before image-to-video regeneration in a private editing workflow.

The clip is the user's own casually captured footage (handheld / home-video style). Describe it so another model can regenerate the same shot starting from only the first frame.

""" + VEO_PROMPT_STYLE_RULES + """

Include:
- Overall action and the visible sequence of events from beginning to end.
- Camera behavior (mostly static or gentle handheld), framing, shot scale, and composition.
- Generic subject descriptions: clothing, pose, gesture, motion, and interactions—avoid unique facial identity or celebrity-like wording.
- Background, objects, environment, lighting, color, atmosphere, and everyday visual style.
- Motion continuity: what keeps moving, what stays mostly static, and how the scene evolves.
- Constraints to preserve: aspect ratio, black bars, scene layout, and natural realism.

Do NOT mention that you are analyzing frames. Do NOT invent events that are not visually supported.
Do NOT use movie-production or celebrity language.
Return a detailed English prompt only.
"""

VIDEO_EDIT_I2V_REWRITE_PROMPT = """You are rewriting a source-video story prompt for first-and-last-frame image-to-video generation in a private, lawful editing workflow.

The I2V model will receive the EDITED first frame and EDITED last frame as visual anchors from the user's own casual footage. Rewrite the story prompt so the generated video follows the same motion, camera behavior, timing, and scene continuity as the original, while consistently preserving the edits visible in both anchor frames.

""" + VEO_PROMPT_STYLE_RULES + """

Original source-video story prompt:
{story_prompt}

Performed edit operation:
{edit_operation_prompt}

Rules:
- Output English only.
- The edited first frame is the exact visual starting state; the edited last frame is the exact visual ending state.
- Preserve the original clip's action, camera motion, subject motion, pacing, composition, lighting, and natural realism.
- Apply the performed edit consistently throughout the whole clip; do not let edited targets revert.
- Do not add new edits beyond the performed edit.
- Preserve all unedited people, objects, background, black bars, layout, and regions.
- Avoid pasted, flat, warped, duplicated, or flickering artifacts.
- Describe all people as generic individuals from user-shot footage—not celebrities or copyrighted characters.

Return only the final I2V prompt, no JSON and no markdown.
"""

VIDEO_EDIT_VEO_PROMPT = (
    VEO_CONTENT_SAFETY_PREAMBLE
    + "\n\n"
    + VEO_ANCHOR_FRAME_INSTRUCTION
    + " "
    + "{rewritten_story_prompt}"
)

SEEDANCE_VIDEO_EDIT_PROMPT = (
    VEO_CONTENT_SAFETY_PREAMBLE
    + "\n\n"
    + VIDEO_EDIT_FIRST_FRAME_CONSISTENCY
    + "\n\n"
    + "We edited the first frame of the source video with the following operation: "
    + "{edit_operation_prompt}\n\n"
    + "The edited first-frame image is provided as the visual reference for the opening frame.\n\n"
    + "Use the reference video as motion and scene context. Apply the same edit described above "
    + "consistently across the entire clip. Preserve camera motion, timing, pacing, unedited "
    + "regions, letterboxing / black bars, and scene continuity. "
    + "Do not revert the edit on any frame and do not add edits beyond what is described."
)

GENERIC_VIDEO_EDIT_PROMPT = (
    VEO_CONTENT_SAFETY_PREAMBLE
    + "\n\n"
    + VIDEO_EDIT_FIRST_FRAME_CONSISTENCY
    + "\n\n"
    + "We edited the first frame of the source video with the following operation: "
    + "{edit_operation_prompt}\n\n"
    + "The attached edited first-frame image shows the result of this edit on the opening frame.\n\n"
    + "Please use the edited first frame as the visual reference and extend the same edit "
    + "consistently across the entire video clip. Preserve camera motion, timing, pacing, "
    + "unedited regions, letterboxing / black bars, and scene continuity. "
    + "Do not revert the edit on any frame and do not add edits beyond what is described."
)

VIDEO_EDIT_QA_VALIDATION_PROMPT = """You are a VLM quality inspector for video-edit propagation in a private editing workflow.

VLM inputs (in order):
- Image 1 = REFERENCE edited frame (the target opening-frame anchor from keyframe editing).
- Image 2 = FIRST FRAME extracted from the edited output video clip.
- Text below = edit operation instructions that the video editor should have applied.

Edit operation instructions:
{edit_operation_prompt}

Verify ALL of the following (priority order):
1. FIRST-FRAME CONSISTENCY: image 2 must match image 1 in edited targets, composition, layout,
   colors, and overall opening-frame appearance. Minor motion-blur differences are acceptable;
   missing edits, reverted edits, or a clearly different opening state are not.
2. EDIT TASK COMPLETED: every required edit in the instructions is visible and correctly applied
   in image 2 (including deletions, additions, and modifications).
3. NO major artifacts on the opening frame: no collage layout, duplicated subjects, flat blank
   replacement, or obvious corruption outside the edit scope.

Return ONLY valid JSON:
{{
  "passed": true,
  "score": 0.0,
  "first_frame_consistent": true,
  "edit_completed": true,
  "failed_aspects": [],
  "feedback": "short explanation of pass or fail",
  "retry_focus_prompt": "if failed: list editing mistakes/errors to AVOID on the next retry (undesired outcomes that must NOT happen again, e.g. 'do not leave the deleted person visible', 'do not drift away from the reference opening frame'); empty string if passed",
  "positive_prompt": "if failed: what was done correctly and should be KEPT on retry; empty if passed"
}}

Rules:
- passed=true only if first_frame_consistent=true AND edit_completed=true AND score>=0.7.
- failed_aspects: specific failures (e.g. "duck missing", "deleted person still visible", "opening frame drift").
- retry_focus_prompt must describe editing operations or outcomes to AVOID — not new edit goals.
- positive_prompt (REQUIRED when failed; empty when passed): list the editing operations that were done CORRECTLY in this attempt and should be KEPT/MAINTAINED on the next retry. Phrase as positive instructions.
- Do NOT phrase retry_focus_prompt as additional edits to apply; phrase as mistakes/errors to prevent.
"""

VIDEO_CHUNK_EDIT_DIFF_PROMPT = """You are analyzing visual edits between two consecutive video sub-clips in a private editing workflow.

Image 1 is the ORIGINAL first frame of the current sub-clip (before any edit).
Image 2 is the EDITED last frame from the previous sub-clip (after the edit was applied there).

Compare image 1 vs image 2 and describe ONLY the edits that must be carried forward into the current sub-clip so it stays visually consistent with the previous edited sub-clip.

The reference edited frame (image 2) is the exact opening-frame anchor for the output video: frame 0 of the generated sub-clip must match image 2 in composition, edited targets, and layout.

Rules:
- Ground your answer in visible differences between image 1 and image 2.
- Do NOT invent changes that are not supported by the two images.
- Describe targets, actions taken, and what must remain unchanged outside edited regions.
- Use generic, everyday descriptions for people (clothing/pose), not celebrity or movie-character language.
- Do NOT assume any external instruction list; rely only on the two images.

Return JSON only:
{{
  "edit_operation_prompt": "Concise English description of the edit(s) to apply in the current sub-clip."
}}
"""

KEYFRAME_LOCATION_ENTITY_REF_GUIDES_SECTION = """
ENTITY REFERENCE GUIDES (identity + edit-outcome hints — NOT the output frame):
Image 1 is the target video keyframe (mandatory output base). Images below are auxiliary.

{guide_blocks}

Reference usage rules:
- Before/after comparison card (preferred, if attached): each panel is the mask-bounded
  rectangular crop of the entity only (not the full scene). Left (Original) = entity crop
  before the instruction; right (Edited) = same crop after the instruction, or empty
  (Removed) for delete. Caption shows instruction_id and entity_id. Identify WHO to edit
  from the left panel in image 1. For modify/add, borrow ONLY the edited attribute from the
  right panel.
- Delete-target identification crop (if attached): mask-bounded crop showing WHO to remove —
  NOT an output template. Never whiten or blank the full scene because of this reference.
- Delete: use the left entity crop only to recognize which entity to remove in image 1;
  right panel is intentionally empty on comparison cards only (do not copy that emptiness).
- """ + KEYFRAME_DELETE_REMOVAL_CLAUSE + """
- Legacy src / mask / overlay (fallback only): find the matching subject in image 1.
- NEVER use any reference as the output base, background, crop, or full-frame substitute.
- Execute the ORIGINAL EDIT INSTRUCTIONS on image 1.
"""

SHOT_CLIP_VLM_ANALYSIS_PROMPT = """You are a professional video analyst reviewing ONE physical shot clip.

The clip was produced by automatic hard-cut detection (PySceneDetect). It may still contain
**undetected editorial sub-cuts** inside it — but ONLY true post-production transitions such as
dissolves, cross-dissolves, fades, or wipes between TWO different scene compositions.

SHOT METADATA:
- shot_id: {shot_id}
- scene_id: {scene_id}
- Index: {shot_index} of {shot_total}
- Duration in this clip: {duration_sec:.2f}s
- Position in the FULL source video: {video_start_sec:.2f}s – {video_end_sec:.2f}s (absolute)

The attached images are frames sampled evenly from this clip, in chronological order.

Your tasks:
1. Write a rich **plot_description** (4–8 sentences) of everything that happens in this clip:
   who appears, what they do step-by-step, how the situation evolves, setting/background,
   camera behavior (static/pan/zoom/handheld), lighting/mood, and any dialogue or interaction cues
   visible from the frames.
2. List **keyframes** — narratively important moments. **MANDATORY:** you MUST include an **opening** keyframe at **timestamp_in_shot_sec = 0.0** (the first frame of the clip). This opening keyframe is required even if nothing notable happens yet. For every keyframe provide:
   - description: **2–4 detailed sentences** — describe visible subjects (appearance, clothing, pose),
     the action beat at this instant, camera framing (shot size, angle), background elements,
     lighting, and why this moment matters narratively. Be specific and visual, not generic.
   - timestamp_in_shot_sec (0.0 = first frame)
   - role: e.g. "opening", "action peak", "dialogue beat", "camera move", "closing"
   - Do NOT mark rack-focus / depth-of-field shifts as "transition" keyframes unless a true editorial dissolve exists.
3. Decide **has_undetected_sub_cuts** — DEFAULT **false**. Set true ONLY when you are highly confident
   the clip contains 2+ DISTINCT scene compositions joined by a visible editorial blend (dissolve/fade/wipe),
   not continuous cinematography within one shot.

**NOT a sub-cut (keep has_undetected_sub_cuts=false):**
- Rack focus / pull focus / depth-of-field change (foreground vs background sharpness swap)
- Shallow-focus ↔ deep-focus change, bokeh shift, lens breathing
- Zoom, push-in, pull-out, pan, tilt, dolly, handheld drift
- Subject walking, gesture change, lighting/exposure shift within the same scene layout
- Single continuous location and camera setup with only focus or exposure changing

**IS a sub-cut (rare — require strong evidence):**
- Visible dissolve/cross-dissolve/fade/wipe between two clearly different compositions or locations
- Two separate visual "scenes" with a blended overlap between them

When has_undetected_sub_cuts is true, also provide:
- **sub_cut_detection_confidence**: 0.0–1.0 (only use ≥0.75 when truly certain)
- **sub_cut_rationale**: one sentence explaining why this is editorial, not focus/camera motion
- **false_positive_risks**: list any focus/camera effects you considered and rejected (may be [])
- **undetected_sub_cuts**: each sub-segment (start_sec_in_shot / end_sec_in_shot,
  sub_plot_description: **3–5 sentences** with the same visual/narrative detail as plot_description)
- **transition_boundaries**: precise peak of each editorial blend (preferred over wide zones):
    boundary_sec_in_shot = the single second where the cross-dissolve is strongest
- **transition_zones** (optional narrow backup): only the blended overlap, typically ≤0.5s wide

Return ONLY valid JSON (no markdown fences):
{{
  "plot_description": "...",
  "has_undetected_sub_cuts": false,
  "sub_cut_detection_confidence": 0.0,
  "sub_cut_rationale": "",
  "false_positive_risks": [],
  "undetected_sub_cuts": [],
  "transition_boundaries": [
    {{
      "boundary_sec_in_shot": 3.5,
      "transition_type": "gradual dissolve",
      "confidence": 0.85
    }}
  ],
  "transition_zones": [],
  "keyframes": [
    {{
      "description": "Detailed multi-sentence visual description of this moment: subjects, clothing, pose, action, framing, background, lighting, and narrative significance.",
      "timestamp_in_shot_sec": 0.0,
      "role": "opening"
    }}
  ]
}}

RULES:
- plot_description and every keyframe description must be **rich, specific, and visually grounded** — avoid one-liners.
- **keyframes MUST include one entry with timestamp_in_shot_sec=0.0 and role="opening".**
- timestamp_in_shot_sec must be within [0, {duration_sec:.2f}].
- **Default has_undetected_sub_cuts to false** — false positives are worse than misses.
- If unsure between focus change and dissolve, choose false and explain in false_positive_risks.
- transition_boundaries must pinpoint the blend peak; do NOT span entire focus-rack durations.
- If has_undetected_sub_cuts is false, undetected_sub_cuts, transition_boundaries, transition_zones must be [].
- Use English for all string values.
"""

from video_editing_agent.prompts.templates_recovered import (  # noqa: E402
    DIRECT_SCENE_VIDEO_EDIT_PROMPT,
    DIRECT_SCENE_SEEDANCE_VIDEO_EDIT_PROMPT,
    ENTITY_MULTIVIEW_CANDIDATE_SELECT_PROMPT,
    ENTITY_MULTIVIEW_EDIT_ATTRIBUTE_QA_PROMPT,
    ENTITY_MULTIVIEW_EDIT_PROMPT,
    ENTITY_MULTIVIEW_EDIT_QA_PROMPT,
    ENTITY_MULTIVIEW_EDIT_VIEW_OCCLUSION_QA_PROMPT,
    ENTITY_MULTIVIEW_SOURCE_APPEARANCE_QA_PROMPT,
    ENTITY_MULTIVIEW_SYNTHESIS_PROMPT,
    ENTITY_MULTIVIEW_SYNTHESIS_QA_PROMPT,
    ENTITY_REFERENCE_KEYFRAME_SELECT_PROMPT,
    KEYFRAME_ENTITY_DETECTION_PROMPT,
    SCENE_ENTITY_EXISTENCE_VOTE_PROMPT,
    SCENE_KEYFRAME_GRID_EDIT_PROMPT,
    SCENE_KEYFRAME_GRID_EDIT_QA_PROMPT,
    SCENE_KEYFRAME_GRID_ENTITY_LOCATION_PROMPT,
    SCENE_VIDEO_EDIT_DERIVATION_PROMPT,
    SCENE_VIDEO_EDIT_KEYFRAME_GRID_QA_PROMPT,
    SCENE_VIDEO_EDIT_BEST_ATTEMPT_SELECT_PROMPT,
)

KEYFRAME_SINGLE_ENTITY_PRESENCE_PROMPT = """You are screening ONE edit target in ONE video keyframe with robust person/object identity matching.

Image 1 = TARGET KEYFRAME from the scene (the frame to be edited).
Image 2 = ENTITY IDENTITY SOURCE REFERENCE from entity_refs/instr_00N_ref_src.png showing WHO the target person/object is.

ENTITY TO FIND:
- instruction_id: {instruction_id}
- entity_id: {entity_id}
- subject_features: {subject_features}
- edit_prompt: {edit_prompt}

{prior_detection_block}

TWO-STEP WORKFLOW (MANDATORY — follow this order strictly, do not skip steps):

STEP 1 — LOCATE FIRST in image 1:
Scan image 1 and determine whether entity_id {entity_id} appears. Output location_description for the
precise screen position (region, background anchors, spatial relation to other people/objects). If the
entity is only partially visible (edge crop, small sliver, fragment), describe exactly which part of the
frame edge it occupies and how much of the body is showing. For physical placement edits: presence and
editability are separate decisions — report presence even when the attachment point is not visible.

STEP 2 — VISIBLE PARTS & FEATURE AUDIT AT THE LOCATED POSITION ONLY:
Now inspect ONLY the pixel area described in STEP 1. List every body part you can ACTUALLY see at that
exact position (visible_parts). Never list "face" or "hair" at a position where they are not physically
visible — edge-crop torso/shoulder/arm regions do NOT contain face or hair. Then run the feature audit
below against ONLY those visible pixels.

RULES:
- REFERENCE-PRIMARY IDENTITY (CRITICAL): Images 2+ from entity_refs are the primary identity evidence. Scene story
  context can help explain continuity for a real visible candidate, but it must never override a visual mismatch
  with the reference image. Do not mark a person present merely because they are prominent, central, a narrative
  lead, or named by shots_analysis.
- IDENTITY vs APPEARANCE (CRITICAL): Match the SAME person/object (entity_id), NOT an identical costume across frames. Image 2 may include views from other scenes with different outfits — that is expected. NEVER reject solely because clothing in image 1 differs from a panel in image 2 if the face/body identity matches. Reject look-alikes, background extras, and limb-only fragments — not the same person in a different outfit.
- VISIBLE-PART-ONLY IDENTITY (CRITICAL): compare ONLY the target body regions actually visible in image 1
  against the corresponding regions in the reference. Do NOT use reference clothing/body parts that are
  cropped out or hidden in image 1 as negative evidence. If a visible key region has high similarity to the
  reference, mark present=true even when other reference regions are not visible or current clothing differs.
  Key regions are face, head, hair, hairline, clear side/profile face, or a substantial torso/headwear region
  with distinctive identity cues. A half shoulder, arm/hand sliver, tiny edge crop, or generic clothing corner
  is NOT a key region and cannot make identity_verifiable_from_visible_parts=true by itself.
- SMALL / BLURRY / BACK-VIEW CAUTION (CRITICAL): if the candidate is tiny, blurry, heavily shadowed,
  occluded, or mainly back-facing, lower confidence substantially. If no directly visible face/head/hair/profile
  or other unmistakably unique cue is present, prefer present=false. Generic silhouette, body build, posture,
  or same-region continuity alone are NOT enough.
- FEATURE-BY-FEATURE IDENTITY AUDIT (MANDATORY — decide present ONLY after this audit):
  1) List every STRONG identity feature from subject_features and image 2 (face shape, hair color/style,
     suspenders, vest, hat/cap, dress, beard, glasses, build, gender/body type, distinctive accessories).
  2) For each feature, inspect image 1 pixels only and assign exactly one status in feature_audit:
     - match: clearly visible in image 1 AND matches the target entity/reference.
     - clear_mismatch: clearly visible in image 1 BUT contradicts the target (wrong color, wrong garment,
       wrong accessory, wrong face structure, wrong body type). Example: subject requires suspenders but
       visible torso clearly shows a plain shirt with no suspenders.
     - uncertain_blurry: the feature region is blurry, occluded, off-frame, too small, or otherwise not
       reliably visible — you cannot confirm match OR mismatch.
     - not_visible: feature area is not in frame.
  3) DECISION RULE (strict):
     - If ANY feature has clear_mismatch → present=false, existence_confidence_score=0. One clear contradiction
       is enough; other matching features cannot override it.
     - If the only non-matching features are uncertain_blurry or not_visible, and at least one strong feature
       clearly matches (face/head/hair/profile OR two+ distinctive clothing/accessory matches), present=true
       is allowed.
  4) ANTI-HALLUCINATION (EXTREMELY CRITICAL): Mark "match" ONLY when you can clearly and physically see that feature in image 1. Never mark "match" because subject_features text or the reference image mentions it. Never invent face, hair, suspenders, vest, hat, or dress details that are not visible.
     - Specially, if only a shoulder, arm, or clothing fragment is visible at the edge of the frame, and the head/face/hair is NOT in the frame or is occluded, you MUST set the status of hair, head, and face features to "not_visible" or "uncertain_blurry", and you MUST NOT list "face" or "hair" in `visible_parts`. Listing features or parts that are physically absent from the visible region is a severe hallucination failure.
  5) Mirror clear_mismatch items in identity_conflicts with prefix "clear:" and uncertain items with
     "uncertain/blurry:".
- SPATIAL DISAMBIGUATION: when two entities could overlap at the same screen region (e.g. both described at the extreme left edge), assign the partial figure to the entity whose distinctive clothing/accessories actually match the visible pixels (vest vs suspenders vs cap). Mark present=false for the other entity at that region — never double-assign one cropped limb to two instruction_ids.
- CROSS-ENTITY CANDIDATE ARBITRATION: treat every visible person/object candidate as assignable to at most one
  single-scope instruction_id. If two instruction_ids could describe the same candidate, choose the one with
  stronger current-frame identity evidence and mark the other present=false or list an identity_conflict. Do
  not bind one candidate to both a removal target and a placement target.
- Respect target_instance_scope:
  - single: exactly ONE tracked individual matching subject_features
  - multiple: every instance in the frame matching subject_features
- Compare side/back/3-4 keyframe views against the source reference using stable identity cues
  (face shape, hairline, hairstyle, build, clothing/accessories, body proportions, distinctive props).
- Be robust to scene lighting, facial expression, body motion, walking/leaning/bending/crouching posture,
  close-up framing, wide/full-body shots, side profile, partial face, soft focus, and camera angle changes.
  Do NOT reject the same entity solely because image 1 has different lighting, expression, action, pose,
  scale, or clothing state from image 2.
  Do NOT lower existence_confidence_score for lighting/exposure, shadows, expression, gaze, action, pose,
  walking/leaning/bending/crouching, or other state differences when visible identity cues still match.
- BACK / SIDE PROFILE VALIDITY (CRITICAL): A back view, side profile, or three-quarter view of the entity
  is a VALID detection when stable identity cues (head/hair shape, hairline, body build, body proportions,
  clothing, accessories) are visible and match the reference. Do NOT require a full frontal face. A clear
  back-of-head with matching hair style/silhouette/build is sufficient for present=true.
- EXPRESSION / GAZE / MOUTH STATE: Smiling, frowning, talking, mouth open, eyes closed, looking away —
  these are non-identity motion state. Keep present=true when face structure, hairline, and hair match.
- LIGHTING-INDUCED APPEARANCE SHIFT: Indoor/outdoor, warm/cool color temperature, backlight, shadow,
  overexposure, underexposure can shift apparent hair/skin/clothing tones. Match structural identity
  cues (face shape, hairline, hair silhouette, body build), not exact colorimetry.
- LIGHTING-INDUCED HAIR COLOR SHIFT (CRITICAL): The same person's apparent hair color can shift significantly
  under different lighting conditions and camera distances. A person with medium-brown, dark-blond, or
  warm-brown hair in a close-up/indoor/controlled-light shot may appear noticeably lighter blond in
  outdoor/bright wide shots, or darker brown in shadowed/backlit/soft-focus shots. When facial structure,
  hair silhouette, hairline shape, hair parting, hair wave/curl pattern, and face shape clearly match the
  reference, do NOT reject the identity based on apparent hair color tone difference alone — this is a
  lighting artifact, not an identity mismatch. In the feature_audit, mark hair color as "match" when
  the hair silhouette/style/hairline/wave pattern match even if the tone appears shifted by lighting;
  only mark "clear_mismatch" for hair when the style, silhouette, texture, length, or parting
  fundamentally contradicts the reference. For close-up faces with warmer/darker hair but matching
  facial structure and hairline, keep present=true.
- MISSING-CROPPED FEATURES ≠ IDENTITY CONTRADICTION (CRITICAL): When subject_features mentions features
  that are cropped out, below the frame edge, or otherwise not visible due to camera framing (close-up,
  tight shot, or partial crop), assign feature_audit status "not_visible" or "uncertain_blurry" — NEVER
  "clear_mismatch". A feature that is simply outside the frame (e.g. suspenders on a torso cropped out
  in a close-up face shot; a dress whose skirt is below the frame edge) is absent due to framing, not a
  contradiction of identity. Only mark "clear_mismatch" when the feature IS clearly visible in frame AND
  demonstrably contradicts the reference (e.g. a visible torso shows a plain T-shirt when the reference
  clearly requires visible suspenders on the same torso region). When the face/head/hair is visible and
  matches the reference while other features are "not_visible" due to framing, the entity is present=true.
- ANTI-NAME/ROLE INTERFERENCE (CRITICAL): Do NOT use character names, actor names (e.g. "Jack", "Rose",
  "Leonardo DiCaprio", "Kate Winslet"), known celebrity identities, or narrative story roles as reasons
  to reject an entity. A person matched by name/role to a "main character" or "protagonist" could still
  be the SAME tracked person as the entity in the reference. ONLY compare visible facial features, hair
  structure, hair silhouette, facial landmarks, and visible body/clothing regions between image 1 and the
  reference image (image 2+). The fact that a person is narratively "the lead" or "a different named
  character" must NEVER override visible visual feature matching. If facial structure, hair silhouette,
  face shape, eye/nose/jaw, and hairline match the reference, mark present=true regardless of any
  name/role association. The reference image identity is the ground truth for visual matching.
- BENDING/CROUCHING CANDIDATES: actively inspect small, low, bent-over, crouching, leaning, or partially
  occluded faces/heads/hair. A smaller front/side face with matching hair/face identity is a better target
  than a larger back-view body whose face and identity are not visible.
- For physical placement targets, choose the candidate with the strongest current-frame identity evidence
  (face/head/hair/profile plus stable cues) before judging attachment visibility. Do not bind the entity to
  a large back-view/turned-away shoulder merely because the requested shoulder area is prominent.
- For removal/delete edits: partial edge crops are SUFFICIENT only when a substantial torso/head/cap/hat/vest
  region is visible with distinctive cues. A tiny shoulder sliver, arm sliver, or corner of clothing is NOT
  sufficient, even if it resembles the reference.
- For shoulder/torso accessory edits: partial torso/shoulder is sufficient only when the visible region is large
  enough to edit and identity cues are verifiable; half a shoulder or a tiny cropped fragment is not enough.
- present=true when you are confident the specific entity is visually recognizable. For physical placement,
  if the entity is recognizable but the requested attachment point is hidden or anatomically ambiguous, keep
  present=true and set target_attachment_visible=false rather than marking the entity absent.
- If present=false, location_description must be empty string.

FIELDS:
- entity_visibility_completeness: sufficient | partial | fragment
  - sufficient: enough visible to edit per edit_prompt (face/head ok for hair-hat edits).
  - partial: face/upper body visible but lower body occluded — still acceptable for head edits.
  - fragment: limb-only / edge sliver only.
- visible_parts: body regions visible for the TARGET in image 1.
- identity_verifiable_from_visible_parts: true only when visible key regions (face/head/hair/profile or a
  substantial distinctive torso/headwear region for non-head edits) are sufficient to compare identity. It
  must be false for shoulder-only, arm-only, hand-only, tiny edge-crop, or generic clothing-only fragments.
- visibility_quality: clear | too_small | blurry | ambiguous | occluded
  - occluded is OK when the TARGET face/upper body remains clearly visible despite foreground blur, but
    non-target occlusion/foreground blur must lower existence_confidence_score.
- localization_clarity: high only when the TARGET region is unambiguous.
- existence_confidence_score: integer 0-100 for whether this exact entity exists in image 1.
  - Use a continuous score, not a binary 0/100 score. Avoid extreme 0 or 100 unless the evidence is truly extreme.
  - 0 means definitely absent with no plausible visible candidate for this entity.
  - If any plausible part of the target entity is visible, do NOT use 0; use a low or medium score based on evidence.
  - 90-100 means absolutely present with clear face/head/hair/profile or other unique identity evidence.
  - 70-89 means likely present but partially occluded, small, truly blurry enough to reduce identity clarity,
    or with some viewpoint/crop uncertainty.
  - 40-69 means weak/partial evidence such as distinctive torso/headwear/clothing without clear face/head/hair.
  - 1-39 means only generic clothing, shoulder/arm/hand, edge crop, tiny fragment, or look-alike evidence.
  - Clothing-only or torso/shoulder/arm-only evidence MUST NOT receive 90-100, even if the edit target point is visible.
  - If the target is obscured by non-target people, hands, props, railing/pillars, foreground blur, or other
    non-entity visual interference, reduce the score according to how much identity evidence remains visible.
    Do not lower the score for lighting/exposure/shadow, expression, gaze, action, or pose changes alone.
    Lower for blur only when it genuinely makes visible identity regions less verifiable.
  - Use 100 only for a fully clear, unobstructed, close/large view with unmistakable identity from key identity regions;
    otherwise prefer 70-95 for strong matches instead of 100.
  - If present=false because there is definitely no candidate, existence_confidence_score MUST be 0.
  - If the target is partially visible, blurred, cropped, or uncertain, mark present=true only when identity is still
    plausible/verifiable and assign a non-extreme reduced score.
  - For present=true, use the score to express identity certainty after considering viewpoint, crop, blur,
    clothing changes, and look-alikes.

For non-placement edits, present=true ONLY when the target identity is visually verifiable from image 1 using
the source reference in image 2, localization_clarity=high, and entity_visibility_completeness is sufficient
or partial for the requested edit. A close-up face, side-profile face, distinctive hair/headwear, upper body,
or full-body view can all be sufficient when the identity match is strong. Limb-only fragments and generic
outfit-only matches are not sufficient.

For physical placement edits, present=true depends on identity visibility; target_attachment_visible depends
on whether the exact requested attachment point is currently visible with clear anatomical side.

If present=false, location_description must be empty. Use English for all string values.

Return ONLY valid JSON:
{{
  "instruction_id": "{instruction_id}",
  "entity_id": "{entity_id}",
  "present": false,
  "existence_confidence_score": 0,
  "location_description": "",
  "visibility_quality": "ambiguous",
  "approximate_area_fraction": 0.0,
  "localization_clarity": "low",
  "entity_visibility_completeness": "fragment",
  "visible_parts": [],
  "identity_verifiable_from_visible_parts": false,
  "target_attachment_point": "left_shoulder|right_shoulder|shoulder|hand|body|none",
  "target_attachment_visible": false,
  "attachment_visibility": {{"left_shoulder": false, "right_shoulder": false}},
  "body_orientation": "front|side_profile|three_quarter|back|bending|crouching|unknown",
  "anatomical_left_screen_side": "screen-left|screen-right|hidden|unknown",
  "anatomical_right_screen_side": "screen-left|screen-right|hidden|unknown",
  "attachment_visibility_reasoning": "brief note on whether the requested attachment point is visible",
  "feature_audit": [
    {{
      "feature": "suspenders|hair color|face shape|vest|...",
      "status": "match|clear_mismatch|uncertain_blurry|not_visible",
      "note": "brief visible-pixel justification"
    }}
  ],
  "candidate_evaluations": [
    {{
      "candidate_location": "brief candidate location",
      "visible_parts": ["face", "hair"],
      "identity_matches": ["specific visible identity match"],
      "identity_conflicts": ["clear: wrong required feature, or uncertain/blurry: feature cannot be verified"],
      "decision": "present|reject_lookalike|uncertain"
    }}
  ],
  "reasoning": "brief explanation"
}}
"""

KEYFRAME_BATCH_ENTITY_PRESENCE_PROMPT = """You are locating ALL edit targets in ONE video keyframe in a single pass.

Image 1 = TARGET KEYFRAME from the scene (the frame to be edited).
Images 2+ = per-entity front-view IDENTITY REFERENCE images, one per entity below.

ENTITIES TO DETECT (evaluate EVERY entity; reference image mapping):
{entity_catalog_block}

SCENE TEMPORAL CONTEXT (earlier keyframes in the SAME scene, time order):
{scene_prior_detection_block}

TEMPORAL CONSISTENCY (same continuous scene):
- Use the scene temporal context above. An entity absent in ALL earlier keyframes should remain absent
  unless it clearly enters the frame with strong identity evidence — not merely because a similar-looking
  person was previously rejected as a non-match at the same region.
- If an earlier keyframe explicitly rejected a look-alike at a region (e.g. "man on the left does not
  match"), do NOT mark present=true at that same region in a later keyframe — even with high confidence.
- If earlier keyframes marked this entity absent while describing the SAME distinctive appearance
  (e.g. flat cap + brown vest, railing leaner on the left), do NOT flip to present=true later at the
  same appearance unless visibility genuinely changed (occlusion cleared) with new evidence.
- Do NOT block entities that newly appear in a middle keyframe with undeniable identity: clear face,
  sufficient visibility, high confidence, and distinctive match to the entity reference. Mark present=true
  when those criteria are met, even if the entity was absent in earlier keyframes (as long as no earlier
  keyframe rejected a look-alike at the same region).
- Do NOT flip absent→present at a spatial region that was already evaluated and rejected for this entity.
- If this entity was present=true in earlier keyframes of this scene, it likely remains present unless
  clearly occluded or off-frame — do not mark absent due to minor field wording differences.
- When an entity was present=true in earlier keyframes, keep location_description spatially consistent:
  same anchor object (e.g. wooden railing vs support pole), same screen region (left/center/right),
  and same lean/pose context — do not drift to a different structural element unless the entity clearly moved.
- entity_visibility_completeness must be exactly one of: sufficient | partial | fragment (never "full" or "high").

TASK (TWO-STEP WORKFLOW — follow this order strictly for EACH entity):
STEP 1 — LOCATE: For each entity, first scan image 1 and output a precise location_description for where
the entity appears (screen region, background anchors, relation to other people/objects). If partially
visible (edge crop), describe exactly the visible fragment.
STEP 2 — VERIFY AT LOCATION: After locating, inspect ONLY the pixel area from STEP 1. List visible_parts
that are ACTUALLY visible at that position. Never list "face" or "hair" for edge-crop torso/shoulder/arm
regions. Run the feature_audit strictly against those pixels.
For physical placement edits, presence and editability are separate decisions.
Return one result object per entity.

SHARED RULES (apply to every entity):
- Match each entity ONLY against its OWN reference image (see mapping above). Do NOT assign the same
  partial edge figure to multiple entities, and do NOT confuse different people wearing similar clothing.
- SPATIAL DISAMBIGUATION: when two entities could overlap at the same screen region (e.g. both described
  at the extreme left edge), assign the partial figure to the entity whose distinctive clothing/accessories
  actually match the visible pixels (vest vs suspenders vs cap). Mark present=false for the other entity
  at that region — never double-assign one cropped limb to two instruction_ids.
- Respect target_instance_scope for each entity (see catalog block):
  - single: mark present=true for exactly ONE best-matching instance — the same tracked individual
    from the reference. Do NOT treat every similar-looking extra as the target. location_description
    must refer to that one instance only.
  - multiple: all matching instances in the frame are in scope (rare; only when explicitly requested).
- Each reference image is a SINGLE front-view reference. When image 1 shows a side/back/3-4 view,
  compare against stable identity cues from the front-view reference. Side/back/three-quarter views
  are VALID detections when identity cues (face profile, head/hair shape, hairline, body build,
  clothing, accessories) match — do NOT require a frontal face.
- NON-EDIT FACTOR ROBUSTNESS (CRITICAL — do NOT lower confidence or mark present=false for these):
  - FACIAL EXPRESSION / GAZE / HEAD POSE: expression (smile, frown, talking, blinking), gaze direction,
    and head turn/tilt are motion state, not identity. Keep present=true when underlying identity matches.
  - LIGHTING / EXPOSURE / COLOR TEMPERATURE: backlight, shadow, overexposure, underexposure, warm/cool
    color shifts are environmental. The same person under different lighting is still the same entity.
  - HAIR COLOR SHIFT: apparent hair tone can lighten/darken under different lighting — match hair
    silhouette/style/hairline, not exact tone.
  - MOTION BLUR / SOFT FOCUS: acceptable when enough identity cues remain visible.
  - CLOTHING DIFFERENCES: match the PERSON identity, not the outfit.
  - CAMERA DISTANCE / SHOT SCALE: close-up, medium, wide shots are all valid.
  Do NOT lower confidence for these non-edit factors when stable identity cues still match.
- Match the SAME person/object (entity_id), NOT an identical costume across frames.
- Reference sheets may show outfits from other scenes — NEVER reject solely for clothing mismatch if face/body identity matches.
- VISIBLE-PART-ONLY IDENTITY (CRITICAL): compare only the regions of the candidate that are actually visible
  in image 1 against the corresponding reference regions. Missing/cropped clothing, lower body, or accessories
  from the reference must not count as negative evidence. A high-similarity visible key region is enough for
  present=true; key regions include face, head, hair, hairline, clear side/profile face, or a substantial
  torso/headwear region with distinctive identity cues. A half shoulder, arm/hand sliver, tiny edge crop, or
  generic clothing corner is not a key region and cannot establish identity by itself.
- SMALL / BLURRY / BACK-VIEW CAUTION (CRITICAL): for tiny subjects, soft-focus candidates, heavy shadow,
  back-facing bodies, or distant silhouettes, lower confidence sharply. If there is no directly visible
  face/head/hair/profile or another unmistakably unique cue, keep present=false rather than guessing.
- Strong vs weak features: clothing, hair style/color, hat/cap/headwear, suspenders, vest, beard/stubble,
  and distinctive accessories are STRONG identity features. Pose, action, gaze, and expression are WEAK
  state features. A weak-state mismatch is allowed; a strong-feature contradiction means wrong person.
- If subject_features identify hair (e.g. blond/middle-parted hair) and do NOT mention a hat/cap, do not
  accept a candidate whose matching evidence is a hat/cap that hides or contradicts that hair cue.
- If subject_features require suspenders, vest, flat cap, color-blocked dress, or another strong cue, the
  visible candidate must show that cue or an equally strong face/body identity match. Do not promote a
  nearby extra based only on one shared weak cue.
- For physical placement edits on shoulders/arms/torso, strong clothing/accessory cues in subject_features
  are mandatory current-frame evidence. If subject_features mention suspenders, the current frame must visibly
  show suspenders or an unmistakable face identity match; do not accept a random blond/profile passenger from
  build, pose, or story role alone. If it is a close-up where clothing is cropped out but the face identity is unmistakable, mark present=true.
- Side profile, back view, 3/4 angle, soft focus, or shallow depth-of-field blur STILL count as present when
  unmistakable identity cues match (cap/hat, vest, hair, build, accessories) — do NOT require a frontal face.
- BENDING/CROUCHING CANDIDATES: actively inspect small, low, bent-over, crouching, leaning, or partially
  occluded faces/heads/hair. Do not let a large back-view body or shoulder candidate crowd out a smaller
  front/side face that has stronger identity evidence.
- For physical placement targets, first choose the candidate whose face/head/hair/profile and stable cues
  best match the reference, then separately decide whether the requested attachment point is visible. Do not
  choose a prominent back-view shoulder over a smaller identifiable face just because the shoulder is easier
  to place an object on.
- identity_verifiable_from_visible_parts=true only when the visible key region is sufficient for identity
  comparison. Clear side profile with matching hair/face or distinctive current-frame clothing/build cues can
  qualify even when both eyes are not visible; shoulder-only, arm-only, tiny edge-crop, or generic clothing-only
  fragments must keep identity_verifiable_from_visible_parts=false.
- Reject look-alikes, background extras, and limb-only fragments at the frame edge.
- ANTI-HALLUCINATION FOR CORNER CROPS (CRITICAL): A partial arm/shoulder/torso sliver cropped by the frame edge WITHOUT a visible face is NOT sufficient for present=true. You MUST NOT list "face" or "hair" in `visible_parts` or claim "match" for hair/face features (e.g. "hair color" or "face shape") if they are physically cropped out or occluded. Hallucinating missing core features on cropped limbs is strictly forbidden.
- For hair/head/hat edits: clearly visible face/head/upper body is SUFFICIENT even if lower body is occluded
  by foreground blur — occlusion by OTHER people does not mean the target is absent.
- For removal/delete edits: head+torso in side profile is SUFFICIENT when identity cues are clear.
- For shoulder/torso accessory edits (e.g. placing an object on a shoulder): partial torso/shoulder
  with distinctive clothing cues (suspenders, vest, shirt color) is SUFFICIENT for editability only when
  identity is already strong and localization_clarity=high.
- For body-side placement, determine body_orientation plus where the subject's anatomical left and right
  sides appear in screen coordinates. If the requested side is occluded, set target_attachment_visible=false
  even when the opposite shoulder is visible.
- Estimate approximate_area_fraction from the target silhouette, not foreground occluders.
- visibility_quality must be exactly one of: clear | too_small | blurry | ambiguous | occluded (never "high").
  Use blurry (not absent) when the target is identifiable but soft-focus in the midground/background.
- present=true ONLY when: entity_visibility_completeness in {{sufficient, partial}} (partial ok for head/removal edits),
  identity_verifiable_from_visible_parts=true, localization_clarity=high,
  approximate_area_fraction>=0.05 OR clear face/upper-body for head edits, confidence>=0.92 for side/blur matches
  or >=0.99 for frontal matches.
- If present=false, location_description must be empty.

Per-entity fields:
instruction_id, entity_id, present, location_description, confidence, visibility_quality,
approximate_area_fraction, localization_clarity, entity_visibility_completeness, visible_parts,
identity_verifiable_from_visible_parts, reasoning

Return ONLY valid JSON:
{{
  "entities": [
    {{
      "instruction_id": "instr_001",
      "entity_id": "entity_01",
      "present": false,
      "location_description": "",
      "confidence": 0.0,
      "visibility_quality": "ambiguous",
      "approximate_area_fraction": 0.0,
      "localization_clarity": "low",
      "entity_visibility_completeness": "fragment",
      "visible_parts": [],
      "identity_verifiable_from_visible_parts": false,
      "reasoning": "brief explanation"
    }}
  ]
}}

Use English for all string values. Include one entry for EVERY entity in the catalog.
"""

KEYFRAME_CANONICAL_ALIGNMENT_PROMPT = """You are a focused VLM inspector for canonical reference alignment.

Image 1 = EDITED keyframe (full scene).
Image 2 = CANONICAL TARGET panel (RIGHT side of the entity reference card — the intended after-edit look).

EDIT INSTRUCTION for this entity:
{edit_instruction}

ENTITY LOCATION in image 1:
{entity_location}

First verify the edited attributes are on the SAME located entity described by ENTITY LOCATION. If the
attributes appear on a different person, a newly inserted person, or a pasted reference-card-like full body,
set alignment_ok=false even if the colors/styles match Image 2.

Then compare ONLY the instructed visual attributes on that located entity in image 1 against image 2.
Ignore background, other people, scene layout differences, and non-instructed face/pose/clothing differences
only when they were preserved from the original keyframe rather than copied from the canonical card.

MODERATE style checks (not too strict, not too loose):
- hat/cap/headwear: same style and silhouette — pillbox (flat top, minimal/no brim) vs fedora/trilby
  (creased crown + brim) vs beret vs baseball cap vs wide-brim — color match alone is NOT enough
- hair: the target hair color/tone family and the broad intended hair style from image 2 must be visibly
  applied to the located entity in image 1. Reject clear mismatches such as blonde/brown/red/black/blue
  target color not matching, unchanged original hair, wrong highlight pattern, or a visibly different
  requested style. Tolerate small differences caused by image 1 lighting, shadows, blur, distance,
  compression, occlusion, and the current keyframe's original hair volume/silhouette.
- clothing/accessory edits: same form, cut, and color as image 2

Judgment calibration:
- alignment_ok=true when the instructed attribute is recognizably the same as image 2 after adapting to
  the video frame's lighting, blur, scale, and occlusion, and when the edit is applied to the correct
  current-frame entity without replacing that entity.
- alignment_ok=false only for a clear attribute mismatch or missing/very weak edit. Do not fail for tiny
  pixel-level color differences, minor lighting shifts, or slight boundary variation.

Return ONLY valid JSON:
{{
  "alignment_ok": false,
  "mismatched_attributes": ["hat style"],
  "feedback": "short explanation",
  "retry_focus_prompt": "if failed: what to fix on retry; empty if passed",
  "positive_prompt": "if failed: what was done correctly and should be KEPT on retry; empty if passed"
}}

Rules:
- alignment_ok=true only when instructed attributes recognizably match image 2's style/shape/color family,
  not merely a vague or partial similar color.
- alignment_ok=false when image 1 shows a different hat style (e.g. fedora when image 2 shows pillbox).
- alignment_ok=false when a hair edit leaves the old hair mostly unchanged or changes it to a clearly
  different color/tone/style than image 2.
- alignment_ok=false when the method appears to paste/insert the canonical target person, a full-body
  replacement, a new face/head, or the RIGHT-panel clothing/pose/body instead of locally editing the original
  entity's requested attributes.
- alignment_ok=false when the requested attributes are visible but on the wrong person/location.
- Use English for all string values.
"""

KEYFRAME_SINGLE_CANONICAL_EDIT_PROMPT = """You are editing ONE video keyframe for a real-world video editing pipeline.

Image 1 = OUTPUT BASE — the original keyframe. You MUST return ONE image with the EXACT same width,
height, aspect ratio, letterboxing/pillarboxing black bars, camera framing, and overall scene layout as image 1.

Images 2+ = ENTITY EDIT REFERENCE cards from entity_refs/instr_00N_ref_canonical.png.
Each card's LEFT panel is the source/original entity identity to find in image 1; RIGHT panel shows ONLY the
intended edited attributes named by the text instruction. The RIGHT panel is NOT a full-person replacement target.

No prior edited keyframe images are provided or allowed as references:
{prior_scene_edit_block}
Edit this keyframe independently from image 1, the current keyframe's entity locations, and the canonical
entity edit cards only. Do NOT infer, copy, or replicate pixels, pose, expression, background, framing,
lighting, or completed outcomes from any earlier edited keyframe in the same scene.

SCENE STORY CONTEXT (for identity continuity and natural integration only; do NOT invent edits from it):
{scene_story_context}

CRITICAL:
- For each listed instruction, find the same entity as the card's LEFT panel in image 1 at the VLM-provided
  location description in parentheses, then apply the corresponding text edit instruction.
- Use the text EDIT INSTRUCTIONS as the authority for exactly which attributes may change.
- Use the card's RIGHT panel only as the visual reference for the explicitly requested edited attribute(s).
  Ignore every unrelated difference in the RIGHT panel.
- This is a surgical local edit, not image compositing. Never paste, transplant, upscale, or redraw the
  RIGHT-panel person/object into image 1. The output must still be the original target entity from image 1
  with only the requested attributes changed.
- Keep scene lighting and camera integration from image 1. Unless the edit instruction explicitly changes
  facial expression, body action, pose, gaze, clothing, or another state attribute, those non-edit states must
  remain unchanged in the edited result.
- Image 1 is the ONLY source of scene state: pose, action, facial expression, gaze, body posture,
  occlusion, motion blur, local lighting, shadows, highlights, color temperature, and camera integration.
- The canonical RIGHT panel is only an appearance/attribute target. Do NOT copy its pose, expression,
  lighting, background, crop, or studio/reference-card rendering style into the video frame.
- CURRENT FRAME STATE IS THE ONLY SCENE STATE. The output entity must keep image 1's original morphology,
  silhouette, expression, gaze, head angle, body posture, action, occlusion, visible extent, background,
  and local lighting except for the explicitly requested edit region.

ENTITY LOCATIONS IN IMAGE 1 (from VLM):
{entity_locations_block}

EDIT INSTRUCTIONS (one per reference card; ONLY these apply):
{canonical_edit_block}

ORIGINAL VISIBILITY CONSTRAINTS (image 1 — do NOT exceed):
{visibility_constraints_block}

CRITICAL:
- Before editing, internally reason from ORIGINAL VISIBILITY CONSTRAINTS which pixels and attributes are
  locked. Treat this as a pre-edit LLM lock plan: head orientation, face/expression/gaze, clothing,
  pose/action, body/hand positions, local lighting/shadows, and background outside explicit edit regions
  must remain exactly from image 1 unless the text instruction names that attribute.
- Apply ONLY the edit instructions and canonical reference cards listed above — do not edit any other entities.
- Never create a new person, actor, body, face, torso, or full entity that was not already visible in image 1.
  Entity reference cards are identity/attribute references only; they must NOT be used to paste or synthesize
  an absent entity into the frame.
- When an instruction has target_instance_scope=single, edit ONLY that one located instance — never apply
  the same change to other people/objects with similar clothing or features.
- For target_instance_scope=single removal/delete, remove exactly the one located target instance. Do NOT
  remove multiple similar people, a group of extras, or adjacent look-alikes unless target_instance_scope=multiple.
- Edit ONLY the located entity regions listed in ENTITY LOCATIONS; leave everything else unchanged.
- If an instruction is not listed in ENTITY LOCATIONS / EDIT INSTRUCTIONS for this keyframe, that entity is
  absent or out of scope: do NOT generate that entity, do NOT add its accessories, and do NOT use its reference card.
- Respect the VLM confidence/location text. If confidence is lower or the location is ambiguous, make a
  conservative localized edit only where the target is unambiguous; do NOT edit adjacent people or background.
- Edit ONLY within each entity's ORIGINAL visible silhouette in image 1. Never paint in, reveal, or
  complete body/face regions that were occluded, cropped, behind pillars, or off-frame in image 1.
- If the target is overlapped or partially occluded by another foreground/person hair/head/body, the occluding
  person is locked. Do NOT recolor, repaint, blur, or attach accessories to the occluder's hair/head/body.
  Edit only the target entity's own visible pixels, respecting the original depth ordering and occlusion edge.
- Preserve each subject's pose, expression, head angle, gaze direction, and body posture except where
  the edit instruction explicitly changes appearance attributes (e.g., hair color, hat).
- Preserve the exact action and emotional/facial state from image 1: open/closed mouth, smile/frown,
  eye direction, head turn, hand/body gesture, walking/standing/leaning posture, and interaction with
  nearby objects or people must not drift.
- Change ONLY attributes named in EDIT INSTRUCTIONS. If an instruction changes hair/clothing color, do NOT
  also change facial expression, pose, body action, gaze, or the entity's local lighting/shadow pattern.
- If an instruction edits hair/hair color/headwear, DO NOT change clothing color, clothing pattern,
  dress/shirt/vest shape, face, facial expression, mouth, eyes, gaze, body pose, arm/hand position,
  walking/standing action, or background pixels.
- For hair/headwear edits, NEVER replace, redraw, smooth, beautify, relight, or paste a new face/head.
  The original face pixels, facial identity, expression, gaze, head pose, skin texture, and local shadows
  from image 1 must remain intact; only the requested hair/headwear pixels may change.
- For hair/headwear edits, never paste the reference-card head, face, upper body, dress, or full figure into
  the scene. Do not enlarge the target, move it toward camera, make it front-facing, change its clothing, or
  replace its body silhouette to resemble the RIGHT panel.
- If an instruction edits clothing, DO NOT change hair, headwear, face, expression, pose, action, or body shape
  unless those exact attributes are named in EDIT INSTRUCTIONS.
- Treat all non-requested attributes as locked pixels from image 1. Small blending changes are allowed only
  along the edited attribute boundary, not across the whole person.
- Preserve the entity's local lighting as it appears in image 1: same light direction on the subject,
  shadow placement, highlight intensity, rim light, and ambient color temperature at that screen position.
- Adapt the edited attribute to the original frame's lighting and texture. The result must look like the
  edit was performed in the real video frame, not pasted from a clean reference card.
- Preserve original image noise, sharpness/blur, depth of field, compression texture, contact shadows,
  and foreground/background occlusion relationships around the entity.
- SMALL / DISTANT TARGETS: when approximate_area_fraction is small or the target is in the background,
  the edit must remain subtle and low-detail like the original pixels. Do NOT paste a sharp, clean,
  high-resolution, front-facing, or reference-card-like face/hat/object onto a small blurry target.
  Match the original target's blur, grain, compression, scale, occlusion, and lighting exactly.
- Preserve all black bars, margins, background structure, and unedited subjects pixel-for-pixel where possible.
- Background pixels outside a removal/inpainting silhouette are locked. Do not smooth, repaint, brighten,
  darken, duplicate, or invent small patches in unrelated background regions.
- Background furniture/props outside the target silhouette are locked: do not add, remove, duplicate, or reshape
  chairs, benches, tables, railings, lamps, doorways, wall panels, posters, luggage, pillars, stairs, or floor
  features unless the text instruction explicitly targets that object.
- Non-target edge/corner people are locked. Do not remove, erase, shift, replace, or repaint the far-right,
  far-left, partially cropped, or background woman/person when the instruction targets one specific different
  individual.
- Use image 1 as the immutable background reference. For non-removal edits, do not touch any background
  pixels. For removal edits, change only the removed target's original silhouette plus the smallest plausible
  inpaint seam; walls, pillars, floor, ceiling, distant people, and side-edge patches outside that silhouette
  must remain from image 1.
- For removal/delete edits, inpaint the target's original silhouette with plausible EMPTY background only.
  Do NOT replace the removed target with another person, a look-alike, a person from a reference card,
  a celebrity-like actor, a full-body substitute, or an entity from another instruction.
- When multiple instructions apply in the same keyframe, removal/delete instructions are mandatory and must
  remain completed on every retry. Do NOT reintroduce a removed target while fixing a separate accessory,
  color, or placement edit. Positive feedback about another edit never overrides the deletion requirement.
- Do NOT crop, reframe, collage, or change canvas size.
- Preserve entity_id identity except where the edit instruction explicitly changes appearance.
- Per-keyframe consistency comes from the shared instruction text and canonical cards only. Never use a
  previous edited keyframe as a visual or textual outcome template. For hair/hat edits, keep the current
  keyframe's original hair volume boundary where visible, head pose, face, expression, gaze, neck connection,
  body action, and occlusions; change only the requested hair color/headwear attribute.
- Edited regions must blend naturally with the surrounding scene environment (e.g., lighting direction,
  intensity, color temperature, shadows, and reflections) so the change looks physically consistent.
- If the RIGHT panel has different expression, pose, gaze, body angle, lighting, or studio-like rendering,
  ignore those non-edit states and keep image 1's original state.
- If the RIGHT panel differs in clothing, face, expression, body shape, or pose but those differences are not
  explicitly requested by text, DO NOT copy them.
- Any output that looks like the canonical RIGHT-panel person was pasted or composited into the keyframe is
  a failed edit. Instead, repaint only the target attribute pixels on the original entity already visible in
  image 1, preserving its original scale, position, silhouette, depth ordering, occlusions, and clothing.
- PHYSICAL PLACEMENT / OCCLUSION: When an instruction places an object on a specific body side
  (e.g. left shoulder), use image 1's body orientation and ORIGINAL VISIBILITY CONSTRAINTS to decide if that
  exact side is visible. If the requested side is turned away, behind the body, behind another person/object,
  or otherwise not visible in image 1, the added object must be hidden/absent in the output. Never place it
  on the visible opposite shoulder just to satisfy the text.
- For placed objects, preserve the canonical reference object's relative scale to the target shoulder/head.
  The object should be small enough to physically sit on the shoulder and must not become oversized,
  toy-like, sticker-like, or larger than the reference ratio.
- The placed object must have physical contact with the shoulder: matching local light direction, contact
  shadow/occlusion, perspective, blur/noise, and partial support by the shoulder surface. It must not float,
  hover, cover too much torso/head, or look like a flat sticker pasted on top of the frame.
- Orient placed objects with the subject's body/camera perspective. If the canonical reference shows the
  object facing the same direction as the entity, keep that relative orientation in the keyframe (for example,
  when the subject faces forward, the duck should also face forward; when the subject turns, rotate the duck
  consistently with the shoulder plane).
{avoid_section}

Return the edited keyframe as a single full-frame image matching image 1's structure.
"""

KEYFRAME_SINGLE_EDIT_QA_PROMPT = """You are a strict VLM quality inspector for single-keyframe editing.

Image 1 = EDITED keyframe (candidate result).
Image 2 = ORIGINAL keyframe (before editing).
Images 3+ = entity canonical reference cards (left=original entity, right=edited target).

ENTITY LOCATIONS (VLM):
{entity_locations_block}

ORIGINAL VISIBILITY CONSTRAINTS (from detection on image 2):
{visibility_constraints_block}

EDIT INSTRUCTIONS:
{canonical_edit_block}

Compare image 1 vs image 2. The two highest-priority questions are:
1. Did image 1 satisfy EVERY planned edit instruction on the correct listed target/location?
2. Did image 1 preserve all non-edited regions, background, non-target people/objects, and non-requested
   target attributes from image 2?
If either answer is no, passed=false. Good realism, framing, or partial success must never compensate.

CRITICAL (must pass — any clear failure → passed=false):
1. FRAME STRUCTURE PRESERVED — same canvas size, aspect ratio, camera angle, depth ordering,
   foreground/background layering, black bars, and global scene layout. Reject reframes or collage.

2. EDIT COMPLETED — each required edit is correctly applied where the entity was present in image 2,
   AND instructed visual attributes match the canonical RIGHT panel (see item 5), not merely approximate.

3. ENTITY IDENTITY — same person/object (entity_id), not a substitute.

4. BACKGROUND / NON-TARGET REGIONS PRESERVED (CRITICAL) — scan the FULL frame. If the EDIT INSTRUCTIONS do not explicitly request changing the background, areas OUTSIDE the
   edited entity silhouettes and minimal inpaint fill for removals must match image 2. Reject if walls,
   pillars, floor, ceiling, sky, distant scenery, or unrelated regions were repainted, relit, blurred,
   or texture-changed. Removal edits may change pixels only where the removed entity stood plus a thin
   inpaint seam — not broad background regions.
   Distinguish a tiny local seam directly touching the edit boundary from a real non-edit-region change:
   only the former may be rated `trace`; any clearly visible changed patch, recolored non-target region,
   shifted object/person, or repeated texture away from the boundary must be rated `moderate` or `severe`.

5. CANONICAL REFERENCE ALIGNMENT (CRITICAL when images 3+ are provided) — for EVERY instructed attribute
   change, compare image 1's edited entity region to the RIGHT panel of its canonical card:
   - hat/cap/headwear: same style/silhouette (pillbox vs fedora vs beret vs wide-brim), crown shape,
     brim width, tilt, and color — NOT just "a hat of similar color"
   - hair: same target color/tone and general style as RIGHT panel
   - clothing/object edits: same form and color as RIGHT panel
   Reject canonical_reference_alignment_ok=false if image 1 differs in style/shape even when the text
   instruction is loosely satisfied (e.g. green fedora when RIGHT panel shows a green pillbox).

6. ORIGINAL ENTITY STATE & REALISM (CRITICAL) — the edit must preserve image 2's original pose,
   expression, gaze, action, body/head angle, local lighting, shadows, blur/noise, occlusion, and scene
   interaction. Reject if the edited entity looks fake, pasted, studio-lit, over-smoothed, relit, or if it
   copies pose/expression/lighting from the reference card instead of image 2.

IMPORTANT (aim to pass; one minor miss alone should not fail if others are strong):
7. VISIBILITY EXTENT PRESERVED — do not reveal or complete large body/face regions that were clearly
   occluded, cropped, or off-frame in image 2. Minor edge softness is acceptable.

8. POSE & EXPRESSION PRESERVED — same general head angle, gaze, expression, and body posture as image 2.
   Small expression or micro-pose drift is acceptable if the instructed edit still reads correctly.

9. ENTITY LOCAL LIGHTING PRESERVED — edited entity lighting should broadly match image 2 at that position
   (direction, shadows, highlights). Reject obvious re-lit or flat pasted looks, not subtle grade shifts.

10. UNRELATED EDIT CHANGES ABSENT — only instructed attributes should change. Reject any clear collateral
   change to non-target people/objects, background, pose, expression, clothing, visible extent, or lighting
   beyond the edit scope.

11. ENVIRONMENT BLEND — edited pixels should integrate with the background at contact edges; reject strong
   cut-out / pasted appearance, not minor edge imperfections.

Return ONLY valid JSON:
{{
  "passed": false,
  "score": 0.0,
  "frame_structure_preserved": false,
  "background_unedited_regions_preserved": false,
  "canonical_reference_alignment_ok": false,
  "edit_instruction_requirements_met": false,
  "visibility_extent_preserved": false,
  "pose_expression_preserved": false,
  "entity_local_lighting_preserved": false,
  "unrelated_edit_changes_absent": false,
  "environment_blend_ok": false,
  "edit_completed": false,
  "entity_identity_preserved": false,
  "non_edit_region_change_severity": "none",
  "non_edit_region_change_summary": "",
  "failed_aspects": [],
  "feedback": "short summary",
  "retry_focus_prompt": "if failed: negative guidance for retry; empty if passed",
  "positive_prompt": "if failed: what was done correctly and should be KEPT on retry; empty if passed"
}}

Rules:
- edit_instruction_requirements_met is the primary verdict: true only when all required edit instructions are
  satisfied on the correct target entities/locations. If false, passed must be false and score should be <=0.35.
- background_unedited_regions_preserved and unrelated_edit_changes_absent are hard gates. If either is false,
  passed must be false even if all requested edits were completed.
- passed=true only when ALL critical items (1–6) are clearly satisfied, ALL important booleans (7–11) are true,
  AND score>=0.6.
- canonical_reference_alignment_ok=false when hat/cap/hair/accessory style or color in image 1 does not
  match the canonical RIGHT panel.
- background_unedited_regions_preserved=false when ANY non-target background region differs from image 2 (unless the edit instruction explicitly targets the background).
- non_edit_region_change_severity must be one of: `none`, `trace`, `moderate`, `severe`.
  Use `trace` only for a tiny local seam/halo directly touching the edit boundary with no recognizable non-edit content changed.
  Use `moderate` or `severe` for clearly visible changed non-edit patches, non-target people/objects, or broader repainting/drift.
- If non_edit_region_change_severity is `moderate` or `severe`, background_unedited_regions_preserved must be false and passed must be false.
- non_edit_region_change_summary: short English summary of the changed non-edit regions; empty if severity=`none`.
- Fail on structural, identity, background, non-target, or edit-completion errors; tolerate only tiny seam/edge
  imperfections that do not change any recognizable non-edited content.
- retry_focus_prompt must be prohibitions/mistakes to avoid, not new edit goals.
- positive_prompt (REQUIRED when failed; empty when passed): list the editing operations that were done CORRECTLY in this attempt and should be KEPT/MAINTAINED on the next retry. Phrase as positive instructions.
- Use English for all string values.
"""

KEYFRAME_ENTITY_DETECT_PROMPT = """You are detecting edit-target entities in ONE video keyframe.

Image 1 = TARGET KEYFRAME (the frame to be edited).
Images 2+ = per-entity FRONT-VIEW reference images from entity_refs/instr_00N_ref_front.png.

SCENE STORY CONTEXT (use only as identity/continuity evidence; never hallucinate absent entities):
{scene_story_context}

ENTITIES TO DETECT (evaluate EVERY entity; reference image mapping):
{entity_catalog_block}

TASK:
For EACH entity, decide whether the same tracked person/object appears anywhere in image 1. If present,
describe its precise location. This is a recognition/localization step, not the final editability gate:
mark present=true when the identity is visually recognizable, even if the subject is side-facing, side-body,
small, blurry, expression-changed, camera-angle changed, or partially occluded.

RULES:
- IDENTITY vs APPEARANCE (CRITICAL): Match the SAME person/object (entity_id), NOT an identical costume across frames. Image 2 may include views from other scenes with different outfits — that is expected. NEVER reject solely because clothing in image 1 differs from a panel in image 2 if the face/body identity matches. Reject look-alikes, background extras, and limb-only fragments — not the same person in a different outfit.
- VISIBLE-PART-ONLY IDENTITY (CRITICAL): compare only the parts of each candidate that are actually visible in
  image 1 with the corresponding parts of the reference. Do NOT penalize the candidate because reference
  clothing, lower body, shoulders, or accessories are hidden/cropped in image 1. If a visible key region
  (face/head/hair/hairline/clear profile, or a substantial torso/headwear/accessory region with distinctive
  cues) has high similarity to the reference, mark present=true. A half shoulder, arm/hand sliver, tiny edge
  crop, or generic clothing corner is not a key region and must not be accepted by itself.
- SPATIAL DISAMBIGUATION: when two entities could overlap at the same screen region (e.g. both described at the extreme left edge), assign the partial figure to the entity whose distinctive clothing/accessories actually match the visible pixels (vest vs suspenders vs cap). Mark present=false for the other entity at that region — never double-assign one cropped limb to two instruction_ids.
- Respect target_instance_scope:
  - single: exactly ONE tracked individual matching subject_features
  - multiple: every instance in the frame matching subject_features
- Compare side/back/3-4 views against stable identity cues in the front-view reference image.
- For removal/delete edits: partial edge crops are SUFFICIENT only when a substantial torso/head/cap/hat/vest
  region is visible with distinctive cues. A tiny shoulder sliver, arm sliver, or corner of clothing is NOT
  sufficient, even if it resembles the reference.
- For shoulder/torso accessory edits: partial torso/shoulder is sufficient only when the visible region is large
  enough to edit and identity cues are verifiable; half a shoulder or a tiny cropped fragment is not enough.
- For shoulder/torso accessory edits whose subject_features mention suspenders, verify suspenders are visible
  in the current keyframe or the face identity is unmistakable. Do not correct to present=true from hair color,
  side profile, build, or story role alone. If it is a close-up where suspenders are cropped out but the face identity is unmistakable, mark present=true.
- ANTI-NAME/ROLE INTERFERENCE: Do NOT mark present=false by identifying a person as "the main character",
  "Jack", "the lead", or by any actor name (e.g. "Leonardo DiCaprio"). Character names and actor identities are
  NOT visual evidence. If the visible facial structure, hair silhouette, and features match the reference image,
  mark present=true regardless of any name/role labels from story context.
- HAIR-COLOR LIGHTING SHIFT: The same person's hair may appear lighter blond in bright/outdoor shots and darker
  brown in close-up/indoor shots. When the hair style, silhouette, hairline, and facial structure match the
  reference, do NOT reject based on apparent hair color tone alone — this is a lighting artifact.
- MISSING CROPPED FEATURES: When features like suspenders, a dress, or a vest are not visible because they are
  cropped out of frame in a close-up, mark present=true if the visible face/head/hair matches. Cropped features
  are absent due to framing, not because the person is different. Only mark present=false for feature absence
  when the same body region is visible in frame and clearly shows contradicting clothing.
- present=true only when you are confident the specific entity is visible with enough area to perform its edit.
- confidence: float 0.0–1.0 representing match certainty.
- If present=false, location_description must be empty string.

Per-entity output fields:
instruction_id, entity_id, present, confidence, location_description, visibility_quality,
approximate_area_fraction, visible_parts, reasoning

Return ONLY valid JSON:
{{
  "entities": [
    {{
      "instruction_id": "instr_001",
      "entity_id": "entity_01",
      "present": false,
      "confidence": 0.0,
      "location_description": "",
      "visibility_quality": "ambiguous",
      "approximate_area_fraction": 0.0,
      "visible_parts": [],
      "reasoning": "brief explanation"
    }}
  ]
}}

Use English for all string values. Include one entry for EVERY entity in the catalog.
"""

KEYFRAME_ENTITY_LOCATION_VERIFY_PROMPT = """You are verifying entity localization on ONE video keyframe.

Image 1 = TARGET KEYFRAME.
Images 2+ = per-entity front-view reference images.

INITIAL DETECTION RESULTS (from a prior pass — verify and correct if needed):
{detection_results_block}

ENTITY CATALOG:
{entity_catalog_block}

TASK (run ONCE):
For EACH entity listed above, verify whether the initial detection is correct. This verification should
improve localization and remove clear wrong-person matches, but it should NOT downgrade a plausible
same-identity detection merely because the face is non-frontal, expression changed, camera angle changed,
the body is side-facing, or the person is small but identifiable by stable visual cues.
- FRONT-VIEW REFERENCE IMAGES ARE PRIMARY. Scene story context may only support a visible candidate that also
  matches the reference identity; it must not turn a visually different narrative lead, prominent person, or
  look-alike into the target entity.
- IDENTITY vs APPEARANCE: Do NOT reject solely because clothing differs if the face/body identity matches the reference. However, do NOT accept look-alikes or background extras just because they wear similar clothing.
- VISIBLE-PART-ONLY IDENTITY: verify identity only from regions actually visible in image 1. Missing or cropped
  reference regions are not contradictions. If the visible face/head/hair/profile or another substantial
  distinctive key region matches the reference, keep or correct to present=true even when the first-appearance
  outfit is not visible. Shoulder-only, arm-only, tiny edge-crop, and generic clothing-only fragments remain
  insufficient.
- SPATIAL DISAMBIGUATION: Do not double-assign one cropped limb to two instruction_ids. Assign it to the entity whose distinctive clothing/accessories actually match the visible pixels.
- For removal/delete edits: partial edge crops (e.g. arm+torso sliver) are SUFFICIENT for present=true when distinctive clothing cues (vest, suspenders, cap) match the reference.
- For shoulder/torso accessory edits: partial torso/shoulder with distinctive clothing cues is SUFFICIENT for present=true.
- If present=true but location is wrong, correct location_description and adjust confidence.
- If present=true but the match is actually a look-alike or wrong person, set present=false and confidence low.
- If present=false but the entity IS clearly visible (or is a valid partial edge crop for removal/accessory edits), set present=true with accurate location and confidence.
- ANTI-NAME/ROLE CORRECTION: If initial detection rejected an entity citing character names, actor names, or
  narrative roles (e.g. "this is Jack/the main character, not the target"), CORRECT to present=true when the
  visible face/hair/facial-structure matches the reference. Names and roles are not visual evidence.
- LIGHTING/TONE CORRECTION: If initial detection cited hair color mismatch (e.g. "dark-haired" vs "blond") but
  facial structure and hair silhouette match the reference, CORRECT to present=true — hair color shifts with
  lighting and camera distance.
- MISSING-CROPPED CORRECTION: If initial detection cited missing cropped features (e.g. "missing suspenders")
  but the visible face/head/hair matches the reference, CORRECT to present=true. Cropped features are absent
  due to framing, not identity mismatch.
- Do NOT invent entities that are not in the frame.

Per-entity output fields:
instruction_id, entity_id, present, confidence, location_description, location_corrected,
visibility_quality, approximate_area_fraction, visible_parts, body_orientation,
anatomical_left_screen_side, anatomical_right_screen_side, target_attachment_point,
target_attachment_visible, attachment_visibility, attachment_visibility_reasoning, reasoning

location_corrected=true when you changed present status, confidence, or location_description from the initial detection.

Return ONLY valid JSON:
{{
  "entities": [
    {{
      "instruction_id": "instr_001",
      "entity_id": "entity_01",
      "present": true,
      "confidence": 0.95,
      "location_description": "center of frame, standing on stairs",
      "location_corrected": false,
      "visibility_quality": "clear",
      "approximate_area_fraction": 0.15,
      "visible_parts": ["face", "torso"],
      "reasoning": "verified — initial detection accurate"
    }}
  ]
}}

Use English for all string values. Include one entry for EVERY entity in the catalog.
"""

SCENE_KEYFRAME_PRESENCE_CONSISTENCY_PROMPT = """You are correcting entity presence across keyframes from ONE continuous scene.

Images 1..{keyframe_count} = scene keyframes in chronological order:
{keyframe_catalog_block}

Images {first_ref_image_index}+ = per-entity FRONT-VIEW reference images:
{entity_catalog_block}

SCENE STORY CONTEXT (identity/continuity evidence from shots_analysis; visible pixels still required):
{scene_story_context}

INITIAL PER-KEYFRAME DETECTION RESULTS:
{initial_detection_block}

ALL-NEGATIVE RECOVERY TARGETS:
{all_negative_recovery_block}

TASK:
Review the whole scene temporally and correct inconsistent entity presence/location results.

Rules:
- FRONT-VIEW REFERENCE IMAGES ARE THE PRIMARY IDENTITY EVIDENCE. Use scene story context only to interpret
  ambiguous but visible pixels of the same referenced person/object. Do not mark a candidate present merely
  because they are the narrative lead, prominent, centered, or described in shots_analysis if their face/hair/body
  identity conflicts with the entity_refs reference.
- Treat the scene as a continuous shot unless the images clearly show a cut. A large, distinctive person/object
  detected at the same screen region in a neighboring keyframe should not disappear in another keyframe unless
  it is actually outside frame, fully occluded, or no visible parts remain.
- For removal/delete edits, a partial edge crop (torso/arm/head/hat at the left or right border) is enough for
  present=true when distinctive cues match the reference and neighboring keyframes support continuity.
- Do NOT recover a removal target from a neighbor when the current keyframe evidence explicitly contradicts
  the target identity (for example the current frame contains only women, lacks the required flat cap/vest,
  or the visible candidate is a different look-alike). Neighbor continuity cannot override a clear identity
  rejection in the current frame.
- A consistency correction still requires visible evidence in the current keyframe. Neighboring keyframes can
  help interpret ambiguous/cropped pixels, but must not create an entity where no pixels of that entity are visible.
- Use SCENE STORY CONTEXT to infer that a recurring character remains the same entity when only a small but
  meaningful visible region remains (for example distinctive hair/back/head at the expected story position).
  This can correct false negatives for partial visibility, but cannot justify hallucinating an entity into a
  frame with no visible pixels of that person/object.
- ALL-NEGATIVE RECOVERY: If an instruction_id is listed in ALL-NEGATIVE RECOVERY TARGETS, the initial detector
  marked it absent in every keyframe. You MUST still actively scan every scene keyframe for large or salient
  unassigned candidates (faces, heads, hair, upper bodies, full bodies) and compare those candidates to the
  reference image and reference identity context. Scene consistency is not only propagation from positive
  neighbors; it must also recover obvious misses when a large side-profile, close-up, expression-changed, or
  wardrobe-changed instance was never bound by the initial detector.
- BENDING/CROUCHING ROBUSTNESS: also scan small, low, bent-over, crouching, leaning, or partially occluded
  faces/heads/hair. A smaller identifiable front/side face is stronger evidence than a larger back-view body
  whose face and identity are hidden.
- For all-negative recovery, enumerate the best current-frame candidate in reasoning. If you recover it,
  output present=true with location_description, visible_parts, identity_cues, and
  identity_verifiable_from_visible_parts=true. If you reject a large/salient candidate, explain the concrete
  visual contradiction (for example incompatible face structure, hair color, body identity, or a different
  tracked person). Do NOT reject solely because clothing differs, the face is side-profile/expressive, or
  subject_features uses a different age/gender word.
- The 'subject_features' field often describes the entity's clothing in their FIRST appearance. Do NOT use it as a strict filter for later scenes. If the character has changed clothes but their face/hair identity matches the reference, they are the SAME person.
- VISUAL REFERENCE OVERRIDES TEXT: If the visible face/hair matches the Reference Image (Image 2+), you MUST mark present=true, even if the current clothing completely contradicts the 'subject_features' text.
- AGE/GENDER TERMINOLOGY: Terms like "girl", "woman", "boy", "man" in subject_features are often used interchangeably. Do NOT reject an identity match by claiming the reference is a "child/girl" while the frame shows a "woman" (or vice versa). Rely strictly on visual face/hair matching, not semantic age labels.
- SIDE PROFILE + WARDROBE CHANGE ROBUSTNESS: A side profile or angled face will naturally look different from the front-view reference. You MUST extrapolate identity using hair color, hair silhouette, and general facial structure. If these plausibly match the reference, mark present=true EVEN IF their clothing has completely changed. Do NOT require a frontal face to confirm identity when the outfit changes.
- CRITICAL RULE FOR IDENTITY: Core facial and hair features determine identity. Wardrobe changes (different clothes) or temporary hairstyle changes (e.g., updo vs. loose hair) do NOT constitute an identity change. If the face and core physical features match the reference, mark as present=true even if the outfit differs. Do not claim identity conflicts due to clothing changes.
- When recovering an entity across keyframes, explicitly state the current-frame identity evidence in
  reasoning and identity_cues, such as "matching face", "matching hair color", or
  "matches reference identity". Do not output only vague continuity wording.
- CANDIDATE AUDIT REQUIRED: for every present=true result, candidate_evaluations MUST contain the selected
  candidate, concrete identity_matches, any identity_conflicts, and why it is not a look-alike. Do not leave
  candidate_evaluations empty for present=true.
- Weak story evidence is not identity evidence: scene role, conversation role, prominence, expression, pose,
  and a name/alias from story context can only support a concrete visual match. They must not be the primary
  reason for present=true.
- Reject tiny fragments: a very small shoulder sliver, arm sliver, corner of clothing, or edge fabric WITHOUT
  face/head OR a substantial torso/cap/hat/vest region must remain present=false, even if a neighboring
  keyframe contains the entity. Do not infer an identity from half a shoulder.
- Do NOT blindly propagate detections through true entrances/exits. If the entity clearly enters after a keyframe
  or leaves before it, keep present=false for frames where it is not visible.
- Preserve spatial disambiguation: do not assign the same cropped person/object to multiple instruction_ids.
- For target_instance_scope=single, enforce temporal identity consistency: the same instruction_id must refer
  to the same tracked individual across all keyframes. If two keyframes appear to assign the instruction to
  different look-alikes, keep present=true only for the candidate(s) matching the reference and strong
  subject_features best; set the inconsistent look-alike to present=false and explain the conflict.
- Maintain one stable target track per instruction_id across the scene. Do not edit person A in one keyframe
  and person B in another because both are near the center/right side or both wear similar clothing. If the
  current keyframe cannot identify the same tracked person, mark present=false rather than switching targets.
- Strong features (hair style/color, hat/cap/headwear, suspenders, vest, clothing pattern/color, beard/stubble,
  distinctive accessories) outweigh weak state features (pose, expression, gaze, action). Do not switch targets
  across keyframes because a different extra has a similar pose or occupies a similar screen region.
- If you change present from false to true, provide a precise location_description and visible_parts from that
  specific keyframe, not copied mechanically from another frame.
- For body-side placement edits, preserve or correct body_orientation, anatomical_left_screen_side,
  anatomical_right_screen_side, target_attachment_point, target_attachment_visible, and attachment_visibility.
  Do not copy shoulder visibility from a neighbor if the subject turns between keyframes.
- For physical placement edits, entity presence and editability are different decisions. A person can be
  present while the requested shoulder/hand/body attachment is NOT editable in that keyframe. If the current
  keyframe only shows a back-of-head/back-view/turned-away/occluded view or ambiguous anatomical left/right,
  keep present=true only if identity is visible, but set target_attachment_visible=false and explain that the
  placement instruction should not be applied to this frame.
- For physical placement edits, do not choose a prominent back-view shoulder/body candidate over a smaller
  candidate with visible face/head/hair identity. Select the identity-best candidate first, then decide
  target_attachment_visible from that candidate's current-frame anatomy.
- If initial detection is already correct, keep it.

Return ONLY valid JSON:
{{
  "keyframes": [
    {{
      "keyframe_id": "keyframe_0001",
      "entities": [
        {{
          "instruction_id": "instr_001",
          "entity_id": "entity_01",
          "present": true,
          "confidence": 0.95,
          "location_description": "left side of frame, partially cropped",
          "visibility_quality": "clear",
          "approximate_area_fraction": 0.2,
          "visible_parts": ["torso", "arm"],
          "identity_cues": ["matching face", "brown shirt"],
          "identity_verifiable_from_visible_parts": true,
          "body_orientation": "three-quarter back view",
          "anatomical_left_screen_side": "screen-right",
          "anatomical_right_screen_side": "screen-left",
          "target_attachment_point": "left_shoulder",
          "target_attachment_visible": true,
          "attachment_visibility": {{"left_shoulder": true, "right_shoulder": false}},
          "candidate_evaluations": [
            {{
              "candidate_location": "left side of frame, partially cropped",
              "visible_parts": ["torso", "arm"],
              "identity_matches": ["specific visible identity match"],
              "identity_conflicts": [],
              "decision": "present"
            }}
          ],
          "location_corrected": true,
          "reasoning": "corrected by scene temporal consistency"
        }}
      ]
    }}
  ]
}}

Use English for all string values. Include one entry for EVERY entity in EVERY keyframe.
"""

SCENE_KEYFRAME_ALL_NEGATIVE_RECOVERY_PROMPT = """You are performing a focused recovery pass for edit-target entities that were marked absent in EVERY keyframe of one scene.

Images 1..{keyframe_count} = scene keyframes in chronological order:
{keyframe_catalog_block}

Images {first_ref_image_index}+ = per-entity FRONT-VIEW reference images:
{entity_catalog_block}

SCENE STORY CONTEXT (identity/continuity evidence from shots_analysis; visible pixels still required):
{scene_story_context}

INITIAL PER-KEYFRAME DETECTION RESULTS:
{initial_detection_block}

TASK:
For EACH listed all-negative entity, actively search EACH scene keyframe for large or salient unassigned candidates
(foreground faces, side profiles, heads/hair, upper bodies, full bodies, and small low/bent/crouching faces)
and compare them to the reference image
plus the REFERENCE IDENTITY CONTEXT. This is a recovery pass: do not merely repeat the initial absence result.

Rules:
- Enumerate the best candidate(s) in candidate_evaluations for every keyframe, especially any large foreground
  person/object and any smaller front/side face in a bent-over, crouching, leaning, or low posture. Include
  candidate_location, visible_parts, identity_matches, identity_conflicts, and decision.
- Candidate priority is identity-first, not size-first. Prefer a smaller candidate with visible matching
  face/head/hair/profile over a larger back-view shoulder/body whose identity is not verifiable.
- Recover present=true when a salient candidate plausibly matches the reference identity by face structure, hair
  color/silhouette, head/upper-body shape, recurring role/name alias, or multi-view reference continuity, even if
  the candidate is side-profile, expressive, close-up, partially cropped, or wearing different clothing.
- VISUAL REFERENCE AND IDENTITY CONTEXT OVERRIDE TEXT: subject_features may describe an early outfit or initial
  scene. Clothing, scene, pose, expression, gaze, camera angle, distance, and "girl/woman/man/boy" wording are
  mutable appearance and are not sufficient rejection reasons.
- If you keep present=false for a large/salient candidate, identity_conflicts must name concrete visual conflicts
  such as incompatible face structure, hair color/silhouette, body identity, or a different tracked individual.
  Do NOT reject solely because clothing differs, the face is side-profile, expression differs, or age/gender
  terminology differs.
- A recovery still requires visible current-frame pixels. Do not hallucinate an entity into empty background,
  pure limb fragments, or indistinguishable crowds.
- Maintain one stable target track per instruction_id. Do not switch to a look-alike just because of similar
  clothing or location.
- PHYSICAL PLACEMENT RECOVERY IS STRICT: for add/place-on-body instructions (for example shoulder/hand/body
  placements), do NOT recover present=true from a cropped torso/shoulder/clothing patch alone. The current
  frame must show face/head/hair or another unmistakable unique identity feature. If identity is strongly
  verifiable but the exact requested attachment point is hidden or anatomical side is unclear, keep
  present=true with target_attachment_visible=false so the frame can be tracked but the placement edit is not
  attempted. If identity is not strongly verifiable, keep present=false.
- SPATIAL CONFLICT CHECK: if the best candidate overlaps with, was previously assigned to, or is visually more
  consistent with another instruction/entity, do not rebind it to the all-negative target unless current-frame
  face/head/hair or unique identity evidence clearly proves the previous assignment was wrong.
- ALL-NEGATIVE CANNOT OVERRIDE STRONG REJECTS WITHOUT NEW EVIDENCE: when initial_detection_block rejected an
  entity for identity_not_verifiable, missing required placement features, or wrong-person/entity conflict,
  recovery must provide new current-frame identity evidence stronger than clothing/torso/shoulder similarity.

Return ONLY valid JSON:
{{
  "keyframes": [
    {{
      "keyframe_id": "keyframe_0001",
      "entities": [
        {{
          "instruction_id": "instr_001",
          "entity_id": "entity_01",
          "present": true,
          "confidence": 0.82,
          "location_description": "large foreground side-profile face on the left",
          "visibility_quality": "clear",
          "approximate_area_fraction": 0.35,
          "visible_parts": ["face", "hair", "head", "upper_body"],
          "viewpoint": "side_profile",
          "identity_cues": ["matching hair silhouette", "matching face structure", "same recurring character alias"],
          "identity_verifiable_from_visible_parts": true,
          "localization_clarity": "high",
          "entity_visibility_completeness": "sufficient",
          "target_attachment_point": "none",
          "target_attachment_visible": false,
          "attachment_visibility": {{"left_shoulder": false, "right_shoulder": false}},
          "body_orientation": "side_profile",
          "anatomical_left_screen_side": "unknown",
          "anatomical_right_screen_side": "unknown",
          "attachment_visibility_reasoning": "not a body-side placement decision",
          "candidate_evaluations": [
            {{
              "candidate_location": "large foreground side-profile face on the left",
              "visible_parts": ["face", "hair", "head"],
              "identity_matches": ["hair silhouette and face profile match reference identity"],
              "identity_conflicts": [],
              "decision": "recover_present"
            }}
          ],
          "location_corrected": true,
          "reasoning": "Recovered by focused all-negative candidate comparison."
        }}
      ]
    }}
  ]
}}

Use English for all string values. Include one entry for EVERY listed all-negative entity in EVERY keyframe.
"""

KEYFRAME_EDIT_COMPLETION_QA_PROMPT = """You are validating whether a keyframe edit pipeline completed correctly.

Image 1 = EDITED keyframe (candidate result).
Image 2 = ORIGINAL keyframe (before editing).
Images 3+ = ENTITY EDIT REFERENCE cards from entity_refs/instr_00N_ref_canonical.png.
Each card's LEFT panel is the original entity reference; RIGHT panel is the intended edited entity reference.

SCENE STORY CONTEXT (for identity continuity and naturalness checks only):
{scene_story_context}

PLANNED EDIT INSTRUCTIONS (what SHOULD have been done):
{canonical_edit_block}

ENTITY LOCATIONS (where edits should apply):
{entity_locations_block}

ORIGINAL VISIBILITY / LOCKED REGION CONSTRAINTS:
{visibility_constraints_block}

TASK:
Determine whether:
1. FRAME STRUCTURE PRESERVED: same canvas size, aspect ratio, camera framing, and black bars (letterboxing/pillarboxing). The edited image must NOT lose any black bars that were in the original.
2. Every planned edit instruction was completed correctly on the right entity at the right location.
3. The edited appearance matches the RIGHT panel of the corresponding canonical reference card and the PLANNED EDIT INSTRUCTIONS.
4. ORIGINAL ENTITY STATE PRESERVED: pose, action, facial expression, gaze, head/body angle, occlusion,
   visible extent, blur, texture, and scene interaction from the ORIGINAL keyframe must remain unchanged
   except for explicitly instructed attributes. Any visible change to mouth/eye state, smile/frown, gaze
   direction, head turn, body posture, gesture, or action is a hard failure even for hair/headwear edits.
   LIGHTING is evaluated separately under item 5 — minor local lighting/shadow shifts at edited pixels are
   acceptable and must NOT by themselves fail this item.
5. PHOTOREALISTIC SCENE INTEGRATION & LIGHTING (LENIENT): the edited result should look broadly natural in
   the scene. Lighting only needs to be roughly appropriate — same general direction/brightness mood as the
   original frame is enough. Do NOT require pixel-perfect shadow/highlight preservation.
   Fail only for OBVIOUS problems: fake/pasted/studio-flat/over-smoothed faces, sticker-like objects, or
   lighting that is clearly impossible for the scene (for example a face lit from the wrong side like a
   reference card pasted onto a differently lit frame).
   Tolerate minor relighting, small shadow/highlight shifts, slight color-temperature changes, and local
   brightness differences around hair/hat/removal/inpaint boundaries when the edit otherwise looks natural.
   For hair/headwear edits, reject pasted/redrawn/over-smoothed faces, not subtle lighting grade changes.
   For face-adjacent edits, compare skin texture, jaw/cheek edges, hairline boundary, shadows, blur/noise,
   and expression against Image 2. Any obvious cut-out mask edge, pasted-face boundary, beautified/redrawn
   face, or reference-card-like head replacement is a hard failure.
   For small/distant/background targets, reject if the edited attribute is sharper, cleaner, higher-detail,
   brighter, or more front-facing than the original small target region. The edit must inherit the original
   blur/noise/compression and distance scale; a tiny target must not look like a pasted high-resolution sticker.
6. NO unrelated people, objects, or background regions were incorrectly edited.
   CRITICAL BACKGROUND CHECK: If the PLANNED EDIT INSTRUCTIONS do not explicitly request changing the background, the background MUST remain exactly as it is in the ORIGINAL keyframe. Any alteration to walls, floors, ceilings, furniture, props, or scenery is a failure.
   Also reject unrelated changes ON THE TARGET ENTITY itself: clothing/outfit, face, expression, gaze,
   pose, action, body shape, hands/arms, or visible extent must not change unless explicitly requested.
   Reject if a foreground occluder, adjacent person, or non-target person's hair/head/body was recolored or
   received the hat/accessory intended for the target.
   Inspect the full frame for small local background artifacts, including right/left edge wall patches,
   texture smears, duplicated/missing details, or inpaint bleed. Minor color/lighting shifts inside a
   removal/inpaint seam or directly around an edited attribute are acceptable and must NOT fail QA by themselves.
   Compare Image 1 to Image 2 area-by-area outside the listed edit locations and removal silhouettes. Even a
   small changed wall/floor/pillar/edge patch is background_unedited_regions_preserved=false.
   Hard reject if the edited image contains any newly created/pasted-in person, actor-like figure, look-alike,
   full body, face, torso, or entity that was not visible in the original keyframe. Removing a target must not
   replace it with another person or with an entity from a different instruction.
7. PHYSICAL PLACEMENT / OCCLUSION: for placed objects on a named body side (e.g. left shoulder), verify
   the object appears only if that exact side/attachment point is visible in the ORIGINAL keyframe. If the
   character shows the opposite shoulder or the requested side is hidden, the object should be occluded/absent.
   Reject if the object is placed on the wrong shoulder or visibly floats on an occluded side.
8. PLACED OBJECT SCALE / ORIENTATION: for placed objects (e.g. the yellow duck), verify the object's size
   is physically reasonable and matches the relative scale in the canonical reference card. It should sit on
   the shoulder, not dominate the torso/head. Verify the object orientation follows the subject's body
   orientation and the canonical relative orientation, not an arbitrary pasted/sticker direction.
   Reject if the duck is too large/small relative to the shoulder/head, lacks contact shadow, floats, is
   pasted flat over clothing, ignores the shoulder plane, points in an inconsistent direction, or appears
   on the wrong screen side for the subject's anatomical left/right.

EDIT COMPLETION IS THE PRIMARY GATE:
- Before judging preservation/naturalness, inspect EACH planned instruction against ENTITY LOCATIONS.
- For every non-removal attribute edit, verify the target entity at the listed location visibly has every
  requested attribute from the PLANNED EDIT INSTRUCTIONS (for example hair color AND hat/headwear).
- For every removal/delete edit, verify the located target is actually absent, not shifted, replaced, or
  partly visible.
- If any planned instruction is missing, partially missing, applied to the wrong entity, or only applied in
  a weak/ambiguous way, set edit_completed=false.
- If a canonical reference is provided and any instructed target attribute does not match the RIGHT panel at
  a moderate standard, set canonical_reference_alignment_ok=false.
- Do not let good background preservation, good framing, or good state preservation compensate for a missing
  planned edit. A frame with the target entity unedited is a failed edit even if it otherwise preserves the
  original image well.
- If edit_completed=false, passed must be false and score should normally be <=0.35.

NON-EDITED REGION PRESERVATION IS ALSO A HARD GATE:
- After checking edit completion, compare Image 1 against Image 2 across the FULL frame.
- If the instructions do not explicitly target the background, every background region outside the exact edited
  entity silhouette or minimal removal inpaint seam must remain unchanged: walls, floors, ceilings, furniture,
  props, scenery, black bars, edge/corner regions, lighting, texture, and perspective.
- Non-target people/objects must remain unchanged. If any non-target person is recolored, removed, shifted,
  replaced, blurred, relit, or receives the target edit, set unrelated_edit_changes_absent=false and passed=false.
- Non-requested target attributes must remain unchanged. For example, a hair/hat edit must not alter face shape,
  expression, gaze, pose, clothing, hands/arms, or body silhouette.
- If background_unedited_regions_preserved=false or unrelated_edit_changes_absent=false, passed must be false
  and the retry_focus_prompt must identify the changed non-edit regions to preserve.
- non_edit_region_change_severity must be one of: `none`, `trace`, `moderate`, `severe`.
  Use `trace` only for a tiny local seam/halo directly touching the edit boundary with no recognizable non-edit content changed.
  Use `moderate` or `severe` for clearly visible changed non-edit patches, non-target people/objects, or broader repainting/drift.
- If non_edit_region_change_severity is `moderate` or `severe`, background_unedited_regions_preserved=false and passed=false.

Return ONLY valid JSON:
{{
  "passed": false,
  "score": 0.0,
  "frame_structure_preserved": false,
  "edit_instruction_requirements_met": false,
  "edit_completed": false,
  "canonical_reference_alignment_ok": false,
  "original_entity_state_preserved": false,
  "photorealistic_scene_integration_ok": false,
  "unrelated_edit_changes_absent": false,
  "background_unedited_regions_preserved": false,
  "non_edit_region_change_severity": "none",
  "non_edit_region_change_summary": "",
  "failed_aspects": [],
  "feedback": "short summary of pass/fail reasons",
  "retry_focus_prompt": "if failed: specific operations to AVOID on retry; empty if passed",
  "positive_prompt": "if failed: what was done correctly and should be KEPT on retry; empty if passed"
}}

Rules:
- frame_structure_preserved=false if the edited image lost the black bars (letterboxing) from the original, or if it was cropped/reframed.
- edit_instruction_requirements_met is the PRIMARY verdict. Set it true only when the result roughly satisfies
  ALL planned edit instructions on the correct entities/locations. If any planned edit is missing, wrong,
  applied to the wrong target, or only partially done, set edit_instruction_requirements_met=false.
- edit_completed=true only when every planned instruction is visibly completed on the right entity at the
  right location. Set edit_completed=false for unchanged target hair/headwear/accessories, missing requested
  attributes, remaining removal targets, wrong-entity edits, or partial completion of only one instruction
  when multiple instructions are planned.
- original_entity_state_preserved=false for clear identity/state drift: changed face shape/identity,
  changed mouth/eye state, changed pose/action/gesture/posture, changed gaze/head orientation, changed
  clothing/outfit color/pattern/shape, or changed occlusion/visible extent unless that exact attribute is
  explicitly requested. Minor local lighting/shadow differences alone must NOT set this false.
- photorealistic_scene_integration_ok=false only for OBVIOUS fake/pasted/studio-flat/over-smoothed results,
  or lighting that is clearly impossible/unnatural for the scene. Roughly appropriate scene lighting is
  enough. Tolerate minor relighting, shadow/highlight shifts, color-temperature drift, and local brightness
  changes at edit boundaries. Treat pasted-on faces/heads and sticker-like placed objects as obvious failures
  even if the requested attribute is present.
- LIGHTING TOLERANCE: if the only reported problem is lighting/shadow/highlight/color-temperature mismatch,
  but the edit is completed, identity/pose/expression/clothing/background are preserved, and the result looks
  broadly natural, keep photorealistic_scene_integration_ok=true and do not fail solely for lighting.
- passed=true only when frame_structure_preserved, edit_instruction_requirements_met, edit_completed,
  canonical_reference_alignment_ok, original_entity_state_preserved, photorealistic_scene_integration_ok,
  unrelated_edit_changes_absent, and background_unedited_regions_preserved are all true, score>=0.55,
  and no placed-object scale/orientation problems are visible.
- unrelated_edit_changes_absent=false when the edit changes non-requested attributes on the target entity,
  including clothing/outfit color or shape, face/expression, gaze, pose, action, body shape, or hand/arm position.
- unrelated_edit_changes_absent=false and background_unedited_regions_preserved=false when a removal/inpaint
  result creates or pastes a new person/entity, replaces the target with another person, or synthesizes an
  absent instruction entity (for example adding entity_03 where entity_03 was not present in the original).
- edit_completed=false and unrelated_edit_changes_absent=false when a removal/delete target is replaced by
  another person, a look-alike, a new face/body/torso, or any actor-like substitute instead of empty background.
- unrelated_edit_changes_absent=false when hair/headwear/accessory edits spill onto an occluding foreground
  person, adjacent person's hair, or any non-target head/body region.
- original_entity_state_preserved=false for clear clothing/outfit drift, face identity/shape drift,
  pose/action/body-angle/gesture drift, gaze/head-orientation drift, or expression drift on an edited or
  unedited person. Do not fail this item for lighting/shadow/highlight differences alone.
- photorealistic_scene_integration_ok=false when a placed object has wrong relative size, wrong orientation,
  missing contact shadow, pasted/sticker appearance, or grossly impossible lighting/perspective on the shoulder.
  Minor object-lighting mismatch is acceptable if the object otherwise sits naturally.
- photorealistic_scene_integration_ok=false when a small/distant edited person or object looks pasted on,
  overly sharp, over-detailed, or disconnected from the original frame blur/noise/depth of field. Do not fail
  solely because the small target is slightly brighter/darker than before if it still looks scene-consistent.
- edit_completed=false and canonical_reference_alignment_ok=false when a planned removal/delete target remains
  visibly present at the listed location.
- background_unedited_regions_preserved=false when any visible background/object/person outside the intended
  edit silhouette changes. If the PLANNED EDIT INSTRUCTIONS do not explicitly target the background, ANY change
  to the background (walls, floors, furniture, scenery, etc.) is a hard failure, even if the changed area is small.
  This includes added/removed furniture or props such as a chair, bench, railing, lamp, table, doorway, wall
  panel, poster, luggage, or other background object not covered by the edit instruction.
- non_edit_region_change_severity must be one of: `none`, `trace`, `moderate`, `severe`.
  Use `trace` only for a tiny local seam/halo directly touching the edit boundary with no recognizable non-edit content changed.
  Use `moderate` or `severe` for clearly visible changed non-edit patches, non-target people/objects, or broader repainting/drift.
- If non_edit_region_change_severity is `moderate` or `severe`, background_unedited_regions_preserved=false and passed=false.
- non_edit_region_change_summary: short English summary of the changed non-edit regions; empty if severity=`none`.
- unrelated_edit_changes_absent=false and background_unedited_regions_preserved=false if a non-target person is
  removed, erased, shifted, replaced, or altered. For single-person removals, explicitly check edge/corner
  passengers such as the far-right or far-left woman/person and preserve them unchanged.
- For target_instance_scope=single, edit_completed=false and unrelated_edit_changes_absent=false if the edit
  removed or changed multiple matching instances instead of the one located target.
- For target_instance_scope=single, explicitly inspect whether a second similar person/object was also removed,
  replaced, or altered. If two people/objects were edited for one single-instance instruction, set passed=false
  even if the intended target was also edited correctly.
- For planned removal/delete instructions, inspect Image 1 directly. edit_completed=true when the located target
  is actually absent from Image 1 and the inpainted region is plausible. Do not require any separate observed
  edit-comparison report.
- For small/background removal targets, zoom mentally into the exact ENTITY LOCATIONS region from Image 2 and
  compare against Image 1. If any recognizable target pixels remain (cap/hat, vest, face, torso silhouette,
  person-shaped remnants, or the same target shifted slightly), set edit_completed=false even if the full frame
  looks plausible at first glance.
- edit_completed=false and canonical_reference_alignment_ok=false when a placed object appears on the wrong
  physical side (right shoulder instead of requested left shoulder) or is visible when the requested side is
  occluded in the original frame.
- retry_focus_prompt must list mistakes to avoid, not new edit goals. When original_entity_state_preserved=false,
  explicitly mention each locked state that drifted, such as "do not change head orientation", "do not change
  facial expression/gaze", or "do not change clothing". Do not require perfect lighting match on retry.
- If edit_instruction_requirements_met=false or edit_completed=false, retry_focus_prompt is REQUIRED and must
  list the concrete edit-instruction failures to avoid repeating, such as skipped removal, wrong target,
  missing hair color, missing hat, wrong shoulder, or pasted/reference-card replacement.
- positive_prompt (REQUIRED when failed; empty when passed): list the editing operations that were done CORRECTLY in this attempt and should be KEPT/MAINTAINED on the next retry. Phrase as positive instructions.
- If no planned edit was done correctly, positive_prompt should still preserve safe context: keep the original
  frame structure, camera framing, target identity/state, non-target people, and background outside the exact
  edit regions. Do not use positive_prompt to cancel a planned edit.
- If a planned instruction is removal/delete, positive_prompt must NEVER say to keep, preserve, maintain,
  restore, or leave visible the removal target. It may say to preserve non-target people/background outside
  the removal silhouette, but the located removal target must still be removed on the retry.
- Use English for all string values.
"""

KEYFRAME_ENTITY_DETECT_PROMPT = """You are detecting edit-target entities in ONE video keyframe.

Image 1 = TARGET KEYFRAME (the frame to be edited).
Images 2+ = per-entity FRONT-VIEW reference images from entity_refs/instr_00N_ref_front.png.

SCENE STORY CONTEXT (use only as identity/continuity evidence; never hallucinate absent entities):
{scene_story_context}

ENTITIES TO DETECT (evaluate EVERY entity; reference image mapping):
{entity_catalog_block}

TASK:
For EACH entity, decide whether it appears in image 1. If present, describe its precise location.

RULES:
- IDENTITY vs APPEARANCE (CRITICAL): Match the SAME person/object (entity_id), NOT an identical costume across frames. Image 2 may include views from other scenes with different outfits — that is expected. NEVER reject solely because clothing in image 1 differs from a panel in image 2 if the face/body identity matches. Reject look-alikes, background extras, and limb-only fragments — not the same person in a different outfit.
- VISIBLE-PART-ONLY IDENTITY (CRITICAL): compare only the parts of each candidate that are actually visible in
  image 1 with the corresponding parts of the reference. Do NOT penalize the candidate because reference
  clothing, lower body, shoulders, or accessories are hidden/cropped in image 1. If a visible key region
  (face/head/hair/hairline/clear profile, or a substantial torso/headwear/accessory region with distinctive
  cues) has high similarity to the reference, mark present=true. A half shoulder, arm/hand sliver, tiny edge
  crop, or generic clothing corner is not a key region and must not be accepted by itself.
- SPATIAL DISAMBIGUATION: when two entities could overlap at the same screen region (e.g. both described at the extreme left edge), assign the partial figure to the entity whose distinctive clothing/accessories actually match the visible pixels (vest vs suspenders vs cap). Mark present=false for the other entity at that region — never double-assign one cropped limb to two instruction_ids.
- Respect target_instance_scope:
  - single: exactly ONE tracked individual matching subject_features
  - multiple: every instance in the frame matching subject_features
- For target_instance_scope=single, never return a location that covers a group or multiple similar people.
  If several similar people are visible, choose only the one best matching the reference and subject_features;
  all other look-alikes are out of scope.
- VIEWPOINT TOLERANCE (IMPORTANT): side face, 3/4 profile, back view, head turned away, head tilt,
  sitting/standing pose changes, bending/crouching/leaning posture changes, soft focus, and partial
  occlusion can still be present=true. Do NOT require a frontal face.
- STATE-CHANGE ROBUSTNESS (IMPORTANT): expression, gaze direction, mouth state, gesture, action beat,
  body posture, head angle, seated/standing/bending/crouching/leaning state, and walking/running motion are
  TEMPORARY state changes, not identity contradictions. Never mark present=false solely because these differ
  from the front-view reference if stable visual identity cues still match.
- For side/back/profile views, rely on stable visual cues: distinctive hair silhouette/color, hairline,
  concrete facial structure details, clothing colors/pattern, accessories, body build, and neighboring-frame
  continuity. Scene role/posture can support but cannot be one of the required stable visual identity cues.
- PRIORITIZE STABLE IDENTITY OVER STATE: when state cues (expression, gaze, pose, action, gesture, head angle)
  differ but face/hair/head/body identity cues match, keep present=true and lower confidence only if the
  visible identity evidence itself is weak.
- Use SCENE STORY CONTEXT to decide whether a partially visible head/hair/back likely belongs to the same
  continuing character in this shot. Story context can raise confidence for real visible pixels, but cannot
  make present=true when no pixels of that entity are visible.
- STORY/ROLE/ALIAS ARE SUPPORTING ONLY: role names, scene role, conversation/engagement, prominence,
  narrative lead status, and story context can only support an already strong visual match. They must NEVER be
  the primary reason for present=true. A candidate needs concrete visual identity evidence such as hair
  color+hair silhouette, hairline, face shape, jaw/nose/eyes/eyebrows/mouth, distinctive accessories, or
  stable clothing/body cues.
- ANTI-NAME/ROLE IDENTITY REJECTION (CRITICAL): Do NOT mark present=false by claiming a visible person is
  "Jack", "the main character", "a different character", "a named protagonist", or by referencing any actor
  name (like "Leonardo DiCaprio") or narrative role. Character names, actor identities, and "main character"
  labels are NOT visual identity evidence. They must NEVER be used to reject a candidate whose face/hair
  features match the reference image. If the visible facial structure, hair silhouette, hairline, and head
  shape match the reference, mark present=true regardless of any name/role association from story context.
  The reference image identity is the ground truth — not narrative character names.
- WEAK IDENTITY EVIDENCE REJECTION: generic phrases like "facial structure", "expression", "scene role",
  "engagement in conversation", "same story role", "prominent person", or a character/name alias alone are
  insufficient. If identity_cues do not contain specific visible face/hair/body/accessory details, mark
  present=false or lower confidence and explain the uncertainty.
- HEAD/HAIR EDIT IDENTITY STRICTNESS: when the edit_prompt changes hair, hair color, hat, cap, or headwear,
  "hair color / hair silhouette + facial structure" is NOT enough by itself. You must cite concrete hair or
  face details such as updo, hairline, hair parting, curls/waves/straightness, bun/braid/bangs, face shape,
  eye/nose/jaw/mouth/cheek/chin details, or a clear "matches the reference face" comparison. If the current
  hairstyle/hairline/volume/silhouette visibly conflicts with the reference identity, put that conflict in
  identity_conflicts and mark present=false.
- A candidate-level comparison such as "matches face and hair", "matches hair and facial features", or
  "matching face shape/profile plus hair silhouette" counts as concrete visual identity evidence when the
  candidate_evaluations entry names the visible candidate and has no concrete identity_conflicts.
- CANDIDATE AUDIT REQUIRED: for every present=true result, candidate_evaluations MUST contain the selected
  candidate, concrete identity_matches, any identity_conflicts, and why it is not a look-alike. Do not leave
  candidate_evaluations empty for present=true.
- The 'subject_features' field often describes the entity's clothing in their FIRST appearance. Do NOT use it as a strict filter for later scenes. If the character has changed clothes but their face/hair identity matches the reference, they are the SAME person.
- VISUAL REFERENCE OVERRIDES TEXT: If the visible face/hair matches the Reference Image (Image 2+), you MUST mark present=true, even if the current clothing completely contradicts the 'subject_features' text.
- AGE/GENDER TERMINOLOGY: Terms like "girl", "woman", "boy", "man" in subject_features are often used interchangeably. Do NOT reject an identity match by claiming the reference is a "child/girl" while the frame shows a "woman" (or vice versa). Rely strictly on visual face/hair matching, not semantic age labels.
- SIDE PROFILE + WARDROBE CHANGE ROBUSTNESS: A side profile or angled face will naturally look different from the front-view reference. You MUST extrapolate identity using hair color, hair silhouette, and general facial structure. If these plausibly match the reference, mark present=true EVEN IF their clothing has completely changed. Do NOT require a frontal face to confirm identity when the outfit changes.
- CRITICAL RULE FOR IDENTITY: Core facial and hair features determine identity. Wardrobe changes (different clothes) or temporary hairstyle changes (e.g., updo vs. loose hair) do NOT constitute an identity change. If the face and core physical features match the reference, mark as present=true even if the outfit differs. Do not claim identity conflicts due to clothing changes.
- BENDING/CROUCHING ROBUSTNESS: for bent-over, crouching, leaning, seated, or low-posture people, actively
  inspect small visible face/head/hair regions. A small but identifiable face/profile should outrank a larger
  back-view body candidate with no face/hair identity.
- Strong vs weak features: clothing, hair style/color, hair silhouette, hairline, face shape, jaw/nose/eyes,
  hat/cap/headwear, suspenders, vest, beard/stubble, and distinctive accessories are STRONG identity features.
  Pose, action, gaze, expression, scene role, conversation role, prominence, and alias/story context are WEAK
  supporting features. Weak features cannot make present=true without concrete visual identity evidence.
- If subject_features identify hair (e.g. blond/middle-parted hair) and do NOT mention a hat/cap, do not
  accept a candidate whose matching evidence is a hat/cap that hides or contradicts that hair cue.
- If subject_features mention specific clothing (e.g., suspenders, vest, dress), the visible candidate must show that cue OR an equally strong face/body identity match. Do not promote a nearby extra based only on one shared weak cue.
- MISSING-CROPPED FEATURES ≠ ABSENCE (CRITICAL): When subject_features mentions clothing/accessories that are
  cropped out or below the frame edge due to camera framing (close-up face shot, tight framing, partial crop),
  the entity is STILL present=true when the visible face/head/hair matches the reference. Features like
  suspenders, a vest, a dress hem, or a belt that are physically outside the frame in a close-up must NEVER
  cause present=false. Such features are absent due to framing, not because the person is different. When
  a close-up large face fills the frame and facial features/hairline/hair structure match the reference,
  mark present=true even if no clothing features are visible.
- LIGHTING-INDUCED HAIR COLOR SHIFT (CRITICAL): The same person's hair color appearance can vary dramatically
  under different lighting. A person with dark-blond/medium-brown hair in close-up indoor lighting may appear
  bright blond in outdoor wide shots, or vice versa. When facial structure, hair silhouette, hairline shape,
  hair parting, and face shape match the reference, do NOT treat apparent hair color tone differences
  (e.g. "blond" vs "medium-brown" or "warm-brown") as identity contradictions. These are lighting artifacts.
  Only mark "clear_mismatch" for hair when the style, silhouette, length, texture, or parting fundamentally
  contradicts the reference.
- If image 1 contains a large central/close-up person whose face/head/hair matches the reference, do NOT
  reject that person solely because the current outfit differs from subject_features from an earlier frame.
- FACE-ONLY / EXPRESSION ROBUSTNESS: a close-up or angled face can be present=true even when only face/hair
  are visible and the expression, mouth shape, gaze, or emotion differs from the reference. Expression and
  action are temporary state, not identity. If face/hair identity is clear, do not require the original
  clothing/dress to be visible in the close-up.
- For large central faces, prioritize concrete facial structure details, hairline/updo, hair color, and hair
  silhouette over clothing. Do not mark a large matching face absent just because expression, gaze, lighting,
  or camera angle differs from the front-view reference. Do not mark a large face present when the only
  evidence is generic "facial structure", story role, prominence, or expression.
- LOW-LIGHT / BACKLIT / SOFT-FOCUS ROBUSTNESS: if the face is shadowed, backlit, softly blurred, or partly
  silhouetted, still keep present=true when the visible profile, hairline, hair silhouette, body build, or
  accessory cues concretely match the reference. Write those specific cues into identity_cues instead of using
  generic wording.
- Mark side/back/profile/side-body present=true when enough head/hair/torso/body/back is visible and at least
  two stable visual identity cues match. Stable cues include hair color/silhouette, hairline, concrete face
  details, clothing color/pattern, distinctive accessories, and body build. Use confidence around 0.72-0.92
  if identity is plausible but not frontal.
- WIDE-SHOT / SMALL-SUBJECT TOLERANCE: in wide-angle or long-shot keyframes, the entity may occupy a small
  fraction of the frame. Do NOT mark absent solely because the target is small, distant, or in the background.
  Mark present=true when the whole body/head+torso is locatable and stable cues (clothing colors/pattern,
  hair silhouette, accessories, body build, continuity) match. If hair color and clothing match
  but facial details are too small, this can still be present=true with lower confidence. Use visibility_quality="too_small"
  only when small but still identifiable; use identity_cues to explain why.
- Reject only when the small region is just a meaningless speck, limb-only fragment, shoulder sliver,
  or cannot be distinguished from similar people.
- CLOSE-UP / NEAR-CAMERA TOLERANCE: if the entity is very close to the camera, fills a large part of the
  frame, or is cropped by frame edges, still mark present=true when the face/head/hair or upper torso is
  visible and identity cues match. Do NOT reject just because the full body is not visible.
  For close-ups, set entity_visibility_completeness="partial" or "sufficient" as appropriate and describe
  the visible edit region precisely.
- FACE-ONLY / EXPRESSION ROBUSTNESS: a close-up or angled face can be present=true even when only face/hair
  are visible and the expression, mouth shape, gaze, or emotion differs from the reference. Expression and
  action are temporary state, not identity. If face/hair identity is clear, do not require the original
  clothing/dress to be visible in the close-up.
- PHYSICAL PLACEMENT VISIBILITY: for instructions that place/add an object on a body side
  (e.g. "left shoulder", "right shoulder"), determine whether that exact side/attachment point is visible
  in image 1. If the target turns so only the opposite shoulder is visible, the requested-side object should
  be hidden/occluded, not visibly placed on the opposite shoulder.
- For physical placement targets, choose the identity-best candidate first, then judge attachment visibility.
  Do not select a large turned-away/back-view shoulder over a smaller matching face/head/hair candidate just
  because it exposes a shoulder-like region.
- For physical placement targets, do not accept hair/shoulder/back-of-head alone when another nearby candidate
  better matches a different entity. Require strong current-frame identity evidence and no cross-entity conflict.
- For physical placement edits, do NOT mark target_attachment_visible=true from scene continuity alone.
  The exact requested attachment point must be visible in the current keyframe. Back-of-head/back-view,
  turned-away, heavy occlusion, or ambiguous anatomical left/right should set target_attachment_visible=false
  even if the person is present.
- For body-side placement, output body_orientation plus anatomical_left_screen_side and
  anatomical_right_screen_side. These fields should describe where the subject's own left/right sides appear
  in screen coordinates (for example "screen-left", "screen-right", "hidden/occluded", or "unknown").
- For removal/delete edits: partial edge crops are SUFFICIENT only when a substantial torso/head/cap/hat/vest
  region is visible with distinctive cues. A tiny shoulder sliver, arm sliver, or corner of clothing is NOT
  sufficient, even if it resembles the reference.
- For shoulder/torso accessory edits: partial torso/shoulder is sufficient only when the visible region is large
  enough to edit and identity cues are verifiable; half a shoulder or a tiny cropped fragment is not enough.
- present=true when you are confident the specific entity is visually recognizable. Do NOT require a frontal
  face or large editable area. Use present=false only for true absence, pure limb/shoulder fragments without
  identity cues, indistinguishable extras, or clear identity contradictions.
- confidence: float 0.0–1.0 representing match certainty.
- If present=false, location_description must be empty string.

Per-entity output fields:
instruction_id, entity_id, present, confidence, location_description, visibility_quality,
approximate_area_fraction, visible_parts, viewpoint, identity_cues, identity_verifiable_from_visible_parts,
localization_clarity, entity_visibility_completeness, target_attachment_point, target_attachment_visible,
attachment_visibility, body_orientation, anatomical_left_screen_side, anatomical_right_screen_side,
attachment_visibility_reasoning, candidate_evaluations, reasoning

Return ONLY valid JSON:
{{
  "entities": [
    {{
      "instruction_id": "instr_001",
      "entity_id": "entity_01",
      "present": false,
      "confidence": 0.0,
      "location_description": "",
      "visibility_quality": "ambiguous",
      "approximate_area_fraction": 0.0,
      "visible_parts": [],
      "viewpoint": "front|side_profile|three_quarter|back|unknown",
      "identity_cues": [],
      "identity_verifiable_from_visible_parts": false,
      "localization_clarity": "low",
      "entity_visibility_completeness": "fragment",
      "target_attachment_point": "left_shoulder|right_shoulder|shoulder|none",
      "target_attachment_visible": false,
      "attachment_visibility": {{"left_shoulder": false, "right_shoulder": true}},
      "attachment_visibility_reasoning": "brief note on which shoulder/side is visible or occluded",
      "candidate_evaluations": [
        {{
          "candidate_location": "brief candidate location",
          "visible_parts": ["face", "hair"],
          "identity_matches": ["specific visible identity match"],
          "identity_conflicts": [],
          "decision": "present|reject_lookalike|uncertain"
        }}
      ],
      "reasoning": "brief explanation"
    }}
  ]
}}

Use English for all string values. Include one entry for EVERY entity in the catalog.
"""

KEYFRAME_ENTITY_LOCATION_VERIFY_PROMPT = """You are verifying entity localization on ONE video keyframe.

Image 1 = TARGET KEYFRAME.
Images 2+ = per-entity front-view reference images.

SCENE STORY CONTEXT (use only as identity/continuity evidence; never hallucinate absent entities):
{scene_story_context}

INITIAL DETECTION RESULTS (from a prior pass — verify and correct if needed):
{detection_results_block}

ENTITY CATALOG:
{entity_catalog_block}

TASK (run ONCE):
For EACH entity listed above, verify whether the initial detection is correct.
- IDENTITY vs APPEARANCE: Do NOT reject solely because clothing differs if the face/body identity matches the reference. However, do NOT accept look-alikes or background extras just because they wear similar clothing.
- VISIBLE-PART-ONLY IDENTITY: verify identity only from regions actually visible in image 1. Missing or cropped
  reference regions are not contradictions. If the visible face/head/hair/profile or another substantial
  distinctive key region matches the reference, keep or correct to present=true even when the first-appearance
  outfit is not visible. Shoulder-only, arm-only, tiny edge-crop, and generic clothing-only fragments remain
  insufficient.
- SMALL / BLURRY / BACK-VIEW CAUTION: for tiny, blurry, shadowed, or mostly back-facing candidates, confidence
  must drop noticeably. If the visible pixels do not show direct face/head/hair/profile evidence or another
  unmistakably unique cue, prefer present=false instead of preserving a weak prior detection.
- SPATIAL DISAMBIGUATION: Do not double-assign one cropped limb to two instruction_ids. Assign it to the entity whose distinctive clothing/accessories actually match the visible pixels.
- CROSS-ENTITY CANDIDATE ARBITRATION: treat every visible person/object candidate as assignable to at most one
  single-scope instruction_id. If two instruction_ids could describe the same candidate, choose the one with
  stronger current-frame identity evidence and mark the other present=false or list an identity_conflict. Do
  not bind one candidate to both a removal target and a placement target.
- VIEWPOINT TOLERANCE: If initial detection says present=false only because the face is side-facing,
  back-facing, turned away, soft-focus, not frontal, head-tilted, or pose-shifted, correct it to present=true
  when stable cues match.
- STATE-CHANGE ROBUSTNESS: expression change, gaze change, mouth state, gesture, action beat, body posture,
  head angle, seated/standing/bending/crouching/leaning state, and walking/running motion are TEMPORARY state
  changes, not identity contradictions. Do not downgrade a plausible same-identity match solely for these
  differences.
- For side/back/profile/side-body views, accept enough visible head/hair/torso/body/back plus at least two
  stable visual identity cues (hair silhouette/color, hairline, concrete face details, distinctive clothing,
  accessories, body build, continuity). Scene role/posture can support but cannot be one of the required
  stable visual identity cues. Use confidence around 0.72-0.92 for plausible non-frontal/small matches.
- PRIORITIZE STABLE IDENTITY OVER STATE: when expression/gaze/pose/action differs but stable face, hair,
  head-shape, body-build, or accessory cues match, keep or correct to present=true and lower confidence only
  if the visible identity evidence itself is weak.
- Use SCENE STORY CONTEXT and neighboring-shot role descriptions to interpret partial hair/head/back views
  when visible pixels are ambiguous. Do not override a true absence; context supports visible evidence only.
- STORY/ROLE/ALIAS ARE SUPPORTING ONLY: role names, scene role, conversation/engagement, prominence,
  narrative lead status, and story context can only support an already strong visual match. They must NEVER be
  the primary reason for present=true. A candidate needs concrete visual identity evidence such as hair
  color+hair silhouette, hairline, face shape, jaw/nose/eyes/eyebrows/mouth, distinctive accessories, or
  stable clothing/body cues.
- ANTI-NAME/ROLE CORRECTION FOR VERIFY (CRITICAL): If the initial detection marked present=false with
  reasoning that references character names (e.g. "Jack", "the main character"), actor names (e.g.
  "Leonardo DiCaprio"), or narrative role labels as the rejection reason, CORRECT this to present=true
  when the visible face/hair/facial-structure in image 1 matches the reference image. Character names,
  actor identities, and "main character" labels are NOT visual evidence — they must NEVER be accepted as
  valid reasons to keep an entity absent. The reference image identity is the ground truth.
- LIGHTING/TONE VERIFY CORRECTION: If initial detection rejected an entity citing hair color mismatch
  (e.g. "dark-haired/medium-brown" vs "blond") but the candidate's facial structure, hair silhouette, and
  hairline shape match the reference, CORRECT to present=true. Hair color appearance shifts significantly
  under different scene lighting in video footage; this is a lighting artifact, not identity mismatch.
- MISSING-CROPPED VERIFY CORRECTION: If initial detection marked present=false citing features that are
  cropped out of the frame (e.g. "missing required suspenders", "missing suspenders for placement"),
  CORRECT to present=true when the visible face/head/hair and head/torso features match the reference.
  Cropped-out clothing features are absent due to camera framing, not because the person is different.
  Such features should be treated as "not_visible", never as identity contradictions.
- WEAK IDENTITY EVIDENCE REJECTION: generic phrases like "facial structure", "expression", "scene role",
  "engagement in conversation", "same story role", "prominent person", or a character/name alias alone are
  insufficient. If identity_cues do not contain specific visible face/hair/body/accessory details, keep or
  correct to present=false and explain the uncertainty.
- HEAD/HAIR EDIT IDENTITY STRICTNESS: when the edit_prompt changes hair, hair color, hat, cap, or headwear,
  "hair color / hair silhouette + facial structure" is NOT enough by itself. You must cite concrete hair or
  face details such as updo, hairline, hair parting, curls/waves/straightness, bun/braid/bangs, face shape,
  eye/nose/jaw/mouth/cheek/chin details, or a clear "matches the reference face" comparison. If the current
  hairstyle/hairline/volume/silhouette visibly conflicts with the reference identity, put that conflict in
  identity_conflicts and mark present=false.
- A candidate-level comparison such as "matches face and hair", "matches hair and facial features", or
  "matching face shape/profile plus hair silhouette" counts as concrete visual identity evidence when the
  candidate_evaluations entry names the visible candidate and has no concrete identity_conflicts.
- CANDIDATE AUDIT REQUIRED: for every present=true result, candidate_evaluations MUST contain the selected
  candidate, concrete identity_matches, any identity_conflicts, and why it is not a look-alike. Do not leave
  candidate_evaluations empty for present=true.
- The 'subject_features' field often describes the entity's clothing in their FIRST appearance. Do NOT use it as a strict filter for later scenes. If the character has changed clothes but their face/hair identity matches the reference, they are the SAME person.
- VISUAL REFERENCE OVERRIDES TEXT: If the visible face/hair matches the Reference Image (Image 2+), you MUST mark present=true, even if the current clothing completely contradicts the 'subject_features' text.
- AGE/GENDER TERMINOLOGY: Terms like "girl", "woman", "boy", "man" in subject_features are often used interchangeably. Do NOT reject an identity match by claiming the reference is a "child/girl" while the frame shows a "woman" (or vice versa). Rely strictly on visual face/hair matching, not semantic age labels.
- SIDE PROFILE + WARDROBE CHANGE ROBUSTNESS: A side profile or angled face will naturally look different from the front-view reference. You MUST extrapolate identity using hair color, hair silhouette, and general facial structure. If these plausibly match the reference, mark present=true EVEN IF their clothing has completely changed. Do NOT require a frontal face to confirm identity when the outfit changes.
- CRITICAL RULE FOR IDENTITY: Core facial and hair features determine identity. Wardrobe changes (different clothes) or temporary hairstyle changes (e.g., updo vs. loose hair) do NOT constitute an identity change. If the face and core physical features match the reference, mark as present=true even if the outfit differs. Do not claim identity conflicts due to clothing changes.
- BENDING/CROUCHING ROBUSTNESS: for bent-over, crouching, leaning, seated, or low-posture people, actively
  inspect small visible face/head/hair regions. A small but identifiable face/profile should outrank a larger
  back-view body candidate with no face/hair identity.
- Treat expression, gaze, head angle, camera distance, and clothing changes as state/wardrobe
  changes, not identity changes. A close foreground person with matching face/hairline
  should be corrected to present=true even when they are not wearing the original outfit.
- Correct false negatives for large faces caused by expression, gaze, head angle, profile/three-quarter view,
  or lighting changes. These are temporary state differences, not identity mismatches.
- WIDE-SHOT / SMALL-SUBJECT TOLERANCE: If the initial result says present=false only because the entity is
  small, distant, or in a wide shot, correct it to present=true when the whole body/head+torso is locatable
  and at least two stable visual identity cues match. Hair color plus distinctive clothing is sufficient when
  the face is too small to resolve. Use confidence around 0.72-0.92 for plausible small targets.
- Do NOT correct to present=true for tiny meaningless specks, limb-only fragments, or indistinguishable extras.
- CLOSE-UP / NEAR-CAMERA TOLERANCE: If the initial result says present=false only because the target is too
  close, cropped, or only face/head/upper torso is visible, correct it to present=true when the visible face,
  hair, head shape, clothing, or other stable cues match. Full body visibility is NOT required for close-ups.
- PHYSICAL PLACEMENT VISIBILITY: for add/place instructions on a specific side (left shoulder/right shoulder),
  verify whether that exact side is visible. If only the opposite shoulder is visible, keep the entity present
  but set target_attachment_visible=false for the requested side.
- For physical placement targets, choose the identity-best candidate first, then judge attachment visibility.
  Do not select a large turned-away/back-view shoulder over a smaller matching face/head/hair candidate just
  because it exposes a shoulder-like region.
- For physical placement targets, do not accept hair/shoulder/back-of-head alone when another nearby candidate
  better matches a different entity. Require strong current-frame identity evidence and no cross-entity conflict.
- For add/place instructions, current-frame evidence is mandatory. Do not copy target_attachment_visible,
  anatomical_left_screen_side, or anatomical_right_screen_side from a neighbor/scene-continuity assumption.
  If the view is back-of-head/back-view, turned away, or anatomical left/right cannot be determined reliably,
  set target_attachment_visible=false even when the entity itself remains present.
- For removal/delete edits: partial edge crops are SUFFICIENT only when a substantial torso/head/cap/hat/vest
  region is visible with distinctive cues. A tiny shoulder sliver, arm sliver, or corner of clothing is NOT
  sufficient, even if it resembles the reference.
- For shoulder/torso accessory edits: partial torso/shoulder is sufficient only when the visible region is large
  enough to edit and identity cues are verifiable; half a shoulder or a tiny cropped fragment is not enough.
- If present=true but location is wrong, correct location_description and adjust confidence.
- If present=true but the match is actually a look-alike or wrong person, set present=false and confidence low.
- If present=false but the entity IS visually recognizable from stable cues, set present=true with accurate
  location and confidence. Only set present=false when the entity is truly absent, indistinguishable, a pure
  limb/shoulder fragment, or a clear wrong-person/look-alike.
- Do NOT invent entities that are not in the frame.

Per-entity output fields:
instruction_id, entity_id, present, confidence, location_description, location_corrected,
visibility_quality, approximate_area_fraction, visible_parts, viewpoint, identity_cues,
identity_verifiable_from_visible_parts, localization_clarity, entity_visibility_completeness,
target_attachment_point, target_attachment_visible, attachment_visibility, attachment_visibility_reasoning,
candidate_evaluations, reasoning

location_corrected=true when you changed present status, confidence, or location_description from the initial detection.

Return ONLY valid JSON:
{{
  "entities": [
    {{
      "instruction_id": "instr_001",
      "entity_id": "entity_01",
      "present": true,
      "confidence": 0.95,
      "location_description": "center of frame, standing on stairs",
      "location_corrected": false,
      "visibility_quality": "clear",
      "approximate_area_fraction": 0.15,
      "visible_parts": ["face", "torso"],
      "viewpoint": "three_quarter",
      "identity_cues": ["distinctive dress", "hair silhouette"],
      "identity_verifiable_from_visible_parts": true,
      "localization_clarity": "high",
      "entity_visibility_completeness": "sufficient",
      "target_attachment_point": "left_shoulder",
      "target_attachment_visible": true,
      "attachment_visibility": {{"left_shoulder": true, "right_shoulder": true}},
      "attachment_visibility_reasoning": "both shoulders are visible enough for placement",
      "candidate_evaluations": [
        {{
          "candidate_location": "center of frame, standing on stairs",
          "visible_parts": ["face", "torso"],
          "identity_matches": ["specific visible identity match"],
          "identity_conflicts": [],
          "decision": "present"
        }}
      ],
      "reasoning": "verified — initial detection accurate"
    }}
  ]
}}

Use English for all string values. Include one entry for EVERY entity in the catalog.
"""

KEYFRAME_EDIT_COMPARISON_PROMPT = """You are comparing an original video keyframe with its edited version.

Image 1 = ORIGINAL keyframe (before editing).
Image 2 = EDITED keyframe (after editing).

PLANNED EDIT INSTRUCTIONS:
{canonical_edit_block}

PRE-EDIT LOCKED REGION / ORIGINAL STATE CONSTRAINTS:
{visibility_constraints_block}

TASK:
Carefully compare the two images. List EVERY visible edit operation you observe in image 2 relative to image 1.
First reason from PLANNED EDIT INSTRUCTIONS and LOCKED REGION constraints: only explicitly requested
attributes may change; all other state is locked from image 1.

Pay special attention to unintended target-entity changes: clothing/outfit color or shape, face, facial
expression, gaze, head orientation, head tilt, pose, action, body/hand/arm position, visible extent, and
local lighting/shadows. If any of these changed, list them as observed edit operations even if the requested
edit was hair/headwear.
Also inspect background and unedited people pixel regions. List small but visible unintended changes such as
repainted wall patches, background texture/color shifts, missing/duplicated objects, altered shadows, or
inpaint bleed outside the target silhouette.
For removal edits, count how many people/objects were removed or materially altered. If more than one
similar instance changed, list that explicitly as a separate observed operation.
Explicitly look for newly created or pasted-in people/entities. If any person, body, face, torso, actor-like
figure, look-alike, or entity that was not present in the original appears in the edited image, list it as
an observed operation with operation="added" or "replaced".
For placed objects, describe the object's screen side/anatomical side, orientation/facing direction,
relative size compared with the shoulder/head, contact shadow, and whether it appears pasted or physically
integrated.

For each observed change, describe:
- where in the frame (spatial location, anchor objects)
- what was edited (which person/object/region)
- what operation was performed (added, removed, recolored, replaced, relit, head-turned, clothing-changed, etc.)
- what changed from what to what (e.g. "brown hair → blue hair", "head turned left → facing forward",
  "plain cream dress → patterned yellow dress", "face shadow changed", "man removed, background inpainted")

Return ONLY valid JSON:
{{
  "observed_edit_operations": [
    {{
      "region": "where in the frame",
      "target": "who/what was affected",
      "operation": "type of edit",
      "change_description": "what changed from X to Y, or what was added/removed",
      "confidence": 0.95
    }}
  ],
  "summary": "one-paragraph summary of all changes"
}}

If no edits are visible, return an empty observed_edit_operations list and say so in summary.
Use English for all string values.
"""

SCENE_ENTITY_DETECT_PROMPT = """You are detecting edit-target entities across an entire video scene.

Image 1 = SCENE KEYFRAME GRID (multiple keyframes from the scene, labeled with keyframe indices).
Images 2+ = per-entity FRONT-VIEW reference images (one per entity below).

ENTITIES TO LOCATE:
{entity_catalog_block}

TASK:
For EACH entity, find the BEST keyframe in Image 1 where it appears most clearly.
Provide a detailed location description of where it is in that specific keyframe.
If the entity does not appear in ANY keyframe in the scene, mark it as not found (NONE).

Return ONLY valid JSON:
{{
  "entities": [
    {{
      "instruction_id": "instr_001",
      "entity_id": "entity_01",
      "found": true,
      "best_keyframe_index": "keyframe_0001",
      "location_description": "detailed location description in that keyframe"
    }}
  ]
}}
"""

GENERATE_SCENE_ENTITY_REFERENCE_PROMPT = """You are an image generation model.
Your task is to create a "Scene Entity Reference Image" that combines the appearances of specific entities from the provided scene keyframes.

SCENE KEYFRAMES:
(Provided as the reference image)

ENTITY LOCATIONS:
{entity_locations_block}

TASK:
Generate a single image grid. For each entity listed above that was 'found', extract/crop its appearance from the specified keyframe and location, maintaining its original state exactly as it appears in the keyframes.
For entities marked as 'not found' or NONE, output a blank placeholder with the text 'NONE'.
Arrange these crops into a single reference image.
"""

EDIT_SCENE_ENTITY_REFERENCE_PROMPT = """You are an image editing model.
Your task is to edit the "Scene Entity Reference Image" according to the provided instructions.

Image 1 = SOURCE SCENE ENTITY REFERENCE IMAGE (contains the original appearances of entities in this scene).
Images 2+ = ENTITY EDIT REFERENCE cards (left panel = original front-view entity appearance, right panel = edited front-view entity appearance after the instruction). Use the RIGHT panel as the visual target for the edit.

EDIT INSTRUCTIONS (one per reference card — left=before, right=after; ONLY these apply):
{canonical_edit_block}

TASK:
Edit Image 1 so that each entity's appearance matches the target edited appearance described in the EDIT INSTRUCTIONS and shown in the RIGHT panel of the corresponding reference card.
Do NOT change the layout, background, or any entities not listed in the instructions.
Maintain the same image dimensions and structure as Image 1.

{avoid_section}
"""

SCENE_ENTITY_REFERENCE_EDIT_QA_PROMPT = """You are validating whether the Scene Entity Reference Image was edited correctly.

Image 1 = EDITED scene entity reference image (candidate result).
Image 2 = SOURCE scene entity reference image (before editing).
Images 3+ = entity canonical reference cards (LEFT panel = original front-view appearance, RIGHT panel = intended edited front-view appearance).

PLANNED EDIT INSTRUCTIONS:
{canonical_edit_block}

TASK:
Determine whether:
1. FRAME STRUCTURE PRESERVED: same canvas size, aspect ratio, and layout of entities.
2. Every planned edit instruction was completed correctly on the corresponding entity crop.
3. The edited appearance matches the RIGHT panel of the canonical reference card where applicable.
4. NO unrelated entities or background regions were incorrectly edited.

Return ONLY valid JSON:
{{
  "passed": false,
  "score": 0.0,
  "frame_structure_preserved": false,
  "edit_completed": false,
  "canonical_reference_alignment_ok": false,
  "unrelated_edit_changes_absent": false,
  "failed_aspects": [],
  "feedback": "short summary of pass/fail reasons",
  "retry_focus_prompt": "if failed: specific operations to AVOID on retry; empty if passed",
  "positive_prompt": "if failed: what was done correctly and should be KEPT on retry; empty if passed"
}}

Rules:
- passed=true only when all planned edits are done correctly AND no significant unrelated changes exist AND frame_structure_preserved is true AND score >= 0.7.
- score: 0.0-1.0 overall quality.
- retry_focus_prompt must list mistakes to avoid, not new edit goals.
- positive_prompt (REQUIRED when failed; empty when passed): list the editing operations that were done CORRECTLY in this attempt and should be KEPT/MAINTAINED on the next retry. Phrase as positive instructions.
- Use English for all string values.
"""

# Front-view entity_refs mode. These assignments intentionally override the
# historical "multiview" prompt constants while keeping the constant names for
# compatibility with existing code paths.
ENTITY_MULTIVIEW_SYNTHESIS_PROMPT = """You are generating ONE front-view entity reference image from video keyframes.

Image 1 = keyframe grid containing up to 6 entity-related keyframes for the target entity.

TARGET ENTITY:
- instruction_id: {instruction_id}
- entity_id: {entity_id}
- subject_features: {subject_features}
- appearance_time_hint: {appearance_time_hint}

KEYFRAME NOTES:
{keyframe_notes}

TASK:
Create a single SQUARE photorealistic FRONT-VIEW or near-front reference image of the target entity.
Use all relevant evidence from the up-to-6 keyframes in Image 1 and the notes to show the entity's characteristic features clearly.
Do NOT create a 2x2 grid, back view, left profile, right profile, contact sheet, collage, or multiple panels.
Preserve the entity's original pre-edit appearance: identity, face, hair style/color, clothing, build, and distinctive accessories.
Use a simple neutral background if needed, center the entity, and keep enough head/upper-body/body context to display the entity features.
The output canvas MUST be square.

{avoid_section}
"""

ENTITY_MULTIVIEW_EDIT_PROMPT = """You are editing ONE front-view entity reference image.

Image 1 = original front-view entity reference.

TARGET ENTITY:
- instruction_id: {instruction_id}
- entity_id: {entity_id}
- subject_features: {subject_features}

EDIT INSTRUCTION:
{edit_prompt}

TASK:
Return one SQUARE front-view edited reference image of the same entity after applying the edit instruction.
Do NOT create a 2x2 grid, side/back views, contact sheet, collage, or multiple panels.
Preserve identity, pose, framing, and all attributes not mentioned in the edit instruction.
The output canvas MUST remain square.

{avoid_section}
"""

ENTITY_MULTIVIEW_SYNTHESIS_QA_PROMPT = """You are validating a generated front-view entity reference image.

Image 1 = input keyframe grid.
Image 2 = generated front-view entity reference candidate.

TARGET ENTITY:
- instruction_id: {instruction_id}
- entity_id: {entity_id}
- subject_features: {subject_features}

KEYFRAME NOTES:
{keyframe_notes}

Return ONLY valid JSON:
{{
  "passed": false,
  "score": 0.0,
  "front_view_orientation_correct": false,
  "entity_identity_matches_reference": false,
  "source_appearance_matches_reference": false,
  "photorealistic": false,
  "panel_structure_preserved": false,
  "neutral_background_ok": false,
  "failed_aspects": [],
  "feedback": "short explanation of pass or fail",
  "retry_focus_prompt": "if failed: list editing mistakes/errors to AVOID on the next retry; empty string if passed",
  "positive_prompt": "if failed: what was done correctly and should be KEPT on retry; empty if passed"
}}

Rules:
- passed=true only when this is a SINGLE SQUARE front-view or near-front entity reference, not a 2x2 grid or multi-panel sheet.
- source_appearance_matches_reference=true only when the candidate preserves the original pre-edit identity and visible appearance from the keyframes.
- panel_structure_preserved=false if the candidate is a collage/contact sheet/2x2 multi-view layout.
- Use English for all string values.
"""

ENTITY_MULTIVIEW_SOURCE_APPEARANCE_QA_PROMPT = """You are checking whether a front-view entity reference preserves the source appearance.

Image 1 = input keyframe grid.
Image 2 = generated front-view entity reference candidate.

TARGET ENTITY:
- entity_id: {entity_id}
- subject_features: {subject_features}

KEYFRAME NOTES:
{keyframe_notes}

Return ONLY valid JSON:
{{
  "alignment_ok": false,
  "mismatched_attributes": [],
  "feedback": "short explanation",
  "retry_focus_prompt": "if failed: source-appearance mistakes to AVOID on retry; empty if passed",
  "positive_prompt": "if failed: correct source-appearance details to KEEP on retry; empty if passed"
}}
"""

ENTITY_MULTIVIEW_EDIT_QA_PROMPT = """You are validating an edited front-view entity reference image.

Image 1 = original front-view entity reference.
Image 2 = edited front-view entity reference candidate.
Image 3 = input keyframe grid context.

TARGET ENTITY:
- instruction_id: {instruction_id}
- entity_id: {entity_id}
- subject_features: {subject_features}

EDIT INSTRUCTION:
{edit_prompt}

Return ONLY valid JSON:
{{
  "passed": false,
  "score": 0.0,
  "front_view_orientation_correct": false,
  "entity_identity_matches_reference": false,
  "source_appearance_matches_reference": false,
  "photorealistic": false,
  "panel_structure_preserved": false,
  "neutral_background_ok": false,
  "edit_completed": false,
  "edit_attributes_match_instruction": false,
  "failed_aspects": [],
  "feedback": "short summary",
  "retry_focus_prompt": "if failed: negative guidance for retry; empty if passed",
  "positive_prompt": "if failed: what was done correctly and should be KEPT on retry; empty if passed"
}}

Rules:
- passed=true only when Image 2 remains a SINGLE SQUARE front-view reference and applies the edit correctly.
- Do NOT accept 2x2 grids, side/back views, contact sheets, or multiple panels.
- The edited reference must preserve identity and all non-edited attributes from Image 1.
- Use English for all string values.
"""

ENTITY_MULTIVIEW_EDIT_ATTRIBUTE_QA_PROMPT = """You are checking edited attributes on a front-view entity reference.

Image 1 = original front-view reference.
Image 2 = edited front-view reference.

EDIT INSTRUCTION:
{edit_prompt}

Return ONLY valid JSON:
{{
  "alignment_ok": false,
  "mismatched_attributes": [],
  "feedback": "short explanation",
  "retry_focus_prompt": "if failed: attribute mistakes to AVOID on retry; empty if passed",
  "positive_prompt": "if failed: correctly edited attributes to KEEP on retry; empty if passed"
}}
"""

ENTITY_MULTIVIEW_EDIT_VIEW_OCCLUSION_QA_PROMPT = """You are checking that an edited front-view entity reference is not a multi-view sheet.

Image 1 = edited front-view reference.

EDIT INSTRUCTION:
{edit_prompt}

Return ONLY valid JSON:
{{
  "alignment_ok": true,
  "mismatched_attributes": [],
  "feedback": "short explanation",
  "retry_focus_prompt": "if failed: layout/view mistakes to AVOID on retry; empty if passed",
  "positive_prompt": "if failed: correct front-view details to KEEP on retry; empty if passed"
}}
"""

ENTITY_MULTIVIEW_CANDIDATE_SELECT_PROMPT = """You are choosing the best front-view entity reference candidate.

Task type: {task_type}
Entity: {entity_id} / {instruction_id}
Subject features: {subject_features}
{edit_section}

{keyframe_notes_section}

Choose the candidate that is a single front-view or near-front image, preserves identity, and is not a 2x2 grid/contact sheet.
For edit tasks, also prefer the candidate that best satisfies: {edit_prompt}

Return ONLY valid JSON:
{{
  "best_candidate_index": 0,
  "confidence": 0.0,
  "reasoning": "brief explanation"
}}
"""

ENTITY_REFERENCE_KEYFRAME_SELECT_PROMPT = """You are selecting the best keyframe for a front-view entity reference image.

ENTITY:
- entity_id: {entity_id}
- instruction_id: {instruction_id}
- subject_features: {subject_features}
- appearance_time_hint: {appearance_time_hint}
- video_duration_sec: {video_duration_sec}

AVAILABLE APPEARANCES:
{appearances_catalog}

Select up to {select_count} appearance_index values for the input_keyframe_grid.png, prioritizing:
1. keyframes strongly associated with the target entity,
2. clear front-facing or near-front view of the same entity,
3. visible face/head and distinctive identity cues,
4. high quality, sharpness, and sufficient size,
5. appearance closest to the requested time hint.

Prefer 6 distinct relevant keyframes when available. Avoid side/back views unless they are needed to supplement fewer than 6 front/near-front sightings.

Return ONLY valid JSON:
{{
  "selected_indices": [0],
  "reasoning": "brief explanation"
}}
"""
