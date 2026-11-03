# EPIC main-method parameters

The release implements one method: FLUX.2-klein-9B generation/editing, one parsed visual program per prompt, programmatic verification, SigLIP-only color scoring, late action verification, rule-based refinement, and conditional in-loop auditing. All supported predicates gate acceptance; a successful in-loop auditor can override persistent failures for eligible predicates. Uncertain predicates remain blocking until satisfied or covered by that audited override. When no accepted candidate is found, the best verified partial image is saved with `final_candidate_accepted=false`.

## Public algorithm controls

These five options are shared by the direct CLI and all six benchmark runners. Definitions and validation live in [cli.py](src/graph/cli.py); the Python API accepts the corresponding arguments in [graph_main.py](src/graph/graph_main.py).

| CLI option | Default | Meaning |
| --- | --- | --- |
| `--seed` | `42` | Base seed for image sampling, rewritten-prompt selection, and VLM requests. Benchmark runners derive sample seeds from this base. |
| `--max_round` | `32` | Maximum total rounds: the initial generation is round 1, and each edit or resample consumes another round. Python API name: `max_rounds`. |
| `--llm_temperature` | `0.0` | Parser, conditional reviewer, rewriter, and in-loop auditor sampling temperature. The text/action HTTP verifiers use fixed temperature 0. |
| `--rewritten_prompt_count` | `8` | Requested number of detailed rewrites; empty, duplicate, and unchanged rewrites are removed. |
| `--persistent_stuck_in_loop_auditor_streak` | `2` | Minimum consecutive blocked rounds before in-loop auditing. Every remaining blocker must be eligible and meet this streak; this is not a periodic audit interval. |

Round budget, rewrite count, and auditor streak must be positive; seed must be non-negative; temperature must be finite and non-negative.

Eligible auditor targets are color/material/shape/pattern/action/pose/state attributes, contrastive size attributes, `in`/`inside`/`on` relations, and global-scene constraints. Count, exclusion, text, and other spatial failures cannot be bypassed by this auditor. Resampling resets the current trajectory's stuck-predicate history. The same blocked-predicate set is not repeatedly audited without a relevant state change.

## Fixed model and sampling settings

These are implementation defaults, not extra CLI switches. See [state.py](src/graph/state.py), [image_runtime.py](src/graph/image_runtime.py), and [models_dm.py](src/graph/models_dm.py).

| Setting | Value |
| --- | --- |
| Image generator and editor | `black-forest-labs/FLUX.2-klein-9B`, one shared BF16 pipeline |
| Diffusion steps | `4` for both generation and editing |
| Guidance scale | `1.0` for both generation and editing |
| Initial image size | `1024 × 1024`; edits retain the input image size |
| Image RNG | Generated seed from `random.Random(base_seed + offset).randint(0, 2**31 - 1)`; generation uses a CPU generator and editing uses a CUDA generator |
| Per-round seed offset | Initial image: `0`; edits/resamples: `round_index - 1` |
| VLM | `Qwen/Qwen3-VL-32B-Instruct`; served separately through the OpenAI-compatible API |
| VLM image maximum side | `1024` pixels |
| Text/action verification | Temperature `0`, request timeout `180` seconds, response token caps `96` / `128` |
| Region/text scorer | `google/siglip2-base-patch16-224` |
| Object proposals | `facebook/sam3`, confidence cutoff `0.30` |
| SAM3 proposal deduplication | Intersection-over-smaller-area cutoff `0.85`, using masks when available and boxes otherwise |
| Depth | MiDaS `DPT_Large`; normalized depth values `0–255`, smaller means nearer |

The no-op small-mask area filter has been removed: main-method counting keeps all SAM3 proposals that survive confidence and deduplication checks. Color verification requires a SigLIP score; missing scores produce a no-signal score of 0 (violated), without substituting RGB heuristics. SAM3's BF16 autocast is scoped to SAM3 inference and cannot affect later models.

## Verifier thresholds and action policy

[VerifierThresholds](src/graph/visual_program_verifier.py) contains the fixed score cutoffs actually used by the main method. Scores at or above the satisfied cutoff are satisfied; the intermediate range is uncertain, and lower scores are violated, subject to each predicate evaluator's evidence checks.

| Predicate/evidence | Satisfied | Uncertain |
| --- | ---: | ---: |
| Object proposal confidence | `0.65` | `0.35` |
| Object attributes | `0.60` | `0.35` |
| Action attributes | `0.55` | `0.35` |
| Spatial relations | `0.60` | `0.35` |

`scene_background_min_area_ratio=0.10` rejects scene-grounding regions that occupy less than 10% of the image and gates residual-background scoring. It is separate from object counting. Exclusion predicates reverse proposal confidence: below 0.35 supports absence, 0.35–0.65 is uncertain, and at least 0.65 indicates a violation. The action cutoffs apply to early SigLIP evidence; a late VLM result maps `true` / `false` / `null` directly to satisfied / violated / uncertain (scores 1 / 0 / 0.5).

Action verification runs after non-action predicates have been checked. The VLM action check is deferred while other predicates are blocking. Directed actions use a joint subject/target crop with `target_joint_crop_padding_ratio=0.10` on each side of the union box, clipped to the image ([ActionScoringConfig](src/graph/visual_program_verifier.py)).

The [selector](src/graph/visual_program_action_selector.py) uses the fixed rule-based policy. Attribute and global-scene repairs can edit first; relation failures resample. `edit_retry_score_delta=0.12` allows another edit when the target mean score improves by at least 0.12; an increased satisfied count or predicate-state rank can independently permit retry. Best-partial selection prioritizes satisfied predicate families in this order: count, spatial relation, text, exclusion, attributes, global scene; it then considers satisfied count, unsatisfied-predicate score, and earlier round. There are no random-selector, edit-only, or resample-only modes.

## Spatial scoring coefficients

All 28 fields below remain active in [RelationScoringConfig](src/graph/visual_program_verifier.py). They define geometric scoring and are fixed in the released main method; there is no separate command-line override for each coefficient. Image geometry is normalized before these terms are combined, except the depth scale, which uses the normalized MiDaS output above.

| Group | Field | Value |
| --- | --- | ---: |
| Directional relations | `directional_centroid_weight` | `0.45` |
| | `directional_gap_weight` | `0.35` |
| | `directional_orthogonal_weight` | `0.20` |
| | `directional_gap_scale` | `0.12` |
| | `directional_gap_tolerance_ratio` | `0.02` |
| | `directional_centroid_scale` | `0.50` |
| | `directional_orthogonal_scale` | `0.55` |
| | `directional_orthogonal_overlap_weight` | `0.75` |
| | `directional_orthogonal_center_weight` | `0.25` |
| | `directional_axis_overlap_penalty_weight` | `0.05` |
| | `directional_overlap_penalty_weight` | `0.45` |
| | `directional_containment_penalty_weight` | `0.30` |
| | `directional_on_penalty_weight` | `0.35` |
| Near | `near_centroid_weight` | `0.45` |
| | `near_edge_weight` | `0.55` |
| | `near_diagonal_scale` | `0.18` |
| Inside | `inside_containment_weight` | `0.80` |
| | `inside_centroid_weight` | `0.20` |
| In | `in_bbox_coverage_weight` | `0.80` |
| | `in_centroid_bbox_weight` | `0.20` |
| Overlap | `overlap_iou_weight` | `0.65` |
| | `overlap_intersection_weight` | `0.35` |
| On | `on_contact_weight` | `0.55` |
| | `on_horizontal_overlap_weight` | `0.25` |
| | `on_above_weight` | `0.20` |
| | `on_inside_penalty_weight` | `0.25` |
| | `on_gap_tolerance_ratio` | `0.02` |
| In front of / behind | `depth_scale` | `28.0` |

Other fixed geometric terms are local to their evaluators: scene scoring blends full-image/residual-background scores with weights `0.45/0.55`, dilates foreground masks by `max(4 pixels, round(0.01 × min(image width, image height)))`, and support-contact evidence blends contact closeness/column coverage with weights `0.70/0.30`. Depth-order scoring combines the scaled depth delta with a `0.80 + 0.20 × support` factor; support blends planar alignment/near evidence with weights `0.60/0.40`. These are score definitions, not additional run-time switches.

Exclusion verification distinguishes a forbidden detection from a duplicate of a permitted object using all three overlap cutoffs: `duplicate_intersection_threshold=0.90`, `duplicate_area_similarity_threshold=0.70`, and `duplicate_iou_threshold=0.75`. An unexplained residual area ratio at or below `residual_ignore_threshold=0.10` is ignored. These four settings are defined on `ExclusionPredicateEvaluator` in the verifier module.

## Runtime and benchmark controls

These affect execution and data handling, rather than the refinement policy:

- `--base_url`: VLM endpoint, default `http://127.0.0.1:8000` or `BASE_URL`.
- `MODEL`: API model identifier, default `openai/Qwen/Qwen3-VL-32B-Instruct`; must match the server's `--served-model-name` after removing `openai/`.
- `OPENAI_API_KEY`: optional server authentication. Local unauthenticated servers use `EMPTY`.
- `--llm_max_retries`: retries after the initial structured LLM request, default `20` or `EPIC_LLM_MAX_RETRIES`. Explicit CLI values take precedence. Structured requests time out after `180` seconds; parser retries wait `3` seconds, and timeout backoff is capped at `60` seconds. No hidden warmup request is sent.
- `--disable_saving_files`: omit detailed per-round debug artifacts; final images and result/LLM/timing traces are still written.
- `--n_samples` / `--images_per_prompt`, indices, shards, paths, resume checks, and runner retry limits: see each runner's `--help`. These determine which independent samples are generated; they do not change the per-image round budget.
- `HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `TORCH_HOME`, and SAM3 asset-path overrides: model cache/storage controls.

Constructing `Context()` directly reads typed `SEED` and `LLM_TEMPERATURE` defaults, plus `BASE_URL` and `MODEL`. The public `async_generate_with_prompt` API always supplies its own seed/temperature/endpoint defaults, even when those arguments are omitted; pass explicit arguments to change them. The CLI reads `BASE_URL` and `EPIC_LLM_MAX_RETRIES`, while seed and temperature come from CLI options/defaults. `MODEL` remains environment-driven in both interfaces.
