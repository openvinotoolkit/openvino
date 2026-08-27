# OpenVINO Transformations — Review Feedback Analysis (`v-Golubev`)

**Scope:** pull requests in [`openvinotoolkit/openvino`](https://github.com/openvinotoolkit/openvino) that were reviewed by **v-Golubev**, that carry (or ever carried) a `*transformations*` label, and that were **created between 2025-08-27 and 2026-08-27**.

**Method**

1. `gh search prs --repo openvinotoolkit/openvino --reviewed-by v-Golubev --created ">=2025-08-27"` → **248** PRs (all states).
2. For each PR, the full label *history* was fetched via GraphQL (`labels` + `timelineItems(LABELED_EVENT, UNLABELED_EVENT)`), so PRs whose `transformations` label was later removed are also included. → **61** matching PRs.
   *Note:* the repository has no bare `transformations` label. The two matching labels are **`category: transformations`** (47 PRs in the reviewed set) and **`category: LP transformations`** (20 PRs). Both were included.
3. For each matching PR, all **review summaries**, **review threads (incl. resolved and outdated)**, **line comments**, and **issue comments** authored by `v-Golubev` were downloaded (`reviewThreads(first:100){comments}` — no thread pagination overflow was observed).
   → **516** comments by `v-Golubev`.
4. Comments on PRs **authored by** `v-Golubev` (15 PRs) were excluded — those are author replies, not review feedback. Administrative comments (`LGTM`, `build_jenkins`, `Done, thanks`, review-request pings, empty suggestions) were also excluded.
   → **400 technical review comments across 43 PRs** form the analysis corpus.

---

# 1. Analyzed PR Inventory

* **Total PRs matching the criteria:** **61**
* **PRs with technical review feedback from `v-Golubev`:** **43**
* **PRs authored by `v-Golubev`** (kept in the inventory, excluded from feedback statistics): **15**
* **PRs with review activity but no technical comments** (approval-only / CI-only): **3**
* **Date range (PR creation):** **2025-09-11 → 2026-08-18**
* **Status split:** Merged **51**, Open **7**, Closed-without-merge **3**

| PR | Title | State | Created | Author | VG cmts | Technical | transformations label(s) |
|---|---|---|---|---|---|---|---|
| [#32050](https://github.com/openvinotoolkit/openvino/pull/32050) | [TRANSFORMATIONS] Add Sinks and Scale inputs handling for PA gpt-oss | Merged | 2025-09-11 | CuriousPanCake | 1 | 1 | transformations |
| [#32266](https://github.com/openvinotoolkit/openvino/pull/32266) | [LPT] QDQ stripping | Merged | 2025-10-01 | v-Golubev | 14 | 0 | LP transformations; transformations |
| [#32323](https://github.com/openvinotoolkit/openvino/pull/32323) | [LPT] Fix accessing out-of-range dimension in MultiplyToGroupConvolutionTransformation | Merged | 2025-10-07 | Lyamin-Roman | 0 | 0 | LP transformations |
| [#32450](https://github.com/openvinotoolkit/openvino/pull/32450) | [CPU] Introduce GatherMatmul operation to optimize MoE pattern | Merged | 2025-10-16 | maxnick | 3 | 2 | transformations |
| [#32983](https://github.com/openvinotoolkit/openvino/pull/32983) | [OV CPU/GPU] modify pass ConvertWeightCompressedConv1x1ToMatmul for both CPU/GPU | Merged | 2025-11-21 | bopeng1234 | 37 | 37 | transformations |
| [#33072](https://github.com/openvinotoolkit/openvino/pull/33072) | [CPU][ARM] Non i32 conv bias support | Merged | 2025-11-28 | alvoron | 54 | 52 | LP transformations |
| [#33159](https://github.com/openvinotoolkit/openvino/pull/33159) | Disable fuseElementWise for FQ with mult-power | Merged | 2025-12-08 | nazanin-beheshti | 12 | 11 | LP transformations |
| [#33160](https://github.com/openvinotoolkit/openvino/pull/33160) | resolve args datatype mismatch for split LPT | Merged | 2025-12-08 | nazanin-beheshti | 3 | 2 | LP transformations |
| [#33162](https://github.com/openvinotoolkit/openvino/pull/33162) | ReduceSum LPT issue with different precision | Merged | 2025-12-08 | nazanin-beheshti | 6 | 5 | LP transformations |
| [#33269](https://github.com/openvinotoolkit/openvino/pull/33269) | fix overflow issues for unstripped uint16 FQs | Merged | 2025-12-16 | nazanin-beheshti | 15 | 13 | LP transformations; transformations |
| [#33581](https://github.com/openvinotoolkit/openvino/pull/33581) | [LPT][DOCS] LPT tests guide introduced | Merged | 2026-01-13 | v-Golubev | 1 | 0 | LP transformations |
| [#33911](https://github.com/openvinotoolkit/openvino/pull/33911) | Fix string literal conditions used in OPENVINO_ASSERT | Merged | 2026-01-30 | aobolensk | 3 | 0 | LP transformations; transformations |
| [#33964](https://github.com/openvinotoolkit/openvino/pull/33964) | [GPU] applying activations scaling if decompressed_to_f32 is present | Merged | 2026-02-04 | e-ddykim | 1 | 1 | transformations |
| [#33989](https://github.com/openvinotoolkit/openvino/pull/33989) | [LPT] FQStripping transformation rework | Merged | 2026-02-05 | v-Golubev | 12 | 0 | LP transformations; transformations |
| [#33990](https://github.com/openvinotoolkit/openvino/pull/33990) | Fix SliceToStridedSlice when axes are empty | Merged | 2026-02-06 | shiyi9801 | 1 | 1 | transformations |
| [#34177](https://github.com/openvinotoolkit/openvino/pull/34177) | [Transformations] Apply SDPA scale after MatMul(Q, K^T) per specification | Merged | 2026-02-18 | evkotov | 3 | 3 | transformations |
| [#34186](https://github.com/openvinotoolkit/openvino/pull/34186) | MoE adoption fixes | Merged | 2026-02-18 | v-Golubev | 3 | 0 | transformations |
| [#34963](https://github.com/openvinotoolkit/openvino/pull/34963) | [Core][CPU] Rename batch_gather_matmul to gather_matmul, move ops to common transformations | Merged | 2026-03-26 | EgorDuplensky | 27 | 24 | transformations |
| [#34965](https://github.com/openvinotoolkit/openvino/pull/34965) | [Transformations] QDQStripping: added horizontal DQ fusion pass | Merged | 2026-03-26 | v-Golubev | 2 | 0 | LP transformations; transformations |
| [#35088](https://github.com/openvinotoolkit/openvino/pull/35088) | [LPT] StridedSliceTransformation fix | Merged | 2026-03-31 | v-Golubev | 1 | 0 | LP transformations |
| [#35215](https://github.com/openvinotoolkit/openvino/pull/35215) | [Op] GroupedMatMul init | Merged | 2026-04-08 | mitruska | 13 | 13 | transformations |
| [#35218](https://github.com/openvinotoolkit/openvino/pull/35218) | [ITT] Standardize all OV_ITT_DOMAIN display names with ov:: prefix | Merged | 2026-04-08 | eparshut | 2 | 1 | LP transformations; transformations |
| [#35280](https://github.com/openvinotoolkit/openvino/pull/35280) | [CPU][ARM] Support Convolution int8 mixed precision | Merged | 2026-04-13 | alvoron | 24 | 23 | LP transformations |
| [#35283](https://github.com/openvinotoolkit/openvino/pull/35283) | [GPU] Enable dynamic quantization for MXFP8 & regular FP8 dtypes | Merged | 2026-04-13 | tkrupa-intel | 18 | 17 | LP transformations; transformations |
| [#35311](https://github.com/openvinotoolkit/openvino/pull/35311) | [GPU] Use gather matmul in gpu pipeline | Merged | 2026-04-13 | EgorDuplensky | 27 | 26 | transformations |
| [#35426](https://github.com/openvinotoolkit/openvino/pull/35426) | [Transformations] Transformations utils: added extract_subgraph helper | Merged | 2026-04-20 | v-Golubev | 13 | 0 | transformations |
| [#35518](https://github.com/openvinotoolkit/openvino/pull/35518) | [GPU] applying activations scaling to MOECompressed | Merged | 2026-04-24 | e-ddykim | 9 | 8 | transformations |
| [#35618](https://github.com/openvinotoolkit/openvino/pull/35618) | [Transformations] MoE 3 Gemm pattern: Gelu activation support | Merged | 2026-04-30 | v-Golubev | 5 | 0 | transformations |
| [#35691](https://github.com/openvinotoolkit/openvino/pull/35691) | [GPU] MoE 3 GeMM: separate Router Subgraph and MoE body kernels | Merged | 2026-05-06 | v-Golubev | 2 | 0 | transformations |
| [#35744](https://github.com/openvinotoolkit/openvino/pull/35744) | [GPU] Support MoE pattern from gemma4 in GPU composite operation | Merged | 2026-05-08 | v-Golubev | 9 | 0 | transformations |
| [#35769](https://github.com/openvinotoolkit/openvino/pull/35769) | [Snippets][Transformations][CPU] Fix GCC 15 shared_ptr warnings | Closed | 2026-05-11 | aobolensk | 1 | 0 | transformations |
| [#35902](https://github.com/openvinotoolkit/openvino/pull/35902) | [LPT] Replace stripped FakeQuantize with Clamp to preserve activation range | Open | 2026-05-15 | liubo-intel | 2 | 2 | LP transformations |
| [#35941](https://github.com/openvinotoolkit/openvino/pull/35941) | Allow ConvertQuantizeDequantize to handle mixed fp32/fp16 float precision | Merged | 2026-05-15 | mdvoretc-intel | 6 | 6 | LP transformations; transformations |
| [#35945](https://github.com/openvinotoolkit/openvino/pull/35945) | [TRANSFORMATIONS] Add NormalizeDequantizeFP16 pass to enable QDQ fusion for FP16 models | Open | 2026-05-16 | rayngun | 8 | 6 | transformations |
| [#35969](https://github.com/openvinotoolkit/openvino/pull/35969) | [LPT] Remove mandatory weight folding in ConvolutionTransformation | Merged | 2026-05-18 | mdvoretc-intel | 7 | 4 | LP transformations |
| [#36042](https://github.com/openvinotoolkit/openvino/pull/36042) | [CPU][ARM] Implement FuseClampAndFakeQuantize transformation | Merged | 2026-05-22 | alvoron | 12 | 11 | transformations |
| [#36058](https://github.com/openvinotoolkit/openvino/pull/36058) | [Transformations] Shared low_precision_dequantize decomposition helper | Merged | 2026-05-25 | mvafin | 4 | 4 | transformations |
| [#36075](https://github.com/openvinotoolkit/openvino/pull/36075) | [Transformations] EliminateSequentialFakeQuantize | Merged | 2026-05-26 | mryzhov | 14 | 14 | transformations |
| [#36170](https://github.com/openvinotoolkit/openvino/pull/36170) | [GPU][Core] Add UINT2 compressed weights support with transpose and ZP fixes | Merged | 2026-06-01 | Prithviraj-R | 7 | 7 | transformations |
| [#36192](https://github.com/openvinotoolkit/openvino/pull/36192) | [Transformations] Added a new `TransposeFQ` matcher to common transpose sinking | Merged | 2026-06-02 | alvoron | 7 | 7 | transformations |
| [#36265](https://github.com/openvinotoolkit/openvino/pull/36265) | [GPU] Fix compilation failure for uncompressed (FP16) MoE models | Merged | 2026-06-04 | andrew-k-park | 4 | 3 | transformations |
| [#36454](https://github.com/openvinotoolkit/openvino/pull/36454) | [Transformations] Fix `TransposeFQ` to only fire when FQ consumer is a Transpose | Merged | 2026-06-17 | alvoron | 5 | 4 | transformations |
| [#36542](https://github.com/openvinotoolkit/openvino/pull/36542) | [Transformations] ov-transformation-tests skill and tests writing documentation | Merged | 2026-06-23 | v-Golubev | 5 | 0 | transformations |
| [#36543](https://github.com/openvinotoolkit/openvino/pull/36543) | [Core/Transformations] Internal-op input-validation sweep + matcher cleanup | Open | 2026-06-23 | pjordanandrsn | 12 | 10 | transformations |
| [#36581](https://github.com/openvinotoolkit/openvino/pull/36581) | [CPU] Enable Convert Gather fusion for hybrid precisions | Merged | 2026-06-26 | xuchen-intel | 5 | 4 | transformations |
| [#36595](https://github.com/openvinotoolkit/openvino/pull/36595) | [GHA] Use lychee-action for links verification | Merged | 2026-06-26 | v-Golubev | 2 | 0 | transformations |
| [#36616](https://github.com/openvinotoolkit/openvino/pull/36616) | [LPT] Remove CMake GLOB from openvino_lpt library | Open | 2026-06-29 | v-Golubev | 2 | 0 | LP transformations |
| [#36666](https://github.com/openvinotoolkit/openvino/pull/36666) | [GPU] Grouped matmul support for GPU | Merged | 2026-07-01 | isanghao | 6 | 6 | transformations |
| [#36711](https://github.com/openvinotoolkit/openvino/pull/36711) | [CPU][GPU] MoveFCReshapeToWeights unification across CPU and GPU | Merged | 2026-07-03 | v-Golubev | 3 | 0 | transformations |
| [#36719](https://github.com/openvinotoolkit/openvino/pull/36719) | [TRANSFORMATIONS][CPU][SNIPPETS] Keep LTX-Video RoPE angle computation in f32 under bf16 | Merged | 2026-07-04 | goyaladitya05 | 17 | 15 | transformations |
| [#36737](https://github.com/openvinotoolkit/openvino/pull/36737) | [GPU] Enable sdpa_micro for bidirectional attention and sliding window | Merged | 2026-07-06 | hyunback | 7 | 7 | transformations |
| [#36756](https://github.com/openvinotoolkit/openvino/pull/36756) | [Transformations] Add BroadcastMatMulFusion common optimization | Merged | 2026-07-07 | mryzhov | 12 | 11 | transformations |
| [#36865](https://github.com/openvinotoolkit/openvino/pull/36865) | [GPU] Enable RMS fusion to match RMS without gamma | Merged | 2026-07-14 | hyunback | 7 | 7 | transformations |
| [#36870](https://github.com/openvinotoolkit/openvino/pull/36870) | [GroupedMatmulCompressed] Keep constant precision of u4 weight | Merged | 2026-07-14 | isanghao | 5 | 5 | transformations |
| [#36879](https://github.com/openvinotoolkit/openvino/pull/36879) | [GPU] Enable post-op fusion for eltwise with FC/MatMul/Transpose when reshape in middle | Open | 2026-07-14 | clee30 | 4 | 4 | transformations |
| [#36944](https://github.com/openvinotoolkit/openvino/pull/36944) | [GPU] Enable dynamic quantization for MXFP4 & regular FP4 dtypes | Merged | 2026-07-17 | merezman | 1 | 1 | LP transformations; transformations |
| [#37021](https://github.com/openvinotoolkit/openvino/pull/37021) | [NPU] Make gather matmul compressed and MOE compressed support weights as param | Closed | 2026-07-23 | liyihao-1ntel | 16 | 16 | transformations |
| [#37133](https://github.com/openvinotoolkit/openvino/pull/37133) | [TRANSFORMATIONS] Fix NCHW/NHWC layout mismatch in ConvertWeightCompressedConv1x1ToMatmul | Closed | 2026-07-29 | mloktukh | 3 | 2 | transformations |
| [#37181](https://github.com/openvinotoolkit/openvino/pull/37181) | [TRANSFORMATIONS] Fix NCHW/NHWC layout mismatch in ConvertWeightCompressedConv1x1ToMatmul | Merged | 2026-07-31 | v-Golubev | 7 | 0 | transformations |
| [#37411](https://github.com/openvinotoolkit/openvino/pull/37411) | [LPT] Skip PadTransformation for non-finite CONSTANT pad values | Open | 2026-08-13 | pj3iL | 1 | 1 | LP transformations |
| [#37490](https://github.com/openvinotoolkit/openvino/pull/37490) | [Transformations] Fix per-channel axis mapping in MoveEltwiseUpThroughDataMov | Open | 2026-08-18 | wilson-seok | 3 | 3 | transformations |

**Review-volume concentration:** five PRs account for ~44 % of all technical feedback — [#33072](https://github.com/openvinotoolkit/openvino/pull/33072) (52), [#32983](https://github.com/openvinotoolkit/openvino/pull/32983) (37), [#35311](https://github.com/openvinotoolkit/openvino/pull/35311) (26), [#34963](https://github.com/openvinotoolkit/openvino/pull/34963) (24), [#35280](https://github.com/openvinotoolkit/openvino/pull/35280) (23). All five are large cross-component (CPU/GPU/common) transformation PRs.

---

# 2. Review Feedback Statistics

Base: **400 technical review comments** over **43 PRs**. Each comment is assigned to exactly one primary category (a strict priority order was applied; many comments legitimately touch two themes).

| # | Category | Comments | % of total | PRs |
|---|---|---|---|---|
| 1 | Test coverage & test design | 102 | 25.5 % | 34 |
| 2 | Minor code simplification / local style | 75 | 18.8 % | 29 |
| 3 | Pattern-matching quality | 59 | 14.8 % | 21 |
| 4 | Graph-rewrite correctness & semantics | 55 | 13.8 % | 22 |
| 5 | Validation, invariants & fail-fast | 25 | 6.2 % | 13 |
| 6 | Documentation & explanatory comments | 21 | 5.2 % | 13 |
| 7 | Pipeline placement & pass design | 16 | 4.0 % | 11 |
| 8 | Code reuse & duplication | 16 | 4.0 % | 12 |
| 9 | Op & pass API design | 12 | 3.0 % | 8 |
| 10 | Scope control & change hygiene | 10 | 2.5 % | 8 |
| 11 | Naming & readability | 7 | 1.8 % | 5 |
| 12 | Performance & memory | 2 | 0.5 % | 2 |

> If comments are allowed to carry **multiple** labels, the picture shifts: *Pattern-matching quality* reaches ~60 comments / 21 PRs, *Pipeline placement & pass design* ~65 / 31, *Graph-rewrite correctness* ~73 / 27, *Code reuse* ~38 / 22, *Op & pass API design* ~44 / 19, *Performance & memory* ~16 / 9. The multi-label view is the more faithful measure of **breadth** (how many PRs a theme touches); the single-label view is the more faithful measure of **volume**.

### 2.1 Test coverage & test design — 102 comments / 34 PRs (25.5 %)

**Description.** Requests to add a regression test for the exact reported bug, to use the canonical `TransformationTestsF` + reference-model comparison instead of ad-hoc assertions, to parametrize instead of copy-pasting instances, to reuse shared model builders, and to cover negative/corner cases.

Representative examples:

* “Could you please also add a test case in `.../reduce_sum_transformation.cpp` which reproduces the original issue?” — [#33162](https://github.com/openvinotoolkit/openvino/pull/33162#pullrequestreview-3602498189)
* “all the transformation tests must use `TransformationTestsF` base class and use functions comparator to verify the model correctness after transformation (by comparing it with the reference)” — [#35280](https://github.com/openvinotoolkit/openvino/pull/35280#discussion_r3161420603)
* “If the transformation shouldn’t be applied, we don’t need to create `model_ref`: it will be automatically cloned from `model`. This [is] actual for all negative test cases” — [#33072](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2613488601)
* “This change make the tests completely useless: refs are not calculated at all” — [#35311](https://github.com/openvinotoolkit/openvino/pull/35311#discussion_r3091909130)
* “Please align the tests with the OV transformations tests standards: `src/common/transformations/docs/writing_tests.md` … you can use the `ov-transformation-tests` skill” — [#36719](https://github.com/openvinotoolkit/openvino/pull/36719#discussion_r3536715504), [#36870](https://github.com/openvinotoolkit/openvino/pull/36870#discussion_r3639199713), [#36879](https://github.com/openvinotoolkit/openvino/pull/36879#discussion_r3674067408)

Source PRs: [#32983](https://github.com/openvinotoolkit/openvino/pull/32983), [#33072](https://github.com/openvinotoolkit/openvino/pull/33072), [#33159](https://github.com/openvinotoolkit/openvino/pull/33159), [#33160](https://github.com/openvinotoolkit/openvino/pull/33160), [#33162](https://github.com/openvinotoolkit/openvino/pull/33162), [#33964](https://github.com/openvinotoolkit/openvino/pull/33964), [#33990](https://github.com/openvinotoolkit/openvino/pull/33990), [#34963](https://github.com/openvinotoolkit/openvino/pull/34963), [#35280](https://github.com/openvinotoolkit/openvino/pull/35280), [#35311](https://github.com/openvinotoolkit/openvino/pull/35311), [#36042](https://github.com/openvinotoolkit/openvino/pull/36042), [#36058](https://github.com/openvinotoolkit/openvino/pull/36058), [#36075](https://github.com/openvinotoolkit/openvino/pull/36075), [#36170](https://github.com/openvinotoolkit/openvino/pull/36170), [#36192](https://github.com/openvinotoolkit/openvino/pull/36192), [#36454](https://github.com/openvinotoolkit/openvino/pull/36454), [#36543](https://github.com/openvinotoolkit/openvino/pull/36543), [#36581](https://github.com/openvinotoolkit/openvino/pull/36581), [#36666](https://github.com/openvinotoolkit/openvino/pull/36666), [#36719](https://github.com/openvinotoolkit/openvino/pull/36719), [#36737](https://github.com/openvinotoolkit/openvino/pull/36737), [#36756](https://github.com/openvinotoolkit/openvino/pull/36756), [#36870](https://github.com/openvinotoolkit/openvino/pull/36870), [#36879](https://github.com/openvinotoolkit/openvino/pull/36879), [#37021](https://github.com/openvinotoolkit/openvino/pull/37021), [#37133](https://github.com/openvinotoolkit/openvino/pull/37133), [#37490](https://github.com/openvinotoolkit/openvino/pull/37490), and others.

### 2.2 Minor code simplification / local style — 75 comments / 29 PRs (18.8 %)

**Description.** Short, mostly `suggestion`-block edits: shortening expressions, removing dead locals, `""` vs `<>` includes, structured bindings, `-1` instead of `Dimension::dynamic()`, removing stale empty lines, copyright year. These are low-severity but consume review bandwidth; several are symptoms of the deeper categories below.

Examples: [#32983 r2669614824](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2669614824), [#32983 r2735893721](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2735893721), [#33072 r2601587457](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2601587457), [#33072 r2638463744](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2638463744).

### 2.3 Pattern-matching quality — 59 comments / 21 PRs (14.8 %)

**Description.** By far the most *characteristic* theme. Two recurring asks:
(a) **move constraints out of the callback into the pattern**; (b) **use the declarative predicate vocabulary** (`shape_matches`, `rank_equals`, `type_matches[_any]`, `attrs_match`, `value_matches`, `has_static_shape`, `wrap_const`, `optional<>`, `operator|`) instead of hand-rolled lambdas and manual `if`s.

Representative examples:

* “Please move this check to the matcher: you can use `ov::pass::pattern::type_matches` or `type_matches_any` predicates for that” — [#33072 r2601523538](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2601523538)
* “Shape checks can be done without custom predicate logic using `ov::pass::pattern::shape_matches` … Can we try to reuse it here to simplify the code?” — [#32983 r2556924250](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2556924250)
* “Can we perform this `GroupedMatMul` type routing at matcher level? This would allow to avoid 2d2d case handling (=return false) and would make the transformation callback code shorter and cleaner.” — [#35215 r3414766660](https://github.com/openvinotoolkit/openvino/pull/35215#discussion_r3414766660)
* “Instead of manual graph traversal, let’s include weights precision checks in the pattern matcher via `type_matches_any` predicate.” — [#36265 r3401658531](https://github.com/openvinotoolkit/openvino/pull/36265#discussion_r3401658531)
* “We can use `optional` here to avoid one more manual ‘Or’.” — [#35280 r3190612533](https://github.com/openvinotoolkit/openvino/pull/35280#discussion_r3190612533)
* “this can be written shorter (check `operator|` in `Or`): `auto weights_convert_m = weights_5d_convert_m | weights_4d_convert_m;`” — [#32983 r2669534476](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2669534476)
* “Since `ov::compare_constants` works only with constants, let’s reflect it in the matcher: use `pattern::wrap_const()`.” — [#36075 r3460296628](https://github.com/openvinotoolkit/openvino/pull/36075#discussion_r3460296628)
* “Consider using `ov::pass::pattern::attrs_match` instead of a custom lambda.” — [#36756 r3713824716](https://github.com/openvinotoolkit/openvino/pull/36756#discussion_r3713824716), [#37490 r3861280827](https://github.com/openvinotoolkit/openvino/pull/37490#discussion_r3861280827)

### 2.4 Graph-rewrite correctness & semantics — 55 comments / 22 PRs (13.8 %)

**Description.** Whether the rewrite is *semantically* valid: broadcasting/shape semantics, precision and overflow, spec conformance, metadata preservation (`copy_runtime_info`, friendly names), correct node replacement helpers, `TypeRelaxed` handling, constant folding.

Representative examples:

* “Using `get_type_name() == "Power"` as a signal is brittle and not semantically tied to whether the transform is safe. The real criterion should be shape semantics… Per the FQ specification, the output shape is always equal to the data input shape.” — [#33159 r2667823897](https://github.com/openvinotoolkit/openvino/pull/33159#discussion_r2667823897)
* “`TypeRelaxed` is used in the wrong way (`_input_data_types`/`_output_data_types` not set) … just call `dequantization.multiply->clone_with_new_inputs({...})`.” — [#33160 r2673190213](https://github.com/openvinotoolkit/openvino/pull/33160#discussion_r2673190213)
* “we should compute new shift in `dequantization.multiplyConstant` precision, because `subtractConstant` may be in low precision (…overflow).” — [#33162 r2686883213](https://github.com/openvinotoolkit/openvino/pull/33162#discussion_r2686883213)
* “Let’s use `ov::replace_output_update_name` helper instead.” — [#33072 r2601530295](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2601530295), [#36042 r3288893713](https://github.com/openvinotoolkit/openvino/pull/36042#discussion_r3288893713), [#36756 r3713863925](https://github.com/openvinotoolkit/openvino/pull/36756#discussion_r3713863925)
* “`copy_runtime_info(weights_reshape, {weights_new_constant, weights_reshape_new});`” — [#32983 r2704344082](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2704344082)
* “This dead consumer will be removed during the next model validation, we never take care about such things manually.” — [#34963 r3000257650](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r3000257650)
* “Let’s check it in a general way using ops from the OV opset (it allows to check non-scalar ranges and automatically handle broadcasting).” — [#36454 r3453598774](https://github.com/openvinotoolkit/openvino/pull/36454#discussion_r3453598774)

### 2.5 Validation, invariants & fail-fast — 25 comments / 13 PRs (6.2 %)

**Description.** Two mirror-image rules: **remove** defensive code that the matcher already guarantees, and **add** hard assertions where an invariant must hold instead of silently bailing out.

Representative examples:

* “These casts are not needed: if the matcher matched the nodes, they are not nullptr.” — [#33072 r2735803886](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2735803886)
* “`swish_m` describes a node with `Swish` type, so the cast must be successful here: I’d recommend replacing `if` with an assert… this approach allows us to avoid static analyzer errors (Coverity) and helps to track casting/matching problems at early stages.” — [#34963 r3000174491](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r3000174491), [r3009363156](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r3009363156)
* “I prefer always using `.at()` for the nodes which must be matched by the matcher.” — [#32266 r2491324024](https://github.com/openvinotoolkit/openvino/pull/32266#discussion_r2491324024)
* “This node must have constant type (we define it as constant in the matcher), so let’s throw an exception if `const_node` is `nullptr` instead of silently creating a zero sliding window.” — [#36737 r3542552399](https://github.com/openvinotoolkit/openvino/pull/36737#discussion_r3542552399)
* “I think it is better to prohibit the creation of such an operation. Otherwise, the user could create 3d3d grouped matmul with offsets and wrongly think that offsets affect something.” — [#35215 r3452305892](https://github.com/openvinotoolkit/openvino/pull/35215#discussion_r3452305892)
* “This ‘if’ is not needed: `validate_inputs_count` already validated that the op always has 6 inputs.” — [#35311 r3091733303](https://github.com/openvinotoolkit/openvino/pull/35311#discussion_r3091733303)

### 2.6 Documentation & explanatory comments — 21 comments / 13 PRs (5.2 %)

* “Let’s describe here what this transformation does (moves DQ on outputs to weights) and why (to avoid f16 overflow).” — [#33072 r2789188154](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2789188154)
* “Could you please mention in the description why this transformation is needed?” — [#35945 r3333386516](https://github.com/openvinotoolkit/openvino/pull/35945#discussion_r3333386516)
* “Since we use `pattern::shape_matches` predicates, we don’t need these comments anymore: the code is self descriptive enough.” — [#32983 r2704321705](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2704321705)
* “It looks like the comment contradicts the expression: if a callback returns **true**, the corresponding **transformation doesn’t run**.” — [#33072 r2638461238](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2638461238)
* “The justification is wrong: shapes are actually known at compilation time… Please correct or remove the comment.” — [#37021 r3737163464](https://github.com/openvinotoolkit/openvino/pull/37021#discussion_r3737163464)

### 2.7 Pipeline placement & pass design — 16 comments / 11 PRs (4.0 %)

* “Please keep `InitNodeInfo` first in the pipeline.” — [#32983 r2557026080](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2557026080)
* “I’d prefer leaving `Validate` as the last pass in this pipeline: it costs not so much, but it allows to check that we don’t break precision or shape inference.” — [#32266 r2491310877](https://github.com/openvinotoolkit/openvino/pull/32266#discussion_r2491310877)
* “this transformation is a prerequisite for `ConvertQuantizeDequantize`. If so, let’s register it in all pipelines where `ConvertQuantizeDequantize` is registered. I see at least 3 more places.” — [#35945 r3333394821](https://github.com/openvinotoolkit/openvino/pull/35945#discussion_r3333394821)
* “I believe this pass may be beneficial for all plugins: let’s move it to common optimizations and register in the same places as `ReluFakeQuantizeFusion`.” — [#36042 r3288925505](https://github.com/openvinotoolkit/openvino/pull/36042#discussion_r3288925505)
* “CPU doesn’t always need to decompose SDPA layers… this fusion currently happens before `CommonOptimizations`, so this change will break LLM scenarios.” — [#34177 r2826716723](https://github.com/openvinotoolkit/openvino/pull/34177#discussion_r2826716723)
* “we should remove transpose/reshape optimizations from this transformation at all… several simple passes with a single responsibility.” — [#37181 r3722382961](https://github.com/openvinotoolkit/openvino/pull/37181#discussion_r3722382961)

### 2.8 Code reuse & duplication — 16 comments / 12 PRs (4.0 %)

* “It looks like this pass duplicates the existing `SharedOpOptimization` pass logic… Maybe something is missed there and we can fix the existing pass instead of introducing a similar one.” — [#35518 r3161441359](https://github.com/openvinotoolkit/openvino/pull/35518#discussion_r3161441359)
* “this is a horizontal fusion optimization … we have a general `ov::pass::SharedOpOptimization` that should be used for such purposes… avoid this pass complication.” — [#35283 r3465877494](https://github.com/openvinotoolkit/openvino/pull/35283#discussion_r3465877494)
* “I suppose we can use `ov::op::util::visit_path` for this traversal instead of a custom implementation.” — [#36719 r3536607065](https://github.com/openvinotoolkit/openvino/pull/36719#discussion_r3536607065)
* “we can use `ov::op::util::visit_path_forward` … to cover the general multi-consumer case and avoid code duplication.” — [#36075 r3460326848](https://github.com/openvinotoolkit/openvino/pull/36075#discussion_r3460326848)
* “this duplication doesn’t look good to me: this not so trivial code should be placed in one place and reused where needed.” — [#34963 r2997128624](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r2997128624)
* “Let’s extract it in a common place as a helper and reuse here.” — [#36454 r3453598774](https://github.com/openvinotoolkit/openvino/pull/36454#discussion_r3453598774)

### 2.9 Op & pass API design — 12 comments / 8 PRs (3.0 %)

* “Can we remove this constructor? I suppose we should always configure config when `MOECompressed` is created.” / “External modification of the existing config may be dangerous. Can we remove the setter?” — [#34963 r2997054108](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r2997054108), [r2997060808](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r2997060808)
* “Instead of a custom method introduction, I propose overriding `visit_attributes` … this would allow to use `SharedOpOptimization` … In addition, it is needed for correct model caching: `visit_attributes` is used during `ov::Model` serialization/deserialization.” — [#35283 r3465920997](https://github.com/openvinotoolkit/openvino/pull/35283#discussion_r3465920997), [r3468455516](https://github.com/openvinotoolkit/openvino/pull/35283#discussion_r3468455516)
* “`element::dynamic` became a standard for the optional inputs in CPU and GPU plugin. Can the NPU plugin align its behavior in this aspect?” — [#37021 r3737225088](https://github.com/openvinotoolkit/openvino/pull/37021#discussion_r3737225088)
* “Let’s regulate the scale and decompression precision without hardcoded values … I suggest introducing new `decompression_precision` and `scale_precision` params.” — [#35283 r3452639217](https://github.com/openvinotoolkit/openvino/pull/35283#discussion_r3452639217)
* “Let’s avoid default value for this parameter: we anyway use it only at one place where it is required.” — [#35280 r3190604036](https://github.com/openvinotoolkit/openvino/pull/35280#discussion_r3190604036)
* “Semantically, `has_batch_dim` should have boolean type.” — [#35311 r3091642677](https://github.com/openvinotoolkit/openvino/pull/35311#discussion_r3091642677)

### 2.10 Scope control & change hygiene — 10 comments / 8 PRs (2.5 %)

* “Let’s avoid this change in this PR: I believe we should test I8 MaxPool support … within a separate activity.” — [#33072 r2606259912](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2606259912)
* “This file is not just moved+renamed, I see some unexpected code changes … Could you please double check this file to avoid the unnecessary changes?” — [#34963 r2997090104](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r2997090104)
* “yes, let’s better apply `state_management_pattern.cpp` corrections within a separate PR.” — [#36543 #issuecomment-4902452899](https://github.com/openvinotoolkit/openvino/pull/36543#issuecomment-4902452899)
* “Why is this test removed?” — [#34963 r2997150804](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r2997150804)

### 2.11 Naming & readability — 7 comments / 5 PRs (1.8 %)

* “The current name doesn’t look descriptive: please try to reflect in the name that we fall back unsupported low-precision convolutions to fp16 execution.” — [#33072 r2793984258](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2793984258)
* “`gptoss_gemma3_mask` name is confusing now: it reflects gemma4 case as well. So let’s rename … or just to `mask`.” — [#36737 r3542532081](https://github.com/openvinotoolkit/openvino/pull/36737#discussion_r3542532081)
* “Let’s align the pass name with the logic: `DisableBF16CompForLtxVideoRopePattern`.” — [#36719 r3656395295](https://github.com/openvinotoolkit/openvino/pull/36719#discussion_r3656395295)
* “Can we use `hidden_in` and `seq_len` notation instead of `?`.” — [#32983 r2669523451](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2669523451)

### 2.12 Performance & memory — 2 comments / 2 PRs (0.5 % primary; ~16 / 9 PRs multi-label)

* “The main concern I have is that we may potentially introduce performance degradations for the quantized models … if earlier we had just one `Convolution(Quantized)`, now we have `Transpose->MatMul(Quantized)->Transpose` … it looks like we should avoid the transformation if the model is quantized.” — [#32983 r2557041267](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2557041267)
* “If `H*W == 1`, transpose shouldn’t be inserted: this is crucial to keep the performance of the already working models.” — [#37133 r3690205497](https://github.com/openvinotoolkit/openvino/pull/37133#discussion_r3690205497)
* “after conversion to matmul, transpose between bias and matmul will break plugin fusings, so it is better to insert transpose after the bias.” — [#32983 r2669464837](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2669464837)
* “a general pass may end up skipping more layers/subgraphs than actually necessary, which could potentially lead to performance degradation.” — [#36719 #issuecomment-4957140930](https://github.com/openvinotoolkit/openvino/pull/36719#issuecomment-4957140930)

---

# 3. Recurring Review Guidelines

Twenty guidelines were distilled. Each states the rule, the reasoning behind it, how often it recurred, and the evidence.

---

### G1 — Express every matching constraint in the pattern, not in the callback

**Explanation.** Any condition that decides *whether the pass applies* (op type, rank, shape, element type, attribute value, constant-ness, static-ness) must be encoded as a pattern node or predicate. The callback should only *build* the replacement.

**Rationale.** Constraints in the pattern are (a) self-documenting, (b) reusable/composable, (c) checked by the matcher engine so the callback needs no defensive code, and (d) make “why didn’t my pass fire?” debuggable via matcher logs instead of stepping through `return false` branches.

**Frequency.** ~40 comments across **17 PRs** — the single most repeated *structural* request.

**Representative comments**
* “Please move this check to the matcher: you can use `type_matches` or `type_matches_any` predicates.” — [#33072 r2601523538](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2601523538)
* “Let’s check this in the matcher: use `pattern::value_matches` predicate for that.” — [#35311 r3179971210](https://github.com/openvinotoolkit/openvino/pull/35311#discussion_r3179971210)
* “Let’s reflect it in the matcher instead of checking in the callback: `wrap_type<v1::Reshape, v0::Squeeze, v0::Unsqueeze>(...)`.” — [#36879 r3674023965](https://github.com/openvinotoolkit/openvino/pull/36879#discussion_r3674023965)
* “Let’s move these checks to pattern predicates: `rank_equals(N)` suits perfectly here.” — [#36666 r3528510946](https://github.com/openvinotoolkit/openvino/pull/36666#discussion_r3528510946)
* “Shouldn’t we require static shape in `CompressedWeightsBlock` then? `pass::pattern::has_static_shape` predicate can be used.” — [#37021 r3730560625](https://github.com/openvinotoolkit/openvino/pull/37021#discussion_r3730560625)
* “If the shape is not scalar, should this transformation even be called for such a pattern? Maybe it is better to check the shape in the `sw_const` predicate?” — [#36737 r3542565902](https://github.com/openvinotoolkit/openvino/pull/36737#discussion_r3542565902)

**Source PRs.** [#32983](https://github.com/openvinotoolkit/openvino/pull/32983), [#33072](https://github.com/openvinotoolkit/openvino/pull/33072), [#33269](https://github.com/openvinotoolkit/openvino/pull/33269), [#35215](https://github.com/openvinotoolkit/openvino/pull/35215), [#35280](https://github.com/openvinotoolkit/openvino/pull/35280), [#35311](https://github.com/openvinotoolkit/openvino/pull/35311), [#36042](https://github.com/openvinotoolkit/openvino/pull/36042), [#36075](https://github.com/openvinotoolkit/openvino/pull/36075), [#36192](https://github.com/openvinotoolkit/openvino/pull/36192), [#36265](https://github.com/openvinotoolkit/openvino/pull/36265), [#36666](https://github.com/openvinotoolkit/openvino/pull/36666), [#36719](https://github.com/openvinotoolkit/openvino/pull/36719), [#36737](https://github.com/openvinotoolkit/openvino/pull/36737), [#36756](https://github.com/openvinotoolkit/openvino/pull/36756), [#36865](https://github.com/openvinotoolkit/openvino/pull/36865), [#36879](https://github.com/openvinotoolkit/openvino/pull/36879), [#37021](https://github.com/openvinotoolkit/openvino/pull/37021), [#37490](https://github.com/openvinotoolkit/openvino/pull/37490)

**Bad**
```cpp
auto conv_m = pattern::wrap_type<v1::Convolution>();
matcher_pass_callback callback = [=](Matcher& m) {
    auto conv = ov::as_type_ptr<v1::Convolution>(m.get_match_root());
    if (!conv) return false;
    if (conv->get_strides() != Strides{1, 1}) return false;
    const auto& w = conv->get_input_partial_shape(1);
    if (w.rank().get_length() != 4 || w[2] != 3 || w[3] != 3) return false;
    if (conv->get_input_element_type(1) != element::i8) return false;
    ...
};
```

**Good**
```cpp
using namespace ov::pass::pattern;
auto weights_m = any_input(type_matches(element::i8) && shape_matches("OC, IC, 3, 3"));
auto conv_m    = wrap_type<v1::Convolution>({any_input(), weights_m}, {{"strides", Strides{1, 1}}});
matcher_pass_callback callback = [=](Matcher& m) {
    const auto& pm  = m.get_pattern_value_map();
    const auto& sym = m.get_symbols();          // "OC"/"IC" are available here
    ...
};
```

**Confidence.** **Strong** (17 independent PRs).

---

### G2 — Prefer the declarative predicate vocabulary over hand-written lambdas

**Explanation.** Use `shape_matches` (with symbols), `rank_equals`, `type_matches` / `type_matches_any`, `attrs_match`, `value_matches`, `has_static_shape`, `wrap_const()`, `optional<Op>()`, and `operator|` on pattern nodes. Reach for a custom `pattern::Predicate` lambda only when nothing in the vocabulary fits.

**Rationale.** The built-in predicates already handle dynamic dimensions, symbol propagation, broadcasting and precision edge cases correctly; custom lambdas repeatedly got these wrong. They also make patterns shorter and remove the need for explanatory comments.

**Frequency.** ~25 comments across **11 PRs**.

**Representative comments**
* “Shape checks can be done without custom predicate logic using `pattern::shape_matches` which uses the symbolics feature.” — [#32983 r2556924250](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2556924250)
* “let’s use `pattern::shape_matches("?, ?, ?, 1, 1") || pattern::shape_matches("?, ?, 1, 1")` predicate instead and remove a custom `filter1x1_path`.” — [#32983 r2735887731](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2735887731)
* “We can use `optional` here to avoid one more manual ‘Or’: `optional<v1::Subtract>()`.” — [#35280 r3190612533](https://github.com/openvinotoolkit/openvino/pull/35280#discussion_r3190612533)
* “this can be written shorter … `weights_convert_m = weights_5d_convert_m | weights_4d_convert_m;`” — [#32983 r2669534476](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2669534476)
* “Ideally, `pattern::attrs_match` predicate helper should be reused (something like `attrs_match({{"mode", "numpy"}})`).” — [#37490 r3861280827](https://github.com/openvinotoolkit/openvino/pull/37490#discussion_r3861280827)
* “a big portion of this logic can be replaced with `pattern::shape_matches` predicate. Could you please try it? This should simplify the code.” — [#36756 r3713886726](https://github.com/openvinotoolkit/openvino/pull/36756#discussion_r3713886726)
* “Since we use `pattern::shape_matches` predicates, we don’t need these comments anymore: the code is self descriptive enough.” — [#32983 r2704321705](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2704321705)

**Source PRs.** [#32983](https://github.com/openvinotoolkit/openvino/pull/32983), [#33072](https://github.com/openvinotoolkit/openvino/pull/33072), [#35280](https://github.com/openvinotoolkit/openvino/pull/35280), [#35311](https://github.com/openvinotoolkit/openvino/pull/35311), [#36042](https://github.com/openvinotoolkit/openvino/pull/36042), [#36075](https://github.com/openvinotoolkit/openvino/pull/36075), [#36666](https://github.com/openvinotoolkit/openvino/pull/36666), [#36719](https://github.com/openvinotoolkit/openvino/pull/36719), [#36756](https://github.com/openvinotoolkit/openvino/pull/36756), [#37021](https://github.com/openvinotoolkit/openvino/pull/37021), [#37490](https://github.com/openvinotoolkit/openvino/pull/37490)

**Confidence.** **Strong** (11 independent PRs).

---

### G3 — Trust the matcher: delete redundant casts and null checks; assert on violated invariants

**Explanation.** If a node was matched by `wrap_type<T>`, it *is* a `T` — do not re-check. Use `pattern_map.at(label)` (never `[]` or `count()`-guarded access) for mandatory pattern nodes. When an invariant that the matcher guarantees appears to be broken, `OPENVINO_ASSERT` / `OPENVINO_THROW` — never `return false` silently.

**Rationale.** Redundant checks hide real bugs behind “transformation silently did nothing”, which is the hardest class of transformation defect to diagnose. Asserts surface matcher/pipeline breakage immediately and also silence Coverity-style static-analysis findings.

**Frequency.** ~34 comments across **12 PRs**.

**Representative comments**
* “These casts are not needed: if the matcher matched the nodes, they are not nullptr.” — [#33072 r2735803886](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2735803886)
* “`as_type_ptr` is not necessary here: we don’t use `Convert`-class specific API.” — [#32983 r2669592589](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2669592589)
* “I prefer always using `.at()` for the nodes which must be matched by the matcher.” — [#32266 r2491324024](https://github.com/openvinotoolkit/openvino/pull/32266#discussion_r2491324024); “symbols must have ‘OC’ and ‘IC’ values, so let’s use `.at()`.” — [#33072 r2741310170](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2741310170)
* “`swish_m` describes a node with `Swish` type, so the cast must be successful here: I’d recommend replacing `if` with an assert.” — [#34963 r3000174491](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r3000174491)
* “This is a standard practice … this approach allows us to avoid static analyzer errors (such as Coverity) and helps to track casting/matching problems at early stages.” — [#34963 r3009363156](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r3009363156)
* “I would throw an exception here instead: we match only 2 gmm variations; if we successfully matched something that is not `gmm_3d_3d` and `gmm_2d_3d`, then something went wrong.” — [#35215 r3452243662](https://github.com/openvinotoolkit/openvino/pull/35215#discussion_r3452243662)
* “Shouldn’t we assert that `s.size() < 3` instead of a silent return?” — [#35311 r3179918990](https://github.com/openvinotoolkit/openvino/pull/35311#discussion_r3179918990)
* “Shouldn’t we throw an exception in case of unexpected shape instead of silently returning the original node?” — [#37021 r3730517184](https://github.com/openvinotoolkit/openvino/pull/37021#discussion_r3730517184)
* “I think that `fq_attr` mustn’t be empty at this stage, so let’s check it with an assertion.” — [#35280 r3161406390](https://github.com/openvinotoolkit/openvino/pull/35280#discussion_r3161406390)

**Source PRs.** [#32266](https://github.com/openvinotoolkit/openvino/pull/32266), [#32983](https://github.com/openvinotoolkit/openvino/pull/32983), [#33072](https://github.com/openvinotoolkit/openvino/pull/33072), [#33269](https://github.com/openvinotoolkit/openvino/pull/33269), [#34963](https://github.com/openvinotoolkit/openvino/pull/34963), [#35215](https://github.com/openvinotoolkit/openvino/pull/35215), [#35280](https://github.com/openvinotoolkit/openvino/pull/35280), [#35311](https://github.com/openvinotoolkit/openvino/pull/35311), [#36042](https://github.com/openvinotoolkit/openvino/pull/36042), [#36543](https://github.com/openvinotoolkit/openvino/pull/36543), [#36737](https://github.com/openvinotoolkit/openvino/pull/36737), [#37021](https://github.com/openvinotoolkit/openvino/pull/37021)

**Bad**
```cpp
auto swish = ov::as_type_ptr<v4::Swish>(pm[swish_m].get_node_shared_ptr());
if (!swish) return false;                      // cannot happen
auto it = pm.find(weights_m);
if (it == pm.end()) return false;              // mandatory pattern node
```
**Good**
```cpp
const auto& pm = m.get_pattern_value_map();
const auto swish = ov::as_type_ptr<v4::Swish>(pm.at(swish_m).get_node_shared_ptr());
OPENVINO_ASSERT(swish, "SwishFusion: matched node is expected to be v4::Swish");
const auto weights = pm.at(weights_m);
```
**Confidence.** **Strong** (12 independent PRs).

---

### G4 — Every behavioural change ships with a `TransformationTestsF` test that compares against a reference model

**Explanation.** Unit tests for a pass must build `model`, register the pass on the fixture’s `manager`, build `model_ref` as the expected post-transformation graph, and let the fixture’s `FunctionsComparator` do the checking. Manual node counting, manual `manager.run_passes()` calls, and hand-rolled assertions are not accepted. See `src/common/transformations/docs/writing_tests.md`.

**Rationale.** Reference comparison checks the *whole* topology (types, shapes, attributes, and optionally runtime info), so it catches collateral damage that targeted assertions miss; it also gives a uniform, greppable test style across the component.

**Frequency.** ~55 comments across **21 PRs** — the highest-volume single request.

**Representative comments**
* “all the transformation tests must use `TransformationTestsF` base class and use the functions comparator to verify the model correctness after transformation.” — [#35280 r3161420603](https://github.com/openvinotoolkit/openvino/pull/35280#discussion_r3161420603)
* “Let’s use `TransformationTestsF` tests: this is the most common way to test transformation-related code … Additionally, this would allow to avoid custom helpers creation.” — [#36058 r3310270387](https://github.com/openvinotoolkit/openvino/pull/36058#discussion_r3310270387)
* “I don’t see a reason why we should avoid a standard approach here. Let’s create a reference model and compare the graphs.” — [#35311 r3094024816](https://github.com/openvinotoolkit/openvino/pull/35311#discussion_r3094024816)
* “Ideally, we need to build `manager` … and build a `model_ref` which will be compared with the model after transformation. This would help to avoid manual checks and improve the code coverage (since we will check all the topology, not only the sliding-window related parts).” — [#36737 r3542662965](https://github.com/openvinotoolkit/openvino/pull/36737#discussion_r3542662965)
* “Please align the tests with the OV transformations tests standards: `writing_tests.md`… you can use the `ov-transformation-tests` skill.” — [#36719 r3536715504](https://github.com/openvinotoolkit/openvino/pull/36719#discussion_r3536715504)
* “Please align the new tests with existing ones: 1. No manual `manager.run_passes` calling 2. `model_ref` should be built as the model after transformation.” — [#36879 r3674067408](https://github.com/openvinotoolkit/openvino/pull/36879#discussion_r3674067408)
* “Shouldn’t we enable runtime info comparison for these tests? I am not sure that this is enabled by default.” — [#35280 r3217579965](https://github.com/openvinotoolkit/openvino/pull/35280#discussion_r3217579965)

**Source PRs.** [#32983](https://github.com/openvinotoolkit/openvino/pull/32983), [#33072](https://github.com/openvinotoolkit/openvino/pull/33072), [#34963](https://github.com/openvinotoolkit/openvino/pull/34963), [#35280](https://github.com/openvinotoolkit/openvino/pull/35280), [#35311](https://github.com/openvinotoolkit/openvino/pull/35311), [#36042](https://github.com/openvinotoolkit/openvino/pull/36042), [#36058](https://github.com/openvinotoolkit/openvino/pull/36058), [#36075](https://github.com/openvinotoolkit/openvino/pull/36075), [#36719](https://github.com/openvinotoolkit/openvino/pull/36719), [#36737](https://github.com/openvinotoolkit/openvino/pull/36737), [#36870](https://github.com/openvinotoolkit/openvino/pull/36870), [#36879](https://github.com/openvinotoolkit/openvino/pull/36879), and others.

**Confidence.** **Strong** (21 independent PRs).

---

### G5 — For negative cases, leave `model_ref` empty

**Explanation.** To assert “this pass must not change the graph”, build only `model`, register the pass, and do **not** create `model_ref`. `TransformationTestsF` clones the original model and verifies invariance automatically.

**Rationale.** Removes duplicated model-building code and guarantees the negative expectation is exactly “bit-identical graph”.

**Frequency.** 4 comments across **3 PRs**.

**Representative comments**
* “We are trying to duplicate `TransformationTestsF` logic here. Let’s better make a `TransformationTestsF` … 3. Leave `model_ref` empty. In this case, `TransformationTestsF` internal logic will check that the model hasn’t been changed by the target transformation: this is exactly what we need.” — [#32983 r2704267785](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2704267785)
* “If the transformation shouldn’t be applied, we don’t need to create `model_ref`: it will be automatically cloned from `model`. This [is] actual for all negative test cases.” — [#33072 r2613488601](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2613488601)
* “Let’s add a test case where `output_fq` has no `conv_input_precision` in the precisions attribute (e.g. int16 FQ): in this case, the transformation mustn’t change the graph.” — [#35280 r3190724466](https://github.com/openvinotoolkit/openvino/pull/35280#discussion_r3190724466)

**Confidence.** **Moderate** (3 independent PRs), but mechanically implied by G4.

---

### G6 — Parametrize tests and reuse the shared model builders

**Explanation.** Convert near-duplicate `TEST_F` bodies into `TEST_P` with a parameter struct; move boolean flags into the parameter tuple rather than duplicating instantiations; use existing builders (`ov::test::utils::make_fake_quantize`, `initMatMulDecompressionSubgraph`, `initGatherDecompressionSubgraph`, `ov::test::CheckNumberOfNodesWithType`) instead of local helpers.

**Rationale.** Parametrization multiplies coverage at near-zero cost and prevents the “fixed one instance, forgot the other five” failure mode; shared builders keep test graphs consistent with the graphs the rest of the suite exercises.

**Frequency.** ~20 comments across **10 PRs**.

**Representative comments**
* “Let’s move the last 2 boolean params from `Conv1x1ToMatmulActNotTranParams` and use `std::tuple<Params, bool, bool>` … This would allow to avoid unnecessary instance duplication and increase the coverage.” — [#32983 r2704315244](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2704315244)
* “Let’s reuse `ov::test::utils::make_fake_quantize` to avoid code duplication.” — [#33072 r2735765972](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2735765972), [#36075 r3460423751](https://github.com/openvinotoolkit/openvino/pull/36075#discussion_r3460423751)
* “You can try to reuse `ov::test::utils::initMatMulDecompressionSubgraph` to avoid code duplication.” — [#32983 r2556899149](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2556899149)
* “Consider using a common build helper for matmul decompression subgraph — `initMatMulDecompressionSubgraph` — it would allow to set up the needed configuration via `decompression_precision` and `scale_precision` params without any boolean flags introduction.” — [#35283 r3452615180](https://github.com/openvinotoolkit/openvino/pull/35283#discussion_r3452615180)
* “Let’s use `ov::test::CheckNumberOfNodesWithType` instead of custom implementation.” — [#35311 r3180006146](https://github.com/openvinotoolkit/openvino/pull/35311#discussion_r3180006146)
* “Can we cover `is_value_preserving` ops between fake quantizes in parametrized tests? We could use a test parameter like `intermediate_op_type`, this would allow to reduce test code.” — [#36075 r3460435941](https://github.com/openvinotoolkit/openvino/pull/36075#discussion_r3460435941)

**Source PRs.** [#32983](https://github.com/openvinotoolkit/openvino/pull/32983), [#33072](https://github.com/openvinotoolkit/openvino/pull/33072), [#35283](https://github.com/openvinotoolkit/openvino/pull/35283), [#35311](https://github.com/openvinotoolkit/openvino/pull/35311), [#36042](https://github.com/openvinotoolkit/openvino/pull/36042), [#36075](https://github.com/openvinotoolkit/openvino/pull/36075), [#36170](https://github.com/openvinotoolkit/openvino/pull/36170), [#36192](https://github.com/openvinotoolkit/openvino/pull/36192), [#36581](https://github.com/openvinotoolkit/openvino/pull/36581), [#36756](https://github.com/openvinotoolkit/openvino/pull/36756)

**Confidence.** **Strong** (10 independent PRs).

---

### G7 — A bug fix must include a regression test built from the reporting model’s subgraph

**Explanation.** For every fix, add a test case that reproduces the original failure, ideally derived from the failing model’s actual subgraph, in the pass’s existing test file.

**Rationale.** Without it there is no proof the fix addresses the reported defect and no protection against reintroduction.

**Frequency.** ~15 comments across **12 PRs**.

**Representative comments**
* “Could you please also add a test case in `.../reduce_sum_transformation.cpp` which reproduces the original issue?” — [#33162](https://github.com/openvinotoolkit/openvino/pull/33162#pullrequestreview-3602498189)
* “Is it possible to add a test case, reproducing the issue, to `.../variadic_split_transformation.cpp`?” — [#33160 r2667938238](https://github.com/openvinotoolkit/openvino/pull/33160#discussion_r2667938238)
* “Could you please add a test case to `.../optimize_strided_slice_test.cpp` which covers this case?” — [#33990](https://github.com/openvinotoolkit/openvino/pull/33990#pullrequestreview-3778229248)
* “Please update `activations_scaling_test.cpp` with the new test case, covering the fixed issue.” — [#33964 r2904450980](https://github.com/openvinotoolkit/openvino/pull/33964#discussion_r2904450980)
* “Let’s better build the graph from the problem ONNX model: `Transpose->FQ->Convert->Convert->Subtract->Multiply`.” — [#36454 r3453604513](https://github.com/openvinotoolkit/openvino/pull/36454#discussion_r3453604513)
* “Please create a new test case from the CVS-186600 model subgraph, where the clamp should be inserted instead of FQ.” — [#35902 r3281444388](https://github.com/openvinotoolkit/openvino/pull/35902#discussion_r3281444388)
* “Could you please confirm that the current unit tests cover this properly? I mean if we remove one of the comparison components (relative or absolute), the corresponding test case fails.” — [#35941 r3703060419](https://github.com/openvinotoolkit/openvino/pull/35941#discussion_r3703060419)

**Confidence.** **Strong** (12 independent PRs).

---

### G8 — Fix the general problem, not the observed topology

**Explanation.** A transformation must be gated by the *semantic* property that makes the rewrite valid, not by a proxy signal (op type name, a neighbouring node’s presence, a model-specific shape). Hard-coded, model-specific behaviour in a plugin pipeline is not an acceptable long-term solution.

**Rationale.** Proxy conditions are simultaneously too narrow (miss valid cases) and too broad (fire on invalid ones); model-specific workarounds become permanent and couple plugin behaviour to one model.

**Frequency.** ~14 comments across **9 PRs**.

**Representative comments**
* “Using `get_type_name() == "Power"` as a signal is brittle and not semantically tied to whether the transform is safe. The real criterion should be shape semantics… the transformation must not be executed when the eltwise output shape differs from the shape of its non-constant input.” — [#33159 r2667823897](https://github.com/openvinotoolkit/openvino/pull/33159#discussion_r2667823897)
* “The current solution is too topology dependent … I believe we should solve this problem in a general way.” — [#35969 r3280185386](https://github.com/openvinotoolkit/openvino/pull/35969#discussion_r3280185386)
* “I still have serious concerns about this workaround and the hard-coded logic it introduces … In practice, such workarounds tend to become permanent … using an explicit, model-specific plugin configuration would be a more maintainable and transparent temporary solution.” — [#35518 r3203932706](https://github.com/openvinotoolkit/openvino/pull/35518#discussion_r3203932706)
* “Let’s regulate the scale and decompression precision without hardcoded values.” — [#35283 r3452639217](https://github.com/openvinotoolkit/openvino/pull/35283#discussion_r3452639217)
* “Let’s avoid mentioning exact values (since we use this pass for several precisions).” — [#33269 r2699059592](https://github.com/openvinotoolkit/openvino/pull/33269#discussion_r2699059592)
* “Please use `std::numeric_limits<ov::float16>::max()` instead of a hardcoded value.” — [#33269 r2683512891](https://github.com/openvinotoolkit/openvino/pull/33269#discussion_r2683512891)
* “This transformation doesn’t look semantically right … When dequantize ops are fused into a quantize-like FQ, only output ranges must be affected.” — [#35945 r3281638941](https://github.com/openvinotoolkit/openvino/pull/35945#discussion_r3281638941)

**Confidence.** **Strong** (9 independent PRs).

---

### G9 — One pass, one responsibility; compose passes instead of growing them

**Explanation.** Do not accumulate special-case branches inside a pass. Split variants into separate `MatcherPass`es aggregated by a `GraphRewrite`; let existing generic passes (`TransposeToReshape`, `TransposeFusion`, `ReshapeFusion`, `SharedOpOptimization`, `Validate`) do their job instead of re-implementing their effects inline.

**Rationale.** Single-responsibility passes are individually testable, reusable across plugins, and their interaction is governed by explicit pipeline order rather than hidden coupling.

**Frequency.** ~12 comments across **8 PRs**.

**Representative comments**
* “I see a lot of 4D vs 5D code branches in this transformation, which makes it quite cumbersome. What if we create separate pattern-matchers for 4D and 5D cases, and register them in a single `GraphRewrite`?” — [#32983 r2669611035](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2669611035)
* “it looks like we should remove transpose/reshape optimizations from this transformation at all … This would allow us to have several simple passes with a single responsibility and avoid logic overcomplication.” — [#37181 r3722382961](https://github.com/openvinotoolkit/openvino/pull/37181#discussion_r3722382961)
* “Since callback and matcher are not reusable across 2 matchers, let’s extract ‘Matcher 2’ case to a separate `MatcherPass` in this file.” — [#36879 r3674039822](https://github.com/openvinotoolkit/openvino/pull/36879#discussion_r3674039822)
* “Looking at this description, I see a potential approach that could allow us to avoid introducing a new transformation … what if we apply the MoE optimizations after the activation-scaling pipeline?” — [#35518 r3162184053](https://github.com/openvinotoolkit/openvino/pull/35518#discussion_r3162184053)
* “Ideally, we should avoid performing graph traversal and transposes search + elimination during the translation to the OpenVINO opset. Transpose merging and elimination should be handled at the `ov::Model` transformations stage.” — [#35215 r3414863522](https://github.com/openvinotoolkit/openvino/pull/35215#discussion_r3414863522)
* “Now the matcher pass requires `BackwardGraphRewrite`, so I think it’s better to force it semantically: inherit `RMSFusion` from `BackwardGraphRewrite` … all the details are hidden.” — [#36865 r3656697405](https://github.com/openvinotoolkit/openvino/pull/36865#discussion_r3656697405)
* “Within this PR, a lot of low-precision related callbacks were added … maybe we could extract all these callbacks to a separate file? This would help prevent the file from growing to an unreasonably large size.” — [#33072 r2735849495](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2735849495)

**Confidence.** **Strong** (8 independent PRs).

---

### G10 — Reuse existing utilities and passes; never introduce a near-duplicate

**Explanation.** Before adding a helper or a pass, search for an existing one. Known reuse targets seen in reviews: `ov::pass::SharedOpOptimization` (horizontal fusion), `ov::op::util::visit_path` / `visit_path_forward` (graph traversal), `ov::replace_output_update_name`, `ov::op::util::get_constant_from_source`, `ov::op::util::is_on_constant_path`, `ov::compare_constants`, the FQ-range comparison helper in `qdq_stripping.cpp`, `ov::fundamental_type_for`, `is_type_any_of`.

**Rationale.** Duplicated non-trivial logic diverges; fixes then land in one copy only. If the existing utility is insufficient, extend it rather than fork it.

**Frequency.** ~38 comments across **12 PRs**.

**Representative comments**
* “It looks like this pass duplicates the existing `SharedOpOptimization` pass logic … Maybe something is missed there and we can fix the existing pass instead of introducing a similar one.” — [#35518 r3161441359](https://github.com/openvinotoolkit/openvino/pull/35518#discussion_r3161441359)
* “we have a general `ov::pass::SharedOpOptimization` that should be used for such purposes … and avoid this pass complication.” — [#35283 r3465877494](https://github.com/openvinotoolkit/openvino/pull/35283#discussion_r3465877494)
* “I suppose we can use `ov::op::util::visit_path` for this traversal instead of a custom implementation.” — [#36719 r3536607065](https://github.com/openvinotoolkit/openvino/pull/36719#discussion_r3536607065)
* “this not so trivial code should be placed in one place and reused where needed.” — [#34963 r2997128624](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r2997128624)
* “The only difference between `FuseMultiplyToFakeQuantizeTransformation` and `FuseSubtractToFakeQuantizeTransformation` is this node type. Let’s extract the common code to a parametrized helper.” — [#33072 r2735816858](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2735816858)
* “We already have such a check in the FQStripping pass … Let’s extract it in a common place as a helper and reuse here.” — [#36454 r3453598774](https://github.com/openvinotoolkit/openvino/pull/36454#discussion_r3453598774)
* “We deduct `use_micro_sdpa` in at least 2 places … Isn’t it better to introduce some helper?” — [#36737 r3542643320](https://github.com/openvinotoolkit/openvino/pull/36737#discussion_r3542643320)

**Confidence.** **Strong** (12 independent PRs).

---

### G11 — Preserve graph metadata and use the canonical replacement helpers

**Explanation.** Use `ov::copy_runtime_info(old_nodes, new_nodes)`, propagate friendly names, prefer `ov::replace_output_update_name` for 1:1 output replacement and `replace_node` with a freshly cloned node over a sequence of `replace_source_output` calls. Clone `TypeRelaxed` nodes with `clone_with_new_inputs` so mixed-precision configuration is inherited. Do not manually clean up dead consumers — `Validate` does it.

**Rationale.** Runtime info carries precision-markup, fusing hints and provenance used by downstream passes and plugins; losing it silently changes behaviour. The canonical helpers preserve tensor names, which the API and caching rely on.

**Frequency.** ~20 comments across **8 PRs**.

**Representative comments**
* “Let’s use `ov::replace_output_update_name` helper instead.” — [#33072 r2601530295](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2601530295); “Please use `replace_output_update_name` helper.” — [#36042 r3288893713](https://github.com/openvinotoolkit/openvino/pull/36042#discussion_r3288893713); “Can this be replaced with `replace_output_update_name`?” — [#36756 r3713863925](https://github.com/openvinotoolkit/openvino/pull/36756#discussion_r3713863925)
* “Please also set `add`’s friendly name for `new_add`.” — [#33072 r2613475344](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2613475344)
* “Instead of several `replace_source_output` calls, I would recommend creating a new convolution (using `clone_with_new_inputs`) and replacing multiply with the new conv using `replace_node`.” — [#33072 r2793948421](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2793948421)
* “`TypeRelaxed` is used in the wrong way … just call `dequantization.multiply->clone_with_new_inputs({parent, splitedMul[i]})`.” — [#33160 r2673190213](https://github.com/openvinotoolkit/openvino/pull/33160#discussion_r2673190213)
* “This dead consumer will be removed during the next model validation, we never take care about such things manually.” — [#34963 r3000257650](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r3000257650)
* “The original subgraph’s nodes should be already marked with `keep_const_precision` attribute by the `KeepConstPrecision` pass … remove this extra markup.” — [#36870 r3639191868](https://github.com/openvinotoolkit/openvino/pull/36870#discussion_r3639191868)
* “The root cause is that we don’t copy runtime info to a new Subgraph node during Snippets tokenization.” — [#36719 r3536691457](https://github.com/openvinotoolkit/openvino/pull/36719#discussion_r3536691457)

**Confidence.** **Strong** (8 independent PRs).

---

### G12 — Get precision right in low-precision code; never let a rewrite narrow silently

**Explanation.** When rebuilding dequantization arithmetic, compute in the precision that cannot overflow (typically the multiply-constant precision), use `foldConvert` / `ov::fundamental_type_for` instead of hardcoding, respect `element::f16` limits via `std::numeric_limits`, and validate that constant values fit the data type’s representable range.

**Rationale.** Silent narrowing in LPT produces accuracy loss or `inf`/`nan` that surfaces only on real models, long after the PR merges.

**Frequency.** ~12 comments across **7 PRs**.

**Representative comments**
* “we should compute new shift in `dequantization.multiplyConstant` precision, because `subtractConstant` may be in low precision (in which we can’t compute multiplication without accuracy loss because of overflow).” — [#33162 r2686883213](https://github.com/openvinotoolkit/openvino/pull/33162#discussion_r2686883213)
* “Please use `std::numeric_limits<ov::float16>::max()` instead of hardcoded value.” — [#33269 r2683512891](https://github.com/openvinotoolkit/openvino/pull/33269#discussion_r2683512891)
* “Shouldn’t we always mark the nodes despite the constant values? Theoretically, an overflow may occur during subgraph computation.” — [#33269 r2699071080](https://github.com/openvinotoolkit/openvino/pull/33269#discussion_r2699071080)
* “the check should be even stricter: we can’t perform the transformation if the pad value is out of range of the `dequantization.data`’s precision.” — [#37411 r3862191484](https://github.com/openvinotoolkit/openvino/pull/37411#discussion_r3862191484)
* “Please consider adding a `Round` op before `Convert` to improve accuracy.” — [#33072 r2601529101](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2601529101)
* “I applied it using `ov::fundamental_type_for` to avoid a manual precision specification.” — [#32266 r2491316465](https://github.com/openvinotoolkit/openvino/pull/32266#discussion_r2491316465)

**Confidence.** **Strong** (7 independent PRs).

---

### G13 — Prefer OV-opset computation over hand-written constant arithmetic

**Explanation.** To compare or transform constant values (FQ ranges, scales, zero points), build a small OV subgraph and fold it rather than iterating raw buffers.

**Rationale.** The opset implementation handles broadcasting, mixed precisions and corner cases for free; the cost is paid once at compile time.

**Frequency.** 4 comments across **3 PRs**.

**Representative comments**
* “Using ops from the OV opset allows to natively handle corner cases (such as broadcasting) or different precisions support … this outweighs the perf overheads on OV ops creation and folding (taking into account that all these computations happen at model compilation stage).” — [#35941 r3454130505](https://github.com/openvinotoolkit/openvino/pull/35941#discussion_r3454130505)
* “Let’s check it in a general way using ops from the OV opset (it allows to check non-scalar ranges and automatically handle broadcasting).” — [#36454 r3453598774](https://github.com/openvinotoolkit/openvino/pull/36454#discussion_r3453598774)
* “We can simplify this code: 1. Always create Reshape … 2. Then use `get_constant_from_source(reshape)` … 3. If the result is constant, use it as the new bias input.” — [#32983 r2704362761](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2704362761)

**Confidence.** **Moderate** (3 independent PRs).

---

### G14 — Register a pass in *every* pipeline that needs it, and respect pipeline invariants

**Explanation.** `InitNodeInfo` stays first; `Validate` stays last. A pass that is a prerequisite of another pass must be registered wherever that pass is registered (Common Optimizations, MoC, CPU pipeline, GPU pipeline, plugin FQ-stripping pipelines). Verify that a pass isn’t already registered elsewhere before adding a second call.

**Rationale.** Pipeline order *is* the transformation contract; a pass registered in one plugin only silently produces divergent graphs across devices.

**Frequency.** ~12 comments across **8 PRs**.

**Representative comments**
* “Please keep `InitNodeInfo` first in the pipeline.” — [#32983 r2557026080](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2557026080)
* “I’d prefer leaving `Validate` as the last pass in this pipeline: it costs not so much, but it allows to check that we don’t break precision or shape inference.” — [#32266 r2491310877](https://github.com/openvinotoolkit/openvino/pull/32266#discussion_r2491310877)
* “let’s register it in all pipelines where `ConvertQuantizeDequantize` is registered. I see at least 3 more places.” — [#35945 r3333394821](https://github.com/openvinotoolkit/openvino/pull/35945#discussion_r3333394821)
* “We already call this pass in the GPU pipeline before `ConvertPrecision` (L552). Could you please check if we need a separate call here?” — [#33269 r2699056432](https://github.com/openvinotoolkit/openvino/pull/33269#discussion_r2699056432)
* “I’d suggest registering the reshape elimination pass before the MoE passes are called: this would allow to avoid the pattern modification.” — [#35311 r3179964020](https://github.com/openvinotoolkit/openvino/pull/35311#discussion_r3179964020)
* “CPU doesn’t always need to decompose SDPA layers … this change will break LLM scenarios.” — [#34177 r2826716723](https://github.com/openvinotoolkit/openvino/pull/34177#discussion_r2826716723)

**Confidence.** **Strong** (8 independent PRs).

---

### G15 — If a pass is device-agnostic, put it in common transformations

**Explanation.** A pass whose benefit is not tied to a specific backend belongs in `src/common/transformations` and should be registered next to its peers, not duplicated per plugin. Conversely, device-specific behaviour is expressed as a *pass parameter*, not a fork.

**Rationale.** Prevents CPU/GPU divergence and duplicated maintenance; parameters keep one implementation testable.

**Frequency.** ~8 comments across **6 PRs**.

**Representative comments**
* “I believe this pass may be beneficial for all plugins: let’s move it to common optimizations and register in the same places as `ReluFakeQuantizeFusion`.” — [#36042 r3288925505](https://github.com/openvinotoolkit/openvino/pull/36042#discussion_r3288925505)
* “Since GPU already handles this correctly with a set of targeted per-pattern passes … we could move them from the GPU-specific code into `src/common/transformations/src/transformations/fp16_compression` and make them available for both plugins.” — [#36719 #issuecomment-4957140930](https://github.com/openvinotoolkit/openvino/pull/36719#issuecomment-4957140930)
* “Can we pass `supported_compressed_weights_types` as [a parameter of] `ConvertGroupedMatMulToGroupedMatMulCompressed`? This would simplify new precisions support.” — [#36666 r3528560646](https://github.com/openvinotoolkit/openvino/pull/36666#discussion_r3528560646)
* “let’s modify the `ConvertTiledMoeBlockToGatherMatmuls` matcher: we can add a pass parameter which would indicate if non-compressed inputs are supported.” — [#36265 r3398325418](https://github.com/openvinotoolkit/openvino/pull/36265#discussion_r3398325418)
* “The transformation is used in both plugins, so it should be implemented in both places.” — [#37181 r3727694189](https://github.com/openvinotoolkit/openvino/pull/37181#discussion_r3727694189)

**Confidence.** **Strong** (6 independent PRs).

---

### G16 — Do not over-constrain a transformation; justify every restriction

**Explanation.** Restrictions such as “single consumer”, “per-tensor only”, “constant path only”, “static shapes only” must be justified by a semantic requirement. If the rewrite is safe without them, drop them and add covering tests.

**Rationale.** Unjustified guards silently disable optimizations on real models and are invisible in tests, because the tests are written to match the guard.

**Frequency.** ~10 comments across **6 PRs**.

**Representative comments**
* “Why do we need to have this restriction? Semantically, we can eliminate FQ (equal to the target FQ) from several consumer branches.” — [#36075 r3460305890](https://github.com/openvinotoolkit/openvino/pull/36075#discussion_r3460305890)
* “consumers count check is not necessary here: we can still benefit from the fusion even if the broadcast has other consumers.” — [#36756 r3713837813](https://github.com/openvinotoolkit/openvino/pull/36756#discussion_r3713837813)
* “Do we have a reason for limiting this optimization only to constant paths? Actually, it can be beneficial in the non-const case as well.” — [#36756 r3713964776](https://github.com/openvinotoolkit/openvino/pull/36756#discussion_r3713964776)
* “we can easily apply this transformation to any FQ, not only per-tensor … Could you please add a test case covering per-channel FQ elimination, and correct the description?” — [#36075 r3460418072](https://github.com/openvinotoolkit/openvino/pull/36075#discussion_r3460418072)
* “A limitation on static shapes looks too strict from my perspective … `ov::symbol::util::dims_are_equal` … also works for dynamic shapes whose symbols are equal. Could you please double check if the static shapes requirement can be relaxed?” — [#37490 r3861806902](https://github.com/openvinotoolkit/openvino/pull/37490#discussion_r3861806902)
* “Actually the plugin implementation also supports float zero points, so the `is_integral_number` check is too strict here.” — [#36543 r3518687845](https://github.com/openvinotoolkit/openvino/pull/36543#discussion_r3518687845)
* “It looks like `output_low_pattern` and `output_high_pattern` shouldn’t be necessarily constant, let’s use `any_input()` instead.” — [#33269 r2683506183](https://github.com/openvinotoolkit/openvino/pull/33269#discussion_r2683506183)

**Confidence.** **Strong** (6 independent PRs).

---

### G17 — Document the pass in its header; keep every comment true

**Explanation.** Each pass header carries a short description of *what* it does, *why* it is needed, and an ASCII/scheme of the matched and produced subgraph (including optional nodes). Stale or contradictory comments are treated as defects.

**Rationale.** Transformation passes are otherwise opaque; the header is the only place a future maintainer can learn the intent. Wrong comments are worse than none — several reviews found comments that contradicted the code.

**Frequency.** ~21 comments across **13 PRs**.

**Representative comments**
* “Let’s describe here what this transformation does (moves DQ on outputs to weights) and why (to avoid f16 overflow).” — [#33072 r2789188154](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2789188154)
* “Let’s reflect optional subtract in the scheme.” — [#35280 r3190622573](https://github.com/openvinotoolkit/openvino/pull/35280#discussion_r3190622573)
* “It looks like the comment contradicts the expression: if a callback returns **true**, the corresponding **transformation doesn’t run**.” — [#33072 r2638461238](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2638461238)
* “The comment justification is misleading: in the GPU pipeline, `CommonOptimizations` are called after MoE passes.” — [#35311 r3179964020](https://github.com/openvinotoolkit/openvino/pull/35311#discussion_r3179964020)
* “The justification is wrong: shapes are actually known at compilation time … Please correct or remove the comment.” — [#37021 r3737163464](https://github.com/openvinotoolkit/openvino/pull/37021#discussion_r3737163464)
* “please add an explanatory comment on why we need the relaxed check in this case, and how exactly the relaxed check looks like (a pseudo formula would be enough).” — [#35941 r3689263469](https://github.com/openvinotoolkit/openvino/pull/35941#discussion_r3689263469)
* “let’s avoid mentioning the specific ops, since the list may be potentially expanded.” — [#36075 r3474402447](https://github.com/openvinotoolkit/openvino/pull/36075#discussion_r3474402447)

**Confidence.** **Strong** (13 independent PRs).

---

### G18 — Names must state semantics and scope

**Explanation.** Pass and variable names must describe *what is done* and *when*, not the model that motivated them. Rename when behaviour broadens. Use domain names for pattern dimensions (`hidden_in`, `seq_len`) instead of `?`.

**Frequency.** ~9 comments across **6 PRs**.

**Representative comments**
* “The current name doesn’t look descriptive: please try to reflect in the name that we fall back unsupported low-precision convolutions to fp16 execution.” — [#33072 r2793984258](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2793984258)
* “`gptoss_gemma3_mask` name is confusing now: it reflects the gemma4 case as well.” — [#36737 r3542532081](https://github.com/openvinotoolkit/openvino/pull/36737#discussion_r3542532081)
* “Let’s align the pass name with the logic: `DisableBF16CompForLtxVideoRopePattern`.” — [#36719 r3656395295](https://github.com/openvinotoolkit/openvino/pull/36719#discussion_r3656395295)
* “I agree to align it with other transformations: so I renamed it to `MoEMatMulsFusion`.” — [#32450 r2487254615](https://github.com/openvinotoolkit/openvino/pull/32450#discussion_r2487254615)
* “I would rename it to something like `mul_input`, as it is not always subtract.” — [#32983 r2704341885](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2704341885)

**Confidence.** **Strong** (6 independent PRs).

---

### G19 — Internal ops: minimal, immutable, self-validating API

**Explanation.** Internal (`ov_ops`) operations should not expose constructors that leave them unconfigured, nor setters that mutate configuration after construction. They must override `visit_attributes` (needed for serialization/caching and for `SharedOpOptimization`). `validate_and_infer_types` must reject configurations that the spec/implementation does not support instead of silently accepting them. Use `element::dynamic` (with empty shape) consistently to denote an absent optional input. Type-prop tests live in `src/core/tests/type_prop/`.

**Frequency.** ~15 comments across **8 PRs**.

**Representative comments**
* “Can we remove this constructor? I suppose we should always configure config when `MOECompressed` is created.” / “External modification of the existing config may be dangerous. Can we remove the setter?” — [#34963 r2997054108](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r2997054108), [r2997060808](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r2997060808)
* “I propose overriding `visit_attributes` … it is needed for correct model caching work: `visit_attributes` is used during `ov::Model` serialization/deserialization.” — [#35283 r3465920997](https://github.com/openvinotoolkit/openvino/pull/35283#discussion_r3465920997), [r3468455516](https://github.com/openvinotoolkit/openvino/pull/35283#discussion_r3468455516)
* “`ScaledDotProductAttention::validate_and_infer_types` lacks the check on scalar scale, which makes it possible to create an op which is not aligned with the spec.” — [#34177 r2993821065](https://github.com/openvinotoolkit/openvino/pull/34177#discussion_r2993821065)
* “I think it is better to prohibit the creation of such an operation … If we throw an exception with a clear message in such a case, it will be easier to understand that the issue is in wrong node configuration.” — [#35215 r3452305892](https://github.com/openvinotoolkit/openvino/pull/35215#discussion_r3452305892)
* “`element::dynamic` became a standard for the optional inputs in CPU and GPU plugin. Can the NPU plugin align its behavior in this aspect?” — [#37021 r3737225088](https://github.com/openvinotoolkit/openvino/pull/37021#discussion_r3737225088)
* “Such tests are usually placed in `src/core/tests/type_prop/`. Can we move this one there?” — [#34963 r2997158288](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r2997158288), [#36543 r3518666077](https://github.com/openvinotoolkit/openvino/pull/36543#discussion_r3518666077)
* “Semantically, `has_batch_dim` should have boolean type.” — [#35311 r3091642677](https://github.com/openvinotoolkit/openvino/pull/35311#discussion_r3091642677)
* “Minor: consider using `std::optional` instead of a special value meaning disabled scaling.” — [#35518 r3203939897](https://github.com/openvinotoolkit/openvino/pull/35518#discussion_r3203939897)

**Confidence.** **Strong** (8 independent PRs).

---

### G20 — Keep the PR scoped; move refactors and extensions to follow-ups

**Explanation.** A PR contains only changes required by its stated purpose. Unrelated edits (formatting drift, pipeline tweaks, build-flag changes, removed tests) must be reverted or split. Conversely, a reviewer-suggested broader refactor is scheduled as a follow-up rather than bolted on.

**Frequency.** ~10 comments across **8 PRs**.

**Representative comments**
* “Let’s avoid this change in this PR: I believe we should test I8 MaxPool support … within a separate activity.” — [#33072 r2606259912](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2606259912)
* “This file is not just moved+renamed, I see some unexpected code changes … Could you please double check this file to avoid unnecessary changes?” — [#34963 r2997090104](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r2997090104)
* “Why do we need this change?” (build flags) — [#34963 r2996290902](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r2996290902)
* “Why is this test removed?” — [#34963 r2997150804](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r2997150804)
* “yes, let’s better apply `state_management_pattern.cpp` corrections within a separate PR.” — [#36543 #issuecomment-4902452899](https://github.com/openvinotoolkit/openvino/pull/36543#issuecomment-4902452899)
* “Fair enough, let’s do the tests refactoring within a separate PR.” — [#35283 r3465691017](https://github.com/openvinotoolkit/openvino/pull/35283#discussion_r3465691017)

**Confidence.** **Strong** (8 independent PRs).

---

### Lower-confidence observations (not promoted to guidelines)

| Observation | Evidence | Confidence |
|---|---|---|
| Skip functional tests via `skip_tests_config.cpp` with a referenced ticket, not inline; a skipped test must still be a real test (`SKIP_IF_CURRENT_TEST_IS_DISABLED` after `run()`, or override `validate()`) | [#35311 r3182556103](https://github.com/openvinotoolkit/openvino/pull/35311#discussion_r3182556103), [#36870 r3673257204](https://github.com/openvinotoolkit/openvino/pull/36870#discussion_r3673257204) | Moderate (2 PRs) |
| Report unsupported configurations as unsupported rather than adding complexity (e.g. training-only semantics) | [#35215 r3414582306](https://github.com/openvinotoolkit/openvino/pull/35215#discussion_r3414582306) | One-off |
| Avoid RAII guards where failure means the whole compilation must fail anyway | [#35426 r3283288244](https://github.com/openvinotoolkit/openvino/pull/35426#discussion_r3283288244) | One-off (own PR) |
| `using namespace ov::pass::pattern;` at function/file scope in pass sources is acceptable and improves readability | [#33072 r2793343098](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2793343098) | One-off |
| Header-only `inline` implementations of pass logic should move to `.cpp` | [#35280 r3217584484](https://github.com/openvinotoolkit/openvino/pull/35280#discussion_r3217584484) | One-off |
| Don’t create deep namespaces for a single utility | [#33072 r2793909024](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2793909024) | One-off |
| Use `""` (not `<>`) for OpenVINO includes | [#33072 r2638463744](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2638463744) | One-off |

---

# 4. Top 10 Review Recommendations (ranked)

| Rank | Recommendation | Frequency | Why it matters | Representative references |
|---|---|---|---|---|
| 1 | **Add a `TransformationTestsF` test with a reference model** covering the exact changed behaviour (and a regression test for every bug fix) | ~55 comments / 21 PRs | The reference comparison validates the whole resulting topology, catches collateral damage, and prevents the fix from silently regressing | [#35280](https://github.com/openvinotoolkit/openvino/pull/35280#discussion_r3161420603), [#36058](https://github.com/openvinotoolkit/openvino/pull/36058#discussion_r3310270387), [#36737](https://github.com/openvinotoolkit/openvino/pull/36737#discussion_r3542662965), [#33162](https://github.com/openvinotoolkit/openvino/pull/33162#pullrequestreview-3602498189) |
| 2 | **Move every matching condition from the callback into the pattern** | ~40 / 17 | Makes the applicability contract explicit, removes defensive code, makes non-firing passes debuggable | [#33072](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2601523538), [#35215](https://github.com/openvinotoolkit/openvino/pull/35215#discussion_r3414766660), [#36265](https://github.com/openvinotoolkit/openvino/pull/36265#discussion_r3401658531) |
| 3 | **Reuse existing helpers/passes; never introduce a near-duplicate** | ~38 / 12 | Duplicated non-trivial logic diverges and fixes land in only one copy | [#35518](https://github.com/openvinotoolkit/openvino/pull/35518#discussion_r3161441359), [#36719](https://github.com/openvinotoolkit/openvino/pull/36719#discussion_r3536607065), [#34963](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r2997128624) |
| 4 | **Trust the matcher; assert instead of silently returning false** | ~34 / 12 | Silent no-ops are the hardest transformation defect to diagnose; asserts also clear static-analysis findings | [#33072](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2735803886), [#34963](https://github.com/openvinotoolkit/openvino/pull/34963#discussion_r3009363156), [#36737](https://github.com/openvinotoolkit/openvino/pull/36737#discussion_r3542552399) |
| 5 | **Use declarative predicates (`shape_matches`, `type_matches_any`, `attrs_match`, `optional<>`, `\|`) instead of custom lambdas** | ~25 / 11 | Built-in predicates handle dynamic dims, symbols and broadcasting correctly and remove explanatory comments | [#32983](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2556924250), [#36756](https://github.com/openvinotoolkit/openvino/pull/36756#discussion_r3713886726), [#37490](https://github.com/openvinotoolkit/openvino/pull/37490#discussion_r3861280827) |
| 6 | **Parametrize tests and reuse shared model builders** | ~20 / 10 | Multiplies coverage at near-zero cost; prevents partially-updated duplicated instances | [#32983](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2704315244), [#35283](https://github.com/openvinotoolkit/openvino/pull/35283#discussion_r3452615180), [#36075](https://github.com/openvinotoolkit/openvino/pull/36075#discussion_r3460435941) |
| 7 | **Preserve runtime info / friendly names; use canonical replacement helpers** | ~20 / 8 | Runtime info drives downstream precision markup and plugin fusing; losing it changes behaviour invisibly | [#33072](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2601530295), [#32983](https://github.com/openvinotoolkit/openvino/pull/32983#discussion_r2704344082), [#33160](https://github.com/openvinotoolkit/openvino/pull/33160#discussion_r2673190213) |
| 8 | **Document the pass in its header and keep comments truthful** | ~21 / 13 | The header is the only intent record for an otherwise opaque pass; wrong comments actively mislead | [#33072](https://github.com/openvinotoolkit/openvino/pull/33072#discussion_r2789188154), [#35945](https://github.com/openvinotoolkit/openvino/pull/35945#discussion_r3333386516), [#37021](https://github.com/openvinotoolkit/openvino/pull/37021#discussion_r3737163464) |
| 9 | **Fix the general cause, not the observed topology; no hardcoded model-specific workarounds** | ~14 / 9 | Proxy conditions are both too narrow and too broad; workarounds become permanent | [#33159](https://github.com/openvinotoolkit/openvino/pull/33159#discussion_r2667823897), [#35969](https://github.com/openvinotoolkit/openvino/pull/35969#discussion_r3280185386), [#35518](https://github.com/openvinotoolkit/openvino/pull/35518#discussion_r3203932706) |
| 10 | **Keep pass responsibility single, place it correctly, and don’t over-constrain it** | ~22 / 12 | Composable passes are individually testable; unjustified guards silently disable optimizations | [#37181](https://github.com/openvinotoolkit/openvino/pull/37181#discussion_r3722382961), [#36075](https://github.com/openvinotoolkit/openvino/pull/36075#discussion_r3460305890), [#36042](https://github.com/openvinotoolkit/openvino/pull/36042#discussion_r3288925505) |

---

# 5. Transformations Code Review Checklist

### Transformation design
- [ ] Is this a new pass, or can an existing pass be extended/parameterized? (G10, G15)
- [ ] Does the pass have **one** responsibility? Would splitting into several matchers under a `GraphRewrite` be clearer? (G9)
- [ ] Is the correct base class used — `MatcherPass` for local rewrites, `ModelPass` when nodes outside the matched pattern are modified, `BackwardGraphRewrite` when bottom-up order is required? (G9)
- [ ] Is device-specific behaviour expressed as a *pass parameter* rather than a plugin-local copy? (G15)
- [ ] Is the change the general fix, not a fix for the one model that reported it? (G8)
- [ ] Are hardcoded precisions / magic values replaced by parameters or `std::numeric_limits`? (G8, G12)

### Pattern matching
- [ ] Are **all** applicability conditions in the pattern (types, ranks, shapes, attributes, constant-ness, static-ness)? (G1)
- [ ] Are declarative predicates used — `shape_matches`, `rank_equals`, `type_matches[_any]`, `attrs_match`, `value_matches`, `has_static_shape`, `wrap_const`? (G2)
- [ ] Are optional nodes modelled with `pattern::optional<Op>()` instead of manual `Or` chains? (G2)
- [ ] Are alternatives written with `a | b` instead of explicit `Or` construction? (G2)
- [ ] Is there any manual graph traversal in the callback (or in the plugin pipeline) that the pattern could express? (G1)
- [ ] Are symbols (`"OC, IC, 3, 3"`) used to extract dimensions instead of index arithmetic? (G2)
- [ ] Is the pattern still the minimal one required — no extra nodes added only to work around ordering issues? (G14)

### Graph correctness
- [ ] Is `ov::copy_runtime_info(old, new)` called for every created node? (G11)
- [ ] Are friendly names propagated (`replace_output_update_name`, explicit `set_friendly_name`)? (G11)
- [ ] Is `replace_node` / `replace_output_update_name` used instead of ad-hoc `replace_source_output` sequences? (G11)
- [ ] Are `TypeRelaxed` nodes recreated via `clone_with_new_inputs` so mixed-precision config is inherited? (G11)
- [ ] Is manual dead-node cleanup avoided (`Validate` handles it)? (G11)
- [ ] Are markup attributes (`keep_const_precision`, `DisableFP16Compression`, precision attributes) preserved/propagated rather than re-applied? (G11)
- [ ] Does the rewrite match the operation **specification** (e.g. FQ output shape == data input shape)? (G8)

### Validation and invariants
- [ ] Are redundant `as_type_ptr` + null checks on matched nodes removed? (G3)
- [ ] Is `pattern_map.at(label)` used for mandatory nodes? (G3)
- [ ] Where an invariant must hold, is `OPENVINO_ASSERT`/`OPENVINO_THROW` used instead of `return false`? (G3)
- [ ] For new/changed internal ops, does `validate_and_infer_types` reject unsupported configurations with a clear message? (G19)
- [ ] Are dynamic/empty optional inputs handled with the `element::dynamic` convention? (G19)

### Runtime behaviour
- [ ] Is the pass registered in **every** pipeline that requires it (Common Optimizations, MoC, CPU, GPU, NPU, FQ-stripping)? (G14)
- [ ] Is `InitNodeInfo` still first and `Validate` still last in modified pipelines? (G14)
- [ ] Was the pass already registered elsewhere (duplicate registration)? (G14)
- [ ] Does the new ordering break an existing consumer (e.g. SDPA fusion before `CommonOptimizations`)? (G14)

### API consistency
- [ ] No unused constructors/setters on internal ops; configuration is immutable after construction. (G19)
- [ ] `visit_attributes` overridden for any op with custom attributes (serialization, caching, `SharedOpOptimization`). (G19)
- [ ] Semantic types used (`bool` for flags, `std::optional` instead of sentinel values). (G19)
- [ ] No default parameter values that are used at a single call site. (G19)
- [ ] Type-prop tests placed in `src/core/tests/type_prop/`. (G19)

### Performance
- [ ] Could the rewrite degrade the quantized path (e.g. replacing a fused `Convolution` with `Transpose→MatMul→Transpose`)? (G16 / §2.12)
- [ ] Do inserted `Transpose`/`Reshape` nodes break plugin post-op fusing? Can they be placed after bias instead? (§2.12)
- [ ] Does the pass avoid firing when it is a no-op (e.g. `H*W == 1`)? (§2.12)
- [ ] Is a "general" markup pass disabling optimizations for more subgraphs than needed? (§2.12)

### Memory considerations
- [ ] Does the rewrite keep the weight-decompression subgraph unfolded where that is the memory-efficient form? (§2.12)
- [ ] Is constant folding used deliberately (`get_constant_from_source`) rather than accidentally materializing large tensors? (G13)

### Test coverage
- [ ] `TransformationTestsF` used, with `model` / `manager` / `model_ref`? (G4)
- [ ] Negative cases leave `model_ref` empty? (G5)
- [ ] Regression test derived from the reporting model’s subgraph? (G7)
- [ ] Tests parametrized rather than copy-pasted; shared builders reused? (G6)
- [ ] Corner cases covered: dynamic shapes, per-channel vs per-tensor, odd dimensions, missing optional inputs, unsupported precisions? (G6, G16)
- [ ] Runtime-info comparison enabled where the pass sets runtime info? (G4)
- [ ] Skips go through `skip_tests_config.cpp` with a ticket; no test is neutered by removing reference computation. (lower-confidence)

### Maintainability
- [ ] Are duplicated code blocks extracted into a shared, parametrized helper? (G10)
- [ ] Is the diff free of unrelated changes (build flags, formatting, removed tests)? (G20)
- [ ] Do names describe semantics and current scope, not the originating model? (G18)
- [ ] Are large plugin pipeline files kept from growing without bound (callbacks extracted)? (G9)

### Documentation
- [ ] Header describes *what* the pass does and *why* it is needed. (G17)
- [ ] Subgraph scheme included, with optional nodes marked. (G17)
- [ ] All comments verified against the code (no contradictions, no stale justifications). (G17)
- [ ] Non-obvious numeric criteria explained (at least a pseudo-formula). (G17)
- [ ] PR title/description updated to match the final implementation. (G20)

---

# 6. Draft "Writing Transformations" Guidelines Document

The recurring review feedback was converted into a contributor-facing guide, written by analogy with
the existing `writing_tests.md`:
**[`src/common/transformations/docs/writing_transformations.md`](src/common/transformations/docs/writing_transformations.md)**

It is linked from [`src/common/transformations/docs/README.md`](src/common/transformations/docs/README.md).

---

# 7. Evidence and Confidence Assessment

| ID | Guideline | Independent PRs | Comments (approx.) | Confidence |
|---|---|---|---|---|
| G1 | Express matching constraints in the pattern, not the callback | **17** | ~40 | **Strong** |
| G4 | `TransformationTestsF` + reference model for every behavioural change | **21** | ~55 | **Strong** |
| G17 | Document the pass in its header; keep comments truthful | **13** | ~21 | **Strong** |
| G3 | Trust the matcher; assert instead of silent `return false` | **12** | ~34 | **Strong** |
| G7 | Regression test built from the reporting model’s subgraph | **12** | ~15 | **Strong** |
| G10 | Reuse existing utilities/passes; no near-duplicates | **12** | ~38 | **Strong** |
| G2 | Declarative predicate vocabulary over custom lambdas | **11** | ~25 | **Strong** |
| G6 | Parametrize tests; reuse shared model builders | **10** | ~20 | **Strong** |
| G8 | Fix the general cause, not the observed topology | **9** | ~14 | **Strong** |
| G9 | One pass, one responsibility; compose passes | **8** | ~12 | **Strong** |
| G11 | Preserve metadata; canonical replacement helpers | **8** | ~20 | **Strong** |
| G14 | Register in every pipeline; respect pipeline invariants | **8** | ~12 | **Strong** |
| G19 | Internal ops: minimal, immutable, self-validating API | **8** | ~15 | **Strong** |
| G20 | Keep the PR scoped; split follow-ups | **8** | ~10 | **Strong** |
| G12 | Precision correctness in low-precision code | **7** | ~12 | **Strong** |
| G15 | Device-agnostic passes belong in common transformations | **6** | ~8 | **Strong** |
| G16 | Do not over-constrain; justify every restriction | **6** | ~10 | **Strong** |
| G18 | Names must state semantics and scope | **6** | ~9 | **Strong** |
| G5 | Negative tests leave `model_ref` empty | **3** | 4 | **Moderate** |
| G13 | Prefer OV-opset computation over manual constant arithmetic | **3** | 4 | **Moderate** |
| — | Test-skip discipline (`skip_tests_config.cpp` + ticket) | 2 | 2 | **Moderate** |
| — | `using namespace ov::pass::pattern` in pass sources | 1 | 1 | **One-off** |
| — | Move `inline` pass logic out of headers | 1 | 1 | **One-off** |
| — | Avoid deep namespaces for a single utility | 1 | 1 | **One-off** |
| — | Report unsupported configurations rather than adding complexity | 1 | 1 | **One-off** |
| — | Avoid RAII guards when failure must abort compilation anyway | 1 | 1 | **One-off** |
| — | `""` vs `<>` for OpenVINO includes | 1 | 1 | **One-off** |

**Promotion rule applied:** only patterns observed in **≥ 3 independent PRs** were promoted to numbered guidelines (G1–G20); everything below that threshold is listed as a lower-confidence observation and is *not* normative in the companion document.

### Threats to validity

* **Label history is only as good as the GraphQL timeline.** Labels removed *and* re-added, or applied before the timeline window, are captured; labels never applied are not. PRs touching transformations but never labelled `category: transformations` / `category: LP transformations` are out of scope by construction.
* **Category assignment is heuristic.** A rule-based classifier with an explicit priority order was applied to all 400 comments and then sanity-checked against a full manual read of the corpus. Comments frequently span two themes; the single-label counts under-report breadth (see the multi-label figures in §2).
* **Reviewer bias toward large PRs.** Five PRs contribute ~44 % of the comments. Guidelines supported *only* by those PRs would be weakly generalizable; every Strong guideline above is therefore additionally required to appear in at least 6 distinct PRs, most of them outside that top-five set.
* **“Technical vs administrative” filtering** uses conservative heuristics; a small number of short technical remarks (e.g. bare ` ```suggestion ` blocks with substantive code) may have been dropped, and a few near-administrative replies retained.
* **Author replies excluded.** The 15 PRs authored by `v-Golubev` (81 comments) were excluded from statistics; they were still read, and several corroborate the guidelines from the author side (e.g. `.at()` preference in [#32266](https://github.com/openvinotoolkit/openvino/pull/32266#discussion_r2491324024), `Validate`-last in [#32266](https://github.com/openvinotoolkit/openvino/pull/32266#discussion_r2491310877)). Where a corroborating quote comes from such a PR, it is marked in context.
