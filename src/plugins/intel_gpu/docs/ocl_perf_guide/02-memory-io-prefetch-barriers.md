# 02. 메모리 IO, prefetch, barrier (Xe2 / Xe-HPG, sdpa_ocl 계열)

대상: `sdpa_ocl*.cl` 커널 군(`src/plugins/intel_gpu/src/graph/impls/ocl_v2/`)에서 실제로 쓰였거나 시도된 메모리 경로 기법.
DPAS/타일링은 `01-dpas-and-tiling.md`, 수치/양자화는 `03-numerics-softmax-quantization.md`, ISA/spill/프로파일링은
`04-spill-isa-profiling.md`, 방법론은 `05-methodology-and-pitfalls.md`를 본다.

## 0. 표기와 증거 등급

- 하드웨어: **B580** = Arc B580 (Xe2, 초기 측정), **B70** = Arc Pro B70 (Xe2, 2026-07-05 이후 개발 장비), **DG2** = Arc A770 (Xe-HPG, SG8),
  **iGPU** = 이 PC의 GPU.0 (XMX 없음). B580과 B70은 같은 Xe2라 builtin 존재/레이아웃/ISA 분석은 그대로 유효하지만 절대 ns는 장비 의존이다.
- **MEASURED** = 사용자가 돌린 측정(출처 명시). **ISA** = ocloc/IGC ISA 정적 계수(런타임 시간 아님; 이 프로젝트에서 여러 번 시간과 어긋남, §13).
  **ASSUMED** = 가설, 측정 없음. AGENTS.md 규칙: 측정 없이 성능 개선을 주장하지 않는다.
- 코드 인용은 `ocl_v2/` 기준 `path:line` (현재 working tree 기준; 줄 번호는 편집으로 흔들린다).
- 메모리 노트의 날짜는 노트 파일 이름(`~/.claude/projects/.../memory/<name>.md`)으로 표기한다.

## 1. 2D block IO builtin 요약

확장: `cl_intel_subgroup_2d_block_io` (`#pragma OPENCL EXTENSION ... : enable`). Xe2(bmg)에서만 사용 가능.
**DG2/ARL-H(xe_hpg)는 이 확장을 광고하지 않고 read/write/prefetch 전부 컴파일 거부**(MEASURED, DG2 S0 프로브, `sdpa-ocl-xe-hpg-facts`; compute-runtime
`compiler_product_helper_before_xe_hpc.inl`). 그래서 SG8 arm은 2D block을 jit 스위치로 끈다: `block2d_io_allowed()` (`sdpa/sdpa_gen_ocl.cpp:720`).
DG2에서 pragma는 경고만 내고 매크로는 정의되지 않으므로 `#ifdef`로는 끌 수 없다.

시그니처 형태(read): `intel_sub_group_2d_block_read_<variant>_<bits>_<rows>r<cols>x<count>c(void* base, int width_bytes, int height, int pitch_bytes, int2 coord, T* dst)`.
`coord.x`의 단위는 **요소**(8b면 byte, 16b면 half, 32b면 dword), `coord.y`는 행. width/pitch는 **byte**.

| 용도 | builtin | lane 의미 | 사용처 |
|---|---|---|---|
| 일반 read | `read_16b_16r16x1c` | lane=열, 요소=행(16) | f16 K 입력/페이지의 KQ A operand (`sdpa_ocl_qk_load.cl:256,330`) |
| 일반 read 32b | `read_32b_8r16x1c` | lane=dword 열, 요소=행(8) | u4 Kc를 DWORD surface로 (`sdpa_ocl_qk_load.cl:466`) |
| transpose | `read_transpose_32b_16r8x1c` | lane=행, 8 dword = 한 행의 16 half | Q B operand (`sdpa_ocl_qk_load.cl:24`), decode K B operand (`sdpa_ocl_decode.cl:547`) |
| transform(VNNI) 16b | `read_transform_16b_16r16x1c` | lane=열, dword = 행 2개 | V B operand (`sdpa_ocl_v_load.cl:55`), decode V |
| transform 8b | `read_transform_8b_32r16x1c` (`_32r16x2c`, `_32r16x4c`) | lane=열, uint = 행 4개를 byte로 | i8/u4 K·V (`sdpa_ocl_qk_load.cl:275`, `sdpa_ocl_v_load.cl:87-99`) |
| write | `write_16b_8r16x1c` | lane=열 | O 출력 (`sdpa_ocl.cl:1033`) |
| prefetch | `prefetch_16b_16r16x1c`, `prefetch_8b_32r16x1c`, `prefetch_32b_16r8x1c` | 목적지 없음(null destination) | `sdpa_ocl_v_load.cl:30`, decode `sdpa_ocl_decode.cl:190,197` |

builtin 존재 여부(MEASURED, ocloc `-device bmg`, `sdpa-ocl-8b-transform-32row-min` 2026-07-08):

| 종류 | 존재 | 없음 |
|---|---|---|
| 8b transform | `32r16x1c`, `32r16x2c`, `32r16x4c` | `16r16x1c/2c/4c` (**16행 transform 없음**) |
| 8b 일반 | `16r16x4c`, `16r32x1c`, `32r16x4c` | `32r16x1c` (1바이트는 열 수가 4의 배수여야 함, transform 제외) |
| transpose | 32b 만 (`transpose_32b_16r8x1c`, `_32r8x1c`) | `transpose_8b_*`, `transpose_16b_*` |
| 16b 일반 | `16r16x1c`, `32r16x1c`(컴파일됨, 레이아웃 미검증, `sdpa-ocl-tk32-bug-hunt`) | |
| prefetch 8b | `32r16x1c`, `16r16x4c`, `16r32x1c/2c`, `32r16x2c/4c`, `32r32x1c/2c`, 1/2/4/8행 | (prefetch는 transform/transpose 구분 없음) |

lane/요소 의미는 반드시 **디바이스 프로브로 확인**한다 (`test/microbench/probe_v_layouts.cpp`, `probe_k_16b_32r.cpp`, `probe_v_multiblock.cpp`,
`run_probe.sh`). 읽은 값으로 (key, head)를 역추적하는 방식(K[key][h]=key로 채워 각 slot이 어떤 key를 담는지 출력)이 근거다.
함정: 프로브가 "일치"를 보고했지만 두 builtin 모두 아무것도 쓰지 않은 경우(all-zero 비교)가 있었다. 반드시 sentinel로 채우고 "쓰인 slot 수"를 확인한다
(`sdpa-ocl-tk32-bug-hunt`, 교훈 5). **`16b_32r16x1c`가 `2x16b_16r16x1c`의 drop-in인지는 아직 검증되지 않았다** (프로브가 무효였음).

## 2. 규칙: spec vs 우리 helper

spec(`cl_intel_subgroup_2d_block_io` §6.13.X.6, 2026-08-11 조회, `sdpa-ocl-8b-transform-32row-min`)의 정의되지 않은 동작(UB) 조건과, 우리 host helper
(`sdpa/sdpa_ocl_utils.hpp`)가 적용하는 규칙, 그리고 B70에서 실제 측정한 경계를 비교한다.

| 규칙 | spec | 우리 helper | B70 측정 |
|---|---|---|---|
| subgroup | 크기 16, full | `block2d_io_allowed(sg==16)` (`sdpa_gen_ocl.cpp:720`) | - |
| `coord.x` | 8b면 4의 배수, 16b면 2의 배수 | 커널에서 보장(`sg_j0_sv`, `db*DPAS_K` 등) | - |
| base 주소 | 64 B 정렬 | strict tier `%64`; fixup tier는 in-kernel 보정 | **4/16/32/48 B 어긋난 base는 16 B 배수 stride면 fixup 없이도 정상 읽힘**, 2 B 어긋나면 오답 (`sdpa-ocl-ki-t7`) |
| width | 64..224 B, 4의 배수(8/16b) | `row_bytes >= 64` | **224 B 상한은 실제로 어겨도 동작**: f16 K read가 head 128에서 width 256, head 512에서 1024로 정상(B70). 의존은 금지 |
| pitch | width 이상, 16 B 배수 | `row_bytes % 16` (fixup) / `% 64` (strict) | token stride 260 B/264 B(16의 배수 아님)는 **오답** (0.0118~0.068 오차) |
| height | (0, 224] | page read는 clamp (`kp_rows = PA_PAGE_ROWS(k, key0)`) | OOB 행은 HW가 0으로 채움 |

`block2d_surface_ok(row)` = `row >= 64 && row % 64 == 0` (`sdpa_ocl_utils.hpp:52`): 스펙보다 **엄격**하다. 스펙은 사실상 `width >= 64`와 `pitch % 16`만 요구한다.
head 80/96/112의 i8 페이지(row 80/96/112 B)는 스펙상 합법이지만 우리 helper가 제외한다 (`sdpa-ocl-8b-transform-32row-min`).

### 2.1 Block IO legality checklist

새 2D block 경로를 추가하거나 게이트를 바꾸기 전에 모두 통과해야 한다.

| # | 점검 | 확인 방법 | 위반 시 증상 |
|---|---|---|---|
| 1 | subgroup 크기 16인가 (DG2 SG8이면 builtin 자체 없음) | host `block2d_io_allowed` | DG2 컴파일 실패 또는 조용한 오답 |
| 2 | 해당 (bits, rows, cols, variant) 조합이 존재하는가 | `test/microbench/probe_dpas_api.sh`, ocloc | 컴파일 실패 |
| 3 | 8b transform이면 32행 소비/over-read를 감수하는가, transpose면 32b인가 | §4 | 16행 변형 부재 |
| 4 | width >= 64 B, 8b/16b면 4의 배수 | row_bytes 계산 | UB |
| 5 | pitch가 16 B 배수인가 (rank-2 PA는 padding 포함 token stride) | §3.5 | **조용한 오답** (B70 측정) |
| 6 | base 64 B 정렬을 증명했는가, 아니면 fixup(round-down, x 이동, width 확장)을 켰는가 | §3.2 | 오답 또는 (B70에서는) 우연히 정상 |
| 7 | 32b transpose의 fixup이면 `prem % 4 == 0`가 증명되는가 (Q, A) | 현재 Q/A에는 fixup 자체가 없음 | WRONG result, 성능 문제가 아님 |
| 8 | `coord.x`의 단위를 맞췄는가 (8b=byte, 16b=half, 32b=dword) | 코드 리뷰 | 1/2배 어긋난 열 |
| 9 | 표면 height를 실제 유효 행으로 clamp했는가 (page의 미기록 slot, 마지막 k0 타일) | `kp_rows`, `y_sub` | 미기록 slot의 NaN이 점수에 유입 |
| 10 | height <= 0인 읽기를 발행하지 않는가 | `if (kp_rows > 0)` / zero-fill | 비합법 read |
| 11 | head tail(`d`가 `DPAS_K`의 배수가 아님)은 surface width(`d*sizeof`)로 HW zero-fill에 맡기거나 커널 가드를 두었는가 | `KP_w = d * sizeof(half)` | 다음 head 값이 섞임 |
| 12 | 동적 padding이면 host가 증명 불가임을 인정하고 전제(stride %16, 시작 %4)를 문서화했는가 | `block2d_layout_fixup_ok` 주석 | 동적 K/V 오답(잔여 위험) |
| 13 | block read 대상이 다음 읽기에 이어지는 인접 행이라는 가정(paired read)이 PA 페이지에서 성립하는가 | 페이지는 비인접 | 다른 페이지 읽음 |
| 14 | `SDPA_OCL_*_2D=1` 강제 override가 안전한가 (fixup flag가 override 뒤에 파생되는가) | §3.2 | 타이밍은 맞고 결과는 틀림 (head 72에서 12/16 head 오답) |

## 3. Host 게이트: 계층, base fixup, padding

### 3.1 계층 (현재 코드)

| 계층 | 함수 | 조건 | 대상 |
|---|---|---|---|
| strict | `block2d_layout_ok(layout, row)` (`sdpa_ocl_utils.hpp:112`) | `row >= 64 && row % 64 == 0` + padding 증명 | Q, A(출력), plain i8 K/V, fixup 없는 base |
| fixup | `block2d_layout_fixup_ok` | `row >= 64 && row % 16 == 0` + padding 증명 | plain f16 K/V, MIXED Kc/Vc, `BLOCK2D_KV_BASE_FIXUP`/`CUR_BASE_FIXUP` 켬 |
| page | `block2d_page_ok(row)` | `row >= 64 && row % 16 == 0` | PA 캐시 페이지 (f16, i8 BY_TOKEN/BY_CHANNEL) |
| u4 | `block2d_surface_ok` (strict) | `%64` 유지 | u4 K/V 페이지, 1D uc16 읽기와 상호 배타 (§5.2) |
| decode | `block2d_surface_ok` (`sdpa_gen_ocl_decode.cpp:49-50`) | strict `%64` | decode K/V 페이지: f16은 `h%32`, i8 `h%64`, u4 K `h%128` |

host 호출 지점: `sdpa_gen_ocl.cpp:733-744` (Q, KV, A), `:792-823` (PA 페이지), `:835-848` (Kc/Vc), `:857-868` (plain i8).

### 3.2 `%64 -> %16` 완화와 base fixup (gemma-4 head-72 prefill)

**무엇**: `block2d_surface_ok()`가 폭/pitch/base **세 규칙을 `row_bytes % 64`로 뭉쳤다.** head 72 f16(row 144 B)에서 USE_2D_BLOCK_IO_Q/KV/A가 전부 0이 되어
K/V가 subgroup당 k0당 **256개 per-lane scalar gather**(micro는 약 13개 block read)로 떨어졌다. 규칙을 따로 검사해 보니 위반은 base 정렬 하나뿐이었다
(`base + head*144`; width 144 B, pitch 2304 B는 통과).

**해결(2개 독립 레버, 둘 다 반영, `191722ba0e`)**:
1. 게이트를 `row_bytes % 16`으로 완화(pitch와 base가 row의 정수배이므로 `%16`이 width/pitch를 함의하고 base만 남음). 남은 base는 in-kernel로 보정한다
   (micro `block2d_load`가 항상 했던 방식): base를 64 B로 round-down, `x += prem/elem`, `w += prem`.

```c
// sdpa_ocl.cl:300-309 (BLOCK2D_KV_BASE_FIXUP)
const uint k_prem = (uint)(as_long(K) & 63);
K_b2d   = (const global KEY_DATA_T *)((const global char *)K - k_prem);
KD_w_b2d = KD_w + (int)k_prem;
KD_x0   = (int)(k_prem / sizeof(KEY_DATA_T));
// read: (int2)(KD_x0 + db*DPAS_K, key_base) 에 KD_w_b2d 폭으로
```

   widening이 **뒤쪽(낮은 주소)으로** 확장되므로 head dim을 넘는 열은 계속 OOB이고 HW가 0으로 채운다. 기존 `head < d` 가드와 같은 효과이며 다음 head의 값이 새지 않는다.
   `base - prem`이 할당 밖으로 내려가지 않음은 `base = buf_base + m*144`, `buf_base`가 64 B 정렬이라는 사실로 증명(`sdpa-ocl-block2d-padded-view-gate`). 비용 0
   (AND 2개를 루프 밖으로 hoist, MEASURED 0.7% 이내, B70/B580 `sdpa_ocl_head72_analysis.md`).
   → head-72 prefill 6.96x (MEASURED, 같은 문서).
2. `DKS_ACTIVE = ceil(HEAD_SIZE/DPAS_K)` (5 vs 8 at head 72, 3 타일이 0만 로드해 DPAS에 0을 먹였음). → 추가 1.33x, 작업량 감소(-19%)보다 큰 이유는
   `Q_slm` 8192->5120으로 WG SLM이 17536->14464 B가 되어 **Xe2 Xe-core(128 KB)당 상주 WG가 7->9**(MEASURED 해석, ISA 아님; 점유율 논거는 이 프로젝트에서
   두 번 빗나간 전례가 있으므로 ASSUMED로 취급).

종합: gemma-4-26b-a4b-it SigLIP prefill(head 72, 16 heads, q=k~2528) 9,570,971 -> 1,030,869 ns avg (**9.28x, micro 대비 1.14x 빠름**)(MEASURED, B70, cliloader,
`sdpa-ocl-block2d-gate-relaxation`). 정확성: `*ScaledAttn*-*486*:*387*` 전부 PASS (`--device_suffix=1`).

**fixup flag는 override 뒤에 파생**한다(`sdpa_gen_ocl.cpp:741`: `BLOCK2D_KV_BASE_FIXUP = kv_2d && !kv_aligned`). 그 전에는 `SDPA_OCL_KV_2D=1`로 강제하면
head 72에서 12/16 head가 틀렸다 (타이밍만 맞음). 지금은 이 override가 "rebuild 없이 가설을 증명하는 A/B 레버"로 안전하다.

**부수 결과(세 가지 직관과 반대, 재도출 금지)**:
- **256 GRF가 오히려 47% 느림**(9.57 -> 14.12 ms) spill을 없앴는데도. LSC 메시지 수가 병목이지 spill이 아니고 상주 스레드 절반이 더 비쌌다 (`get_build_options`의 256GRF 주석, `sdpa_gen_ocl.cpp:1038-1041`).
- SPILL=2688은 원인이 아니라 증상: 로드 방식을 바꾸자 부수적으로 0이 됨.
- `Q_2D`/`A_2D`는 perf-irrelevant (Q staging은 prologue, A store는 epilogue, k0 루프 약 20회 밖): `SDPA_OCL_Q_2D=1` 측정 +0.35%. Q/A에는 base fixup이 없으므로 강제는 **오답** 위험.

### 3.3 padded view 게이트 (`bd8d90097d`, phi-4 vision tower)

fixup tier가 `!l.data_padding`을 요구해서 fused-QKV의 `Split` crop view(in-place padded)가 scalar gather에 남았다. phi-4-multimodal 비전 타워(head 72, 16 heads,
q=k=1024, 마스크 kind 2)가 micro 대비 **7.5x 느렸다**. B70 측정(208 call): micro b=2 411,976 ns, ocl 전 3,097,727 ns(7.52x 느림), `SDPA_OCL_KV_2D=1` 393,355 ns(1.047x 빠름);
b=5 micro 1,022,198 / ocl 전 8,769,732 / 971,180. SPILL 9,344 B -> 0 (MEASURED, `sdpa-ocl-block2d-padded-view-gate`).

핵심 논리: simple data format에서 모든 OV pitch는 padded 차원들의 곱 = `X_PITCH`(= padded_x)의 정수배다. **X가 padding 없음**이면 `row%16`이 `pitch%16`과 `>=64`를 함의한다.
batch/head/seq 패딩은 pitch를 **곱할 뿐**이다(`pitch = (16+padY)*72*2`는 항상 144의 배수). pad-before는 `INPUT*_OFFSET`에 행의 정수배로 접혀 base는 16의 배수만큼 어긋나고
fixup이 정확히 복구한다. 구현: `axis_unpadded(layout, ChannelName::X)` (`sdpa_ocl_utils.hpp:69`)가 `get_channel_index`로 `padding::_lower_size/_upper_size/_dynamic_dims_mask`를
`jitter.cpp:90-92`와 같은 방식으로 읽는다. 폴백 비용(subgroup/k0당): K 5*2*8 = **80 gather vs 5 block read**, V 8*16 = **128 vs 8** (16x 메시지 수), 약 16개 live gather 주소가 spill 원인.

### 3.4 rank-2 PA padding (T7, `1fece9b9c1`)

PA의 Q/K/V는 rank-2 `[tokens, heads*head_size]`다. head 차원이 FEATURE 안에 있어서 feature padding이 base(padding-before만큼)와 모든 head의 token stride를 **임의의 양**으로 움직인다.
rank-4에서 쓰던 "pitch와 base는 row의 정수배" 논거가 성립하지 않고 `axis_unpadded(X)`가 vacuous하게 참이다(포맷에 X가 없음 -> `get_channel_index` = -1).

| 입력 | strict | fixup |
|---|---|---|
| rank-4 | simple format && X unpadded | 동일 |
| rank-2 static padding | `first_head % 64 == 0 && token_stride % 64 == 0` | `first_head % 16 == 0 && token_stride % 16 == 0` (장치보다 보수적) |
| rank-2 dynamic padding | **불가(false)**: Q/A는 scalar로, K/V/Kc/Vc는 fixup tier | **신뢰(true)**: 전제 `stride % 16`, 첫 head 시작 `% 4` |
| 그 외 rank | 거부 | 거부 |

B70 측정(`paged_attention_feature_pad_test`, 55 case, `sdpa-ocl-ki-t7-block2d-padding`): **재현되는 결함은 token stride가 16 B 배수가 아닐 때(260/264 B)와 첫 head 시작이 2 B 어긋날 때**다.
base가 4/16/32/48 B 어긋나도 16 B 배수 stride면 fixup 없이 정상 읽힘 - 내 사전 예측("64 B base가 깨진다")은 틀렸다. 그래도 게이트는 문서화된 64 B/16 B 규칙을 유지한다
(스펙 + 더 이른 B580 phi-4 실험). 따라서 **base 규칙은 호스트 테스트 `sdpa_block2d_gate`로만 지켜지며 디바이스 테스트는 pitch만 증명한다.**
잔여(호스트로 해결 불가): 동적 K/V padding의 stride %16 != 0 또는 시작 %4 != 0 -> 오답. 재현: `--gtest_also_run_disabled_tests --gtest_filter=*paged_attention_feature_pad_residual_test*` (4 FAIL).
증거: pset rT8->rT7 정확히 105개 config의 `USE_2D_BLOCK_IO_Q 1->0` (pa-mixed 75 + pa-prefill 30, 모두 DYNAMIC_INPUT_PAD), 그 외 변화 0, `.inc` 동일(host-only).

**데이터 함정(MEASURED)**: PA harness 데이터 N(0,0.1)에서는 q.k ~0.08이라 softmax가 거의 균일하여 **Q/K 오독이 가려진다** (V 오독만 보임). `logit_scale_gain`(상수 scale x128,
sharp softmax)을 추가해야 Q misread가 드러났다 (case 47: 0.068 vs 비-sharp 13: 0.003). 그래도 항상은 아니다(case 30: 0.0021). 상세는 `05-methodology-and-pitfalls.md`.

### 3.5 pitch 단위 함정 (head 크기와 무관한 K 페이지 규칙)

K 페이지가 d-major `[head_size, block_size]`이면 row = `block_size` 토큰 = 16 * 2 = **32 B**로 폭 64 B 미만, `pitch >= width` 규칙도 위반한다. row 길이가 head_size가 아니라
block_size이므로 어떤 head 크기도 이를 피하지 못한다 (`pa-k-cache-layout-d-major`). 이것이 §9의 token-major relayout의 근본 이유다.

## 4. 8b transform은 32행 전용, transpose는 32b 전용

### 4.1 사실

- 8b VNNI transform read는 bmg에서 **32행 형태만** 존재한다 (§1 표). K/V 페이지는 16 토큰이므로 16행 호출을 못 한다.
- 결과: 기본 head-128 설정(`kq_sg_tile_keys=16`, cp step=16)에서 K·Q 읽기와 S·V 읽기가 32 key를 가져오지만 앞 16개(uint 0..3)만 소비한다 -> **행 50% over-read**(디코드되지만 버려짐).
- **그런데 read COUNT를 줄이는 시도는 시간을 줄이지 못했다**(§12 표: cp-pair 재사용 3회 모두 실패). 병목은 dequant mov 폭풍이었고 이는 bias trick으로 해결됐다.
  따라서 "over-read 제거"는 (a) 16행이면서 VNNI 정렬 + shuffle-free 경로가 존재하고 (b) 실제로 device time을 줄일 때만 의미 있다 - **둘 다 미증명**.
- micro는 **over-read가 0**이다 (gemmstone 소스 추적, `sdpa-ocl-8b-transform-32row-min`): nGEN이 block2D descriptor의 높이를 직접 프로그램하므로 OpenCL builtin의 고정 32행 지오메트리에 묶이지 않는다.
  micro의 A operand는 `problem.A.layout`으로 결정: T -> Block2DTranspose(K, `ka_load=16`, s8을 u32로 재해석한 d32 transposed read), N -> Block2DVNNI(V, crosspack=4).
  OpenCL에서는 descriptor 제어가 노출되지 않아 이를 동일하게 흉내 낼 수 없다.

### 4.2 transpose가 32b 전용일 때: i8 surface를 dword로 본다

`read_transpose_*`는 32b만 있다. i8 K 페이지를 transpose-read 하려면 **같은 `transpose_32b_16r8x1c` 호출에 byte 단위 width/pitch를 주면 된다**
(4 byte = 연속 4 요소가 한 dword). pitch가 16 B 배수이면 합법이다. 8 dword = 32 byte = head dim 32개 = **DPAS 타일 2개**, 그래서 K block 메시지가 절반(head 128: 16 -> 8)
(`sdpa-ocl-decode-kernel`; ISA 계수). `x = r*8` dword 좌표가 f16/i8/u4 모두 동일해 K read와 K prefetch가 한 호출 지점을 공유한다.

```c
// sdpa_ocl_decode.cl:547 - 모든 정밀도 공통 (f16: 8 dword=16 half, i8: 32 byte, u4: 64 nibble)
intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void*)(key_cache + k_page_off[g]),
        K_ROW_BYTES, PAGED_ATTENTION_BLOCK_SIZE, K_ROW_BYTES, (int2)(r * 8, 0), (private uint*)&kt);
```

### 4.3 u4와 Kc: 채널 쌍을 dword로 (그리고 DPAS depth 순열)

u4 K 페이지의 byte 하나가 채널 쌍(2b, 2b+1)이라 8b transform read는 lane L에게 채널 (2L, 2L+1)을 주고 DPAS가 원하는 연속 `base+L`을 못 준다. depth는 축약 축이므로 A와 B에 같은
순열을 쓰면 결과가 보존된다 (`PA_K_U4_CHANNEL(db,L) = win + 2L + (db&1)`; `docs/sdpa_ocl.md` "u4 K: the permuted depth axis"). Kc(raw f16 current token)를 같은 순서로 읽기 위해
**Kc를 DWORD surface로** 본다: `read_32b_8r16x1c`, x는 dword (`sdpa_ocl_qk_load.cl:466`). 각 window를 parity마다 두 번 읽으나(L1 hit) 페이지 dequant보다 약 4x 적은 명령.
`kc_dword_ok = ((KcD_x0 & 1) == 0)`로 런타임 검사하고 실패 시 per-lane gather로 폴백한다(`sdpa_ocl.cl:284`). 상세는 `03-numerics-softmax-quantization.md` (dequant).

### 4.4 d-major i8 페이지를 재해석 surface로 block read 하는 안 (NOT hw-verified)

2026-08-12에 BY_CHANNEL을 검토하다 사용자가 token-major를 택해서 버린 안이지만 "d-major는 block read 불가"라는 단정을 반박하므로 기록한다. 페이지는 연속이므로 **다른 surface**를 선언할 수 있다:
BY_CHANNEL `[k_head_size cols][block_size+4]`(열당 20 B)에서 width = pitch = 80 B(4열), height = `k_head_size/4`로 두면 재해석된 행 r이 head dim 4r..4r+3이고
`transform_8b_32r16x1c`를 `coord.x = 20*c`에 걸면 head dim 4r+c의 데이터 16 byte만 읽는다(inline comp byte 16..19는 창 밖). spec 규칙 전부 만족(width 80, pitch %16, x=20c %4, base %64).
**함정**: depth 축이 순열된다(slot i = head dim 4i+c) -> Q와 per-channel scale/zp도 같은 순서로 읽어야 한다. **재해석 surface 읽기 자체는 하드웨어에서 돌려본 적이 없다**; 소작은 프로브가 필요하다.

## 5. 클래식 block read, 1D page read, local block IO

### 5.1 클래식 subgroup block read

- `intel_sub_group_block_read_us8`, `intel_sub_group_block_read8` (SLM `S_slm`의 pA, `sdpa_ocl.cl:916-920`), `intel_sub_group_block_read(const global uint*)` (SG8 K tile, per-channel comp 로드 `sdpa_ocl_qk_load.cl:181`).
- 주소 4 B 정렬 필수. **DG2 측정 UB**: 홀수 row pitch에서 `block_read uint`는 오답(err 49), 2 B 오프셋의 `block_read_us`도 오답 (`sdpa-ocl-xe-hpg-facts`, S2). 연속 K reader는 "pitch 짝수 + 4 B 정렬"일 때만 block read, 아니면 폴백.
  실제 코드: `k_tile_dword`의 `dword_ok && whole_tile` (`sdpa_ocl_qk_load.cl:479-522`), 폴백에 `addr & ~3`으로 정렬된 dword 로드 후 시프트. (`ushort` 2개 로드 폴백은 DG2에서 query block 1이 틀리게 컴파일된 이력이 있다.)

### 5.2 u4 head-64 1D whole-page read (`intel_sub_group_block_read_uc16`)

문제: u4 head 64의 K row = 32 B, V row = `Align(32,16)` = 32 B로 둘 다 block2d 64 B 최소 미달 -> K와 V가 모두 per-lane byte gather. k0당 **K 64 + V 128 = 192 `load.ugm.d8u32` vs 16 dpas**(ISA),
+ 5952 B spill. gpt-oss-20b MIXED가 micro 대비 3.67x 느렸다 (원래 보고된 sink 원인이 아님: 13/3529 명령).

해결: 페이지의 데이터 영역은 `16 * ROW` byte 연속이고, `block_read_uc16`은 component i를 lane L의 byte `SUBGROUP_SIZE*i + L`에 놓는다. `COLS = ROW/16`일 때

```
PA_PAGE_UC16 * r + i == t * COLS + c          (r = read index, i = component, t = token, c = 16 B 열 그룹)
```

shuffle도 per-lane 주소도 없다 (`sdpa_ocl_config.cl:340-362`, 매크로 `PA_PAGE_R/I`, 읽기 `sdpa_ocl_qk_load.cl:236`, `sdpa_ocl_v_load.cl:291`).

- `r`은 `COLS`가 2의 거듭제곱이라 c가 런타임이어도 컴파일 타임 상수. `i`는 아니므로 **V는 base를 `SUBGROUP_SIZE*c`만큼 편향하고 c=0을 넘긴다**(런타임 `i`는 `uchar16` subscript를 indirect register addressing으로 바꾼다).
  K의 c는 `db>>1`(상수)라 plain page base에서 읽고 read 2개가 `db` 루프 밖으로 hoist된다(그 hoist가 이득 대부분: 64 -> 2).
- ISA(bmg): `uc16` -> `load.ugm.d32x64t.a64` 1개(256 B), `uc8` -> `d32x32t`. `__local`에서도 동작(`load.slm.d32x64t`). 마지막 접근 byte는 head-64 u4 V에서 527 B (페이지 576 B) 이내.
- 가드 의도적 제거: 키 `>= k`는 점수에 -INF가 더해지므로 로드는 **유한성**만 보장하면 된다(u4 nibble은 [0,15]). per-key `key < k` 제거로 cmp 240 -> 39. `head < d`는 scale에 접어 넣음.
- 정적 결과(head 64): 명령 5733 -> **3516**, d8u32 gather 192 -> 0, spill 138 -> 0. 이 경로는 block2d가 꺼진 곳(u4 head 32/64, V만 head 48)에서만 켜진다. 완화하면 u4는 strict `%64`를 유지: 1D 읽기가 "block2d off"에 게이트되어 있어서 `%16`으로 풀면 이득이 입증된 3.92x 경로를 대체해 버린다.
- **MEASURED(B70, gpt-oss-20b whole run)**: 1474574702 -> 632343047 (+1D) -> 397934594 (+`KQ_TILE_QUERIES=32`, 이미 micro 401436373보다 빠름) -> **376221254** (+V uniform-shift nibble) = **3.92x**, micro 대비 6.3% 빠름 (`sdpa-ocl-u4-head64-page-read`).
  주의: 총 subgroup 수 약 4096이 sweet spot (8192: 632 ms, 4096: 398 ms, 2048: 428 ms)이고 "메시지 수 최소"(h 구성)가 오히려 졌다 -> 정적 메시지 수는 이 커널에서 틀린 지표.
- V 4x 중복 dequant를 SLM으로 공유하는 안(약 -24% ops)은 SLM +16 KB -> WG 4 -> 3(-25% 점유율)이라 **폐기**(블라인드 베팅). 점유율은 이미 두 번 틀렸다.
- 미적용: `sdpa_ocl_decode`도 같은 결함(gpt-oss dump에서 128 `load.ugm.d8u32`). uc16 매핑이 그대로 적용되지만 decode 갭이 측정된 적 없어 범위 밖.

### 5.3 local(SLM) block IO와 xe_hpg(DG2)

DG2 S0/S2(MEASURED, 컴파일+ISA + 실기 36/36 mini-SDPA): local uint/ushort block IO는 **pragma 없이 성공**, global uint/ushort block IO 성공, `S_slm`의 pA를
`as_int8(intel_sub_group_block_read8((local uint*)&S_slm[...]))`로 읽는 SG8 arm이 동작한다 (`sdpa_ocl.cl:911-914`). DG2는 2D block IO와 subgroup buffer prefetch가 모두 거부되며 generic `prefetch()`는
null-destination `load.ugm` 1개를 추가하지만 효과는 미측정이다. DG2에서 SG16 `short8` DPAS operand는 **오류 없이 DPAS를 버린다**(ISA에 dpas 없음): 조용한 오답 위험이므로 `hpg` 서브커맨드에서 `dpas` 열이 0이면 실패로 본다.

## 6. SLM 사용 vs 미사용

| 항목 | 내용 | 근거 |
|---|---|---|
| sdpa_ocl SLM 구성 | `Q_slm`(Q를 DPAS B operand로 staging), `S_slm`(softmax 결과를 pA로 되먹임), `S_sum_slm`, `S_max_slm`(local fmax atomic) | `sdpa_ocl.cl:301-304` |
| micro SLM | wrapper의 Q/S staging이 전부 (107,008 B vs ocl 17,536 B, 6.1x). **spliced GEMM(`ugemm_kq/vs_slm_size`)은 SLM 0**: K/V를 global에서 곧바로 systolic array로 흘림(prefetch만) | `sdpa-ocl-slm-vs-micro` (ISA/소스 공식, MEASURED diff 0) |
| 차이 원인 | 타일 크기 둘뿐: query tile 128 vs 32, key tile 256 vs 128. 구현 방식 차이가 아님 | 동일 노트 |
| 점유율 | Xe2 Xe-core SLM 128 KB. micro 1 WG/Xe-core(82%), ocl 7 WG. 그래도 micro는 causal 효율(80% vs 50%) + 256GRF에서 이김 -> **점유율 헤드룸이 곧 처리량이 아님** | 동일 노트 |
| decode의 S*V | head-dim 분할로 `slm_out`(16,640 B) -> `slm_p`(2,304 B), -86%: Xe-core 점유율 7/8 WG 상한 해소. 이 변경 자체의 시간 효과는 명령/SLM 계수로만 보였고(1290->1202 instCount) **단독 시간 측정 없음** | `sdpa-ocl-decode-kernel` |
| `slm_p` 레이아웃 | key-indexed + head가 innermost vector 원소여야 한다. head-major로 쓰면 64개의 별도 32 B SLM 읽기(SLM load 44->72, instCount +12%) | 동일 |
| ISA 계수 지표 | 한번 첫 버전이 1290->1440으로 **더 느렸다** (slm_p layout, 페이지 lookup 18 scalar load) | 동일 |
| SLM 제외 | 4개 서브그룹 이상이 key 축을 나누면 `slm_out` + barrier가 추가됨: `V_TILES < SG_PER_WG`(head < 128)이면 sg16이 중립~손해, head 256/512(V_TILES 16/32)는 무료 | `sdpa-ocl-decode-tiling-sg-per-wg` (MEASURED B70) |
| SLM 공유 dequant | u4 V 4x 중복 dequant를 SLM으로 공유: +16 KB, WG 4->3, 폐기 | §5.2 |

SLM 전치(DG2 S2 K3)는 K 방향 후보 중 최악(D=128 t_med: K0 연속 286 us, K1 scalar gather 365 us 채택, K2 vload8+pack 370, **K3 SLM 전치 449**, K4 A=Q/B=K 344) (MEASURED, DG2, `sdpa-ocl-xe-hpg-facts`).

## 7. Software prefetch

### 7.1 사용처 (현재 코드)

| 위치 | 무엇 | 조건 |
|---|---|---|
| decode V | `PREFETCH_V_TILE`: `intel_sub_group_2d_block_prefetch_16b_16r16x1c` / i8·u4는 `_8b_32r16x1c` (`sdpa_ocl_decode.cl:177-197`), `PREFETCH_DIST`(기본 4, `SDPA_OCL_DECODE_PREFETCH`, `sdpa_gen_ocl_decode.cpp:256`) 청크 앞서 | `USE_PREFETCH_V && USE_2D_BLOCK_IO_V` |
| MIXED Vc | `vc_prefetch` (`sdpa_ocl_v_load.cl:21-40`): S_max 집계 barrier **직후**, softmax qb 루프 앞. `!from_cache && sg_i_sv == 0`만 발행 | `IS_PA_K_U4 && PA_CUR_KV_F16` (u4 전용) |
| K | **없음** (decode K prefetch는 삭제됨) | |
| 마스크 | 없음 (micro는 `PREFETCH_MASK` 있음) | |

주의: 2D block prefetch는 목적지 레지스터가 필요 없어서 "점유율로는 살 수 없는 메모리 병렬성(MLP)"을 준다(`sdpa_ocl_decode.cl:177-182` 주석). 각 prefetch는 a64 주소/descriptor 세팅 약 8 명령(32개로 instCount 1202 -> 1469, 대부분 mov).
8b prefetch builtin은 존재하므로(§1) i8/u4에서도 2D prefetch를 유지한다. 호출 지점은 실제 읽기와 **같은 surface/base fixup/좌표**를 써야 한다 (`vc_prefetch`의 `Vc_b2d`, `VcD_w_b2d`, `VcD_x0`).

### 7.2 측정 결과 (decode, llama-3.1-8b, head 128, M=4, `sdpa-ocl-decode-kernel` 2026-08-10; HW는 노트에 미기재, B70 이전 시점)

| 실험 | 결과 | 해석 |
|---|---|---|
| K prefetch (`prefetch_32b_16r8x1c`) | **3.6% 느림** | KQ 루프가 단일 4 KB 페이지 위에서 `KEY_GROUPS` 독립 DPAS 체인을 이미 인터리브 -> 로드가 이미 파이프라인됨 |
| V prefetch | **2.1% 빠름** (기본 ON) | S*V가 16개 다른 페이지를 단일 accumulator 체인으로 걷기 때문에 로드가 노출되어 있었음 |
| K+V 둘 다 | +0.8% | K 손실이 V 이득을 삼킴 |
| 거리 | 1: -1.45%, 2: -1.87%, **4: -2.10%**, 8: -0.56%, 16: **+2.21%** (비단조) | 거리는 **어디서 발행하느냐만** 바꾸고 개수는 안 바꿈: `min(dist, chunks)`개는 barrier 창, 나머지는 한 iteration 앞. 거리 > ~4이면 in-flight 집합(8 KB/subgroup x 8)이 L1을 넘겨 쓰기 전에 evict |
| "전부 barrier 창에 밀어 넣기" | **틀렸다** | 위 비단조 결과 |
| 절대 시간 | prefetch off 1.1194e9 ns vs 거리 1/2/4: 1.1031e9 / 1.0984e9 / 1.0958e9 | `docs/sdpa_ocl.md` "V is prefetched ..." |
| barrier 창의 split arrive/wait가 차지한 비중 | 2.1%p 중 0.7%p | §8 |

### 7.3 MIXED Vc prefetch (u4, llama-3.2-1b-instruct, B70, cliloader 평균 ns)

V_PREFETCH on 145,603 ns vs off 151,866 ns = **-4.1240%** (64 call, `sdpa_ocl_mixed_..._sa`, SIMD16 REG128 SLM=17664, GWS[272x512x1], LWS[16x16x1]); micro 147,825 ns;
최초 정확 baseline 158,661 ns 대비 총 **-8.23%** (`sdpa-ocl-mixed-exact-complete`, MEASURED). **i8 BY_CHANNEL MIXED에서는 측정된 적 없음**(u4 전용 `#if`).

### 7.4 실패/무효 prefetch

- plain prefill에 V prefetch를 k0 iteration 안에서 곧바로 같은 타일 소비 직전에 발행: **약 11 us 순손해**(소비 직전이라 숨길 계산이 없는 순수 오버헤드). 코드 제거(`ddd9ab8ef4` 직전 `#if 0` 블록, `sdpa_ocl.cl` 이력).
- `MAX_BARRIER_V_PREFETCH`(plain prefill의 split barrier 창 안에서 V prefetch)는 `ddd9ab8ef4`에서 토글로 들어가 `SDPA_OCL_256GRF=1 ... MAX_BARRIER_V_PREFETCH=1`로 llama-2-7b 4096-token 벤치(`test/run.sh`)에 쓰였지만
  **이득이 입증된 적 없고** 2026-09 리팩토링에서 삭제됐다(`sdpa-ocl-refactor-2026-09`: "never shown to help"). 개별 측정 결과 파일은 찾지 못했다 -> 효과는 "미입증"으로만 기록한다.
- micro는 `PREFETCH_K/V/MASK`를 한 k0 iteration 앞서 L1+L3(`LSC_LDCC_L1C_L3C`, `cooperative_prefetch_2d_*`)로 쓴다. sdpa_ocl은 이 방식을 그대로 이식하지 않았다(Step 3로 계획만 존재: `sdpa-ocl-block2d-gate-relaxation`).
- DG2: generic `prefetch()` 효과 없음(MEASURED, S2 P8: 기록만).

## 8. Split barrier (`intel_work_group_barrier_arrive/wait`)

실제 코드 위치(현재 working tree):

| 위치 | 코드 | 의도 |
|---|---|---|
| `sdpa_ocl.cl:766` | `barrier(CLK_LOCAL_MEM_FENCE)` | `S_max_slm` atomic max 집계 완료. 이후 `vc_prefetch`(`:770`) |
| `sdpa_ocl.cl:842` `intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE)` ... `:868` `intel_work_group_barrier_wait(...)` | softmax의 `S_slm` 쓰기/`S_sum_slm` 이후 arrive, **그 사이에 `alpha`로 A_tile을 rescale**(`!first`일 때) 하고 wait | S_slm을 모든 subgroup이 쓴 뒤에야 pA를 읽도록 하는 동기화를 "A_tile rescale" 계산과 겹침 |
| `sdpa_ocl_decode.cl:727-732` | arrive -> `PREFETCH_V_TILE` x `min(PREFETCH_DIST, CHUNKS)` -> wait | slm_p/slm_sum 쓰기 후 barrier 대기 시간을 V 첫 타일 prefetch로 채움 (`:721-722` 주석) |
| `sdpa_ocl_decode.cl:734` | `#else barrier(...)` (prefetch 꺼짐) | |

규칙(코드 주석과 노트 근거):
- arrive와 wait 사이에는 **그 barrier가 보호하는 SLM 데이터를 읽지 않는** 독립 작업만 둔다 (alpha rescale은 private A_tile, prefetch는 SLM 무관).
- 분할 barrier 안에서 **barrier 자체의 개수와 배치를 바꾸지 않는다**. MIXED의 `SV_TRIM`은 `cp*DPAS_K >= k_chunk` 블록을 `continue`로 건너뛰되 barrier는 루프 밖에 둔다(`sdpa_ocl.cl:891`, "the barriers stay outside").
- local fmax atomic(`__builtin_IB_atomic_max_local_f32`)과 split barrier는 DG2 SG8에서도 실기 PASS(S2 미니 SDPA 36/36). `cl_intel_split_work_group_barrier`.
- 측정된 효과: decode V prefetch 2.1% 중 **0.7%p**가 barrier 창 발행분(MEASURED, `sdpa-ocl-decode-kernel`). plain prefill의 split barrier **단독** 효과를 분리 측정한 기록은 찾지 못했다 -> ASSUMED로 둔다. MIXED V_PREFETCH 4.12%는 barrier 직후 발행이며 barrier 자체는 변경하지 않음("새 barrier를 만들지 않았다").
- 그 외 이력: tk=32 정확성 버그 조사에서 split barrier 주변 컴파일러 순서 차이를 용의 목록에 올렸으나 입증되지 않음(`sdpa-ocl-tk32-bug-hunt`, UNRESOLVED).

## 9. Scalar gather와 K 캐시 레이아웃 (d-major -> token-major)

### 9.1 왜 K가 SIMD-1 scalar gather였나

PA K 캐시(원본)는 `[num_blocks, kv_heads, k_head_size, block_size]`(**token이 innermost, d-major**), V는 `[.., block_size, v_head_size]`(**d innermost, token-major**)이다
(`paged_attention_gpu_test.h`, `paged_attention_opt.cl:105-106`, `plugin/ops/paged_attention.cpp:45-46`; 2026-08-05 검증). KQ의 A operand는 lane=head(d), 요소=key를 원한다.
- V(token-major)는 `transform_16b_16r16x1c`가 페이지에 곧바로 적용된다.
- K(d-major)는 coalesced read가 lane=token(틀린 축)을 준다 -> **per-key SIMD-1 scalar gather**: head 128 mixed ocloc 총 op 2651, K scalar gather 128개.
- 빠져나갈 길이 모두 막혀 있다: transpose는 32b 전용(f16 두 개를 묶으면 두 *토큰*이 한 dword), pitch는 32 B(< 64 B), operand swap(A=Q, B=K)은 micro가 하는 방식이지만 S^T가 되어 per-query softmax max/sum이 free in-register에서 cross-lane `sub_group_reduce`로 바뀐다.
- **해결**: K를 V처럼 token-major로 저장 `[num_blocks, kv_heads, block_size, k_head_size]`. 그러면 K 페이지가 prefill 지오메트리와 동일 -> `read_16b_16r16x1c`를 `(x=db*DPAS_K, y=0)`, pitch `HEAD_SIZE*2`로 호출(`sdpa_ocl_qk_load.cl:330`).
  (ISA, head 128 mixed) 총 op 2651 -> 1353 (-49%), K scalar gather 128 -> **0**, dpas 32 불변. head 64 mixed instCount 2322 -> 1288 (-45%), `load.ugm.d16u32` 64 -> 0.

### 9.2 relayout 사다리 (2026-08-05..06, `pa-k-token-major-progress`)

| rung | 내용 | 검증 |
|---|---|---|
| 1 | dim order(`keyCacheDimOrder` `transformations_pipeline.cpp`), writer(`pa_kv_cache_update_ref.cl` KEY_TOKEN_STRIDE/KEY_HIDDEN_STRIDE), decode reader | toggle off 129/129, on 124/129(MIXED 5개 기대 실패) |
| 2 | sdpa_ocl MIXED K를 페이지 위 `read_16b_16r16x1c`로 (`USE_2D_BLOCK_IO_K_PA`) | 3 경로 모두 129/129, cliloader로 실제 `sdpa_ocl__generate` 확인 |
| 3 | rotate | **실제 버그 발견**: d-major 커널이 token-major 캐시에서 *다른 토큰*끼리 짝지어 512 쌍 전부 오염. 기본 suite 59 케이스는 모두 통과(blind). 캐시를 되읽는 `smoke_kv_cache_rotation_content`(4) 추가 후 4/4 FAIL -> PASS |
| 4 | adaptive_rkv, reorder | adaptive_rkv 15 FAIL -> 24/24; `pa_kv_reorder_gpu`는 **테스트가 같은 d-major 가정을 공유해 blind**(8/8 통과), 커널과 테스트를 동시에 수정 |

교훈: **green suite는 소비자가 레이아웃에 맞다는 증거가 아니다.** 해당 커널이 쓰는 값을 실제로 관찰하는 테스트가 있는지 확인한다.
`IS_KEY_TOKEN_MAJOR`는 5개 커널(decode, writer, rotate, adaptive_rkv, reorder)에 닿는다. 단일 origin: `keyCacheDimOrder`; XAttention/CM은 이미 token-major.

`pa_sdpa_opt`(decode)는 token-major에서 `vload16(0, key_cache + block_offset + sglid*K_HEAD_SIZE + ...)`로 "각 lane이 자기 토큰 행"을 읽는다. 정확하고 **빠르지는 않다**(사용자 실측: 변화 없음~약간 느림).
이유 (ISA): d-major의 `load.ugm.d32x8t (1|M0)`은 **block load**(SIMD-1은 주소 하나가 32 B 연속 데이터를 lane들에 분산, 이미 최적)였는데 이를 "scalar gather로 강등됨"으로 **오독**했다(한 세션에서 두 번).
`vload16`은 `load.ugm.d32 (16|M0)` 16개 서로 다른 주소(lane당 256 B 간격)의 **scatter**: qk_idx당 d-major 16 block 메시지(~8 cache line) vs token-major 8 scatter(16 line) -> 명령 수는 줄고 traffic은 늘었다.
**지표 교훈: 접근 패턴이 바뀔 때 instCount/메시지 수는 틀린 지표**(1076->980, load 178->114로 좋아졌지만 wall-clock 불변). 메시지 *종류*(block vs scatter vs transform)와 touched line 수를 비교한다.
그 위에 `transpose_32b_16r8x1c`를 쓰면 8 scatter를 1 block으로 줄일 수 있으나 **`pa_sdpa_opt`가 pre-Xe2에서도 돌아야 해서 막혔다**(사용자 결정 2026-08-06): arch gate + 비 2D 폴백이 필요.

### 9.3 페이지 레이아웃 표 (현재 `docs/sdpa_ocl.md` "Paged-attention cache layouts", h = head size, block 16)

| 캐시 | 데이터 row pitch | comp 영역 | K 페이지 stride |
|---|---|---|---|
| f16 | `2h` | 없음 | `32h` |
| i8 BY_TOKEN | `h` (**`h+4`가 아님**: +4는 페이지 끝의 token당 f16 scale/zp 배열 공간) | token당 scale `[t]`, zp `[16+t]` | `16*(h+4)` |
| i8 BY_CHANNEL (token-major) | `h` | 채널당 (scale, zp) f16 pair, dword 하나 `[c]` | `20h` |
| u4 BY_CHANNEL K (token-major) | `h/2` (**정렬 안 함**: `16*(h/2)+4h = 12h`가 upstream d-major INT4 페이지 할당에 byte 단위로 맞음) | i8 BY_CHANNEL과 동일 | `12h` |
| u4 BY_TOKEN V | `Align(h/2, 16)` (trailing comp slack이 흡수: `16*PV+64 == 16*(PV+4)`) | i8 BY_TOKEN과 동일 | `16*(Align(h/2,16)+4)` |

- 페이지 크기는 d-major <-> token-major 전환에서 **불변**(`k_head_size*(block_size+4)`) -> 할당/풀 크기 무변경, 페이지 내부 주소만 이동.
- **K는 BY_TOKEN이 token-major가 되어도 comp는 token 축에 있고, BY_CHANNEL은 token-major에서 comp가 채널 축**(scale/zp가 d에 의존 -> Q에 접히는 KQ depth 축과 일치, 페이지마다 A operand가 달라져 `AS_A(qv)` 재구성이 g 루프 안으로 들어가는 구조적 비용: +Q_PER_WG half 곱/타일).
- BY_CHANNEL comp는 `4*k_head_size` = 512 B/page(head 128)로 BY_TOKEN의 64 B보다 크다: K 페이지 2560 B vs 2112 B (+21%) - 포맷에 내재(K+V ctx 4352, 8 kv-head: 10.2 MB vs 9.2 MB, f16 17.8 MB). 커널로 못 줄임.
- 소비자가 4개뿐(writer, rotate, sdpa_ocl MIXED, sdpa_ocl_decode)이며 나머지는 d-major 가정이다. 레이아웃은 모델당 한 번 `transformations_pipeline.cpp`에서 결정하고 소비자는 **캐시의 물리 shape에서 파생**한다
  (`k_by_channel_token_major_layout(shape, adjusted_block)`; dim 2가 adjusted block이면 token-major). 불일치 방향 한쪽은 `PagedAttentionOptImpl::update_rt_params()`가 throw, `pa_kv_reorder`가 거부한다
  (`pa-mixed-layout-gate-use-ocl-hole`: MIXED K-page 검사가 `use_ocl`에 게이트되어 `TEST_USE_SDPA_OCL=0`에서 micro-MIXED가 token-major 페이지를 읽던 구멍 -> `96b457a4f0`).
- 사실 3개(재도출 금지, `sdpa-ocl-pa-i8-by-token-task`): (1) `OV_GPU_PA_K_TOKEN_MAJOR=1`은 i8 캐시에 no-op였던 것은 *우리 코드의* 범위 제한(`aaf3ca3ae2`)이었다. (2) i8 BY_TOKEN의 row pitch는 `h`다.
  `h+4`로 오해해 "132 B pitch -> 2D block 불가"라는 틀린 결론이 한 설계 옵션을 죽였다. (3) PREFILL은 K/V 캐시를 읽지 않고 f16 입력 텐서를 읽는다(`sdpa_gen_ocl.cpp:594`).

### 9.4 per-key scale/zp 로드도 gather였다

plain i8 prefill에서 128개의 SIMD-1 `(1|M0)` 로드의 정체는 K 데이터가 아니라 **per-key scale/zp** (key 단위 값이 innermost에서 읽히면 lane-uniform이라 IGC가 lane당 한 메시지, 15/16 lane 유휴)였다.
수정: k0 타일 상단에서 subgroup 협력 wide load(lane=key) 한 번 + `sub_group_broadcast`(lane이 컴파일 타임 상수라 region에 fold). PA MIXED i8에서도 k0당 256 -> 0 (head 128, instCount 5450 -> 4868, ISA).
`block_read`로 바꾸는 안은 이미 SIMD16 gather 1개 메시지로 coalesce되어 있어 이득 ≈ 0이라 기각(ISA).
`block_indices[]` 페이지 lookup도 같은 방식으로 hoist: `DKS*kq_key_blocks*DPAS_ROWS`회(head 64: 64) -> `kq_key_blocks`회. decode는 한 partition이 정확히 `SUBGROUP_SIZE`개 페이지를 덮어서 페이지 테이블을 lane당 한 번에 읽고 `sub_group_broadcast`(global load 53 -> 36, 18개 scalar load 회피).

## 10. 쓰기 경로: kvup writer와 store sector floor

### 10.1 u4 token-major writer 회귀와 수정 (`pa-kvup-u4-token-major-writer`, 2026-08-13)

token-major는 requantize 루프가 걷는 "채널의 토큰 벡터"의 연속성을 깬다. d-major는 16 nibble이 8 연속 byte라 IGC가 읽기와 쓰기를 wide 메시지 하나로 합친다(2 토큰/trip). token-major는 `K_HEAD_SIZE/2` 간격이라 합쳐지지 않고,
byte가 채널 쌍이라 채널-per-lane은 `intel_sub_group_shuffle` + divergent `if (!hi_nibble)`가 토큰마다 필요했다. 토큰당 명령 기울기: d-major 18.3 vs token-major 40.3.
수정: lane이 **byte와 token parity**를 소유(`col = sglid % (SG/2)`, `par = sglid / (SG/2)`, `pa_kv_cache_update_ref.cl:391-392`), min/max는 butterfly(`sglid ^ half_sg`) 1회(fmax/fmin 교환/결합 법칙이라 **bit-identical**).
**세 가지 함정(각각 측정)**: (1) lane에 따라 변하는 루프 변수가 배열의 register를 빼앗는다(`for (t = par; ...; t += 2)`로 `r[a0]` 2 -> 274, **1.8x 느림**) -> uniform `it/npo/np` 사용. (2) uniform 가드의 **모양**이 안쪽 루프 unroll 여부를 정한다:
loop bound를 0으로: unroll 소실(7850 -> 2590 static), 인라인 helper 맨 위 `if (skip) return;`: unroll 소실 + **18.7 us(분할 안 한 8.2 us보다 2.3x 느림)**, **call site**에서 같은 검사: unroll 유지 + 5.6 us. 가드는 call site에, helper 본문이나 loop bound가 아니라.
(3) 타입 승격을 식 단위로 보존(`(max==min) ? 0.004 : (max-min)`는 half에서 뺄셈 후 widen).
V를 별도 WG에: V는 BY_TOKEN이라 head dim 전체 reduce를 한 WG가 소유해야 한다. **prefill에서만** 이득(prefill WG가 K 전 채널도 하므로 max(K,V)): ablation에서 generate는 floor 580, V-only 895, K-only 2755, K+V 2777 ns -> V가 단독 315 ns인데 in-situ 22 ns(K의 긴 dependent chain의 idle issue slot이 흡수).
`HAS_SEPARATE_V_WG_PREFILL`은 prefill만 건드리고 `get_global_id(2)` -> `get_local_id(2)`이 **필수**(gws[2]가 두 WG에 걸치므로).
측정(probe B70 head 128, 8 kv head, ns/dispatch): generate u4 d-major 2904 / token-major(회귀) 3427 / **fixed 2776**, prefill(2 blk) 29444 / 29920 / **14324**.
**real-model(llama-3.1-8b, cliloader, debug build)**: generate `[1x8x128]` 31620 call 192,259,083(d-major) / 216,920,855(회귀) / **192,274,023(fixed)** -> **회귀 완전 제거, d-major와 정확히 동률**(probe가 예측한 -4.4%는 실현되지 않음: 실제 run은 probe가 없는 ~3200 ns/dispatch를 더 가져 희석).
prefill 1984 call 74,837,410 -> 76,483,353 -> **36,547,697**(-52.2%, probe -52.1%와 일치, `GWS[16x8x32]`가 V WG 추가 dispatch의 증거). 총합 275,999,683 / 303,455,125 / **237,128,428** (-21.9% vs 회귀, -14.1% vs upstream d-major).
프로브는 **큰 비율은 잘 예측하고 작은 비율은 못 한다**(절대 시간은 generate에서 약 2.2x 과소보고).

### 10.2 store-sector floor (`pa-kvup-token-major-store-sector-floor`, 2026-08-18, B70)

남은 회귀: head-256/kv-8 generate dispatch(`GWS[1x8x256] LWS[1x1x16]`)가 6312 -> 7074 ns (**+12.1%**); head-512는 그대로(8099 -> 7956). 원인(ablation): thread 하나가 **8 byte x 16 토큰 행**을 소유하고 행 간격이 `BC_TOKEN_STRIDE = K/2 = 128 B`이므로
**16개의 서로 다른 128 B sector를 dirty**한다. d-major의 16 채널은 `16*ADJUSTED_BLOCK = 192` 연속, line-정렬, 단독 소유 byte = **2 sector**.

| ablation (ns/dispatch, head 256) | floor | old-token loads | **data stores** | total |
|---|---|---|---|---|
| d-major | 2201 | 723 | **423** | 3353 |
| token-major | **1283** | 889 | **2014** | 3700 |

store 비용은 dispatch 전체의 sector write 수에 비례(~0.5-0.8 ns/sector): K=128 1024 sector -> 912 ns, K=256 2048 -> 2014, K=512(2 group이라 per-thread 재사용) 512 -> 746.
상한(timing-only 프로브, head 256): sector/thread 16 -> 3698, 8 -> 2851, ~1 -> 2712 -> **기회 총 986 ns**(0.024 ms/token), 16 -> 8이 847을 차지.
**ISA/메시지 수는 틀린 지표였다**: token-major가 명령 수(4249 vs 4448)와 indirect `r[a0]`(125 vs 188)가 모두 더 적고 store 메시지 수가 같은데(50 vs 51) store가 4.8x 느렸다. ablation만이 찾았다.
네 후보 수정은 **모두 측정으로 기각**(§12 표). 결론: 코드 변경 없음, +0.019 ms/token 수용, e2e 갭의 나머지 89%에 노력을 쓴다.
(그 e2e 갭 자체: 벽시계 23.3 s vs device 16.6 s, 즉 wall의 29%가 device 외 시간, enqueue 1,680,102회. 미검증 가설: enqueue당 고정 비용.)

기타: `reqd_work_group_size(1,1,SUBGROUP_SIZE)`(`pa_kv_cache_update_ref.cl:807`)는 `lws[2] != 16`을 CL_INVALID_WORK_GROUP_SIZE(-54)로 거부한다 -> generate lws를 넓히려면 속성을 제거해야 한다(prefill은 16 유지).
i8 BY_CHANNEL token-major는 **모든 h_sub에서** d-major보다 느리다(parts=16 2882 vs 2432, +18.5%; prefill +10%): u4와 다른 원인 - non-int4 arm이 d-major 한 열 `vload16`을 16회 scalar gather로 바꿈(메시지 수 회귀, sector 아님). gemma-4(u4)는 영향 없음. 미수정.

## 11. Dynamic shape, padding, OOB 위험

| 위험 | 사실 | 대응 |
|---|---|---|
| 2D block OOB | HW가 0으로 채운다(width/height 밖). **global scalar OOB read는 Xe2에서 0이 아니라 CL_OUT_OF_RESOURCES 또는 쓰레기** | 2D block의 surface를 유효 영역으로 clamp, scalar는 반드시 가드 |
| 미기록 페이지 slot | NaN이 점수에 마스크 후에도 살아남을 수 있다(`-inf + NaN`) | height clamp(`kp_rows`), V는 scale AND zp를 0으로 강제(`sdpa_ocl_decode.cl` comp 주석, K와 달리 V는 나중에 덮어쓸 곳이 없음) |
| key >= k의 페이지 lookup | page 0은 항상 할당됨 -> `block_indices` 범위 밖은 page 0으로 읽는다 (`pa_v_page_base` `sdpa_ocl_v_load.cl:11`) | 이 키의 확률이 정확히 0이므로 값은 유한하기만 하면 된다 (`1D` 경로가 per-key 가드를 제거한 근거) |
| k0 마지막 타일 | `key_base`가 `k`를 넘을 수 있음 -> height <= 0 비합법 read | zero-fill (`pa_k_tile_b2d16`의 `else` 가지) |
| head tail (`d % DPAS_K != 0`) | surface width를 `d*sizeof(T)`로 두면 HW zero-fill; 다음 head 누수 없음 | `KP_w = d * sizeof(half)` |
| 동적 padding | 크기가 shape_info로 런타임에 옴 | fixup tier는 전제를 신뢰, strict tier는 false (§3.4); 위반 시 오답(잔여 위험) |
| 동적 mask shape | JIT 시점에 trailing dim이 동적이면 `mask_kind = m_is_prefill ? 2 : 1`로 추론된다. 런타임이 `[B,H,1,K]`(per-key)면 kind 2 가정이 1-row 버퍼를 q행만큼 읽어(최대 15x) **CL_OUT_OF_RESOURCES/NaN** (`sdpa_gpu_test_64_32_990_128_2`, 2026-09-17, `test/SDPA_OCL_MASK_KIND_OOB_SESSION.md`) | 커널에서 `MSK_D2 == 1 ? 0 : row`, `MSK_D3 == 1`이면 broadcast, 상한을 `MSK_D2/D3`로(static 마스크는 컴파일 타임 fold되어 바이트 동일 코드젠, 동적만 프롤로그에 compare 1개). `MASK_KIND=-1`로 되돌리는 안은 IGC 회귀라 채택 안 함. micro는 `sdpa_micro.cl:620`의 런타임 `MSK_D2 == 1 && MSK_D3 > 1` 검사로 이미 안전 |
| u4 Kc dword 정렬 | 런타임 padding에 따라 head 첫 채널의 half offset이 홀수일 수 있다 | `kc_dword_ok` 런타임 검사 + exact scalar Kc 폴백(cache로 돌아가면 오답: raw와 cache의 현재 토큰은 수치적으로 다름) |
| PA scores 출력/adaptive RKV/alibi/qq_bias | token-major BY_CHANNEL 페이지를 d-major로 읽음 -> 쓰레기 | 레이아웃 gate `allow = !has_scores_output && !has_adaptive_rkv` (`PA_BY_CHANNEL_TOKEN_MAJOR_SCORES_DECODE_SESSION.md`, 2026-09-21): `/6` 케이스가 `TEST_USE_SDPA_OCL=0`에서도 실패하던 것의 근본 원인은 writer는 token-major로 쓰는데 `paged_attention_opt__single_token`이 `IS_KEY_BY_CHANNEL_TOKEN_MAJOR`를 jit하지 않아 d-major로 읽은 것 |
| i8 BY_CHANNEL requantize | `pa_kv_cache_update_ref.cl:297`이 `in_data_pitch`를 무시 -> padded key를 j>=1 새 토큰에서 오독 | **미수정**, Known issues에 기록(`sdpa-ocl-ki-t7`) |

### 11.1 V page split 버그 (`test/V_PAGE_SPLIT_BUG_HANDOFF.md`, 역사)

MIXED가 `[0,past_len)`은 캐시, `[past_len,k)`는 raw Kc/Vc에서 읽을 때, 소스 전환을 `align_up(past_len, PAGED_ATTENTION_BLOCK_SIZE)`(GRAN=1 구버전, 페이지 단위)에 두면 **K는 정상이고 V만 오답**이었다.
증상 규칙: `pa_key_end`가 `kq_wg_tile_keys`의 배수가 **아닐 때** 한 k0 iteration이 두 소스를 섞는다. K의 `from_cache = (key_base < pa_key_end)`는 KQ key 분할(subgroup별), V의 `v_from_cache = ((k0 + cp*SUBGROUP_SIZE) < pa_key_end)`는 S*V `cp` 분할로 **다른 매핑**이다.
f16 캐시(캐시 읽기와 Vc 읽기가 bit-identical)에서도 실패하므로 양자화 오차가 아니다. 이 버그의 **메커니즘은 끝내 규명되지 않았다**(많은 용의 배제: if/else 중첩, surface descriptor, Vc pointer, base fixup, y>=0, parity/VNNI row pairing, V page block read).
**해결은 원인 규명이 아니라 설계 변경**: `GRAN=1`의 의미를 "past_len에서 정확히 자르는 exact split"으로 바꿨다. `k_chunk`를 `past_len`에서 clip하고 chunk tail mask/`last`를 실제 chunk로 계산, `k0 += k_chunk`로 진행 -> 각 iteration의 소스 선택이 **WG-uniform**(micro PR #37377과 동일).
(2026-09-09, `sdpa-ocl-mixed-exact-complete`; GRAN/SIDE/page-rounded 경로는 2026-09 리팩토링에서 삭제.) 교훈: **한 k0 타일 안에서 소스가 subgroup별로 갈리는 설계는 K와 V의 서로 다른 key 분할 때문에 위험하다 - WG-uniform 경계를 만든다.**
옛 문서의 "GRAN=0 정답" 주장과 실패 개수는 현재 근거가 아니다.

## 12. 실패/기각 실험표

| # | 시도 | 결과 | 원인(또는 가설 표시) | 출처 / HW |
|---|---|---|---|---|
| 1 | int8 K를 `8b_16r16x4c`(head 64, shuffle 없음 가정) + scale 접기 | variant1 5888 / 30,780 ns **scalar보다 느림** (정확성 1024/1024 PASS) | shuffle/추가 작업이 coalescing 이득을 상쇄. spill이 병목이 아님 | `sdpa-ocl-int8-perf`, B580 |
| 2 | i8 V cp-pair read 재사용 (32행 read 한 번으로 cp 두 블록, 8 -> 4 read) 3회(`if(cp&1)` 게이트 / `cp+=2` 명시 unroll / grouped-mad 스케줄) | v1 21,666 -> 22,812 ns **느림**, 재작성 두 개도 무개선, 전부 revert | read count가 병목이 아님; 게이트와 iteration-carried state가 IGC 스케줄링을 흔듦; HW/L2가 버려지는 반을 흡수 | 동일, B580 |
| 3 | K 진단(dequant 제거 K_DIAG=2) | 오히려 **느려짐** (32,916) | scale/zp 로드를 지우면 IGC 스케줄링/regalloc이 변함(observer effect). 로드 vs dequant 비용을 분리 못 함 | 동일 |
| 4 | V transform_8b + V_SCALE_CACHE=0 | 29,687 **scalar보다 느림** | per-byte global scale 재로드가 transform 이득을 삼킴. 둘을 같이 출하해야 함 | 동일 |
| 5 | KQ per-lane `char16 vload16` K gather 대체 (mixed i8, "1b") | instCount 4868 -> 4565이지만 mov 2185 -> 2275, d32 16 -> 48 | "메시지는 줄고 mov가 늘어나는" 형태가 decode에서 졌던 것과 같다 | `sdpa-ocl-pa-i8-by-token-task`, ISA |
| 6 | decode K prefetch | **-3.6%** (느림) | KQ가 이미 독립 DPAS 체인 인터리브 | `sdpa-ocl-decode-kernel` |
| 7 | decode prefetch 거리 16 | +2.21% | in-flight 집합이 L1을 넘김 | 동일 |
| 8 | plain prefill의 소비 직전 V prefetch | 약 11 us 순손해 | 숨길 계산이 없음 | `sdpa_ocl.cl` 이력(`#if 0` 주석), 측정 HW 미기재 |
| 9 | `MAX_BARRIER_V_PREFETCH` | 이득 입증 못 함, 삭제 | 개별 측정 파일 없음 | `sdpa-ocl-refactor-2026-09` |
| 10 | head-72에서 256 GRF | **+47% 느림** (9.57 -> 14.12 ms) | spill 제거해도 resident thread 절반 비용이 더 큼, LSC 메시지 수가 병목 | `sdpa-ocl-block2d-gate-relaxation`, B70 |
| 11 | u4 head-64에서 256 GRF | **+70% 느림** (1474574702 -> 2502030737) | gather-bound 설정은 스레드 수로 지연을 숨김 | `sdpa-ocl-u4-head64-page-read`, B70 |
| 12 | u4 head-64 `PER_WG_KEYS=4`(메시지 수 최소 구성 (h)) | 397.9 M -> 427.7 M ns **느림** | sg_per_wg 16->8로 스레드 수 절반 | 동일 |
| 13 | u4 V dequant 4x 중복을 SLM으로 공유 | 폐기(미측정) | -24% ops vs WG 4->3(-25% 점유율), 블라인드 베팅 | 동일 |
| 14 | `Q_2D=1`, `A_2D=1` 강제 (prologue/epilogue) | +0.35% (무의미) + fixup 없어 **오답 위험** | 루프 밖 | `sdpa-ocl-block2d-gate-relaxation` |
| 15 | decode K를 `vload16` per-lane row (token-major) | 명령 1076 -> 980이지만 wall-clock 불변~약간 느림 | scatter 16 line vs block 8 line | `pa-k-token-major-progress` |
| 16 | kvup `PA_K_SGS_PER_WG` (같은 sector를 공유하는 partition을 한 WG에) sgs 1/2/4/8/16 | 이득 없음(sgs=2는 +479 ns) | Xe2는 같은 L1을 쓰는 다른 스레드의 sub-line write를 합치지 않음 | `pa-kvup-token-major-store-sector-floor`, B70 |
| 17 | kvup 루프 컴파일 타임 unroll + predication | 3700 -> 3618 (2%) | 한계 요인 아님 | 동일 |
| 18 | kvup partition 수 16 -> 8 (h_sub>=2) | 비율은 좋아져도(0.90) 절대는 4337 vs 3700 **느림** | 스레드당 직렬 작업 2배; production baseline은 d-major@parts16=3353 | 동일 |
| 19 | kvup partition-tiled 페이지 레이아웃 `[K/16][16][8 B]` | 범위상 기각(시도 안 함) | decode/MIXED의 DPAS operand 조립을 다시 써야 함 | 동일 |
| 20 | u4 token-major writer lane-varying 루프 변수 | 1.8x 느림 | `token_vals[]` register 소실 | `pa-kvup-u4-token-major-writer` |
| 21 | `skip` 가드를 helper 본문 안에 | 18.7 us (분할 안 한 것보다 2.3x 느림) | 안쪽 루프 unroll 소실 | 동일 |
| 22 | paired Kc DWORD parity 사이 보관 (u4 MIXED) | 177,716 vs 158,661 ns (**약 12% 악화**) | 장기 private 배열이 spill/register 압박 | `sdpa-ocl-mixed-exact-complete`, B70 |
| 23 | MIXED `KQ_FAST`, `KQ_TRIM` | 172,310 / 157,932 (악화), 제거 | 컴파일러 수준 원인 미입증 | 동일 |
| 24 | rung 3 이전 "basic suite green이니 rotate도 OK" | rotate가 512 쌍 전부 오염 | 기본 suite는 캐시 내용을 관찰 안 함 | `pa-k-token-major-progress` |
| 25 | `16b_32r16x1c`를 `2x 16r`의 drop-in으로 | 미확정(프로브가 모두 0이라 무효) | sentinel 검증 누락 | `sdpa-ocl-tk32-bug-hunt` |
| 26 | `V_I8_PAIRED_READ`를 PA 경로에 | **금지** | 연속 key group이 비인접 페이지(`block_indices`) | `sdpa-ocl-pa-i8-by-token-task` |
| 27 | head-72에서 `DKS_ACTIVE`가 scalar 폴백의 spill을 키움 (head 48/96) | 정적 2688 -> 16064(DKS_ACTIVE 8/6/5) | **device 미측정(2026-08-14 시점)** | `sdpa-ocl-block2d-gate-relaxation` |
| 28 | f16 V-read 병합 `16r16x2c` (VTune 기반) | **소폭 이득**(실패 아님): 524.34 -> 521.01 ms (-0.64%), MD5 동일; SBID stall 43.9–45.5% -> 41.5–43.5% | 단일 workload 사례, 다른 GPU/모델 보장 아님 (MEASURED) | 04장 §4.1.1, 05장 §5.5.3, B70 |

## 13. 이 장의 불확실/stale 항목 (확인 필요)

- `sdpa-ocl-int8-perf`의 "K dequant를 float으로 하는 것이 half보다 이긴다"(2026-07-05, ISA + 사용자 실측)는 현재 코드와 어긋난다: `k_tile_i8_b2d`(`sdpa_ocl_qk_load.cl:~275`)는 bias trick `as_half(0x6480 ^ byte)`를 half로 계산하고 bf16에서만 float를 쓴다 -> 이후 변경으로 대체됐을 가능성. 내용은 `03-numerics-softmax-quantization.md`.
- `docs/sdpa_ocl.md`의 "Performance opportunities" 두 항목은 미측정 후보다(decode `% 16` page rule, MIXED V page lookup 2회 등).
- decode prefetch 측정(§7.2)의 장비 표기 누락. B70 이전(2026-07-05) 이후 날짜이므로 B70으로 추정하되 ASSUMED.
- tk32 override-only 버그(`sdpa-ocl-tk32-bug-hunt`)는 UNRESOLVED(2026-09-25: 기본 설정에서는 재현 안 됨, `MICRO_MATH=1`이 가림). split barrier 주변 순서가 용의 후보였으나 증거 없음.
- `sdpa-ocl-block2d-gate-relaxation`의 `MAX_BARRIER_V_PREFETCH`, `DKS_ACTIVE`, `SDPA_OCL_PA_CUR_*` 토글은 리팩토링으로 삭제됨(코드에서 확인): 노트의 env 이름은 역사 기록이다.
