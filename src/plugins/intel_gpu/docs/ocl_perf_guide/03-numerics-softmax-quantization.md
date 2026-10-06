# 03. 수치 설계: online softmax, 마스크, 양자화 dequant, 정확도 검증

대상: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/` 의 `sdpa_ocl.cl`, `sdpa_ocl_decode.cl`, `sdpa_ocl_mask.cl`,
`sdpa_ocl_qk_load.cl`, `sdpa_ocl_v_load.cl`, `sdpa_ocl_config.cl`, writer `pa_kv_cache_update_ref.cl`.
호스트 jit: `sdpa/sdpa_gen_ocl.cpp`, `sdpa/sdpa_gen_ocl_decode.cpp`. 설계 문서: `docs/sdpa_ocl.md`.

표기: **MEASURED** = 출처(노트/커밋/하드웨어) 있는 측정값, **ASSUMED** = 코드 읽기 또는 가설. 하드웨어 약어: B580 / B70 = Xe2 dGPU,
DG2 = Xe-HPG, 모든 B580 수치는 B70 이전 것이라 stale 로 취급(`hardware-migration-b580-to-b70` 노트). 정적 ISA 수치(instCount, mov 수)는
`ocloc -device bmg` 결과이며 시간이 아니다.

관련 장: DPAS operand 역할/타일링 → `01-dpas-and-tiling.md`, 2D block read/prefetch/barrier → `02-memory-io-prefetch-barriers.md`,
spill/ISA/프로파일링 → `04-spill-isa-profiling.md`, 검증 방법론 일반론 → `05-methodology-and-pitfalls.md`.

---

## 0. 한 장 요약 (수치 규칙)

| 규칙 | 근거 |
|---|---|
| 누적(accumulator)은 항상 f32: DPAS `float8` acc, softmax max/sum f32, S*V acc f32. DPAS 입력만 f16/bf16 | `sdpa_ocl.cl:365` `float8 A_tile`, decode `S_VEC_TYPE` = `SOFTMAX_ACCUMULATOR_TYPE` |
| softmax 확률 P 는 입력 dtype(f16/bf16)으로 반올림해 SLM 에 저장, 분모 합은 반올림 전 f32 exp 로 합산 | `sdpa_ocl.cl:807-823`(PACK_SOFTMAX8), decode `:702-710` |
| scale 에 log2(e) 를 접어 `exp2` 사용 (prefill). decode 는 Q 에 scale 을 미리 곱하고 `native_exp` | `sdpa_ocl.cl:156,807`, decode `:408,702` |
| fully-masked 행/타일은 NaN 이 아니라 0 출력 (`l > 0 ? 1/l : 0`, `isfinite(m_new)` 가드) | `sdpa_ocl.cl:780,1017` |
| 양자화 dequant 은 affine 이라 scale/zp 를 DPAS 밖(점수 또는 확률)으로 뺄 수 있다 | §5 |
| f16 정수 widen 트릭은 정수 zp 에서만 exact. 비정수 zp 는 ulp=1 (1024~2048) 에서 반올림됨 | §4.5, §10 |
| 마스크가 얹히는 NaN: `NaN + -INF = NaN`. 미기록 캐시 슬롯의 NaN 은 마스크로 가려지지 않으므로 로드 단계에서 0 처리 | §2.6 |
| 종단 지표(WWB)로 커널 정오 판정 금지. layer-0 동일 입력 + 출력 dtype ULP 와 비교 | §9 |

---

## 1. Online (flash) softmax 설계 — `sdpa_ocl.cl`

### 1.1 상태와 점화식 (a: 무엇)

key 타일 k0 마다 query 별 running max `m`, running sum `l`, 누적 출력 `A` 를 갱신:

```
m_new = max(m_old, max_k s_k)            // s = raw QK^T (scale 미적용) + mask*iscale
alpha = exp2((m_old - m_new) * scale')    // scale' = scale * LOG2E
p_k   = exp2(s_k * scale' - m_new * scale')
l     = alpha * l + sum_k p_k
A     = alpha * A + P @ V
O     = A / l   (epilogue 1회)
```

- S 타일(`S_tile`)은 **스케일하지 않은** raw 점수로 두고 max 도 raw 도메인(`S_max_slm`)에서 구한다. `scale' = scale * LOG2E`
  를 exp2 인자에서 한 번만 곱한다 (`sdpa_ocl.cl:130-156`). 마스크/sink 는 raw 도메인이므로 `* iscale`(= 1/scale) 로 얹는다
  (`:651,700,717`; sink `:163`).
- 기본 경로는 스케일된 max 를 저장(`S_max_tile = m_new * scale'`)하고 `exp2(S*scale' - m_log2)` 로 계산 → 인자당 fma 1회.
  `MICRO_MATH` 경로는 raw max 를 빼고 나서 스케일(`exp2((S - m) * scale')`) 하며 키를 순서대로 합산, S*V 를 `A_tile1`(0 시작)에서 누적한 뒤
  더한다 — `sdpa_micro` 의 반올림을 재현하기 위함 (`sdpa_ocl.cl:781-831, 870-879`, 호스트 `sdpa_gen_ocl.cpp:677-688`: PA PREFILL f16,
  sink/alibi/sliding-window/token_type_ids 없을 때만 기본 ON, `SDPA_OCL_MICRO_MATH=0` 으로 OFF).
- 이유: 두 경로는 **수치 순서가 다르다**. 기본 경로는 `exp2` 인자의 반올림이 다르므로 bit-exact 하지 않다. micro 와 bit 근접이 목표면
  MICRO_MATH, 아니면 기본. (§9 의 "order-changing" 분류 참고)

핵심 스니펫 (`sdpa_ocl.cl:780-831` 축약):

```c
const bool ok = isfinite(m_new);                       // m_new = S_max_slm[query] (subgroup 간 atomic max 결과)
const float m_log2 = ok ? m_new * scale : 0.0f;
const float a = ok ? native_exp2(S_max_tile[qb] - m_log2) : 1.0f;   // alpha
S_max_tile[qb] = ok ? m_log2 : S_max_tile[qb];
float8 exp_tile = ok ? native_exp2(S_tile[mb][qb] * scale - m_log2) : (float8)0.0f;
lsum += exp_tile[0] + ... + exp_tile[7];               // f32 합, P 반올림 전
S_sum_tile[qb] = a * S_sum_tile[qb] + lsum;
vstore4(as_uint4(PACK_SOFTMAX8(exp_tile)), 0, &S_slm[...]);   // P 는 f16/bf16 으로 SLM
// alpha 로 이전 A 를 rescale (첫 타일 제외):  A_tile[r][cd] *= av
```

### 1.2 pitfalls (d)

| 함정 | 설명 / 근거 |
|---|---|
| `-inf - -inf = NaN` | 해당 타일까지 유효 key 가 하나도 없는 query 행(causal 앞쪽, sliding window, fully-masked) 은 `m_new = -inf`. 가드 없이 `S - m` 을 만들면 S/A 가 NaN 으로 오염. `ok = isfinite(m_new)` 로 exp 인자를 0 으로 두고 `alpha = 1` (`:777-795`) |
| `first`/`last` 플래그 | `first` 는 alpha rescale 게이트, `last` 는 `S_sum_slm` 기록 게이트. causal/window 로 루프 시작·끝이 움직이면 **둘 다 같이 움직여야** 한다 (`:481-485`). 안 그러면 결과 손상/합 미기록 (causal-bound 노트) |
| `alpha[]` 런타임 인덱싱 | private 배열을 런타임 인덱스로 읽으면 IGC 가 scratch 로 보냄(llama-3.2-1b MIXED: private memory 128 B + k0 루프 내 scratch ld/st 2+2). 컴파일 타임 `kq_query_blocks` 에 대한 select chain 으로 해결 (`:849-861`) — 수치 변화 없음, 성능 목적 |
| `native_exp2(-inf)` | 첫 유효 타일이 `first` 가 아닐 때 `alpha = native_exp2(-inf - m)` 가 0 이어야 `A*alpha` 가 0 으로 유지됨. **ASSUMED**: Xe2 HW exp2 가 -inf→0. (OpenCL `native_*` 의 극단값 동작은 구현 정의) 테스트로 확인된 경로(causal 초반 타일)가 있으나 별도 단위 검증은 못 찾음 |
| S_slm 의 P 정밀도 | P ∈ [0,1] 이라 f16 overflow 없음. 분모는 반올림 전 f32 합 vs 분자는 반올림된 P: 상대 ~5e-4 불일치. micro 도 동일 구조 |
| 단일 max 소스 | `S_max_slm` 은 subgroup 들이 `__builtin_IB_atomic_max_local_f32` 로 합치는 running max. 초기값이 -INFINITY(또는 sink) (`:319-327`) |

### 1.3 decode(`sdpa_ocl_decode.cl`) 의 차이

- 키가 lane 에 1개씩(`lane == key`) 이라 max/sum 이 subgroup reduce 2회 (`:667-700`). 분할(partition) 별로 자기 max 에 대해 정규화하고
  `exp_sums`/`max_logits` 를 쓰면 `pa_sdpa_finalization_stage` 가 합친다.
- **-INFINITY 대신 `SOFTMAX_ACCUMULATOR_VAL_MIN`** 으로 마스크(`:627-640`). 이유: partition 전체가 가려질 수 있고(슬라이딩 윈도우가 훨씬 뒤)
  `MIN - MIN = 0` 이라 exp 가 유한(=1)하게 남고, finalization 이 `exp(max_logit - global_max)` 로 그 partition 을 0 으로 지움.
  prefill 커널과 다른 관례이므로 혼용 금지 (partition 합성 구조 때문).
- Q 에 `scale_val` 을 미리 곱한다(`:408`) → 점수가 이미 스케일됨. 그래서 decode 에선 mask·sink 가 raw 가 아님. prefill 의 sink 규칙
  (`* iscale`)을 decode 에 복사하면 틀린다.
- partition 이 1개이면 직접 output 을 쓰고, 여러 개면 `tmp_out` 을 partition 자기 합으로 나눠 쓴다. 이 분기 조건은 **finalization 의
  `effective_seq_len = seq_len - swa_start_token`** 와 정확히 상보적이어야 한다 (T4: SWA 윈도우 240..256 에서 출력 미기록 버그, 수정 `03c9121efc`).

---

## 2. 마스크

### 2.1 causal / sliding window key-loop 상한·하한 (a/b/c)

mask 만 믿고 전체 key 를 도는 것이 root cause 였다: `for (k0=0; k0<k; ...)` + `if (key > query) s = -INF` 는 대각선 너머 타일도 K/V 로드+DPAS 후 버림.
q=k=1024, head 128, 32 heads 에서 ocl(query tile 32) causal 효율 50% vs micro(128) 80% (causal-bound 노트, 산술치).

```c
// sdpa_ocl.cl:383-395, 417-434 (축약)
int causal_k = min(k, query_position_offset + (int)wg_j0 + kq_wg_tile_queries);
int window_k_begin  = max(0, query_position_offset + (int)wg_j0 - SLIDING_WINDOW_SIZE + 1);
const int window_k0_begin = (window_k_begin / kq_wg_tile_keys) * kq_wg_tile_keys;   // 타일 정렬로 내림
for (int k0 = window_k0_begin; k0 < causal_k; k0 += kq_wg_tile_keys) { first = (k0 == window_k0_begin); last = ...; }
```

- window 하한은 **k0 타일 경계로 내림** — 2D block read 와 `S_slm` 인덱싱이 `key_base` 의 타일 정렬을 가정.
- 0 iteration 불가: `window_k0_begin <= wg_j0 < causal_k` (노트: 7.76M (query,key) 쌍 brute-force, W∈{1..512}×k∈{16..512} 에서 필요 쌍 skip 0).
- 블록 단위 skip `causal_block_clear` (`:668-682`): 블록 전체가 causal(+window) 안쪽이면 per-element 술어 생략. 서브그룹 uniform.
- 측정(B70, 2800MHz 고정, PA prefill q=1024, 32/8 heads, head 128, f16, min ns): causal_k 없음 3,421,875 → 3,300,312 (**tile 수 1.78x 감소인데 시간 1.04x**),
  window(W=256) 2,266,562 (**타일 1.71x 감소, 1.46x 빠름**). 타일 수 감소가 시간으로 선형 환산되지 않음(WG 내 reduce/barrier 오버헤드 가설, **미해결**).
  llama-3.1-8b 실모델에서 causal_k 효과는 사용자 확인, window 는 해당 모델에 SW 가 없어 효과 없음(컴파일 아웃).
- 테스트 커버리지 구멍(노트, 이후 변경 가능): sdpa_ocl PREFILL + sliding_window 를 타는 케이스가 당시 없었음 (다른 backend 로 라우팅).

### 2.2 causal lower-right 정렬 (plain SDPA)

stateless decode(`q < k`) 에서 마스크를 우하단 정렬하면 query i 는 keys `[0, i + (k - q)]` 를 본다. 정렬 안 하면 `q=1,k=512` 에서
`causal_k = min(512, 0+32) = 32` 로 key 32..511 을 **방문조차 안 해** 분모가 틀림 (`decode_40q_40kv_512seq`).

```c
// sdpa_ocl_config.cl:178  (PA 는 past_len 으로 같은 shift 를 이미 하므로 plain 에서만 ON)
#define LOWER_RIGHT_SHIFT(pos) pos + causal_offset       // causal_offset = max(0, k - q)
```
- 적용 3곳: `causal_k` 상한, `causal_block_clear`, per-element 술어 + window 하한 (모두 같은 offset, `sdpa_ocl.cl:388,671,740-743`).
- 매크로가 **괄호 없음**: 비교 연산자 피연산자 위치에서만 안전. 산술식 안에서 쓰면 우선순위 버그 (잠재).
- 호스트: `jit.make("CAUSAL_MASK_LOWER_RIGHT", config.is_paged_attention ? false : config.causal_lower_right)` (세션 기록 `test/SDPA_OCL_CAUSAL_LOWER_RIGHT_SESSION.md`).
  `=0` 일 때 전처리 결과 byte-identical, ocloc inst 1190 → 1194 (+4). 사용자 확인: `*sdpa_gpu_causal_mask*` 전체 통과.

### 2.3 마스크 종류 (`MASK_KIND`) 와 OOB

| MASK_KIND | 모양 | 처리 |
|---|---|---|
| 2 | full 2D `[.,.,q>1,k>1]` | lane=query, 타일을 `half16` vload 후 `* iscale` (`sdpa_ocl_mask.cl:56-84`) |
| 1 | per key `[.,.,1,k]` | `mask_tile` 을 lane=key 로 로드, `sub_group_broadcast` |
| 0 | scalar/broadcast | `HAS_SCALAR_ATTN_MASK`: `s += MASK_TO_FLOAT(msk[0]) * iscale` (`:697-701`) |
| -1 | 런타임 판정 | `MSK_D2`/`MSK_D3` 로 분기 |

- **동적 shape 함정**: 마스크 trailing dim 이 동적이면 호스트가 stage 로 kind 를 **추정**(prefill → 2). 실제 런타임 마스크가 `[B,H,1,K]` 이면 full-2D 로더가
  query 행을 `~990` 까지 읽어 1행 버퍼를 ~15배 초과 → Xe2 에서 `CL_OUT_OF_RESOURCES`(OOB read 가 0 이 아니라 크래시) 또는 garbage→NaN.
  수정: `mask_query = (MSK_D2 == 1) ? 0 : ...`, 행 상한을 `mask_query < MSK_D2`, 키 쪽도 `MSK_D3 == 1` 브로드캐스트 + `mask_key + kk < MSK_D3` (`sdpa_ocl_mask.cl:62-80`).
  static 마스크는 리터럴이라 select 가 fold → 코드젠 byte-identical (OOB 세션 기록 `test/SDPA_OCL_MASK_KIND_OOB_SESSION.md`).
- **scalar 런타임 마스크**: rank-0/1-element 입력은 "real mask" 로 취급되지 않아 값이 무시되었고 `CL_OUT_OF_RESOURCES` 도 났음. 별도 헬퍼(`has_scalar_runtime_attn_mask_input`) +
  `HAS_SCALAR_ATTN_MASK` 로 처리 (세션 `test/SDPA_OCL_SCALAR_RUNTIME_MASK_SESSION.md`). sdpa_micro 는 이 게이트를 유지(미지원).
- bool/half 마스크 값은 `MASK_TO_FLOAT*` 로 float 로 푼 뒤 `* iscale` — bf16 이면 `as_ushort` → bit-decode (§8).

### 2.4 bidirectional (`token_type_ids`, 이미지 그룹)

allowed key 집합은 **합집합** `(causal ∩ window) ∪ (자기 이미지 그룹)` 이며 쌍(pair) 술어가 아니다. 그래서 **query 쪽 `[group_begin, group_end)` 만** 있으면 되고
(WG 당 1회 스캔, lane 당 int2), 핫루프는 `key < gb || key >= ge` 한 줄 (`sdpa_ocl.cl:745-752`).
- key 상한/하한 확장: 그룹은 연속이므로 **WG 의 마지막 query 하나**만 스캔하면 충분(`bidir_scan_end`, subgroup cooperative `sub_group_reduce_min/max`, `sdpa_ocl_mask.cl:13-46`).
  스캔을 k0 루프 범위로 clamp 하는 것은 근사가 아니라 **정확** (범위 밖 key 는 방문 안 되거나 이미 -INF).
- `causal_block_clear` 는 그대로 유효: 이미지 규칙은 un-mask 만 하고, 블록이 causal+window 내부임을 증명하는 skip 은 un-mask 할 대상이 없음.
- 좌표계: `token_type_ids` 는 **LOCAL(새 토큰 `[0,q)`)**, `key/causal_k/window_k_begin/bidir_group_*` 는 **KEY(= query_position_offset + local)**. 혼동이 가장 흔한 버그원. 캐시 토큰은 새 query 와 같은 이미지 그룹이 될 수 없다.
- GENERATE 는 불필요(증명): 1 토큰/서브시퀀스라 그룹 = 자기 자신 → plain causal. MIXED 는 `kv_cache_update` 가 SDPA 보다 먼저 돌아야 그룹의 미래 key 를 읽는다 (아무것도 assert 하지 않음).
- `[B_token | 0]` 런타임 게이트: `token_type_ids_count > 0` 이어야 접근 (`bidir_active`). 빈 텐서를 읽으면 OOB (`sdpa_ocl.cl:199-216`).

### 2.5 qq_bias (speculative tree mask, MIXED 전용)

새 key 구간 `[past_len, past_len+spec_num)` 안의 (query, key_spec) 쌍에서 `qq_bias == 0` 이면 `s = -INFINITY` (`sdpa_ocl.cl:720-733`). jit/시그니처/인자가 모두
`HAS_QQ_BIAS && IS_PA && !IS_PREFILL` 로 일치해야 한다 (한때 jit 이 0 으로 하드코딩돼 인자 shift). decode 는 새 토큰 1개라 1x1 identity 라 읽을 필요 없음(세션 `test/QQ_BIAS_SDPA_OCL_DECODE_SESSION.md`).

### 2.6 fully-masked 행, NaN/Inf 취급 요약

| 상황 | prefill (`sdpa_ocl.cl`) | decode |
|---|---|---|
| 한 타일에 유효 key 없음 | `ok=false` → exp_tile=0, alpha=1, S_max_tile 유지 | `VAL_MIN` 마스크, partition 합성에서 제거 |
| 행 전체 유효 key 없음(`l == 0`) | 출력 **0** (`inv_l = (l > 0) ? recip(l) : 0`, `:1017`) | N/A |
| 미기록 캐시 슬롯(NaN 가능) | 로드 단계에서 height clamp / scale=zp=0 강제. **마스크의 `+ -INF` 는 NaN 을 못 지운다** (`NaN + -inf = NaN`) | K: 마스크가 `s` 를 **덮어써서**(`s[g] = VAL_MIN`) NaN 소거. V: 확률 0 은 곱셈이라 `0*NaN = NaN` → V 는 `v_valid ? v_sc : 0` 가드 필요 (`:788-803`) |
| causal 술어 | `s = -INFINITY` 대입(덧셈 아님) | — |

`fmax(lmax, NaN)` 은 NaN 을 무시하므로 max 는 멀쩡한데 합만 NaN 이 되는 식으로 **증상이 늦게 나타난다**.
출력 0 이 reference 와 일치하는지(reference 가 NaN 을 내는지)는 **검증하지 못함 (ASSUMED)**.

---

## 3. Attention sink (gpt-oss)

sink = head 당 추가 logit 하나, **value 벡터는 0** → softmax 분모만 키운다.

- **prefill/MIXED (`sdpa_ocl.cl`)**: online-softmax 상태를 **seed** 한다. 핫루프 변경 0.
  ```c
  const float sink_raw = SINK_TO_FLOAT(sink_ptr[b0]) * iscale;        // :163 raw 도메인(= mask 와 동일)
  S_max_slm[qi]   = sink_raw;                                          // :324 running max 가 sink 로 시작 (WG 전체 1회)
  S_max_tile[qb]  = sink_raw * scale;                                  // :354 같은 곱 → 첫 alpha 가 정확히 1.0
  S_sum_tile[qb]  = (sg_i_kq == 0) ? 1.0f : 0.0f;                      // :358 exp2(sink - sink) = 1, 한 subgroup 에만
  ```
  `S_sum_tile` 을 모든 subgroup 에 seed 하면 epilogue 의 `S_sum_slm` 합이 sink 를 `kq_sg_per_wg_keys` 번 센다.
- **decode**: partition 0 에서만 주입(`:685-694, 746-752`), `m_wg` 에 sink 로 max, `lw` 에 `native_exp(sink - m_wg)`. partition 마다 넣으면 finalization 이 P 번 센다.
  `SINK_HEAD(m)` 은 leftover-clamped 인덱스 (마지막 WG 의 가짜 head 가 OOB 읽지 않게).
- 호스트: `HAS_SINK_INPUT` 는 jit 되었는데 커널이 인자를 선언하지 않아 **arg shift** 가 나던 잠재 버그가 있었다. jit/시그니처/push 순서 3자 일치 확인.
- 성능: sink 는 3529 inst 중 13 inst (**MEASURED**, ISA A/B `u4_bychannel_1d_sink` vs `u4_bychannel_1d`, B580). gpt-oss 가 느렸던 원인은 sink 가 아니라 u4 head-64 K/V 로드(`sdpa-ocl-u4-head64-page-read`).
- 테스트 데이터 함정: 데이터가 N(0,0.1) 이라 logit≈0.01, 분모≈kv_len → sink 값을 `log(kv_len) + spread[h%4]` 처럼 분모 규모에 맞춰야 한다. 아니면 sink 를 빼도 통과(vacuous).
- 음성 대조 중 "정상적으로 아무 변화 없는" 것: **sink 를 max 에서 빼기** 는 결과를 바꾸지 않는다(max 는 자유 안정화 파라미터; 분자·분모가 같이 스케일). 단 partition 0 전체가 가려진 경우만 필수.

---

## 4. int8 → f16 변환 트릭 (`0x6480`)

### 4.1 무엇 / 왜

Xe2 에는 byte→half 직접 convert 가 없다. `convert_half16(as_char16(x))` 은 byte deinterleave(`mov :b`), `:b→:w<2>`, `:w→:hf<2>` 의 3-단 계단이 되어 element 당 mov 3개 수준.
대신 **XOR 한 번 + (나중에) half 뺄셈**으로 해결한다.

```c
// sdpa_ocl_qk_load.cl:294-295, sdpa_ocl_v_load.cl:157, sdpa_ocl_decode.cl:120-127
const ushort wbits = (ushort)0x6480 ^ (ushort)(w_byte & 0xFFu);   // w_byte = signed int8 의 2의 보수 비트
const half   wide  = as_half(wbits);                              //  == (float)s + 1152.0
```

### 4.2 비트 유도

f16 = `S(1) E(5) M(10)`, 값 = `(1 + M/1024) * 2^(E-15)`.

1. `0x6400 = 0 11001 0000000000` → E = 25 → 2^10 = **1024**, M = 0.
2. E=25 이면 `[1024, 2048)` 구간이고 ulp = 2^10 / 1024 = **1.0**. 따라서 `0x6400 | n` (n ∈ [0,1023]) = **1024 + n**, 정수 1 간격으로 정확히 표현.
3. `0x6480 = 0x6400 | 0x0080` → M = 128 → 1024 + 128 = **1152.0**. (`0x6480` 은 "1152.0h")
4. signed byte s (2의 보수) 는 오프셋 이진 `s + 128` 의 **bit 7 반전** 이다: `u = s & 0xFF`, `u ^ 0x80 = s + 128 ∈ [0,255]`.
   - s ≥ 0: u = s, bit7 = 0 → 세팅 → s + 128.
   - s < 0: u = s + 256, bit7 = 1 → 해제 → s + 256 − 128 = s + 128.
5. 따라서 `0x6480 ^ u` = `0x6400 | ((u ^ 0x80) )` = `0x6400 | (s+128)` → 값 = 1024 + s + 128 = **s + 1152**. XOR 이 (a) 부호→오프셋 이진 변환과 (b) 지수 필드 주입을 동시에 한다.
   (`0x80` 비트가 mantissa 의 한 비트이므로 지수 필드 `0x6400` 은 건드려지지 않는다: byte 는 0..255 → mantissa 하위 8비트만 변함.)

### 4.3 예제 (worked)

| s (int8) | u = s&0xFF | u ^ 0x6480 (=h bits) | mantissa | 값 | s + 1152 |
|---|---|---|---|---|---|
| +5 | 0x05 | 0x6485 | 133 | 1024+133 = 1157 | 1157 |
| 0 | 0x00 | 0x6480 | 128 | 1152 | 1152 |
| −3 | 0xFD | 0x647D | 125 | 1024+125 = 1149 | 1149 |
| −128 | 0x80 | 0x6400 | 0 | 1024 | 1024 |
| +127 | 0x7F | 0x64FF | 255 | 1279 | 1279 |

노트 기록: 256개 int8 전체에서 `as_half(0x6480 ^ u) - 1152.0 == (float)s` 가 정확히 성립함을 C 프로그램으로 확인(`/tmp/verify_bias.c`, int8-perf 노트; 파일 자체는 휘발성).
IGC 는 이를 정확히 `xor(:w, 0x6480)` + `add(:hf, 0xE480 = -1152)` 2 op 로 컴파일 (ISA 확인, 노트). 이것은 sdpa_micro/gemmstone `planInt8ToHF`(copy_plan.cpp:1201-1220)와 같은 lowering.

### 4.4 scale / zp 를 접는 방법

- **bias 를 zp 에 접는다** (`zpb = zp + 1152.0h`): element 당 `(wide - zpb) * scale` = sub 1 + mul 1, 별도 bias 감산 add 가 없어야 이득이다
  (bias 감산을 분리한 v1 은 145 inst/62 mov 로 add 폭증, 접은 v2 는 130/62 — 마이크로벤치, B580 정적 ISA).
- 뺄셈 결과는 `s - zp` 이므로 **zp 가 정수이면 f16 에서 정확** (`1152+s` 정확, `zp+1152` 정확, 차 ∈ ±255 정확). scale 곱 1회만 반올림.
- 바이트 추출은 `(w >> (8*bb)) & 0xFF` (shift+mask). `as_char4(w)[bb]` 는 `:b` region deinterleave 를 만든다.
- VNNI 용 dword 로 바로 구성: 4 byte → 2 dword (`K_WIDEN_DWORDS`, decode `:120-127`):
  ```c
  dst[j*2+0] = as_int(((w & 0xFF) | ((w & 0xFF00) << 8)) ^ 0x64806480u);        // byte0, byte1 → half2
  dst[j*2+1] = as_int((((w >> 16) & 0xFF) | ((w >> 8) & 0x00FF0000)) ^ 0x64806480u);
  ```
  VNNI half2 dword 의 (2i, 2i+1) 짝이 byte 순서와 그대로 일치 → sub-register 이동 없는 순수 dword 산술.
- 부호 없는 u8 이면 XOR 대신 `0x6400 | u` (= 1024+u). 부호 byte 에만 `0x6480 ^` 가 필요(bit 7 반전 목적).

### 4.5 측정 효과와 한계

| 항목 | 값 | 출처/HW |
|---|---|---|
| K dequant 마이크로벤치(dpas 소비, 32 keys, asym) | float 132 inst/77 mov/48 `:b` → bias 100 inst/24 mov/0 `:b`. mov −69% | 정적 ISA, B580 (int8-perf 노트) |
| 실 커널 loop body (head 128) | 2751 → 2534(K) → 2247(K+V) inst, mov 1535→1055→543 | 정적 ISA, B580 |
| 디바이스 시간 (head 128, q=4096, plain int8 prefill, floor ns) | 5,620,520 → 5,406,562(K, −3.8%) → 5,085,625(K+V, 누적 −9.5%); ocl/micro 1.396→1.261 | **MEASURED**, B580, 2800MHz 고정 기재, 노트. B70 재측정 아님 |
| decode i8 (head128, M=4) | dword widen 4045 inst/2022 mov vs `convert_half16(as_char16)` 4431/2852 | 정적 ISA (docs/sdpa_ocl.md:624-628) |

**한계 (pitfalls)**
- **정수 zp 에서만 exact.** `zp + 1152` 는 f16 ulp = 1 이므로 비정수 zp 는 정수로 반올림된다 (`np.float16(7.3+1152)-1152 == 7.0`).
  plain SDPA i8 은 KV compression 이 zp 를 **i8** 로 저장(XMX 에서, `kv_cache_compression.cpp:244`)하므로 안전. **int4(zp = query type)·PA 캐시(writer 의 비정수 zp)에는 쓰면 안 된다.**
  → PA 경로는 `(q - zp) * scale` 을 half 로 명시 (`PA_DEQ`, `sdpa_ocl_config.cl:256`), decode 는 bias 를 **float 로 따로** 제거(§5).
- bf16 에는 같은 트릭이 없다(§8). u4 는 1024 base (§4.6).
- `K_WIDEN_BIAS` 와 `zp` 의 결합이 깨지면 조용히 틀린다: 단위 테스트 임계(6e-3)에 못 걸리는 수준(§5.3).
- 노트의 "denormal reinterpret" 표현은 부정확: 결과는 정상 정규수 (E=25).

### 4.6 u4 nibble

- `as_half(0x6400 | n) == 1024.0 + n` (n ∈ [0,15]), bias = **1024** (`K_WIDEN_BIAS` u4=1024, i8=1152; decode `:115-119`).
- K 는 **adjacent** 패킹(byte b = 채널 2b, 2b+1; writer 의 `NUM_K_HEAD_SIZE_PARTITIONS` 가 채널을 WG 간 분할하므로 byte 의 두 채널이 서로 다른 WG 에 있으면 race → 강제). VNNI dword 의 (2i, 2i+1) 짝과 일치 → depth 순서가 자연스러움.
  ```c
  // decode :129-139  K_WIDEN_U4_DWORDS: byte → dword(half2) = (lo nibble, hi nibble)
  as_int(((p & 0x0000000Fu) | ((p & 0x000000F0u) << 12)) | 0x64006400u)   // p = w >> (8*b)
  ```
  hi nibble(bit4..7) 을 `<< 12` 로 bit16..19 (두 번째 half 의 mantissa 하위) 로 보내고 `0x6400` 지수를 OR → 두 half = (1024+lo, 1024+hi).
- V 는 **split** 패킹(byte b = dim b 와 dim b+PV): `transform_8b_32r16x1c` 는 lane == byte column 이라 adjacent 면 lane c 가 dim 2c, 2c+1 둘 다 소유해 DPAS N 축을 표현 못 함.
  split 이면 `V_TILE_COL(cd)` 로 읽어 lane c = dim `cd*16+c` (하위/상위 nibble 모두) → S*V 타일 루프/출력 store 불변. nibble 선택은 `cd < V_READS ? (x & 0xF) : (x >> 4)` (decode `:830-836`).
  V 는 bias 트릭 없이 plain widen (V zp 는 요소 단위, f16 ulp@1024 = 1 vs 0..15 범위).
- sdpa_ocl MIXED 의 u4 K: A operand 가 K(lane == head dim)라 adjacent 패킹의 byte column 이 채널 **쌍**이 된다 → lane-local 해법 없음. **DPAS depth(contraction) 축을 permute** 한다:
  `PA_K_U4_CHANNEL(db, L) = win + 2L + par` (`win = (db>>1)*32`, `par = db&1`, `sdpa_ocl_config.cl:278-280`). 타일 db(짝수)는 32채널 창의 짝수 채널, 홀수는 홀수 채널.
  depth 는 contraction 이므로 A 와 B 에 동일하게 permute 하면 무료. **Q 만 비용을 낸다**: WG 당 1회 SLM staging 에서 `q_pack[j] = par ? ((a>>16)|(b&0xFFFF0000)) : ((a&0xFFFF)|(b<<16))` (`sdpa_ocl_qk_load.cl:58`).
  `DKS_ACTIVE` 는 짝으로 묶이므로 u4 에서 항상 짝수로 올림. 검증: `test/check_u4_offsets.py` (Q staging 이 K lane i 와 같은 채널을 element i 에 놓는지 .cl dword 식을 기호 실행, 9 head × 3 kv-head, **음성 대조 5개가 모두 검출**).
- u4 nibble 선택은 **lane-uniform**(타일 parity) 이라 shift 량에 접힘 (`U4_NIBBLE_SEL`, `qk_load:400-407`). writer 가 `[0,15]` 로 clamp 하므로 부호/CHAR_MIN 이 없다.

### 4.7 u4 "exact MIXED" 의 정확한 의미

"exact" 는 dequant 산술이 정확하다는 뜻이 **아니다**. 의미: MIXED(chunked prefill / prefix caching)에서 캐시 `[0, past_len)` 은 u4 dequant 로 읽고,
**현재 chunk 의 새 행 `[past_len, k)` 은 양자화되지 않은 raw f16 `Kc/Vc`** 에서 읽는다 (압축 캐시의 current row 는 raw 와 같지 않으므로 캐시로 가면 틀림).
구현: `PA_CUR_KV_F16`; k0 타일이 `past_len` 을 가로지르면 `k_chunk = past_len - k0` 로 **타일을 잘라** 한 iteration 이 한 소스만 읽게 하고(`sdpa_ocl.cl:473-476`),
고정 DPAS 타일 모양은 유지하되 `k_chunk` 이후 행을 mask(`k_mask`: `key < k0 + k_chunk`, `:642-646`), `k0 += k_chunk`.
- 옛 `GRAN=0`(16-key 페이지 단위 분할)은 regression `/0`,`/1` 에서 **오답**이었다. 지금은 exact split 만 있다.
- 홀수 Kc 오프셋은 dword block read 불가 → `kc_dword_ok` false 면 raw Kc scalar fallback. 이 때 cache 로 돌아가면 오답.
- 성능(**MEASURED**, llama-3.2-1b 실행 중 cliloader 평균 커널 시간, 사용자 측정): OCL 145,603 ns vs micro 147,825 ns (−1.50%). 하위 실험 — SV_TRIM 채택(151,677 vs 158,422), V_PREFETCH 채택(145,603 vs 151,866, −4.12%),
  paired Kc DWORD 보관 177,716(약 12% 악화, 제거), KQ_FAST 172,310(악화, 제거), KQ_TRIM 157,932(악화, 제거). (일부 knob 은 이후 리팩터로 제거됨)

---

## 5. Dequant 항등식 — scale/zp 를 DPAS 밖으로

dequant 가 affine(`x = sc * (q - zp)`) 이므로 곱셈 구조를 따라 인수분해한다. 어떤 operand 에 head dim 이 lane 으로 놓이는지에 따라 접히는 위치가 다르다
(→ `01-dpas-and-tiling.md` operand 역할표). **sdpa_ocl 과 decode 는 역할이 반대**(prefill KQ: A=K,B=Q / decode KQ: A=Q,B=K)이므로 다른 커널의 대수를 복사하지 말고 자기 mapping 으로 다시 유도.

```
BY_TOKEN K  (decode): S[key] = sc[key] * (sum_d Q[d]*q[key][d]  -  zp[key] * sum_d Q[d])
BY_CHANNEL K(decode): S[key] = sum_d (Q[d]*sc[d]) * q[key][d]  -  sum_d (Q[d]*sc[d]) * zp[d]
V (BY_TOKEN)        : O[d]   = sum_key (P[key]*sc[key]) * (q[key][d] - zp[key])
softmax 분모         : sum(P)  — V 의 scale 은 값에 속한다. 분모에 넣으면 틀림.
```
(`sdpa_ocl_decode.cl:22-27`)

### 5.1 어디에 접는가

| 항 | 위치 | 이유 |
|---|---|---|
| BY_TOKEN K `sc[key]`, `zp[key]` | 점수 후처리 (per-lane scalar, lane == key, broadcast 없음) | decode: `s = sc*(s - (zp+bias)*q_sum)`, `:613-624`. `q_sum = sum_d Q[d]` 는 head 당 reduce 1회 (float, `:414-424`) |
| BY_CHANNEL K `sc[d]` | **Q (A operand)** 에 곱: `qv = q_reg * k_sc` (half) | lane == head dim 이므로 per-lane scalar. 페이지마다 달라 g 루프 안에서 재구성(`:557-566`) |
| BY_CHANNEL K `zp[d]` | 점수에서 상수 하나 `k_corr[g][m]` 을 뺌 (키 무관, 페이지·head 당 1회) | `k_corr = sum_d (Q[d]*sc[d]) * (zp[d] + K_WIDEN_BIAS)` (`:480-497`) |
| V `sc[key]` | **P (A operand)** 에 per-lane 곱 (`pv *= v_sc`) | P 는 이미 lane == key. V 에 곱하면 head dim lane 으로 broadcast 필요 |
| V `zp[key]` | V B operand 에서 빼기 (`convert_half16(...) - vzp`) | zp 가 DPAS depth(key) 축을 따라 변하므로 유일하게 broadcast 필요; lane index 가 컴파일 타임 상수라 source region 에 접혀 shuffle 안 생김 |

### 5.2 안전장치 (NaN / OOB)

- seq_len 이후 key 의 comp(scale/zp)는 쓰인 적이 없어 NaN 가능. K 쪽은 마스크가 `s` 를 덮어쓰므로 가드 불필요, **V 쪽은 `0 * NaN = NaN`** 이므로 `v_valid ? v_comp : 0` 로 scale **과 zp 모두** 0 강제 (`decode :788-803`). 그러면 widen 된 byte 는 유한 → 확률 0 이 기여를 지움.
- plain/PA prefill 의 `k_comp_per_key`: `sc_key < k ? scale : 0`, zp 는 `1152`(bias만) — 범위 밖 key 가 `scale=0` 으로 0 이 됨 (`qk_load.cl:105-114`).
- block read 는 페이지 높이를 `PA_PAGE_ROWS` 로 clamp: 안 쓴 슬롯의 NaN 이 마스크된 점수에서도 **살아남기** 때문 (`qk_load pa_k_tile_b2d16` 주석).

### 5.3 BY_CHANNEL `k_corr` 의 f16 반올림 규칙 (조용히 틀리는 곳)

DPAS 가 보는 값은 `Q*sc` 를 **f16 으로 반올림한** 곱이다. bias(1152/1024)는 B operand 안에 `q+bias` 로 들어 있고 `k_corr` 의 `bias` 항과 **그 같은 half** 에 대해서만 정확히 상쇄된다.
`k_corr` 를 float 곱으로 계산하면 잔여가 남아 worst-case error 가 7.21e-4 → 1.68e-3 (2.3x, bias/signal 비 × half ulp). 6e-3 임계 아래라 **테스트가 통과해 보호되지 않는다** (docs/sdpa_ocl.md:630-634; 측정 장비 미기재, probe/gtest 값).
→ 규칙: bias 를 되돌리는 항은 DPAS 가 실제로 본 half 값으로 계산. `zp` 를 f16 에 접으면 writer 의 비정수 zp 가 반올림되므로 decode 는 bias 를 float 로 점수에서 제거 (`:620` `zp_f32 + K_WIDEN_BIAS`).

### 5.4 writer 와 comp 레이아웃 (수치 관점)

`pa_kv_cache_update_ref.cl:103-136`:
```c
diff = (max == min) ? 0.004 : (max - min);              // range 0 보호 (0 나눗셈/inf scale 방지)
// u4:  scale = 15/diff,  zp = -min*scale                // 비정수, [0,15] clamp, 부호 없음
// i8:  scale = 255/diff, zp = -min*scale + CHAR_MIN
q = clamp(convert_int_rte(x*scale + zp), 0, 15)          // round-to-nearest-even
comp[0] = 1.0/scale; comp[1] = zp                        // writer 는 1/scale 저장 → 읽은 값이 곧 곱셈 인자
```
- scale/zp 는 float 로 계산한 뒤 **half 로 narrow** (`(INPUT1_TYPE)`). prefill 헬퍼는 half 로 narrow 한 뒤 quantize, requantize 헬퍼는 float 유지: 교체 시 **교체 대상 arm 의 승격 규칙을 표현식 단위로 보존** (`range = (max==min)?0.004:(max-min)` 는 half 로 뺄셈 후 widen, `fabs(max*0.1f)` 는 float).
- BY_CHANNEL 은 블록에 새 토큰이 들어올 때마다 범위가 넓어지면 블록 전체를 **requantize**(이중 반올림). 범위 안 넓어지면 건너뛰는 최적화는 후속(미구현).
- **requantize 피치 버그 (미수정, Known issue)**: `pa_kv_cache_update_ref.cl:297` i8 BY_CHANNEL requantize 가 `in_data_pitch` 를 무시(`j*K_HEAD_SIZE*KV_HEADS_NUM` stride) → 패딩된 key 입력에서 한 블록 내 j≥1 새 토큰을 잘못 읽음. T7 리뷰에서 발견, 문서화만.

---

## 6. V 의 in-place VNNI-stride dequant

DPAS B operand 는 VNNI-2(`half2` 짝이 depth 축 연속). V 를 `transform_8b_32r16x1c` 로 읽으면 lane == head dim, 각 uint 가 4 keys 를 byte 로 가진다.
이 uint 에서 shift+mask 로 byte 를 뽑아 `0x6480 ^` 로 widen → `deq4 = wide4 - zpb4` → `as_int(deq4.lo)`, `as_int(deq4.hi)` 가 이미 VNNI 순서(key (2i,2i+1))의 dword 이다:

```c
// sdpa_ocl_v_load.cl:155-170  (f16 경로)
const half4 wide4 = (half4)(as_half((ushort)(0x6480 ^ ((w >>  0) & 0xFFu))), ..., as_half((ushort)(0x6480 ^ ((w >> 24) & 0xFFu))));
const half4 deq4 = wide4 - zpb4[u];
vb[cd][u*2+0] = as_int(deq4.lo);  vb[cd][u*2+1] = as_int(deq4.hi);
```
- scale 은 pA 쪽에 이미 접힘 → V dequant 는 widen + zp 감산뿐 (`v_i8_comp_fold`, `v_load.cl:114-133`).
- 마이크로벤치: VNNI B operand 팩은 IGC 의 floor — `as_int2(deq4)` 전체 재해석과 `as_int(.lo)/as_int(.hi)` 분리가 동일(97 inst / 35 `:uw`), 소스 형태로 줄일 수 없음 (v_dequant_bias2_dpas.cl, 정적 ISA, B580).
- V 적용 단계가 K 단계보다 이득이 컸다: floor 5,406,562 → 5,085,625 ns (V, −5.9%) vs 5,620,520 → 5,406,562 ns (K, −3.8%) (B580, int8-perf 노트). sv_key_blocks=8 이라 V dequant 가 k0 당 더 자주 돌기 때문.
- 프로파일(head 128, K+V bias 적용 후, 정적 ISA B580): S*V 가 loop body 의 46% (1027/2247 inst); 남은 mov 는 `pA*vs_c` VNNI `.16` 재조립 64개, bias-trick shr/xor, VNNI B-팩. scale/zp broadcast 는 상수 lane 이면 consuming op 의 source region 에 접혀 mov 0개(사실 확인) → "scale 공유" 는 레버가 아님.
- 상수 lane 의 `sub_group_broadcast` 는 공짜지만 **런타임 lane 은 간접 mov**. 그리고 per-key scale/zp 를 depth/key 루프 안에서 읽으면 lane-uniform 이라 SIMD-1 load 가 되어 k0 당 128~256개 → 반드시 hoist (docs/sdpa_ocl.md:512-518, MIXED 커널의 load 수가 scale/zp 지배였음: `d16u32` 272 vs 데이터 `d8u32` 128).

---

## 7. 동적 양자화와 캐시 쓰기 (수치 경계)

- **`dynamic_quantize` head-size 상한 (T5)**: `kernel_selector/.../dynamic_quantize_kernel_opt_kv_cache.cpp` `Validate()` 가 `input_dims.back().v > 256` 을 거부 → head 512 에서 ref 커널이 선택되는데
  **ref 커널은 append 를 못 해** KV cache 가 깨지고 q=1 iteration 이 틀림. sdpa_ocl 의 V `_8b_32r16x4c` read 를 의심했던 것은 **오진**. 수정: 256 → 512, upstream PR #38466. 교훈: **먼저 `TEST_USE_SDPA_OCL=0`(micro) 로 대조** — micro 도 틀리면 캐시 생산자(양자화)를 먼저 의심.
- **bf16 + compressed KV (14 tests)**: `add_required_reorders` 에서 `dynamicquantize` 의 i8 layout 없음으로 SDPA 선택 전 실패. sdpa_ocl 의 bf16 압축 분기는 이 때문에 실모델에서 도달 불가·무테스트 상태 (T10).
- **PA 압축 경로는 f16 가정**: `pa_v_comp_fold` 가 `as_half8(pA)` 를 쓴다. bf16 PA 는 `PagedAttentionOpt::validate_impl()`(f32/f16 만)가 막고 있을 뿐, `sdpa_ocl supported()` 는 통과시킴 → 호스트 가드 필요(T9 L1, 코드 읽기만 한 상태).
- **int4 plain**: zp 가 query type(비정수)이고 plain 경로에 nibble unpack 이 없음 → dispatch 되면 zp 반올림 이상으로 틀림. 현재는 `sdpa_opt.cpp` 가 `!is_int4_kv` 로 막지만 `supported()` 는 컴파일은 함 (T9 L2/L3).
- writer 의 u4 규칙(K BY_CHANNEL + V BY_TOKEN): `execution_config.cpp:312` 가 4-bit BY_TOKEN key 를 assert → int4 BY_TOKEN K 는 존재하지 않음. u4 only (i4 아님).

---

## 8. bf16

| | f16 | bf16 (`INPUT0_IS_BF16`) |
|---|---|---|
| 저장 | `half` | `ushort` 비트 (jitter 가 bf16→`ushort`) |
| DPAS | `intel_sub_group_f16_f16_matrix_mad_k16` | `..._bf16_bf16_matrix_mad_k16` (packing 동일: `short8` A, `int8` B, `float8` acc) |
| P 저장 | `convert_half8` | `_convert_bfloat168_as_ushort8` |
| 마스크 | `convert_float*` | `_convert_as_bfloat16*_float*(as_ushort*)` |
| int8 dequant | `0x6480` 트릭 + zp+1152 | **float dequant → bf16 encode**, bias 접기 없음 |
| scale/zp 텐서 | f16 | **여전히 f16** (micro 와 동일) |

- bf16 에는 트릭이 없다: ulp=1 구간이 `[128,256)` 뿐(mantissa 7 bit → 128 슬롯)이라 signed byte 256 개가 한 binade 에 안 들어간다. 같은 비트를 bf16 으로 읽으면 1152 가 아니라 ~2^74.
  하이브리드(f16 widen → float → bf16 encode)는 DPAS operand 를 그 비트로 채울 수 없고 convert 가 늘어서 기각.
- **`as_char((uchar)b)` 를 써야 한다**: `(char)(uchar)b` 는 OpenCL `convert_char` 라 128–255 가 **127 로 saturate**(조용한 오답).
- 스칼라 V 로드: `ushort→half` **수치 변환 금지**, `as_ushort(V[...])` 비트 복사.
- `INPUT0_IS_BF16` 분기로 f16 전처리 결과는 byte-identical. `MICRO_MATH` 는 f16 PA prefill 전용이라 bf16 에 안 켠다.
- 검증: 사용자 확인으로 `*SDPAWithKVCacheTest.MultipleIterationStateful*bf16*` 압축/비압축 통과. 성능 수치는 측정 안 함(목표만 합의: ≥ sdpa_micro bf16).
- `SDPAOclGenerator::supported()` 는 이미 bf16 을 통과시켜 왔으나 **커널 산술은 16-bit payload 를 f16 으로 취급**하고 있었다 — 게이트 통과 ≠ 산술 지원.

### 8.1 scale dtype (T8, `2cc6103678`)

런타임 scale 텐서는 bf16 일 수 있다. `SCALE_DATA_T` 가 `half` 로 하드코딩돼 bf16 scale 비트를 half 로 읽어 **오차 3.5546**(f16 0.0003, bf16 const 0.0021).
수정: `runtime_scale_layout()` 에서 dtype 을 읽어 `SCALE_DATA_T` = half|float|ushort + `SCALE_IS_BF16`; `SCALE_TO_FLOAT`: bf16 은 `as_float(((uint)(x)) << 16)`, 나머지 `convert_float` (`sdpa_ocl_config.cl:55-57`). 지원 밖 dtype 은 `supported()` 가 거부.
- 같은 버그가 `sdpa_micro`(`SCALE_DATA_T half`)와 `sdpa_opt.cl:154`(PA 에서 `SCALE_TYPE = INPUT3_TYPE` → f16 scale 을 int32 로 읽음)에도 있음: 문서화만, 수정 안 함(실모델 PA scale 은 Constant 라 미도달).
- 증명: 음성 대조(수정 3 파일 stash) → 새 테스트 8개 중 bf16 4개 FAIL, 수정 후 8/8 PASS (B70, 사용자 확인).

---

## 9. 정확도 검증 방법론

### 9.1 테스트 데이터가 오류를 가린다

| 함정 | 증상 | 대응 | 근거 |
|---|---|---|---|
| PA 하네스 N(0, 0.1), 같은 seed → Q=K=V 같은 배열 | logit ≈ 0.08, softmax 거의 uniform. **Q/K 를 완전히 잘못 읽어도 출력 변화 0.003 (tolerance f16 0.002)**, V 오독은 ~0.1 로 크게 보임 | `logit_scale_gain=128`(상수 scale) 또는 `runtime_scale_multiplier`(런타임 scale, Xe2+) 로 softmax 를 날카롭게 한 복사 케이스를 항상 같이 | T7: Q-only stride 오류 sharp 0.068 vs plain 0.003 |
| gtest LCG 거듭제곱-2 주기 | `InputGenerateData(-1,2,32,seed)` → range*res = 64 = head size → **모든 행이 동일**, logit 동일, softmax 가 scale 을 무시 → 버그 코드에서도 PASS | resolution 을 head size 의 약수가 아닌 값(31, 37, 1000)으로; **LCG 를 10줄 파이썬으로 재생**해 logit spread/버그 효과 확인 후 전달; Q,K,V 에 같은 seed 금지 | T8: spread 0.00(res 32) vs 13.9(res 31) vs 17.1(res 1000), misread 오차 0.85 vs 0.000 |
| sink 값이 분모보다 작음 | sink 제거해도 통과 | sink = log(kv_len) + spread | §3 |
| 임계 6e-3 | bias 상쇄 오류(2.3x)가 안 걸림 | 오차 수치를 기록해 이전 값과 비교 | §5.3 |
| 참조(reference)를 **dequant 된** k_log/v_log 로 구성 | 양자화 오차가 상쇄되어 u4 오차가 2.5e-5..5.3e-4 로 i8 과 같은 band | 임계를 완화하지 말 것 | decode-u4 노트 |

### 9.2 layer-0 동일 입력 oracle

연쇄 네트워크에서 **첫 레이어의 op 만** 두 설정에서 입력이 동일하다. 이후 레이어는 이미 섭동된 입력을 받아 `커널 오차 + 전파` 를 잰다.
minicpm4-0.5b (24 PA layers, f16 출력, relL2 per (token, head), B70):

| pair | L0 mean | L0 max |
|---|---|---|
| pwk8 vs micro | 1e-6 | 5.07e-4 |
| pwk4(기본) vs micro | 2.5e-5 | 6.82e-4 |

출력이 f16 이라 자체 표현 입도가 ~4.9e-4 상대 → **max ≈ 1.0~1.4 ULP, mean ≈ 0.05 ULP**: 인덱싱/커버리지 결함으로는 불가능한 수준. 이후 `3e-5(L0) → 7e-3(L1) → 5e-2(L2) → 0.08 plateau` 는 1600x 증폭이고
정상·버그 설정 모두 같은 plateau → 정보 없음. 자기 일관성 검사: causal 이라 토큰 `[0, tile)` 은 자기에만 의존 → 두 설정에서 bit-identical 이어야 하며 24 레이어 평균 0.0000 이었다(덤프가 비교 가능함의 증명).
**VERDICT: minicpm4-0.5b 의 WWB −0.070 은 sdpa_ocl 버그가 아니다 (재추적 금지).** WWB 유사도는 마지막 비트 변화에 ±0.05 민감 → **커널 회귀 게이트로 쓰지 말 것**; per-op 정확 reference(gtest)를 쓴다.

### 9.3 토글 분류: bit-preserving vs order-changing

| 분류 | 예 | "변화 없음"의 의미 |
|---|---|---|
| bit-preserving | `SDPA_OCL_KV_2D=0`, `_Q_2D=0`, `_A_2D=0`, `_BLOCK_SKIP=0`, `_DKS_ACTIVE=0`, `_256GRF=1` (DKS 의 추가 depth 타일은 f32 에서 정확히 +0.0) | 정답 커널이 예측하는 null result. **증거 아님** |
| order-changing | `SDPA_OCL_KQ_TILE_KEYS/QUERIES`, `_PER_WG_KEYS/QUERIES` (k0 분할 → 누적 순서) | 민감 모델에서 지표를 움직여도 **주사위 재굴림**일 뿐 |

"tiling 노브만 점수를 움직인다" 는 결함의 서명이 아니라 정상 커널의 서명이다(7 runs 허비). 토글을 믿기 전에 먼저 분류. (일부 노브는 이후 리팩터로 제거됨.)

### 9.4 귀인(attribution)

- 한쪽 대조만으로 실패를 귀속하지 말 것: A/B 는 정확히 한 가지만 달라야 한다 (T5 오진: `TEST_USE_SDPA_OCL=0` 을 안 돌림).
- 새 가드/게이트 예측은 **옛 코드에서의 음성 대조**와 함께 (T7: 예측과 달리 모든 16B 패딩 케이스가 옛 게이트에서 통과 → sharp 데이터 추가 후에야 구분).
- "어느 커널이 실제 실행됐는가"를 cliloader/`get_kernels_dump_info()` 로 확인 (게이트가 거절하면 이미 정답인 다른 backend 로 조용히 빠진다 → 초록 ≠ 증명).
- set-equality 논증: 스위치만 켠 run(writer 만 flip, reader 없음) vs reader 토글까지 켠 run 의 FAIL→PASS 집합이 게이트 허용 집합과 정확히 일치.
- 오프셋/nibble 일치 사전 검증 스크립트: `test/check_by_channel_offsets.py`(writer/test fill/reader 2D/gather 4자 일치 + comp 읽기가 comp 영역을 벗어나지 않음), `test/check_u4_offsets.py`, `test/check_u4_page_read.py`(1D `uc16` 페이지 읽기 vs scalar gather 의 집합·원소 동치, 컴파일 타임 상수 인덱스 확인). 모두 Python, 음성 대조 포함.

---

## 10. 수치 함정 표 (요약)

| # | 함정 | 증상 | 예방 | 근거 |
|---|---|---|---|---|
| 1 | `-inf - -inf` | 행/타일 NaN 전파 | `ok = isfinite(m_new)` 가드, alpha=1, exp_tile=0 | `sdpa_ocl.cl:777-795` |
| 2 | partition 전체 마스크 (decode) | exp 폭주/NaN | `-INF` 대신 `VAL_MIN`, finalization 이 `exp(max_logit-global_max)` 로 제거 | decode `:627-640` |
| 3 | 미기록 캐시 슬롯 NaN | 마스크 `+ -INF` 로 안 지워짐 | 로드 높이 clamp, scale=zp=0 강제, V 확률 0×NaN | `qk_load`, decode `:788-803` |
| 4 | 비정수 zp 에 +1152 접기 | zp 정수 반올림 | PA/u4/decode 는 float 로 bias 분리 | §4.5 |
| 5 | `k_corr` 를 float 로 | 오차 2.3x, 테스트 통과 | f16-rounded `Q*sc` 사용 | §5.3 |
| 6 | `(char)(uchar)b` | 128-255 → 127 saturate | `as_char((uchar)b)` | bf16 세션 |
| 7 | 분모에 V scale | 출력 스케일 오류 | scale 은 P 에 접고 분모는 sum(P) | decode `:26-28` |
| 8 | sink 를 모든 subgroup/partition 에 | 분모 중복 계상 | `sg_i_kq==0` / `partition_idx==0` 만 | §3 |
| 9 | sink 도메인 | scale 불일치 | prefill: raw(`*iscale`), decode: Q 가 pre-scaled | §1.3 |
| 10 | causal 상한만 적용 | 틀린 `first/last` | `first`/`last` 를 루프 경계에 맞춤 | §1.2 |
| 11 | lower-right 미적용 | q<k decode 에서 key 대부분 미방문 | `CAUSAL_MASK_LOWER_RIGHT` 3곳 | §2.2 |
| 12 | 동적 마스크 kind 추정 | 1행 마스크 OOB → `CL_OUT_OF_RESOURCES`/NaN | `MSK_D2==1`/`MSK_D3==1` clamp | §2.3 |
| 13 | LOCAL vs KEY 좌표 | bidir 마스크 틀림 | `token_type_ids` 만 LOCAL | §2.4 |
| 14 | 런타임 scale dtype | bf16 scale 오독(3.55) | `SCALE_TO_FLOAT` + layout dtype | §8.1 |
| 15 | writer: range 0 | 0 나눗셈 | `0.004`, rte 반올림, clamp | §5.4 |
| 16 | requantize pitch | 패딩 입력에서 j≥1 오독 | **미수정** (`:297`) | §5.4 |
| 17 | 동적 quantize 256 상한 | head 512 에서 ref 커널이 append 불가 → 캐시 손상 | upstream PR #38466 | §7 |
| 18 | 테스트 데이터 | 오류 은폐 | sharp 복사 케이스, LCG 재생 | §9.1 |
| 19 | 종단 지표로 판정 | 카오스 증폭 오진 | layer-0 + ULP | §9.2 |
| 20 | accumulator 전환 | f16 acc 로 바꾸면 길이에 비례해 오차 증가 | f32 acc 유지 (이 코드베이스는 전부 f32, 변경 실험 없음 — **ASSUMED**) | — |

---

## 11. 확인하지 못한 것 / 노트-코드 불일치

- `native_exp2(-inf) == 0` 의 단독 검증 없음 (§1.2).
- fully-masked 행 출력 0 이 reference(NaN 여부)와 일치하는지 미확인.
- int8-perf 노트의 실측 ns 는 B580(+clock 2800MHz 표기)이며 B70 재측정 없음. 정적 ISA 수치는 시간이 아님.
- "bias trick = 69% mov 감소" 는 마이크로벤치 K dequant 블록 한정; 실 커널 전체 효과는 −3.8%(K) / −9.5%(K+V) (B580).
- 노트에 있는 환경변수 일부(`SDPA_OCL_BLOCK_SKIP`, `_DKS_ACTIVE`, `_PA_CUR_*`, `_V_I8_PAIRED` 등)는 2026-09 리팩터로 제거됨. 위 §9.3 표의 항목은 개념 분류로만 읽을 것.
- `LOWER_RIGHT_SHIFT` 의 괄호 없는 매크로는 현재 사용처에서 안전하지만 잠재 위험.
- T9(bf16 PA 가드, plain int4 가드)는 "분석됨/미착수" 상태 노트이며 현재 코드에 적용됐는지 확인 안 함.
