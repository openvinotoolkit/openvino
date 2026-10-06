# 01. DPAS/XMX 사용법과 타일링 설계

대상: Intel Xe2(Arc B580/B70, SG16) 및 Xe-HPG(DG2, SG8)에서 OpenCL C 로 DPAS(XMX) 커널을 직접 짤 때 필요한 지식.
출처: `sdpa_ocl` 커널 군 (`src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_ocl*.cl`, 호스트 `.../sdpa/sdpa_gen_ocl*.cpp`,
설계 문서 `src/plugins/intel_gpu/docs/sdpa_ocl.md`) 과 개발 중 누적된 측정 기록.

표기 규칙
- **MEASURED**: 실제 장치에서 측정된 값 (하드웨어 + 출처 표기). **ASSUMED**: 가설/추론. 측정 없는 성능 주장은 하지 않는다.
- 하드웨어: **B580** = Arc B580 (Xe2), **B70** = Arc Pro B70 (Xe2, 클럭 2800MHz 고정, 현재 개발 장비), **DG2** = Arc A770 (Xe-HPG, SG8), iGPU 는 XMX 없음.
  B580 에서 나온 옛 절대값은 B70 에서 재측정되지 않았으면 stale 로 본다.
- 코드 인용은 `path:line` (경로는 `ocl_v2/` 기준 생략 가능한 곳은 파일명만). HEAD 기준 2026-10-06 에 grep 으로 재확인한 위치이며 라인은 이동할 수 있다.
- 다른 장 참조: 메모리/IO/배리어 → `02-memory-io-prefetch-barriers.md`, 수치/양자화 → `03-numerics-softmax-quantization.md`,
  spill/ISA → `04-spill-isa-profiling.md`, 방법론/함정 → `05-methodology-and-pitfalls.md`.

---

## 1. DPAS 빌트인과 레지스터 레이아웃

### 1.1 사용하는 API

| 용도 | 빌트인 (OpenCL, `cl_intel_subgroup_matrix_multiply_accumulate`) | 정의 위치 |
|---|---|---|
| f16 | `intel_sub_group_f16_f16_matrix_mad_k16(A, B, acc)` | `sdpa_ocl_config.cl:36` (`DPAS_MAD_K16`) |
| bf16 | `intel_sub_group_bf16_bf16_matrix_mad_k16(A, B, acc)` | `sdpa_ocl_config.cl:22` |
| 그 외 (미사용) | i8/u8 `k32`, i4/u4 `k64`, split-matrix 변형 | Khronos 스펙; split-matrix 는 DG2 전용이라 필수 경로로 쓰지 않음 |

pragma: `cl_intel_subgroup_matrix_multiply_accumulate`, `cl_intel_subgroup_2d_block_io`, `cl_intel_subgroups`, `cl_intel_subgroups_short`
(`sdpa_ocl.cl:1-6`). 빌드 옵션에 `-Dcl_intel_subgroup_matrix_multiply_accumulate -Dcl_intel_subgroup_split_matrix_multiply_accumulate` 를 넣는다
(`sdpa_gen_ocl.cpp: get_build_options`, ~L1033).

### 1.2 연산 형태 (SG16, Xe2): M=8, N=16, K=16

```c
// sdpa_ocl.cl:621 (KQ), :992 (S*V)
float8 C = intel_sub_group_f16_f16_matrix_mad_k16(short8 A, int8 B, float8 C);
```

| 피연산자 | 타입 (SG16) | lane 의 의미 | 성분(component)/레지스터 의미 |
|---|---|---|---|
| A | `short8` | lane = **K(축약) 인덱스** 1개 (half 1개) | 성분 i = **행 M=i** (8행 고정: `DPAS_ROWS=8`, `sdpa_gen_ocl.cpp:693`) |
| B | `int8` | lane = **열 N** | 8 dword = K=16 half 를 **VNNI**(연속 K 두 개가 한 dword) 로 패킹 |
| C | `float8` | lane = **열 N** | 성분 i = 행 M=i |

핵심 규칙: **모든 피연산자에서 lane 은 그 행렬의 "열 인덱스"** (A 는 K 가 열, B 와 C 는 N 이 열). 이 규칙으로 아래 모든 매핑이 도출된다.
- `DPAS_K = 16` 은 f16/bf16 k16 에서 고정 (`sdpa_gen_ocl.cpp:692`). `DPAS_ROWS` 는 ISA 상 repeat count 이므로 1/2/4/8 만 가능 (decode 가 M 으로 사용, §8).
- 누산기는 **float** (`float8`). half/short 누산기는 SG16 전용이고 이 커널들은 쓰지 않는다. 출력은 `ACC_TO_OUT8` (`convert_half8` /
  `_convert_bfloat168_as_ushort8`, `sdpa_ocl_config.cl:23,38`) 로 에필로그에서 한 번만 내린다 → 누산 중 정밀도 손실 없음.
- **P(softmax 확률)는 S*V 의 A 로 들어가기 전에 입력 dtype(f16/bf16) 으로 반올림**된다 (`PACK_SOFTMAX8`, `sdpa_ocl.cl:822`).
  반면 분모 `lsum` 은 반올림 전 f32 `exp_tile` 합 (`sdpa_ocl.cl:803-806`). 정밀도 변환 비대칭이므로 수치 비교 시 의식할 것 (→ 03장).
- bf16: DPAS 빌트인 이름만 바뀌고 변환 함수 (`DT_FROM_F32`, `MASK_TO_FLOAT*`) 가 per-dtype 매크로로 분기 (`sdpa_ocl_config.cl:21-49`).

### 1.3 SG8 (Xe-HPG / DG2) 형태

| | SG16 (Xe2) | SG8 (DG2) |
|---|---|---|
| A 타입 | `short8` | **`int8`** (lane 당 dword = half 2개) |
| lane 의 K 매핑 | lane l = K 원소 `l` | lane l = K 원소 `2l, 2l+1` (low half = `2l`) |
| B | `int8`, lane = 열 | 동일 형태, N=8, dword j = K `2j, 2j+1` |
| C | `float8` | `float8` (lane = 열 N=8) |
| 정의 | `DPAS_A_T short8` | `DPAS_A_T int8` (`sdpa_ocl_config.cl:12-17`, `#define SG8 (SUBGROUP_SIZE == 8)`) |

- **틀린 SG 폭의 결과 (MEASURED, DG2 A770, `test/sdpa_ocl_xe_hpg/probe/S0_RESULTS.md`)**: SG16 `short8` A 형태를 DG2 에 컴파일하면
  **오류 없이 컴파일 성공하지만 ISA 에 DPAS 가 없다** (입력 load 도 사라지고 store 만 남음; optimized IR 에는 `__builtin_IB_sub_group16_fdpas` 가 남음).
  즉 **조용한 쓰레기 출력**. 초기 예측("컴파일 실패")은 틀렸다. 그래서 `sdpa_ocl_config.cl` 은 SG8 에서 조용히 틀릴 지점마다 `#error` 를 박았다
  (`sdpa_ocl_config.cl:335-339` 등; 문서 `sdpa_ocl.md` "xe_hpg bring-up").
  검출법: 오프라인 `ocloc -device dg2` 컴파일 후 ISA 의 `dpas` 개수가 0 이 아닌지 확인 (`sdpa_ocl_ab.py hpg` 의 `dpas` 열). (→ 04장)
- **SG8 DPAS 매핑 H1 확정 (MEASURED, DG2 A770 IP12.55.8, `test/sdpa_ocl_xe_hpg/s2/S2_RESULTS.md`)**: f16/bf16 × M=1/2/4/8 에서 8/8 항등.
  A: lane l = K(2l,2l+1). B: dword j = K(2j,2j+1), lane n = 열. C: lane = 열, 성분 m = 행. 부정 대조군(NC-A/NC-B)이 예측 순열과 24/24 일치.
  Khronos 문서는 lane 별 원소 표를 주지 않아서 "추론"이었고 실기로 확정했다.
- SG8 에서 `float8` 은 레지스터 8 개(GRF 32B) → 같은 타일이면 lane 당 float 수가 SG16 의 2배 필요 (SG8: 16×16 S 타일 = DPAS 4 개, SG16 은 2개).

---

## 2. 연산자 매핑: 어떤 행렬이 A/B 가 되는가

### 2.1 sdpa_ocl (prefill / mixed) 의 매핑

```
KQ  : S^T[key,query] = sum_d K[key][d] * Q[query][d]
      A = K  (lane = head dim d,   성분 = key)       <- K 행(row-major [key,head])을 block read 하면 그대로 이 모양
      B = Q  (lane = query,        8 dword = d 16개 VNNI)   <- Q_slm 에서 block_read8 (SLM 에 미리 VNNI 로 스테이징)
      C = S_tile float8 (lane = query, 성분 = key)
SV  : O[query,v] = sum_key P[query][key] * V[key][v]
      A = P  (lane = key,          성분 = query)       <- S_slm 에서 8행 block read
      B = V  (lane = value 열,     VNNI)               <- V 토큰-major 페이지를 16b VNNI transform read
      C = A_tile float8 (lane = value 열, 성분 = query)
```
코드: `sdpa_ocl.cl:542` (qB 읽기), `:619-621` (KQ DPAS), `:912-920` (pA 읽기), `:992` (SV DPAS). 문서: `sdpa_ocl.md` "Kernel flow".

**왜 이 매핑인가 (설계 귀결)**
1. KQ 결과의 **lane = query** → softmax 의 query 별 최대/합이 **lane 내부 8 성분 (+ key 블록들) 상의 스칼라 reduce** 가 된다. subgroup shuffle 불필요.
   반대로 A=Q/B=K 로 바꾸면 S 가 전치되어 (lane = key) 최대/합이 `sub_group_reduce` 가 된다 (`pa-k-cache-layout-d-major` 분석; micro 가 실제로 이 쪽).
2. P 는 레지스터 레이아웃(lane = query, 성분 = key)이 SV 의 A 가 요구하는 (lane = key, 성분 = query) 와 **전치 관계**이므로 SLM 을 한 번 경유한다
   (`vstore4` 로 쓰고 `block_read8` 로 읽음; `sdpa_ocl.cl:822`, `:916`). 이것이 S_slm 존재 이유이고 SLM 사용의 핵심 (§7).
3. SV 의 C 는 (lane = value 열, 성분 = query) 라서 online softmax 의 **alpha 재스케일**은 "query 별 스칼라" 를 value 열 lane 들에 **성분 단위로 broadcast**
   해야 한다: `av[rr] = sub_group_broadcast(alpha_sel, alpha_lane0 + rr)` (`sdpa_ocl.cl:855-862`). → alpha[] nesting 불변식(§4)의 근원.
4. B 는 VNNI 고정 피연산자다 (SPV_INTEL_subgroup_matrix_multiply_accumulate: 레이아웃 고정, 전치 플래그 없음, 타입 재해석 플래그만).
   그래서 VNNI 로 읽히는 V (`transform_*` read) 는 반드시 B 로 가고, S 쪽이 A 가 된다. 이는 micro (A=V) 와 전치-대칭 관계이며 결과는 동일 (§2.2).

### 2.2 sdpa_micro 의 매핑 (비교 기준; blob 디스어셈블로 확정, B70 2026-07-09/10)

| GEMM | 선언 | 실제 DPAS | 로드 | VNNI 쪽 |
|---|---|---|---|---|
| KQ (`ugemm_kq`) | A=K, B=Q, `C.layout=T` | `transC` → `problem.transpose()` 로 A/B **교환** → **A=Q, B=K** (M=query, N=key) | K: `load_block2d.ugm.d8` + `d32` transpose, Q: SLM | B(K) |
| VS (`ugemm_vs`) | A=V, B=S, `C.layout=N` | 교환 없음 → **A=V, B=S** (M=v_head, N=query, K=key) | V: `load_block2d.ugm.d8v` (VNNI transform), S: SLM | **A(V)** |

- VNNI 가 어느 피연산자인지는 A/B 역할이 아니라 `globalCM` 이 결정 (`matrix_multiply.cxx:551-557`): `globalCM(VS)` → A 가 VNNI, `!globalCM(KQ)` → B 가 VNNI.
- 두 DPAS 명령은 동일(`dpas.8x8 hf`); 다른 것은 피연산자 **공급 방식**뿐. `bdpas`(dequant 융합) 는 **Xe3p 전용**이라 Xe2/B70 에서 불가.
- 두 매핑이 수학적으로 같다는 것을 `test/microbench/verify_micro_kq_dpas.cpp` 케이스 D/E 로 확인 (maxerr=0, MEASURED B70).
- **micro 의 int8 V dequant "in-place on VNNI stride"** (디스어셈블 확인): `mov`(i8→w, `<4;2,1>:b` region) → stride-2 `add`(zp, 짝/홀 따로) → `mov hf` → `rol 0x10` → stride-2 `mul`(scale).
  VNNI crosspack-2 stride 를 유지한 채 dequant 하여 **repack 0**. OpenCL 은 GRF region/stride 를 제어할 수 없어 IGC 가 `half4 → as_int` 재조립 mov 를 넣는다 → **OpenCL 바닥(floor)**이며
  흉내 불가 (→ 03장 dequant). sdpa_ocl 은 대신 V scale 을 score(P) 쪽에 접어 V dequant 에서 scale 곱을 없앤다 (다른 trade-off; "고치지 말 것").

### 2.3 sdpa_ocl_decode 의 매핑 (q=1, 전부 lane == 그 행렬의 열)

```
KQ  S[key] = sum_d Q[d] K[key][d] :  A = Q (short, M = Q_PER_WG heads, lane = d)   B = K (int8, lane = key)   C lane = key
SV  O[d]   = sum_key P[key] V[key][d]: A = P (short, lane = key, KQ 결과 레이아웃 그대로, shuffle 없음)  B = V (int8, lane = d, VNNI)
```
(`sdpa_ocl_decode.cl:9-20`). prefill 과 **반대**(A=Q, B=K)인 이유: query 가 1개뿐이라 M 축을 GQA 의 q-head 들(`Q_PER_WG`)로 채울 수 있고
(`M ∈ {1,2,4,8}`, DPAS repeat count 로 인코딩), K/V 타일을 한 번 읽어 여러 head 가 공유한다. softmax 는 SLM 타일+alpha 재스케일 대신 subgroup reduce 2회.

### 2.4 K 캐시 레이아웃이 매핑을 강제한다
- PA K 캐시가 d-major(`[head, token]`, token innermost) 이면 coalesced read 는 lane = token 을 주므로 sdpa_ocl 의 A=K(lane = head dim) 가 필요로 하는 축이 아니다
  → K 가 **SIMD-1 스칼라 gather** 가 되어 mixed 커널의 지배 비용이었다 (`pa-k-cache-layout-d-major`, `sdpa-ocl-int8-perf`: int8 prefill ocl 1.69x slower than micro, B580).
  해결: K 를 token-major 페이지로 재배치 (`OV_GPU_PA_K_TOKEN_MAJOR`, BY_CHANNEL 은 `by_channel_token_major_readable()` 가 허용하는 경우; 현재 기본값 범위는 `sdpa_ocl.md` "Paged-attention cache layouts" 확인; → 02장). 트랜스포즈 read 는 32-bit 전용이라 8-bit/16-bit K 를 전치로 못 읽는다.
- u4 K: 바이트가 인접 채널 쌍이라 lane L 이 채널 `(2L,2L+1)` 을 받는다. 축약축 depth 는 **A 와 B 에 동일하게 치환하면 공짜**이므로
  `PA_K_U4_CHANNEL(db,L) = win + 2L + par` 로 depth 를 **순열**하고 Q 스테이징에서 한 번만 deinterleave 한다 (`sdpa_ocl_qk_load.cl:15-40`; `sdpa_ocl.md` "u4 K").
  → "축약축은 두 피연산자에 같은 순열을 적용하면 결과 불변" 은 레이아웃 불일치를 푸는 일반적 도구다.

---

## 3. 일 분배: 서브그룹/워크그룹 타일링

### 3.1 용어와 식 (`sdpa_ocl_config.cl:67-77`)

| 매크로 | 정의 | 의미 |
|---|---|---|
| `kq_sg_tile_keys` / `kq_sg_tile_queries` | 호스트가 JIT 로 주입 | 서브그룹 1개가 KQ 로 계산하는 S 타일 (keys × queries) |
| `kq_sg_per_wg_keys` / `_queries` | 〃 | WG 내 서브그룹 배열 |
| `sg_per_wg` | `pwk * pwq` | WG 당 서브그룹 수 = `reqd_work_group_size(SUBGROUP_SIZE, sg_per_wg, 1)` (`sdpa_ocl.cl:26-27`) |
| `kq_wg_tile_keys` (wgTK) | `tile_keys * pwk` | k0 루프 1 반복이 소비하는 키 수 |
| `kq_wg_tile_queries` (wgTQ) | `tile_queries * pwq` | WG 가 소유하는 query 수 |
| `kq_key_blocks` / `kq_query_blocks` | `tile_keys/8`, `tile_queries/SG` | S_tile[mb][qb] 배열 크기 (float8 × 개수) |
| `sv_sg_tile_values` / `_scores` | S*V 단계에서 서브그룹이 맡는 value 열 / query 수 | |
| `sv_sg_per_wg_values` / `_scores` | S*V 단계에서 같은 서브그룹을 재분할 | |
| `DKS`, `DKS_ACTIVE` | `D_MAX/16`, `ceil(K_HEAD_SIZE/16)` (u4 는 짝수로 올림) | KQ depth 반복 수 (§9) |
| `D_MAX` | `K_HEAD_SIZE` 를 2의 거듭제곱으로 올림 (`get_d_max`, `sdpa_gen_ocl.cpp`) | S*V 분할과 alpha 중첩이 이 값으로 유도됨 → head 72 도 `D_MAX=128` |

- 디스패치 (`sdpa_gen_ocl.cpp: get_dispatch_data_func`, ~L1373): `LWS = {SG, sg_per_wg, 1}`, `GWS[0] = SG * ceil(q / wgTQ)`,
  `GWS[1] *= heads_num` (**헤드마다 별도 WG**), plain SDPA 는 `GWS[2] *= batch`. PA 는 서브시퀀스를 `blocked_indexes_start_and_gws_mapping` 으로 커널 안에서 해석.
  **PA 의 query-block stride 는 반드시 jit 된 `kq_wg_tile_queries` 와 같아야 한다** (`get_query_block_size`; `make_problem` 이 host/jit/dispatch 공통 유도). env 오버라이드도 `choose_config` 를 거치므로 자동 일치.
- **총 서브그룹 수 = (aligned_q / wgTQ) × heads × sg_per_wg**. 이 값이 성능 지표로 의미가 있다 (§6.3).

### 3.2 커널 흐름과 서브그룹 역할 (`sdpa_ocl.cl`)

```c
// sdpa_ocl.cl:118-128 -- KQ 와 S*V 가 같은 sg_ij 를 서로 다르게 해석한다
sg_i_kq = sg_ij % kq_sg_per_wg_keys;   sg_j_kq = sg_ij / kq_sg_per_wg_keys;   // KQ: (key, query)
sg_i_sv = sg_ij / sv_sg_per_wg_values; sg_j_sv = sg_ij % sv_sg_per_wg_values;  // S*V: (score, value)
```
1. Q 를 WG 가 SLM(`Q_slm`)에 협력 스테이징 (타일을 서브그룹에 round-robin: `for (tile = sg_ij; tile < q_blocks*DKS_ACTIVE; tile += sg_per_wg)`, `sdpa_ocl.cl:~330`).
   → `q_blocks*DKS > sg_per_wg` 인 head ≥ 256 에서 1:1 할당이 일부 타일을 비워 **조용한 오답** (C7 버그; round-robin 으로 수정, `sdpa-ocl-headsize-work`).
2. 키 범위 상한/하한 (causal_k, window_k0_begin, bidir 확장; `sdpa_ocl.cl:~380-430`). 이 경계가 없을 때 PA prefill 이 micro 보다 느렸던 근본 원인이었다 (`sdpa-ocl-causal-bound`; q=k=1024, wgTQ=32 에서 256 vs 144 타일 = 1.78x 낭비).
3. k0 루프 (kq_wg_tile_keys 씩): K 타일 로드 → KQ DPAS(depth db 루프) → 마스크 + `lmax` → `__builtin_IB_atomic_max_local_f32(&S_max_slm[query], lmax)` (`:763`) →
   barrier → `exp2` / P→`S_slm` (`vstore4`) / `S_sum` → alpha 재스케일 → S*V DPAS (cp 루프, cp = 16키 = 페이지 1개).
4. 에필로그: `S_sum_slm` 합산 후 `inv_l`, 출력 저장. 완전 마스크된 행은 `l > 0 ? recip(l) : 0` 로 NaN 대신 0.
- **키 축은 reduction 축**이다: `kq_sg_per_wg_keys` 개 서브그룹이 같은 query 집합을 서로 다른 키 구간으로 나누어 계산하고 최대값은 SLM atomic max, 합은 `S_sum_slm` 으로 병합한다
  (`sdpa_ocl.cl:763, 838`). 서브그룹 수나 k0 타일을 키우면 병합/배리어/SLM 왕복이 늘고 유효 일은 안 는다 (§6).
- GQA: prefill 커널은 **head 별 WG** (`b0_kv = b0 / KV_GROUP_SIZE`, `sdpa_ocl.cl: b0_kv`) 라 같은 kv head 를 쓰는 WG 들이 K/V 를 L2 로 공유할 뿐 레지스터/SLM 공유는 없다.
  decode 는 `Q_PER_WG` 개 q-head 를 한 WG 에 넣어 K/V 로드를 amortize (§8).

### 3.3 SLM 크기 (정확한 식, 호스트 `slm_bytes` 와 커널 선언이 동일; `sdpa_gen_ocl.cpp:~148`, `sdpa_ocl.cl:314-317`)

```
Q_slm  = DKS_ACTIVE * q_blocks * 8 * SG * 4B     = D_MAX*wgTQ*2  (DKS 상한 사용)
S_slm  = wgTK * wgTQ / 2 * 4B                    = wgTK*wgTQ*2
S_sum  = wgTQ * pwk * 4B ,   S_max = wgTQ * 4B
```
기본 h128 (wgTQ=32, wgTK=128, pwk=8) = 8192 + 8192 + 1024 + 128 = **17536 B** (MEASURED 일치: cliloader 보고값과 diff 0, `sdpa-ocl-slm-vs-micro`).
호스트 `tiling_fits_device` 가 SLM(Xe2 128KiB / xe_hpg 64KiB)과 WG ≤ 1024 work-item 을 검사하고 `choose_config`(assert) 와 `supported()`(false) 가 같은 함수를 쓴다.

---

## 4. 조용히 오답을 만드는 타일링 불변식

위반해도 **컴파일/실행은 성공**하고 정확도 비교에서만 틀린다 (크래시도 spill 경고도 없음).

| # | 불변식 | 검사 위치 | 위반 시 증상 |
|---|---|---|---|
| 1 | `sv_sg_tile_values * sv_sg_per_wg_values >= V_HEAD_SIZE` (정확히는 `== vd_max`) | 커널 `#error` (`sdpa_ocl_config.cl:112`) + `solve_sv_split` | 일부 value 열 미계산 |
| 2 | `sv_sg_tile_scores * sv_sg_per_wg_scores == kq_wg_tile_queries` | `#error` (`:109`) + `solve_sv_split` | 일부 query 행 미계산 |
| 3 | `sv_sg_per_wg_values * sv_sg_per_wg_scores == sg_per_wg` (= `kq_sg_per_wg_keys * _queries`) | `#error` (`:106`) | `reqd_work_group_size` 는 KQ 쪽이 정하므로, S*V 곱이 다르면 **존재하지 않는 서브그룹이 value 열을 소유**하고 그 출력은 영영 안 써짐. `kq_sg_per_wg_keys` 만 오버라이드한 초기 env 스윕이 이로 인해 **무효 타이밍**을 냈다 |
| 4 | **alpha[] nesting**: 각 서브그룹에서 `sg_i0_sv >= sg_j0_kq` 그리고 `sg_i0_sv + sv_sg_tile_scores - 1 < sg_j0_kq + kq_sg_tile_queries` (⟹ `sv_sg_tile_scores <= kq_sg_tile_queries`) | **`#error` 없음**. 호스트 `solve_sv_split` 가 서브그룹별 루프로만 검사 (`sdpa_gen_ocl.cpp:~53-90`) | alpha 재스케일이 `alpha[]` (자기 KQ query 만 보관) 를 S*V 좌표 `rel_query = sg_i0_sv + r*8 - sg_j0_kq` 로 읽는데 범위를 벗어나 **엉뚱한 query 의 alpha 를 곱함**. MEASURED 실패: (kq_tile_q=16, sv_tile_scores=32), (32, 64) 가 `*paged*96` 정확도 FAIL, 같은 KQ 에 `sv_tile_scores=16` 으로 재유도하면 PASS |

- 4번은 **타일 크기만이 아니라 서브그룹별로** 확인해야 한다 (두 단계가 `sg_ij` 를 다르게 (key,query)/(score,value) 로 매핑하기 때문).
  alpha 선택은 런타임 인덱스 private 배열이 scratch 로 쫓겨나는 것을 피하려 select chain 으로 구현 (`sdpa_ocl.cl:855-858`; 이 변경 하나가 llama-3.2-1b MIXED 에서 `private memory size 128` + k0 루프 안 scratch ld/st 4개를 제거, B70 약 0.8%, `sdpa-ocl-mixed-kc-vc-split`).
- 그 외 구조적 제약: `kq_sg_tile_keys ∈ {16,32}` (`#error`, `:115`; k_mask/mask_tile 이 16키 엔트리 2개까지), `PAGED_ATTENTION_BLOCK_SIZE == DPAS_K` (cp 블록 = 페이지 1개, `:126`),
  u4 는 `DKS_ACTIVE` 짝수, `S_slm` 행 길이 = `DPAS_K` (SG8 에서도 8 dword = block_read8 한 행).
- **`SDPA_OCL_KQ_*` 오버라이드** (`SDPA_OCL_KQ_TILE_KEYS / _TILE_QUERIES / _PER_WG_KEYS / _PER_WG_QUERIES`): 하나라도 설정되면 S*V 분할을 불변식 1-4 로 재유도하고
  (widest `sv_sg_per_wg_values` 우선) 불가능하면 `OPENVINO_ASSERT`. 미설정 시 튜닝 표는 바이트 동일. 타일이 JIT `#define` 이므로 **재빌드 없이** 스윕 가능 (`SDPA_OCL_TRACE_CONFIG=1` 로 선택 확인).
  함정: `setupvars.sh` 가 `set --` 로 위치 인자를 지워서 env 가 실제로 안 들어간 채 "서로 다른 설정"이 동일 결과를 냈던 사고 (→ 05장).

### 4.1 검증된 유효 config 예 (D_MAX=128, wgTQ=64, REG128; `sdpa-ocl-tiling-constraints`)

| tile_q | pwk | pwq | sg/wg | wgTK | SLM(B) | sv (tv,ts,pv,ps) |
|---|---|---|---|---|---|---|
| 16 | 4 | 4 | 16 | 64 | 25856 | 32,16,4,4 |
| 64 | 4 | 1 | 4 | 64 | 25856 | 32,64,4,1 |
| 16 | 2 | 4 | 8 | 32 | 21248 | 64,16,2,4 |
| 64 | 8 | 1 | 8 | 128 | 35072 | 16,64,8,1 |
| 32 | 4 | 2 | 8 | 64 | 25856 | 32,32,4,2 |
| 32 | 8 | 2 | 16 | 128 | 35072 | 16,32,8,2 |

---

## 5. 출고 기본 타일 표 (`choose_config_kq_only`, `sdpa_gen_ocl.cpp:~180-250`)

Xe2 (SG16), `k_head_size == v_head_size`. 모두 `tile_keys=16`. 형식: `(tile_q, pwk, pwq)` → `wgTQ/wgTK/sg_per_wg`; sv=(tv,ts,pv,ps).

| d_max | KQ (tile_q, pwk, pwq) | wgTQ / wgTK / sg | sv (tv, ts, pv, ps) | 비고 |
|---|---|---|---|---|
| ≤ 32 | (32, 8, 2) | 64 / 128 / 16 | 16, 8, 2, 8 | 32폭 head 는 8-way 분할 불가(tv ≥ 16) → query 타일 2배 |
| ≤ 64 | (32, 8, 2) | 64 / 128 / 16 | 16, 16, 4, 4 | "wide micro math" (commit `6b0e442b09`): k 타일을 micro 의 128 과 맞춰 누산 순서 일치 |
| ≤ 128 | (16, 8, 2) | 32 / 128 / 16 | 16, 16, 8, 2 | SLM 17536 B |
| ≤ 256 | (16, 8, 2) | 32 / 128 / 16 | 32, 16, 8, 2 | |
| ≤ 512 | (16, 8, 2) | 32 / 128 / 16 | 64, 16, 8, 2 | 512 초과는 assert |
| xe_hpg | (16, 4, 2) | 32 / 64 / 8 | `solve_sv_split` 유도 | DG2: 256GRF, SLM 64KiB 한도 |

- `k_head_size != v_head_size`: S*V 분할을 `vd_max` 로 재유도. `d_max ≥ 128` 에서 query 타일 32 이면 V head 32폭이 2-way 까지만 되어 score 타일 4/2행(< DPAS 8행 최소)이 된다.
  **Tier 2**: KQ `(16, 4, 4)` (wgTK=64, wgTQ=64) 로 재시도 → 모든 해당 쌍이 `(16,8,2,8)` 로 해결. 이 구멍은 무해하지 않았다: MIXED 폴백 `pa_multi_token` 이 token-major K 를 d-major 로 읽어 **NaN** 이 나왔다 (`sdpa_ocl.md` "Tiling").
- **현재 코드와 일부 메모리 노트의 불일치**: `sdpa-ocl-mixed-kc-vc-split` 은 "config A (`d_max ≤ 64`: tq16/pwk4/pwq4)" 가 출고라고 적었지만 현재 코드는 위 표(pwk8)이며 `6b0e442b09` 에서 되돌려졌다. 표는 코드 기준.
- **아래 §6 의 이기는 config 들 (256GRF + tq32/pwk4/pwq2, A' = tq16/pwk4/pwq4)은 2026-07-30 이후 기본값으로 승격되지 않았다 (코드 확인).** env 로만 도달 가능. 승격 시 §10 절차로 재측정 필요.

---

## 6. 측정된 타일링 결과와 인과

### 6.1 `kq_sg_tile_keys` 16→32 가 느린 진짜 원인 (B70, 2800MHz 고정, head128 v128 i8-compressed prefill q=4096, `sdpa-ocl-kq-tile-keys-32-slower`)

사용자 기대: 8b transform 32행 read (`transform_8b_32r16x1c` 는 32키를 읽는데 16키만 사용) 가 tile_keys=32 에서 효율적이 될 것. **결과는 반대.**

| config | tile_keys | pwk | sg/wg | SLM | occ% | prefill min ns |
|---|---|---|---|---|---|---|
| k16pwg4 | 16 | 4 | 8 | 12928 | 54.7 | **3,526,979** (micro 4.17M 보다 빠르게 보고됨; 아래 주의) |
| k32wg | 32 | 4 | 8 | 17024 | 58.5 | 4,346,666 |
| k16 (기본) | 16 | 8 | 16 | 17536 | 63.6 | 5,102,600 |
| k32 | 32 | 8 | 16 | 25728 | 65.3 | 5,587,600 (+9.5%) |
| k16pwg16 | 16 | 16 | 32 | 26752 | 70.2 | 6,864,687 |

- 독립 두 축, 둘 다 "클수록 느림": 축 A = `kq_wg_tile_keys` (같은 sg_per_wg 에서 64<128<256), 축 B = `sg_per_wg` (같은 wgTK 에서 8<16<32).
  16→32 변경은 wgTK 를 128→256 으로 키운 것(축 A) → +9.5%. `tile_keys` 자체는 거의 중립 (k16pwg4 3.53M < k32wg 4.35M, 같은 sg=8).
- **먼저 세운 "SLM → occupancy 감소" 가설은 틀렸다 (반증됨)**: 총 body 명령 수는 k32 가 오히려 2% 적고 (0.978x), sync 도 적고, spill 둘 다 0 (GRF 128), dispatch 동일 (GWS[2048×256×1], LWS[16×16×1]).
  **cliloader occupancy% 는 이 커널에서 속도와 역상관**(가장 느린 k16pwg16 이 occ 최고 70%). EU 스레드 상주 ≠ throughput (재측정 없는 occ% 신뢰 금지).
- 총 WG 수는 모든 config 에서 2048 로 동일 → grid underfill 이야기가 아니다.
- 메커니즘 (ASSUMED, 구조적 설명): 키 축이 reduction 이라 서브그룹/k0 타일이 클수록 atomic max·barrier·S_slm 왕복·S_tile 라이브 레인지만 늘고 유효 일은 불변.
- ⚠ **미해결 모순**: `sdpa-ocl-tiling-constraints` (07-30) 는 "이 표의 pwg4/pwg2 데이터는 `kq_sg_per_wg_keys` 만 오버라이드해서 불변식 3 을 위반한 무효 config 였으므로 타이밍을 버려라" 라고 기록했고,
  `kq-tile-keys-32-slower` 는 "4개 모두 head=64 ref-check ACCURACY PASS" 라고 기록했다. 두 기록을 같이 만족시키는 설명을 찾지 못했다.
  따라서 **"k16pwg4 가 micro 를 이긴다" 는 미확정**으로 취급하고, 일반 법칙 "작은 WG 가 빠르다" 는 아래 6.2 가 반증한다.

### 6.2 유효 config 로 다시 한 스윕: 16 서브그룹이 이긴다 (B70, 2800MHz 고정, PA prefill `*paged*96` q=1024 32/8 heads head128 f16, causal_k+window 경계 포함, 8회 min 3패스 ±4%, `sdpa-ocl-tiling-constraints`)

| label | sg/wg | wgTQ | wgTK | SLM | min ns | vs micro |
|---|---|---|---|---|---|---|
| default | 16 | 32 | 128 | 17536 | 3,273,229 | 1.087x |
| **A' (tq16, pwk4, pwq4)** | **16** | **64** | **64** | 25856 | **2,268,020** | **0.753x** |
| E (pwk4, pwq2) | 8 | 32 | 64 | 12928 | 2,795,208 | 0.928x |
| C (tq64, pwk4, pwq1) | 4 | 64 | 64 | 25856 | 3,892,187 | 1.292x |
| D (tq64, pwk8, pwq1) | 8 | 64 | 128 | 35072 | 4,189,687 | 1.391x |
| micro | 32 | 128 | 256 | 107008 | 3,011,041 | — |

- 이 형상에서는 **sg/wg=16 (A') 이 8(D,E), 4(C) 를 이긴다** → 6.1 의 "작은 WG 가 빠름" 은 **일반 법칙이 아니라 형상 의존**.
  이기는 조건: **wgTQ=64 와 wgTK=64 를 16 서브그룹 풀 WG 에서 동시에** (causal 효율 50%→66% + 작은 k0 타일). wgTQ=64 유지 + wgTK=128 복귀(D) 는 1.85x 느리고, wgTQ=64 에서 sg 를 줄이면(C) 1.72x 느림.
- 풀 143-test PA 스위트가 A' 로 PASS (모든 head 32-512, prefill/generate/mixed) → 기본값 후보였으나 **승격 안 됨**.

### 6.3 서브그룹 총수의 sweet spot (gpt-oss-20b u4, B70 전체 실행 ns, `sdpa-ocl-u4-head64-page-read`)

| config | 총 서브그룹 | 전체 실행 ns |
|---|---|---|
| 1D page read + 기본 (tq16) | 8192 | 632,343,047 |
| (e) `KQ_TILE_QUERIES=32` | 4096 | **397,934,594** (micro 401,436,373 보다 빠름) |
| (h) (e)+`PER_WG_KEYS=4` (sg 16→8) | 2048 | 427,729,494 |

(h) 는 단위 일당 load 수가 가장 적은데도(80 vs 288 block read / 4096 key-query 쌍) 졌다 → **정적 "message per work" 는 단독으로 틀린 지표**. 스레드 수 반감이 더 컸다.
이 (e) 가 현재 `d_max ≤ 64` 기본값의 `tile_queries=32` 로 반영됨.

### 6.4 MIXED(past 있는 PA) 에서 좁은 키 타일이 이기는 이유 (llama-3.2-1b, past=32 q=1027, u4, B70; `sdpa-ocl-mixed-kc-vc-split`)

| config | sg/wg | wgK | SLM | SPILL | ns/call (page bound 시점) |
|---|---|---|---|---|---|
| default | 16 | 128 | 26880 | – | 153,227 |
| **A** | 16 | 64 | 17664 | – | **146,417** |
| B | 8 | 64 | 17664 | 512 | 157,877 |
| C | 8 | 128 | 26880 | 11776 | 372,628 |

A vs C 가 핵심: **키 타일은 좁히고 스레드는 유지해야** 이긴다. 원인: cache/current 경계 k0 타일 하나가 cache 쪽 블록이 current(Kc) 쪽보다 ~2.2x 느린데 KQ 단계가 WG barrier 로 끝나서
가장 느린 서브그룹 비용을 전원이 치른다. wgK 를 반으로 줄이면 유휴가 반 (ASSUMED 메커니즘 + 측정 일치). V 쪽은 cp 루프가 서브그룹 로컬이라 barrier 가 없어 split 이득이 온전.
B/C 는 **ocloc 가 spill 0 으로 예측했는데 런타임 spill 발생** (§7.3).

### 6.5 256GRF + 타일 쌍 (llama-3.1-8b f16, q=4096 prefill, B70 2800MHz, `_prefill` 장치 시간 3패스 <0.5%, `sdpa-ocl-beats-micro-256grf`)

| config | sg | wgTQ | wgTK | SLM | spill | avg ns |
|---|---|---|---|---|---|---|
| sdpa_micro | – | – | – | – | – | 1,930,454 |
| ocl 기본 (REG128) | 16 | 32 | 128 | 17536 | 0 | 2,181,376 (13.0% slower) |
| **256GRF + tq32/pwk4/pwq2** | 8 | 64 | 64 | 25856 | 0 | **1,794,065** (7.0% faster; e2e 1st token 466.98→462.85 ms) |
| 256GRF tq32 pwk4 pwq4 | 16 | 128 | 64 | 51712 | 0 | 1,874,063 |
| 256GRF tq64 pwk8 pwq1 | 8 | 64 | 128 | 35072 | 0 | 1,890,184 |
| 256GRF 기본 타일 | 16 | 32 | 128 | 17536 | 0 | 2,725,145 (**25% 더 느림**) |
| 256GRF T128k256 | 16 | 128 | 256 | 107008 | 3328 | 4,148,942 |
| 256GRF T128k128 | 8 | 128 | 128 | 70144 | 13952 | 13,795,231 |

- 풀 143-test PA 스위트 PASS. SDPA prefill 은 1st token 지연의 ~10-15% (69.8/475 ms) → 7% 커널 이득 ≈ 0.9% e2e. **커널 단위 시간이 지표**, e2e 는 거의 안 움직임.
- **k0 반복 수는 목표가 아니다**: T128k256 은 반복이 가장 적은데(272 = micro 와 동일) spill 로 2.2x 느림.
- 근본 원인(ISA 영역 프로파일, 기본 타일 k0 루프 body 379 inst): KQ 26.1%, **softmax 47.5%** (causal mask+lmax 61, barrier+S_max 13, exp+lsum 49, S→SLM VNNI store 57), SV prep 12.4%, SV dpas 14%.
  dpas 는 32 inst (8.4%) 뿐. ocl 은 head 당 k0 반복 2112 vs micro 272 (7.76x), 계산한 score 원소는 오히려 2.9% 적음 → **반복당 고정 오버헤드**(atomic fmax → barrier → S_max 읽기, S→SLM, alpha)가 문제. (→ 04장)

---

## 7. GRF 모드와 spill

### 7.1 선택 방법
- `SDPA_OCL_256GRF=1` → `-cl-intel-256-GRF-per-thread` (`SDPAOclGenerator::get_build_options`, `sdpa_gen_ocl.cpp:1041-1042`). **xe_hpg 는 항상 켜짐**(128GRF 에서 DG2 의 모든 측정 타일이 spill).
  decode 는 `SDPA_OCL_DECODE_256GRF`. Xe2 기본은 **128GRF 유지** (기본 타일에서는 256 이 오히려 손해, 아래).
- sdpa_micro 는 마이크로커널이 128 초과를 요구하면 항상 REG256 으로 돈다. 그래서 ocl REG128 vs micro REG256 비교는 처음부터 불공정했고, **큰 타일이 spill 로 막혀있던 것을 256GRF 가 풀었다**
  (`sv_sg_tile_scores >= 64` 인 모든 config 가 128GRF 에서 3.8-19KB spill; 반면 256GRF 에서 spill 0).
- GRF 크기: Xe2 64B, Xe-HPG 32B → DG2 256GRF = 8KB/thread = Xe2 128GRF 와 같은 바이트. 스레드 수는 EU 당 반으로 줄어든다.

### 7.2 256GRF 는 공짜가 아니다 (MEASURED)
| 형상 | 128GRF | 256GRF | 출처 |
|---|---|---|---|
| llama-3.1-8b f16 prefill 기본 타일 | 2,181,376 | 2,725,145 (+25%) | B70, §6.5 |
| gemma-4 head 72 prefill (scalar-gather 상태) | 9,570,971 (spill 2688B) | 14,119,903 (spill 0, **+47% 느림**) | B70, `test/sdpa_ocl_head72_analysis.md` §2.7 |
| decode gemma-4 head512 sg8 | – | 157.1M ≈ sg16 의 158.6M | B70, `sdpa-ocl-decode-tiling-sg-per-wg` |
| decode gemma-4 head256 | 408.9M | 505.3M (손해) | 〃 |

→ 256GRF 는 **스레드 수가 반으로 줄어드는 대가**를 spill 제거 이득이 넘을 때만 이득. 큰 타일(spill 유발)과 **쌍으로** 튜닝해야 한다. head 72 에서 spill 은 원인이 아니라 **증상**이었다
(스칼라 gather 128 개 → LSC message 수가 비용; 로드를 block read 로 바꾸자 spill 2688→0 이 부수효과로 사라졌다).

### 7.3 spill 예측의 한계 (MEASURED)
- `ocloc -device bmg` 의 spill 예측은 **여기서 쓸모없다**: 256GRF 후보 8개가 모두 ocloc spill=0 이었는데 4개가 런타임에 3.3-19KB spill (`sdpa-ocl-beats-micro-256grf`). MIXED config B/C 도 동일.
  **런타임 cliloader 의 `SPILL=` 만 신뢰** (→ 04장). 반면 ocloc ISA 의 명령 수/dpas/send 수는 런타임과 정확히 일치 (k16: dpas 32=32, send.ugm 98=98, inst 3039=3039, 같은 해시).
- decode 는 spill 게이트가 아예 호스트 모델 (`live_grf_estimate`, 예산 112 GRF) 로 구현 (§8.2).

---

## 8. decode 와 MIXED 의 타일링

### 8.1 decode: `SG_PER_WG` 가 M 보다 중요 (gemma-4-26b-a4b-it, B70, 어텐션 장치시간 합 ns, `sdpa-ocl-decode-tiling-sg-per-wg`)

| | head512 | head256 | 합 | e2e |
|---|---|---|---|---|
| pa_opt (기준) | 197.9M | 520.2M | 718.1M | 11.43 ms |
| 구 기본 (sg8, M=8) | 623.2M | – | 1342M | 12.14 ms |
| **신 auto** | **118.3M** | **407.6M** | **525.9M (0.73x)** | 11.40 ms |

- 서브그룹 = 스레드 1개. sg8 은 pa_opt (`LWS[1x1x512]` = SIMD16 스레드 32개) 의 1/4 스레드. M=1 에서 sg8 대비 **sg16 = 2.07x (head512), 1.26x (head256) 빠름, sg4 = 2.2x 느림** (단조, 양방향 가파름).
- **TLP-bound 진단법**: 파티션 수 2p→5p (일 2.5x) 에서 장치 시간이 121,728→122,290 ns (+0.5%) 로 거의 불변인데 기준은 +103% → "일 2.5배를 공짜로 흡수" = idle thread slot 이 있다는 신호.
- 규칙 `get_sg_per_wg`: `(v_head/16 >= 16) ? 16 : 8` (`sdpa_gen_ocl_decode.cpp:142-147`). 이유: S*V 가 서브그룹마다 V head-dim 타일을 갖는데 `SG_PER_WG > V_TILES` 가 되면 남는 서브그룹이 **키 축을 쪼개**
  (`SV_KEY_SGS>1`) `slm_out` 스테이징 + 배리어 1개가 추가됨. head128 (V_TILES 8) 에서 sg16 이 "중립 또는 손해"로 기록된 것과 head 256/512 에서 2.07x 인 것은 모두 사실이며 `V_TILES` 가 판별자.
- **M(=Q_PER_WG) 은 레지스터 한도로 제한**: `q_reg[M][K_TILES]` half 만으로 head512 M=8 이 128 GRF 전체 → **SPILL=34432, pa_opt 대비 3.14x 느림**.
  `live_grf_estimate()` 예산 112 (`sdpa_gen_ocl_decode.cpp:~70-100`). 보정표 (u4 BY_CHANNEL): h512 s8 M=8/4/2/1 = 추정 332/220/164/136 → 측정 spill 34432/15936/6400/0;
  h512 s16 M=2/1 = 126/100 → 640/0; h256 s16 M=2 = 94 → 0. **약 15 GRF 미만 차이는 해상 불가**(136 은 spill 0, 126 은 spill 640) → 거친 게이트일 뿐. 모델은 보수적(h256 sg8 M=2 = 116 은 spill 0 인데도 거부).
- `SDPA_OCL_DECODE_M` 은 레지스터 상한을 우회하지만 SLM clamp 는 우회하지 못한다 (`SDPA_OCL_DECODE_M=8` 이 `kv_group_size < 8` 이면 조용히 무시 → "구 기본 재현"이 아닐 수 있음).
- **diagnosis 가 무재빌드였던 이유**: M / SG_PER_WG / 256GRF / K_2D 가 env 로 노출. 13 개 config 를 재빌드 0 회로 진단.
- 함정: gemma-4 는 llm_bench 에 `-lc '{"ATTENTION_BACKEND" : "PA"}'` 가 없으면 stateful SDPA 경로를 타서 `sdpa_ocl_decode` 가 아예 디스패치되지 않고 모든 env config 가 byte 동일하게 측정됨 (5회 스윕 낭비).
  "A/B 해석 전에 코드 경로가 live 인지 확인" (→ 05장).
- 일반 교훈: **occupancy 는 두 축 (WG 수 vs WG 당 스레드)이며 서로 대체 불가**; "한 형상에서 중립이던 knob 이 죽은 knob 이 아니다" (SG_PER_WG=16 이 head128 에서 중립, head512 에서 2.07x).
- 남은 불확실성: 장치시간 1.37x 이득이 e2e 에서는 parity (pa_opt 가 ~0.18 ms/token 더 얻음, 원인 미규명; 벽시계 29% 가 장치 시간 아님, 823 enqueue/token). 커널 타일링으로 풀 문제가 아닐 수 있음.

### 8.2 decode 의 다른 사실
- 확률 P 는 SLM 에 **key 인덱스 × head 벡터 성분**으로 저장 (head-major 로 저장하면 32B read 64회 → SLM load 44→72, instCount +12%).
- 파티션 page table 은 lane 당 1 chunk 로 한 번 읽고 broadcast (사용처마다 조회하면 scalar load 18개).
- V prefetch distance: 2D block prefetch 로 V 만 (+2.1% llama-3.1-8b head128 M=4, 1.1194e9 → 1.0958e9 ns at dist 4), K prefetch 는 3.6% 손해 (KQ 가 이미 KEY_GROUPS 개 독립 DPAS 체인) (→ 02장).
- "트래픽 2배 → +23%, 명령 6.8%·SLM 트래픽 86% 절감 → +1.2%": 메모리-레벨 병렬성이 한계였고 모든 occupancy knob (SG_PER_WG 2/4/16, 256GRF) 은 중립 이하 (llama head128).

---

## 9. head size 일반성 (32 ~ 512, 72, 80, 48/96)

- 범위: `d_max ≤ 512`, 초과는 assert (`choose_config_kq_only`). head 32 는 테스트에서 일부 실패하나 **sdpa_micro 도 같은 케이스에서 동일 실패** (커널과 무관한 테스트/참조 문제, B580 기록).
- **head 256/512 의 Q 스테이징 버그 (C7)**: 1:1 `sg_ij` 할당이 `q_blocks*DKS > sg_per_wg` 일 때 타일을 비웠음 → round-robin 으로 수정. 수정 후 h64/128/256/512 PASS (B580, `sdpa-ocl-headsize-work`).
- **`D_MAX` 패딩 비용**: head 72 → `D_MAX=128`, depth 타일 8개 중 3개는 완전 0 인데 DPAS 는 실행됨. `DKS_ACTIVE = ceil(K_HEAD_SIZE/16)` 로 **KQ depth 루프와 Q 스테이징만** 줄이고 `D_MAX` 는 유지(S*V 분할과 alpha 중첩이 이 값에서 유도되므로).
  micro 는 `ugemm_kq` 가 축약 길이를 런타임 인자로 받아 이 비용을 안 냄. u4 는 depth 순열이 쌍으로 묶여 `DKS_ACTIVE` 를 짝수로 올림.
- **head 72 사례 (gemma-4 SigLIP 비전 prefill, 16 heads q=k≈2528, B70, `test/sdpa_ocl_head72_analysis.md`, 커밋 `191722ba0e`)**:
  8.11x 느림 (9,570,971 vs micro 1,179,897 ns) → micro 대비 **1.14x 빠름** (1,030,869 ns). 분해: (1) block2d 게이트가 `row_bytes % 64` 로 세 하드웨어 규칙(width ≥64B, pitch %16, base 64B 정렬)을 뭉쳐
  head 72 (144B 행) 에서 전부 꺼짐 → K/V 가 per-lane 스칼라 gather 256개 → `%16` 로 완화 + 커널 안 base fixup (`prem = base & 63`, x 보정, width 확장) → **6.96x** (KV_2D 강제만으로 1.375M ns).
  (2) `DKS_ACTIVE` → 추가 1.33x (-19% 작업 이상: Q_slm 8192→5120, WG SLM 17536→14464 B, 상주 WG 7→9). Q/A 2D 는 prologue/epilogue 라 +0.35% (노이즈).
  교훈: 타일/SLM/occupancy/WG 구성은 두 커널이 byte 동일했고 **원인은 한 줄짜리 과엄격 호스트 게이트**였다 — "타일링을 의심하기 전에 IO 경로가 켜져 있는지 확인".
  역설: `DKS_ACTIVE` 는 **스칼라 폴백 경로의 spill 을 오히려 늘린다** (DKS_ACTIVE 8/6/5 → spill 2688/13568/16064, instCount 7043→5631; ocloc 정적, 장치 미측정; head 48/96 에서만 해당).
- 비-64의 배수 head (80/96/112 등) 는 block2d 사양상 합법이나 우리 헬퍼가 더 엄격 (§ `sdpa-ocl-8b-transform-32row-min`; → 02장). head 486/387 `*ScaledAttn*` 스윕에서 `CL_OUT_OF_RESOURCES` 는 SLM ~41KB 의 스위트 수준 자원 압박(단독 재현 안 됨, 이 변경과 무관): 단, 단독 재현 안 되므로 **양방향(on/off) 대조 없이 귀속 금지**.
- decode: head 가 커질수록 `V_TILES` 가 늘어 sg16 이 무료가 됨 (§8.1).
- head 64 u4 는 행 32B 가 block2d 최소(64B) 미만이라 1D 전체-페이지 read (`uc16`) 로 해결 → gpt-oss 3.92x (`sdpa-ocl-u4-head64-page-read`, → 02장).

---

## 10. SLM vs 무-SLM 설계 (micro 와 비교)

PA prefill head128 (`*paged*96`), B70, **소스 식으로 정확 계산하고 cliloader 값과 diff 0** (`sdpa-ocl-slm-vs-micro`):

| 버퍼 | micro | ocl |
|---|---|---|
| S_slm | 256·128·2 = 65,536 | 128·32/2·4 = 8,192 |
| Q_slm | 128·128·2 = 32,768 | 8,192 |
| S_sum | 8,192 | 1,024 |
| S_max | 512 | 128 |
| ugemm SLM | **0** | n/a |
| 합계 | **107,008** | **17,536** (6.1x 작음) |

- 차이는 **정확히 타일 두 개** (micro: query 128 / k0 256, ocl: 32 / 128). micro 의 GEMM 자체는 **SLM 을 전혀 쓰지 않는다** — K/V 를 global 에서 시스톨릭 어레이로 직접 스트리밍(프리패치만). 107KB 는 100% wrapper 의 Q/S 스테이징.
  "micro 가 SLM 을 많이 쓴다" 는 GEMM 구현 사실이 아니라 **타일 크기** 사실.
- occupancy (Xe2 Xe-core SLM 128KB): micro 는 WG 1개/Xe-core (82% SLM), ocl 은 7개 (ASSUMED 산술; occ% 는 §6.1 대로 성능 지표 아님). 그래도 micro 가 이긴 이유 후보: causal 효율 (80% vs 50%), GRF 256 + spliced ukernel ILP.
- **ocl 이 SLM 을 쓰는 이유**: (1) Q: B 피연산자 VNNI 패킹을 WG 전체가 한 번만 수행 (prologue), (2) S: (lane = query) → (lane = key) 전치 (§2.1), (3) S_max/S_sum: 키 분할 서브그룹 간 병합.
  S*V 의 A 를 SLM 에서 `block_read8` 로 읽는 것은 레이아웃 전치의 값싼 수단이지 GEMM 피연산자 공급이 아님.
- SLM 을 줄이려면 (ASSUMED, 미시도): 서브그룹 간 병합 없이 한 서브그룹이 모든 키를 맡거나 (pwk=1) micro 식으로 S 를 레지스터 내에서 소비. 다만 §6 에서 pwk 를 줄이는 방향이 이미 이득이었다.

---

## 11. xe_hpg (DG2, SG8) 재작성 상세

현재 상태: `TEST_USE_SDPA_OCL_HPG=1` 에서만 xe_hpg 가 sdpa_ocl lane 을 탄다. `kHpgTiersReady = PLAIN_F16_STATIC | PLAIN_EXT | PLAIN_I8` (`sdpa_ocl_hpg.hpp:33`) 이고 나머지는 micro 로 라우팅
(`sdpa_ocl_hpg.hpp`, `sdpa_ocl.md` "xe_hpg bring-up", 최근 커밋 `10e4f00b68`, `087d2bd171`, `b739dc5881`). **DG2 에서의 sdpa_ocl 대 micro 성능 수치는 이 저장소/PC 에 없다** (S6/S7 측정은 DG2 PC 에 있음 — 미확인). 아래는 S0/S2 실기 결과 중심.

### 11.1 하드웨어 사실 (S0/S2, MEASURED DG2 A770 IP12.55.8, driver 25.13.33276.16)
| 항목 | 결과 |
|---|---|
| SG8 DPAS f16/bf16 × M=1/2/4/8 | 16/16 컴파일 성공 (`dpas.8xM`, exec8) |
| SG16 `short8` A | 컴파일 성공 + **DPAS 없음** (조용한 폐기) |
| 2D block IO / prefetch / `cl_intel_subgroup_buffer_prefetch` | **미지원 (컴파일 거부)**. micro 도 xe_hpc 이상에서만 block2d. → 2D 경로는 JIT 스위치로 꺼야 한다 (`#ifdef` 로는 못 끔: pragma 는 경고만, 매크로 미정의; `block2d_io_allowed()`) |
| global uint/ushort block IO | 성공. 단 **홀수 row pitch 에서 `block_read uint` 오답**(err 49), 2B 오프셋 `block_read_us` 오답 → pitch 짝수 + 4B 정렬일 때만 block read, 아니면 폴백 |
| local block IO | 성공 (`addr = base + j*8 + lane`) |
| local fmax atomic, split barrier, sub_group 연산 | PASS |
| `-cl-intel-256-GRF-per-thread` | numGRF 128→256 |
| SLM / max WG | 64KiB / 1024 |
| generic `prefetch()` | 효과 없음 |

### 11.2 SG8 로 새로 짜야 했던 곳은 3군데뿐 (나머지는 lane = query / value 구조라 SG16 과 공유)
1. **KQ 의 K A 피연산자 로드**: `ushort8 k_raw` (lane = head dim) → `int8`, 성분 = key, lane = dword(head dim 2개). `k_tile_dword` (`sdpa_ocl_qk_load.cl:487`):
   키 행당 `intel_sub_group_block_read((const __global uint *)row)` 1회 (8 lane × 4B = 32B). 가드: `dword_ok` (짝수 pitch + 4B 정렬 base, WG 당 1회 증명), `whole_tile` (head_base+16 ≤ d). 꼬리/비정렬은 각 lane 의 두 half 를 정렬된 dword 로 읽는 폴백 (두 `ushort` load 는 DG2 에서 일부 pitch/view 에서 **오컴파일**(query block 1 오답) → dword 로).
   head 꼬리는 K 나 Q **한쪽만** 0 으로 가드하면 충분 (S2). NaN 방어: `d` 이후 half 와 `k` 이후 key 는 0 으로 유지 — Q 가 0 이어도 `0 * NaN = NaN` 이라 K 쪽 0 이 필수.
2. **S*V 의 pA 읽기**: `block_read_us8` → `as_int8(intel_sub_group_block_read8((local uint *)&S_slm[...]))` (`sdpa_ocl.cl:916`). S_slm 한 행 = DPAS_K half = 8 dword = SG8 block read 한 행.
3. **DPAS 호출 A 타입** (`DPAS_A_T`).
나머지(Q staging `vload16` 폴백, V `v_tile_gather`, alpha `sub_group_broadcast`, O 스칼라 저장, sink/bidir/qq_bias/causal) 는 그대로. **softmax 상태가 lane = query 라서 SG8 에서도 동일** — 이 설계 선택이 이식성을 샀다.
- `plain i8`: `k_tile_dword_i8` 이 `(q - zp) * scale` 을 per-element 로 dword A 에 직접 dequant (byte load, 정렬 가정 없음), V 는 공통 스칼라 `v_tile_gather`.
- 음성 대조군: `SDPA_OCL_NEG_SG8=1..6` (K 쌍 순서 교환 / pA 전치 읽기 / S_slm 쌍 교환 / 비정렬 K 폴백 강제(=4, PASS 해야 함) / 마스크 lane 폭 오류) — **각각 sharp-softmax 테스트가 실패해야 하는** 용도. 통과하면 테스트가 그 기능을 관측하지 못한다는 뜻 (→ 05장).

### 11.3 SG8 타일/GRF (S2 microbench, MEASURED DG2; `test/sdpa_ocl_xe_hpg/s2/S2_RESULTS.md` — D=128 미니 SDPA, 시간은 t_med/min µs)
- 256GRF, 16×16 서브그룹 타일, **키 방향 서브그룹 4개** 가 spill 0 으로 최속 (sg4x2 3353 µs min, sg4x1 3400 µs). 128GRF 는 모든 config 에서 3.8~11KB spill (최고 5896 µs).
- k32q16: sg8x2 4553 µs (spill 0), sg4x2 spill 64 B; k16q32 는 256GRF 에서도 spill 21KB(키 방향 4sg) / 96B(8sg); k32q32 는 9.6-12KB spill. → 출고 시드 = KQ 16×16, sg 4×2 (`sdpa_gen_ocl.cpp:~195`; MEASURED 근거가 microbench 라는 점이 한계, 실모델 성능은 미검증).
- K 방향 대안 (D=128 t_med µs): **K1 스칼라 gather 365 (채택)**, K0 연속 286 (정렬 전제), K2 행 vload8+pack 370, K3 SLM 전치 449, K4 A=Q/B=K 344 (KQ 만, 미채택). V: V0 lane=value gather.
- micro xe_hpg 설정(튜닝 시드): `xehpg_h128 = {16,16,32,8, 8,2, 4,4}` (`sdpa_gen_micro.cpp:322`) 는 sdpa_ocl `solve_sv_split` 의 alpha nesting 을 만족.
- 실패 모드 주의: (1) SG8 코드젠 실패는 `add_stage` 가 삼켜 **opt 커널로 조용히 강등**, census 로 확인; (2) PA MIXED f16 는 `USE_2D_BLOCK_IO_*=0` 이어도 `v_tile_b2d16` 이 2D transform 을 무조건 써서 DG2 컴파일 오류 (스칼라 Kc/Vc reader 필요);
  (3) 위조 arch (`OV_GPU_ARCH_OVERRIDE`) 로 덤프한 소스를 B70 에서 ocloc `-device dg2` 로 오프라인 검증하는 것은 가능하지만 **실행 결과는 의미 없음**.

---

## 12. 새 커널의 타일링 config 를 고르는 절차

1. **연산 형태와 피연산자 매핑을 먼저 고정한다.**
   - 각 GEMM 에서 "lane = 열" 규칙으로 A/B/C 의 (lane, 성분) 을 적는다. VNNI 가 필요한 쪽은 B (OpenCL 빌트인 기준) 이므로 VNNI 로 읽히는 입력(V 류)을 B 로, 전치로 읽어야 하는 입력을 로드 가능한 축에 맞춰 A/B 중 선택.
   - reduction(softmax 최대/합 등) 대상 축이 **lane 인지 성분인지** 정한다. lane 이면 shuffle/reduce, 성분이면 레지스터 내 합. 가능하면 reduction 축을 **성분**으로 (여기서는 lane = query).
   - 두 GEMM 사이에 레이아웃 전치가 필요하면 SLM 경유 비용과 불변식(alpha 중첩 같은 "두 단계의 서브그룹 매핑이 다름" 문제)을 이 시점에 도출.
2. **불변식을 호스트 solver 로 코드화** (§4). 타일 곱 = 커버 조건, 서브그룹 수 일치, 서브그룹별 중첩 조건. 가능한 건 커널 `#error` 로, 안 되는 건 solver 단위 테스트로. 오버라이드 경로도 같은 solver 를 통과하게.
3. **자원 모델**로 후보를 거른다: SLM 식(§3.3), WG ≤ 1024 work-item, 라이브 레지스터 추정 (float8 C 개수 = 타일 수, `k_raw`/`qB` 임시). 정적 예측(ocloc spill)은 **참고용**이며 런타임 `SPILL=` 로 최종 판정 (§7.3).
4. **재빌드 없는 knob 을 먼저 만든다**: 타일/GRF/경로 env 노출 + `SDPA_OCL_TRACE_CONFIG` 식 trace 출력 (config 가 실제로 반영됐는지 소스 덤프/trace 로 확인; 동일 결과가 나오면 knob 이 경로에 없는 것). 덤프 소스 `#define` 편집 + ocloc 으로 ISA A/B (→ 04장).
5. **후보 격자**: (총 서브그룹 수 ≈ 수천 단위) × (wgTK 64/128) × (wgTQ 32/64) × (sg_per_wg 8/16) × (GRF 128/256). **GRF 와 타일은 쌍으로** 본다 (§7.2). 서브그룹 수가 너무 적으면(≤ 2048) TLP 부족, 너무 많으면(8192) 오버헤드 (gpt-oss: 4096 최적, ASSUMED 일반화 금지, 형상마다 재측정).
6. **정확도 먼저**: 각 후보를 전체 정확도 스위트(예: 143-test PA)에 통과시킨 후 타이밍. 불변식 위반 config 의 "빠른 타이밍"은 무효 (§4, 불변식 3).
7. **측정**: B70 클럭 고정(2800MHz), 커널 단위 장치시간(cliloader), 3패스 이상 min/avg, 형상(head/q/k/GQA/dtype/causal)을 대표하는 실모델 + 합성 둘 다. SLM/occ% 를 지표로 쓰지 않는다. 한 형상의 결과를 일반 법칙으로 승격하지 않는다 (`V_TILES` 처럼 판별자를 찾는다).
8. **다른 형상에 회귀 확인**: decode/generate, 다른 head, MIXED, 다른 dtype(i8/u4), 다른 arch(DG2). 기본값 변경은 모든 경로의 정확도+성능+dispatch+메모리 회귀 확인 후 (AGENTS.md).
9. **기록**: 실패한 config 도 (config, 장치, 결과, 원인 가설과 반증 여부) 로 남긴다 (§13).

---

## 13. 실패/반증된 타일링 실험 표

| 실험 | 결과 | 원인/교훈 | 장치·출처 |
|---|---|---|---|
| `kq_sg_tile_keys` 16→32 (wgTK 128→256) | 5.10M→5.59M ns (+9.5%) | 타일 자체는 중립, 파생된 `kq_wg_tile_keys` 와 sg_per_wg 가 reduction 오버헤드 증가. 32행 transform "낭비 제거" 는 레버가 아니었음 | B70, kq-tile-keys-32 |
| "SLM 증가 → occupancy 감소" 가설 | 반증 | spill 0, 명령 수 -2%, sync 감소, dispatch 동일. occ% 는 속도와 역상관 | B70, 〃 |
| k32 후보를 sg_per_wg 16→8 로 (k32wg) | 4.35M (k16pwg4 3.53M 보다 느림) | 같은 sg=8 에서 tile16 이 tile32 보다 빠름 (단 6.1 의 유효성 의문) | B70, 〃 |
| `kq_sg_per_wg_keys` 만 오버라이드 | 타이밍 무효 | 불변식 3 위반: 존재 안 하는 서브그룹이 value 열 소유 → 출력 일부 미계산 | tiling-constraints |
| (kq_tile_q=16, sv_tile_scores=32), (32, 64) | `*paged*96` 정확도 FAIL | alpha[] nesting 위반 | B70 |
| tq64 pwk4 pwq1 (C, sg=4) | 3.89M (A' 대비 1.72x 느림) | wgTQ=64 에서 서브그룹 감소 = 스레드 부족 | B70 |
| tq64 pwk8 pwq1 (D, wgTK=128) | 4.19M (A' 대비 1.85x 느림) | wgTQ=64 유지해도 wgTK 128 이 문제 | B70 |
| 256GRF 를 기본 타일에 적용 | 2.18M→2.73M (+25%) | 스레드 수 반감 | B70 |
| 256GRF @ head 72 (스칼라 gather 상태) | 9.57M→14.12M (+47%) | spill 제거해도 LSC message 수가 비용, 스레드 반감이 더 큼 | B70, head72 |
| T128k256 / T128k128 (큰 타일, 256GRF) | 4.15M / 13.80M | spill 3.3KB / 14KB. k0 반복 수 최소(=micro)여도 spill 이 지배 | B70, beats-micro |
| ocloc 로 spill 후보 선별 | 8개 중 4개 런타임 spill | ocloc spill 예측 무용 | B70 |
| block-level causal-mask skip (`causal_block_clear`) | 명령 +16 (cmp 20→36, sel 불변) | IGC 가 분기를 take 하지 않고 flatten. 코드는 남아 있고 무해 (이후 문서에서 "항상 on") | B70 ISA, beats-micro |
| `k_mask` remainder add 제거 | 명령 -3 (add 60→20, mov 59→82) | IGC 가 mov 로 치환, 천장 낮음. 장치 미측정 | ISA |
| decode sg8, M=8 (구 기본) | head512 623.2M vs 118.3M | M=8 가 q_reg 128GRF = spill 34432 | B70 gemma-4 |
| decode sg4 | 2.2x 느림 | 스레드 부족 | B70 |
| decode 256GRF (head256) | 408.9M→505.3M | 스레드 반감 | B70 |
| decode K prefetch | +3.6% 느림 | KQ 가 이미 독립 체인 다수 | B70 llama-3.1-8b |
| MIXED B/C (sg8) | 157,877 / 372,628 vs A 146,417 | 좁은 키 타일 + 스레드 유지가 핵심. spill 이 ocloc 에 안 보임 | B70 llama-3.2-1b |
| u4 MIXED paired Kc DWORD 보관 (장수명 private 배열) | 177,716 (약 12% 악화) | 장수명 private 배열 금지 | B70 llama-3.2-1b |
| u4 MIXED KQ_FAST / KQ_TRIM | 172,310 / 157,932 (vs off 151,746 / 151,529) | 악화, 제거 | B70 |
| h(sg8) at gpt-oss (총 서브그룹 2048) | 427.7M vs 397.9M | 단위 일당 load 최소여도 스레드 부족 | B70 gpt-oss |
| tile_keys=32, pwk≥4, query_blocks≥2 | 오답 (출력 index 35264 = 두 번째 query block 시작) | **원인 미규명**. 메모리 경로 스위치, 256GRF, tiling 불변식, S_slm 용량·충돌 모두 배제. 2026-09-25 현재 `MICRO_MATH=1` 경로에서는 재현 안 됨 (`MICRO_MATH=0` 에서 재확인 필요). tuned 표는 항상 16 키라 override 로만 도달. **tile_keys=32 로 출고 금지** | B70, tk32-bug-hunt |
| 프로브 "DROP-IN CONFIRMED" | 무효 | 두 builtin 모두 0 반환(미기록) 상태 비교 | tk32-bug-hunt |

tk32 디버깅에서 얻은 재사용 방법: (1) 각 env 를 단독/조합으로 돌려 상호작용 격리, (2) 모든 선택적 fast path 를 스칼라로 강제해 인덱싱 vs 메모리 op 분리,
(3) 틀린 출력 index 를 (token, head, dim) → 타일 좌표로 해독, (4) 분석 모델은 **정상 config 로 먼저 검증**, (5) probe 는 sentinel 로 "정말 기록됨" 확인.
Intel subgroup block read 는 **strided**: 성분 i 의 lane L 은 `p[i*SG + L]` (lane-contiguous 가 아님) — 이 오해가 S_slm 레이아웃 분석을 한 번 틀리게 했다 (2026-09-11 정정, 도출값이며 장치 미측정).

---

## 14. 메모리 노트와 현재 코드 사이의 불일치 / 주의

- `sdpa_gen_ocl.cpp:157-164` 를 인용하는 노트(head72, decode)가 있으나 현재 그 위치에는 해당 주석이 없음 (stale 참조; "256GRF 가 스레드 수를 줄인다" 는 사실 자체는 `get_build_options` 의 주석에 남아 있음).
- `sdpa-ocl-mixed-kc-vc-split` 의 config A (`d_max ≤ 64`: pwk4/pwq4) 는 현재 코드에 없음 (§5).
- `SDPA_OCL_BLOCK_SKIP`, `SDPA_OCL_DKS_ACTIVE`, `SDPA_OCL_PA_CUR_GRAN/SIDE/SV_TRIM/V_PREFETCH`, `SDPA_OCL_MAX_BARRIER_V_PREFETCH` 는 2026-09 리팩터로 제거됨 (현재 문서의 knob 표에 없음). 옛 노트의 이 env 를 재사용 가능한 knob 으로 취급하지 말 것.
- `kq_sg_tile_keys` 는 현재 `#error` 로 16/32 만 허용.
- `TEST_USE_SDPA_OCL=0 (default)` 주석이 `sdpa_opt.cpp:~52`, `paged_attention_opt.cpp:~1413` 에서 코드와 반대라는 노트가 있음 (sdpa_ocl 이 기본). 코드 동작 기준.
- 모든 절대 ns 는 해당 줄의 장치 기준이며 B580 수치는 B70 재측정 전까지 stale.
