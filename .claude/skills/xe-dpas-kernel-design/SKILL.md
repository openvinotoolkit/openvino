---
name: xe-dpas-kernel-design
description: Playbook for designing Intel Xe2 (SG16) and Xe-HPG/DG2 (SG8) OpenCL DPAS/XMX kernels - A/B/C operand mapping, VNNI, tiling across subgroups and workgroups (sg_per_wg, tile_keys, wgTQ, alpha nesting), 256GRF vs 128GRF, head-size generality. Use for sdpa_ocl, choose_config, DPAS tiling, SG8 port, "wrong output after tiling change".
---

# xe-dpas-kernel-design

DPAS 커널의 피연산자 매핑과 타일링 config 를 새로 정하거나 바꿀 때의 체크리스트.
깊이 있는 근거/표/측정값은 `src/plugins/intel_gpu/docs/ocl_perf_guide/01-dpas-and-tiling.md` (이하 "1장"). 메모리/IO → 02장, 수치 → 03장, spill/ISA → 04장, 방법론 → 05장.

저장소 규칙 (AGENTS.md): 빌드/GPU 테스트/벤치는 **사용자가 실행**한다 (정확한 명령 블록을 제시하고 결과를 받는다). 측정 없이 성능 개선을 주장하지 않는다. 모든 수치에 장치(B580/B70/DG2)와 출처를 붙이고 MEASURED/ASSUMED 를 구분한다. NaN/Inf, 누산 dtype, 동적 shape 를 항상 확인.

## 0. 시작 전 확인 (1분)
- 대상 arch: Xe2 (SG16, `short8` A, 2D block IO 있음) vs xe_hpg (SG8, `int8` A, 2D block IO 없음). 코드 위치: `sdpa_ocl_config.cl:12-17` (`SG8`, `DPAS_A_T`).
- 현재 상태는 소스가 진실: `sdpa/sdpa_gen_ocl.cpp` `choose_config_kq_only`, `solve_sv_split`, `slm_bytes`, `tiling_fits_device`; 커널 `sdpa_ocl.cl`, `sdpa_ocl_config.cl`; 문서 `docs/sdpa_ocl.md`.
- 메모리 노트의 knob 이름은 stale 일 수 있다 (`SDPA_OCL_BLOCK_SKIP`, `_DKS_ACTIVE`, `_PA_CUR_GRAN/SIDE/SV_TRIM/V_PREFETCH` 는 2026-09 리팩터로 제거). 사용 전에 grep.

## 1. DPAS 피연산자 규칙 (모든 매핑의 출발점)
- `float8 C = intel_sub_group_f16_f16_matrix_mad_k16(short8 A, int8 B, float8 C)` (bf16 은 `bf16_bf16`). M=8 행(`DPAS_ROWS`), K=16(`DPAS_K`), N=SG 폭. 누산기는 float.
- **lane = 각 행렬의 열 인덱스**: A 는 lane=K 원소, 성분=행(M). B 는 lane=열(N), 8 dword = K 16개를 VNNI(연속 K 2개 = 1 dword). C 는 lane=N, 성분=M.
- B 가 VNNI 고정 피연산자 (전치 플래그 없음). VNNI 로 읽히는 입력(V)은 B, 반대편(P)이 A.
- SG8: A 가 `int8` (lane l = K 2l, 2l+1, low half = 2l), B dword j = K 2j, 2j+1, N=8. **SG16 `short8` 형태를 DG2 에 쓰면 에러 없이 컴파일되지만 DPAS 가 ISA 에서 사라져 조용한 쓰레기** → 오프라인 `ocloc -device dg2` 후 ISA 의 `dpas` 개수 0 이 아닌지 확인.
- sdpa_ocl 매핑: KQ = A:K(lane=head dim, 성분=key) × B:Q(lane=query) → C lane=query. S*V = A:P(lane=key, 성분=query; SLM 경유) × B:V(VNNI, lane=value 열) → C lane=value, 성분=query.
  이점: softmax 최대/합이 lane 내 성분 reduce (shuffle 없음). 대가: S 를 SLM 으로 전치, alpha 는 `sub_group_broadcast` 로 성분별 재분배.
- micro 는 KQ 에서 A=Q/B=K (전치), VS 에서 A=V/B=S. 결과 동일 (verify_micro_kq_dpas maxerr=0). micro 의 in-place VNNI stride dequant 는 OpenCL 로 흉내 불가 (IGC 바닥).
- decode(q=1): A=Q(M=Q_PER_WG ∈ {1,2,4,8} = GQA head), B=K(lane=key). M 은 DPAS repeat count 라 1/2/4/8 만.
- 축약축(depth)은 A,B 에 같은 순열을 적용하면 결과 불변 → 레이아웃 불일치(u4 인접 nibble)를 Q 스테이징에서 한 번 치환으로 해결.

## 2. 타일링 구조 이해
- KQ: 서브그룹 타일 `kq_sg_tile_keys(16|32) × kq_sg_tile_queries`, WG 배열 `pwk × pwq`. wgTK = tile_keys*pwk, wgTQ = tile_queries*pwq, `sg_per_wg = pwk*pwq`. S*V 는 같은 서브그룹을 `(sv_sg_per_wg_scores × sv_sg_per_wg_values)` 로 재분할.
- 디스패치: `LWS={SG, sg_per_wg, 1}`, 헤드마다 WG, PA query-block stride = jit 된 wgTQ (host/jit/dispatch 가 `make_problem` 으로 공통 유도).
- 키 축은 reduction: pwk 개 서브그룹이 SLM atomic max + `S_sum_slm` 으로 병합 → 서브그룹/k0 타일을 키울수록 병합·배리어 오버헤드 증가.
- SLM = `D_MAX*wgTQ*2 + wgTK*wgTQ*2 + wgTQ*pwk*4 + wgTQ*4` (기본 h128 = 17536 B). Xe2 128KiB / xe_hpg 64KiB, WG ≤ 1024 work-item.

## 3. 조용히 오답을 내는 불변식 (config 후보마다 점검)
1. `sv_tile_values * sv_per_wg_values >= vd_max`. 2. `sv_tile_scores * sv_per_wg_scores == wgTQ`.
3. `sv_per_wg_values * sv_per_wg_scores == sg_per_wg` — `kq_sg_per_wg_keys` 만 바꾸면 깨짐 (그때의 타이밍은 무효).
4. **alpha[] nesting**: 서브그룹마다 `sg_i0_sv >= sg_j0_kq` 이고 `sg_i0_sv + sv_tile_scores - 1 < sg_j0_kq + kq_tile_queries`. 커널 `#error` 없음 → `solve_sv_split` 의 서브그룹별 루프만이 보호. 타일 크기만 보지 말 것.
5. 그 외: `kq_sg_tile_keys ∈ {16,32}`, cp 블록 = DPAS_K = 페이지 크기, u4 `DKS_ACTIVE` 짝수, Q 스테이징은 round-robin (`q_blocks*DKS > sg_per_wg` 인 head ≥ 256 에서 1:1 할당은 오답).
- 오버라이드 `SDPA_OCL_KQ_TILE_KEYS/_TILE_QUERIES/_PER_WG_KEYS/_PER_WG_QUERIES` 는 S*V 분할을 재유도하고 assert (재빌드 없이 스윕). `SDPA_OCL_TRACE_CONFIG=1` 로 실제 반영 확인. `setupvars.sh` 가 위치 인자를 지우므로 env 는 소스 전에 캡처.
- **tile_keys=32 + pwk≥4 + query_blocks≥2 는 원인 미규명 오답** (현재 MICRO_MATH=1 경로에선 재현 안 됨). 출고 config 에 쓰지 말 것.

## 4. config 선택 절차
1. A/B/C (lane, 성분) 표를 쓰고 reduction 축을 성분 쪽에 둔다 (1장 §2).
2. 불변식 solver 를 호스트에 두고 오버라이드도 같은 solver 통과 (§3).
3. 자원 모델로 후보 필터: SLM 식, WG 크기, 라이브 레지스터 (float8 C 타일 개수 + 임시). ocloc spill 예측은 **참고만** — 8개 후보 전부 spill 0 으로 예측했는데 4개가 런타임 3.3-19KB spill. 최종 판정은 런타임 cliloader `SPILL=`.
4. 재빌드 없는 env knob + trace 먼저. 덤프 소스의 `#define` 편집 + `ocloc` 으로 ISA A/B (04장).
5. 후보 격자: 총 서브그룹 수 (`aligned_q/wgTQ × heads × sg_per_wg`), wgTK(64/128), wgTQ(32/64), sg_per_wg(8/16), **GRF(128/256) 를 타일과 쌍으로**.
6. 정확도 스위트 통과 후 타이밍. 클럭 고정(B70 2800MHz), 커널별 장치시간, 3패스 이상, 대표 형상 + 합성 병행.
7. 형상별 일반화 금지: 한 형상의 결과를 법칙으로 만들지 말고 판별자(예: `V_TILES`)를 찾는다. 기본값 변경 전 prefill/mixed/decode, 다른 head·dtype·arch 회귀 확인.

## 5. 측정으로 확정된 규칙 (장치/출처는 1장)
- **GRF**: Xe2 기본 128. `SDPA_OCL_256GRF=1` 은 스레드 수를 반으로 줄여 기본 타일에서 +25% 느림(B70 llama-3.1-8b), head 72 에서 +47% 느림. spill 로 막힌 큰 타일(sv_sg_tile_scores ≥ 64)과 **쌍일 때만** 이득 (256GRF + tq32/pwk4/pwq2 = 1,794,065 ns vs micro 1,930,454, B70 f16 q=4096). xe_hpg 는 항상 256GRF.
- **occupancy%(cliloader)는 속도와 역상관** 가능 → 지표 금지. "정적 message 수 / 일당 load" 도 단독으론 틀린 지표 (gpt-oss: 총 서브그룹 4096 이 최적, 2048 은 졌음).
- **WG 크기 축 vs k0 타일 축은 독립**이고 형상 의존: 키-reduction 오버헤드(큰 sg_per_wg, 큰 wgTK = 느림) vs 스레드 부족(작은 sg_per_wg = 느림). PA prefill q=1024 head128 에서 sg16/wgTQ64/wgTK64 (A') 가 sg8, sg4 를 이김 (2.27M vs 2.80M/3.89M ns, B70).
- **MIXED (cache+current)**: 키 타일을 좁히고(wgK 64) 스레드는 유지(sg16). 경계 k0 타일이 WG barrier 때문에 느린 쪽 비용을 전원이 치름. sg 줄이면 폭락 (372k vs 146k ns).
- **decode**: `SG_PER_WG = (v_head/16 >= 16) ? 16 : 8` (V_TILES 판별자), M 은 `live_grf_estimate` 예산 112 로 제한 (M=8 head512 = spill 34432, 3.14x). TLP-bound 징후: 일이 2.5배인데 시간 +0.5%.
- **head size**: `D_MAX` = 2의 거듭제곱 올림. head 72 는 `DKS_ACTIVE=ceil(d/16)` 로 depth 루프만 줄임 (S*V 분할은 D_MAX 유지). 72 의 8.11x 격차 원인은 타일이 아니라 block2d 게이트(`%64`→`%16` + base fixup) (6.96x) — **타일링 의심 전에 IO 경로가 켜져 있는지 확인**.
- k≠v head: S*V 분할을 `vd_max` 로 재유도, 안 되면 Tier 2 (KQ 16x16, pwk4 pwq4).

## 6. xe_hpg (SG8) 포팅 체크
- 새로 써야 하는 곳은 3군데: K A-operand 로드(`k_tile_dword`: 행당 `block_read(uint)`, 가드 `dword_ok`+whole_tile, 비정렬/꼬리는 정렬 dword 폴백), S*V pA 읽기(`as_int8(block_read8(local uint*))`), DPAS A 타입. softmax 등은 lane=query 라 공유.
- 2D block IO/prefetch 없음 → JIT 스위치로 꺼야 함 (`#ifdef` 불가). global `block_read uint` 는 홀수 pitch/2B 오프셋에서 오답. `d` 이후와 `k` 이후 K 값은 0 (0*NaN=NaN 방어).
- 시드 타일 KQ 16x16, sg 4x2, 256GRF (DG2 microbench: sg4x2 3353 µs min, 128GRF 는 전 config 3.8-11KB spill; MEASURED DG2 S2). 실모델 DG2 성능은 이 저장소에 없음 (DG2 PC) — 주장 금지.
- `SDPA_OCL_NEG_SG8=1..6` 음성 대조군은 각각 sharp-softmax 테스트를 **실패시켜야** 한다. `add_stage` 는 코드젠 실패를 삼켜 opt 로 조용히 강등 → census 로 sdpa_ocl dispatch 가 실제 있는지 확인.

## 7. 흔한 실수
- 불변식 위반 config 의 "빠른" 타이밍을 믿음 (정확도를 먼저).
- env 가 실제 적용됐는지 확인 안 함 (동일 SLM/GWS/LWS 가 나오면 knob 이 안 먹은 것). 경로가 live 인지 확인 (예: `ATTENTION_BACKEND=PA`).
- ocloc spill=0 을 근거로 배포. cliloader `SPILL=` 확인 필수.
- 단일 대조군으로 귀속 (양방향 on/off 필요).
- B580 수치를 B70/DG2 에 재사용.

## 8. 산출물에 포함할 것
선택한 config 표(타일, sg_per_wg, SLM, GRF, spill), 불변식 1-4 점검 결과, 정확도 스위트 결과, 장치/클럭/방법론, 실패한 후보와 이유(1장 §13 형식), MEASURED/ASSUMED 구분.
