---
name: xe-block-io-prefetch-barrier
description: Playbook for Intel Xe2/Xe-HPG OpenCL kernels' memory path - 2D block read/transform/transpose/prefetch legality (width/pitch/base, 8b 32-row, 32b transpose), base fixup gates, padded views, uc16 page reads, V prefetch, split barriers, K token-major KV-cache layout, OOB hazards. Use for sdpa_ocl, block2d, prefetch, barrier_arrive, scalar gather.
---

# Xe 메모리 IO / prefetch / split barrier 플레이북

깊은 근거와 측정 출처: `src/plugins/intel_gpu/docs/ocl_perf_guide/02-memory-io-prefetch-barriers.md` (이하 "ch02").
관련: DPAS/타일링 `01-dpas-and-tiling.md`, 수치 `03-numerics-softmax-quantization.md`, ISA/spill `04-spill-isa-profiling.md`, 방법론 `05-methodology-and-pitfalls.md`.

## 0. 시작 전 규칙 (AGENTS.md)
- 빌드/gtest/ab/dump/벤치는 **사용자만 실행**한다. 정확한 명령 블록을 주고 결과를 받는다 (`--device_suffix=1` 필수, 아니면 iGPU를 측정함).
- 성능 개선 주장 금지: 측정 없으면 ASSUMED로 표기. 장비(B580/B70/DG2)와 방법(cliloader 평균 ns, probe, ISA 계수)을 명시.
- 정적 지표(instCount, 메시지 수, spill)는 접근 패턴이 바뀌면 틀린 지표였다 (ch02 §9.2, §10.2). 메시지 *종류*(block/scatter/transform)와 touched line/sector를 보고, 최종 판정은 시간.
- 수치 영향 점검: NaN/Inf(미기록 slot, OOB 행), 누산기 타입, 동적 shape, padding.

## 1. 2D block builtin을 쓰기 전 체크리스트 (ch02 §2.1)
1. subgroup 16인가? DG2/ARL-H(xe_hpg SG8)는 2D block 자체가 없다 -> `block2d_io_allowed()`로 jit에서 끈다 (`#ifdef` 불가).
2. 변형이 존재하는가? 8b transform은 **32행만** (`16r` 없음), transpose는 **32b만**, 8b 일반 read는 열 수 4의 배수. 불확실하면 `test/microbench/probe_dpas_api.sh`.
3. width >= 64 B (8/16b는 4의 배수), pitch >= width 이고 **16 B 배수**, base 64 B 정렬(또는 fixup), `coord.x` 단위 = 요소(8b byte / 16b half / 32b dword).
4. surface height를 유효 행으로 clamp (미기록 page slot의 NaN, 마지막 k0 타일). height <= 0 read는 발행하지 말고 0 채움.
5. head tail은 surface width = `d*sizeof(T)`로 HW zero-fill. 넓히는 fixup은 **뒤쪽(낮은 주소)**으로.
6. lane/요소 의미는 **프로브로 확인** (K[key][h]=key 채우기). 프로브가 sentinel 없이 all-zero 일치를 보고하는 함정 주의.

## 2. 게이트 선택 (host, `sdpa/sdpa_ocl_utils.hpp`)
| 대상 | 규칙 |
|---|---|
| Q, A(출력), plain i8 K/V (fixup 없음) | strict `row%64==0`, `row>=64` + padding 증명 |
| plain f16 K/V, MIXED Kc/Vc | fixup `row%16==0` + `BLOCK2D_KV_BASE_FIXUP` (round-down 64 B, `x += prem/elem`, `w += prem`) |
| PA 캐시 페이지 f16/i8 | `block2d_page_ok` (`%16`) |
| u4 페이지 | strict 유지 (1D uc16 읽기와 배타. `%16`으로 풀면 입증된 3.92x 경로를 대체) |
| decode 페이지 | strict `%64` (f16 `h%32`, i8 `h%64`, u4 K `h%128`) |
- **fixup flag는 `SDPA_OCL_*_2D` override 뒤에 파생**해야 override가 안전하다. `SDPA_OCL_KV_2D=1`은 rebuild 없는 가설 증명 레버.
- Q/A에는 fixup이 없다 (32b transpose는 `prem%4`, 비용도 무의미): **강제 2D는 오답 위험**.
- rank-4 padded: simple format && X unpadded이면 통과 (pad는 pitch를 곱할 뿐). rank-2 PA 토큰 행렬: static이면 first_head/stride 검사, **dynamic padding은 증명 불가** -> strict false, fixup은 전제(stride%16, start%4)를 신뢰.
- B70 측정: 오답을 내는 건 token stride가 16 B 배수가 아닐 때와 시작 2 B 어긋남. 4/16/32/48 B base는 정상(디바이스 테스트는 pitch만 증명, base 규칙은 host 테스트 `sdpa_block2d_gate`만).

## 3. 패턴 카탈로그
- **8b i8/u4 K/V**: `transform_8b_32r16x1c`가 lane=열, uint=행 4개. 페이지(16 토큰)는 height를 clamp하고 uint 0..3만 쓴다 (50% over-read는 HW/OpenCL 한계, micro는 descriptor 직접 제어로 0). read 수를 줄이려는 cp-pair 재사용은 3회 모두 시간 개선 없음.
- **i8 K transpose**: `transpose_32b_16r8x1c`에 byte width/pitch를 주면 i8 surface를 dword로 본 transpose가 된다 (`sdpa_ocl_decode.cl:547`). 8 dword = 32 head dim = DPAS 타일 2개 -> K 메시지 절반.
- **u4**: byte=채널 쌍 -> A/B에 같은 depth 순열(`PA_K_U4_CHANNEL`), Kc는 DWORD surface(`read_32b_8r16x1c`), `kc_dword_ok` 런타임 검사 + scalar 폴백.
- **head 32/64 u4 (row<64 B)**: `intel_sub_group_block_read_uc16` 1D 전체 페이지 읽기 `PA_PAGE_R/I(ROW,t,c)`. `r`은 컴파일 상수, 런타임 `c`는 base를 `SUBGROUP_SIZE*c` 편향 + c=0 (`uchar16` subscript가 indirect addressing이 되는 것 방지). per-key 가드는 유한성만 필요하면 제거 (gpt-oss 3.92x, B70).
- **per-key scale/zp**: innermost에서 읽으면 lane-uniform이라 SIMD-1 로드가 key마다 발생. k0 상단에서 lane=key wide load 1회 + `sub_group_broadcast`(컴파일 상수 lane). `block_indices[]`도 hoist.
- **SLM**: micro의 GEMM은 SLM 0, wrapper의 Q/S staging이 전부. SLM 크기는 타일 크기의 함수이며 점유율 헤드룸이 곧 처리량은 아니다. `slm_p`는 key-indexed + head가 innermost vector.
- **DG2(xe_hpg)**: local block IO 동작, global `block_read uint`는 4 B 정렬 + 짝수 pitch일 때만 (홀수 pitch 오답). SG16 `short8` DPAS는 오류 없이 DPAS를 버림 (ISA에 `dpas` 있는지 확인).

## 4. Prefetch
- 쓸 곳: **독립 체인이 없어 로드 지연이 노출된 루프** (decode S*V의 16 페이지 단일 accumulator 체인). 이미 K처럼 독립 DPAS 체인을 인터리브하는 루프에는 해롭다 (K prefetch -3.6%).
- 거리 비단조 (decode, 1/2/4/8/16 = -1.45/-1.87/**-2.10**/-0.56/**+2.21 %**). 거리는 *어디서* 발행하느냐만 바꾸고 개수는 같다. 거리 > ~4이면 L1 evict.
- 발행 지점은 실제 읽기와 **같은 surface/fixup/좌표** 사용. 소비 직전 prefetch는 오버헤드만 (plain prefill 약 11 us 손해, 제거됨).
- 비용: prefetch당 약 8 명령 (a64 descriptor 세팅). 목적지 레지스터가 없으므로 점유율로 살 수 없는 MLP를 준다.
- MIXED Vc prefetch (u4, softmax 직전, `!from_cache && sg_i_sv==0`): -4.12% (B70 llama-3.2-1b, cliloader). i8 BY_CHANNEL MIXED에서는 미측정.

## 5. Split barrier
- 쓰는 곳: `arrive` 와 `wait` 사이에 **그 barrier가 보호하는 SLM을 읽지 않는** 독립 작업 (alpha로 A_tile rescale `sdpa_ocl.cl:842-868`, V 첫 타일 prefetch `sdpa_ocl_decode.cl:727-732`).
- barrier 개수/배치를 바꾸지 말 것. 조건부 skip(`SV_TRIM`의 `continue`)은 barrier를 루프 밖에 둔다.
- 측정된 단독 효과: decode V prefetch 이득 2.1% 중 0.7%p. plain prefill split barrier 단독 효과는 미측정(ASSUMED). `MAX_BARRIER_V_PREFETCH`는 이득 미입증으로 삭제됨.

## 6. KV 캐시 레이아웃 결정
- K d-major 페이지(`[head, block]`)는 row=32 B < 64 B 라 block read 불가 -> per-key SIMD-1 scalar gather. token-major `[block, head]`로 바꾸면 prefill 지오메트리 재사용 (ISA: head 128 mixed 총 op -49%, gather 128->0).
- 레이아웃 바꾸는 순간 **모든 소비자**(writer, rotate, reorder, adaptive_rkv, decode, MIXED, 테스트 harness)가 영향. green suite != layout-correct: 캐시를 되읽는 테스트가 있는지 확인 (rotate 512쌍 오염이 기본 suite에서 안 보였음).
- 결정은 한 곳(`transformations_pipeline.cpp`), 소비자는 캐시 물리 shape에서 파생. 불일치는 throw/거부.
- i8 BY_TOKEN row pitch는 `h` (`h+4` 아님: +4는 끝의 scale/zp 배열). u4 K row는 `h/2` 정렬 안 함 (`12h` 페이지가 INT4 할당에 byte 단위로 맞음).
- token-major의 대가: kvup writer가 thread당 16 sector를 dirty (d-major 2). u4 head-256 generate +12% (MEASURED B70), 4개 수정 모두 측정으로 기각 (sector 공유 WG, unroll, partition 수 축소). store 비용 ~ 총 sector write 수.
- writer 최적화 함정: lane-varying 루프 변수 -> register 소실(1.8x), 가드는 **call site**에 (helper 본문은 unroll 소실, 2.3x), 타입 승격을 식 단위로 보존.

## 7. OOB / 동적 shape 위험
- 2D block OOB = HW 0 채움, **global scalar OOB = CL_OUT_OF_RESOURCES 또는 쓰레기** (Xe2). 
- 동적 mask는 JIT에서 kind를 추론하므로 런타임 `MSK_D2==1`/`MSK_D3==1`을 커널에서 다시 검사 (static은 fold되어 코드젠 동일).
- 미기록 slot의 NaN: 점수는 -INF로 가려져도 V 쪽은 scale AND zp를 0으로 강제.
- 한 k0 타일 안에서 K/V 소스가 subgroup별로 갈리면(page 분할) V만 오답이 났다. **WG-uniform한 exact split + chunk clip**으로 해결 (메커니즘은 미규명).
- 테스트 데이터가 오독을 가린다: N(0,0.1)은 softmax가 균일해 Q/K 오독 불가시 (`logit_scale_gain`), 2의 거듭제곱 LCG는 행이 전부 동일. 오답 판정 전 데이터 민감도부터 확인.

## 8. 검증 절차 (사용자 실행 명령 준비)
1. 가설은 `SDPA_OCL_*_2D=0/1` env A/B (rebuild 없음)로 먼저. 같은 build/workload에서 **한 가지만** 바꾼다. 양쪽 control 필수.
2. ISA/spill 비교는 ocloc splice (rebuild 없음, `04-spill-isa-profiling.md`).
3. 정확성: `ov_gpu_unit_tests --device_suffix=1` (`smoke_paged_attention/*`, `paged_attention_feature_pad_test`, `sdpa_block2d_gate`), 실제로 어떤 커널이 돌았는지 cliloader로 분류.
4. 성능: cliloader 평균 ns + 호출 수, 장비 명시. probe는 큰 비율만 신뢰 (작은 비율은 실모델로).
5. 보고 형식: 측정/가정 구분, 회귀 위험(동적 padding, DG2, pre-Xe2 fallback), 테스트 커버리지 공백.

## 9. 흔한 오판 (재도출 금지)
- "`load.ugm.d32x8t (1|M0)`은 scalar gather" -> 아니다, 한 주소가 32 B를 lane들에 분산하는 block load.
- "read 수를 줄이면 빠르다" -> i8 V/K에서 3회 실패; geometry와 dequant mov가 병목이었다.
- "spill을 없애면 빠르다" -> 256 GRF가 head-72 +47%, u4 head-64 +70% 느림.
- "점유율/SLM 헤드룸이 처리량" -> 두 번 빗나갔다.
- "d-major는 block read 불가" -> 재해석 surface(80 B pitch)로 이론상 가능하나 depth 순열 + **HW 미검증**.
