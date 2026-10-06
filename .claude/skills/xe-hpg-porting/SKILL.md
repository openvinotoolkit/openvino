---
name: xe-hpg-porting
description: Playbook for porting or adapting an Xe2 DPAS/2D-block OpenCL kernel (sdpa_ocl) to Xe-HPG (DG2/Arc-A, ARL-H, SG8, no 2D block IO) in OpenVINO intel_gpu - SG8 vs SG16, HpgTier mask, micro-lane fallback, gates and landmines. Use when touching xe_hpg, DG2, ARL-H, SG8, TEST_USE_SDPA_OCL_HPG, tiers, or sdpa_micro fallback.
---

# Xe-HPG (DG2/ARL-H) 이식 playbook

깊이: `src/plugins/intel_gpu/docs/ocl_perf_guide/05-methodology-and-pitfalls.md` 5.7장 (단계표, DG2 사실, tier 표), 설계 문서 `src/plugins/intel_gpu/docs/sdpa_ocl.md` "xe_hpg bring-up" 절 (:62).
방법론(A/B, 귀속, 성능 규율)은 스킬 `ocl-kernel-ab-methodology`. 커널 기법은 01~04장.
작업 사본/계획: 메모리 `sdpa-ocl-xe-hpg-plan`(허브), `-facts`, `-s7-performance`(최우선), `test/sdpa_ocl_xe_hpg/`.

## 0. 현재 상태 (2026-10-06)
- S0~S6c 완료: plain SDPA f16/bf16/mask/causal/sink/dynamic/q<=1, plain i8 KV가 xe_hpg SG8 arm에서 동작. committed HEAD `d20cac7`의 `kHpgTiersReady`는 `PLAIN_F16_STATIC | PLAIN_EXT | PLAIN_I8`이다.
- 미커밋 작업 트리에는 `PA_PREFILL` bit와 PREFILL/MIXED 개별 routing 및 `KV_TILED` 전처리 prototype이 있다. 이 변경은 standalone 게이트를 통과한 vISA raw-kernel 정책과 다르고, 제품 빌드·정확도·성능으로 검증되지 않았다. 현재 소스에 ready bit가 보이더라도 S7a 제품 준비 완료로 분류하지 않는다. raw-kernel 제품 통합은 별도 후속 작업이다.
- **S9 전까지 xe_hpg 기본 동작은 불변**: `TEST_USE_SDPA_OCL_HPG=1`일 때만 sdpa_ocl lane. 거부된 op는 `TEMP(S9)` 2곳 (`sdpa/sdpa_opt.cpp:86`, `sdpa/paged_attention_opt.cpp:1623`)이 sdpa_micro lane으로 돌린다.
- 실기 검증은 **DG2 (Arc A770)만**. ARL-H(12.74)는 S9 기본값/캐시 태그에 포함되지만 **런타임 정확도/성능 미검증**. B70/Xe2 회귀 + corpus/pset 불변 증명은 DG2-only로 면제되지 않는다.
- S7a PA PREFILL 초기 product path는 head128/32:8/seq4096에서 micro보다 약 24–28배 느린 것이 device-USM ABBA로 확인됐다. 후속 standalone raw K/V kernel은 2026-10-04에 명시된 3% gate를 측정 matrix에서 통과했으나, product integration/end-to-end/S6/SG16/B70/ARL-H 증거가 아니다. 자세한 원인과 결과는 [성능 사례 §6.3](../../../src/plugins/intel_gpu/docs/ocl_perf_guide/06-performance-gap-investigation.md)에 있다.
- S7 이후 성능 우선순위 PA PREFILL > PA MIXED > plain q=1, 이슈별 FIXED/REFUTED/승인된 ACCEPTED·DEFERRED. DG2 실측을 다른 Xe 장치나 미측정 shape로 일반화하지 않는다.

## 1. 하드웨어 차이표 (MEASURED = DG2 S0/S2 실기)
| 항목 | Xe2 (B70) | Xe-HPG (DG2) |
|---|---|---|
| subgroup | SG16 | **SG8** (`get_subgroup_size(arch)` sdpa_gen_ocl.cpp 에서 이미 8) |
| DPAS A 피연산자 | `short8` (lane l=K(l)) | **`int8`**, lane l = K(2l, 2l+1) 한 dword (H1 확정 8/8, 순열 대조 24/24) |
| DPAS B / C | VNNI B, lane=열 | B `int8` dword j=K(2j,2j+1), lane n=열; C lane=열, 성분 m=행 |
| 2D block IO (read/write/prefetch) | 있음 | **없음** (컴파일 거부, 확장 미광고). pragma는 경고만 + 매크로 미정의 -> `#ifdef`가 아니라 jit 스위치로 끈다 |
| GRF | 64 B (128GRF = 8KB) | 32 B (128GRF = 4KB): Xe2 타일이 spill. `-cl-intel-256-GRF-per-thread` 필수로 시작 |
| SLM / WG | 128 KB | **64 KB**, max WG 1024 (호스트에 체크 없었음 -> `tiling_fits_device`) |
| local block IO | 있음 | `local uint/ushort` block read/write는 pragma 없이 OK |
| split-matrix MAD | 있음 | DG2 전용 (ARL-H 없음) -> 필수 경로로 쓰지 않음 |
| decode (PA GENERATE) | `sdpa_ocl_decode` (SG16, 2D block, block_size==SG) | 원래 micro가 아니라 `pa_single_token/pa_gqa`; SG8 `sdpa_ocl_decode`는 범위 밖 (opt 유지, xe2+ reader gate) |
- **SG16 `short8` DPAS를 DG2에서 쓰면 오류 없이 컴파일되고 ISA에 DPAS가 없다** (입력 load도 사라짐) -> 조용한 쓰레기. 모든 SG8 변경에서 `dpas` 개수 확인 (`sdpa_ocl_ab.py hpg` TSV의 `dpas` 열 0이면 실패).
- DG2 `ocloc -device dg2`, ARL-H `-device arl-h` (오프라인, 선택적, 실행 전 승인). `sdpa_ocl_ab.py`의 일반 레벨은 `-device bmg` 고정, `hpg` 서브커맨드만 장치를 파라미터로.

## 2. 이식 구조 (S2~S6에서 확정)
- 공유 커널 + **`#if SG8`** (`sdpa_ocl_config.cl`의 `SUBGROUP_SIZE == 8`). Xe2 경로는 바이트 동일해야 한다 (S4: 키 수를 `SUBGROUP_SIZE`가 아닌 **`DPAS_K` 단위**로 세는 SG16-neutral 리팩터 -> L1 same 2491 + modes 8937, ISA SAME 92+162).
- 레이아웃 의존 재작성은 plain f16 기준 **3곳**: (1) KQ의 K A-operand 로드 (`k_tile_dword`: row당 dword block read, 꼬리/비정렬은 2-ushort fallback), (2) S*V의 pA 읽기 (`as_int8(intel_sub_group_block_read8((local uint*)&S_slm[(cp*WQ+q0)*8]))`), (3) DPAS 호출 A 타입. softmax 상태/alpha/O 저장/mask/sink/bidir/scale/causal/V gather는 lane=query/value라 SG16과 공유 (이식 불필요).
  S_slm 쓰기: `vstore4(uint4)`로 half offset `(key_block16*WQ+query)*16+(key%16)`. alpha는 `intel_sub_group_shuffle`.
- K 방향 선택 (D=128 DG2 미니 벤치, 단위 us): K1 스칼라 gather 365 (채택), K2 vload8+pack 370, K3 SLM 전치 449, K4 A=Q/B=K 344 (KQ만; 미채택), K0 연속 286.
  V0 = lane=value gather. 꼬리 규칙: `db*16+2*idx >= D`이면 dword 0, **K나 Q 한쪽만** 가드.
- 타일/GRF: **256GRF, KQ 16x16, 키 방향 sg 4개 (`sg4x2`), spill 0** (S2 소형 프로토타입). 128GRF는 모든 구성에서 3.8~11KB spill, k16q32/k32q32는 256GRF에서도 spill. 한계: 제품 PA 전체 최적성 증명이 아님. `get_build_options`가 HPG에서 256GRF를 **강제**하므로 `SDPA_OCL_256GRF=0/1`은 128/256 실험이 아님.
- SG8에서 조용히 틀릴 수 있는 곳 (SG8 `#error`로 막음, `sdpa_ocl_config.cl`): 2D block IO 경로 전부, 압축/PA MIXED의 K/V reader, `pa_k_comp_by_token`(토큰 0..7 재읽기), `pa_v_comp_fold`/`v_i8_comp_fold`의 scale(lane당 키 2개 vs half 1개), `mask_tile_2d` half16 폭, S_slm이 `SUBGROUP_SIZE==DPAS_K==페이지(16)`를 가정하는 곳.
- `v_tile_b2d16` (`#if USE_2D_BLOCK_IO_KV || IS_PA_MIXED`)이 PA MIXED f16에서 2D transform builtin을 무조건 사용 -> `USE_2D_BLOCK_IO_*=0`이어도 DG2 컴파일 오류 (S7b가 스칼라 Kc/Vc reader로 대체해야 함).
- UB 게이트: 홀수 row pitch의 `block_read uint`는 오답 (err 49), 2 B 오프셋 `block_read_us`도 오답 -> 연속 reader는 **pitch 짝수 + 4 B 정렬**일 때만 (런타임 `k_dword_ok`), 아니면 fallback.
- d-major BY_CHANNEL K 페이지: 설계상 읽지 못함 -> **DM reader** (S8b) 먼저 (DM-first). u4 nibble 순서는 S8c에서 writer 확인.

## 3. 호스트: 게이트, tier mask, 라우팅
- 술어 분리 (S3): `sdpa_ocl_arch_ok` / `sdpa_ocl_hpg_enabled` / `sdpa_ocl_decode_reader_available`. 옛 단일 술어 `sdpa_ocl_selected = env && immad && arch>=xe2`는 여러 의미를 겸했다.
- **landmine**: `by_channel_token_major_readable()`(paged_attention_opt.cpp)가 `sdpa_ocl_selected`를 "decode reader 있음"의 대용으로 써서, 술어만 넓히면 xe_hpg에서 token-major 페이지가 생기고 d-major GENERATE reader가 예외를 낸다.
- **라우팅 함정**: `sdpa_ocl_selected` true인데 `supported()` false -> `none` (opt 커널). micro tail로 안 떨어짐 (`paged_attention_opt.cpp:1615-1620`). `add_stage`는 codegen 예외를 삼켜 SG8 빌드 실패가 **조용히 opt로 강등** -> census로 stage 존재 확인.
- Tier 비트 (`sdpa_ocl_hpg.hpp:20-29`): committed HEAD에서는 PLAIN_F16_STATIC, PLAIN_EXT, PLAIN_I8만 READY. PA_PREFILL/PA_MIXED_F16/PA_FEATURES/PA_I8_TOKEN/PA_I8_CHANNEL/PA_U4는 제품 게이트 미완료다. 현재 작업 트리의 PREFILL-only routing과 `KV_TILED` pre-pass는 미검증 prototype이며, 별도-pre-pass 없는 raw assembly 측정 결과를 제품에 연결하지 않는다. Tier별 단계 상태는 [성능 사례 §6.3](../../../src/plugins/intel_gpu/docs/ocl_perf_guide/06-performance-gap-investigation.md)에서 다시 확인한다.
- 새 tier 비트를 켤 때: `kHpgTiersReady`에 비트 추가 + `hpg_tier_required` 경계 확정 + 테스트 미러 `expected_dpas_backend_for(...)`에 op별 기대 추가 (tier 0 동안은 "SKIP"이 아니라 **정확한 기대**로 미러) + `sdpa_ocl_serves_prefill_mixed()` 가드 정합.
- `SDPA_OCL_*_2D/_1D/PA_CUR_F16` env는 xe_hpg에서 `block2d_io_allowed(sg)`가 **무시**하고 0으로 강제. tiling은 `tiling_fits_device()` (SLM 64KiB, WG 1024)로 `choose_config`와 `supported()`가 공유 (KQ override도 평가).
- 모델 캐시 descriptor (`compiled_model.cpp:282-303`)에 lane/스테이지 레이아웃/`TEST_USE_SDPA_OCL`이 없어 옛 micro-lane blob이 재사용된다 -> S9 캐시 태그 필요 (xe_hpg DG2 + ARL-H 둘 다; Xe2/기타 arch/no-XMX/oneDNN 없는 빌드의 캐시 문자열은 불변).
- ARL-H 정적 decode 제외 규칙(`sdpa_opt.cpp` `is_ARL_H = gfx 12.74`)은 기본 lane 전환과 별개로 유지. 식별은 `gfx_ver` (`device.cpp`가 12.74를 DG2 12.55~12.57과 별도 행으로 둠).
- 호스트에서 micro-lane 한계는 upstream 그대로: k==v, runtime mask 1원소 불가, unaligned single-token 불가, f32 Q/O 거부, `micro MIXED + bidir tti` 미지원 (opt가 mask 무시).

## 4. 검증 절차
1. **B70에서 가능한 것**: 호스트 라우팅/jit 생성/fake-device 단위 테스트, SG16-neutral 변경의 바이트 동일 증명, 위조 arch 덤프의 오프라인 컴파일:
   ```bash
   OV_GPU_ARCH_OVERRIDE=xe_hpg TEST_USE_SDPA_OCL_HPG=1 SDPA_OCL_HPG_TIERS=all \
     bash test/sdpa_ocl_gtests.sh dump <tag> U1 U3 F2     # ENABLE_DEBUG_CAPS 빌드 (+F1, U2)
   python3 test/sdpa_ocl_ab.py corpus <dir> --base <snapshot>
   python3 test/sdpa_ocl_ab.py hpg --corpus <dir> --base <snapshot> --device dg2 --grf256 [--define SDPA_OCL_SG8_ARM_READY] --out <dir>
   ```
   TSV: rc, first_error, sg, dpas, dpas_forms, simd, numGRF, spill(B), slm, inst, any_2d, any_1d. **`dpas` 열이 in-tier 커널 전부에서 0이 아니어야 한다.** 2D 누수는 `any_2d` + `ARM_READY` 변이 pass2로 확인 (first_error만 보면 가려짐). **위조 arch로 돈 gtest 결과는 해석 금지, 덤프된 소스만 의미.**
2. **DG2 실기 필수**: SG8 정확도, census, 성능. 실행마다 사전 승인 (실행자, workload, 예산, arm/순서/warmup/반복). 새 키트/로그는 `test/sdpa_ocl_xe_hpg/` 아래.
3. 새 SG8 매핑은 **sharp-softmax 음성 대조**와 쌍: `SDPA_OCL_NEG_SG8=1..3` (K pair 교환 / S*V A 전치 / S_slm pair 교환)은 FAIL해야 하고, `=4` (unaligned-K fallback 강제)는 PASS해야 하며 `=5/6` (mask 폭/lane 오프셋)도 대조군. KVC 계열은 `SDPA_KVC_NEG=pair_swap|scale_alias|zp_off`로 **참조를 교란** (전부 FAIL). 그룹은 `test/sdpa_ocl_gtests.sh`의 `HPG_SDPA, NEG_sg8_*, KVC*`.
4. **판정은 census**: tier mask가 거부하는 동안 "HPG=1에서 PASS"는 sdpa_ocl 증거가 아니다. `OV_VERBOSE=4`의 `Enqueue stage`에서 `sdpa_ocl` 줄 >=1을 확인 (S5 DG2: U1 FAIL 0/PASS 328/SKIP 109인데 sdpa_ocl dispatch 0이었음). 단일 gtest 재현은 `TEST_USE_SDPA_OCL_HPG=1 ./bin/intel64/Debug/ov_gpu_unit_tests --device_suffix=1 --gtest_filter=...`.
5. **Xe2 불변**: 새 SG8 코드가 B70 corpus/pset/L1/L2에서 A (또는 의도된 config만 변경). `gtests.sh diff` 14그룹 identical, NEG_* FAIL 수 불변.
6. 기존 DG2 실패 귀속은 **S1/dg2base + 현재 micro/opt 두 통제**: F1 133건은 sdpa_ocl 무관 (bf16 head 486/512 `CL_OUT_OF_RESOURCES` 128, bf16 runtime-scale 4, f16 runtime-scale 1은 **프로세스 내 실행 순서 의존**, 단독 PASS). 이 이름들을 baseline 실패 목록으로 고정하고 S9 후 opt로 떨어진 op가 새 회귀처럼 보이지 않게 한다.
7. **DG2 micro-only 통과 목록** (S9 전 ocl lane에서 전부 PASS여야 회귀 아님): U1 24개 = `micro_sdpa_prefill` 13 + `u4_mixed_micro` 5 + `update_shape` 1 + `sink/{0,1,2,3,6}` 5. 계속 SKIP: `micro_sdpa_prefetch_k` (xe_hpc 미만), `token_type_micro_sdpa_mixed` 6개, `qq_bias_token_major`, runtime-scale (`arch<xe2`).
8. 변이(mutation) 검증: 게이트/미러를 일부러 깨서 예측한 테스트만 FAIL하는지 확인 (변이는 **단독 빌드**, 변수 제거 대신 `&& false`).

## 5. 성능 점검 루프 (S7+)
정확도/실제 dispatch -> 정적 점검 (기존 corpus, 타일/SLM/GRF/private 배열 수명/gather 중복/barrier) -> 승인된 대표 케이스 짧은 스크리닝 -> 트리거 시 한 변경 통제 실험 -> §4.5 회귀 + 사용자 처분.
- 큰 격차는 [ocl-kernel-performance-investigation](../ocl-kernel-performance-investigation/SKILL.md) 순서로 비교 유효성을 먼저 검사하고, VTune/GTPin/ISA를 각각 한 질문에 연결한다. S7a에서 usm_host 입력, 최종 fusion 전의 micro wrapper, 그리고 소스만 보고 예상한 K/V gather 종류는 모두 잘못된 결론을 낼 수 있었다.
- S7a raw-kernel PASS는 vISA inline-assembly standalone harness 결과다. 실제 product dispatch/census, 제품 정확도, end-to-end, S6와 SG16/B70 invariant를 별도로 닫기 전에는 “sdpa_ocl이 micro를 대체했다”고 보고하지 않는다. 미커밋 `KV_TILED`/전처리 branch는 해당 gate와 다른 prototype이며, 최신 체크포인트에서 final raw-kernel 경로에 필요 없는 이전 구현으로 기록됐다.
- 비교축: **현재 OCL vs 현재 실제 micro/opt** (이름이 아니라 실제 stage 확인), 같은 Release 빌드/DG2/driver/입력/캐시. 이전 OCL checkpoint는 별도 축. S1 Debug micro 표는 성능 baseline이 아님.
- ocloc spill (plain h64 7,968 B / h128 12,128 B / h256 28,064 B)은 **정적 보고값**, DG2 런타임 spill/시간이 아니다. 런타임 `CL_KERNEL_SPILL_MEM_SIZE_INTEL` 조회는 가능 (S2 PASS) 하지만 latency가 아니다.
- 공통 SG8 튜닝은 완료된 S6 plain/q=1에 영향을 주면 해당 정확도/성능 회귀 필수.
- 도구: `test/sdpa_perf_hpg.py` (plain f16 정적만, 누적 primitive 평균, warmup 포함, 정수 us -> median이라 부르지 말 것), 기존 PA gtest + CLIntercept (첫 실행/cold 섞임, steady-state 주장 금지). 필요하면 최소 C++ PA 반복 측정 확장을 승인 후.

## 6. 하지 말 것
- tier mask 우회(`SDPA_OCL_HPG_TIERS=all`)로 준비 안 된 reader를 **실행**하지 않는다 (덤프 전용).
- SG16 `short8` DPAS 코드를 SG8에 그대로 두지 않는다 (조용한 DPAS 소실).
- 보지 못한 DG2 결과/수치를 만들거나, 로컬 자료 누락을 단계 미완료로 해석해 완료 단계를 재실행하지 않는다 (정확한 태그/경로/빌드 식별자를 사용자에게 요청).
- ARL-H 런타임 정확도/성능을 DG2 결과나 `ocloc -device arl-h`로 보장하지 않는다.
- 범위 밖: micro 코드 삭제, SG8 `sdpa_ocl_decode`/token-major 새 경로, 승인 없는 새 성능 하한.
