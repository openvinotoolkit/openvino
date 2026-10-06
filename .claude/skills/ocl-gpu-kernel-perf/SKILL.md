---
name: ocl-gpu-kernel-perf
description: Entry point for high-performance Intel Xe OpenCL kernel work in OpenVINO intel_gpu (sdpa_ocl, DPAS/XMX, 2D block IO, prefetch, split barrier, softmax, int8/u4 dequant, register spill, VTune profiling). Routes to the specialised xe-* / ocl-* skills and the ocl_perf_guide docs. Use first when optimizing or debugging a GPU OpenCL kernel.
---

# Xe OpenCL 고성능 커널: 진입 스킬

가이드 전체: `src/plugins/intel_gpu/docs/ocl_perf_guide/README.md`

## 증상/작업 → 스킬

| 상황 | 스킬 | 챕터 |
|---|---|---|
| 타일링/DPAS operand/SG·WG 배치/GRF 모드 결정, tiling 변경 후 출력 오류 | `xe-dpas-kernel-design` | 01 |
| block2d 읽기 불법/오염, pitch/base 규칙, prefetch, split barrier, KV 페이지 레이아웃 | `xe-block-io-prefetch-barrier` | 02 |
| softmax/mask/sink, int8·u4 dequant(`0x6480`), scale/zp, 정확도 회귀 | `ocl-quant-softmax-numerics` | 03 |
| reference 대비 큰 성능 격차, 원인 불명의 stall/저활용, 비교 대상/측정 유효성 확인 | `ocl-kernel-performance-investigation` | 06, 이후 필요하면 04/05 |
| spill/TPM, 256GRF, ISA dump, VTune source-analysis/GTPin, ocloc splice | `xe-spill-isa-profiling` | 04 |
| A/B, 원인 귀속, 속도 향상 주장, 검증 하네스 | `ocl-kernel-ab-methodology` | 05 |
| DG2/ARL-H/SG8 포팅 | `xe-hpg-porting` | 05 |
| cliloader/VTune/GTPin/ocloc 명령·수집·해석, 독립 하네스, 마이크로벤치 | `xe-profiling-tools-cookbook` | 07 |
| IGC가 load를 sink/패킹이 mov로 풀림 → 인라인 vISA, verifier 오류, asm 오답 | `xe-visa-inline-asm` | 08 |
| 새 커널 시작/성능 계획/게이트 설계, 일반 설계·실험 규율 | `ocl-kernel-dev-lessons` | 09 |

## 공통 원칙 (AGENTS.md 와 작업 규율)

1. 수치 정확성이 성능보다 우선. NaN/Inf, 누산기 dtype, 정밀도 변환, dynamic shape, shape inference 영향을 항상 확인.
2. 성능 개선은 측정 없이 주장 금지. 하드웨어(B580/B70/DG2)·방법론·MEASURED/ASSUMED 를 명시.
3. A/B 는 정확히 한 가지만 달라야 한다. 한쪽 대조군만으로 원인 귀속 금지. 음성 대조군을 먼저 확인.
4. 정적 ISA 지표(명령/메시지 수, spill 예측)는 속도를 보장하지 않는다. 런타임 `SPILL=`/`TPM=` 와 디바이스 측정이 최종 근거.
5. 테스트 데이터가 오류를 가릴 수 있다 (N(0,0.1), LCG power-of-two). 음성 대조군/NaN-poison 으로 검증 민감도부터 확인.
6. 빌드·테스트·벤치·덤프 등 실행은 기본적으로 사용자가 수행한다(정확한 명령 블록 제공). 사용자가 해당 세션에서 직접 실행을 명시적으로 허용했을 때만 직접 실행하고, 그 허용은 다른 세션/작업으로 이어지지 않는다. 2-GPU 환경에서는 `ov_gpu_func_tests --device_suffix=1` (B70) 확인.
7. 범위 확장·기회적 정리 금지. 커밋 메시지는 한 줄.

## 첫 5분 체크

- 어떤 하드웨어 계열인가 (Xe2 SG16 / xe_hpg SG8)? 다르면 `xe-hpg-porting` 먼저.
- 실제 어느 커널/장치가 실행됐고 입력이 device memory에 있는지 확인한다. 비교 arm의 최종 native binary도 확인한다.
- 이유를 모르는 큰 성능 격차: `ocl-kernel-performance-investigation` 의 runtime → profiler → ISA → matched A/B 순서로 조사한다. spill 후보가 보이면 `xe-spill-isa-profiling` 을 적용한다.
- 틀림: 레이아웃/게이트(02), tiling invariant(01), 마스크 종류/OOB(02·03) 순으로 의심하고, 소비자 커널을 탓하기 전에 생산자(예: dynamic_quantize)와 `TEST_USE_SDPA_OCL=0` 대조를 확인.
