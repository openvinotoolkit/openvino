# Intel Xe 고성능 OpenCL(DPAS) 커널 개발 가이드

`sdpa_ocl` (prefill / MIXED / decode, PA + plain SDPA, f16/bf16/i8/u4 KV-cache) 개발 전 과정에서 얻은 기법, 제약, 실패한 실험, 검증 방법을 정리한 문서 모음. Xe2/B70의 제품 성능 조사와 DG2 PA PREFILL standalone kernel gate 결과는 06장에, 분석 도구 사용법은 07장, inline vISA는 08장, 다른 커널에 옮길 일반 교훈은 09장에 정리했다.
근거는 프로젝트 메모리 노트와 세션 로그, `test/` 의 handoff/분석 문서·스크립트, 그리고 커널 소스(`src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_ocl*.cl`)다.
모든 수치는 챕터 안에서 하드웨어(B580/B70/DG2)와 MEASURED/ASSUMED 여부를 표기한다. 하드웨어가 노트에 명시되지 않은 경우 "B70 추정"으로 적혀 있다.

## 챕터

| 챕터 | 내용 | 대응 스킬 |
|---|---|---|
| [01-dpas-and-tiling.md](01-dpas-and-tiling.md) | DPAS API, operand 매핑(lane = column), SG/WG 타일링, 4개 tiling invariant, 256GRF, head-size 일반화, SG8(xe_hpg) 재작성 | [`xe-dpas-kernel-design`](../../../../../.claude/skills/xe-dpas-kernel-design/SKILL.md) |
| [02-memory-io-prefetch-barriers.md](02-memory-io-prefetch-barriers.md) | 2D block IO 합법성 규칙(width/pitch/base), transform/transpose 제약, base fixup, prefetch, split barrier, K token-major 재배치, uc16 page read | [`xe-block-io-prefetch-barrier`](../../../../../.claude/skills/xe-block-io-prefetch-barrier/SKILL.md) |
| [03-numerics-softmax-quantization.md](03-numerics-softmax-quantization.md) | online softmax, mask/sink/causal, `0x6480` XOR 트릭 유도, u4 unpack, dequant identity, 테스트 데이터 함정 | [`ocl-quant-softmax-numerics`](../../../../../.claude/skills/ocl-quant-softmax-numerics/SKILL.md) |
| [04-spill-isa-profiling.md](04-spill-isa-profiling.md) | spill/TPM 진단, ISA dump·ocloc splice·iga64, `always_inline`, 정적 ISA의 한계 | [`xe-spill-isa-profiling`](../../../../../.claude/skills/xe-spill-isa-profiling/SKILL.md) |
| [05-methodology-and-pitfalls.md](05-methodology-and-pitfalls.md) | A/B 하네스, attribution 규칙, 성능 주장 규율, 대표 결과 표, 타임라인, 28개 pitfall, xe_hpg(S0-S9) | [`ocl-kernel-ab-methodology`](../../../../../.claude/skills/ocl-kernel-ab-methodology/SKILL.md), [`xe-hpg-porting`](../../../../../.claude/skills/xe-hpg-porting/SKILL.md) |
| [06-performance-gap-investigation.md](06-performance-gap-investigation.md) | B70의 9.28x/7% 개선, DG2 초기 24–28x 격차의 프로파일링·A/B·raw-kernel 결과, 도구별 해석 경계와 다른 GPU 커널에 재사용할 조사 절차 | [`ocl-kernel-performance-investigation`](../../../../../.claude/skills/ocl-kernel-performance-investigation/SKILL.md) |
| [07-profiling-tool-cookbook.md](07-profiling-tool-cookbook.md) | cliloader·VTune(overview/lsc-slm/source-analysis)·GTPin·IGC dump/ocloc/iga64·cycle counter·마이크로벤치·standalone 하네스의 명령, 해석 한계, 실제로 겪은 함정, 시간 측정 규율 | [`xe-profiling-tools-cookbook`](../../../../../.claude/skills/xe-profiling-tools-cookbook/SKILL.md) |
| [08-inline-visa-asm.md](08-inline-visa-asm.md) | 인라인 vISA(`__asm__` + `.decl`/alias/`lsc_load`/`dpas`)를 쓴 이유·문법·검증된 패턴, verifier 제약, 깨졌던 16가지와 검증 프로토콜 | [`xe-visa-inline-asm`](../../../../../.claude/skills/xe-visa-inline-asm/SKILL.md) |
| [09-general-lessons-for-new-kernels.md](09-general-lessons-for-new-kernels.md) | 병목 진단 결정 트리, 효과 크기 감각, 설계 원칙 10개, 정확도·실험 관리 규율, 도구 함정 표, 새 커널 시작 체크리스트 | [`ocl-kernel-dev-lessons`](../../../../../.claude/skills/ocl-kernel-dev-lessons/SKILL.md) |

진입 스킬: [`ocl-gpu-kernel-perf`](../../../../../.claude/skills/ocl-gpu-kernel-perf/SKILL.md) (아래 기법 → 챕터 지도).

## 기법 → 위치 지도

| 기법 | 챕터 |
|---|---|
| DPAS (`intel_sub_group_*_matrix_mad_k16`), VNNI, SG8 vs SG16 | 01 |
| subgroup/workgroup 블록 할당 (`sg_per_wg`, `kq_sg_tile_keys`, `wgTQ/wgTK`, `V_TILES`) | 01 |
| 256 GRF 모드와 tile 크기 | 01, 04 |
| 2D block IO read/transform/transpose/prefetch | 02 |
| software prefetch (V_PREFETCH), split barrier | 02 |
| K d-major scalar gather → token-major 재배치 | 02 |
| softmax (exp2, alpha rescale, -inf/NaN 가드), causal 키 루프 상한 | 03 |
| int8↔f16 `0x6480 ^ byte` 트릭, u4 nibble, scale/zp 분리 | 03 |
| register spill, TPM, ISA 분석, ocloc splice | 04 |
| A/B 증명 레벨, 음성 대조군, layer-0 oracle | 05 |
| unexplained performance gap, 실제 native와 VTune/GTPin/ISA 교차 분석, 비교 arm 오류 검사 | 06 |
| 단계별 격차 폐쇄 사다리(24x → micro 3% 이내), 짧은 길이/작은 shape/head-count 잔여 결함 | 06 |
| cliloader/VTune/GTPin/ocloc 명령과 수집 모드, standalone 하네스, phase cycle counter, 마이크로벤치 상한 | 07 |
| 인라인 vISA 문법, LSC/DPAS 메시지, verifier 제약, asm 정확도 검증 | 08 |
| 병목 결정 트리, 설계 원칙, fixed-policy 게이트, 새 커널 시작 체크리스트 | 09 |

## 읽기 전에 알아둘 점 (정직성 노트)

- VTune: B70 GPU Hotspots와 A770/DG2 GPU Hotspots/source-analysis를 사용한 세션 기록이 있다. counter 수집 성공, native PC 매핑, GTPin trace 유효성을 각각 확인해야 한다. 실측 예와 실패한 수집은 [04장 §4.1.1](04-spill-isa-profiling.md)과 [06장](06-performance-gap-investigation.md)에 기록했다.
- 메모리 노트의 일부 커밋 해시는 rebase 로 현재 브랜치에 없다. 05장의 매핑표를 먼저 확인할 것.
- 2026-09 리팩터링으로 제거된 env 토글(`SDPA_OCL_BLOCK_SKIP`, `_DKS_ACTIVE`, `_PA_CUR_*`, `_MAX_BARRIER_V_PREFETCH`)은 역사적 설명으로만 남아 있다.
- 일부 노트 간 모순은 각 챕터 끝의 "불확실" 항목에 남겼다 (예: k16pwg4 가 micro 를 이긴다는 수치는 tiling-constraints 노트와 충돌하여 미확정).
- DG2의 S7a PA PREFILL standalone kernel 비교는 측정됐다. 3% gate PASS는 제품 통합·end-to-end·전체 지원 shape의 증명이 아니다. 현재 경계는 [05장 §5.7](05-methodology-and-pitfalls.md)과 [06장](06-performance-gap-investigation.md)을 참조한다.
- 줄 번호는 2026-10-06 작업 트리 기준이며 곧 이동한다. 심볼 이름으로 재검색할 것.
- `VERIFICATION.md`는 06장 추가 전의 한 시점 문서·스킬 대조 기록이다. 이후 결과의 검증 보고서로 읽지 않는다.
