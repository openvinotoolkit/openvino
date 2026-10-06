---
name: xe-spill-isa-profiling
description: Playbook for diagnosing register spill, scratch/TPM, 256-GRF and occupancy problems in Intel Xe2/Xe-HPG OpenCL (DPAS) kernels by dumping and reading GEN ISA (IGC ShaderDump, ocloc splice, iga64) and cliloader SPILL=/TPM= lines. Use for "spill", "TPM", "scratch", "256 GRF", "ISA dump", "VTune", "ocloc A/B", "always_inline".
---

# Xe spill / ISA / profiling playbook

깊이 있는 근거와 수치: `src/plugins/intel_gpu/docs/ocl_perf_guide/04-spill-isa-profiling.md` (이하 "ch4"). 타일/DPAS는 01, 메모리 I/O는 02, 수치는 03, 방법론은 05 장.

## 불변 규칙 (repo AGENTS.md + 사용자 메모리)
- 빌드, gtest, cliloader 실행, ab 실행은 **사용자만** 실행한다. 정확한 명령 블록을 제시하고 결과를 받는다. 읽기 전용 분석(grep, 소스/덤프 읽기)만 직접 한다.
- 성능 개선을 측정 없이 주장하지 않는다. 숫자마다 하드웨어(B580/B70/DG2)와 출처를 적고 MEASURED vs ASSUMED를 구분한다.
- NaN/Inf, accumulator(f32), precision 변환 순서, dynamic shape(shape_info)를 항상 점검한다. 타일 override는 크래시/spill 없이 *조용히 틀린 출력*을 낼 수 있다.
- 신규 `.cl`은 cmake 재실행 필요 (GLOB은 configure-time). 커밋 메시지는 한 줄.

## 1. 관측 먼저 (런타임 ground truth)
```bash
# setupvars.sh가 "$@"를 지우므로 인자는 source 전에 저장
cliloader -d -dv python <llm_bench>/benchmark.py -d GPU.1 -m <model> -n 1 -ic 4 -pf <jsonl> 2>&1 | grep <kernel>
```
- 커널 이름줄: `SIMD16 REG128|256 [SPILL=N] [TPM=N] SLM=N GWS[..] LWS[..]`. spill 0이면 `SPILL=` 없음.
- **spill/TPM 판단은 런타임 줄로만.** ocloc 오프라인 spill은 head-72에선 정확(2688 B)했으나 tiling sweep 4건과 kc/vc B/C config에서 0으로 틀렸다.
- 2-GPU 박스: `ov_gpu_func_tests --device_suffix=1`(B70) 없으면 iGPU에서 sdpa_opt가 돌아 비교 무효.
- config가 달라야 하는데 SLM/GWS/LWS/시간이 동일하면 토글이 적용 안 된 것(setupvars `set --`, PA 커널이 안 뜨는 stateful 경로).
- occupancy%는 속도 지표 아님(더 높은데 더 느린 사례). cliloader `-d`는 host-bound라 작은 커널 이득이 e2e에서 안 보임.
- VTune은 실제 B70 GPU Hotspots 분석에 사용했다 (상세와 제한: `04-spill-isa-profiling.md` §4.1.1). 확인된 사례: occupancy 99.0%, XVE active 47.8%/stalled 52.2%, 5.581 ms/2 instances; 별도 OCL-vs-micro 비교에서 SBID 59.1% vs 34.2%, barrier stall 10.6% vs 2.5%. 한 수집은 source/DWARF mapping은 정상인데 stall PC가 0개여서 소스 행 귀속은 불가했다.
- 수집 전 VTune, driver, Metrics Discovery 버전과 GPU BDF를 기록한다. B70에서 Metrics Discovery 1.13.545 초기화가 실패했고 1.16.190으로 해결한 이력이 있다. OpenVINO workload와 VTune target GPU를 같은 BDF에 고정한다. JIT source 보존 시 `OV_GPU_DUMP_SOURCES_PATH`와 `OV_GPU_MAX_KERNELS_PER_BATCH=1`을 함께 설정한다.
- GPU active/stalled, SBID, barrier stall은 병목 가설을 좁히는 지표이지 대역폭이나 특정 소스 행의 직접 증거가 아니다. VTune result의 정확한 zebin archive로 주소를 해석한다. stall PC가 없거나 주소 매핑이 불확실하면 ISA/source-level 귀속을 주장하지 않는다. 판정은 동일 조건의 before/after VTune + cliloader device time + correctness A/B로 한다.
- XE/i915 counter 및 source attach 권한은 system sysctl에 좌우된다. 관리자 설정이 필요하면 보안 정책에 따라 최소 기간만 허용하고 원복한다. `GTPin` 경고와 수집 실패를 동일시하지 말고 result에 GPU metric 데이터가 실제 기록됐는지 확인한다.

## 2. spill 종류 구분

원인 불명의 성능 격차 전체를 조사할 때는 [ocl-kernel-performance-investigation](../ocl-kernel-performance-investigation/SKILL.md)와 [06장](../../../src/plugins/intel_gpu/docs/ocl_perf_guide/06-performance-gap-investigation.md)을 먼저 참고한다.

추가 A770/DG2 사례에서는 OCL 원본 XVE active/stalled가 15.6/81.8%, occupancy 47.9%, XMX active 2.0%, runtime spill 7,872 B였다. micro occupancy는 49.7–49.8%, XMX active 24.4–24.5%였다. K는 이미 dword block read였고 V는 gather/packing이었으므로 실행 ISA를 확인해 reader별 가설을 고쳤다. VTune/GTPin 원인 분석과 한계는 06장 §6.3에 있다. 저spill kernel은 AMPLXE_MORE_GTPIN_OPTIONS=-allow_sregs 1 없이는 exit 0이어도 event/kernel 데이터가 없을 수 있으며, fused micro final binary의 ELF symbol size가 native text보다 작아 source-analysis가 실패한 사례도 있었다. GPU event와 exact zebin/PC mapping을 확인한다.

| 관측 | 의미 | 우선 대응 |
|---|---|---|
| `TPM>0`, spill 0, hot loop 안 scratch store/load | **런타임 인덱스 private 배열** | 인덱스를 unroll 상수로, 또는 select chain + `sub_group_broadcast` (alpha[] 사례 0.8%) |
| `SPILL>0` | register pressure (독립 gather N개 동시 live 등) | 로드 geometry 개선 먼저, 그 다음 타일/GRF |
| `.asm`에서 `grep spill` 매치 | **무의미**: `//.full_options`의 `-abortOnSpill` | `//.spill size`, `//.private memory size`, `.zeinfo` 를 읽는다 |

## 3. ISA 확보 (리빌드 없이)
1. 런타임 덤프: `OV_GPU_DUMP_SOURCES_PATH=./ OV_GPU_MAX_KERNELS_PER_BATCH=1` (jit prelude + standalone .cl) 와 `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir>` (.asm/.ll/.zeinfo). `MAX_KERNELS_PER_BATCH=1`이면 덤프당 entry 1개. 파일명은 hash라서 `grep -l 'kernel sdpa_ocl__prefill'`로 식별. 스크립트: `test/dump_isa*.sh` (`dump_isa.sh`는 옛 ROOT 하드코딩).
2. ocloc: `ocloc compile -file X.cl -device bmg -options '-cl-mad-enable -cl-std=CL3.0 -D...4개' -internal_options '...'` (`-device bmg`; `xe2`는 iGPU로 해석. `-cl-mad-enable`은 `-options` 안). 옵션 원본은 덤프의 `*_options.txt`/`*_internal_options.txt`.
3. splice A/B: 덤프 prelude + 작업트리 `.cl`을 `#include`. 스윕할 `#define`은 prelude에서 `sed`로 제거 후 재공급. 예: `test/splice_head72.sh`, `isa_ab_mixed.sh`, `kvup_splice.sh`, `isa_ab_kvup.sh`, `sdpa_ocl_ab.py` (L0/L1/L1'/L2/pset/built/hpg).
4. **먼저 재현 증명**: HEAD 소스 splice가 런타임 값(instCount/spill/dpas/hash)과 일치하는가 (`kvup_splice.sh check`, head-72 spill 2688 일치). 절대 instCount/spill은 같은 표의 행끼리만 비교.
5. 디스어셈블: zebin이면 `readelf -S -W`로 `.text.<kernel>` 오프셋 -> `dd` -> `iga64 -p=xe2 -d`. `/usr/bin/iga64` 있음. micro의 GEMM은 IGC 덤프에 안 보이며(더미 `dpas.8x1 null`, `0xFADECAFE`), ocloc 컴파일하면 가짜 ~709 instr -> micro 숫자 인용 금지.
6. 빌드 후: `python3 test/sdpa_ocl_ab.py built --base <rev>`로 .inc(chunk 연결)/.so/`ov_gpu_unit_tests` 안에 의도한 소스가 있는지 확인 (unit test는 정적 링크라 stale 가능).

## 4. ISA 판독 치트시트
| 시그니처 | 의미 / 대응 |
|---|---|
| `load.ugm.d8u32/d16u32 (1\|M0)` 다수 | SIMD-1 scalar gather (15/16 lane 낭비). per-key scale/zp면 lane=key wide load + broadcast (128 -> 0, B580) |
| `load.ugm.d8u32 (16\|M0)` 수십 개 | 중간 메시지 과다. `d32xKt` wide transposed / `load_block2d` 로 |
| `load_block2d.ugm.*` | 2D block read (목표 형태; 규칙은 02장) |
| `dpas.8x8` 개수 | steady-state 개수. IGC가 첫 iteration을 peel (x2). 0이면 DPAS가 버려진 것(DG2 SG16 short8) |
| `dpas.8x1 null` + `0xFADECAFE` | micro blob 더미. 무시 |
| 대량 strided `mov` (`<2>`, `<4;1,0>`) | int8 -> f16 widen/VNNI repack. IGC floor. dequant 산술은 **float**, 끝에 `(half)` 1회 |
| `r[a0]` 급증 | lane-varying 루프 변수/런타임 인덱스 배열 (2 -> 274, 1.8x 느림) |
| backward branch (target label이 더 작은 줄) | 실제 rolled loop. forward goto는 predicated skip |
| `goto/join` 급감, `d32x8t` 감소 | unroll이 풀림 (guard 모양 문제) |
| `sync.*` +-1..5 | SWSB jitter. send/dpas/math/load/store 개수 불변이면 무시 |
| zeinfo `disable_mid_thread_preemption` true -> None | IGC ~600 instr 임계 (<=598 true, >=601 없음). 기능 변화 아님 |
| `numGRF=256` / `-TotalGRFNum 256` | 256 GRF 모드 |

## 5. 256 GRF 판단 (모두 MEASURED, ch4 §4.5)
- 기본 타일에 256만: **손해** (llama-3.1-8b 2.18M -> 2.73M ns, B70).
- 큰 타일 + 256 한 쌍(tq32/pwk4/pwq2): micro보다 7% 빠름 (1.796M vs 1.930M ns, B70).
- gather-bound 커널(head-72 scalar, u4 head-64): 256이 spill을 0으로 만들어도 +47%/+70% 악화. spill 바이트를 목적함수로 쓰지 말고 *왜* live가 큰지를 본다.
- xe_hpg(DG2)는 256 강제 (128에선 모든 측정 타일 spill). `SDPA_OCL_256GRF=0/1`만으로는 HPG 128/256 실험이 안 된다.
- occupancy는 2축 (WG 수 vs threads/WG). `sg_per_wg`와 `kq_wg_tile_keys`가 클수록 느린 사례 (head-128 prefill, 3.53M -> 6.86M ns, B70). SLM 점유 가설은 device 측정으로 반증됨.

## 6. IGC 파이프라인 함정
- helper는 **`SDPA_OCL_INLINE` = `__attribute__((always_inline)) inline`**만. plain `inline`은 인라인되어도 *모듈 전체* 파이프라인이 바뀌어 무관한 config의 ISA까지 움직인다 (122개 중 121개 C등급).
- always_inline이어도: 확장 파라미터 캐스트(`size_t lane` + `int lane_i` 분리), 상수가 되는 파라미터로 분기, 합쳐진 좌표 인자, **helper 내부 배열**(caller 선언 + `__private` 포인터), loop-carried 출력(out-param) 에서 ISA가 움직인다.
- unroll: trip count는 매크로, guard는 **호출 지점**에서 (helper 맨 위 `if(skip) return;`은 unroll을 잃어 2.3x 느림, 호출 지점은 5.6 us 유지). lane-varying 루프 변수 금지. 루프 bound를 0으로 만들어 ablation (`if(0)` 금지).
- 소스의 분기 의도 != IGC 코드젠 (causal block skip은 flatten되어 +16 instr).

## 7. 제거 레버 요약 (증거 있는 것만)
select chain(TPM 제거) / block read로 gather 대체 (head-72 9.28x, spill 2688 -> 0, B70) / scale-zp hoist / M 또는 tile 축소 (`live_grf_estimate`, budget 112, gemma-4 head-512 M=8 SPILL=34432, 3.14x 느림) / 256GRF+타일 쌍 / tile 줄이기(config A) vs thread 줄이기(config C, 372,628 ns). **증거 없음**: `#pragma unroll N` 제한, 커널 분할로 spill 감소.

## 8. 판정 절차
1. L2 ISA A/B로 "의도하지 않은 변화 없음"(instCount + opcode 히스토그램 동일)을 증명 (`isa_ab_*.sh`, `sdpa_ocl_ab.py l2`). C등급이면 `compile_one(..., keep_dir=)`로 `beforeUnification -> afterUnification -> optimized -> ISA`를 diff해 처음 갈라지는 단계를 찾는다.
2. 성능은 **device 측정**으로만 판정: env toggle로 한 binary 안 A/B, min/median, 같은 shape의 production baseline 절대값 대비. 정적 instr가 같거나 좋은데 시간이 다르면 ablation (cache sector, thread당 dirty line: store 4.8x 사례).
3. **한쪽 control만으로 귀인 금지**: A/B는 정확히 한 가지만 달라야 한다 (head 486 사례).
4. 작은 비율(<~5%)은 microprobe로 예측되지 않는다 (probe -4.4% 예측 vs 실모델 parity). 큰 비율(12%)은 1 pp 내 일치.
5. 실패한 실험도 기록: int8 K 2D block 30.8 us > scalar 26.1 us, K_DIAG=2 observer effect(더 느림), half dequant 더 느림, V cp-pair reuse 효과 없음.

## 9. GPU hang / DG2
`CL_OUT_OF_RESOURCES`/`clFinish` 예외는 hang-reset 증상일 수 있다: `dmesg | grep -i 'GPU HANG'`와 PID 대조. DG2 ocloc spill(h64 7,968 B 등; 출처 `s6a/pass_real/results.tsv`, 05장 §5.7.2)은 컴파일러 보고값이지 DG2 런타임 값이 아니다. `ab.py hpg`의 `dpas` 컬럼이 0이면 DPAS가 버려진 것.

## 10. 보고 형식
변경/가설/측정을 표로: 하드웨어, 명령, 지표(cliloader SPILL/TPM/time 또는 ocloc instCount), MEASURED/ASSUMED, 회귀 위험, 검증한 correctness 항목, 열린 질문. 빌드/실행이 필요하면 사용자에게 정확한 명령 블록을 요청한다.
