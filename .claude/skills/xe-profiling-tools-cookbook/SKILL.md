---
name: xe-profiling-tools-cookbook
description: Commands and interpretation limits for Intel Xe/Xe-HPG GPU performance tools in OpenVINO intel_gpu - cliloader (-d -dv), usm_device inputs, final-linked micro binary dump, VTune GPU Hotspots (overview/lsc-slm/source-analysis), GTPin bb-latency, IGC ShaderDump/ocloc/iga64, cycle-counter phase timers, microbench ceilings, standalone OpenCL harness. Use when you must actually collect or read profile data for a GPU kernel.
---

# Xe GPU 성능 분석 도구 쿡북

전체 설명과 수치: `src/plugins/intel_gpu/docs/ocl_perf_guide/07-profiling-tool-cookbook.md` (이 스킬은 요약). 큰 격차의 *조사 절차*는 `ocl-kernel-performance-investigation`, 증명/귀속 규율은 `ocl-kernel-ab-methodology`.

## 질문 → 도구

| 질문 | 도구 | 말해 주지 않는 것 |
|---|---|---|
| 어떤 커널이 어떤 geometry로 돌았나 / device time | `cliloader -d -dv`, `OV_VERBOSE=4` census | 원인 |
| spill/scratch | cliloader `SPILL=`/`TPM=`, ocloc `.zeinfo` | 병목 여부 |
| 실행 유닛·XMX 활용 | VTune GPU Hotspots `characterization-mode=overview` | 소스 행 |
| 메모리/SLM 대역폭 | VTune `lsc-slm` (`-allow-multiple-runs`) | 개별 load latency |
| BB/메모리 latency, 명령 수 | VTune `source-analysis` bb-latency/mem-latency, `instruction-count` + GTPin | 계측 스케줄에서의 제품 latency |
| 컴파일러가 낸 ISA | IGC ShaderDump, ocloc, iga64 | 실행 빈도·latency 은닉 |
| phase별 비율 | `__builtin_IB_read_cycle_counter` (DG2 1 tick ≈ 1.08 ns) | 계측 안 한 커널의 시간 |
| 하드웨어 상한 | 마이크로벤치 (dpas/ld/slm/alu) | 전체 커널이 상한에 막혔는지 |

## 핵심 명령

```bash
CL=/home/shingyuk/work/opencl-intercept-layer/install/bin/cliloader   # 항상 -d -dv
TEST_USE_SDPA_OCL_HPG=1 PA_PERF_DEVICE_MEM=1 PA_PERF_ITERS=10 $CL -d -dv $BIN --device_suffix=1 --gtest_filter='perf_paged_attention_prefill/*'
# 소스/바이너리 덤프 (절대경로 + 끝 슬래시)
OV_GPU_DUMP_SOURCES_PATH=/abs/dir/ OV_GPU_MAX_KERNELS_PER_BATCH=1 ...
# VTune (BDF로 GPU 고정, 결과 디렉터리 부모 미리 mkdir)
source /opt/intel/oneapi/setvars.sh; mkdir -p /tmp/vt
vtune -collect gpu-hotspots -knob gpu-profiling-mode=characterization -knob characterization-mode=overview -knob target-gpu=0:3:0.0 -allow-multiple-runs -result-dir /tmp/vt/NAME -- <cmd>
vtune -report hotspots -r /tmp/vt/NAME -group-by computing-task -format csv
# GTPin source-analysis는 -allow_sregs 필수, 호출 수 최소화
AMPLXE_MORE_GTPIN_OPTIONS='-allow_sregs 1' vtune -collect gpu-hotspots -knob gpu-profiling-mode=source-analysis -knob source-analysis=bb-latency -knob computing-tasks-of-interest='kernel*#2#1#3' ...
# IGC dump / 오프라인 ISA (ocloc은 cwd에 *.bin/*.spv를 쓴다 → 임시 디렉터리에서)
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=/abs/dir <run>
```

## 반드시 지킬 것

1. **입력은 `usm_device`**. 단위 테스트 엔진은 dGPU에서 기본 `usm_host`(PCIe) → 151.8 vs 110 ms. `PA_PERF_DEVICE_MEM=1`.
2. **sdpa_micro 기준은 final-linked PREFILL native binary**. wrapper 재컴파일/초기 clBuildProgram binary는 nGEN GEMM 누락으로 FAIL. 같은 hash 파일에 GENERATE도 있으니 **ELF 안의 커널 이름**(`sdpa_micro__prefill_`)을 확인. head/heads가 바뀌면 micro geometry도 바뀌므로 새 shape는 제품을 다시 dump.
3. 2-GPU: OpenVINO `--device_suffix`와 VTune `target-gpu` BDF를 **각각** 확인. A770 = OpenCL p0/d0 = `GPU.1` = BDF `0:3:0.0`.
4. VTune 해석: occupancy는 속도 판정값이 아니다(같은 occupancy에서 24배 차이, 반대로 5%대 occupancy에서도 동률). "XMX active"는 latency도 센다(실제 처리량 ~32%인데 60%로 보임). 모든 비율은 시간 비율이지 wall time 아님. stall PC가 0개면 소스 귀속 금지.
5. GTPin: `-allow_sregs 1` 없으면 exit 0인데 "kernels not found"로 데이터 없음 → 수집 후 event/PC 존재 확인. sdpa_micro native는 ELF `st_size`가 fused `.text`보다 작아 assertion 실패 → 프로파일링용 *복사본*만 수정, 제품 native는 그대로. BB 합만 신뢰(BB 안 분배는 모델).
6. cycle-counter 계측은 ISA를 바꾼다(+1~8%). 가설 생성용이며 acceptance timing 금지.
7. **GPU는 한 번에 하나만**. 벤치/덤프/VTune 겹침은 해당 측정 무효.
8. 같은 round의 raw/ref paired ratio의 median으로 판정. queued32/gap0 **와** batch1/wait_each/gap200 둘 다. arm median과 paired를 둘 다 보고. 반올림으로 임계값 통과 금지.
9. 출력 위생: VTune 진행 로그 `tr '\r' '\n' | grep -v "Executing actions"`, IGC `.asm`의 `//.declare` 헤더, `strings libigc.so`는 출력 금지. Bash `cd` 금지(절대경로). `setupvars.sh`는 `"$@"`를 지우므로 source 전에 인자 저장.
10. `isa.sh`류 스크립트는 cwd 변경/정리를 하므로 읽고 실행. 실행 중인 스크립트는 편집 금지.

## 독립 하네스 (제품 빌드 없이)

`test/sdpa_ocl_xe_hpg/s7a/perf/opt/` (untracked): `sdpa_bench`(arm `이름=src:file.cl[:옵션]` / `이름=bin:file:micro`, `--verify --rows N`/`--all-rows`, `--shape seq,heads,kvheads,head`), `assemble.py`(제품 `.cl`+JIT 헤더 → 실행 커널), `mkvar.py`/`tile.py`/`sweep.py`/`reparam.py`, `vt*.sh`, `isa.sh`/`isasum.py`, `mb/`. 함정: 변형 소스의 `#define`이 하네스 `-D`를 덮음(`#ifndef` 가드), `--rows`는 복원 추출 샘플링(전수는 `--all-rows`), 틀린 커널도 빠르게 벤치마크됨(verify 줄을 본다), micro.bin은 shape 전용.

## A770(DG2) 상한 (MEASURED)

dpas.8x8 f16: 스레드당 1 in-flight(~30 ns), EU당 ≥4 스레드에서 포화(128 GRF 104 TFLOPS, 256 GRF 116). dpasw = dpas 속도. L1 hit block read 64 B 2.84 msg/clk, 256 B 0.93 msg/clk, L1 latency ~137 ns, L3 ~4.1 TB/s, SLM 12.4 TB/s, GDDR 2–3 TB/s. 256 GRF = 4 threads/EU, 128 GRF = 8. SLM 47.7 KB/WG에서 occupancy 절반.
