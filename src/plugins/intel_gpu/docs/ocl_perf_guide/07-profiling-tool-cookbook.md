# 07. 성능 분석 도구 쿡북 (cliloader / VTune / GTPin / IGC / ocloc / 하네스 / 마이크로벤치)

06장의 DG2 사례(24배 격차 → micro 대비 3% 이내)에서 실제로 쓴 도구와 명령, 해석 한계, 빠졌던 함정을 한곳에 모았다. 새 커널의 성능 문제를 조사할 때 "어떤 질문에 어떤 도구를 쓰는가"와 "그 도구가 말해 주지 않는 것"을 먼저 보고, 명령은 복사해서 쓴다.

모든 수치는 Arc A770 (DG2, xe_hpg, OpenVINO `GPU.1`, driver 26.27.39122.14), VTune 2026.4 기준의 MEASURED다. 다른 GPU(Xe2/B70 등)에서는 같은 도구를 쓰되 수치와 BDF를 다시 확인한다. 원자료·스크립트는 `test/sdpa_ocl_xe_hpg/s7a/perf/opt/`(untracked)에 있다. 이 장에서 경로를 `$O`로 줄여 쓴다.

```bash
repo=/home/shingyuk/work/openvino-eddy
O=$repo/test/sdpa_ocl_xe_hpg/s7a/perf/opt
```

## 7.0 질문 → 도구 한눈에 보기

| 질문 | 1차 도구 | 이 도구가 말해 주지 않는 것 |
|---|---|---|
| 어떤 커널이 실제로 몇 번, 어떤 geometry로 실행됐나? | cliloader `-d -dv`, `OV_VERBOSE=4` dispatch census | 왜 느린지 |
| 한 호출의 순수 GPU 시간은? | cliloader device time, 하네스의 OpenCL event | e2e latency, host 오버헤드 |
| 레지스터 spill/scratch가 있나? | cliloader `SPILL=`/`TPM=`, ocloc `.zeinfo` | spill이 *병목인지* |
| 실행 유닛이 놀고 있나, DPAS를 쓰나? | VTune GPU Hotspots `overview` (XVE active/stalled, XMX active, occupancy) | stall의 소스 행/명령 |
| 메모리 계층의 어디가 막히나? | VTune `lsc-slm` / L3·SLM bandwidth, 마이크로벤치 상한 | 개별 load의 latency |
| 어떤 basic block/명령에 시간이 몰리나? | VTune `source-analysis` + GTPin (`bb-latency`, `mem-latency`, `instruction-count`) | 계측으로 바뀐 스케줄에서의 제품 latency |
| 컴파일러가 실제로 무슨 명령을 냈나? | IGC ShaderDump, ocloc, iga64 | 실행 빈도, latency 은닉 |
| 한 커널 안의 phase(KQ/softmax/SV)별 비율은? | `__builtin_IB_read_cycle_counter` 타이머 | 계측하지 않은 커널의 시간 (계측이 ISA를 바꿈) |
| 이 하드웨어의 DPAS/load/SLM 상한은? | `mb/` 마이크로벤치 | 우리 커널이 그 상한에 막혔는지 |
| 한 가지 변경이 원인인가? | 하네스의 paired A/B + 양방향 control | — (도구가 아니라 규율, 05장) |

## 7.1 cliloader (CLIntercept): 실제로 무엇이 돌았나

```bash
CL=/home/shingyuk/work/opencl-intercept-layer/install/bin/cliloader
source build/ov_install/setupvars.sh        # 주의: setupvars가 "$@"를 지우므로 인자는 source 전에 저장
TEST_USE_SDPA_OCL_HPG=1 PA_PERF_DEVICE_MEM=1 PA_PERF_ITERS=10 \
  $CL -d -dv bin/intel64/Debug/ov_gpu_unit_tests --device_suffix=1 \
  --gtest_filter='perf_paged_attention_prefill/*'
```

- `-d`는 device time, `-dv`는 GWS/LWS(dispatch geometry)까지 기록한다. 초기에는 `-d -ko`를 썼고 이후 `-d -dv`로 통일했다. geometry를 기록하면 "의도한 tile/subgroup 설정으로 실제 launch됐는지"를 바로 확인할 수 있다.
- 출력의 min/avg/max는 한 프로세스 안의 호출들이다. 서로 다른 arm은 **별도 프로세스, ABBA 순서**로 돌린다(환경변수가 프로세스 단위로 캐시됨).
- **dGPU에서 입력 메모리 위치가 결과를 바꾼다.** unit-test 엔진은 입력을 `allocate_memory(layout)`로 잡아 dGPU에서 기본 `usm_host`가 되고, 커널이 PCIe로 읽는다. A770 PA prefill에서 usm_host는 151.8 ms로 나왔고 usm_device는 110 ms 수준이었다. `PA_PERF_DEVICE_MEM=1`(테스트 헬퍼에서 Q/K/V/cache를 usm_device로 복사)을 쓰고 usm_host 결과는 폐기한다. 이 모드에서는 reference가 입력을 lock할 수 없으므로 perf 테스트 전용이다.
- 어떤 커널이 선택됐는지는 이름으로 확인한다: `sdpa_ocl_prefill_*`, `sdpa_micro__prefill_*`, `sdpa_opt_*`. tier gate가 거부하면 조용히 opt로 떨어져 "OCL 측정"이 아닌 측정이 된다. HPG 실험에서 `TEST_USE_SDPA_OCL_HPG=1`, 대조군은 `TEST_USE_SDPA_OCL=0`.
- `-d -dv` 출력은 Debug 호스트 라이브러리에서 얻은 값이다. device kernel time은 JIT된 OpenCL이라 호스트 빌드 타입에 거의 영향이 없지만, Release 재확인은 별도 점검 항목으로 남긴다.

### 커널 소스/바이너리 덤프 (참조 커널을 정확히 얻기)

```bash
OV_GPU_DUMP_SOURCES_PATH=/abs/dir/ OV_GPU_MAX_KERNELS_PER_BATCH=1 \
  TEST_USE_SDPA_OCL_HPG=1 PA_PERF_DEVICE_MEM=1 $BIN --device_suffix=1 --gtest_filter=...
```

- 경로는 **절대 경로 + 끝 슬래시**. `MAX_KERNELS_PER_BATCH=1`이면 커널마다 개별 `.cl`/binary가 나와 대조가 쉽다.
- **sdpa_micro는 OpenCL 소스만으로 재현할 수 없다.** 소스 wrapper를 다시 컴파일하면 nGEN이 삽입하는 fused GEMM이 빠져 정확도가 FAIL한다. CLIntercept가 `clBuildProgram` 직후 저장한 첫 binary도 삽입 전이라 FAIL이다. 비교 기준은 **최종 link된 PREFILL native binary**(`*_<hash>_GPU.bin`, build options 빈 값)다.
- 같은 hash 파일에 PREFILL과 GENERATE가 같이 있을 수 있다(`B5C0FBCF`). hash만 믿지 말고 **ELF 안의 커널 이름**(`sdpa_micro__prefill_`)을 확인한다. GENERATE를 PREFILL로 잘못 쓰면 `argument-52` 오류가 나거나 timing이 무효가 된다. 하네스(`run_extended.py`, `run_checked.py`)는 PREFILL 바이트가 없으면 GPU launch 전에 거부하도록 고쳤다.
- head/heads/kvheads가 바뀌면 micro의 geometry(Q tile, SG 수, SLM, spill)도 바뀐다. 예: h16/32 Q128/SG16, h48/64 Q128/SG32(h64 spill 1,440 B, SLM 59,904), h80/96/128 Q32/SG16, >128 대부분 Q32/SG32. 새 shape의 기준 binary는 항상 제품을 다시 dump해서 얻는다.

## 7.2 독립 하네스: 제품 빌드 없이 커널을 반복 실험하기

OpenVINO를 다시 빌드하면 한 번에 몇 분이 걸리고 호스트 코드가 섞인다. 커널 설계 탐색은 **standalone OpenCL 하네스**(`$O/sdpa_bench`, `sdpa_bench.cpp`)에서 했다.

```bash
g++ -O2 -fopenmp -std=c++17 $O/sdpa_bench.cpp -o $O/sdpa_bench -lOpenCL
$O/sdpa_bench --verify --rows 96 --iters 8 \
  --arm c40=src:$O/v/c40_base.cl:TILED \
  --arm micro=bin:$O/micro.bin:micro
```

- arm 문법: `이름=src:파일.cl[:옵션...]`(런타임 컴파일) / `이름=bin:파일:micro`(미리 만든 native binary). `--shape seq,heads,kvheads,head`, `--warm`, `--iters`.
- 제품 커널을 빌드 없이 시험하려면 `assemble.py`가 JIT 헤더 덤프와 현재 제품 `.cl`을 합쳐 실행 가능한 커널을 만든다. 제품 코드가 바뀌면 다시 assemble한다.
- 변형 생성 도구: `mkvar.py`(이름 붙은 소스 편집 — phase timer 삽입 포함), `tile.py`(8개 tiling define 재작성), `sweep.py`(유효 tiling 스윕/랭킹), `reparam.py`(shape 재작성).

하네스가 만든 함정 (각각 실제로 발생했다):

| 함정 | 증상 | 대책 |
|---|---|---|
| 변형 소스의 `#define TILED_NKT 256`이 하네스의 `-DTILED_NKT=<NT>`를 덮어씀 | seq가 맞지 않으면 K'/V' 주소가 틀려 `verify FAIL`(maxabs ≈ 4.9)인데 **시간은 정상처럼 보임** | `#ifndef` 가드. 시간만 보지 말고 verify 줄을 본다 |
| `--rows N`이 reference 행을 **복원 추출로 샘플링** | "fullrow PASS" 표기가 실제로는 샘플이었음 | `--all-rows` 모드(`sdpa_bench_checked`)와 정확도 FAIL 시 rc=2 |
| 하네스가 timing 밖에서 쓰지 않는 K'/V' 버퍼를 할당·업로드 | raw-only 실험에서 pre-pass 흔적이 남음 | raw 전용 하네스(`sdpa_bench_extended`)에서 할당과 pre-pass 실행을 거부 |
| `--prepass-src`는 pre-pass를 한 번만 실행하고 커널만 AB 측정 | `prepass avg 0.000`을 "포함 시간"으로 오독 | pre-pass 포함 파이프라인은 별도로 잰다 |
| micro.bin이 rung-1(h128, 32/8) 전용 | 다른 shape에 쓰면 오답/무효 | shape별로 제품에서 final-linked binary를 새로 받는다 |
| 실패한 변형의 시간 | 틀린 커널이 빠르게 "벤치마크"됨 | 정확도 FAIL 변형의 시간은 폐기하고 로그만 남긴다 |

**정확도 검증은 timing보다 먼저 한다.** 출력을 NaN으로 poison하고, CPU double reference를 모든 row/head/channel에 대해 비교(`maxabs < 1e-2`, tolerance 완화 금지)하며, 이전 OCL 변형과 **bit-identical인지**도 따로 본다(수학이 같은 변경이면 bit-identical이어야 한다). 입력 분포는 두 종류를 쓴다: 일반(Q/K/V std 1)과 sharp(Q/K std √1.28, V std 0.1, scale 1/√head — 제품 gtest의 N(0,0.1)×gain128과 logit 분포를 맞춤). 부드러운 데이터는 softmax·mask 오류를 숨긴다(03장, 05장).

## 7.3 VTune GPU Hotspots

### 환경과 기본 명령

```bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
mkdir -p /tmp/vt                                  # 부모가 없으면 리디렉션이 GPU 시작 전에 실패한다
vtune -collect gpu-hotspots \
  -knob gpu-profiling-mode=characterization -knob characterization-mode=overview \
  -knob target-gpu=0:3:0.0 -allow-multiple-runs \
  -result-dir /tmp/vt/NAME -- <command...>
vtune -report hotspots -r /tmp/vt/NAME -group-by computing-task -format csv
```

- `target-gpu`는 **PCI BDF**다. 2-GPU 시스템에서 OpenVINO `GPU.1`(`--device_suffix=1`)과 VTune의 BDF를 각각 확인한다(A770: OpenCL platform 0 / device 0 / `GPU.1` / BDF `0:3:0.0`). 잘못된 GPU를 프로파일링해도 오류 없이 수집된다.
- `$O/vt.sh <name> <mode> <command...>`가 위를 묶고 `vtrow.py`로 CSV 한 줄 요약을 낸다. `mode`는 `overview`, `lsc-slm`, `full-compute`, `instruction-count` 등. `lsc-slm`은 `-allow-multiple-runs`가 필요하다.
- 진행 로그는 `\r`로 한 줄을 계속 덮어써서 출력이 매우 커진다. `tr '\r' '\n' | grep -v "Executing actions"`로 거른다.

### 읽는 법 (06장 DG2 사례의 실제 값)

| 지표 | OCL 원본 | micro | 해석 |
|---|---:|---:|---|
| XVE active / stalled | 15.6% / 81.8% | 44.2% / 55.6% | 실행 유닛 활용 부족 |
| XVE occupancy | 47.9% | 49.7% | 같은 수준 → occupancy로는 24배를 설명 못 함 |
| XMX active | 2.0% | 24.4% | DPAS를 거의 못 씀 |
| L3 read bandwidth | 247 GB/s | 784–858 GB/s | 단서일 뿐 DRAM 포화 증거가 아님 |

- **occupancy는 속도 판정값이 아니다.** 같은 occupancy에서 24배 차이가 났고, 반대로 occupancy가 낮아도(h128 seq257 MHA 5.6% vs 5.5%) 두 커널이 같은 시간에 끝났다.
- **"XMX active"는 latency도 센다.** 포화 DPAS 마이크로벤치가 100%를 찍지만 우리 최선 커널은 XMX active 60%에서 실제 처리량은 약 32%였다(phase 타이머/명령 수로 교차 확인).
- 이 지표들은 *시간 비율*이지 wall time이 아니다. 프로파일링 중 시간은 clean 측정보다 길다(seq512, OCL/micro: 프로파일 중 178/140 µs vs clean 137/123 µs).
- Xe2/B70의 GPU Hotspots 사례에서는 stall PC가 0개여서 소스 행/ISA 명령에 귀속할 수 없었다. 이 경우 지표는 *가설 생성*에만 쓰고 채택은 A/B로 한다(04장 §4.1.1).

### source-analysis + GTPin (basic block / memory latency / instruction count)

```bash
AMPLXE_MORE_GTPIN_OPTIONS='-allow_sregs 1' vtune -collect gpu-hotspots \
  -knob gpu-profiling-mode=source-analysis -knob source-analysis=bb-latency \
  -knob computing-tasks-of-interest='sdpa_ocl_prefill*#2#1#3' -knob target-gpu=0:3:0.0 \
  -result-dir /tmp/vtsa/NAME -- $O/sdpa_bench --iters 2 --warm 2 --arm NAME=src:variant.cl
```

(`$O/vtbb.sh`, `vtic.sh`(instruction-count), `vtrep.sh`, `vtsa.py`(GTPin 이벤트를 native ISA PC에 join해 BB별 집계)).

- **`-allow_sregs 1`이 없으면** 저-spill 변형에서 exit 0인데 "kernels not found"로 데이터가 비어 있을 수 있다. 수집 뒤 반드시 실제 event와 PC가 생겼는지 확인한다.
- `computing-tasks-of-interest`는 `커널이름패턴#호출순서...` 형태다. 계측은 느리므로 `--iters 2 --warm 2`처럼 적게 돌린다.
- **sdpa_micro native는 GTPin source-analysis가 assertion(`it != _insIdByOrigOffset.end()`)으로 실패**했다. ELF function `st_size`가 실제 fused `.text` 길이보다 작아서였다. 프로파일링 전용 *복사본*의 symbol size만 고쳐 재현했고 제품 native/fuser는 건드리지 않았다. 같은 실패를 근거 없이 재시도하지 않는다. overview characterization은 micro에서도 정상이다.
- **GTPin의 명령별 cycle은 BB 안의 분배가 모델**이다. BB 합만 믿는다. 합성 operand로 ablation하면(특히 128 GRF) 추가 ALU/레지스터 때문에 왜곡되므로, 로드를 SLM 읽기로 치환하는 쪽이 깨끗했다.
- 계측 build는 스케줄이 달라진다. instrumented 결과로 acceptance timing을 대신하지 않는다.
- 얻은 대표 결론(A770, probe 커널): BB cycle은 V 로더 49%, K 로더 28%; 로드 ablation으로도 K 7 ms, V 6.8 ms. 이 두 independent 근거가 맞아서 "operand load가 시간의 ~70%"를 확정했다.

## 7.4 IGC ShaderDump / ocloc / iga64: 컴파일러가 실제로 낸 ISA

```bash
# 런타임 컴파일을 덤프
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=/abs/dir  <run>
# 오프라인 (GPU 불필요): 런타임과 같은 옵션으로
ocloc compile -file build.cl -device dg2 \
  -options "-cl-mad-enable -cl-std=CL3.0 -cl-intel-256-GRF-per-thread ..." \
  -internal_options "-cl-intel-greater-than-4GB-buffer-required -cl-intel-has-buffer-offset-arg -cl-store-cache-default=2 -cl-load-cache-default=4"
```

- `$O/isa.sh <variant.cl> [G128]` → `/tmp/isa/<name>/` (`.asm`, `.zeinfo`), `isasum.py`가 send/dpas/spill 개수를 요약한다. **`isa.sh`는 작업 디렉터리를 바꾸고 정리를 하므로** 실행 위치와 대상을 확인한다.
- spill/scratch: `.zeinfo`의 scratch(런타임 값은 cliloader `SPILL=`).
- `ocloc compile`은 **cwd에 `*.bin/*.spv`를 쓴다.** 저장소 루트에서 실행하면 stray 파일이 생긴다(`dpasw_peak_dg2.*`가 한 번 repo 루트에 남음). 임시 디렉터리에서 돌리거나 정리한다.
- `.asm`의 `//.declare` 헤더와 `strings libigc.so`는 거대하다. 출력을 grep으로 좁힌다.
- 오프라인 ISA는 런타임과 다를 수 있다. 런타임 options와 실제 loaded zebin으로 재현 여부를 확인한 뒤에 결론을 쓴다.
- 판독 시그니처(자세한 표는 04장 §4.4): `dpas` 개수와 사이의 명령, `load.ugm.d32x8t`(block transposed), `send`/`sync.allwr`(로드 완료 대기), `spill`/scratch fill, `goto/join`(dynamic branch).

### ISA에서 확인한 것의 예

| 관측 | 결론 |
|---|---|
| IGC는 로드를 **사용 직전으로 sink**한다. 128 GRF에서는 step마다 `loads → sync.allwr → dpas`, 256 GRF 완전 unroll에서도 약 20줄 앞까지만 hoist | latency가 매 step 노출됨. 소스에서 load를 앞으로 옮겨도 ISA가 안 바뀔 수 있으니 ISA로 확인 |
| 원본 OCL의 K는 이미 32 B dword block read였고 V만 lane gather+pack을 반복 | "K/V 둘 다 scalar gather"라는 가설 정정. 소스 경로가 아니라 **실행된 ISA**를 먼저 본다 |
| unroll 제한 후 scratch fill ~104배 감소, spill 7,872→704 B | 로드 개수는 그대로인데 spill reload가 시간의 큰 부분이었음 |
| full-tile guard로 dynamic goto/join 약 85.8% 감소 | 같은 send 수에서 control-flow 비용만으로 28→19 ms |
| `-cl-intel-no-prera-scheduling`은 받아들여지지만 ISA 동일 | 효과 없는 옵션을 "시도했다"로 남기지 말고 ISA diff로 확인 |
| `__asm__ volatile("":::"memory")`는 block read를 scalar 로드로 풀고 spill 유발 | 컴파일러 barrier는 공짜가 아니다 |
| 같은 의미의 소스 변형이 ±40% 변동, 일부 256-GRF fuse 구성은 오답 | 모든 변형에 정확도 검증. IGC miscompile 가능성을 열어둔다 |

## 7.5 커널 내부 cycle counter phase 타이머

`__builtin_IB_read_cycle_counter()`는 DG2에서 동작한다(1 tick ≈ 1.08 ns). `mkvar.py timing/timing2`가 커널 소스에 phase 타이머를 넣고 debug 버퍼(8번째 인자)에 WG별 tick을 기록한다.

c40 커널(h128, seq4096)의 phase (ticks/iteration): K 5,068 / max 1,784(barrier wait 607) / exp 2,533(split wait 139) / SV 5,116, 합 14.5K. K'/V'를 SLM 읽기로 바꾼 ablation에서 K −1,230, SV −1,460 ticks → global latency ≈ 19%, SLM operand latency ≈ 15%, DPAS 직렬 ≈ phase당 2.7K.

한계(반드시 지킨다):
- **계측은 ISA와 스케줄을 바꾼다.** h64 seq1033에서 계측 커널 158 µs vs 비계측 152 µs, MHA gap에서 +1~8%. phase 비율은 *가설 생성*에만 쓰고, acceptance timing이나 정확한 latency 귀속에 쓰지 않는다.
- phase에는 inactive SG의 wait나 barrier 동기화가 섞여 있다("max" phase가 53%로 보였지만 inactive SG 대기 포함).
- 이 값으로 고친 효과는 반드시 *계측 없는* 커널의 paired A/B로 확인한다.

## 7.6 마이크로벤치 (`$O/mb/`)

`mb_run`(러너), `dpas_peak*.cl`, `ld_bw*.cl`, `slm_bw.cl`, `alu_ipc.cl`, `reorder.cl`, `cyc_run`. 우리 커널이 하드웨어 상한 대비 어디쯤인지 가늠하는 용도다. A770 측정값:

| 항목 | 값 |
|---|---|
| dpas.8x8 f16 | 한 스레드가 **동시에 1개만 in-flight**(~30 ns/개, 독립 누산기 8개여도). EU당 ≥4 스레드에서 포화: 128 GRF 10.05 ns/dpas = 104 TFLOPS, 256 GRF 9.04 ns = 116 TFLOPS. operand 값은 속도에 무관 |
| dpasw | dpas와 같은 속도. src1 공유 순서도 시간에 무관 → opcode 자체를 성능 원인으로 단정하지 않는다 |
| L1 hit block read | 64 B 메시지 2.84 msg/clk/core, 256 B 0.93 msg/clk(~240 B/clk). L1 hit latency ≈ 137 ns |
| L3 stream | ≈ 4.1 TB/s (72 B/clk/core) |
| SLM block read | 최대 12.4 TB/s (215 B/clk/core) |
| GDDR | 2–3 TB/s |
| 점유 | 256 GRF 4 threads/EU(occupancy ~50%), 128 GRF 8 threads/EU. SLM 18–31 KB/WG에서는 변화 없음, **47.7 KB에서 절반**(SLM/WG는 16/32/64 KB로 올림됨) |

마이크로벤치의 처리량을 attention 전체 속도로 환산하지 않는다. 상한과 메시지 비용을 아는 용도다.

## 7.7 시간 측정 규율 요약 (도구 공통)

1. **한 GPU에서는 한 번에 하나만 실행한다.** 벤치, compiler dump, VTune/GTPin, native dump 진단이 겹치면 해당 구간 측정이 무효가 된다. 실제로 8192 측정이 우연히 겹친 KQ-tail 테스트와 겹쳐 acceptance에서 제외했다. 앞선 GPU 프로세스의 완료(`rc`)를 확인한 뒤 다음을 실행한다.
2. **같은 round의 paired ratio를 집계한다.** raw/micro를 같은 round에서 번갈아 재고 비율의 median을 쓴다. 별도 median 시간의 비율과 소수점이 다를 수 있고, 1.03 초과를 반올림으로 PASS 처리하지 않는다. arm별 median과 paired ratio를 **둘 다** 보고하고 하나만 골라 쓰지 않는다.
3. **cadence를 둘 다 본다.** `queued32/gap0`(커널 32개를 연달아 큐잉)과 `batch1/wait_each/gap200µs`(호출마다 대기). wait_each에서 첫 호출은 clock·cache가 차가워 강한 bimodal(20–40 µs vs 100–140 µs)이다. 한쪽에서만 이기는 후보를 채택하지 않는다. cliloader가 원인이 아님도 확인했다(cliloader 없이 native 하네스에서도 같은 격차가 재현).
4. warm-up과 round 수를 명시한다(예: queued warm6/48 rounds, gap warm64/256, 최종 repeat warm8/64·warm100/512). seed와 arm 순서를 독립적으로 바꿔 repeat한다.
5. 정확도 → 성능 순. 정확도 FAIL은 timing 전에 중단하고, FAIL한 변형의 시간은 폐기한다.
6. 결과 파일(JSON/log), 정확한 command, 소스 SHA256, 사용한 micro binary를 같이 남긴다. 실패·제외 run도 사유와 함께 보존한다(재시도 방지).
7. Bash `cd`는 세션에 남는다. 절대 경로를 쓰고 저장소 루트에서 실행한다. 큰 출력(VTune 진행 로그, IGC `.asm` 헤더, `strings`)은 필터한다.
8. idle gap과 clock: gap 0/200/1000 µs로 같은 커널을 재면 일부 shape에서 +0.7~5.9% 차이가 일관되게 나타났지만, power/frequency 원인은 증명하지 못했다. 이 때는 "cadence 민감성"으로만 기록하고 power 귀속을 쓰지 않는다.
