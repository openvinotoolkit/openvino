# 04. Register spill, GEN ISA 분석, 프로파일링

범위: register pressure / spill(scratch, TPM) 진단, GEN ISA 덤프와 판독, 256 GRF 트레이드오프, 리빌드 없는 A/B(ocloc splice), IGC 파이프라인 함정(`inline` vs `always_inline`, unroll), 커널 임베딩 경로, 마이크로벤치, cliloader 사용.
(→ 01-dpas-and-tiling.md: 타일 선택 / 02-memory-io-prefetch-barriers.md: 메모리 메시지 / 03-numerics-softmax-quantization.md: dequant 수치 / 05-methodology-and-pitfalls.md: 측정 방법론)

표기: **MEASURED** = 하드웨어와 출처를 명시한 실측, **ASSUMED** = 가설/추정. 하드웨어 약어: B580 = Arc B580 (Xe2), B70 = Arc Pro B70 (Xe2, `ocloc -device bmg`), DG2 = Xe-HPG, iGPU = TGLLP 등.
모든 수치는 repo 메모리 노트(`~/.claude/projects/.../memory/*.md`), `test/*.md`, 커널 소스, `docs/sdpa_ocl.md`에서 가져왔다. 재측정하지 않았다.

---

## 4.0 한 장 요약

1. **spill은 결과가 아니라 증상인 경우가 많다.** head-72 prefill에서 SPILL=2688 B는 128개 scalar gather의 부산물이었고, 로드 방식을 바꾸자 spill 0 + 6.96x 빨라짐 (B70 MEASURED, `test/sdpa_ocl_head72_analysis.md` §2.7). 반대로 spill 0을 만들려고 256 GRF를 켜면 오히려 47% 느려졌다.
2. **정답 지표는 런타임 cliloader 커널 이름줄의 `SPILL=` / `TPM=` / `SLM=` / `REG`** 이다. `ocloc` 오프라인 spill은 어떤 경우엔 정확히 맞고(head-72: 2688 B 일치) 어떤 경우엔 0으로 틀린다(tiling sweep 4건). 후보 선택은 런타임 측정으로만 한다.
3. **정적 ISA와 런타임 시간 측정은 역할이 다르다.** ISA dump는 기계 코드 구조를 보여주고, cliloader device time으로 성능을 판정한다. 코드 토글로 특정 연산을 빼서 귀속하면 IGC가 프로그램 전체를 재스케줄/재할당하므로 관측 대상이 달라질 수 있다 (K_DIAG=2: dequant를 지웠더니 더 느려짐, 28.1 -> 32.9 us, B580 MEASURED).
4. **리빌드 없이 A/B**: 런타임이 덤프한 jit prelude(`OV_GPU_DUMP_SOURCES_PATH`) + 작업 트리 `.cl`을 `ocloc`으로 컴파일 (`test/splice_head72.sh`, `test/isa_ab_*.sh`, `test/sdpa_ocl_ab.py`). 런타임 IGC 덤프와 ISA가 바이트 단위로 같음을 확인했다 (§4.6).
5. **plain `inline` helper는 IGC 최적화 파이프라인 자체를 바꾼다.** `__attribute__((always_inline))`만 인라인 코드와 동일 ISA (§4.8).

---

## 4.1 Spill의 종류와 용어

| 용어 | 어디서 보이나 | 의미 |
|---|---|---|
| spill (fill/store) | IGC `.asm` 헤더 `//.spill size N`, `//.spill flag store`; cliloader `SPILL=N` (런타임 커널 이름줄) | register allocator가 GRF에 못 담아 scratch(= thread private memory)로 내린 바이트 |
| private memory / TPM | `.asm` 헤더 `//.private memory size N`, `.zeinfo` `*private*/*scratch*`; cliloader `TPM=N` | **소스의 private 배열이 scratch에 할당됨.** register 부족이 아니라 *런타임 인덱스로 private 배열을 접근*해서 생기는 경우가 있다 (§4.9-A). 이 경우 spill과 별개 |
| numGRF | `.asm` `//.thread_config numGRF=128|256`, cliloader `REG128/REG256`, `.zeinfo grf_count` | thread당 GRF 수. 128 vs 256은 thread/EU 점유율과 교환관계 (§4.5) |
| SIMD | cliloader `SIMD16` 등 | subgroup 크기 |
| `-abortOnSpill 4` | `.asm` `//.full_options` 줄 (모든 덤프에 존재) | IGC vISA 옵션. **`grep spill`로 spill 여부를 판단하면 안 된다**: 이 옵션 문자열 때문에 모든 덤프가 매치한다 (옛 `isa_ab_mixed.sh`가 이걸 셌다. `docs`/ab harness 노트). spill 크기는 `//.spill size` 헤더와 `.zeinfo`에서 읽는다 |

`CL_KERNEL_SPILL_MEM_SIZE_INTEL`(clGetKernelWorkGroupInfo)로도 질의 가능하며 DG2에서 128 GRF > 0, 256 GRF < 128 GRF 조건으로 동작을 확인했다 (`test/sdpa_ocl_xe_hpg/s2/t6.cpp:167`, P7 PASS, DG2 MEASURED).

### 4.1.1 VTune GPU Hotspots: 실제 사용과 해석 한계

VTune은 실제 B70(Xe2, `GPU.1`, PCI device `0xe223`) 분석에 사용했다. 다음은 세션 기록과 저장된 결과에서 확인된 관측이며, source-level 원인 확정과는 구분한다.

| 사례 | 관측 | 해석 범위 |
|---|---|---|
| `vtune_sdpa_gpu_hotspots_b70_20260714_111549`, `sdpa_ocl__prefill` | occupancy 99.0%, XVE active 47.8%, stalled 52.2%, 평균 5.581 ms / 2 instances | 커널이 높은 occupancy를 보였어도 절반가량의 XVE 시간이 stalled였다. occupancy만으로 처리량이나 원인을 판정할 수 없다는 실례다. |
| 후속 OCL 대 `sdpa_micro` 비교 | OCL SBID stall 59.1% 대 34.2%, barrier stall 10.6% 대 2.5% | OCL에서 dependency/barrier 대기 비중이 컸다는 병목 가설을 세웠다. SBID는 scoreboard dependency stall이지 메모리 대역폭 포화의 직접 측정치가 아니다. |
| 후속 소스 상관 분석 | source/DWARF 매핑은 정상, VTune 결과 DB의 stall PC 기록은 0개 | 매핑이 정상이어도 샘플링된 stall PC가 없으면 특정 OpenCL 행이나 ISA 명령에 stall을 귀속할 수 없다. 이 수집은 source-line 병목 증거가 아니다. |

#### 수집 환경과 재현할 때의 체크

- 당시 VTune은 2026.2/2026.3 세션이 있었고, B70 Xe 드라이버에서 시스템 `intel-metrics-discovery` 1.13.545는 초기화에 실패했다. Metrics Discovery 1.16.190을 빌드해 해당 `lib`를 `LD_LIBRARY_PATH` 앞쪽에 두자 counter collection이 동작했다. 이 경로는 당시 임시 빌드의 해결책이지 일반 설치 위치가 아니다.
- 2-GPU 시스템에서 OpenVINO workload는 `-d GPU.1`, VTune은 확인된 B70 PCI BDF를 `-knob target-gpu=...`로 지정한다. 기록된 수집은 `target-gpu=0:3:0.0`을 사용했다. 장치 순번은 시스템마다 다르므로 BDF를 재확인한다.
- 수집 형태(타깃 workload와 BDF는 현재 머신에 맞게 교체):
  ```bash
  vtune -collect gpu-hotspots -knob target-gpu=0:3:0.0 -- \
    env TEST_USE_SDPA_OCL=1 python <workload> <args>
  ```
  OpenVINO의 `GPU.1` 지정과 VTune의 BDF 지정은 각각 확인한다. 성능 A/B는 같은 workload를 별도 result directory로 수집하고, GPU metric과 실제 선택된 커널이 기록됐는지 결과에서 검증한다.
- Xe/i915 counter collection은 커널 보안 설정에 막힐 수 있다. 당시에는 `dev.xe.observation_paranoid`, 구형 i915의 `dev.i915.perf_stream_paranoid` 설정이 필요했다. Source attach에는 `kernel.yama.ptrace_scope`도 영향을 줬다. privileged sysctl 변경은 필요한 경우에만 사용하고, 조직 보안 정책을 확인해 수집 후 원래 설정으로 복구한다.
- OpenVINO에서 생성한 JIT 커널 소스를 보존할 때는 `OV_GPU_DUMP_SOURCES_PATH=<dir>`와 `OV_GPU_MAX_KERNELS_PER_BATCH=1`을 함께 설정한다. 후자가 커널별 별도 소스 파일을 보장한다. 이 파일은 VTune의 source mapping 자체를 활성화하는 컴파일 옵션은 아니며, 매핑된 커널을 식별하고 ISA와 대조하는 데 쓴다.
- `GTPin` 계측 실패 경고가 출력됐어도 해당 수집에서는 타깃 B70의 GPU metrics가 생성됐다. 경고만으로 수집 전체를 실패로 판정하지 말고 result의 GPU/device 데이터와 실제 metric row를 확인한다.
- VTune 결과의 `InternalAddress`는 해당 result archive에 보관된 **정확한 zebin**의 `.text.<kernel>`에만 대응시킨다. 별도 IGC dump는 `.text` 크기 차이(관측 사례 128 B)가 날 수 있어 주소를 그대로 교차 적용하면 안 된다.

#### 관측에서 검증된 최적화까지

OCL의 높은 SBID/barrier 비중을 보고 V-read와 barrier 사이에 독립 작업을 두는 V-prefetch를 먼저 실험했다. B70의 후속 benchmark에서는 V-prefetch 단독 결과가 388.04 ms에서 381.54 ms, 재검증 평균 381.17 ms였고 출력 MD5가 일치했다. 32-row prefetch 변형은 16-row와 차이가 없어 제거했다.

별도 f16 V-read 병합 실험(`16r16x2c`)은 같은 workload에서 521.01 ms 대 동일 빌드 x1c 524.34 ms로 3.33 ms(0.64%) 개선했고 출력 MD5가 일치했다. VTune 지표도 SDPA kernel 구간 2.9%/5.2% 단축, Send stall 22.0–24.8%에서 21.0–22.7%, SBID stall 43.9–45.5%에서 41.5–43.5%로 이동했다. 이 결과는 작은 개선의 단일 사례이며 다른 GPU·모델의 보장은 아니다. 세부 측정 규율은 05장 §5.5를 참조한다.

재수집 시 최소 기록 항목은 VTune 버전, driver와 Metrics Discovery 버전, GPU BDF, workload/반복, 실제 선택된 kernel, 수집 knob, metric 정의, result archive 식별자다. GPU Hotspots의 활성/정지 비율은 병목 후보를 찾는 용도다. 최적화 귀속은 같은 조건의 before/after 수집, device-time A/B, correctness check로 확인하고, stall PC가 없으면 ISA/source 행 단위 결론을 보류한다.

---

## 4.2 진단 도구 1: cliloader (런타임 ground truth)

사용 (모든 sdpa_ocl 측정에서 쓰인 형태, B70):
```bash
# setupvars.sh 가 `set --` 로 "$@" 를 지우므로 인자는 source 전에 저장 (§05 참조)
SDPA_ARGS=("$@")
source <build>/ov_install/setupvars.sh
~/work/opencl-intercept-layer/install/bin/cliloader -d -dv \
  python <llm_bench>/benchmark.py -d GPU.1 -m <model> -n 1 -ic 4 -pf <prompt.jsonl> 2>&1 \
  | grep -E "sdpa_(ocl|micro)__prefill"
```
- `-d` = device performance timing, `-dv` = 커널 이름에 SIMD/REG/SPILL/TPM/SLM/GWS/LWS 부착 (출처: `test/clintercept_report_dev.txt`의 실제 출력 형식, `~/work/opencl-intercept-layer`). `-n 1 -ic 4`로 커널 타이밍엔 충분(~2분), e2e 1st token은 `-n 3 -ic 256` (`sdpa-ocl-beats-micro-256grf`).
- 출력 예 (`test/clintercept_report_dev.txt`):
  `sdpa_micro__prefill_..._sa SIMD16 REG256 SLM=107008 GWS[ 112 x 1024 x 1 ] LWS[ 16 x 32 x 1 ], calls, time_ns, %, avg, min, max`
  `sdpa_ocl_decode__..._sa SIMD16 REG128 SLM=1152 GWS[ 16 x 128 x 2 ] LWS[ 16 x 8 x 1 ]`
  spill이 0이면 `SPILL=`은 줄에 나타나지 않는다 (`sdpa_ocl_prefill_..._sa SIMD16 REG128 SLM=14464 (no SPILL)`, head-72 노트 §2.7).
- 리포트 파일: `test/clintercept_report_{ref,dev}.txt`는 gemma-4-26b decode 2040 토큰 전체 실행의 ref(micro/opt) vs dev(sdpa_ocl) 비교로, 커널 그룹별 device time 합계(총 16.5 s)를 `pa-kvup-token-major-store-sector-floor` 노트가 사용했다. 이 파일들 안의 sdpa 커널에는 SPILL이 없었다.
- 한계: cliloader `-d -dv`는 host-bound라서 2.2 ms 커널 이득을 전체 시간에서 가린다 (노트 인용: "`cliloader -d -dv` is host-bound and hides a 2.2 ms kernel win entirely"). 커널별 device time 합으로 비교하고, e2e와 혼동하지 않는다. 같은 파일에서 wall 23.3 s vs device 16.6 s: 29%가 device time이 아님.
- **occupancy%는 속도 지표가 아니다**: head-128 prefill에서 occ%가 높을수록(70%) 오히려 가장 느렸다 (6.86M ns, B70 MEASURED, `sdpa-ocl-kq-tile-keys-32-slower`).
- 검증 함정: "서로 다른 config로 보이는 측정이 byte-identical"이면 토글이 적용되지 않은 것이다 (setupvars `set --`로 인자 소실; stateful 경로에서는 PA 커널 자체가 안 뜸). SLM/GWS/LWS가 동일한지 먼저 본다.

---

## 4.3 진단 도구 2: IGC 덤프 (정적 ISA, observer-effect 없음)

### 4.3.1 런타임에서 덤프
```bash
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir> \
OV_GPU_MAX_KERNELS_PER_BATCH=1 TEST_USE_SDPA_OCL=1 \
<ov_gpu_func_tests> --gtest_filter=<case> --device_suffix=1
```
(스크립트: `test/dump_isa.sh`, `test/dump_isa_h128.sh`, `test/dump_isa_gemma4.sh`. 출처: `test/dump_isa.sh:1-40`)
- `OV_GPU_MAX_KERNELS_PER_BATCH=1`: 커널마다 별도 program으로 컴파일되어 덤프당 entry가 1개(`*_simd16_entry_0001.asm`). 없으면 여러 커널이 `entry_0002.asm`을 공유해 식별이 어렵다. 부수효과로 batching 시 `FUNC()` 이름 충돌이 가려진다 (§4.7).
- 파일명은 커널 이름이 아니라 hash. 내용으로 식별: `grep -lE 'kernel sdpa_ocl__prefill' <dir>/*.asm`.
- `--device_suffix=1` 필수 (2-GPU 시스템: 없으면 iGPU에서 sdpa_opt가 돌고 비교가 무효).
- 덤프 산출물: `.asm`(최종 GEN), `.visaasm`, `.ll`(`*_beforeUnification.ll`, `*_afterUnification.ll`, `*_optimized.ll`), reg info, `*_options.txt`, `*_internal_options.txt`, `.zeinfo`.
- **`cliloader --dump-kernel-isa-binaries`는 쓰지 말 것**: 산출물이 zebin ELF가 아닌 raw GEN이라 `ocloc disasm`이 "Invalid or missing ELF header"로 실패 (`gpu-kernel-isa-dump`).

### 4.3.2 런타임 소스 덤프 (jit prelude 확보)
`OV_GPU_DUMP_SOURCES_PATH=./` (+ `OV_GPU_MAX_KERNELS_PER_BATCH=1`)은 각 커널을 **모든 `#define`이 앞에 붙은 standalone `.cl`** (`clDNN_program_*.cl`)로 덤프한다. 이것이 §4.6 splice의 입력이다. 이 환경변수는 `GPU_DUMP_SOURCES_PATH` 프로퍼티(`internal_properties.hpp:187`)이며 `OV_` 접두사 + `ENABLE_DEBUG_CAPS` 빌드가 필요하다 (`gpu-tensor-dump-env`). 덤프 끝에 런타임이 `/* Build Log: ... */`를 덧붙이므로 비교 전에 제거해야 한다 (`sdpa_ocl_ab.py corpus`의 `BUILD_LOG_RE`).

### 4.3.3 ocloc으로 오프라인 컴파일
```bash
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=./occ \
ocloc compile -file X.cl -device bmg \
  -options '-cl-mad-enable -cl-std=CL3.0 -Dcl_intel_dot_accumulate -Dcl_intel_global_float_atomic \
            -Dcl_intel_subgroup_matrix_multiply_accumulate -Dcl_intel_subgroup_split_matrix_multiply_accumulate' \
  -internal_options '-cl-intel-greater-than-4GB-buffer-required -cl-intel-has-buffer-offset-arg -cl-store-cache-default=2 -cl-load-cache-default=4'
```
- `-device bmg` = B70/B580(Xe2). **`xe2`는 lnl-m iGPU로 해석**되므로 쓰지 말 것 (`sdpa-ocl-kq-tile-keys-32-slower`). DG2는 `-device dg2` (`sdpa_ocl_ab.py hpg --device dg2`).
- `-cl-mad-enable`은 반드시 `-options` 안에. top-level이면 무시된다.
- build options의 정답 원본은 덤프된 `*_options.txt` / `*_internal_options.txt`. sdpa_ocl은 위 `-D` 4개 + `-cl-mad-enable -cl-std=CL3.0`, decode는 `-cl-mad-enable -cl-std=CL3.0`만 (`sdpa_gen_ocl.cpp:1032-1041`, `sdpa-ocl-ab-harness`). 옛 스크립트(`splice_head72.sh`, `isa_ab_kvup.sh`)는 internal_options 없이 `-cl-std=CL2.0`로 컴파일하고, 최신 `sdpa_ocl_ab.py`는 internal_options까지 넣어 production에 맞춘다.
- `.asm` 헤더에서 읽을 것: `//.instCount`, `//.thread_config numGRF=`, `//.spill size`, `//.private memory size`, `//.full_options`. 스크립트들이 쓰는 grep: `grep -oE '^//\.instCount [0-9]+'`, `'^//\.spill size [0-9]+'` (`test/splice_head72.sh:56-60`).
- 실수 방지: `open(p,'w').write(f(open(p).read()))` 패턴은 읽기 전에 파일을 비운다 -> 커널이 빈 채 컴파일되어 "Build succeeded, `kernels: []`", `.asm` 없음 (`igc-inline-helper-pipeline-switch`).

### 4.3.4 디스어셈블 (`iga64`)
zebin ELF에서 커널 `.text`를 꺼낸 후 `iga64`로 디스어셈블 (`sdpa-ocl-kq-tile-keys-32-slower`):
```bash
readelf -S -W X_bmg.bin | grep '] .text.sdpa'      # offset/size 확인
dd if=X_bmg.bin of=kernel.gen bs=1 skip=<off> count=<size>
iga64 -p=xe2 -d kernel.gen > x.asm                 # Xe2/Battlemage; raw nGEN blob도 그대로 가능
```
- `iga64`는 `/usr/bin/iga64`에 있다 (2026-10-06 확인). `gpu-kernel-isa-dump`의 "iga64 CLI is not installed, only libiga64.so"는 **stale**이며 `sdpa-micro-dpas-operand-mapping`(07-09)과 현재 환경이 맞다.
- 그러나 보통은 IGC 덤프의 `.asm`이 이미 텍스트라서 `iga64`는 (a) zebin에서 직접 뽑을 때, (b) micro의 nGEN blob(`micro::Package::binary`, 아래)에서만 필요하다.
- **sdpa_micro의 진짜 GEMM은 IGC 덤프에 안 보인다.** 덤프에는 wrapper만 있고 GEMM은 `dpas.8x1 ... null` 더미 + `0xFADECAFE` sentinel 자리표시다. ocloc으로 micro `.cl`을 컴파일하면 `DUMMY_DPAS` clobber와 ~709개 가짜 instruction이 나온다 -> micro 숫자를 인용 금지 (`sdpa-ocl-u4-head64-page-read`). 실제 blob은 `sdpa_gen_micro.cpp:~904`의 `init_microkernels()` 직후 `gemms[kq_id].binary`를 파일로 쓰는 env hook을 (임시로) 넣어 얻고 `iga64 -p=xe2 -d micro_vs.bin`으로 읽었다 (hook은 사용 후 되돌림; 결과: `test/minicpm4_8b_sdpa_accuracy/static_micro/{kq,vs}.{bin,asm}`).

---

## 4.4 ISA 판독: 시그니처 -> 의미

Xe2 ISA 문법 (B70/B580): 메모리는 `send`가 아니라 LSC opcode (`load.ugm.d8u32.a64`, `store.ugm...`, `load_block2d.ugm.d8v`); `send`는 LSC 메모리 opcode가 아닌 메시지(barrier/fence/EOT 등)에 보인다 (SLM 접근은 `load.slm`/`store.slm` LSC opcode). `(16|M0)` = 전 lane, `(1|M0)` = SIMD-1 (16 lane 중 1개만 사용). 한 줄 형식 `(pred) opcode (exec|Mask) dst src... {deps}` (`gpu-kernel-isa-dump`).

| ISA 시그니처 | 의미 | 대응 / 측정 사례 (하드웨어, 출처) |
|---|---|---|
| `load.ugm.d8u32/d16u32 (1\|M0)` 다수 | 15/16 lane 낭비하는 SIMD-1 scalar gather. 보통 per-key scale/zp | int8 head-64: 128개. lane-per-key로 1회 wide load 후 `sub_group_broadcast` -> 128 -> 0, instr 4154 -> 3504 (B580 MEASURED, int8-perf) |
| `load.ugm.d8u32 (16\|M0)` x72 + `d16u32` x24 | 중간 크기 per-lane gather (lane은 안 낭비, 메시지 수 과다) | micro는 `d32x16t/d32x32t` wide transposed block load 27개 (B580, 정적 ISA) |
| `load.ugm.d32xKt` (K=16,32) | wide transposed 블록 로드 1메시지 | 목표 형태 |
| `load_block2d.ugm.d8v` / `load_block2d.ugm.*` | 2D block read (VNNI transform 포함) | gate 완화 후 head-72 K/V: 256 scalar gather -> ~13 block read (B70 MEASURED, head72 분석) |
| `dpas.8x8` 개수 / 밀도 | systolic matmul. 정상 prefill 루프 steady-state 32 | head-72 NEW_kv2d_dks_on: dpas 52 = 2 x 26 (IGC가 k0 loop 첫 iteration을 peel, steady body 26 dpas / 13 block2d) (`splice_head72.sh` 주석) |
| `dpas.8x1 ... null:f` + `0xFADECAFE` | **micro blob 자리표시 더미**. 실제 compute 아님 | 위 §4.3.4 |
| 다량의 wide `mov` (`<2>`, `<4;1,0>` strided) | int8 -> f16 widen + VNNI repack mov-storm | ocl 루프 body mov 1250 (1181 wide) vs micro wrapper 352; S·V 영역 679 mov 중 609가 strided reshape, 0이 순수 rename (B580, int8-perf). IGC floor, 소스 재배열로 줄지 않음 |
| `mov rX<2>:hf ...:w` 후 `:uw<2>` repack | half 중간값이 dword-stride(`<2>`) 레이아웃을 강제 | half dequant 144 instr/93 mov vs float 134/83 (microbench). **dequant 산술은 float, 마지막에 `(half)` 한 번** |
| `r[a0]` (indirect register addressing) | lane-varying 루프 변수/런타임 인덱스로 private 배열 접근 | `r[a0]` 2 -> 274, **1.8x 느려짐** (u4 writer, B70 MEASURED, `pa-kvup-u4-token-major-writer`) |
| `//.private memory size N` / cliloader `TPM=N` + hot loop 안의 scratch store/load | 런타임 인덱스 private 배열 -> scratch. ISA 증명: `math.exp` -> `cmp lt 0x7F800000` -> `sel 0x3F800000` 직후 store, `asr 4`(/SG) + `mul 64`(float x 16 lane stride) 직후 load, k0 loop 안에 store 2 + load 2 | llama-3.2-1b MIXED `TPM=128`. select chain으로 교체 -> private ops 4 -> 0, inst 3646 -> 3628, device 0.8% (`sdpa-ocl-mixed-kc-vc-split`) |
| `//.spill size` > 0 / SPILL= | register spill. 다수 개별 gather가 동시 live일 때 흔함 | §4.9 |
| backward branch (goto/jmpi target label이 **더 작은 줄**) | 진짜 rolled loop. forward `goto`는 lane-predicated skip | loop body 경계 찾는 awk: `test/sdpa_ocl_ab.py`/세션 awk. k16 body 620-2981줄 (2361 lines), k32 629-5201 (4572) (B70) |
| `goto`/`join` 개수 급감 + `load.ugm.d32x8t` 51 -> 27 | unroll이 풀림 (loop가 rolled로 돌아감) | §4.10 guard shape |
| `sync.nop/allrd/allwr` 개수 변동 | SWSB dependency 표기. +-1..5 jitter는 흔함. barrier와 무관한 sync 증가는 보통 스케줄 변화 | `isa_compare`에서 `sync`가 heavy 목록에 있어 jitter만으로도 C 판정 (74/86건). send/dpas/math/load/store/CALL 개수가 안 변하고 소스에 barrier 추가가 없으면 무시 가능 |
| `//.thread_config numGRF=256` / `-TotalGRFNum 256` | 256 GRF 모드 | §4.5 |
| zeinfo `disable_mid_thread_preemption` true -> None | **IGC 크기 임계값 (~600 instr)** 이지 기능 변화가 아님. 109,802개 캐시된 컴파일에서 instCount <= 598이면 true, >= 601이면 없음 (599/600은 없음) | 작은 커널이 ~600을 넘나들면 이 키가 뒤집힌다 (`sdpa-ocl-ab-harness`, 2026-09-26) |
| `call`/`ret` 존재 | helper가 인라인 안 됨 | sdpa_ocl은 call/ret 0이 정상 |
| `math.exp`/`math.inv` 개수 | softmax 비용 | head-128 f16 prefill k0 body 379 instr 중 softmax 180 (47.5%), dpas는 32 (8.4%) (B70 MEASURED, `sdpa-ocl-beats-micro-256grf`) |

### 4.4.1 영역 프로파일 (ISA를 region으로 쪼개기)
루프 body 안의 지표 instruction(첫 dpas, `math.exp`, SLM barrier `send`, `store.slm`)을 랜드마크로 영역을 나눠 instruction 비율을 계산한다 (예: K·Q dpas L4119-4661, exp L5161-5252, S·V dpas L5459-6433, int8-perf). 이것은 **정적 instruction 개수**이지 시간이 아니다. 아래 §4.11처럼 정적 개수와 시간이 어긋나는 사례가 있으므로 반드시 ablation/실측으로 확인한다.

---

## 4.5 256 GRF 모드와 occupancy 트레이드오프

### 4.5.1 방법
`-cl-intel-256-GRF-per-thread` build option (`sdpa_gen_ocl.cpp:1038-1041`, env `SDPA_OCL_256GRF`; decode는 `SDPA_OCL_DECODE_256GRF`, `sdpa_gen_ocl_decode.cpp:232`; **xe_hpg는 무조건 on**, DG2는 128 GRF에서 모든 측정 타일이 spill (`sdpa_gen_ocl.cpp:1038-1040`, 195-196)). GRF를 두 배로 하면 thread/EU가 반으로 줄어든다 (Xe2에서 128 GRF = 4 KB/thread, 256 GRF = 8 KB; DG2 GRF는 32 B).
micro는 gemm이 128을 넘게 필요하면 항상 REG256으로 컴파일한다.

### 4.5.2 결정 표 (모두 MEASURED)

| 케이스 | 128 -> 256 GRF 효과 | spill | 판정 | 출처 |
|---|---|---|---|---|
| llama-3.1-8b f16 PA prefill q=4096, **기본 타일** | 2.18M -> 2.73M ns (**25% 느려짐**) | 0 -> 0 | 256은 공짜가 아님 | `sdpa-ocl-beats-micro-256grf` (B70, 2800 MHz 고정) |
| 같은 모델, **큰 타일 + 256GRF** (tq32/pwk4/pwq2, sg_per_wg=8, wgTQ=64, wgTK=64) | 1,796,200 ns vs micro 1,930,454 (**7.0% 빠름**; 기본 ocl 대비 17.7%) | 큰 타일은 128GRF에서 spill | **타일과 한 쌍으로 튜닝해야** 이득 | 동일 노트. `sv_sg_tile_scores >= 64` 구성은 128 GRF에서 전부 3.8-19 KB spill |
| gemma-4 head-72 plain prefill (scalar gather 상태) | 9.57 -> 14.12 ms (**47% 느려짐**) | 2688 -> 0 | spill 제거는 이득이 아님. 병목은 LSC 메시지 수 | `sdpa_ocl_head72_analysis.md` §2.7 (B70) |
| gpt-oss-20b u4 head-64 MIXED (gather-bound) | 1,474,574,702 -> 2,502,030,737 ns (**+70% 악화**) | 5952 -> 0 | gather latency를 thread 수로 가리는 커널은 256GRF가 손해. 이긴 수정(c)은 spill 5952 -> 13760 B로 *증가*했음 | `sdpa-ocl-u4-head64-page-read` |
| gemma-4 decode head-512 | 256GRF가 sg16과 동급(157.1 vs 158.6M), head-256에서는 손해 | - | 답이 아님 | `sdpa-ocl-decode-tiling-sg-per-wg` |
| DG2 (SG8) plain prefill 타일 sweep | 256 GRF + 16x16 + 키 방향 sg4: spill 0. 128 GRF는 전 구성 3.8-11 KB spill. k16q32/k32q32는 256에서도 spill | - | HPG 기본 256 | `test/sdpa_ocl_xe_hpg/s2/S2_RESULTS.md` (DG2 소형 프로토타입, 제품 PA 전체 최적성 증명 아님) |

### 4.5.3 규칙
- **spill 바이트를 목적함수로 삼지 않는다.** scratch는 L1 상주라 192개 scattered 메시지에 비하면 싸다. 먼저 *왜* 많은 값이 동시 live인지(독립 gather 128개)를 본다.
- 256 GRF는 "큰 타일이 128에서 spill할 때 + 타일을 함께 키울 때"만 이긴다. 기본 타일에 256만 켜면 손해.
- occupancy는 축이 둘이다 (WG 수 vs threads/WG). `sg_per_wg` 8 -> 16이 gemma-4 head-512 decode에서 2.07x 빨랐고(head-256 1.26x), 같은 노브가 head-128에서는 중립/악화였다. 판별자는 `V_TILES >= 16` (`sdpa-ocl-decode-tiling-sg-per-wg`, B70 MEASURED). 한 shape에서 "중립"이었던 노브는 죽은 노브가 아니다.
- `SDPA_OCL_256GRF=0/1`만 바꿔서는 HPG에서 128/256 실험이 안 된다 (HPG는 항상 256 강제). 진짜 비교는 build option 문자열을 확인한 뒤 별도 코드 변경으로 (`sdpa-ocl-xe-hpg-s7-performance`).
- SLM 점유: Xe2 Xe-core SLM 128 KB, DG2 64 KB (`sdpa_gen_ocl.cpp:155-176`, `max_slm_bytes_for`). k32 변형이 SLM 17536 -> 25728로 늘어 occupancy가 떨어진다고 가설을 세웠으나 **device 측정으로 반증됨** (§4.11-3).

---

## 4.6 리빌드 없는 A/B: ocloc splice

### 4.6.1 원리
jit 상수(`kq_sg_tile_keys`, `USE_2D_BLOCK_IO_KV`, `HEAD_SIZE` ...)는 소스 앞에 붙는 `#define`이므로 OpenVINO 리빌드 없이 바꿀 수 있다. 런타임 소스 덤프의 prelude(첫 `#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate` 앞까지, 또는 첫 `#define UINT4_RANGE` 앞까지)를 떼고, 스윕할 `#define`을 prelude에서 `sed`로 지우고 `-D`/`#define`으로 다시 공급하고, 뒤에 `#include "<작업트리>/sdpa_ocl.cl"`을 붙여 ocloc으로 컴파일한다 (`test/splice_head72.sh:30-48`, `test/isa_ab_mixed.sh:1-30`).

### 4.6.2 스크립트 목록 (모두 untracked, 사용자 실행)

| 스크립트 | 용도 |
|---|---|
| `test/splice_head72.sh` | head-72: HEAD vs 작업트리 x (scalar/kv2d) x (DKS_ACTIVE on/off) 6행 표 (`inst spill dpas block2d load.ugm baseFixup`) |
| `test/isa_ab_mixed.sh {dump,compile,ab,all} [case]` | PA MIXED 커널 모드 sweep (f16_dmajor, i8_bytoken_2d, u4_bychannel_1d, prefill, sink ...). 의도하지 않은 모드는 **instCount와 opcode 히스토그램이 동일**해야 함 |
| `test/kvup_splice.sh {splice,check}` | `pa_kv_cache_update_ref.cl` splice. `NAME=VALUE`(대체), `+NAME=VALUE`(덤프에 없던 상수 추가), `-UNAME`(정의 해제; `#ifdef` 테스트용). 오타 가드: 덤프에 없는 이름은 에러. `check`가 HEAD `.cl`로 덤프 ISA를 정확히 재현하는지 먼저 증명 |
| `test/isa_ab_kvup.sh` | kvup 전 모드 HEAD vs 작업트리. 모드 7개 byte-identical 확인 |
| `test/splice_bidir.sh`, `splice_pa_cur_f16.sh` | bidir mask / PA f16 splice |
| `test/sdpa_ocl_ab.py` | 일반화된 harness (§4.6.4) |
| `test/probe_decode_build.cl` + `probe_decode_run.cpp` | decode 커널용 standalone: 런타임이 만드는 jit prelude를 손으로 재현하여 `ocloc`으로 컴파일(+ISA 조사)하고, 합성 PA GENERATE 문제로 GPU 실행해 수치 검증. `probe_decode_build_bmg.{bin,spv}`는 그 산출물. 빌드: `g++ -O2 -o probe_decode_run probe_decode_run.cpp -lOpenCL` |
| `test/probe_m1_dpas.cl` | M=1/2/4 `intel_sub_group_f16_f16_matrix_mad_k16`가 IGC에 받아들여지는지, dpas rep-count, 2D block transpose/transform 형태 확인 (ocloc 단독) |
| `test/microbench/` | §4.12 |

### 4.6.3 신뢰성 증명 (필수 절차)
splice 결과를 믿으려면 **먼저 알려진 기준을 재현**해야 한다.
- head-72: `HEAD_scalar`가 **spill 2688**을 재현 = cliloader가 device에서 보고한 값과 일치 (B70). `SDPA_OCL_DKS_ACTIVE=0`이 HEAD 대비 바이트 단위 no-op임도 같은 표에서 증명 (identical instCount/spill/dpas/block2d/load.ugm).
- `ocloc(k16)` ISA == 런타임 IGC 덤프 정확히 일치: dpas 32=32, load_block2d 16=16, send.ugm 98=98, 총 instr 3039=3039, spill 0=0, 같은 hash 8d4bbcc78f6ee655 (B70, `sdpa-ocl-kq-tile-keys-32-slower`).
- kvup: `./kvup_splice.sh check`. 단, `dump_ref`는 reference 빌드 산출물이라 현 `.cl`로 재현 불가 -> `dump_dev`를 쓴다 (`pa-kvup-token-major-store-sector-floor`).
- 한계 (스크립트 주석): ocloc build option은 OV와 다르므로 **절대 instCount/spill은 같은 표의 행끼리만 비교**. dpas / block2d / load.ugm 개수는 구조적이라 의미가 있다.

### 4.6.4 `sdpa_ocl_ab.py` (offline A/B 체계)
상태 디렉토리 `test/sdpa_ocl_ab/{snapshots,revs,cache}`. 레벨:
| 레벨 | 비교 대상 | 의미 |
|---|---|---|
| L0 | 임베딩될 소스 텍스트 (`embed()`: kernels_db_gen.py 재현) | 주석/들여쓰기 변경은 L0 동일 (주석은 런타임에 도달하지 않음) |
| L1 | `clang -E` 토큰 리스트 | 공백 재배치 제외한 같은 토큰. `const int` vs `constint`를 구별하도록 토큰 리스트 비교 (selftest `merge`) |
| L1' | IGC front-end IR (`*_beforeUnification.ll`) | 괄호 추가는 L1' 동일/L1 상이 |
| L2 | ocloc ISA 지표: `//.instCount`, `numGRF`, opcode 히스토그램(predicate 제거), `.zeinfo` execution_env(`slm_size`, `grf_count`, `barrier_count`, `eu_thread_count`, scratch/spill/private) | 등급 A=동일, B=heavy op 동일+scratch 증가 없음+inst +-0.5% (실측 필요), C=그 외 |
| `pset` | host-only 변경: jit (name,value) 멀티셋 동일성 | 커널 불변 증명 |
| `built` | 빌드 후 `.inc`(chunk 연결)과 링크된 `.so`, `ov_gpu_unit_tests`에 해당 텍스트가 있는가 | §4.7 |
| `hpg` | DG2 등 다른 device ocloc 컴파일 TSV (`rc, sg, dpas, dpas_forms, simd, numGRF, spill, slm, inst`) | spill은 **ocloc 컴파일러 보고값이지 DG2 런타임 값이 아님** (`sdpa-ocl-xe-hpg-s7-performance`) |

비용: 2332 config + 합성 6110 job, 32 core `-j 28`: cold L1+L2 ~30분, one side cached ~10분, L1만 2-3분. 토큰 키 캐시 `cache/tok/`로 토큰 동일 config는 컴파일 생략. `--l2-from-l1`: L1 동일 job은 base 컴파일 재사용(ocloc은 입력 토큰에 대해 결정적이므로 타당; 단 driver front-end만 predefine하는 매크로에 `#if`가 추가되지 않았을 때). 자기 테스트: `selftest`는 harness 수정 후 항상 재실행 (negative control).
C등급 진단: `compile_one(text, kernel, keep_dir=dir)`로 IGC 덤프를 보존하고 `%name`/label 정규화 후 `beforeUnification -> afterUnification -> optimized -> ISA` 순서로 diff하여 **처음 달라지는 단계**를 찾는다 (§4.8).

---

## 4.7 커널 임베딩 경로 (.cl -> .inc)와 검증

- `ocl_v2/*.cl`은 런타임에 디스크에서 읽지 않는다. `CMakeLists`의 `file(GLOB_RECURSE KERNELS "*.cl")`이 빌드타임에 `build/src/plugins/intel_gpu/graph/impls/ocl_v2/codegen/include/gpu_ocl_kernel_sources.inc`(**`intel_gpu/` 뒤에 `src/` 없음**; `.../intel_gpu/src/graph/...` 사본은 09-17의 stale 잔재)로 임베딩한다. GLOB은 configure-time이므로 **새 `.cl` 파일은 cmake 재실행**이 필요하다.
- `.inc`는 긴 커널을 인접 raw-string chunk(`)__krnl"` 줄바꿈 `R"__krnl(`)로 분할하므로 grep이 chunk 경계의 토큰을 놓친다. 정확한 확인: `python3 test/sdpa_ocl_ab.py built --base <rev|snapshot> [--release]` (chunk를 연결해 비교 + `bin/intel64/<cfg>/libopenvino_intel_gpu_plugin.so`와 `ov_gpu_unit_tests`(그래프 라이브러리를 정적 링크하므로 **자체 커널 사본**을 가짐: 단위 테스트만 relink가 stale이면 옛 커널이 돈다)를 mmap 검색. 링크가 끝난 뒤 실행(쓰는 중인 .so는 false negative). 1초 미만.
- 두 빌드 트리(`build/` Debug, `build_release/`)에서 `.inc`와 `.so`가 중단된 빌드 후 서로 어긋날 수 있다. `.so`를 확인한다.
- 코드젠 규칙 (`src/plugins/intel_gpu/src/graph/common_utils/kernels_db_gen.py`):
  - **top-level** `ocl_v2/*.cl`만 커널 엔트리 (재귀 안 함). sibling header도 top-level `.cl`이며 이름으로 include.
  - non-batch `#include "x.cl"`은 빌드타임 inline, 커널당 **1회**(두 번째 include는 `[[no_opt]]`가 없으면 조용히 무시). 해석 실패한 include는 **조용히 `''`로 치환**. batch-header include는 맨 위로 호이스트되어 런타임에 해석.
  - 주석 제거, 공백 최소화, 미사용 `#define` 제거, 모든 `#define`에 `#undef` 추가. 단 **본문 없는** `#ifndef X`/`#define X` guard 쌍은 제외 -> 클래식 include guard는 batched program의 두 번째 커널이 헤더를 건너뛰게 만든다. 본문이 있는 sentinel(`#define X_INL 1`)을 쓴다.
  - 여러 커널이 한 OpenCL program을 공유하므로(batching) helper/타입은 `FUNC(name)`/`FUNC_CALL(name)`. `OV_GPU_MAX_KERNELS_PER_BATCH=1`은 충돌을 가린다.
- **주석은 런타임에 도달하지 않으므로 주석만 바꾼 편집은 L0로 no-op임을 증명** 가능 (`sdpa_ocl_ab.py l0`).
- 런타임 소스 덤프 == `embed(base)`에서 `#include "include/batch_headers/..."` 줄만 뺀 것 (바이트 단위 증명). 그래서 `corpus`가 prelude/suffix를 분리하면서 *어느 빌드에서 나온 덤프인지*까지 증명한다. index에는 `HEAD`가 아니라 **전체 SHA**로 base를 저장한다 (다음 커밋에서 base가 조용히 바뀌지 않도록).
- host-only(C++) 컴파일 점검: `build/.../ocl_v2_obj.dir/flags.make`의 `CXX_DEFINES/INCLUDES/FLAGS`로 `c++ ... -fsyntax-only <file>` (~3 s). `-Wall -Werror -Wmissing-declarations` 타깃이지만 `-fsyntax-only`는 anonymous-namespace의 미사용 함수를 보고하지 않는다 (codegen에서야 -Wunused-function). 빌드 성격의 실행이므로 사용자 승인을 받는다.

---

## 4.8 `inline` vs `always_inline`: IGC 파이프라인 전환

**사실** (ocloc 26.22.38646.4, bmg, MEASURED 2026-09-24, `igc-inline-helper-pipeline-switch`):
- plain `inline` helper는 IGC가 인라인하지만 **모듈 전체**가 다른 최적화 파이프라인을 탄다. 1줄짜리 `inline size_t FUNC(pa_v_page_base)(...)`를 추출했더니 MIXED 122개 config 중 121개가 C등급 (inst -72..+17).
- 대조 실험: 무관한 identity helper `inline int FUNC(id)(int x){return x;}`를 한 번 사용 -> **같은 delta 재현** (1835 -72, 1818 -28, 1817 +4), 그 helper를 호출하지 않는 plain-SDPA config도 이동(000 +18, 001 +4). 즉 helper 내용이 아니라 "alwaysinline이 아닌 함수가 존재함"이 원인.
- IR 증거: `*_afterUnification.ll`은 value 이름만 빼면 동일, `*_optimized.ll`부터 다름 (helper 빌드에서만 kernel arg에 `nocapture readonly`가 추론됨).
- 해결: `__attribute__((always_inline)) inline`. 현재 `sdpa_ocl_config.cl:268`의 `#define SDPA_OCL_INLINE __attribute__((always_inline)) inline`. front-end IR에는 helper와 호출이 여전히 남지만(clang이 인라인하지 않음) IGC가 alwaysinline 함수를 다르게 취급한다. repo 내 다른 GPU 플러그인 커널은 always_inline을 쓰지 않으므로 이미 "other 파이프라인"에 있다.

`always_inline`이어도 ISA를 움직이는 경우 (LLVM inliner가 인자값으로 clone 본문을 먼저 단순화하기 때문; `docs/sdpa_ocl.md:138-175`):
| 함정 | 원인 | 처방 |
|---|---|---|
| 확장된 파라미터 캐스트 | 커널의 `size_t lane`(= `zext(get_sub_group_local_id())`)을 helper에서 `(int)lane` -> clone 시 `trunc(zext x) -> x`로 entry-block lane id 재사용, 인라인 코드는 사용처마다 새 zext. mov -2/-4 | helper는 `size_t lane`(size_t 산술용)과 `int lane_i`(int 산술용)를 모두 받고 변환은 호출자가. **`uint lane`으로 바꾸면 size_t 사용처가 더 나빠짐(spill 증가)** |
| 상수가 되는 파라미터로 분기 | `USE_BIDIR_GATE=0`에서 `bidir_active=true` -> clone 시 분기 제거 -> CFG 상이, +16 `sync` | 그 테스트는 호출 지점이나 `#if`에 둔다 (bidir query group loop은 inline 유지) |
| 좌표를 합쳐서 전달 | `VcD_x0 + sg_j0_sv`를 하나의 인자로 넘기면 unroll된 루프 밖으로 hoist되어 순서 변경 (+23 sync) | leaf(`VcD_x0`, `sg_j0_sv`)를 따로 넘기고 helper 안에서 합산 |
| **helper 내부 배열** | inliner가 helper 자체 배열을 lifetime marker로 감싸 인라인 코드에 없던 stack 할당 변화. u4 head-512 MIXED(spill-bound)에서 scratch 7680 -> 7552 B | scratch 배열(`kt`, `kw`, `vt_pa`, `vzp4`, `zpb4`, `v_pg`)은 호출자가 선언해서 `__private` 포인터로 전달 |
| loop-carried 출력 | `q_pack`을 caller loop 변수로 두면 SROA가 이전 값을 phi로 유지, helper-local vector는 그것을 버려 u4 head 512에서 register allocation 이동 | out-parameter로 |

원칙(리팩터링 때 확립, 6 commits `1706b74354..0653092a5a`): helper는 인자 타입을 caller 그대로, unroll trip count는 매크로, private 배열은 unroll 상수로만 인덱싱, 포인터에 명시적 `__private/__global`, struct/global 없음. shape_info를 읽는 prelude 매크로(`MSK_*`, `KEY_COMP_OFF`, `VAL_COMP_OFF`)를 쓰는 helper는 `OPTIONAL_SHAPE_INFO_ARG`를 받는다.

---

## 4.9 Spill / scratch를 제거한 방법 (증거가 있는 것만)

### A. 런타임 인덱스 private 배열 -> select chain (TPM 제거)
증상 `TPM=128`: `float alpha[kq_query_blocks]`(2 float x 16 lane x 4 B)가 런타임 인덱스 `alpha_qb`(=`sg_ij`에서 유도)로 접근되어 scratch로 이동. k0 루프 안에 store 2/load 2.
```c
// sdpa_ocl.cl:845-862 (현행)
float alpha_sel = alpha[0];
#pragma unroll
for (int t = 1; t < kq_query_blocks; ++t)
    alpha_sel = (t == alpha_qb) ? alpha[t] : alpha_sel;   // kq_query_blocks는 컴파일타임 상수
#pragma unroll
for (int rr = 0; rr < 8; ++rr)
    av[rr] = sub_group_broadcast(alpha_sel, alpha_lane0 + rr); // runtime lane은 indirect reg move
```
효과: private memory 128 -> 없음, in-loop private ops 4 -> 0, inst 3646 -> 3628, **device 0.8%** (llama-3.2-1b MIXED, 출처 `sdpa-ocl-mixed-kc-vc-split`). pre-existing(HEAD에도 있었음).
일반 규칙: **private 배열은 unroll 상수로만 인덱싱한다.** 런타임 값으로 고르려면 select chain / `sub_group_broadcast`.

### B. scalar gather를 block read로 교체 (spill은 부산물로 사라짐)
head-72 plain prefill: `block2d_surface_ok()`가 폭/pitch/base 정렬 3규칙을 `row_bytes % 64 == 0` 하나로 묶어 head-72(144 B)에서 K/V가 **subgroup당 k0 iteration당 256개 scalar gather**로 떨어짐. `%16`으로 완화(+ base를 64 B 아래로 내림 `x += prem/elem, w += prem` 보정)하자 spill 2688 -> 0 (B70 MEASURED: 9,570,971 -> 1,030,869 ns, 전체 9.28x; 1단계(KV_2D만) 6.96x). 상세 → 02-memory-io-prefetch-barriers.md.

### C. 로드 geometry 개선 (scale/zp hoist)
int8 head-64: per-key scale/zp를 dequant 루프 안에서 읽으면 `(1|M0)` 128개. lane=key로 wide load 1회 + `sub_group_broadcast`(`krel%SG`가 컴파일타임 상수라 단순 reg move). **broadcast는 per-lane `(head<d)` guard 밖**(collective). OOB key는 scale/zp=0 (dequant 결과 0). 128 -> 0, instr 4154 -> 3504 (B580 MEASURED, 사용자가 실모델에서 의미 있는 속도 향상 확인).

### D. 타일/스레드 수 조정 (MEASURED)
- 큰 타일은 128 GRF에서 spill -> `256GRF + 타일 재조정` 한 쌍 (§4.5).
- decode `Q_PER_WG`(M): gemma-4 head-512 M=8은 SPILL=34432 B, pa_sdpa_opt 대비 **3.14x 느림**, M=1은 spill 0. `live_grf_estimate()`(`sdpa_gen_ocl_decode.cpp:68-86`)가 KQ 루프를 가로질러 live인 배열(q_reg[M][K_TILES] half, k_sc/k_zp, k_corr, s[], m/l 5M, S*V accumulator V_TILES x float8 ...)을 세어 **budget 112**(128 GRF보다 낮게; 컴파일러 temp 미포함)로 상한. head-512 sg8에서 점수 136/164/220/332 vs spill 0/6400/15936/34432 B (단조), 그러나 **~15 GRF 미만 차이는 구별 못 함** ((8,M=1)=136은 spill 없음, (16,M=2)=126은 640 B spill). 즉 coarse gate이지 예측기가 아님. 같은 SG_PER_WG에서 M에 대해 단조.
- KV 페이지 key-tile을 좁힘(config A) vs 스레드 수를 줄임(config C): A는 146,417 ns (default 153,227)이고 C(sg8, wgK128)는 spill 11776 B로 372,628 ns — "tile을 줄이면 이기고 thread를 줄이면 무너진다". B/C는 **ocloc이 spill 0으로 예측했으나 런타임 spill 발생** (`sdpa-ocl-mixed-kc-vc-split`, B70).
- SV_TRIM / V_PREFETCH (u4 MIXED): unroll된 cp loop에서 `cp*SUBGROUP_SIZE >= k_chunk` 블록의 S*V 생략(SV_TRIM) 151,677 vs 158,422 ns, V_PREFETCH를 S_max aggregation barrier 직후에 넣어 145,603 vs 151,866 ns (**4.124%**), micro 147,825 ns (B70 MEASURED, `sdpa-ocl-mixed-exact-complete`). 이 노트도 "REG128만으로 spill 여부나 악화 원인을 단정하지 않는다"고 명시.

### E. 배열 lifetime / 재계산 vs 보관
- 증거가 있는 것: helper 내부 배열은 caller 선언으로 (§4.8). u4 writer: `token_vals[]`를 레지스터에 두려면 루프 변수가 uniform해야 한다 (§4.10).
- **증거 없음** (시도 기록이 없어 서술하지 않음): `#pragma unroll N` 횟수 제한으로 spill을 줄인 사례, 커널 분할로 spill을 줄인 사례. `sdpa_ocl.cl`의 `#pragma unroll` 38곳은 전부 무인자 full unroll이고, `sdpa_ocl_decode.cl`은 `unroll_for`(= `__attribute__((opencl_unroll_hint)) for`, `batch_headers/common.cl:25`)를 쓴다. unroll 횟수를 제한한 곳은 sdpa_ocl 계열에 없다 (`#pragma unroll 1`은 `sdpa_ref.cl:341`에 FP16 정규화 오류 회피용으로만 존재하며 spill과 무관). 커널 *분할*은 spill 목적이 아니라 PA kvup의 V workgroup 분리(§4.10, 성능 목적)에서만 쓰였다.

---

## 4.10 unroll 함정과 guard shape

(→ 01-dpas-and-tiling.md: DPAS 루프 구조)

`pa_kv_cache_update_ref.cl`의 u4/i8 BY_CHANNEL writer (B70 MEASURED, `pa-kvup-u4-token-major-writer`):
1. **lane-varying 루프 변수는 배열의 register를 빼앗는다.** `for (t = par; t < tpb; t += 2)` -> `t>>1`이 uniform임을 증명할 수 없어 `token_vals[]`가 indirect register addressing으로 이동 (`r[a0]` 2 -> 274), shuffle 버전보다 **1.8x 느림**. 해결: `it/npo/np` 등 uniform bound와 uniform tail. lane-varying *주소*는 괜찮다(그냥 gather).
2. **uniform guard의 모양이 안쪽 루프의 unroll 여부를 결정한다.** i8 prefill (unsplit 8.2 us) 3가지:

   | guard 모양 | 정적 instr / ISA | 시간 |
   |---|---|---|
   | 루프 bound를 0으로 | unroll 사라짐 (7850 -> 2590) | - |
   | inline helper 맨 위 `if (skip) return;` | unroll 사라짐 (`load.ugm.d32x8t` 51 -> 27, goto/join 24/17 -> 3/3) | **18.7 us (2.3x 느림)** |
   | **호출 지점**에서 같은 테스트 | unroll 유지 | **5.6 us** |

   규칙: guard는 호출 지점에, helper 본문/루프 bound에는 두지 않는다. V 루프도 body가 아니라 *루프를 감싼다*. 이후 `docs/sdpa_ocl.md:172-174`가 일반 규칙으로 승격 ("runtime 또는 select-based bound, helper의 early `return`이 unroll을 잃게 했다").
3. **표현식 단위 type promotion 보존**: `range = (max==min) ? 0.004 : (max-min)`은 **half**에서 뺄셈 후 widen, `fabs(max*0.1f)`는 float. prefill helper는 scale/zp를 half로 narrow한 뒤 quantize, requantize helper는 float 유지. 교체하는 쪽 arm을 그대로 따라야 수치가 같다 (→ 03-numerics-softmax-quantization.md).
4. **ablation은 루프 bound를 0으로** 만든다 (`if (0)`은 shared `for (j...)`에서 `new_idx = j - token_pos_in_block`을 음수로 만들어 OOB 읽기 CL error -14). 한쪽 절반만 끄려면 `for (int j = (int)token_pos_in_block; ...)`.
5. `__attribute__((reqd_work_group_size(1,1,SUBGROUP_SIZE)))`가 있으면 모든 enqueue의 `lws[2] != 16`이 `CL_INVALID_WORK_GROUP_SIZE(-54)`. 진단: `clGetKernelWorkGroupInfo(CL_KERNEL_COMPILE_WORK_GROUP_SIZE)` (40줄 standalone 질의).
6. **정적 instCount는 지표가 아니다**: 고쳐진 u4 kernel은 정적 +643 instruction이고 더 빠르다. 동적 개수가 지표 — `current_token_pos_in_block`을 리터럴로 고정해 모든 루프 bound를 컴파일타임으로 만들면 실모델 generate 비율을 0.7 pp 이내로 예측했다.
7. IGC는 k0 루프 첫 iteration을 **peel**한다 (head-72: dpas 52 = 2 x 26). dpas 개수를 셀 때 steady-state body는 절반.

---

## 4.11 Falsified / 반직관 사례 모음 (모두 MEASURED)

1. **observer effect**: int8 K 경로에서 dequant만 빼기(K_DIAG=2): 28,124 -> 32,916 ns (더 느림). scale/zp load 제거가 IGC 스케줄/할당을 흔든다. K 경로 전체 제거(K_DIAG=1)만 일관적(~9 us 감소). -> ISA 정적 분석 + 구조적 변경만 신뢰 (B580, `sdpa-ocl-int8-perf`).
2. **K를 2D block으로 읽기 (non-transform 8b)**: spill 5888/2944로 올라가고 30.8/30.3 us로 scalar(26.1 us)보다 느림. 이후 **transform(VNNI) read + hoisted scale/zp**를 쓰는 2단계 재설계로 15.1 -> 12.8 us (B580). 교훈: 읽기 *개수*가 아니라 *geometry*(SIMD-1 scale/zp, 중간 크기 메시지)가 병목이었다. V의 cp-pair read 재사용(8 -> 4 reads)은 3가지 구현 모두 효과 없음/악화(22.8 vs 21.7 us) -> V read count는 레버가 아님.
3. **kq_sg_tile_keys 16 -> 32가 느린 이유**: 처음 가설(SLM 17536 -> 25728 B로 occupancy 7 -> 5 WG/Xe-core)은 **device 측정으로 반증**. 실제 원인은 (a) `kq_wg_tile_keys = tile_keys x per_wg_keys`와 (b) `sg_per_wg` 두 축 모두 "클수록 느림" (B70 2800 MHz 고정, head-128 compressed prefill q=4096):

   | config | pwk | sg_per_wg | SLM | occ% | min ns |
   |---|---|---|---|---|---|
   | k16pwg4 | 4 | 8 | 12928 | 54.7 | **3,526,979** (micro 4.17M보다 빠르게 보고됨; 유효성 의문, 아래 주의) |
   | k32wg | 4 | 8 | 17024 | 58.5 | 4,346,666 |
   | k16 (default) | 8 | 16 | 17536 | 63.6 | 5,102,600 |
   | k32 | 8 | 16 | 25728 | 65.3 | 5,587,600 |
   | k16pwg16 | 16 | 32 | 26752 | 70.2 | 6,864,687 |

   ⚠ pwg4/pwg2 계열 타이밍은 `kq_sg_per_wg_keys`만 오버라이드해 불변식 3을 위반한 무효 config였을 수 있다는 기록이 있어 "k16pwg4가 micro를 이긴다"는 **미확정**이다 (01장 §6.1 "미해결 모순"). 유효 config 재스윕은 01장 §6.2.

   ISA로 배제된 것: spill (모두 0, GRF 128), 총 instr(k32가 ~2% 적음), sync(k32가 더 적음), V-load 지연 숨김, dispatch(GWS/LWS 동일), dpas 병렬성. **정적 ISA에서 불리한 것이 하나도 없는데 느림** -> ISA만으로 부족, device 측정 필수. 메커니즘 가설(ASSUMED): key 축이 reduction 축이라 subgroup 증가 = S_max atomic_max + S_sum SLM + barrier 오버헤드 증가. (WG 수는 모든 config에서 2048로 동일하므로 grid underfill 이야기가 아님.)
4. **block-level causal mask skip**: 구현/검증 완료(9.3M 원소 zero change)했으나 IGC가 branch를 취하지 않고 flatten: cmp 20 -> 36, net instr +16. k_mask remainder add 제거도 add 60 -> 20 이지만 mov 59 -> 82, net -3. (B70, `sdpa-ocl-beats-micro-256grf`). **소스의 분기 의도 != IGC 코드젠.**
5. **K dequant를 half로**: 실모델에서도 더 느림. float 134 instr/83 mov vs half 144/93 (`<2>` dst 때문, `:uw` repack +52). V의 `vs_c/vz_c`는 half로 캐시해도 되는 이유(scalar-per-lane multiply, word->half widen 배열 없음)가 있으므로 "통일" 금지.
6. **store sector floor**: pa_kv_cache_update_ref head-256 u4 token-major가 d-major보다 +12.1% (6312 -> 7074 ns, B70). **ISA/메시지 수/indirect addressing 개수는 전부 같거나 더 좋았는데(instr 4249 vs 4448, `r[a0]` 125 vs 188, store.ugm 50 vs 51) 시간은 store 루프만 4.8x**. 원인은 thread당 dirty cache sector (16 vs 2). ablation 표 (probe, head 256, ns/dispatch):

   | | floor | old-token loads | data stores | total |
   |---|---|---|---|---|
   | d-major | 2201 | 723 | 423 | 3353 |
   | token-major | 1283 | 889 | 2014 | 3700 |

   고정된 후보 4개가 측정으로 **반증됨**: (1) workgroup 내 sector 공유 co-location `PA_K_SGS_PER_WG` sgs 1/2/4/8/16 전부 이득 없음(sgs=2 +479 ns): **Xe2는 다른 thread의 sub-line write를 합치지 않는다**, (2) 컴파일타임 unroll+predication 3700 -> 3618 (2%), (3) partitions 16 -> 8은 비율은 좋아지나(0.90) 절대값 악화(4337 vs 3700), (4) partition-tiled page layout은 reader 재작성 위험으로 기각. 천장은 986 ns/dispatch (0.024 ms/token): 도달에는 SLM staging + barrier가 필요해 실질 이득은 parity. 결론: **production 코드 변경 없음.** 교훈: "개선되는 ratio가 악화되는 absolute를 가릴 수 있다" (parts=8), 접근 *패턴*이 바뀌면 지표는 thread당 sector 수이며 ISA 개수로는 안 보임.
7. **spill 수치가 스스로 모순**: DKS_ACTIVE 8/6/5에서 instCount는 7043 -> 5808 -> 5631로 줄지만 spill은 2688 -> 13568 -> 16064로 단조 *증가*(scalar fallback만; head 48/96에서만 도달). **ocloc 정적값이며 device 미측정** (ASSUMED 위험으로 `docs/sdpa_ocl.md`에 기록). 이 위험 때문에 DKS_ACTIVE를 되돌리지 말고 기본값을 좁히라고 기록됨.
8. **ocloc spill 예측의 신뢰도가 일관되지 않음** (§4.2 참조): head-72 splice는 2688 B 정확히 일치, 반면 256GRF sweep 후보 중 4개는 ocloc spill=0 이었으나 런타임 3.3k-19k B spill, kc/vc split config B/C도 마찬가지. 불일치 원인은 이 repo 자료에서 규명되지 않았다 (ASSUMED: build option/internal option 차이 가능성; 최신 `sdpa_ocl_ab.py`는 internal_options를 넣지만 이 점을 spill에 대해 재검증한 기록은 못 찾음). 규칙: 런타임 `SPILL=`만 신뢰.

---

## 4.12 마이크로벤치 (`test/microbench/`)

관측자 효과 없는 도구. 실제 커널을 편집/빌드하지 않고 분리된 .cl을 ocloc `-device bmg`로 컴파일하여 instruction mix/mov 개수 비교 (`test/microbench/README.md`).
- 정적 ISA: `./compare_isa.sh [k_dequant_float.cl k_dequant_half.cl ...]`. 결과: K dequant float 134 instr/83 mov vs half 144/93; V dequant baseline 93 mov/cp-block, `A_char16` 93, `B_nozp` 68, `C_char2pack` 68, `E_reinterpret` 68, `F_vec_zp` 93 (zp 제거만 효과 = 대칭 int8, 비대칭이 일반적이라 narrow). `k_dequant_bias*.cl`, `*_dpas.cl`: 0x6480 xor bias trick과 DPAS 연결 변형.
- 온디바이스 layout probe: `./run_probe.sh {verify_k_transform, probe_v_layouts, probe_v_multiblock, verify_micro_kq_dpas}` — 각 lane/byte가 어느 (key, head)를 가지는지 GPU 실행으로 확인 (K 8b-transform read -> lane==head 이고 shuffle 불필요; V transform은 이미 VNNI 정렬). `probe_dpas_api.sh`: bmg에 존재하는 DPAS/2D-block builtin 확인 (int8 있음, mixed-precision `hf_i8`/`bf_i8`/`u8_hf` 없음 -> 중간 widen 필수).
- 한계: `bmg` 하드코딩(B580/B70), `dump/`는 재생성 가능한 산출물.
- `test/probe_kvup_run.cpp`: 실제 kvup 커널 GPU 실행 + differential oracle. `--bench-iters >= 500`이면 +-1 ns 재현. **큰 비율(12%)은 실모델과 1 pp 일치(+13.2% vs +12.1%)하나 작은 비율은 못 맞춘다**: generate에서 probe는 -4.4%를 예측했으나 실모델은 parity(6080 vs 6080) — probe가 실제보다 ~2.2x 절대시간을 낮게 보고. 작은 튜닝은 실모델로 판정. `test/kvup_negctl.py`: 15개 negative control (앵커 occurrence count 단언).

---

## 4.13 DG2 / GPU hang 진단 메모

- DG2 f32 PA 테스트의 `CL_OUT_OF_RESOURCES` 중단은 dmesg `i915 ... GPU HANG: ecode ... context reset due to GPU hang`와 PID가 일치하는 **GPU hang**으로 확인 (2026-09-28). NEO가 hang-reset context를 `CL_OUT_OF_RESOURCES`로 보고. 문제 커널은 `pa_kv_cache_update_ref` vs `sdpa_opt__multi_tokens` 경로이고 sdpa_ocl 경로가 아님 (`sdpa-ocl-known-issues-repro`). 즉 `clFinish` 에러는 hang의 증상이므로 `dmesg`를 같이 본다.
- head 486/387 `CL_OUT_OF_RESOURCES` + core dump는 전체 sweep(~41 KB WG SLM 최대)에서만 재현, 단독 재현 안 됨 (resource pressure로 추정, 규명 안 됨). 처음 `DKS_ACTIVE` 탓으로 돌렸던 것은 `=0` 대조만 있고 매칭되는 negative control이 없던 오류 (→ 05-methodology-and-pitfalls.md).
- DG2(xe_hpg) 오프라인 점검: `OV_GPU_ARCH_OVERRIDE=xe_hpg` 덤프 + `ab.py hpg --device dg2 --grf256`; 대표 plain 구성의 spill h64 7,968 B, h128 12,128 B, h256 28,064 B는 **ocloc 컴파일러 보고값**이며 DG2 런타임 값이나 시간이 아님. 큰 정적 spill은 "조사 후보"이지 회귀 확정이 아니다 (`sdpa-ocl-xe-hpg-s7-performance`).
- DG2 소형 프로토타입: 128 GRF에서 모든 측정 타일이 3.8-11 KB spill; IGC가 손으로 쓴 많은 레지스터 커널도 0 spill로 최적화하는 경우가 있어 P7은 T3 커널을 128 vs 256에서 비교하도록 설계 (`S2_RESULTS.md`).
- DG2 SG8에서 SIMD16 `short8`은 오류 없이 DPAS를 버린다 (`sdpa-ocl-xe-hpg-facts`). `dpas` 컬럼이 0이 아닌지 hpg TSV에서 항상 확인.

---

## 4.14 Spill triage playbook (단계별)

1. **관측**: cliloader `-d -dv`로 대상 커널 이름줄 확보. `SPILL=`, `TPM=`, `REG`, `SLM=`, GWS/LWS, device time. (device는 `--device_suffix=1`/`-d GPU.1`로 B70 지정.)
2. **path 확인**: 커널이 실제로 뜨는지, config가 서로 다르게 적용됐는지(SLM/GWS/LWS 비교) 먼저 확인. (stateful 경로에서는 PA 커널이 안 뜸; setupvars가 인자를 지움.)
3. **spill 종류 분리**: `TPM`>0이고 spill=0 -> 런타임 인덱스 private 배열(§4.9-A). `SPILL>0` -> register pressure. 둘 다 .asm 헤더로 교차 확인 (`//.private memory size`, `//.spill size`; `grep spill`은 무의미).
4. **ISA 확보** (리빌드 없음): 런타임 소스 덤프 + `OV_GPU_MAX_KERNELS_PER_BATCH=1`, ocloc splice. 먼저 **재현 증명**: ocloc 결과가 runtime의 spill/instCount/dpas와 일치하는지 (아니면 §4.6.3 한계).
5. **루프 body 격리**: backward branch(label이 작은 줄)로 k0 loop body 경계 확정, dpas / block2d / `load.ugm` 종류(SIMD-1, d8u32, d32xKt) / wide mov / sync / `r[a0]` 집계 (§4.4).
6. **원인 가설 -> 레버**:
   - SIMD-1 gather 다수, 독립 로드 N개가 동시 live -> 로드 geometry(block read, lane-per-key + broadcast) (§4.9-B/C)
   - 런타임 인덱스 배열 -> select chain / broadcast (§4.9-A)
   - lane-varying 루프 변수, indirect addressing -> uniform bound (§4.10-1)
   - 루프가 풀렸다/안 풀렸다 -> guard를 호출 지점으로 (§4.10-2)
   - live set이 128 GRF 초과 -> 타일 축소(M, tile) 또는 256 GRF + 타일 재조정; thread 수를 먼저 지키는 쪽을 우선 시도 (§4.5)
7. **ISA A/B**: 의도하지 않은 모드는 instCount + opcode 히스토그램이 **동일**해야 함 (`isa_ab_*.sh`, `sdpa_ocl_ab.py l2`). C등급이면 `keep_dir`로 IR 단계 diff (§4.6.4).
8. **device 측정으로 판정**: env toggle(리빌드 없이 한 binary 안에서 A/B; 8회 min/median; prefill은 invocation당 2회라 avg 대신 min), cliloader 커널별 time. ISA가 좋아졌다고 빨라졌다고 주장하지 않는다. 같은 shape에서 production baseline(절대값) 대비로 판단.
9. **ablation**으로 지표 확인: 의심 phase를 bound=0으로 끄고(`if(0)` 금지) 비용 분해. 정적 개수가 같은데 시간이 다르면 sector/캐시 패턴 의심.
10. **negative control**: 같은 변경을 *되돌렸을 때* 결과가 되돌아오는지, 매칭되는 control 없이는 귀인하지 않는다.
11. **정확성 게이트**: NaN/Inf, accumulator 타입(f32), precision 변환 순서, dynamic shape(shape_info 인자) 영향을 점검. 타일 override는 *조용히 틀린 출력*을 낼 수 있다(§01의 4 invariants; 크래시도 spill도 없이 accuracy만 실패).
12. **임베딩 확인**: 빌드 후 `sdpa_ocl_ab.py built`로 .inc/.so/ov_gpu_unit_tests에 의도한 소스가 들어갔는지.

(빌드/실행은 사용자 담당. 위 명령은 에이전트가 실행하지 않고 블록으로 제시한다.)

---

## 4.15 notes vs 코드 불일치 / 미확정 항목

| 항목 | 상태 |
|---|---|
| VTune 사용 | B70 GPU Hotspots 수집과 후속 A/B가 확인됨 (§4.1.1). 다만 한 수집은 stall PC가 없어 소스 행 단위 귀속 불가 |
| `iga64` 미설치 (gpu-kernel-isa-dump) | stale. 현재 `/usr/bin/iga64` 존재 |
| `docs`/노트의 "head-64 note at `sdpa_gen_ocl.cpp:157-164`" | 줄 번호가 이동함. 현재 256GRF 설명은 `sdpa_gen_ocl.cpp:1038-1041`, 타일 주석은 `:195-196` |
| `SDPA_OCL_DKS_ACTIVE`, `SDPA_OCL_MAX_BARRIER_V_PREFETCH`, `SDPA_OCL_PA_CUR_*` | 리팩터링(2026-09)에서 제거. 노트의 해당 토글은 역사 기록 |
| ocloc spill 예측 불일치 | 원인 미규명 (§4.11-8) |
| `test/dump_isa.sh`가 `ROOT=/home/gta/work/openvino-eddy` 사용 | 옛 경로 하드코딩. 현재 repo는 `/home/shingyuk/work/openvino-eddy` (`dump_isa_h128.sh`는 갱신됨) |
| TPM 의미 | 노트에서 `TPM=N`은 cliloader의 thread private memory 크기로 쓰이며 `//.private memory size N`과 같다고 기록됨. cliloader 소스로 직접 확인하지는 않음 |
