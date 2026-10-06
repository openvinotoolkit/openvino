# 06. 성능 격차 조사 사례: sdpa_ocl vs sdpa_micro

이 장은 Xe2/B70에서 제품 커널의 병목을 찾아 성능을 끌어올린 사례와, Xe-HPG/DG2에서 초기 24배 격차를 조사해 별도 pre-pass 없는 단일 attention 커널의 측정 게이트까지 맞춘 전체 과정을 기록한다. B70의 제품 경로와 DG2의 standalone 실험 경로는 서로 다르다. 숫자를 한 시계열로 이어 붙이거나 다른 GPU에 일반화하지 않는다.

도구별 명령과 해석 한계는 [07장 도구 쿡북](07-profiling-tool-cookbook.md), inline vISA는 [08장](08-inline-visa-asm.md), 다른 커널에 옮길 일반 교훈은 [09장](09-general-lessons-for-new-kernels.md)에 있다. 관련 설계·정확도 원리는 [01장 DPAS와 타일링](01-dpas-and-tiling.md), [02장 메모리 I/O](02-memory-io-prefetch-barriers.md), [03장 수치](03-numerics-softmax-quantization.md), [04장 ISA와 프로파일링](04-spill-isa-profiling.md), [05장 방법론](05-methodology-and-pitfalls.md)에 있다.

## 6.0 요약과 증거 경계

| 사례 | 초기 관측 | 확인된 결과 | 아직 증명하지 않은 것 |
|---|---|---|---|
| Xe2 Arc Pro B70 | 일부 head/경로에서 sdpa_ocl이 sdpa_micro보다 최대 8.11배 느림. 기본 f16 prefill 구성도 대표 케이스에서 13% 느림 | head-72 경로는 9.28배 빨라져 micro보다 1.14배 빨라짐. f16 prefill best 설정은 micro보다 7% 빠름. 기록된 u4 MIXED 두 workload는 micro보다 1.5%와 6.3% 빠름 | 모든 shape와 모델에서 같은 타일이 최선이라는 보장 |
| Xe-HPG Arc A770/DG2 | PA PREFILL head 128, seq 4096에서 초기 OCL 110–113 ms, micro 4.62 ms 수준 | vISA inline assembly를 쓰고 원본 K/V를 attention 안에서 읽고 패킹하는 standalone 커널 행렬이 승인된 3% 게이트를 통과: 32개 block-16 head 크기, 현재 정책/소스 해시와 일치하는 2,642 PASS 기록, 2,434 고유 조건, 최악 paired ratio +2.6234% | OpenVINO 제품 raw-kernel 통합, end-to-end 성능, 모든 head-count·padding·feature 조합, Xe2/B70 회귀 |

DG2 결과는 kernel-only 실험 게이트다. 제품에서 sdpa_micro를 대체하는 작업이 완료됐다는 뜻은 아니다. 제품 raw-kernel 통합, 실제 dispatch 확인, 제품 정확도 게이트, end-to-end 측정, S6 및 SG16/B70 회귀는 별도 단계로 남아 있다. 원자료는 `test/sdpa_ocl_xe_hpg/s7a/perf/`의 2026-10-03/04 기록에 있으며, 이 장에는 재사용할 핵심 근거와 한계를 자체적으로 요약했다.

현재 작업 트리에는 이전 `KV_TILED`/전처리 실험에서 남은 제품 코드와 PREFILL 전용 tier 변경도 있다. 이 코드는 standalone 게이트를 통과한 vISA raw-kernel 정책이 아니며 제품 빌드·정확도·성능으로 승인되지 않았다. committed HEAD의 tier 상태와 미커밋 작업 트리의 상태를 구분한다. 이 경로의 정리와 raw-kernel 제품 통합은 후속 작업이다.

## 6.1 성능 비교를 믿을 수 있게 만드는 첫 단계

큰 격차를 최적화하기 전에 측정 대상이 실제 비교 대상인지 확인한다.

1. **실제로 실행된 커널을 확인한다.** OV_VERBOSE=4 dispatch census 또는 cliloader 커널 이름으로 sdpa_ocl, sdpa_micro, sdpa_opt 중 어느 arm인지 확인한다. HPG tier gate가 거부한 op가 opt로 떨어졌다면 그 실행은 OCL 성능 측정이 아니다. 빌드 실패를 삼킨 경로도 dispatch census로 잡는다.
2. **GPU와 메모리 위치를 고정한다.** 2-GPU 시스템에서는 OpenVINO device suffix와 VTune의 GPU BDF를 각각 확인한다. A770 실험은 OpenVINO GPU.1 / --device_suffix=1, VTune target 0:3:0.0이었다. unit-test 입력은 dGPU에서 기본 usm_host가 될 수 있다. usm_host 입력의 151.8 ms 결과는 PCIe 경로가 섞여 폐기했고, 입력을 usm_device로 복사한 결과만 비교에 사용했다.
3. **기준 커널의 실제 native binary를 쓴다.** OpenCL 소스 wrapper나 초기 clBuildProgram dump만으로는 sdpa_micro의 fused nGEN GEMM이 빠질 수 있다. final-linked PREFILL binary와 커널 이름을 확인한다. 잘못된 binary는 정확도도 시간도 유효하지 않다.
4. **같은 일을 시킨다.** Q/K/V 값과 dtype, shape, cache layout, mask/causal, query/key 범위, subgroup/WG 배치, warm-up, enqueue cadence를 맞춘다. micro와 OCL이 실제 수행한 key 수 또는 padding 양이 다르면 FLOP 효율 비교 전에 그 차이를 계산한다.
5. **시간의 종류를 분리한다.** cliloader device kernel time, primitive pipeline time, end-to-end first-token latency는 서로 다른 지표다. 예를 들어 SDPA가 first-token의 10–15%라면 커널 7% 개선이 first-token에서 약 1%만 보일 수 있다. pre-pass가 있을 때는 attention 커널 시간과 pre-pass 포함 전체 시간을 별도로 측정한다.
6. **반복과 arm 순서를 고정한다.** 작은 호출 수에서는 평균 하나를 믿지 않는다. B70 비교는 2800 MHz 고정, 여러 pass와 낮은 분산, ABBA 또는 독립 재측정을 썼다. DG2 standalone gate는 같은 round의 OCL/micro paired ratio를 계산하고 queued32와 wait-each 두 cadence를 따로 확인했다. 임계값을 반올림해서 통과시키지 않는다.

cliloader -d -dv/-ko는 실행한 커널, launch geometry와 device time을 확인하는 데 쓴다. host-bound일 수 있으므로 작은 device-time 차이는 e2e 결과와 혼동하지 않는다. 실행 환경과 A/B 규칙은 05장 §5.3–5.5를 따른다.

## 6.2 B70: 하나의 성능 문제가 아니라 workload별 다른 원인

### VTune은 조사 방향을 좁혔고, 단독으로 원인을 확정하지 않았다

B70 GPU Hotspots 수집에서 sdpa_ocl__prefill은 occupancy 99.0%, XVE active 47.8%, stalled 52.2%였다. 후속 OCL/micro 비교에서는 OCL SBID stall 59.1% 대 micro 34.2%, barrier stall 10.6% 대 2.5%가 관측됐다. 이는 dependency/barrier 대기 가설을 만들 근거였지만, 해당 수집의 stall PC는 0개였다. source/DWARF mapping이 정상이어도 특정 소스 행이나 ISA 명령에 stall을 귀속할 수 없었다.

그 가설로 V-prefetch를 시험했다. 해당 B70 workload에서 V-prefetch 단독은 388.04 ms에서 381.54 ms로 줄었고 재검증 평균은 381.17 ms, 출력 MD5는 같았다. 32-row prefetch는 16-row와 차이가 없어 제거했다. 별도 16r16x2c V-read 병합은 524.34 ms에서 521.01 ms로 0.64% 개선했고 MD5가 같았지만 작은 단일-workload 이득이다. 이 사례와 VTune 환경/해석 한계는 04장 §4.1.1에 기록돼 있다.

**교훈:** GPU Hotspots의 active/stalled/occupancy는 다음 실험을 고르는 자료다. stall PC가 없거나 주소를 정확한 zebin에 대응시키지 못하면 ISA/source-level 원인으로 단정하지 않는다. 채택은 같은 입력의 device-time A/B와 correctness로 확인한다.

### head-72에서 SPILL보다 먼저 메모리 경로를 고쳤다

gemma-4 vision prefill의 f16 head-72는 OCL이 sdpa_micro보다 8.11배 느렸다. head 한 행은 144 B였고 pitch는 2304 B였지만, host gate가 width/pitch/base 정렬 조건을 row_bytes % 64 == 0 하나로 묶어 K/V 2D block IO를 껐다. 실제로 위반한 규칙은 base 정렬뿐이었다. 결과적으로 K/V가 subgroup당 key-loop 반복마다 256개 scalar gather로 떨어졌다.

수정은 두 단계였다.

1. 게이트를 %16 규칙으로 분리하고, base를 64 B 경계로 내린 뒤 x-coordinate와 width를 보정하는 base fixup을 추가했다. K/V block IO가 다시 켜져 6.96배 빨라졌다.
2. D_MAX 전체가 아니라 실제 head에 필요한 DKS_ACTIVE = ceil(head/DPAS_K)만 처리했다. head-72에서 깊이 타일이 8개에서 5개로 줄고 Q SLM도 감소했다. 추가 1.33배가 나왔다.

합계는 cliloader device time 9,570,971 ns에서 1,030,869 ns로 9.28배 개선됐다. sdpa_micro보다 1.14배 빨랐다. 세부 legality/padding 논증은 02장 §3.2에 있다.

이 사례는 세 가지 단순 지표 해석을 반박했다. 첫째, 256 GRF가 spill 2688 B를 없앴지만 이 workload에서는 9.57 ms에서 14.12 ms로 47% 느려졌다. 원인은 GRF를 두 배로 써 resident thread가 줄면서 scalar-load 경로 비용이 더 커진 것이었다. 둘째, spill은 block-read로 바뀐 뒤 부수적으로 사라졌으므로 root cause가 아니었다. 셋째, Q/A staging은 key-loop 바깥이라 이 사례에서 주 병목이 아니었다.

### 기본 f16 prefill은 spill과 tile을 한 쌍으로 조정했다

B70, llama-3.1-8b, q=4096 f16 prefill에서 sdpa_micro는 1,930,454 ns, OCL 기본 설정은 2,181,376 ns로 13% 느렸다. 256 GRF와 tq32/pwk4/pwq2 타일을 함께 쓴 설정은 1,796,200 ns로 micro보다 7% 빨랐다. first-token latency는 466.98 ms에서 462.85 ms로 바뀌었다.

256 GRF만 켜면 이기지 못한다. 큰 타일의 spill을 없애도록 GRF를 늘리면서, WG key/query 배치도 다시 맞춰야 했다. 반대로 gather-bound head-72 같은 kernel은 spill 0이 돼도 256 GRF에서 더 느려졌다. REG256 또는 spill bytes를 단독 목적함수로 삼지 않는다. B70 pinned-clock 조건과 나머지 구성 표는 05장 §5.5.3, 설계는 01장 §4–7에 있다.

다른 성공 사례도 같은 원칙을 보였다. gpt-oss u4 head-64 MIXED는 cache row가 64 B block2d 경로를 못 타고 scalar gather가 많아졌다. whole-page uc16 read와 타일 조정으로 3.92배 빨라져 micro보다 6.3% 빨랐다. 반면 causal key bound와 token-major K 재배치는 실제 work 또는 message geometry를 줄인 사례다. 이 결과들은 01/02/05장 표에 각각 기록돼 있다.

## 6.3 DG2: 24배 차이에서 raw K/V 단일 커널 게이트까지

### 처음 측정한 격차

대상은 PA PREFILL, f16, head 128, Q/KV heads 32/8, seq 4096, past 0, causal, uncompressed cache, 단일 subsequence였다. A770 DG2에서 TEST_USE_SDPA_OCL_HPG=1로 실제 SG8 OCL arm을 dispatch했다. 첫 통제 ABBA는 OCL 110.42–113.27 ms, sdpa_micro 약 4.616 ms로 약 24배 차이였다. 후속 재현은 OCL 126.74 ms 대 micro 4.543 ms, 27.90배였다. 이 값들은 서로 다른 측정 run이므로 한 세부 최적화의 연속 delta로 더하지 않는다.

초기 host library는 Debug였고 Release 재측정은 미완료였다. 다만 이 비교는 OpenCL device kernel time이었으며, 최종 acceptance는 별도 standalone native-kernel 하네스에서 했다. 첫 usm_host 실행 151.8 ms는 폐기하고 device-USM 입력을 썼다. 정확도 gate는 통과했으므로 격차는 correctness 실패가 아니라 성능 문제였다.

### 프로파일링으로 root cause를 좁히기

| 측정 | OCL 원본 | sdpa_micro | 읽을 수 있는 결론 |
|---|---:|---:|---|
| XVE active / stalled | 15.6% / 81.8% | 44.2% / 55.6% | OCL이 실행 자원을 충분히 활용하지 못했음 |
| XVE occupancy | 47.9% | 49.7–49.8% | occupancy만으로 24–28배 격차를 설명할 수 없음 |
| XMX active | 2.0% | 24.4–24.5% | OCL은 DPAS 처리량을 거의 활용하지 못함 |
| L3 read bandwidth | 246.86 GB/s | 783.61–857.93 GB/s | micro가 더 많은 데이터를 공급받았지만 이 값만으로 DRAM 포화나 단일 원인을 증명하지는 않음 |
| 런타임 spill | 7,872 B | 0 B | 원본 OCL에 큰 register-pressure/scratch 문제가 있었음 |

도구별 역할은 다음과 같았다.

| 도구 | 이 조사에서 한 일 | 해석 경계 |
|---|---|---|
| CLIntercept cliloader -d -ko/-dv | 실제 dispatch 이름과 per-enqueue device time, launch 정보 비교 | host-bound 실행, profiling overhead, 잘못된 device/memory placement가 섞이지 않게 별도 확인 |
| VTune 2026.4 GPU Hotspots | XVE active/stalled, occupancy, XMX, Send, L3 지표로 병목 후보를 좁힘 | 비율은 kernel-time의 비율이 아니다. bandwidth 하나로 대역폭 포화를 단정하지 않는다 |
| VTune source-analysis + GTPin | native PC 기준 memory latency, instruction count, basic-block latency 수집 | 계측 trace와 PC/native mapping이 정상인지 확인해야 한다. latency 합계를 wall-time으로 환산하지 않는다 |
| IGC ShaderDump + ocloc + iga64 | .asm/.zeinfo, scratch/spill, DPAS, global/SLM load, dynamic branch와 send를 확인 | 오프라인 compile은 runtime과 다를 수 있다. runtime options와 실제 loaded zebin으로 재현 여부를 확인한다 |
| __builtin_IB_read_cycle_counter phase probe | KQ, max/softmax, S*V 구간의 상대 비용 가설 생성 | 계측 코드가 ISA/scheduling을 바꾸므로 acceptance timing이나 정확한 latency 귀속에 쓰지 않는다 |
| component microbench | DPAS, L1/L3, SLM, ALU의 대략적인 장치 상한·메시지 비용 확인 | microbench 대역폭·처리량을 attention 전체 속도로 환산하지 않는다 |

VTune 2026.4 수집은 `source /opt/intel/oneapi/setvars.sh` 뒤 진행했고, GPU target은 BDF로 고정했다. A770은 OpenCL device 0 / OpenVINO `GPU.1` (`--device_suffix=1`), 당시 driver는 26.27.39122.14였다. GTPin 저spill 변형에서 SREG 허용이 없으면 exit 0이어도 “kernels not found”로 데이터가 없는 경우가 있었다. `AMPLXE_MORE_GTPIN_OPTIONS='-allow_sregs 1'` 후 실제 event와 PC가 생겼는지 확인했다. micro final fused native는 ELF function `st_size`가 실제 fused `.text` 길이보다 작아 GTPin source-analysis가 assertion을 냈다. profiling 전용 binary 복사본에서 symbol size만 고쳐 source-analysis를 재현했으며 native text와 제품 fuser는 바꾸지 않았다. 잘못된 result directory나 실패한 계측은 수치 집계에서 제외했다.

또 하나의 중요한 정정은 “K와 V가 모두 scalar gather”라는 초기 추측이다. 실행된 native ISA를 보면 해당 OCL의 K는 이미 32 B dword block read였고, V 쪽이 lane gather와 packing을 반복했다. 전체 workload에선 두 reader의 boundary/alignment 검사도 비용이 컸다. 어떤 소스 경로가 선택됐는지보다 실행된 ISA를 확인하는 것이 원인을 빠르게 바로잡았다.

### 한 가지만 바꾼 진단 A/B

실제 OpenVINO 빌드와 제품 코드를 건드리지 않고, runtime source-injection한 kernel을 ABBA로 비교했다. 아래 delta들은 같은 shape와 harness를 쓴 각 matched A/B다. source-injection 결과는 root cause를 확인한 진단 probe이며 제품 gate는 아니다.

| 변경 | OCL device time | 근거 |
|---|---:|---|
| 기준 | 126.74 ms | sdpa_micro 4.543 ms, runtime spill 7,872 B |
| S*V cp loop unroll을 1로 제한 | 126.79 → 57.57 ms | K/V read 수는 그대로, scratch fill과 spill/reload 감소 |
| KQ db loop도 unroll 제한 | 57.57 → 51.01 ms | 두 loop 변경 뒤 spill 704 B, scratch fill 약 104배 감소 |
| V 경계 조건을 full-tile guard로 밖에 두고 64 B row read | 50.97 → 28.20 ms | 16 B gather 여러 개와 packing 대신 block read; V send 수 4분의 1 |
| K full-tile guard 추가 | 28.18 → 19.00 ms | K/V send 수가 그대로인데 dynamic goto/join 약 85.8% 감소 |

이 결과는 “병목 후보 → 예측 → 단일 변경 → 양방향 A/B”의 예다. 앞선 진단에서 K를 잘못 의심했지만, ISA와 reader별 변경 결과가 K/V 역할을 분리했다. 조정 뒤에도 19 ms는 micro보다 약 4.2배 느렸으므로 이 시점에 문제를 해결했다고 기록하지 않았다.

### 실험 경로를 정리하고 pre-pass를 제거했다

후속 standalone 실험에서는 `MICRO_MATH=1`이 추가 O accumulator를 live로 만들어 spill을 늘린 문제를 껐다. 측정된 비교에서 probe 18.9 ms에서 11.8 ms로 줄며 spill은 1,376 B에서 0이 됐다. 임시 tile-major K'/V' 재배열 경로는 각 DPAS operand를 256 B block read로 바꾸어 kernel을 약 5.1 ms까지 낮췄다. 256GRF + c40 tile은 커널 기준 4.25 ms 대 micro 4.51 ms에 도달했다.

이 external pre-pass 경로는 4096-token 길이에서 유용한 조사 수단이었지만, pre-pass 고정 비용이 짧은 sequence에서 불리했다. h128의 seq 128–1024 구간은 attention kernel도 micro보다 10–30% 느렸고, pre-pass는 약 44–47 µs였다. 따라서 pre-pass나 짧은 길이의 micro fallback으로 목표를 덮지 않았다.

다음 설계는 원본 K/V를 attention kernel 안에서 읽고, register/native packing을 거쳐 DPAS에 공급하는 단일 raw kernel이었다. 즉 외부 K'/V' buffer, 별도 launch, micro fallback 없이 수행한다. 이 final candidate는 vISA inline assembly에서 virtual register shape와 operand mapping을 지정하고, 물리 GRF 배치는 IGC에 맡겼다. 이는 standalone gate의 구현 특성이다. OpenVINO 제품 커널에 이 asm을 통합하고 IGC/compiler version별 동작을 확인한 것은 아직 별도 작업이다.

Xe-HPG에서 여러 head·head-count·길이 경계에 최선이었던 geometry가 달라 고정 tile 하나를 고르지 않았다. 독립 실험 정책은 `(head, Qheads, KVheads, max_subsequence_length)`를 사용했다. PA에 길이가 다른 subsequence가 여럿 있을 때는 합산 token 수가 아니라 가장 긴 subsequence 길이를 선택 인자로 썼다. 이 정책은 standalone candidate selector이지 OpenVINO 제품 host policy가 아니다.

### 단계별 사다리: 어떤 증거가 어떤 다음 단계를 낳았나

아래는 matched A/B 결과를 시간 순으로 이은 것이다. 단계마다 shape·하네스·arm이 다르므로(제품 Debug OpenVINO → source-injection → standalone 하네스) 한 줄로 더하지 않는다. 사용한 도구는 07장 §7.0 표의 이름이다.

| # | 단계 | 결과 (A770, h128 32/8 seq4096 f16 causal) | 이 단계의 증거 → 다음 단계의 근거 |
|---:|---|---|---|
| 0 | 제품 sdpa_ocl vs micro (cliloader, usm_device, ABBA) | OCL 110–127 ms, micro 4.5–4.6 ms (24–28×), 정확도 PASS | 실행 커널 이름/geometry 확인, usm_host 폐기. 순수 성능 문제로 정의 |
| 1 | VTune overview + 런타임 spill | XVE active 15.6%/stall 81.8%, XMX 2%, spill 7,872 B | 실행 유닛·DPAS 미활용 + 큰 scratch → ISA 확인 |
| 2 | ISA(IGC dump) | scratch fill/reload 지배, V는 lane gather+pack, K는 이미 dword block read | "K/V 둘 다 gather" 가설 정정, unroll/reader를 분리해 A/B |
| 3 | source-injection ABBA: S*V unroll 1 → KQ unroll 제한 | 126.8 → 57.6 → 51.0 ms (spill 704 B) | spill/reload가 시간의 절반 이상 |
| 4 | V를 full-tile guard 바깥 + 64 B row read | 51.0 → 28.2 ms (V send 1/4) | gather → block read |
| 5 | K full-tile guard | 28.2 → 19.0 ms (goto/join −85.8%) | 같은 send 수에서 control-flow만으로 33% |
| 6 | standalone 하네스로 이전, `MICRO_MATH=0` | 18.9 → 11.8 ms, spill 1,376 → 0 B | live accumulator (A_tile1 64 GRF) |
| 7 | GTPin bb-latency + load ablation | V 로더 49%, K 로더 28% (BB cycle); ablation K 7 ms, V 6.8 ms | 약 70%가 operand load: key row마다 32 B load 16개 + word-mov VNNI pack(operand당 ~80 명령) |
| 8 | tile-major K'/V' pre-pass | 11.8 → 5.1 ms | operand당 256 B block read 한 번 |
| 9 | tile/GRF 스윕(128 GRF 24개, 256 GRF 63개) → c40 | 5.1 → 4.25 ms (micro 4.52) | 256 GRF + KQ 16×32 / SV 32×16. 이후 tile 스윕은 수확 체감 |
| 10 | 길이 스캔 | seq128/512/1024: c40 141/226/460 µs vs micro 117/179/411 µs, pre-pass 44–47 µs | **짧은 길이와 pre-pass 고정비가 새 문제**. 사용자: hybrid 금지, pre-pass 제거 |
| 11 | 짧은 길이 원인 가설 검증(아래) → raw K/V staging 시도 → native asm | 아래 표 | micro가 pre-pass 없이 이기는 이유를 ISA로 분석 |
| 12 | head·길이별 geometry 정책 + 경계 gate | 32개 head 크기, 2,642 PASS 기록 | 아래 절과 게이트 범위 |

### 짧은 길이에서 느린 원인을 하나씩 반증했다

c40 커널은 4096에서는 micro보다 빨랐지만 ≤1K에서 10–30% 느렸다(pre-pass 제외). 아래는 각 가설을 한 번에 하나씩 시험한 결과다.

| 가설 | 시험 | 결과 | 결론 |
|---|---|---|---|
| causal tail 계산량 | WG의 full-key padding 비율 계산: seq128/512/1024/2048/4096 = 1.60/1.176/1.091/1.046/1.023. KQ skip, SV cp 단축을 독립 A/B + branch-only control + source-off control(ISA SHA256 동일) | combined/micro paired +3.14/+6.83/+9.58/−4.65%(128/512/1024/4096), combined/base −3.1/−4.4/−3.2/+0.9%. SV tail −2.5%, KQ tail ≈ 0 | **tail이 격차 전체의 원인이라는 가설 반증.** 일부 이득만 있고 긴 길이는 오히려 악화 |
| 초기화/idle (wait-per-call) | 커널 불변, 하네스 enqueue만 wait-per-call ↔ 32 queued | queued32: 128 → 24/23 µs, 512 → 129/117, 1024 → 409/369, 4096 → 4215/4443 | wait-per-call의 큰 첫 K 지연은 정상 hot-loop latency가 아니다. 그래도 512/1024가 ~10% 열위로 남음 |
| 점유/wave 부족 | VTune overview | seq128 occupancy 21.7% vs 42.6%, idle 56.2% vs 14.6%. 512/1024는 동일(45.2/44.9, 47.3/47.5%) | 128의 wave 부족은 확인, 512/1024는 **다른 원인** |
| micro의 K/V 공급 방식 | final-linked ISA와 wrapper 소스 분석 | micro도 packing이 있다: KQ는 global-next K load → current SLM operand → current DPAS → next K dword 재배열 → SLM write/barrier, VS는 DPASW + `mov <2>:uw` pair 패킹. 별도 global K'/V'/launch 없음 | pre-pass 없는 구조가 가능하다는 증거. SLM single-buffer + barrier 재사용(`slmBuffers=1`)이며 "double-buffer"라고 쓰지 않는다 |
| DPASW/operand 공유 | opcode 교체 `kv_regular_dpasw` (DPAS vs DPASW+P half read) | 222.60 vs 222.63 µs (동일) | opcode 자체는 원인이 아니다 (마이크로벤치의 dpasw ≈ dpas와도 일치) |

### raw K/V를 attention 안으로 가져오려는 시도들 (seq512, h128 32/8, matched µs, 모두 정확도 PASS)

기준: c40(pre-pass K'/V') ≈ 131, micro ≈ 117. raw-only 후보는 별도 pre-pass 없이 원본 K/V만 쓴다.

| 후보 | 핵심 | 결과 | 평가 |
|---|---|---|---|
| `k_first` SG8×8 gather rows | raw K를 협동 load → SLM | 189.6 (pipelined 207) | 미채택 |
| `k_block` uniform block rows | 같은 구조 + block read | 545 (spill 7,808 B) | **spill block 변형 일괄 재시도 금지** |
| `k_sg16` / `k_sg16_block` | 16 SG | 228 / 295 | micro의 ~2.5배, 미채택 |
| `k_sg_sync` | WG barrier → subgroup barrier | 172.8 → 160.9 | barrier 감소만으로 불충분 |
| `k_direct` | gather + 8×8 butterfly transpose (SLM 없음) | 185 | 미채택 |
| `qrows_transpose` | KQ를 Q-rows/K-lanes로 | 159 | 구조 변경만으로 불충분 |
| `k_alias` | K와 P scratch를 alias | 최초 race (maxabs ≈ 0.9); WG barrier 추가 `k_alias_safe`만 유효 (157.7) | **SLM 소유 변경에는 WG barrier 필수** |
| `v_shared` / `v_direct` | raw token-pair → register VNNI pack | 174 / 206 | V pack 비용이 그대로 추가 |
| `independent` | raw K/V, query-lane 방향 | 452 | 16-key tile 온라인 softmax 반복으로 느림 |
| native vISA LSC+DPAS+pack 묶음 (`k16_nativepipe_rw`) | 08장 | 처음 오답 (uniform SLM 주소) → 수정 후 182–195 | 순서 통제의 가능성 확인 |
| raw 주소 8행 선계산 / whole-KQ 단일 asm | | 184.2 → 179.7 → 171.6 | 작은 이득, 모두 PASS |
| `kv_original_native` (원래 KQ 방향 + 레지스터 mov 재배열) | | 140.8 | pre-pass 없는 첫 140대 |
| `kv_sv_whole_addrs` (C에서 cp별 주소 선계산 + whole-SV) | | 131.3 (micro 119.3) | micro보다 ~10% 느림, 4096도 ~14% 열위 |
| **`kv_k_d64`: K d32x8 → d64x4 gather + uq transpose** | 같은 32 B/lane | 138.6 → 129.3 (micro 136.7) | **전환점.** native message grouping이 transpose lowering을 바꿈 |
| `kv_d64_partial_sv` + Q-dword load + output dword store + null-src0 | 작은 변경 누적 | 512에서 micro 대비 −7.8% | 각 변경 단독으로는 ≤5% |

이 표에서 얻은 규칙:
- **구조를 많이 결합한 후보가 느린 이유를 단일 원인으로 해석하지 않는다.** 시간 차이는 SLM staging, barrier, pack, live range가 섞인 결과다.
- raw 협동 staging은 대부분 (a) spill 폭증 (b) pack 비용 추가 (c) barrier 증가 중 하나로 졌다. 이를 이긴 것은 소스 구조 변경이 아니라 *native 메시지 grouping*(d64 gather)과 *tail 코드 생성 변경*(clamped whole-SV)이었다.
- 틀린 결과로 나온 "좋은 시간"(uq ADD, 잘못된 SLM 주소, 잘못된 K operand 매핑)은 모두 폐기했다.

### 짧은 길이: tiny geometry와 길이별 정책

h128 32/8에서 Q8/K64/4SG/128 GRF(`tiny64_safe`)로 seq ≤ 96을 처리하고 그 이상은 Q32/K128/8SG/256 GRF(`kv_d64_part_v64`)를 쓰는 raw-only 정책을 seed2, 32길이로 측정했다(queued32, warm6, 48 rounds, 같은 round paired ratio; 모두 CPU 정확도 PASS, spill 0).

| seq | raw µs | micro µs | paired | seq | raw µs | micro µs | paired |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.190 | 7.558 | −4.73% | 257 | 48.687 | 51.340 | −5.18% |
| 16 | 7.619 | 8.495 | −10.09% | 511 | 109.890 | 117.797 | −6.76% |
| 32 | 8.495 | 9.577 | −11.52% | 513 | 121.895 | 124.981 | −2.50% |
| 64 | 10.457 | 12.545 | −16.52% | 1024 | 359.530 | 369.440 | −2.63% |
| 65 | 13.182 | 13.834 | −4.48% | 2048 | 1225.703 | 1267.259 | −3.30% |
| 129 | 26.275 | 28.218 | −6.74% | 4096 | 4387.617 | 4441.423 | −1.22% |
| 255 | 45.017 | 46.166 | −2.49% | 8192 | 16592.823 | 16821.034 | −1.35% |

(이 표는 h128 32/8 하나의 queued32 결과다. wait_each와 다른 head/head-count에서는 별도 결함이 나타났다. 아래.)

짧은 길이에서 효과를 낸 요인:
- tiny 설정은 sweep이 아니라 **wave/prologue 격차에 대한 근거**(occupancy 21.7% vs 42.6%, 작은 WG 수)로 선택했다. 같은 tiny가 127에서는 +6.05% 손해여서 전 길이에 쓰지 못했고 길이 경계에서 정책을 바꿨다(h128: tiny ≤ 64, 이후 Q16 WG2/clamp).
- tiny에서 KQ skip은 NaN을 냈다(08장 §8.4-5). mask를 유지하고 raw load를 clamp한다.
- h48/h112/h272 등 각 head에서 geometry의 *깊이 수*(KQ depth 5/6/7/9/10), V 폭, padded value group이 달라 별도 소스(`tiny_h48_2_2_v16`, `tiny_h112_exact`, ...)가 필요했다.

### head-count·cadence 의존 잔여 결함과 해결 (작은 MHA/GQA)

처음 정책은 2/2 외 head-count에서 +2~5% 결함이 남았다(예: h128 2/2 seq129/257 gap200 +5.03/+5.02%). 한 번에 하나씩 시험한 요인:

| 시도 | 결과 | 채택 |
|---|---|---|
| V 주소 대수: 16개 token clamp·stride 곱 → 공통 column/first/last 주소 + 고정 offset의 min (`h128_q16wide_address`) | h128 2/2 seq256 gap: +4.06% → +1.70%, 15길이 × 2 cadence 30케이스 모두 3% 이내 | ✔ |
| causal-bounded raw 주소 clamp(`causal_k-1`) | K/V clamp 독립 A/B: +1.64/+2.04/+1.08/−0.08% (h128 129 gap). 주소 locality가 기여 요인 | 부분 |
| SGQ32 → SGQ16 + query SG 2 (`h64_16_16_q16wg2`, K128/WGQ32 유지) | h64 16/16 seq256 gap +10.57% → −13.91% | ✔ (≤512) |
| Q64B 협동 load, `Qwide` (h96) | h96 129/257 gap의 +0.3~0.9% → −1.0~−4.4% | ✔ |
| key8 KQ (SG8 keys / WGK128 + 인접 S 쌍을 16-key SV 레이아웃에) | h512 seq128 +7.6% → −12.3%, h144/160/256 1/1 경계 PASS | ✔ (bound 수정 포함) |
| WG 순서 변경(역순/heavy-first/interleave), L1 bypass, streaming cache, prefetch | 대부분 악화, 격차 미해결 | ✘ |
| Q8/K128 whole-SV, SGQ8×4 32 SG, K64/K96 (h128/96), 큰 Q tile | 컴파일/중복 라벨 또는 +5~29% | ✘ |
| `addc` 64-bit 주소, 첫 K load early reuse, native atomic marshal | <1% 또는 악화 | ✘ |
| phase 계측으로 병목 확정 | 계측이 ISA를 바꿔 acceptance로 불가 | 참고 |

핵심은 이 잔여 결함이 **기능 결함이 아니라 1–5%의 latency/scheduling 차이**였다는 점이다. 그래서 (a) 후보를 먼저 동결하고 (b) 15–20개 경계 길이 × 2 cadence 전체에서 *한 번도* 3%를 넘지 않는지 확인하는 게이트로 닫았다. 한 shape의 개선(h128)이 다른 shape(h64 16/16)의 새 결함을 가렸고, head-count 확장 때마다 새 blocker(head80 16/16 +4.1%, head256 1/1 seq128 +4.4% 등)가 나왔다. head-count와 head 크기를 독립 축으로 취급해야 한다.

### 최종 raw-kernel 게이트의 범위

2026-10-04 체크포인트의 PASS는 다음 비교 절차와 입력 범위에 해당한다.

- A770/DG2에서 device-USM 입력을 사용했다. 제품 micro가 실제로 생성하고 최종 링크한 PREFILL native binary를 대조군으로 썼다. wrapper나 초기 compile binary에 fused nGEN GEMM이 빠진 경우는 대조군으로 인정하지 않았다.
- kernel time은 OpenCL device event의 같은 round OCL/micro paired ratio로 비교했다. queued32/gap0과 batch1/wait-each/gap200 µs를 따로 측정했다. 각 suite는 warm-up과 반복을 명시하고, 경쟁 GPU job 없이 직렬로 실행했다.
- output을 NaN으로 poison한 뒤 모든 row/channel을 CPU double reference와 비교했다. 일반 입력 외에 Q/K 표준편차 `sqrt(1.28)`, V 표준편차 `0.1`, scale `1/sqrt(head)`인 sharp-softmax stress도 썼다. `maxabs < 1e-2`를 유지했고, 정확도 실패는 timing 전에 중단했다. 이 stress 입력은 제품 gtest fixture와 logit 분포를 맞춘 것이며 fp16 입력 byte까지 같지는 않으므로 제품 fixture에서의 정확도 게이트가 여전히 필요하다.
- 게이트는 f16 Q/K/V/output, `k_head_size == v_head_size`, causal PA PREFILL, past=0, uncompressed cache, page block 16에 한정했다. head 크기는 16부터 512까지 16의 배수 32종이다. 모든 head 크기가 모든 Q/KV head 수로 검증된 것은 아니다. 24개 head 크기는 주로 Q/KV=2/2에서만 검증했고, 추가 head-count 행렬은 일부 크기에서만 확인했다. 모든 padding, subsequence 배치, feature, cache 형식을 포괄하지 않는다.
- 게이트는 같은 입력과 cadence에서 `OCL <= micro × 1.03`이었다. 이 3%는 해당 standalone 실험의 승인 기준이지 모든 커널의 기본 허용치가 아니다. authoritative policy 및 원본 source hash와 일치하는 2,642 PASS 기록, 중복 제거 후 2,434 입력/호출 조건이 남았다. 가장 느린 통과 조건은 head=128, Q/KV=16/2, seq=65, queued32이며 10.8975/10.616 µs = 1.026233509, 즉 +2.6234%였다. 두 시간의 별도 median 비율이 아니라 같은 round의 paired ratio를 집계했다.

여러 경계에서 남은 문제는 source별로 풀었다. 아래는 전체 선택표가 아니라 결과를 바꾼 구조적 수정이다.

| 문제 | 확인한 수정과 교훈 |
|---|---|
| H=112, Q/KV=2/2, seq=1 | exact 7-depth KQ와 WGQ8/K64/128GRF tiny geometry가 긴 길이 후보보다 맞았다. CPU reference에 bit-identical한 뒤 짧은 길이 조건을 통과했다. 작은 MHA/GQA는 별도 경계다. |
| H=272 partial head | vector address 계산 변경은 상위 16-channel band를 틀리게 만들었다 (`maxabs=2.432`). Q-only 대조와 고정 +32-byte 상위 band load가 원인을 좁혔지만 정확한 IGC lowering 원인은 확정되지 않았다. 원래 vector pointer를 유지하고 없는 상위 band만 uniform native branch로 건너뛰는 후보가 통과했다. |
| H=144/H=160 경계, H=256 Q/KV=1/1 | KQ를 8-key subgroup로 줄인 후보에서 KQ/SV producer-consumer 폭이 달라지면 SV가 unwritten SLM을 읽어 NaN/Inf가 났다. `min(WGK, causal_k-k0)`로 S*V chunk를 제한한 뒤 독립 경계·sharp/multi 조건을 다시 검사했다. 한 길이 통과만으로 정책을 고정하지 않았다. |
| H=80 padded V subgroup | OpenCL outer-C guard가 seq=257에서 NaN/Inf를 냈다. uniform guard를 native asm 안에 둔 별도 후보가 통과했다. synchronization/SLM 주변 제어 흐름은 소스에서 uniform해 보이는 것만으로 안전하다고 판단하지 않는다. |

상세 candidate/source hash/제외 조건은 `test/sdpa_ocl_xe_hpg/s7a/perf/opt/CHECKPOINT_20261004_ASM_GATE3.md`, `test/sdpa_ocl_xe_hpg/s7a/perf/opt/HANDOFF_20261003.md`, `test/sdpa_ocl_xe_hpg/s7a/perf/opt/short_diag_20261003/raw_stage/RAW_RESULTS.md`에 있다. 그 파일의 전체 선택 정책은 당시 standalone 실험을 재현하는 기록이며 제품 지원 표가 아니다.

작업 트리에 남은 `KV_TILED` OpenCL pre-pass branch는 이 raw assembly gate와 다른 이전 prototype이다. tile-major K'/V'를 외부에서 쓰고 다시 읽는 경로는 긴 seq 조사에 유용했지만, pre-pass launch/write/read 고정비가 짧은 길이에 불리해 최종 3% gate에서 제외했다. 최신 체크포인트도 이 제품 prototype을 inactive/old path로 표시하고, 후속 통합 때 제거할 것을 명시한다. 현재 미커밋 `PA_PREFILL` tier 변경이 존재해도 제품 raw-kernel 완료 증거로 세지 않는다.

### 실패한 후보도 설계 규칙을 남겼다

| 관측 | 다음 커널에 적용할 규칙 |
|---|---|
| 256 GRF는 큰 accumulator를 수용했지만 resident thread를 줄이고, tile에 따라 128 GRF보다 느리거나 spill을 만들었다 | GRF 수, SLM, tile, subgroup 수를 묶어 측정한다. SPILL=0만으로 채택하지 않는다 |
| K/V를 SLM으로 바꾸거나 global prefetch를 넣는 변형 중 일부는 느려졌고, L1 prefetch는 여러 위치/거리에서 손해였다 | prefetch·SLM 이중 버퍼는 이유만으로 넣지 않는다. 실제 제품 micro의 staging 구조도 ISA에서 확인하고 가정을 피한다 |
| 큰 query/key tile, unroll, SLM transpose, KQ/SV fusion 후보가 정확도 오류 또는 성능 손해를 냈다 | 후보마다 전체 출력 검증을 timing 전에 한다. 한 번의 PASS와 좋은 sample은 채택 기준이 아니다 |
| key8 KQ를 도입한 일부 head에서 SV가 WG가 쓴 것보다 많은 cp를 읽어 unwritten SLM을 읽었다 | producer/consumer tile bound를 같은 값으로 전달한다. tail과 partial tile은 별도 테스트한다 |
| padded value subgroup 생략의 OpenCL outer-C 변형이 seq 257에서 NaN/Inf를 냈지만 uniform native guard 변형은 통과했다 | synchronization/SLM이 있는 구간에서 control-flow 변경은 컴파일러 재배치까지 포함해 재검증한다 |
| head-272의 inline address 계산/일부 vector payload 변경이 상위 채널 결과를 깨뜨렸다. 정확한 IGC payload-lowering 원인은 확정하지 못했다 | 성공 결과만 기록하지 말고 실패를 남기며, 원인이 불명확하면 그 변형을 일반화하지 않는다 |

## 6.4 다른 GPU kernel에 재사용할 성능 조사 절차

1. **성능 문제를 정확한 호출 하나로 정의한다.** 연산, 실제 dispatch, shape/layout/dtype, mask, data placement, device/driver, build, timing API, 실행 cadence를 고정한다.
2. **측정이 비교 가능한지 먼저 확인한다.** 올바른 native competitor, source/hash, 실행된 variant와 GPU, warm/cold 상태, host/device memory, 동일한 useful work를 확인한다. 문제가 실제 제품 경로인지 standalone probe인지 표시한다.
3. **낮은 비용으로 runtime 근거를 모은다.** device time과 geometry, SPILL/TPM, SIMD/GRF/SLM을 저장하고 call count와 min/median 또는 paired ratio를 보고한다. occupancy는 활용도 설명의 일부이지 속도 판정값이 아니다.
4. **프로파일러는 한 질문에 연결한다.** GPU Hotspots는 “어디서 stall/underuse가 보이는가”, ISA는 “실제로 어떤 memory/DPAS/control instruction이 있는가”, GTPin은 “정확한 native PC/basic block에서 어떤 event가 발생하는가”를 답한다. 질문과 도구 출력이 맞지 않으면 더 무거운 profile을 추가하지 않는다.
5. **가장 그럴듯한 원인 하나를 반증 가능하게 만든다.** “memory-bound” 대신 “이 reader의 lane gather를 64 B block read로 바꾸면 동일 K/V bytes에서 device time이 내려간다”처럼 예측을 기록한다. bit-identical/no-op arm 등 필요한 control도 둔다.
6. **A/B는 한 변화만 다르게 한다.** 소스만 바꿔 compile flags를 놓치는 식의 이중 변경을 피한다. before/after에서 실제 JIT config, final binary, ISA signature가 의도한 대로 바뀌었는지 확인한다.
7. **빠른 kernel은 정확도부터 확인한다.** 모든 output row/channel, NaN/Inf, tails, partial tiles, dynamic/padded layouts를 검사한다. 테스트 데이터가 Q/K 오류를 숨기지 않는지 확인한다.
8. **성능 게이트를 넓힌다.** 긴 대표 모델뿐 아니라 short/long, boundary, odd tail, 여러 head-count와 실행 cadence를 순차적으로 확인한다. 실험 harness, 제품 dispatch, end-to-end는 서로 다른 증명 단계로 보고한다.
9. **결과를 재현할 수 있게 남긴다.** measured/predicted, device/driver/clock, options/JIT constants, source/native hash, 정확한 비교 arm과 command, warm-up/반복, 실패·제외 run, coverage hole과 다음 조건을 기록한다.
10. **GPU 측정은 직렬화한다.** 같은 GPU에서 별도 benchmark, compiler dump, VTune/GTPin 수집을 겹치지 않는다. tag/result directory는 새로 만들고, 중첩 가능성이나 잘못된 arm이 의심되면 해당 run을 폐기하고 깨끗한 조건에서 다시 잰다.

## 6.5 성능 도구가 답하는 질문

| 도구 | 좋은 질문 | 그 도구만으로 말할 수 없는 것 |
|---|---|---|
| cliloader / CLIntercept | 실제 어떤 kernel이 몇 번 실행됐고 device event time/dispatch geometry는 얼마인가? | end-to-end latency 전체, source-level 원인 |
| VTune GPU Hotspots | GPU active/stalled/occupancy/XMX/메모리 metric이 어떤 후보를 가리키는가? | stall PC가 없을 때 특정 소스 행/명령, stall 비율을 그대로 wall time으로 변환 |
| VTune source-analysis + GTPin | 정확한 native PC/BB에서 load, latency, instruction event가 어디에 집중되는가? | mapping/trace가 불완전하거나 instrumented scheduling이 달라졌을 때 제품 native의 정확한 latency |
| IGC ShaderDump / ocloc / iga64 | compiler가 만든 ISA의 memory message, DPAS, spill/scratch, control flow는 무엇인가? | 실행 빈도, cache hit, latency hiding, device time. offline ISA는 runtime build와 먼저 대조 |
| component microbench | DPAS/load/SLM/ALU의 해당 장치상 대략적인 ceiling과 message 비용은 무엇인가? | 전체 kernel이 같은 병목으로 제한되는지 |
| phase cycle counter | 한 diagnostic build에서 KQ/softmax/SV 등의 상대 비용은 어떻게 보이는가? | 계측이 ISA와 scheduling을 바꾼 뒤의 uninstrumented acceptance timing |

각 도구의 정확한 명령, 환경 변수, 수집 모드, 실패 사례는 07장에 있다. 모든 시간 결과는 correctness와 실제 dispatch 확인 뒤 해석한다. spill byte, instruction count, bandwidth, occupancy는 독립적인 단서다. 이 중 하나의 변화가 속도 변화의 원인이라는 결론은 matched device-time A/B가 지지해야 한다.
