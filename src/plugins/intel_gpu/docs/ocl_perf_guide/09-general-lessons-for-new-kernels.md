# 09. 새 GPU 커널 개발·최적화를 위한 일반 교훈

sdpa_ocl 개발(Xe2/B70 제품 경로, Xe-HPG/DG2 standalone 게이트)에서 얻은 교훈 중 **attention에 한정되지 않는 것**만 모았다. 앞 장들이 "무엇을 했는가"라면 이 장은 "다음 커널(다른 연산, 다른 아키텍처)에서 무엇을 먼저 할 것인가"다. 근거 장은 괄호로 표시한다. 모든 정량 값은 해당 장의 하드웨어/MEASURED 조건을 따른다.

## 9.1 병목 진단 결정 트리

증상 → 먼저 확인할 것 → 가능한 원인 → 확정 방법. "확정 방법"이 없는 원인은 가설이다.

| 증상 | 먼저 확인 | 가능한 원인 | 확정 방법 (A/B) |
|---|---|---|---|
| micro/기준 대비 **수 배~수십 배** 느림 | 실제로 같은 일을 하는가(커널 이름, 입력 메모리 위치, 기준 binary) | 비교 오류(usm_host, 잘못된 binary), tier 거부로 opt fallback | cliloader 이름·geometry, `usm_device`, final-linked binary (07장 §7.1) |
| 큰 격차 + XMX active ≈ 2%, 큰 spill | spill 크기(SPILL=), XMX active | register pressure + 거대한 unroll, 추가 accumulator | unroll 제한 → spill/scratch 변화와 시간. `MICRO_MATH`처럼 live accumulator를 늘리는 옵션 끄기 (06장) |
| 격차 + XMX 낮음 + spill 작음 | ISA의 load 메시지 종류/개수 | lane gather(16 B), word-mov 패킹, 정렬 때문에 block IO 비활성 | gate 조건 분리(legality), block read 대체 (head-72: 9.28×, 02장) |
| `dpas` 사이에 `sync.allwr`가 매 step | ISA 순서 | IGC load sink → latency 노출 | load 위치/메시지 grouping 변경 후 ISA 비교 (native 필요 가능, 08장) |
| 긴 길이는 이기고 **짧은 길이만 느림** | 길이별 tile padding ratio, WG 수, 고정비 | wave 부족, prologue/epilogue, pre-pass 고정비, 큰 tile의 낭비 | 길이별 `tiny` geometry(Q8/K64/128 GRF) 단독 A/B (06장 §6.3 "짧은 길이" 절) |
| queued는 이기고 **wait_each만 짐** (또는 반대) | cadence별 paired ratio, bimodality | clock/cache cold, 첫 K load latency | 같은 커널을 gap 0/200/1000 µs로 측정. 원인을 power로 단정하지 않는다 |
| 정확도가 **특정 길이/head에서만** 틀림 | partial tile, odd depth, tail band | producer/consumer bound 불일치, 미기록 SLM 읽기, clamp 오류 | 경계 길이 스캔 + all-rows + NaN poison |
| 성능 변화가 소스 의미와 무관하게 ±수십 % | ISA diff | IGC 스케줄/할당 변동, `inline` vs `always_inline` | ISA SHA256 비교로 "no-op 변경"을 control로 사용 (04장 §4.8) |
| 256 GRF로 spill은 사라졌는데 더 느림 | 그 커널의 지배 비용(예: gather) | occupancy 반감 + load 비용 증가 | GRF/tile/SG 수를 묶어 측정 (04장 §4.5) |

## 9.2 어디서 성과가 났는가 (효과의 크기 순서에 대한 감각)

DG2 PA prefill (h128, 32/8, seq4096, f16, causal): 126.7 ms → 4.25 ms (micro 4.52 ms). 각 단계는 별도 matched A/B이며 서로 다른 run의 값이다.

| 순서 | 변경 | 효과 | 분류 |
|---:|---|---|---|
| 1 | S*V loop unroll 제한 | 126.8 → 57.6 ms | register pressure/spill 제거 |
| 2 | KQ loop unroll 제한 | 57.6 → 51.0 ms | 동상 (spill 7,872 → 704 B) |
| 3 | V 경계를 full-tile guard 바깥에 두고 64 B row read | 51 → 28.2 ms | gather → block read |
| 4 | K full-tile guard | 28.2 → 19.0 ms | dynamic goto/join 85.8% 제거 |
| 5 | `MICRO_MATH=0` (두 번째 accumulator 제거) | 18.9 → 11.8 ms | live range |
| 6 | tile-major K'/V' (pre-pass) | 11.8 → 5.1 ms | DPAS operand를 256 B block read로 |
| 7 | 256 GRF + c40 tile (KQ 16×32, SV 32×16) | 5.1 → 4.25 ms | tile/GRF/occupancy |
| 8 | pre-pass 제거: raw K/V를 attention 안에서 읽고 native 패킹 (asm) | pre-pass 고정비 제거 | 구조 (08장) |
| 9 | head·길이별 geometry/정책 (tiny, key8, Q16, clamped SV, …) | 짧은 길이/작은 shape 격차 폐쇄 | 06장 §6.3 "짧은 길이" 절 |

**일반화:** 처음 두 자릿수 배율은 (a) 비교 오류 제거, (b) register pressure/spill, (c) memory message 형태(gather vs block)에서 나왔다. tile 튜닝이나 prefetch 같은 "정석 최적화"는 마지막 10% 안쪽이었다. 큰 격차에서는 정석 튜닝보다 **측정의 신뢰성 → spill → message 형태** 순으로 본다.

## 9.3 설계 원칙 (이 프로젝트에서 반복 확인된 것)

1. **message 형태가 먼저, tile이 나중.** 같은 bytes를 16 B 4개로 읽느냐 64 B 한 번으로 읽느냐가 tile 크기보다 컸다(head-72 9.28×, h128 V read 51→28 ms). legality 조건(정렬/pitch/width)을 정확히 분리해야 block IO를 되찾는다(02장).
2. **live range를 설계한다.** 큰 tile이 빨라도 accumulator가 두 벌이면 spill로 진다(`MICRO_MATH=1`의 A_tile1 64 GRF). 어떤 값이 어느 loop에 걸쳐 살아 있는지 그려 본다. 256 GRF는 occupancy 반감(4 vs 8 threads/EU)을 치르므로 "필요한 tile이 256을 요구하는가"를 먼저 본다.
3. **producer/consumer의 bound를 한 값으로.** SLM을 통해 넘기는 tile에서 producer가 쓴 폭과 consumer가 읽는 폭이 다르면 쓰이지 않은 SLM을 읽는다(key8: `min(WGK, causal_k-k0)`). tail과 partial tile은 별도 테스트한다.
4. **SLM ownership이 바뀌는 곳에는 WG barrier가 필요하다.** K와 P scratch를 alias했을 때, 소유가 바뀌는 시점의 모든 reader 완료 barrier가 없으면 race(maxabs ≈ 0.9). "같은 subgroup이 쓰고 읽으니 subgroup barrier로 충분"이라는 가정도 먼저 A/B로 확인했다.
5. **분기/skip은 제어 흐름이 코드 생성을 바꾼다.** tail iteration 수를 줄이는 것과 tail code path를 없애는 것은 다르다. clamped whole-SV처럼 "항상 전체를 실행하고 mask"가 더 빠를 수 있다(분기와 marshal 제거). 반대로 KQ early-exit은 일부 lowering에서 NaN을 냈다. 어느 쪽이든 동기화 근처에서는 uniform branch를 asm 안에 두는 것이 안전했다.
6. **작은 shape는 별도 커널 설정이다.** wave 수 부족·prologue·첫 load latency가 지배하므로 큰 tile(Q32/K128/256 GRF)을 줄여 Q8/K64/128 GRF 같은 "tiny" geometry가 필요했다. 짧은 길이에서 긴 길이 설정을 그대로 쓰면 micro보다 느려진다(h128 seq1/16/32/64에서 normal clamp 패배, 127에서는 tiny가 +6.05% 패배 — 경계에서 정책을 바꿔야 함)(06장 §6.3 "짧은 길이" 절).
7. **head/dtype/layout별로 최선 geometry가 다르다.** 고정 tile 하나를 고르지 않고 `(head, Qheads, KVheads, max_len)` 같은 명시적 선택 인자로 정책을 만든다. 다중 sequence는 **합산 길이가 아니라 최대 길이**로 선택한다.
8. **일반화는 head-size가 아니라 align/depth/active-group 조건으로 한다.** head80(5 depth, 홀수), head48(V tile 폭에 비정렬), head272(partial band)에서 각각 다른 방식으로 깨졌다. "padded value subgroup은 skip", "없는 upper band는 uniform branch로 skip" 같은 규칙이 일반 해법이었다.
9. **pre-pass로 푸는 것은 임시 수단이다.** 외부 layout 재배열(K'/V')은 긴 길이 진단을 빠르게 했지만 고정비(약 44–110 µs)가 짧은 길이에서 진다. 최종 목표가 "대체"이면 pre-pass와 hybrid fallback 없이 attention 안에서 해결해야 한다(사용자 결정).
10. **하드웨어 사실을 확인한 뒤 가정을 쓴다.** DPAS는 스레드당 1개 in-flight, L1 prefetch가 손해, dpasw ≈ dpas, SLM 47.7 KB는 occupancy 반감 같은 값은 설계 선택을 많이 줄였다. 새 아키텍처에서는 07장 §7.6 마이크로벤치를 먼저 돌린다.

## 9.4 정확도 검증 규율

- **timing 전에 정확도.** 틀린 커널도 빠르게 "벤치마크"된다(주소가 틀린 K'/V'). FAIL 변형의 시간은 폐기한다.
- **all-rows, NaN poison, tolerance 고정.** 샘플 행 검증(복원 추출)은 "fullrow"가 아니다. 출력 버퍼를 NaN으로 채우면 미기록 영역이 즉시 드러난다. tolerance를 성능을 위해 완화하지 않는다(`maxabs < 1e-2`).
- **bit-identical 대조.** 수학을 바꾸지 않은 변경은 이전 커널과 전체 출력이 같아야 한다. 같지 않으면 순서가 바뀐 것이므로 의도했는지 확인한다(key8은 reduction grouping이 달라 halfword 몇 개 차이 — 기록 후 허용).
- **sharp 입력 stress.** 부드러운 데이터(N(0,0.1))는 Q/K 오독과 softmax 오류를 숨긴다. logit 분포를 맞춘 sharp 입력(또는 gain)을 반드시 쓴다(03장, 05장).
- **degenerate data 확인.** LCG 범위/해상도가 2의 거듭제곱이면 모든 row가 같은 값이 되는 경우가 있다. 데이터가 의미 있는 분산을 가지는지 offline으로 먼저 확인한다.
- **경계 스캔.** 길이: 1, 16, 31–33, 63–65, 96/97, 127–129, 255–257, 511–513, 1024/1025/1033, 2048, 4096, 8192. head: 홀수 depth, 비정렬 band. 여러 subsequence와 padding(Q before/after, NaN physical pad)도.
- **한 길이의 PASS로 후보를 되살리지 않는다.** seq256만 통과하고 257에서 NaN인 후보가 실제로 있었다.

## 9.5 실험·결과 관리 규율

1. **fixed policy 선 동결, 후 측정.** 후보를 측정한 뒤 잘 나온 것만 고르면(cherry-pick) 게이트가 무의미해진다. 선택 정책과 소스 SHA256을 측정 전에 고정하고, 첫 FAIL에서 중단(`run_policy.py`)한다.
2. **한 번에 한 가지만 바꾼다.** 구조를 많이 결합한 후보의 시간은 총효과일 뿐 단일 원인 증거가 아니다. 양쪽 control이 있어야 귀속한다 ("한쪽 control로 귀속 금지", 05장). 소스만 바꾸고 compile flag를 놓치는 이중 변경, 이름만 같은 build를 비교하는 실수를 피한다.
3. **no-op/branch-only control.** `DIAG_TAIL_LIMIT=k`처럼 일을 줄이지 않고 분기만 넣은 control, 소스는 바꿨지만 기능은 끈 control(최종 ISA SHA256 동일 확인)을 둔다.
4. **repeat는 독립이어야 한다.** seed, arm 순서(reverse), cadence를 바꾼다. 작은 개선(≤2%)은 repeat로 확인하기 전에 채택하지 않는다.
5. **반올림으로 통과시키지 않는다.** 1.03 기준에서 1.0306은 FAIL이다. 한 번의 미세한 승리가 전체 게이트를 대신하지 않는다.
6. **실패를 기록한다.** 폐기한 후보는 *왜* 폐기했고 *무엇을 재시도하면 안 되는지*를 HANDOFF의 dead-end 목록에 남긴다(spill block variant, sweeps, KQ skip, wide SLM load 등). 근거가 새로 생기면 재시도한다.
7. **overlap 사고를 숨기지 않는다.** GPU 작업이 겹쳤을 가능성이 있으면 해당 측정을 제외하고 clean repeat로 닫으며, 사고도 기록한다.
8. **증거의 층을 구분해서 말한다.** (a) standalone 하네스 PASS (b) 제품 dispatch 확인 (c) 제품 정확도 gtest (d) e2e latency (e) 회귀 gate(S6/SG16/B70)는 서로 다른 증명이다. 하네스 PASS를 "제품 대체 완료"로 쓰지 않는다. coverage 구멍(checked한 head 수, head-count, padding, feature)은 PASS 표와 같은 문서에 적는다.
9. **지표는 단서이고 판정은 A/B다.** spill 바이트, occupancy, XMX active, bandwidth, 명령 수, IPC는 각각 독립 단서이며 그중 하나의 변화가 속도 변화의 원인이라는 결론은 matched device-time A/B가 지지해야 한다. 예: 명령 수가 작은 커널(920)이 큰 커널(1585, micro)보다 항상 빠르지 않았고 IPC가 0.22–0.24로 낮아 시간은 *latency/의존*에서 나왔다.

## 9.6 도구/환경 함정 모음

| 함정 | 결과 | 규칙 |
|---|---|---|
| dGPU 입력이 `usm_host` | PCIe로 읽어 커널이 느려짐 (PA prefill에서 약 38%: 151.8 vs 110 ms) | `usm_device`로 복사 (07장 §7.1) |
| micro 소스 wrapper 재컴파일 / 초기 clBuildProgram binary | nGEN GEMM 누락 → 정확도 FAIL, 시간 무효 | final-linked PREFILL binary, ELF 이름 확인 |
| GENERATE binary로 PREFILL 측정 | 인자 오류 `argument-52` | 하네스가 PREFILL 바이트 없으면 거부 |
| 2-GPU에서 `--device_suffix` 누락 | iGPU/sdpa_opt가 돌아 "OCL 측정"이 아님 | dGPU `--device_suffix=1`, dispatch 이름 확인 |
| `setupvars.sh`가 `"$@"`를 지움 | 스크립트 인자가 사라짐 | source 전에 인자를 변수에 저장 |
| Bash `cd`가 세션에 남음 | 상대 경로 스크립트가 엉뚱한 위치에서 실행 | 절대 경로, 저장소 루트 고정 |
| `ocloc compile`이 cwd에 `*.bin`/`*.spv` 작성 | 저장소에 stray 파일 | 임시 디렉터리에서 실행 |
| GTPin이 `-allow_sregs` 없이 exit 0 + 데이터 없음 | 수집 성공으로 오인 | 수집 뒤 event/PC 존재 확인 |
| VTune result dir 부모 없음 | 리디렉션이 GPU 시작 전에 실패 | `mkdir -p` |
| 실행 중인 스크립트를 편집 | bash가 오래된 바이트 offset에서 재개 | 실행 여부 확인 후 편집 |
| Debug 호스트 + DEBUG_CAPS | e2e/host overhead 값 신뢰 불가 | device kernel time만 쓰거나 Release로 재측정 |
| 계측(phase timer/GTPin)을 acceptance로 사용 | ISA가 바뀌어 시간이 달라짐 | 계측은 가설 생성용 |

## 9.7 다음 커널을 시작할 때의 체크리스트

1. 호출 하나를 정의한다: 연산, dtype, shape, layout, mask, device/driver, build, timing API, cadence. 기준 커널(binary)을 확정한다.
2. 비교가 같은 일을 하는지 확인한다(커널 이름, 메모리 위치, useful FLOP과 실제 key 수/padding).
3. runtime 근거를 먼저 모은다: device time + geometry, SPILL/TPM, SIMD/GRF/SLM. 길이/shape 스캔 표를 만든다.
4. 격차가 수 배면 §9.1을 위에서부터 따른다. 프로파일러는 한 질문에 연결해서 쓴다(07장 §7.0).
5. 반증 가능한 예측을 쓴다: "이 reader의 gather를 block read로 바꾸면 같은 bytes에서 시간이 내려간다".
6. 한 변경, 양방향 control, 정확도 → 성능. 변경 전후의 JIT 상수, final binary, ISA 시그니처가 의도대로 달라졌는지 확인한다.
7. 성능 게이트를 넓힌다: short/long, 경계, head-count, cadence 두 가지.
8. 결과·실패·coverage 구멍·재현 command를 남긴다(HANDOFF/CHECKPOINT). 메모와 해시를 함께.
9. 제품 통합은 별도 단계다: dispatch 확인(census), 정확도 gtest(sharp 포함), SG16/타 아키텍처 byte-identity, e2e.
10. asm/native가 필요하면 08장의 체크리스트와 통합 후 유지보수 리뷰를 계획에 넣는다.
