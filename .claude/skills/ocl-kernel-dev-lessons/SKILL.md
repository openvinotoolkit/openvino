---
name: ocl-kernel-dev-lessons
description: Cross-kernel lessons from closing the sdpa_ocl vs sdpa_micro gap (24x to within 3%) on Intel Xe/Xe-HPG - bottleneck decision tree, effect-size ladder, design principles, correctness discipline, fixed-policy gating, tool/environment traps, and a start-a-new-kernel checklist. Use when starting or tuning any GPU OpenCL/DPAS kernel, planning a perf investigation, or reviewing a performance claim.
---

# 새 GPU 커널 개발/최적화 일반 교훈

상세와 근거: `src/plugins/intel_gpu/docs/ocl_perf_guide/09-general-lessons-for-new-kernels.md`, 사례 06장, 도구 07장, asm 08장.

## 진단 순서 (큰 격차일 때)
1. **비교가 같은 일을 하는가** — 커널 이름/geometry, `usm_device`, final-linked 기준 binary, tier fallback(opt) 여부. (24배 격차의 첫 조각이 usm_host였다.)
2. **spill/scratch** — SPILL=, XMX active. 거대 unroll + 추가 accumulator(`MICRO_MATH=1`)가 흔한 원인. unroll 제한만으로 126→51 ms.
3. **memory message 형태** — ISA에서 lane gather(16 B)/word-mov pack vs block read. 정렬/pitch 게이트 조건을 분리해 block IO 복구(head-72 9.28×, h128 V 51→28 ms).
4. **control flow** — full-tile guard로 dynamic goto/join 제거(28→19 ms).
5. **live range와 GRF/occupancy** — 256 GRF는 occupancy 반감. spill 0만으로 채택 금지.
6. 그 다음에야 tile/prefetch/SLM 튜닝. (정석 튜닝은 마지막 10% 안쪽이었다.)
7. 짧은 길이만 느리면 wave/prologue/pre-pass 고정비 → 길이별 tiny geometry.
8. IGC가 load를 sink해 latency가 노출되면 native(vISA) — `xe-visa-inline-asm`.

## 설계 원칙
- producer/consumer의 tile bound를 한 값으로(`min(WGK, causal_k-k0)`); tail/partial을 별도 테스트.
- SLM 소유권이 바뀌는 곳엔 WG barrier(alias race).
- 분기를 줄이는 것과 tail 코드를 없애는 것은 다르다: clamped whole-SV(항상 전체 실행+mask)가 더 빠를 수 있으나, KQ skip은 일부 lowering에서 NaN.
- 작은 shape는 별도 설정(Q8/K64/128 GRF). 고정 tile 하나 대신 `(head, Qheads, KVheads, max_len)` 선택 정책, 다중 sequence는 **최대 길이**로 선택.
- head-count와 head 크기는 독립 축 — 한 shape의 개선이 다른 shape의 결함을 가림.
- pre-pass(외부 layout 재배열)는 진단용; 대체가 목표면 kernel 안에서 해결(고정비 44–110 µs).
- HW 사실부터 측정(DPAS 1 in-flight, L1 prefetch 손해, SLM 47.7 KB → occupancy 반감).

## 정확도
timing 전에 정확도. NaN poison + CPU double **all-rows**(샘플 행은 전수가 아님), tolerance 고정. 수학 불변 변경은 이전 변형과 bit-identical. sharp 입력(부드러운 N(0,0.1)은 Q/K·softmax 오류를 숨김). LCG power-of-two degenerate data 확인. 경계 스캔(1,16,31–33,63–65,127–129,255–257,511–513,1025,1033,2048,4096,8192; 홀수 depth; padding NaN). **한 길이 PASS로 후보 부활 금지**(seq256 PASS/257 NaN 사례).

## 실험 관리
1. fixed policy와 소스 SHA256을 **측정 전에 동결**, 첫 FAIL에서 중단 — cherry-pick 금지.
2. 한 번에 한 가지, 양방향 control, no-op/branch-only control(최종 ISA SHA256 동일 확인).
3. 독립 repeat(seed, reverse order, cadence 둘 다: queued32 / wait_each gap200). 작은 개선(≤2%)은 repeat 전 채택 금지.
4. 반올림으로 임계값 통과 금지. arm median과 same-round paired ratio를 둘 다 보고.
5. 실패·제외 run과 이유, 재시도 금지 목록(dead ends)을 HANDOFF에 남긴다. 겹침 사고도 기록하고 clean repeat.
6. 증거 층을 구분: standalone 하네스 PASS ≠ 제품 dispatch ≠ 제품 gtest ≠ e2e ≠ 회귀 gate(S6/SG16/B70). coverage 구멍(검증한 head/count/padding/feature)을 PASS 표 옆에 적는다.
7. 지표(spill, occupancy, XMX active, 대역폭, 명령 수, IPC)는 단서; 판정은 matched device-time A/B.
8. GPU 작업은 직렬. 계측(phase timer/GTPin)은 acceptance 불가.

## 새 커널 시작 체크리스트
호출 정의(연산/dtype/shape/layout/mask/device/driver/cadence) → 기준 binary 확정 → 같은 일을 하는지 확인 → 길이/shape 스캔 표 → runtime 근거(device time+geometry, SPILL/TPM, GRF/SLM) → 한 가설·예측 → 한 변경·control·정확도→성능 → 게이트 확대(short/long/경계/head-count/cadence) → 결과·실패·구멍·command 기록 → 제품 통합은 별도 단계(census, gtest, 타 아키텍처 byte-identity, e2e).
