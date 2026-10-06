---
name: ocl-kernel-ab-methodology
description: Playbook for proving and measuring changes to high-performance OpenCL/DPAS GPU kernels (sdpa_ocl, paged attention) in OpenVINO intel_gpu - offline A/B (L0/L1/L2/pset/xab/built), attribution with both controls, oracle design, benchmark discipline. Use when A/B testing, bisecting a kernel failure, claiming a speedup, or reviewing kernel results.
---

# OCL 커널 A/B·귀속·벤치마크 방법론 (playbook)

깊은 설명과 표: `src/plugins/intel_gpu/docs/ocl_perf_guide/05-methodology-and-pitfalls.md` (이하 "05장"). 커널 기법은 01~04장.
저장소 규칙(`AGENTS.md`): 수치 정확성 우선, 측정 없는 성능 주장 금지, 방법론 명시, measured vs assumed 구분, 범위 확장/기회적 정리 금지.
비교 arm 오류와 unexplained gap의 조사·도구 선택은 [`ocl-kernel-performance-investigation`](../ocl-kernel-performance-investigation/SKILL.md) 및 [06장](../../../src/plugins/intel_gpu/docs/ocl_perf_guide/06-performance-gap-investigation.md)을 참조한다.

## 0. 먼저 지킬 작업 규칙
- **빌드, gtest, ab(전체 corpus), 덤프, 벤치, `-fsyntax-only`, `built`, `l0`은 사용자가 실행한다.** 에이전트는 정확한 명령 블록 + 예측값을 제시한다 (직접 실행 허락이 명시된 경우만 예외, 그것도 세션마다 다시 확인).
- **커밋은 사용자.** 한 줄 메시지만 제안한다. 근거/이력은 코드 주석과 `docs/sdpa_ocl.md`에.
- 코드 수정 전에 계획 승인. 범위 밖 발견은 고치지 말고 `docs/sdpa_ocl.md` "Known issues"에 기록.
- 사용자가 실행 중일 수 있는 bash 스크립트를 편집하기 전 `ps -eo pid,etime,args | grep <script>` (길이가 바뀌면 bash가 파일 꼬리를 명령으로 실행). 부득이하면 바이트 길이 유지.
- 긴 작업은 세션 말미에 메모리 갱신 + 한국어 재개 프롬프트. 폴링은 `sleep 240` 단위.

## 1. 변경 종류 -> 증명 방법 선택
| 변경 | 증명 |
|---|---|
| `.cl` 동작 보존 (리팩터/헬퍼 추출) | `sdpa_ocl_ab.py l0`, `ab --levels l1,l1p,l2 --l2-from-l1` 전 corpus A=전부. 헬퍼는 **`always_inline` 필수** (plain `inline`은 커널 전체 ISA를 바꿈) |
| `.cl` 버그 수정 | `ab`에서 **대상 config만** A를 벗어나야 함 |
| 호스트-only | 새 `dump` -> `corpus` -> `pset <기준> <새것>` (+`--l1`) + `gtests.sh diff`; 의도한 전이만 |
| 동작/성능 변경 | A/B가 다른 게 정상 -> **측정**하되, 건드리지 않은 config가 그대로임을 증명 |
| 테스트 추가 | 수정 **전** 코드에서 음성 대조군이 예측대로 FAIL |
- 도구/명령: 05장 5.2.3. 하네스를 고쳤으면 `selftest` 재실행 (음성 대조군 5/5).
- **base는 항상 명시 SHA** (HEAD 금지: 커밋하면 움직임). 스냅샷 `--force` 금지. `corpus <dir>`은 `<dir>/configs`를 지우므로 기준 코퍼스(r0, r1, r3, rT*)를 다른 base로 재생성하지 말 것.
- ISA 등급: A 동일 / B heavy op 동일+scratch 증가 없음+inst ±0.5% (성능 측정 필요) / C 그 외.
  C 해석: `sync` 개수 ±는 heavy로 분류됨 -> send/dpas/math/load/store/CALL이 안 움직였고 barrier 추가 없으면 해제 가능. `disable_mid_thread_preemption` 변화는 IGC ~600 inst 임계지 기능 변화가 아님.
- `grep spill`은 무의미 (`-abortOnSpill`이 모든 .asm에 있음). 진짜 신호는 zeinfo `*scratch*/*spill*`, `numGRF`.
- 빌드 후 `python3 test/sdpa_ocl_ab.py built --base <sha|snapshot>`로 `.inc` + `.so` + `ov_gpu_unit_tests`에 내 커널이 들어갔는지 확인 (링크 끝난 뒤에만; unit test는 정적 링크라 별도 사본).

## 2. 귀속(attribution) 규칙 -- 위반 시 실제로 정상 코드를 고칠 뻔함
1. **A/B는 정확히 한 가지만 다르다.** 한쪽 대조는 가설이지 결론이 아니다. "단독 실행+토글" vs "sweep+토글 없음"은 두 가지가 다르다 (head-486 OOM 사고: 원인은 토글이 아니라 suite 메모리 압력).
2. **토글을 on/off 양쪽으로** 돌린다. 긴 실행에서만 나는 문제면 sweep vs sweep.
3. **이질적 실패 집합 = 증거**: 커널과 무관한 테스트가 섞여 있으면 구성 간 실패 **집합**을 diff한 뒤 이론을 세운다.
4. **통제가 진짜 통제인지 확인**: 예) 압축 BY_CHANNEL에서 `TEST_USE_SDPA_OCL(_DECODE)=0`은 레이아웃 버그가 있던 때 통제가 아니었다. 증명 가능한 no-op 토글(ocloc으로 바이트 동일 확인)을 선호.
5. **소비자 전에 생산자**: 잘못된 출력의 첫 bisection 단계는 다른 backend(`TEST_USE_SDPA_OCL=0`)로 같은 입력을 돌려 보는 것. T5: sdpa_ocl을 용의자로 몰았지만 원인은 KV-cache 양자화 커널(`dynamic_quantize` 256 제한).
6. **메트릭이 움직인 것 != 메트릭이 중요한 것**: spill 2688->0이었는데 47% 느려짐 (LSC 메시지 수가 진짜 비용). occupancy%는 속도와 역상관이었음. 정적 모델(ISA, SLM->occupancy)은 장비 측정으로 검증 (SLM-occupancy 가설은 반증됨).
7. **예측과 틀리면 기록하라**: T7에서 "base가 64 B 정렬 아니면 틀림"은 틀린 예측이었고 음성 대조군을 수정 전에 돌려 발견. 정적 모델 예측은 PREDICTED로 표기.
8. 위 한 줄 판정 전에 반드시: 같은 바이너리인가, `--device_suffix=1`인가, env가 실제로 반영됐나 (소스 덤프 또는 `SDPA_OCL_TRACE_CONFIG=1`).

## 3. 오라클/검증 설계
- **연쇄 네트워크는 첫 레이어에서만 입력이 동일**하다. 레이어 평균은 전파 오차(포화값은 정상/버그 커널이 동일). L0 오차를 **출력 dtype ULP** (f16 ~4.9e-4 상대)와 비교. 자기검증: causal에서 `[0, tile)` 토큰이 bit-identical이어야 함.
- **토글 분류**: bit-preserving (값/순서 동일: KV_2D/Q_2D/A_2D/DKS/256GRF)의 "변화 없음"은 올바른 커널의 귀결이라 증거가 아니다. order-changing (타일/per_wg 노브)이 chaos-민감 지표를 움직이는 건 주사위 재굴림 -> "타일 노브만 점수를 움직임"은 정상 커널의 서명.
- **WWB 등 end-to-end 텍스트 지표는 커널 게이트 금지** (±0.05). 정확한 reference의 per-op gtest 사용.
- **테스트 데이터가 감도를 가지는지 확인**: N(0,0.1)은 softmax를 균일하게 만들어 Q/K 오독을 숨김 (`logit_scale_gain`, `runtime_scale_multiplier` 사용). `InputGenerateData`의 LCG는 `range*resolution`이 2의 거듭제곱 <= 행 길이면 모든 행이 동일 (오프라인 재생으로 확인). 허용오차가 all-zero 출력을 받아들이는지 계산.
- **NaN-poison 픽스처**: 출력 버퍼를 NaN으로 채우고 재실행해 미기록 원소를 잡는다 (출력 버퍼는 reset되지 않음).
- **golden-split**: 쿼리 구간을 쪼갠 실행 == 한 번에 실행 (새 golden 없이 동치 oracle).
- **sharp-softmax + 음성 대조 스위치** (`SDPA_OCL_NEG_*`, `SDPA_OCL_BIDIR=0`, `BIDIR_GATE=0`): 끄면 **반드시 FAIL**해야 한다. PASS면 스위트가 기능을 관측하지 못함.
- 완전 열거가 가능하면 열거: 정수 모델 전수 검사(SWA 0..699 x seq 1..2099 등), brute-force 쌍 검사 (causal/window 상한 7.76M 쌍).
- "SKIP/opt fallback = OCL 성공" 금지. 판정은 dispatch census (`OV_VERBOSE=4`의 `Enqueue stage ...`에서 대상 커널 >=1줄).

## 4. 실행 환경 함정 체크 (한 번씩 훑을 것)
- GPU 2장: **`--device_suffix=1`** (B70). 누락 -> iGPU `sdpa_opt__*`, `TEST_USE_SDPA_OCL` 무시, `//.platform TGLLP`. (DG2 PC는 번호 체계 다름, `GPU.1` = HWQ 0.)
- 바이너리: PA/SDPA unit = `ov_gpu_unit_tests`, `ScaledAttn*`/KV-cache subgraph = `ov_gpu_func_tests`. 틀리면 `0 tests` (에러 없음). `bin/intel64/{Debug,Release}/`.
- **`source setupvars.sh`는 `set --`로 `$@` 삭제**: `SDPA_ARGS=("$@")`를 source 전에 캡처. 서로 다른 config의 SLM/GWS/LWS가 같다면 토글이 안 먹은 것.
- 셸에서 `NAME=VALUE`를 `"$@"`로 넘기지 말고 `env NAME=VALUE cmd`.
- 덤프용: `OV_GPU_DUMP_SOURCES_PATH=./ OV_GPU_MAX_KERNELS_PER_BATCH=1` (단일 entry). `-device bmg` (xe2는 lnl-m). `NEO_CACHE_PERSISTENT=0`이 필요할 때 있음.
- `.inc` 경로는 `build/src/plugins/intel_gpu/graph/impls/ocl_v2/codegen/include/` (`src/` 없음), 긴 커널은 raw-string 청크.
- 텐서 덤프: `OV_` 접두, 경로 끝 `/`, 레이어 필터는 대문자에 full regex_match (`.*`로 감쌈).
- Debug 바이너리로 정확도, Release/RelWithDebInfo로 성능. 사용자가 "빌드했다"고 해도 어느 트리인지 확인.

## 5. 성능 측정/주장 규율
- 원인 불명의 큰 격차에서는 VTune/ISA 출력 전에 비교 유효성을 확인한다: 실제 dispatch/device, device memory, final-linked reference binary, 동일 useful work와 cadence. 06장의 DG2 사례에서 usm_host 입력과 wrapper-only micro binary는 둘 다 잘못된 비교를 만들었다.
- 지표: **커널 device time** (cliloader `-d -dv`) vs 전체 primitive vs e2e를 구분 표기. SDPA prefill은 1st token의 ~10-15%라 커널 7% = e2e ~0.9%.
- B70은 **2800 MHz 핀**. 호출 수가 적은 커널은 평균 대신 **MIN/median over N runs**. arm 간 순서 교차(ABBA), 분산이 차이보다 크면 판단 보류.
- 비교는 **같은 소스/빌드/입력/캐시 상태, env 하나만 다른 별도 프로세스** (env는 프로세스 내 캐시됨).
- 일의 양을 맞춰라: causal 상한이 없던 ocl은 1.60x 많은 일을 하고도 1.13x만 느렸다 (효율 비교 전에 macs/head, 타일 반복 수 비교). 타일 반복 감소 != 시간 감소.
- **ocloc spill 예측은 런타임 spill과 다르다** (ocloc 0인 후보 4개가 런타임 3.3k-19k B spill). 후보는 런타임 cliloader `SPILL=`로 선택. REG256은 공짜가 아니다 (기본 타일 25% 악화; 큰 타일과 쌍 튜닝).
- 정적 지표(ISA inst 수, SLM, occupancy%)는 성능 증거가 아니다. 변경마다 한 가지만 바꿔 측정, 반증된 실험도 표로 남긴다 (채택/기각 + 수치).
- 하드웨어 이동: B580 -> B70 이후 절대 ns/비율은 재측정 전엔 STALE. ISA/레이아웃/정확도는 유효. 모든 수치에 장비/클럭/반복/지표/출처 표기, MEASURED vs PREDICTED 구분.
- 실모델 하네스 템플릿은 05장 5.5.1 + `test/run.sh`; 성능 전용 모드 `SDPA_SKIP_REF_CHECK=1`이면 정확도는 별도 게이트.
- 대표 결과/기각 실험은 05장 5.5.3-5.5.4 (예: u4 MIXED 145,603 vs micro 147,825 ns; 256GRF+tq32/pwk4/pwq2 = micro 대비 -7.0%, B70 핀).

## 6. 마무리 체크
1. 증명 레벨/통제/음성 대조군을 표로 정리하고 예측 vs 관측 불일치(`!!`)를 숨기지 않는다.
2. 회귀 위험을 명시: NaN/Inf, 동적 shape/shape inference, 정밀도 변환, accumulator 타입, 양자화, kernel selection, execution graph, 메모리 변화 (AGENTS.md).
3. 범위 밖 발견은 Known issues에만. 코드 주석은 짧은 "왜", 측정/유도는 `docs/sdpa_ocl.md`.
4. 메모리 갱신 + 재개 프롬프트 + 한 줄 커밋 제안.
