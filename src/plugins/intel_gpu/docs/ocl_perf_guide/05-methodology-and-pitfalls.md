# 05. 방법론, 성능 주장 규율, 타임라인, 함정 카탈로그, xe_hpg 이식

이 장은 `sdpa_ocl` 커널군(`ocl_v2/sdpa_ocl*.cl`, 호스트 `ocl_v2/sdpa/`, 설계 문서 `docs/sdpa_ocl.md`)을 만들면서 굳어진 **작업 방법**을 정리한다.
커널 기법 자체는 다른 장을 본다: DPAS/타일링 (→ 01-dpas-and-tiling.md), 메모리 IO/prefetch/barrier (→ 02-memory-io-prefetch-barriers.md),
수치/softmax/양자화 (→ 03-numerics-softmax-quantization.md), spill/ISA/프로파일링 (→ 04-spill-isa-profiling.md).

표기 규칙: **MEASURED** = 사용자 또는 에이전트가 장비에서 측정했고 출처 노트가 있음. **PREDICTED/ASSUMED** = 코드 읽기나 모델에서 나온 예측.
하드웨어: **B580** = Arc B580 (Xe2, 초기 개발기), **B70** = Arc Pro B70 (Xe2, 2026-07-05 이후 개발 장비, `GPU.1`/`--device_suffix=1`),
**iGPU** = TGLLP UHD (XMX 없음, `GPU.0`), **DG2** = Arc A770 (Xe-HPG, IP 12.55.8, 별도 PC), **ARL-H** = gfx 12.74 (실기 검증 없음).
출처 표기 `memory:<name>`은 `~/.claude/projects/-home-shingyuk-work-openvino-eddy/memory/<name>.md` (개발 PC 로컬 메모리; 저장소에 없음).

---

## 5.1 작업 분담과 작업 스타일 (저장소 AGENTS.md + 사용자 피드백에서 반복된 것)

| 규칙 | 내용 | 근거 |
|---|---|---|
| 실행은 사용자 | 빌드, gtest, ab(full corpus), 덤프, 벤치, `-fsyntax-only`, `built`, `l0`까지 사용자가 돌린다. 에이전트는 **정확한 명령 블록**과 **예측값**을 준다 | memory:build-is-user-only, ab-runs-are-user-only (2026-09-27 위임 철회) |
| 커밋은 사용자 | 에이전트는 한 줄 메시지를 제안만 한다. 사유/근거는 코드 주석과 `docs/sdpa_ocl.md`에 쓰고 git 이력에 쓰지 않는다 | memory:short-commit-messages |
| 코드 수정 전 계획 승인 | 상세 계획 -> 승인 -> 구현. 범위 밖 발견은 고치지 않고 `docs/sdpa_ocl.md` "Known issues"에 기록 | AGENTS.md "Do not expand task scope", refactor 2026-09 |
| 긴 작업은 세션 분할 | 세션 종료 시 메모리 갱신 + **재개 프롬프트**를 남긴다 (각 `sdpa-ocl-ki-t*` 노트 끝에 한국어 프롬프트) | PROMPTS.md |
| 대기/폴링 | 백그라운드 작업은 `sleep 240` 단위로 폴링, 600은 쓰지 않는다 | memory:wait-with-sleep-240 |
| 숫자 주장 규율 | 측정 없이 "빨라짐" 금지, 방법론(장비/핀 클럭/반복/지표) 명시, measured vs assumed 구분 | AGENTS.md |
| find->verify 워크플로 | 렌즈별 findings를 **먼저 dedupe**, blocker/major만 반증자 2명, minor는 1명이나 에이전트가 grep으로 직접. 40건에 verify 에이전트 80개는 "과하다"는 지적을 받음 | memory:workflow-verify-dedupe-first (2026-09-27) |
| 실행 중 스크립트 편집 금지 | `ps -eo pid,etime,args \| grep <script>`로 먼저 확인. bash는 옛 바이트 오프셋에서 계속 읽으므로 길이가 바뀌면 파일 꼬리를 명령으로 실행한다. 부득이하면 바이트 길이를 동일하게 유지 | memory:dont-edit-running-scripts (t2_census.sh 사고) |

---

## 5.2 A/B 하네스 — GPU 없이 "동작 보존"을 증명하는 도구들

### 5.2.1 도구와 위치 (모두 untracked, `test/` 아래; 저장소 `.gitignore`에 의해 로컬 제외)

| 도구 | 역할 |
|---|---|
| `test/sdpa_ocl_ab.py` | 오프라인 A/B. 서브커맨드 `snapshot, embed, l0, corpus, ab/l1/l1p/l2, selftest, xab, pset, built, hpg, modes` (`sdpa_ocl_ab.py:1147-1187`) |
| `test/sdpa_ocl_gtests.sh` | **사용자 실행** gtest 러너. `run <tag> [그룹..]`, `dump <tag>`, `diff <a> <b>`, `list`. 그룹 표 `GROUPS_DEF` (`sdpa_ocl_gtests.sh:34`) |
| `test/sdpa_ocl_known_issues/` | 이슈별 재현 키트: `repro.sh list\|<item>\|all`, `t2_census.sh`(dispatch census), `*_syntax.sh`, python 재현기. arm끼리는 환경변수 하나/장치 하나만 다르며 관측 vs 예측을 출력하고 불일치에 `!!` |
| `test/microbench/` | (a) ocloc 정적 ISA 비교 `compare_isa.sh *.cl`, (b) 실제 GPU 레이아웃 probe `run_probe.sh`, (c) `probe_dpas_api.sh` (어떤 DPAS/2D-block builtin이 있나). **디바이스 `bmg` 고정** |
| `test/dump_isa.sh` 등 | 실제 커널 ISA 덤프 (→ 04-spill-isa-profiling.md) |
| `test/sdpa_hpg_classify.sh`, `test/sdpa_perf_hpg.py` | DG2 실패 분류 / plain SDPA 짧은 성능표 (기록용, 게이트 아님) |

### 5.2.2 증명 레벨 (각 레벨이 "무엇이 같음"을 의미하는지)

| 레벨 | 비교 대상 | 같으면 의미 | 한계 |
|---|---|---|---|
| **L0** | 커널 소스의 embed 텍스트 (주석/들여쓰기 제거 후) | `.inc`/.so에 들어가는 바이트가 동일 | 주석·들여쓰기만 L0. 코드 줄바꿈이나 `* / ( ) ? :` 주변 공백은 L1 |
| **L1** | 각 config의 `clang -E` **토큰 리스트** | 전처리 후 토큰 동일 (공백 무관) | clang은 OpenCL 기본 헤더 없이 돌려 드라이버가 정의하는 매크로는 미전개. IGC가 보는 것의 oracle이 아님 |
| **L1'** | IGC front-end `.ll` | FE IR 동일 (괄호 추가 등) | 토큰 캐시 때문에 토큰 동일 텍스트의 L1'는 base 복사(동어반복). 판정은 L1/L2 |
| **L2** | `ocloc -device bmg` 결과 ISA 메트릭 (`instCount, numGRF, opcode 히스토그램, zeinfo, call/ret`) | **ISA 등급 A = 동일** | B = heavy op 동일 + scratch 증가 없음 + inst ±0.5% (성능 측정 필요), C = 그 외 |
| **xab** | 두 코퍼스(예: r0 vs r1) 쌍을 지원 중단된 jit 이름을 제거 후 ISA 비교 | 호스트 jit 정리 후에도 같은 커널 | `XAB_DROPPED`를 단계마다 확장해야 함 |
| **pset** | 두 코퍼스의 host jit `(name, value)` **multiset** | 호스트-only 변경이 jit 집합을 안 바꿈 | 순서 변경은 허용 (prelude는 `#define`뿐, 중복 0을 매번 재검사) |
| **built** | 빌드 산출 `.inc`(청크 결합) + 링크된 `.so` + `ov_gpu_unit_tests`에 base 텍스트가 있는가 | 내가 검증한 커널이 **실제 바이너리**에 들어갔다 | 링크 중 실행하면 거짓 음성 |
| **hpg** | 코퍼스를 `ocloc -device dg2/arl-h`로 오프라인 컴파일, TSV | xe_hpg에서 컴파일/DPAS 개수/spill/SLM 정적 점검 | 실행이 아님 (→ 5.7) |

### 5.2.3 명령 (memory:sdpa-ocl-ab-harness에서 검증된 형태)

```bash
python3 test/sdpa_ocl_ab.py snapshot s1f                                  # 작업 트리 .cl 동결 (--force 금지: 증명 체인 파괴)
python3 test/sdpa_ocl_ab.py l0 --base s1e [--work work]                   # embed 텍스트 동일?
python3 test/sdpa_ocl_ab.py corpus test/sdpa_ocl_corpus/r0 --base 235639f1a2   # dump 색인; base mismatch 0 필수
python3 test/sdpa_ocl_ab.py ab --corpus <dir> --base X --work Y --levels l1,l1p,l2 --l2-from-l1 -j 28
python3 test/sdpa_ocl_ab.py xab <corpusA> <corpusB>
python3 test/sdpa_ocl_ab.py pset <corpusA> <corpusB> [--l1 -j 28] [--selftest]
python3 test/sdpa_ocl_ab.py selftest --corpus <dir> --base <sha>          # 도구를 고친 뒤 항상 재실행 (음성 대조군)
python3 test/sdpa_ocl_ab.py built --base s1g [--release]                  # 빌드가 끝난 후에만
bash test/sdpa_ocl_gtests.sh run r0 [U1 U2 ...]; bash test/sdpa_ocl_gtests.sh dump r0
bash test/sdpa_ocl_gtests.sh diff test/sdpa_ocl_corpus/r1/logs test/sdpa_ocl_corpus/r3/logs
```

규모 (MEASURED, memory:sdpa-ocl-ab-harness, 32코어 `-j 28`): corpus 2332 config + 합성 6110 job. L1+L2 cold ~30분, 한쪽 캐시 ~10분, L1만 2-3분.
ocloc 컴파일 1개 0.2-2초.

### 5.2.4 이 하네스가 "정확"한 이유 (잘못 쓰면 거짓 증명이 되는 지점)

- **Build Log 제거**: 런타임이 덤프한 소스 끝에 `/* Build Log: <덤프 경로> ... */`를 컴파일 **후** 덧붙인다. 지우지 않으면 6177 덤프가 6177 "config"가 된다 (`BUILD_LOG_RE`, `sdpa_ocl_ab.py:235`).
- **base는 SHA로 저장** (`base_id()`, `:121`). 인덱스에 `HEAD`를 적으면 다음 커밋 후 커널 본문이 조용히 바뀐다. 커밋하면 HEAD가 움직이므로 R0 기준은 항상 `235639f1a2`처럼 명시 SHA.
- **덤프 == embed(base) 에서 `#include batch_headers` 줄만 뺀 것과 바이트 동일**: `corpus`가 덤프가 어느 빌드에서 왔는지 증명한다 (9/11 덤프는 81a3ffa944와 일치, 다음 커밋과는 불일치).
- **L1은 문자열이 아니라 토큰 리스트** (2026-09-24 수정): 공백 제거 비교는 `const int` == `constint`, `a - -b` == `a--b`를 같다고 판정했다. selftest `merge` 케이스가 L1 DIFF를 내야 한다.
- **토큰 키 캐시** (`cache/tok/`): 토큰이 같은 텍스트는 컴파일을 재사용. 새 스냅샷은 토큰이 바뀐 config만 컴파일 (B3: 8448 중 1462).
- **`--l2-from-l1`**: L1 동일이면 work측 컴파일 생략. ocloc은 입력 토큰에 결정적이므로 타당하나, 드라이버만 predefine하는 매크로에 `#if`를 추가한 편집이면 틀린다.
- **`grep spill`은 무의미**: 모든 .asm의 `//.full_options`에 `-abortOnSpill`이 있다. 진짜 신호는 zeinfo의 `*scratch*/*spill*/*private*`와 `numGRF`.
- **`sync`는 heavy op**: SWSB `sync.nop/allrd/allwr` ±1..5만으로 C가 나온다(T4에서 86 중 74). send/dpas/math/load/store/CALL 개수가 안 움직였고 소스에 barrier가 없으면 해제 가능. `disable_mid_thread_preemption true->None`은 IGC의 **약 600 inst 임계**(true는 instCount<=598, >=601은 None, 109,802 컴파일 전수)이며 기능 변경이 아니다.
- **합성 모드**: 호스트 의존 jit가 없는 단일 매크로 하나만 뒤집는 `modes`(q2d0, kv2d0, cur0, mm0, bidir0, gate0, dpf0 ...). 호스트가 다른 jit까지 바꾸는 토글(`K_PA_I8_2D=0`은 u4 1D를 켬)은 실제 env 덤프(U4_* 그룹)에서만 나온다.
- **런타임 빌드 옵션** (L2에 사용): sdpa_ocl은 `-Dcl_intel_dot_accumulate -Dcl_intel_global_float_atomic -Dcl_intel_subgroup_matrix_multiply_accumulate -Dcl_intel_subgroup_split_matrix_multiply_accumulate -cl-mad-enable -cl-std=CL3.0`, decode는 `-cl-mad-enable -cl-std=CL3.0`. 드라이버 internal options는 `-internal_options '-cl-intel-greater-than-4GB-buffer-required -cl-intel-has-buffer-offset-arg -cl-store-cache-default=2 -cl-load-cache-default=4'`.
  `-device bmg`가 B70/B580 타깃이다 (`-device xe2`는 lnl-m iGPU로 해석됨). `-cl-mad-enable`은 `-options` 안에 있어야 한다.
- **ocloc 결과 == 런타임 IGC 덤프** (MEASURED, memory:sdpa-ocl-kq-tile-keys-32-slower): k16에서 dpas 32=32, send.ugm 98=98, 전체 inst 3039=3039, spill 0=0, 해시 동일. 따라서 ocloc 수치는 런타임 커널의 ground truth. **단 spill은 예외** (5.4 참조).

### 5.2.5 "빌드 없이" 노브 A/B (jit `#define` 직접 편집)

타일 매크로 등은 소스에 베이크되지 않고 jit `#define`으로 주입되므로, `OV_GPU_DUMP_SOURCES_PATH=./ OV_GPU_MAX_KERNELS_PER_BATCH=1 TEST_USE_SDPA_OCL=1 SDPA_SKIP_REF_CHECK=1 <test> --device_suffix=1`
로 단일 `.cl`을 덤프 -> `#define kq_sg_tile_keys 16`을 편집 -> `ocloc compile -device bmg` -> `readelf -S -W X_bmg.bin`으로 `.text.sdpa` 오프셋 추출 -> `iga64 -p=xe2 -d kernel.gen`.
SLM/GRF/spill은 `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=./occ`의 `*.zeinfo` + `//.thread_config numGRF=`. 이 방법으로 OpenVINO 재빌드 없이 구조 변이를 ISA로 비교했다
(memory:sdpa-ocl-kq-tile-keys-32-slower, sdpa-ocl-block2d-gate-relaxation의 "ocloc splice"). `OV_GPU_MAX_KERNELS_PER_BATCH=1`은 프로그램당 커널 1개로 덤프를 단일 entry로 만든다.

### 5.2.6 코드 변경 종류별 증명 레시피 (known-issues 작업에서 정착, memory:sdpa-ocl-known-issues-repro)

| 변경 | 증명 |
|---|---|
| 커널 `.cl` 동작 보존 리팩터 | `l0`/`l1`/`l2` (코퍼스 전수), 헬퍼 추출은 `always_inline` 필수 (5.6 #17) |
| 커널 버그 수정 | `ab --levels l1,l2`에서 **대상 config만** A를 벗어나야 한다. 나머지는 A 유지 |
| 호스트-only 변경 | 새 `dump <tag>` -> `corpus` -> `pset r3 <tag>` (+`--l1`) + `gtests.sh diff r3/logs <tag>`; 의도한 상태 전이만 허용 (T7: `USE_2D_BLOCK_IO_Q 1->0` 정확히 105 config) |
| 테스트 추가 | 수정 **전** 코드에서 음성 대조군이 예측대로 FAIL 하는지 먼저 (T8: 8개 중 f16 4 PASS / bf16 4 FAIL) |
| 동작 변경 | A/B가 달라지는 것이 정상. 증명 대상이 아니라 **측정** 대상. 건드리지 않은 config가 그대로임만 증명 |

기준 코퍼스 이름: r0 (R0=`235639f1a2` 바이너리), r1 (s1g), r3 (Phase 3), rT3/rT4/rT7/rT8 (이슈 단계), rS4/s5_* (xe_hpg).
"identical to r3"는 항상 `gtests.sh diff test/sdpa_ocl_corpus/r3/logs <tag>`를 뜻한다. r3 기준선: U2 FAIL 2, F2 FAIL 15 SKIP 7, NEG_bidir FAIL 44, NEG_gate FAIL 11, 나머지 PASS.
`NEG_*` 그룹은 **반드시 FAIL해야 하는** 음성 대조 (`SDPA_OCL_BIDIR=0`, `SDPA_OCL_BIDIR_GATE=0`): PASS면 그 스위트가 기능을 관측하지 못한다는 뜻.

---

## 5.3 실행/환경 함정 (비용을 치른 순서대로)

| # | 함정 | 증상 | 대응 |
|---|---|---|---|
| E1 | **GPU 2장** (iGPU `GPU.0` + B70 `GPU.1`) | `--device_suffix=1` 누락 시 iGPU에서 돌아 `sdpa_opt__*`가 나오고 `TEST_USE_SDPA_OCL`을 무시. ocl/micro 덤프가 바이트 동일, `//.platform TGLLP` | 모든 gtest에 `--device_suffix=1`. B70=1, iGPU=0. (DG2 PC에서는 OV `GPU.1` = OpenCL HWQ `0`로 번호 체계가 다름) (memory:dual-gpu-device-suffix) |
| E2 | **gtest 바이너리 잘못 선택** | `0 tests from 0 test suites` (에러 없음, stale build처럼 보임) | `src/.../tests/unit/**` -> `ov_gpu_unit_tests` (PA/SDPA 대부분), `tests/functional/**` -> `ov_gpu_func_tests` (`ScaledAttn*`, F1/F2). 위치 `bin/intel64/{Debug,Release}/` (`build/`가 아님) (memory:gpu-func-test-binary) |
| E3 | **`setupvars.sh`가 `set --`로 인자 삭제** | 스크립트에서 `source setupvars.sh` 후 `for kv in "$@"; do export "$kv"; done`이 아무것도 export 안 함 -> 서로 "다른" 3개 config이 전부 기본값으로 측정됨 (1시간 손실) | `SDPA_ARGS=("$@")`를 source **전에** 캡처. 징후: 다른 config인데 SLM/GWS/LWS가 동일. `SDPA_OCL_TRACE_CONFIG=1` 또는 소스 덤프로 토글 반영 확인. python 항목은 conda activate + setupvars를 스크립트 **밖**에서 |
| E4 | **셸 헬퍼의 `NAME=VALUE`** | `"$@"` 로 넘긴 `P=x cmd`는 확장 단어이므로 명령 이름이 되어 "command not found" | `env NAME=VALUE ... cmd` |
| E5 | **커널 임베딩** | `.cl`은 런타임에 디스크에서 로드되지 않고 빌드 시 `gpu_ocl_kernel_sources.inc`로 임베드. 실제 경로 `build/src/plugins/intel_gpu/graph/impls/ocl_v2/codegen/include/` (`src/` 없음; `.../intel_gpu/src/graph/...` 사본은 stale), 긴 커널은 raw-string 청크로 분할되어 grep이 토큰을 놓친다 | `sdpa_ocl_ab.py built`. `ov_gpu_unit_tests`는 그래프 라이브러리를 **정적** 링크하므로 자기 커널 복사본을 가진다: 단위 테스트 relink가 stale이면 옛 커널로 돈다 (memory:sdpa-ocl-kernel-embedding) |
| E6 | **Debug vs Release 혼동** | 사용자가 "빌드했다"고 했으나 Release 플러그인(tests OFF)만 재빌드됨 (T2 22:1x). Release 바이너리가 Phase 2 빌드라 Release 성능 측정 전 재빌드 필요했음 | 바이너리 mtime/`built`로 확인. 성능 측정은 Release/RelWithDebInfo, 정확도는 Debug 코퍼스 |
| E7 | **프로세스 내 env 캐싱** | 같은 프로세스에서 lane을 뒤집을 수 없다. env는 호스트 컴파일 시 raw `std::getenv`로 1회 읽힘 | lane 비교는 별도 프로세스 |
| E8 | **드라이버 빌드 캐시** | 캐시 hit면 빌드 로그가 비어 build-log/덤프 분석이 불가 | `NEO_CACHE_PERSISTENT=0` (DG2 S2 교훈). 모델 blob 캐시는 lane/스테이지를 키에 포함하지 않아 옛 micro blob을 재사용 (`compiled_model.cpp:282-303`) |
| E9 | **텐서 덤프 env** | `OV_` 접두, `OV_GPU_DUMP_TENSORS_PATH`는 끝에 `/`, `OV_GPU_DUMP_LAYER_NAMES`는 **대문자 이름에 full regex_match** (`.*`로 감쌀 것), `OV_VERBOSE`(`OV_GPU_VERBOSE` 아님)=4면 `Enqueue stage <kernel>` 출력, `OV_GPU_DUMP_ITERATIONS=0`은 warm-up infer (memory:gpu-tensor-dump-env). 덤프 전 `stream.finish()`가 호출되어 레이싱이 아님 |
| E10 | **`-fsyntax-only` 사각** | 익명 namespace의 미사용 함수 (-Wunused-function은 codegen 단계) 못 잡음 | 호출처를 손으로 확인. 플래그는 `flags.make`의 `CXX_(DEFINES\|INCLUDES\|FLAGS)`를 `sed`로 추출. `-Wall -Werror -Wmissing-declarations` |
| E11 | **변이 검사에서 변수 제거** | `-Werror` unused로 빌드 실패 -> 옛 바이너리 결과를 읽음 | 변수는 남기고 `&& false` (S5) |
| E12 | **변이 합치기** | MUT-2+3 한 빌드로 -> 예측 합집합 밖 FAIL | 변이는 단독 빌드 |
| E13 | **성능 측정 시 클럭** | B70 클럭이 흔들리면 부호가 뒤집힘 | 2800 MHz 핀, 별도 프로세스는 ABBA 교차 |

---

## 5.4 귀속(attribution)과 검증 규칙

1. **A/B는 정확히 한 가지만 다르다.** 한쪽 대조(toggle on만 또는 off만)는 가설이지 결론이 아니다.
   - 사고 (2026-08-14, head-486 `CL_OUT_OF_RESOURCES`): 전체 `*ScaledAttn*` sweep이 죽었다. 해당 테스트만 `SDPA_OCL_DKS_ACTIVE=0`으로 단독 실행 -> 통과 -> "definitive, 우리 것"이라 단정하고 수정을 설계. 사용자가 토글 없이 단독 실행 -> 역시 통과. 변수는 토글이 아니라 **sweep 안/밖**이었고 원인은 suite-level GPU 메모리 압력. 하마터면 정상 코드를 고칠 뻔.
   - 긴 실행에서만 나는 크래시면 **실행 맥락이 변수**다: sweep vs sweep을 비교.
   - 증명 가능한 no-op 토글(ocloc splice로 HEAD와 바이트 동일 확인)이 쓸 만한 통제다.
   - **이질적 실패 목록 자체가 증거**: 그래프 변환 테스트가 커널 테스트와 섞여 있으면 최소 하나는 커널과 무관. 구성 간 **실패 집합**을 diff한 뒤 이론을 세운다 (`*paged*` 85 vs 70 -> ocl-only 실패 0).
2. **통제가 진짜 통제인지 확인**: 압축 BY_CHANNEL 테스트에서 `TEST_USE_SDPA_OCL(_DECODE)=0`은 T1 수정 전에는 통제가 **아니었다** (fallback이 token-major 페이지를 d-major로 읽어 스스로 FAIL). 통제는 `OV_GPU_PA_BY_CHANNEL_TOKEN_MAJOR=0` (d-major 페이지, U3 그룹) 또는 "같은 테스트, 수정 전 vs 후".
3. **음성 대조군을 수정 전에 먼저 돌린다.** T7: 대조군이 "base가 64 B 정렬이 아니면 틀린다"는 내 예측이 **틀렸음**을 수정 전에 드러냈다 (B70 실측: stride %16이 규칙, base 4/16/32/48 B 어긋남은 stride가 16 B의 배수면 정상). T8: 기능 테스트 데이터가 퇴행(5.6 #20)이라 옛 코드에서도 통과했을 것 -> 대조군으로 발견.
4. **메트릭이 움직인다고 그 메트릭이 중요한 것은 아니다.** spill 2688->0 (`SDPA_OCL_256GRF=1`)인데 커널은 **47% 느려졌다**: spill은 증상이었고 LSC 메시지 수가 실제 비용 (memory:sdpa-ocl-block2d-gate-relaxation). cliloader occupancy%는 속도와 **역상관**(가장 느린 config가 occ 70%) -> 성능 프록시로 쓰지 말 것.
5. **오라클 위치**: 연쇄 네트워크에서는 **첫 레이어만** 입력이 동일하다 (memory:layer0-identical-input-oracle). 레이어 평균 오차는 전파된 오차이며 올바른 커널과 버그 커널이 같은 값으로 포화한다 (minicpm4: `L0 3e-5 -> L1 7e-3 -> L2 5e-2 -> plateau 0.08`). L0 오차를 **출력 dtype ULP**와 비교 (f16 ~4.9e-4 상대; max relL2 ~5e-4 = 표현 한계 내 동일). 자기검증: causal에서 `[0, tile)` 토큰은 자기에게만 의존하므로 그 구간이 bit-identical이면 덤프가 비교 가능함이 증명됨 (`pwk4 vs pwk8` block 0 = 정확히 0.0000, 24 layer 평균).
6. **토글을 분류한 뒤 결과를 믿는다**: *bit-preserving* (값·순서 동일: `KV_2D/Q_2D/A_2D/BLOCK_SKIP/DKS_ACTIVE/256GRF`)이면 "변화 없음"은 올바른 커널의 귀결이라 **아무 증거도 아니다**. *order-changing* (`KQ_TILE_KEYS/PER_WG_KEYS` 등 누적 순서 변경)은 chaos-민감 모델에서 end-to-end 지표를 움직이는 것이 주사위 재굴림일 뿐이다. "타일 노브만 점수를 움직인다"는 올바른 커널의 서명 (minicpm4에서 7회 실행 낭비).
7. **end-to-end 텍스트 지표(WWB)는 커널 정확도 게이트가 아니다.** 작은 모델에서 마지막 비트 변화에 ±0.05 민감, 프롬프트별 bimodal. 정확한 reference 대비 per-op 비교(gtest)를 쓴다. 사용자가 `TEST_USE_SDPA_OCL=0`으로 재측정한 micro 값(0.819114)을 기준값으로 지정했다.
8. **golden-split 테스트 트릭** (bidir token_type_ids, memory:sdpa-ocl-bidir-token-type-ids): 새 golden 데이터 없이 "쿼리 구간을 둘로 쪼갠 실행이 한 번에 실행한 것과 동일"을 동치 oracle로 쓴다. 그리고 reference도 bidir-aware로 확장 (`PagedAttentionReference`).
9. **gtest 허용오차가 느슨하다** -> 정합의 주 증거는 L1/L2/xab/pset. gtest는 skip 포함 **집합** diff.
10. **하네스 데이터가 오류를 가리는지 확인** (5.6 #19-21): N(0,0.1) 데이터는 softmax를 거의 균일하게 만들어 Q/K 오독이 안 보이고 V 오독만 크다. `logit_scale_gain`(스케일 x128)/`runtime_scale_multiplier` 사용. 출력 버퍼는 리셋되지 않으므로 **NaN-poison 픽스처** (출력을 NaN으로 채우고 재실행해 미기록 원소를 index 0에서 잡음).
11. **tier mask가 거부하는 동안 "HPG=1에서 PASS"는 sdpa_ocl 증거가 아니다**: 판정은 항상 dispatch census의 `sdpa_ocl` 줄(>=1)로 (`OV_VERBOSE=4`, `Enqueue stage`). SKIP/opt fallback을 OCL 성공으로 세지 않는다.
12. **독립 리뷰의 가치/비용**: 읽기 전용 다관점 리뷰(T7: 32 agent가 requantize 버그 발견, T8: 12 agent가 퇴행 데이터 발견, 두번째 3-lens가 산술 재유도)는 긴 빌드/덤프 **전에** 돌릴 가치가 있다. 단 dedupe-first (5.1).

---

## 5.5 성능 주장 규율과 대표 결과

### 5.5.1 측정 방법론

| 항목 | 방법 | 출처 |
|---|---|---|
| 커널 시간 | `cliloader -d -dv` (CLIntercept) 의 "Device Performance Timing", kernel 이름 줄에 `SPILL=` 표시. 사용자가 실행 | memory:sdpa-ocl-beats-micro-256grf |
| 병목 후보 | VTune GPU Hotspots로 GPU active/stalled, SBID, barrier stall을 확인하고, 같은 GPU·workload·빌드에서 before/after 비교. VTune stall PC가 없으면 source/ISA 행 귀속을 보류하고 cliloader device time + correctness A/B로 판정 | B70 GPU Hotspots 기록, 상세는 04장 §4.1.1 |
| 클럭 | B70 **2800 MHz 핀** (미핀 시 micro max 16458->10000으로 분산 큼) | memory:sdpa-ocl-int8-perf |
| 반복 | 소량 호출 커널(prefill 2회/invocation)은 평균이 노이즈 -> **MIN/median over N runs** (예: 8회). `-n 1 -ic 4`는 prefill 62+2 호출로 ~2분 | 〃 |
| 실모델 하네스 | `benchmark.py -d GPU.1 -m <모델> -n 1 -ic 4 -pf <jsonl>` 를 `cliloader`로 감싸 `grep -E "sdpa_(ocl\|micro)__prefill"`. e2e 1st token은 `-n 3 -ic 256` | 〃 |
| 결과 해석 | SDPA prefill은 1st token latency의 ~10-15% (llama-3.1-8b: 69.8/475 ms) -> 커널 7% 개선 = e2e ~0.9%. **커널 단위가 최적화 지표, e2e는 거의 안 움직임** | 〃 |
| 값이 "총합"인가 "호출당"인가 | 노트의 수억 ns 단위는 run 전체 **device time 합**, 수천-수백만 ns는 호출당. 표마다 구분 | 각 노트 |
| 정확도 | 모든 성능 A/B에 ref-check PASS를 같이 기록. 단 `SDPA_SKIP_REF_CHECK=1`로 성능 전용 모드를 쓰는 경우 정확도는 별도 게이트 | kq-tile-keys 노트 |
| xe_hpg 성능 비교 (S7+ 지침) | 같은 Release 빌드/DG2/driver/입력/캐시 상태에서 **현재 OCL vs 현재 실제 micro/opt**(이름이 아니라 실제 stage 확인). OCL/control은 `TEST_USE_SDPA_OCL` 하나만 다른 **별도 프로세스**. device kernel time / 전체 primitive 파이프라인 / E2E를 구분. 진단 실행(dump/verbose)과 timing 실행 분리. 변동이 차이보다 크면 판단 보류. 정적 값(ocloc spill, tile, SLM)은 시간·실제 spill이 아니다 | memory:sdpa-ocl-xe-hpg-s7-performance §4 |

### 5.5.2 하드웨어 이동의 효력 (memory:hardware-migration-b580-to-b70)

2026-07-05에 개발 장비가 B580 -> B70 (둘 다 Xe2). **유효 (하드웨어 무관)**: ISA 명령 믹스, 레이아웃 probe 결과, builtin 가용성, `-device bmg` 타깃, 정확도.
**재측정 필수**: 모든 절대 ns와 ocl-vs-micro 비율. B580 수치(예: int8 prefill 1.69x -> 1.45x -> 1.23x)는 **역사**이며 B70 재측정값은 다르다 (B70: head=64 q=7 pinned 평균 1.055x/floor 1.117x; head=128 q=4096은 1.40x, 즉 소형 microbench가 실제 갭을 숨김).
2026-07-05 이전 날짜의 수치는 B580, 이후는 B70 (노트가 명시한 경우만 확정; 아래 표에서 "추정"은 날짜로 추정).

### 5.5.3 대표 결과표

| 결과 | 전 -> 후 | 하드웨어/방법 | 상태 | 출처 |
|---|---|---|---|---|
| PA MIXED u4 (llama-3.2-1b) | OCL 145,603 ns vs sdpa_micro 147,825 ns (**-1.50%**); 최초 exact baseline 158,661 대비 -8.23%; V_PREFETCH on/off 145,603/151,866 (-4.12%) | 사용자 cliloader 커널 평균 (대표 커널 64 calls, SIMD16 REG128 SLM=17664 GWS[272x512] LWS[16x16]). 장비는 노트에 미기재, 날짜상 B70 추정 | MEASURED | memory:sdpa-ocl-mixed-exact-complete |
| plain f16 prefill, llama-3.1-8b q=4096 | micro 1,930,454 -> ocl 기본 2,181,376 (13% 느림) -> **ocl BEST 1,796,200 (7.0% 빠름)**; e2e 1st token 466.98 -> 462.85 ms | B70, 2800 MHz 핀, 호출당 device time, 3 pass, 분산 <0.5%. BEST = 256GRF + tq32/pwk4/pwq2 | MEASURED | memory:sdpa-ocl-beats-micro-256grf |
| head=128 int8 compressed prefill q=4096 | k16 기본 5,102,600 -> **k16pwg4 3,526,979** (micro 4.17M 보다 빠름); sg_per_wg 8<16<32, wgTK 64<128<256 모두 "클수록 느림" | B70 2800 핀, 호출당 min. 정확도는 head=64로만 ref-check (head=128 실데이터 미검증이었음) | MEASURED | memory:sdpa-ocl-kq-tile-keys-32-slower |
| causal/window key 루프 상한 | causal_k: 3,421,875 -> 3,300,312 (1.04x, 타일 반복 1.78x 감소); window W=256: 2,266,562 (1.46x), micro 2,037,708 | B70 2800 핀, `*paged*96` q=1024 head128 f16. causal_k는 사용자가 llama-3.1-8b 실모델에서 큰 개선 확인. window는 llama에 창 없어서 실모델 무효과(SLIDING_WINDOW_SIZE=0이 블록 제거) | MEASURED | memory:sdpa-ocl-causal-bound |
| gemma-4 head-72 vision prefill | 9,570,971 -> 1,030,869 ns (**9.28x**, 37.28% -> 1.11% of GPU time), micro 대비 8.11x 느림 -> 1.14x 빠름. block2d gate %64->%16 + base fixup (6.96x) + DKS_ACTIVE (1.33x) | 날짜 2026-08-14 (B70 추정), 총 device time | MEASURED | memory:sdpa-ocl-block2d-gate-relaxation |
| phi-4 vision head-72 padded view | 7.5x 느림 -> bd8d90097d 수정 | 〃 | MEASURED(노트) | memory:sdpa-ocl-block2d-padded-view-gate |
| gpt-oss-20b u4 PA MIXED head 64 | 1,474,574,702 -> 376,221,254 ns (**3.92x**), micro 401,436,373 보다 6.3% 빠름. 원인 = u4 head-64 row 32 B가 block2d 64 B 규칙 미달 -> K+V 둘 다 scalar gather (192 `load.ugm.d8u32` / 16 dpas per k0). 수정 = `intel_sub_group_block_read_uc16` 페이지 전체 읽기 + 타일 재조정 | 총 device time (B70 추정) | MEASURED | memory:sdpa-ocl-u4-head64-page-read |
| decode (PA GENERATE) f16 | `pa_gqa_single_token` 1,188,447,305 ns 대비 **-7.79%** (M=4 -4.65%, +head-dim split S*V -5.78%) | llama-3.1-8b, ctx ~4352, 32/8 heads, head 128; cliloader 총 device time. 장비 미기재, 날짜상 B70 추정 | MEASURED | memory:sdpa-ocl-decode-kernel |
| decode gemma-4 (head 512 + SWA256 mix) | pa_opt 718.1M -> sdpa_ocl_decode **525.9M (0.73x)**; **e2e 11.43 -> 11.40 ms (사실상 불변)** | 1024 tok prompt, 2040 decode tokens. SG_PER_WG가 M을 이김, V_TILES>=16 판별식 | MEASURED | memory:sdpa-ocl-decode-tiling-sg-per-wg |
| `pa_kv_cache_update_ref` u4 token-major writer | 총 303.5M(퇴행) -> **237.1M (-21.9%)**, upstream d-major 276.0M 대비 -14.1%; prefill -52.2% (probe -52.1%) | llama-3.1-8b B70 head128, 8 kv heads. generate는 d-major와 정확히 동률 (probe 예측 -4.4%는 빗나감) | MEASURED | memory:pa-kvup-u4-token-major-writer |
| 위 writer의 한계 | token-major 페이지는 thread당 16 sector를 더럽혀 d-major의 2 대비 +12%; 4개 수정안 **측정 후 기각**, 천장 986 ns/dispatch, 코드 미변경 | gemma-4 26b, 2040 tokens, B70. e2e 순효과: decode attention -0.185 ms/token vs kvup +0.019 ms/token | MEASURED | memory:pa-kvup-token-major-store-sector-floor |
| int8 plain prefill (B580, **역사**) | ocl/micro 1.69x -> Stage-1 1.45x -> Stage-2 1.23x (median 15.1 -> 12.8 us). V transform_8b만 첫 승리; 이후 K-load 지오메트리 재설계 | B580 head=64 prefill, MIN over 8 runs | MEASURED, **stale on B70** | memory:sdpa-ocl-int8-perf |
| 리팩터 성능 중립 | R0 157,307 ns -> R2 157,580 ns (+0.17%, 잡음) | L4 측정 (6ac827ff6f 빌드) | MEASURED | memory:sdpa-ocl-refactor-2026-09 |
| VTune 기반 f16 V-read 병합 | `16r16x2c`: 524.34 -> 521.01 ms (-0.64%), MD5 동일; SDPA 구간 2.9%/5.2% 짧아짐, SBID stall 43.9–45.5% -> 41.5–43.5% | B70 GPU Hotspots + 동일-workload benchmark. stall PC가 0인 수집이 있어 명령 단위 귀속은 제한됨 | MEASURED | B70 profiling session, 2026-07-31; 상세와 caveat는 04장 §4.1.1 |

**아직 없는 것**: DG2에서 sdpa_ocl SG8 커널의 성능 수치. DG2 S1 baseline은 micro-lane **Debug** 정확도 기준선이며 현재 성능 baseline이 아니다. S7 성능 단계는 시작 전(5.7). ARL-H는 실기 검증 없음.

### 5.5.4 실험 결과표 (채택/기각 모두; 평균 ns, memory:sdpa-ocl-mixed-exact-complete)

| 실험 | on | off | 결론 |
|---|---:|---:|---|
| 최초 correct GRAN=1 | 158,661 | - | 정확성 baseline |
| legacy GRAN=0 | 152,432 | - | **오답** (regression /0,/1) -> 빠르다고 baseline 삼지 않음 |
| paired Kc DWORD retention (장기 private 배열) | 177,716 | - | 약 12% 악화, 제거. **"장기 생존 private 배열 금지"** |
| SV_TRIM | 151,677 | 158,422 | 채택 |
| KQ_FAST | 172,310 | 151,746 | 악화, 제거 |
| KQ_TRIM | 157,932 | 151,529 | 악화, 제거 |
| V_PREFETCH | 145,603 | 151,866 | 채택 (-4.12%) |

---

## 5.6 마스터 함정/반증 카탈로그

형식: 번호. 상황 -> 틀린 가설/실수 -> 사실/원인 -> 규칙.

### A. 귀속 오류 (잘못된 용의자)

1. **T5 head-512 compressed F2 실패** (memory:sdpa-ocl-ki-t5-plain-i8-head512): "prime suspect = sdpa_ocl V `_8b_32r16x4c` 읽기" (B580에서 64 B 폭/x=0만 probe한 레이아웃 가정). 실제 원인은 `dynamic_quantize_kernel_opt_kv_cache.cpp` `Validate()`가 `input_dims.back().v > 256`을 거부 -> ref 커널이 append 못 함 -> 캐시 손상 -> q=1 단계 출력 오류. 수정은 256->512 (upstream PR #38466, 이 브랜치에는 없음). **규칙: 증거가 "q=1 스텝이 틀림"까지만 증명하는데 "sdpa_ocl이 틀림"으로 점프했다. 첫 bisection 단계 `TEST_USE_SDPA_OCL=0`을 안 돌렸고, 소비자 커널을 의심하기 전에 캐시 생산자를 확인한다.**
2. **minicpm4-0.5b WWB -0.070** (memory:minicpm4-wwb-chaos-not-a-bug): `pwk==4`일 때만 나쁜 비단조 판별식(2 ok/4 bad/8 ok)을 버그로 오독. 실제는 L0 <=1.4 ULP, 24층 증폭. 5.4 #5-#7 규칙. **재추적 금지.**
3. **head-486 CL_OUT_OF_RESOURCES** -> 토글 한쪽 대조 (5.4 #1). 실제는 suite 메모리 압력.
4. **tile_keys=32 오답 (C7)** (memory:sdpa-ocl-tk32-bug-hunt, **미해결**): `basic/31`(head 64 prefill q=1024)에서 **`tk=32 && pwk>=4 && kq_query_blocks>=2`일 때만** FAIL, 두번째 query block(index 35264 = token 275, WG-local q 19)부터 오답, token 0..274는 정확. 모든 메모리 경로 토글/256GRF/BLOCK_SKIP/타일 불변식/S_slm/Q_slm 정적 검사 전부 배제. **2026-09-25: 같은 repro가 더는 실패하지 않음 -- PA prefill이 MICRO_MATH=1을 쓰면서 가려졌다 (가설; `SDPA_OCL_MICRO_MATH=0` arm 계획이 노트 끝에 있음)**. 이 노트 자신도 "minicpm4와 혼동 금지: 이쪽은 정확한 reference에 대한 gtest가 결정적으로 같은 index에서 실패하므로 실제 결함"이라고 구분. **override-only, `kq_sg_tile_keys` 16/32 외는 `#error`. 기본값으로 32를 출하하지 말 것.** 부수 정정: 노트의 옛 S_slm 모델이 틀렸다 (subgroup block read는 **strided**: component i of lane L = `p[i*sub_group_size + L]`; 커버리지/충돌 분석은 두 전단사를 구분하지 못한다).
5. **SLM-occupancy 가설 (kq_sg_tile_keys 16->32가 느린 이유)** (memory:sdpa-ocl-kq-tile-keys-32-slower): ISA로 spill/전체 inst/sync/V-load 지연/dispatch/dpas 병렬성을 배제하고 "SLM 17.5->25.7KB로 occupancy 7->5 WG"를 지목했으나 **디바이스 측정이 반증**: 실제는 workgroup 크기(sg_per_wg)와 kq_wg_tile_keys의 독립 두 축, 둘 다 "클수록 느림". 키 축은 reduction 축이라 subgroup이 늘수록 atomic max/barrier/SLM 왕복이 늘고 유용한 일은 안 는다. WG 수(2048)는 모든 config에서 동일하므로 under-fill 이야기가 아님. **정적 모델 -> 장비 측정으로 검증해야 한다.**
6. **tile 수 감소 != 시간 감소** (causal-bound): causal_k는 타일 반복 1.78x 감소인데 시간 1.04x, window는 1.71x 감소에 1.46x. 1024 WG/sg_per_wg=16에서는 work-bound가 아니라 reduction 오버헤드 지배 (열린 퍼즐이었고 이후 per-iteration 오버헤드로 해소: ocl 2112 k0 iter/head vs micro 272 (7.76x)).
7. **"ocl이 느린 이유는 커널 효율"이라는 직관** (causal-bound): 실제로 ocl은 causal 상한이 없어 **1.60x 많은 일**을 했고, 단위 일당으로는 이미 micro보다 1.41x 빨랐다. 효율 비교 전에 **일의 양**(KQ macs/head, causal efficiency)을 맞춰라.
8. **spill 숫자는 증상** (gate-relaxation): 256GRF가 spill을 2688->0으로 만들었지만 47% 느려졌다 (LSC 메시지 수, 반으로 준 resident thread). **REG256은 공짜가 아니다**: 기본 타일에서 25% 느려지고(2.18M->2.73M), 이전에 spill하던 큰 타일과 **쌍으로** 튜닝해야 이득.
9. **ocloc spill 예측은 쓸모없다** (beats-micro-256grf): `-device bmg`에서 spill=0이던 후보 4개가 런타임에 3.3k-19k B spill. 후보 선택은 **런타임 측정만** (cliloader `SPILL=`). `T128k256`은 k0 반복 수가 가장 적은데 spill로 2.2x 느림 -> "k0 반복 수는 목적함수가 아니다".
10. **"읽기 횟수를 줄이면 빨라진다"** (int8-perf): V read 횟수 감축 3회 시도 실패. ISA가 보여준 병목은 **load 지오메트리** (SIMD-1 scalar scale/zp + 중간 크기 K-data 메시지)이지 횟수가 아니었다. cp-pair read 재사용은 읽기를 반으로 줄이고도 22,812 vs 21,666 ns로 **더 느림** (`if(cp&1)` + 루프 간 `v_pair`가 IGC 스케줄링을 교란).
11. **K 2D-block 로드 실험 (NEGATIVE RESULT)** 및 토글들 (`K_I8_VARIANT`, `V_SCALE_CACHE` 무이득: L1이 이미 재로드 흡수, `K_DIAG`)은 정리 단계에서 전부 삭제하고 V transform 승리만 유지. 마지막 갭은 widening+zp broadcast가 하드웨어 OpenCL의 floor (micro는 gemmstone 레지스터 스케줄로 회피).
12. **block-level causal-mask skip** (beats-micro-256grf): micro의 `if (causal_k_end > causal_q_begin)`를 흉내냈고 9.3M 원소 brute-force로 정확함을 확인했지만 **IGC가 분기를 취하지 않고 평탄화** (cmp 20->36, sel 불변, net inst +16). 상한도 작음 (mask `sel` 17/35, 나머지 16은 fmax reduce). `k_mask` remainder add 제거도 add 60->20이 mov 59->82로 바뀌어 -3 inst뿐.
13. **MIXED 실험** (5.5.4): paired Kc DWORD retention, KQ_FAST, KQ_TRIM 모두 악화. "REG128만으로 spill/악화 원인을 단정하지 않는다."
14. **U4 token-major writer**: probe가 -4.4% 예측한 generate는 정확히 동률. **sector floor**: 4개 후보 수정을 측정해 모두 기각, 천장 986 ns. 코드 변경 없이 결론만 기록하는 것도 유효한 종결.

### B. 정확성/게이트 구멍

15. **mixed layout gate의 `use_ocl` 구멍** (memory:pa-mixed-layout-gate-use-ocl-hole, T1 f81a75eef6(구 96b457a4f0)로 수정): `paged_attention_opt.cpp`의 "페이지가 어떻게 생겼나" 검사가 "어느 커널이 읽나"인 `use_ocl`로 가드되어 `TEST_USE_SDPA_OCL=0`에서 micro MIXED가 token-major 페이지를 읽었다. **B70 맹점**: Xe2에서는 env와 staged backend가 항상 일치하므로 남은 `use_ocl` 읽기가 모든 B70 테스트를 통과한다 (T2).
16. **by-channel layout을 reader 확인 없이 token-major 기본** (T1): `by_channel_token_major_readable()`이 `sdpa_ocl_selected`를 "decode reader 있음"의 대용으로 쓴다 -> 술어를 넓히면 xe_hpg에서 token-major 페이지가 생기고 d-major GENERATE reader가 예외 (S3에서 술어 분리).
17. **plain `inline` 헬퍼가 커널 전체 ISA를 바꾼다** (memory:igc-inline-helper-pipeline-switch, ocloc 26.22.38646.4 bmg): IGC가 인라인하는데도 파이프라인이 바뀐다. **`__attribute__((always_inline))`만** 직접 작성과 ISA 동일. 규칙 R7-R12: 모든 헬퍼 `SDPA_OCL_INLINE`, lane은 `size_t lane` + `int lane_i`(헬퍼 안 `(int)lane` 금지), 일부 설정에서 상수가 되는 인자로 헬퍼 안 분기 금지, 헬퍼 안 배열 선언 금지(호출부 선언 후 `__private` 포인터), 좌표는 잎 변수를 넘겨 헬퍼 내에서 계산, 루프 전달 출력은 호출부 변수(out-param). `bidir_query_groups`는 인라인 유지(gate0에서 C). 비-A 원인 분리: 호출부 hunk를 하나씩 적용한 스냅샷 -> 대상 config만 L2 -> `compile_one(keep_dir=)`로 IR 단계별 정규화 diff (`_beforeUnification -> _afterUnification -> _optimized -> ISA`).
18. **tiling 불변식 위반은 조용히 오답** (memory:sdpa-ocl-tiling-constraints; 상세 -> 01-dpas-and-tiling.md): `kq_sg_per_wg_keys`만 env로 바꾸면 sg_per_wg 불변식(3번)이 깨져 존재하지 않는 subgroup이 value 열을 소유하고 출력이 안 써진다. 그 sweep의 "빨라 보이는" 타이밍은 무효였다.
19. **PA 하네스 데이터 N(0,0.1)이 Q/K 오독을 숨김** (memory:pa-harness-data-hides-qk-errors, T4 "WRONG earlier claim"): logits ~0.008 -> 균일 softmax -> ref std ~0.005, 허용오차 0.025/0.075가 **all-zero 출력을 받아들임**. U[-1,1] 분포라는 내 주장은 틀렸다 (`generate_input_data`는 호출자 없음, 실제는 `generate_realistic_data` mt19937(1234) N(0,0.1)). 출력 버퍼는 `reset=false`라 0 또는 stale. 그래서 T4의 24개 `MISSING -> PASS`는 아무것도 증명하지 못했을 것 -> NaN-poison 픽스처 + f16 U2 쌍(허용 0.002, 70-90% 원소가 초과)만이 값-기반 탐지기.
20. **퇴행 LCG 데이터** (memory:gtest-lcg-power-of-two-degenerate-data, T8): `InputGenerateData(start, range, resolution, seed)`는 `range*resolution`이 2의 거듭제곱 <= 행 길이면 주기가 head size와 같아 **모든 행이 동일** -> logit 동일 -> softmax가 scale을 무시 -> 옛 코드에서도 통과. `(-1, 2, 32)` + head 64가 정확히 그 경우. resolution 31로 교체 (logit spread ~14, 오독 오차 0.85 vs bf16 임계 0.025). **테스트를 믿기 전에 생성기를 오프라인으로 재생.**
21. **T4: SWA 241..256 출력 미기록** (027a9a95ae): decode가 tmp_out/output을 **전체 seq_len**으로 고르고 finalization은 SWA-유효 길이로 골라 출력이 안 써졌다. 정수 모델 전수(SWA 0..699 x seq 1..2099, 10.2M 케이스) 0 위반으로 수정 확인. 알려진 모델 중 (240,256] 창은 없음 (gpt-oss 128, Gemma-3 512/1024, Mistral 4096). 잠복 결합: decode는 `SLIDING_WINDOW_SIZE != 0`, 호스트/finalization은 `SWA_BLOCK_SKIP_ENABLED`로 게이트 (decode가 scores를 거부하므로 일관).
22. **T7 block2d padding** (325f47dfa5): `block2d_layout_ok`가 padding을 안 봄. B70 **실측 규칙**: token stride가 16 B 배수가 아니면(260/264 B) 오답, 첫 head 시작이 2 B 어긋나면 오답, base 4/16/32/48 B 어긋남(stride 16 B 배수)은 정상. 게이트는 문서화된 64 B/16 B base 규칙도 유지(스펙 + 옛 B580 phi-4 실험; B580과 다른 실리콘일 수 있음). 잔여: **동적** K/V padding은 호스트가 증명 못 함 (DISABLED residual suite 4개 FAIL). 부수 발견(미수정): `pa_kv_cache_update_ref.cl:297` i8 BY_CHANNEL requantize가 `in_data_pitch`를 무시 (padded key의 새 토큰 j>=1 오독; 트리거 좁음: i8 BY_CHANNEL + padded key + past_len%16!=0). rank-4에서 fixup 티어의 `axis_unpadded(X)`가 rank-2에선 vacuously true였음.
23. **T8 SCALE_DATA_T=half** (8300f14d92): 문서는 "잠재"라 했으나 bf16 runtime scale에서 live (F1이 bf16 runtime-scale sdpa_ocl 커널 400개 컴파일, scale을 half로 읽음). 수정 후 기존 `sdpa_opt.cl:154` PA scale 버그(`SCALE_TYPE = INPUT3_TYPE`, runtime f16 scale을 int32로 읽음; 실모델은 PA scale이 Constant라 도달 불가) 발견, 미수정, 문서화. 곱셈 배율 64를 쓴 이유: N(0,0.1)은 scale을 숨김.
24. **8b transform / block2d 하드웨어 규칙** (→ 02-memory-io-prefetch-barriers.md; memory:sdpa-ocl-8b-transform-32row-min): 8b transform은 32-row 전용, 우리 helper가 스펙보다 엄격. 64 B 폭 x=0만 probe한 `_32r16x4c` 레이아웃 가정이 head 512에서 검증된 적 없음(결국 무관했음, #1).
25. **plain int4 / bf16 PA / jitter**: plain int4는 nibble 언팩이 아예 없고(컴파일만, 미디스패치), plain i8 bias trick은 현재 정확(`kv_cache_compression.cpp:244` zp_dt = immad && !int4 ? i8 : query type), `jitter.hpp` dup-macro assert 소실 (T9 P2 미착수). F2 bf16 compressed는 dynamicquantize 레이아웃 실패, `SDPAFusion/0`은 upstream 3D-XMX 분해 규칙 (T10: sdpa_ocl 아님).
26. **sink 입력**: 30개 `TEST_USE_SDPA_OCL=1` 실패는 `token_type_ids` 공백 + 설명 안 된 빌드 실패 1건이었다 (memory:sdpa-ocl-sink-input).
27. **`sg_ij` 균일성** (minicpm4 부수 발견, 미수정): micro는 `sub_group_broadcast(get_local_id(1), 0)`, sdpa_ocl은 원본 `get_local_id(1)` -> subgroup block read/write 주소와 broadcast 인덱스가 subgroup-uniform을 요구. `NUM_HEAD_SIZE_GROUPS`(`pa_kv_cache_update_ref.cl:227`)는 괄호 없는 매크로 (현재 사용처는 우연히 안전, head 320에서 1로 잘림).
28. **리팩터 스코프 판정**: 코드 변경이 아니라 "관찰 가능한 동작" 단위로 증명한다. `HEAD` 대신 SHA, 스냅샷 `--force` 금지, `corpus <dir>`은 `<dir>/configs`를 **삭제**하므로 다른 base로 r0/r1을 재실행하면 증명 기준점이 파괴된다.

### C. 도구/프로세스 함정은 5.3 (E1-E13)과 5.4

---

## 5.7 xe_hpg(DG2/ARL-H) 이식 노력 — S0-S9

상세 playbook: `.claude/skills/xe-hpg-porting/SKILL.md`. 계획 원본은 메모리 `sdpa-ocl-xe-hpg-plan`(허브), `-facts`, `-s7-performance`(최우선 보충 지시서, 2026-10-02 승인), `-briefs-s2-s9`, `-plan-full`; 작업 사본 `test/sdpa_ocl_xe_hpg/plan/`. 장비: DG2는 별도 PC(raptorlake-02), B70이 개발 PC(raptorlake-01).

### 5.7.1 단계 표 (2026-10-02 기준)

| 단계 | 내용 | 상태 | 현재 브랜치 커밋 (메모리의 옛 해시 -> 제목으로 대응) |
|---|---|---|---|
| S0 | 하드웨어 probe (DG2 컴파일+ISA, enqueue 없음) | DONE 2026-09-30, DG2 PASS rc0 | `test/sdpa_ocl_xe_hpg/probe/S0_RESULTS.md` |
| S1 | DG2 baseline (micro lane, HEAD 36fb814999) | DONE: U1 0/325/112, U2 0/287/108, U3 0/195/64, F1 133/2066/0, F2 15/101/7, SEL 18/18; UNEXPLAINED=0 | 데이터는 DG2 PC |
| S2 | SG8 프로토타입 (DPAS 매핑 H1, 타일, 미니 SDPA, K 방향, DM spike) | DONE: H1 8/8 항등, NC 24/24 | `s2/S2_RESULTS.md` |
| S3 | 술어 분리 + fake-device 테스트 | DONE | `5af20d13f1` (구 01bcf08592) |
| S4 | SG16 중립 리팩터 (키 수를 `DPAS_K` 단위로) + Xe2 불변 증명 + tripwire | DONE: L1 same 2491 + modes 8937, ISA SAME 92+162 | `976fbf4b0c` (구 7940e525ca) |
| S5 | xe_hpg host jit (2D 끔, SLM/WG 체크, tier mask, TEMP 라우팅, `OV_GPU_ARCH_OVERRIDE`) | DONE | `a0c60df997` (구 0ae73cfa32) + `1799d40f79` (구 d06fc2d3d6) + `79b856f497` (가드) |
| S6a | plain f16 static prefill SG8 코어 | DONE (사용자 확인) | `10e4f00b68` (구 050605239a) |
| S6b | bf16, mask, causal, sink, dynamic, q<=1 | DONE | `087d2bd171` (구 27304008a1) |
| S6c | plain i8 KV | DONE | `b739dc5881` (구 f972cc2d80) |
| S7a | PA PREFILL + 진입 성능 checkpoint | **미착수 (다음)** | - |
| S7b | PA MIXED f16 + 스칼라 Kc/Vc | 미착수 | - |
| S7c | PA 기능 (sink, bidir, qq_bias, window, k!=v, runtime scale) | 미착수 | - |
| S8a/b/c | i8 BY_TOKEN / d-major i8 BY_CHANNEL (DM reader) / u4 | 미착수 | - |
| S9 | 기본값 flip (DG2 + ARL-H 12.74, 캐시 태그, 문서) + DG2 census 2회 + 실모델 | 미착수 | `TEMP(S9)` 2곳 제거 |

현재 `kHpgTiersReady = PLAIN_F16_STATIC | PLAIN_EXT | PLAIN_I8` (`sdpa/sdpa_ocl_hpg.hpp:33`); `TEMP(S9)`는 `sdpa_opt.cpp:86`, `paged_attention_opt.cpp:1623` 정확히 2곳. 안전 정지점: **S9 전까지 xe_hpg 기본 동작 불변** (`TEST_USE_SDPA_OCL_HPG` off).

### 5.7.2 S7 이후 성능 정책 (사용자 승인 2026-10-02, memory:sdpa-ocl-xe-hpg-s7-performance)

- 성능 우선순위 **PA PREFILL > PA MIXED > plain SDPA q=1**. 숫자 합격선/"회귀 감수 후 기록만" 정책 폐기. 각 성능 이슈는 **FIXED / REFUTED / 사용자 명시 승인 ACCEPTED·DEFERRED** 중 하나와 근거, 한계, 다음 revisit 조건을 갖는다. 구조/정확도 blocker를 성능 defer로 닫지 않는다.
- 공통 SG8 config/tile/GRF/reader 튜닝은 완료된 S6 plain/q=1에 영향을 주어도 허용하되 영향 회귀 확인 필수. B70/Xe2 하드웨어 회귀와 compiled-kernel/L1/corpus/pset 불변 증명은 DG2-only로 면제되지 않는다.
- 각 단계 루프: (1) 정확도/실제 dispatch 확인 -> (2) 저비용 정적 점검 (기존 corpus 재사용) -> (3) 승인된 대표 케이스 제품 경로 스크리닝 -> (4) 트리거 시 한 번에 한 변경 통제 실험 -> (5) 재검증/사용자 처분. 측정 가능한 위험을 "게이트 아님"으로 자동 생략하지 않는다. 도구가 없으면 "미측정/판단 불가"로 쓴다.
- 조사 트리거 예: 새/증가 spill, private array 장기 생존, hot loop 내 중복 gather, 크기/페이지/작은-query 경계의 급변, 커널은 빨라졌는데 PA 전체/E2E 악화, host 거부로 실제 lane이 달라짐.
- **정적 값의 의미**: `s6a/pass_real/results.tsv`의 plain 구성 h64=7,968 B / h128=12,128 B / h256=28,064 B spill은 **ocloc register allocator 보고값이지 DG2 런타임 spill/시간이 아니다**. 현재 build option은 HPG에서 256GRF를 **강제** (`sdpa_gen_ocl.cpp` `get_build_options`)하므로 `SDPA_OCL_256GRF=0/1`만으로는 128/256 실험이 아니다.
- `sdpa_perf_hpg.py`의 `prof_us`는 누적 primitive 평균이며 warmup 포함, 정수 us -> warmup 제외 median이라 부르지 않는다. gtest 소요/XML time은 컴파일/참조 포함 wall time이라 성능 증거가 아니다.
- 실행마다 실행자/workload/예산/arm 순서/warmup/반복을 **개별 승인**. 모든 새 키트/로그는 `test/sdpa_ocl_xe_hpg/` 아래.

### 5.7.3 DG2 실기 사실 요약 (MEASURED, memory:sdpa-ocl-xe-hpg-facts, S0/S2/S1)

| 항목 | 사실 |
|---|---|
| 장비 | A770, device 0x56a0, IP 12.55.8, driver 25.13.33276.16; SG 8/16/32, SLM 65536 B, max WG 1024, 512 EU x 8 thread; OV `GPU.1` = OpenCL HWQ `0` |
| SG8 DPAS | f16/bf16 x M=1/2/4/8 컴파일 성공 (`dpas.8xM`, hf/hf, bf/bf). A는 `int/int2/int4/int8`, B `int8`, C `float..float8` |
| SG16 `short8` A | **오류 없이 컴파일되지만 ISA에 DPAS가 없다** (입력 load도 없고 store만 남음, 128/128 결과 0). "컴파일 실패" 예측이 틀렸다 -> **조용한 쓰레기 출력**. 모든 SG8 작업은 `dpas` 개수 확인 필수 |
| lane 매핑 (H1) | A: lane l = K(2l, 2l+1), low half = 2l. B: dword j = K(2j, 2j+1), lane n = 열. C: lane = 열, 성분 m = 행. 8/8 항등, 순열 대조 NC 24/24 일치 |
| 2D block IO | read/write/prefetch 모두 컴파일 거부 + 확장 미광고 (`cl_intel_subgroup_2d_block_io`는 xe_hpc+ 전용). subgroup buffer prefetch도 거부. 2D pragma는 경고만 내고 매크로 미정의 -> **`#ifdef`로 못 끄고 jit 스위치(`block2d_io_allowed`)로** 꺼야 함 |
| block IO | global uint/ushort block IO OK, **local uint/ushort block IO는 pragma 없이 OK**. 홀수 row pitch에서 `block_read uint`는 오답 (err 49; ushort 쌍 fallback은 정상); 2바이트 오프셋 `block_read_us`도 오답 -> 연속 K reader는 pitch 짝수 + 4 B 정렬일 때만 block_read |
| GRF | 32 B (Xe2 64 B): 128GRF = 4 KB/thread. `-cl-intel-256-GRF-per-thread` numGRF 128->256 (스레드 수 반). 정정: "float8 C가 SG8에서 1 GRF"는 틀림 (32 B GRF 8개) |
| 타일/GRF 선택 (S2, 미니 SDPA 벤치) | **256GRF, 16x16, 키 방향 sg 4개 (sg4x2 3353 us) spill 0**; 128GRF는 모든 구성에서 3.8~11 KB spill; k16q32/k32q32는 256GRF에서도 spill. K 방향 D=128 t_med: K0 연속 286 us (노트에 채택 여부 미기재, K1 채택), **K1 스칼라 gather 365(채택)**, K2 vload8+pack 370, K3 SLM 전치 449, K4 A=Q/B=K 344(KQ만, 미채택). 한계: 소형 프로토타입이며 제품 PA 전체 최적성 증명이 아님 |
| split-matrix MAD | DG2 전용 (ARL-H 없음) -> 필수 경로로 쓰지 않음 |
| 꼬리 규칙 | `db*16+2*idx >= D`이면 dword를 0으로; K나 Q **한쪽만** 가드하면 충분 (양쪽 모두 없으면 오답) |
| F1 DG2 baseline 실패 133 | 전부 sdpa_ocl 무관: bf16 head 486/512 `CL_OUT_OF_RESOURCES` 128, bf16 runtime-scale 4 (단독도 실패), f16 runtime-scale 1 (**프로세스 내 이전 테스트 순서에 의존**, `=0`에서도 재현) |
| f32 PA | OK 14개 뒤 rc=134 abort (`CL_OUT_OF_RESOURCES`) |
| DG2 micro-only 통과 목록 | U1 24개 = micro_sdpa_prefill 13 + u4_mixed_micro 5 + update_shape 1 + sink {0,1,2,3,6} 5. **S9 전환 전에 ocl lane에서 전부 PASS여야 회귀가 아님** (sink 5, prefill 13 -> S7a/S7c; u4 5, update_shape -> S8c) |

### 5.7.4 Tier 표와 라우팅 규칙

| 비트 | 대상 | 상태 |
|---|---|---|
| PLAIN_F16_STATIC (1<<0) | plain f16, static, q>1, mask/causal/sink/runtime scale 없음 | READY (S6a) |
| PLAIN_EXT (1<<1) | bf16, mask, causal, sink, runtime scale, 동적, q<=1 | READY (S6b) |
| PLAIN_I8 (1<<2) | plain i8 KV (u4와 scale/zp 없는 i8은 `supported()`가 거부) | READY (S6c) |
| PA_PREFILL (1<<3) + PA_MIXED_F16 (1<<4) | PA는 두 스테이지가 함께 컴파일되므로 **둘 다 필수** (`hpg_tier_required`는 현재 PREFILL/MIXED를 함께 요구; S7a의 "PREFILL=OCL, MIXED=micro" 분리는 **구현할 호스트 변경**) | S7a/S7b |
| PA_FEATURES (1<<5) | sink, tti, qq_bias, window, k!=v, runtime scale | S7c |
| PA_I8_TOKEN/PA_I8_CHANNEL/PA_U4 (1<<6/7/8) | 압축 PA | S8a/b/c |

라우팅 함정: `sdpa_ocl_selected`가 true인데 `supported()`가 false이면 `none`(opt 커널)이다 -- micro tail로 안 떨어진다 (`paged_attention_opt.cpp:1615-1620`). 그래서 S5가 `TEMP(S9)`로 xe_hpg에서 거부된 op를 micro lane으로 되돌린다. `add_stage`는 codegen 예외를 삼키므로 SG8 빌드 실패가 **조용히 opt로 강등**된다 -> census로 stage 추가 확인 (`sdpa_ocl` dispatch 줄). `SDPA_OCL_NEG_SG8=1..6`은 SG8 매핑을 일부러 깨는 음성 대조 (sharp-softmax 테스트가 FAIL해야 함; `=4`는 unaligned-K fallback 강제, PASS해야 함).

### 5.7.5 B70에서 할 수 있는 것/없는 것

- **불가**: SG8 실행 검증 (B70 min SG=16, iGPU는 XMX 없음) -> 실기는 DG2. 업스트림 CI에 DG2 없음.
- **가능**: 호스트 라우팅/jit 생성 테스트, `OV_GPU_ARCH_OVERRIDE=xe_hpg` (ENABLE_DEBUG_CAPS 한정) + `SDPA_OCL_HPG_TIERS=all` 덤프 -> `sdpa_ocl_ab.py hpg --device dg2 --grf256 [--define SDPA_OCL_SG8_ARM_READY]` 오프라인 컴파일, SG16 중립 리팩터의 바이트 동일 증명 (S4). **위조 arch로 돈 gtest 결과는 해석 금지, 덤프된 소스만 의미가 있다.**
- 위조-arch 덤프와 실제 DG2 덤프의 jit 동일성은 S6a에서 `pset`으로 확인하도록 계획.
- pass1/pass2 검출기: first_error만 보면 가려지므로 2D 누수는 TSV `any_2d` 열 + pass2(`ARM_READY` 변이, 2D tripwire 38/38)로 잡는다. 오프라인 `ab.py hpg` 컴파일 1회 수 초~25초.

---

## 5.8 프로젝트 타임라인 (git log + 메모리; 해시는 현재 브랜치 `sdpa_ocl_dpas` 기준)

메모리가 가리키는 옛 해시 중 다수 (예: `235639f1a2`, `1706b74354..0653092a5a`, `96b457a4f0`, `7940e525ca`, `050605239a`)는 브랜치 재정렬 후 현재 브랜치에 포함되지 않는다 (`git branch --contains` 결과 비어 있음). 아래는 제목이 일치하는 현재 해시다.

| 날짜 | 단계/커밋 | 내용 |
|---|---|---|
| 2026-06-17~24 | `0ba4ebfcd0`, `864b2de687`, `1bc17f9186`, `27bd824c9b`, `2d9038bf46`, `feb3a58836` | `sdpa_ocl.cl` 최초: mha, head 128, 2D mask, cooperative Q fetch, config 표 + `choose_config`, GQA |
| 2026-07-05~07 | `13f7e52c8a` k block2d transform read | B580 int8 K-load 재설계 (1.69x -> 1.23x). B70로 이동, B70 재측정 |
| 2026-07-13~30 | (메모리) | kq_sg_tile_keys 32 조사 -> sg_per_wg/wgTK 두 축 발견; causal_k + window_k0_begin 상한 (llama 실모델 개선); 256GRF + tq32/pwk4/pwq2로 micro 대비 -7% 역전 (07-30); `90a05f180f` V_I8_MULTIBLOCK (07-28, 이후 head 512 의심 대상이었다가 무관 판정) |
| 2026-08-08~12 | `53ba9bf880`, `ccb0a0ab81`, `dd55fce6b3`, `01b04aa769`, `ae1908b4b8`, `e1249f0164`, `3cf4c60692` | int8 by-token PA, 새 `sdpa_ocl_decode` (PA GENERATE, f16 -7.79% vs pa_gqa), int8 token-major BY_CHANNEL, int4 KV |
| 2026-08-13 | `d060b84ddc` (rename, default), `a1effb89ad` (kvup) | `sdpa_ocl` 기본 lane, u4 head-64 whole-page read (3.92x), token-major u4 writer (-21.9%) |
| 2026-08-14 | `6f78fd496e` | block2d %64->%16 + base fixup + DKS_ACTIVE: gemma-4 head-72 9.28x. 이 주 `365997e7b1`(08-18) decode head 72 |
| 2026-08-19 | `3705c3b124`, `8a27746c9c`, `1981d0d961` | bidirectional token_type_ids (prefill, MIXED, empty gate) |
| 2026-08-23 | `5cf2d5fc5a` | MIXED의 현재 토큰을 Kc/Vc에서 읽기 |
| 2026-09-09~11 | (메모리) | exact u4 MIXED (SV_TRIM/V_PREFETCH 기본값): 145,603 vs micro 147,825; minicpm4 chaos 결론 (09-11) |
| 2026-09-10~21 | `e6450198b6`, `bda92b80af`, `6b0e442b09`, `c5b2811f54`, `b0dd980faa`, `78a552b414` | k_head != v_head, micro_math softmax, wide micro math, bf16, scalar runtime mask + unaligned head, qq_bias |
| 2026-09-23~24 | `9694bd61ef`, `142706ebd5`, `53296f18b0`, `2ce502f5ce`, `1677e8f28f`, `b72dc44907`, `631e0ce1bb` | Xe2+ 허용, **동작 보존 리팩터** (legacy 제거, 헤더 분리, 주석+docs, K/V 헬퍼 추출, 호스트 jit 재구성, known issues 문서). 코드 6431->5179줄 (-19.5%), `sdpa_ocl.cl` 2767->996. 단계별 L0/L1/L2/pset/xab 증명 |
| 2026-09-25~29 | `ad6e56dae9` T3, `027a9a95ae` T4, `f81a75eef6` T1, `f3167294bf` T2, `325f47dfa5` T7, `8300f14d92` T8, `270ba49532` T11(주석) | known-issues 효력 (5.6 B). T5는 upstream PR #38466로 분리. 09-29 사용자 결정으로 일단 종결 (T9/T10/T11-perf/tk32 on-demand) |
| 2026-09-30 | `5af20d13f1` S3, `976fbf4b0c` S4 | xe_hpg 플랜: 술어 분리, SG16-neutral 리팩터 + Xe2 불변 증명. S0/S1/S2는 DG2 PC |
| 2026-10-01 | `a0c60df997`, `1799d40f79`, `79b856f497`, `10e4f00b68`, `087d2bd171` | S5 (host jit, tier mask, TEMP 라우팅), S6a (plain f16 SG8), S6b (bf16/mask/sink/dynamic) |
| 2026-10-02 | `b739dc5881` S6c | plain i8 KV SG8. S7 성능 지침 승인 |

---

## 5.9 이 방법론에서 뽑은 체크리스트 (스킬 `ocl-kernel-ab-methodology`가 이를 playbook으로 압축)

1. 변경 종류를 먼저 분류 (동작 보존 / 버그 수정 / 호스트-only / 성능 / 신기능) -> 5.2.6의 증명 레시피.
2. 가설은 한 줄로, 예측값을 수치로 적고, **양쪽 통제**를 설계한 뒤 실행 (한 번에 한 변수).
3. 음성 대조군을 수정 **전에** 돌려서 예측과 현실의 차이를 먼저 본다.
4. 장비 확인: `--device_suffix`, 바이너리, Debug/Release, `built`로 임베딩 확인, 클럭 핀.
5. 오라클: 첫 레이어, dtype ULP, bit-preserving vs order-changing 분류, NaN-poison, 비퇴행 데이터 (logit spread 확인).
6. 성능 주장: 지표(kernel device time vs e2e), 핀 클럭, MIN/median, 같은 소스/빌드의 대응 arm, measured/assumed 표기, 하드웨어 이동 후 재측정.
7. 종결: 메모리 갱신 + 재개 프롬프트 + 한 줄 커밋 메시지 제안 (커밋은 사용자).
