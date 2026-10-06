# ocl_perf_guide 문서·스킬 검증 결과

검증 일시: 2026-10-06
검증 대상: `src/plugins/intel_gpu/docs/ocl_perf_guide/` (README + 01~05장) 및 `.claude/skills/` 7개 스킬
기준 코드: 브랜치 `sdpa_ocl_dpas`, HEAD `b739dc5881` (작업 트리)
검증 방법: 코드와의 라인 대조 / 수치 재계산 / 문서 간 일관성 / 참조 존재 확인

---

## 1. 정확한 것 (신뢰 가능)

| 항목 | 검증 결과 |
|---|---|
| 모든 `.cl`/`.cpp`/`.hpp` 파일 존재 | `sdpa_ocl*.cl` 6개, `sdpa/*.cpp|hpp` 19개 전부 존재 |
| `sdpa_ocl_config.cl:12-17` (`SG8`, `DPAS_A_T`), `:36` (DPAS_MAD_K16) | 일치 |
| `sdpa_ocl.cl:26-27` (reqd_work_group_size), `:118-128` (sg_i/sg_j), `:300-309` (base fixup), `:314-317` (SLM 선언), `:763` (atomic max), `:780-831` (softmax), `:842-868` (split barrier + alpha select chain), `:891` (SV_TRIM), `:910-920` (pA block_read8), `:1030-1035` (A 2D write) | 전부 일치 |
| `sdpa_ocl_decode.cl:9-20` (operand mapping), `:547` (transpose_32b), `:727-734` (split barrier + prefetch) | 일치 |
| `solve_sv_split` (alpha nesting, widest-first) | 소스 재현 결과 문서 §4.1 표 6행 전부 **일치** (sv=(32,16,4,4) 등) |
| `choose_config_kq_only` 표 (≤32/64/128/256/512, xe_hpg 시드 16×16+sg4×2) | 코드와 일치 |
| SLM 산식 (`slm_bytes`) | 재계산: 17536/25856/35072/10624 — 문서의 21248만 불일치 (아래 §2-1) |
| `0x6480` 트릭 / u4 `0x6400\|n` | **256개 int8 전수 + u4 16개 전수 확인**, 예제 표(±5/0/−3/−128/127) 전부 정확 |
| `block2d_surface_ok`/`layout_ok`/`layout_fixup_ok`/`page_ok`/`axis_unpadded` | `sdpa_ocl_utils.hpp:52,63,69,112,134` 일치, 주석(rank-2 vacuous 등)도 일치 |
| 게이트 `block2d_io_allowed`(`:720`), fixup flag가 override 뒤 파생(`:741`) | 일치 |
| env 토글 (`SDPA_OCL_KQ_TILE_*`, `_PER_WG_*`, `TRACE_CONFIG`, `_256GRF`, `_DECODE_M/_256GRF/_PREFETCH`, `MICRO_MATH`, `NEG_SG8`) | 전부 존재, `NEG_SG8` 1..6 각각 코드에 존재(=1 qk_load:524/570, =2 :912, =3 :816, =4 :227, =5 :708, =6 config:147) |
| decode `get_sg_per_wg` `(v/16>=16)?16:8`(`:142-147`), `live_grf_estimate`+budget 112(`:68-100`) | 일치 |
| `kHpgTiersReady = PLAIN_F16_STATIC\|PLAIN_EXT\|PLAIN_I8` (`sdpa_ocl_hpg.hpp:33`), `TEMP(S9)` 2곳 (`sdpa_opt.cpp:86`, `paged_attention_opt.cpp:1623`) | 일치 |
| 술어 분리 (`sdpa_ocl_arch_ok`, `sdpa_ocl_hpg_enabled`, `sdpa_ocl_decode_reader_available`, `sdpa_ocl_selected`) | `paged_attention.hpp:178-189`에 실제 존재, `expected_dpas_backend_for`는 테스트 헬퍼에 존재 |
| `block2d_layout_ok` 주석의 head-72 B70 측정 (base 4/16/32/48B 정상, 2B 어긋남 오답) | 코드 주석과 일치 |
| 스크립트 (`test/sdpa_ocl_ab.py`, `sdpa_ocl_gtests.sh`, `splice_*.sh`, `isa_ab_*.sh`, `kvup_splice.sh`, `dump_isa*.sh`, `run.sh`, `check_*_offsets.py`, `probe_*.cpp`, `S2_RESULTS.md`, `S0_RESULTS.md`) | 전부 존재 |
| `docs/sdpa_ocl.md`의 인용 절 ("xe_hpg bring-up", "Kernel flow", "Paged-attention cache layouts", "Known issues", "Performance opportunities") | 전부 존재 |
| 04장의 `r[a0]` 2→274/1.8x, store sector 표(2201/723/423/3353, 1283/889/2014/3700), `//.private memory size` + `//.spill size` 헤더 가이드 | 일치 |
| 02장 §9.2의 `load.ugm.d32x8t (1\|M0)` 해석 ("block load, not scalar gather") | 이 프로젝트의 결론(블록 로드)과 일치 — 훌륭한 정정 기록 |

---

## 2. 발견된 문제

### 중대 (수정 권장)

| # | 위치 | 문제 | 수정안 |
|---|---|---|---|
| 1 | 01장 §4.1 표 | "tile_q=16, pwk=2, pwq=4, sg=8, wgTK=32, **SLM 21248**" 행 — 재계산 시 **10624** B (Q_slm=4096 + S_slm=2048 + S_sum=1024 + S_max=128). 다른 행(25856/35072)과 코드 공식에서는 전부 일치하므로 이 행만 오타 | `21248` → `10624` |
| 2 | 01장 §6.1 표 vs 04장 §4.11-3 표 | 같은 k16pwg4/k32wg 데이터를 두 장이 서로 다른 결론으로 표기 (01장: "무효 config" 경고, 04장: "micro보다 빠름" 주석 없이 나열). 04장 표는 헤더에서 `pwk` 열을 생략해 sg=8 행이 pwk4임을 알기 어려움. 독자가 어느 것이 유효한지 판단 불가 | 04장 표에 `pwk` 열 추가 + "6.1 유효성 의문" 경고 반영. 두 장의 결론을 상호 참조 |
| 3 | 05장 §5.7.4 | `SDPA_OCL_NEG_SG8=1..8`로 표기 — 코드는 **1..6**만 처리 (=7/8은 no-op → 음성 대조 무력). 01장·스킬은 1..6로 정확 | `1..8` → `1..6` |

### 경미 (명확화 권장)

| # | 위치 | 문제 | 수정안 |
|---|---|---|---|
| 4 | 01장 §7.1 | "(`sdpa_gen_ocl.cpp:157-164` 주석 참조)" — 현재 `:155-166`에 해당 주석 없음. 01장 §14는 stale로 명시하면서 본문에서 또 인용 (self-contradiction). 04장 §4.15에 정확한 새 위치(`:1038-1041`) 존재 | 본문 인용을 `:1038-1041`로 정정 |
| 5 | 04장 §4.4 | "`send`는 barrier/fence/EOT에만 보인다" — SLM barrier/atomic은 실제로 LSC `load.slm`/`store.slm`으로 컴파일되며 §4.4.1이 랜드마크로 "SLM barrier `send`"를 사용하는 것과 미묘하게 충돌 | "send는 LSC가 아닌 opcode에만 보인다" 식으로 명확화 |
| 6 | 02장 | VTune 기반 `16r16x2c` V-read 병합 (521.01 vs 524.34 ms, -0.64%, 04장 §4.1.1·05장 §5.5.3)이 02장 카탈로그에 **누락** | 02장에 실험 추가 |
| 7 | 스킬 `xe-spill-isa-profiling` §9 | DG2 ocloc spill (h64 7,968 B 등) 값은 05장 §5.7.2의 `s6a/pass_real/results.tsv`와 일치하나 출처 파일명이 스킬에 없음 | 출처 표기 추가 |

### 잠재적 혼란 (정보만 제공)

| # | 위치 | 내용 |
|---|---|---|
| 8 | 01장 §7.2 vs §8.1 | decode에 256GRF가 유용한 경우(head512 sg16 = 동급)가 04장 §4.5.2에만 있고 스킬에는 없어, decode+256GRF 판단이 어려움. 필요 시 04장 행을 스킬에 추가 |

---

## 3. 결론

- **전체적으로 신뢰도가 매우 높다**: 인용된 코드 위치, 게이트 로직, env 토글, 수치, 산식, 스크립트가 현재 트리와 정확히 일치. 0x6480 유도와 SLM 산식, solve_sv_split은 수치 재현까지 확인됨.
- **수정 권장 3건**: ① SLM 21248→10624, ② 04장 §4.11-3 표 헤더에 pwk 열 추가 + "6.1 무효" 경고 반영, ③ 05장 `NEG_SG8=1..8`→`1..6`.
- **명확화 권장 4건**: ④ stale 주석 인용 정정, ⑤ "send" 표기, ⑥ 02장에 16r16x2c V-read 병합 실험 추가, ⑦ 스킬 DG2 spill 출처 표기.
- **주의**: 검증 기준은 2026-10-06 작업 트리. 커밋이 이동하면 라인 번호가 다시 어긋나므로(문서가 이미 경고), 사용 시 심볼 이름으로 재검색할 것.
---

## 4. 반영 결과 (재검증 후 처리)

| # | 처분 | 근거 |
|---|---|---|
| 1 | **반려 (문서가 맞음)** | 해당 행은 SG16(Xe2), d_max=128, wgTQ=64, wgTK=32: q_slm=4096 + s_slm=32·64/2=1024 + s_sum=64·2=128 + s_max=64 = 5312 dword = **21248 B** (`slm_bytes`). 리뷰의 10624는 s_slm/s_sum을 잘못 계산. 25856 행도 같은 식으로 재현됨. |
| 2 | 반영 | 04장 §4.11-3 표에 `pwk` 열과 "유효성 의문" 경고·01장 §6.1/§6.2 상호 참조 추가 |
| 3 | 반영 | 05장 `NEG_SG8=1..8` → `1..6` (코드는 1..6만 처리) |
| 4 | 반영 | 인용 위치는 01장 §7.1이 아니라 02장 §3.2였음; `:1038-1041`(`get_build_options` 256GRF 주석)로 정정 |
| 5 | 반영 | 04장 §4.4 "send" 표현 명확화 |
| 6 | 반영 | 02장 §12 표에 #28 (소폭 이득 사례) 추가 |
| 7 | 반영 | `xe-spill-isa-profiling` 스킬에 출처 표기 |
| 8 | 미반영 | 정보성; 요청 범위 밖 |
