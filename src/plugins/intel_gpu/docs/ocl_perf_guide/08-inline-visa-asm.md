# 08. 인라인 vISA 어셈블리: 언제, 어떻게, 무엇이 깨지는가

06장의 DG2 raw-kernel 게이트는 OpenCL C만으로는 도달하지 못했다. IGC가 로드를 사용 직전으로 sink하고, operand 패킹을 word-mov로 풀어내고, 같은 의미의 소스 변형이 ±40% 흔들렸기 때문이다. 최종 후보들은 DPAS·LSC 로드·register 패킹·SLM 쓰기를 **vISA inline assembly**로 묶어 메시지와 순서를 직접 지정하고, 물리 GRF 배치만 IGC에 맡겼다. 이 장은 그 방법과, 하면서 실제로 깨졌던 것들을 기록한다.

> 범위: 이 asm은 DG2 standalone 실험 커널(`test/sdpa_ocl_xe_hpg/s7a/perf/opt/short_diag_20261003/raw_stage/`)의 것이다. 제품 `sdpa_ocl.cl`에는 통합하지 않았다. 사용자 결정은 "성능 목표를 먼저 맞추고, 그 뒤 유지보수 리뷰에서 OpenCL C/intrinsic 전환을 판단"이었다. 따라서 아래 "깨지는 것" 목록은 *통합 전 리뷰 체크리스트*로도 읽어야 한다.

## 8.1 asm이 필요했던 이유 (증거)

| 증거 | 출처 |
|---|---|
| OpenCL C `block_read`/gather → IGC가 loads를 DPAS 직전으로 sink: `loads → sync.allwr → dpas`가 step마다 반복, step 간 중첩 0 | ISA 판독 (07장 §7.4) |
| `-cl-intel-no-prera-scheduling`, `__asm__ volatile("":::"memory")`로는 스케줄을 못 바꿈 (전자는 ISA 동일, 후자는 block read를 scalar로 풀고 spill) | HANDOFF §5 |
| sdpa_micro는 fused nGEN native: global-next load → SLM operand → DPAS → next 재배열 → SLM write, 별도 K'/V' 없음 | micro final ISA 분석 |
| 순수 OpenCL로 만든 raw-only 후보 최선이 micro보다 ~10–30% 느렸고, native 패킹+load 순서 통제 후 격차가 닫힘 | 06장 §6.3 |
| VNNI 패킹의 word-mov는 operand당 ~80 명령(OpenCL), native `mov <2>:uw` 두 개로 축소 가능 | ISA 비교 |

판단 기준: **측정으로 "스케줄/메시지/패킹 명령"이 병목임을 확인한 뒤에만** asm을 쓴다(소스 변형 ablation + ISA 확인). 병목이 memory 대역폭이나 알고리즘 구조라면 asm이 도와주지 않는다.

## 8.2 문법과 관용구 (실제 소스에서)

### 구조: `__asm__ volatile("{ ... }" : outputs : inputs)`

```c
SDPA_OCL_INLINE uint16 raw_v_pair_pack(uint8 x, uint8 y) {
    uint16 p;
    __asm__ volatile("{\n"
        ".decl PW v_type=G type=uw num_elts=256 alias=<%0,0>\n"
        ".decl XW v_type=G type=uw num_elts=128 alias=<%1,0>\n"
        ".decl YW v_type=G type=uw num_elts=128 alias=<%2,0>\n"
        "mov (M1_NM,8) PW(0,0)<2> XW(0,0)<16;8,2>\n"      // 두 token의 halfword를 한 dword에 interleave
        "mov (M1_NM,8) PW(0,1)<2> YW(0,0)<16;8,2>\n"
        /* ... 16 pairs ... */
        "}\n" : "=rw"(p) : "rw"(x), "rw"(y));
    return p;
}
```

- `.decl NAME v_type=G type=<ud|uw|uq|f|hf...> num_elts=N [align=wordx32] [alias=<%k,offset>]`: 가상 레지스터 선언. `alias=<%k,off>`로 C 변수(`%k`)의 바이트 offset을 다른 type으로 *재해석*한다. 이것이 VNNI/transposed 재배열의 핵심 도구다.
- 지역 `.decl`은 `{ }` 블록 안에서만 유효하다. `align=wordx32`는 GRF 정렬(raw operand를 payload로 쓸 때 필요).
- 제약자: `"=rw"` 출력, `"rw"` 입력, `"+rw"` 입출력(누산기). 값은 가상 레지스터로 들어가며 물리 GRF는 IGC가 할당한다.
- region 문법 `<vstride;width,hstride>`: `XW(0,0)<16;8,2>`는 stride 2 halfword 8개를 8 lane에 뿌린다. `(M1_NM,8)`은 execution size 8, mask 무시(NM).
- predicate: `.decl P v_type=P num_elts=1` + `cmp.ne (M1_NM,1) P ...` + `(P) jmp (M1_NM,1) LABEL`. 라벨(`KQWORK:`)도 asm 안에 직접 쓴다. **uniform branch를 asm 안에서 쓰는 것이 OpenCL C 분기보다 안전했다**(§8.4-4).

### 메모리/DPAS 명령

```
lsc_load.ugm.ca.ca (M1,8)  %9:d32x8  flat[%13]:a64           // global, 행 8개 dword 벡터 (per-lane 주소)
lsc_load.slm       (M1_NM,1) P0:d32x64t flat[%12+0]:a32       // SLM block, transposed(t), 256 B
dpas.hf.hf.8.8     (M1,8) %0.0 %0.0 P0.0 PD0(0,0)             // C = C + A*B, depth 8 repeat 8
dpasw.hf.hf.8.8    ...                                        // "dpas.w."는 문법 오류
```

## 8.3 검증된 사용 패턴

1. **KQ 전체를 한 asm 블록으로**: depth 반복(8×8 DPAS 묶음)과 next-K 로드, 레지스터 transpose를 한 블록에 둔다 (`raw_kq_original_whole`). chunk별 asm → whole-KQ asm으로 179.5 → 171.6 µs(seq512, 초기 raw 후보 기준).
2. **SV 단계 `raw_sv_step`**: P를 SLM에서 `d32x64t` transposed load → 8개 DPAS와 next V 로드(`RAW_SV_PIPE`로 위치를 토글) → next uw 패킹.
3. **K d64×4 gather** (`kv_k_d64`): K를 `d32x8` 대신 `d64x4` gather + uq register transpose로 읽어 *같은* 32 B/lane, 16K×32Q DPAS를 유지하면서 컴파일러가 transpose를 잘게 쪼개는 것을 막았다. seq512 138.6 → 129.3 µs(micro 136.7). 이 한 변경이 raw-only 후보를 micro 근처로 끌어올린 전환점이었다. **native 메시지 grouping이 compiler lowering을 바꾼다**는 증거다.
4. **clamped whole-SV**: 부분 tile의 SV tail을 OpenCL 분기로 두지 않고, 유효 행으로 주소를 clamp(`min(addr, last_valid)`)한 뒤 항상 전체 DPAS를 실행하고 mask(P=0)로 무효 key를 0 기여시킨다. h64 seq1033 base 146.3 → 134.4 µs, h96 seq2048 323.7 → 232.0 µs(micro 363.7)로 크게 개선. 효과는 tail 반복 수만이 아니라 **분기와 operand marshal 코드 생성 자체**였다(분리해서 증명하지 못했으므로 "tail 반복 감소 때문"이라고 쓰지 않는다).
5. **PV 주소 대수**: 16개 token마다 clamp·stride 곱을 계산하던 것을 공통 column/first/last byte 주소 + 고정 offset의 unsigned min으로 바꿨다(`h128_q16wide_address`). asm/DPAS/geometry/barrier/math 불변인 *주소 계산만의* 변경으로 h128 2/2 seq256 gap의 +4.06% → +1.70%.

## 8.4 실제로 깨졌던 것들 (재발 방지 체크리스트)

| # | 증상 | 원인 | 규칙 |
|---|---|---|---|
| 1 | SLM 주소를 `"rw.u"`(uniform 강제)로 넘기자 오답(maxabs 크게). `asm_mad` 단독은 PASS, `asm_slm_read` 단독은 FAIL로 분리 | subgroup마다 다른 SLM 주소를 uniform으로 선언 | **subgroup-varying 값은 `rw`**. 의심되면 asm 묶음을 기능별로 쪼개 한 개씩 넣고 빼서 오답을 분리한다. memory clobber는 이 오답을 고치지 못했다 |
| 2 | 64-bit 주소 `uq` ADD가 오답(maxabs 1.854) | native ISA가 varying 64-bit ADD의 low word를 scalar 형태로 내림 — carry 처리 누락 | 64-bit 주소 가산은 `addc`(carry low/high)를 쓰거나 C에서 미리 계산한다. "explicit ud strided low-word ADD"는 carry 미처리라 제품 후보 불가 |
| 3 | head-272의 vector 주소 계산을 바꾸자 상위 16-channel band만 틀림(maxabs 2.432) | 정확한 IGC payload-lowering 원인은 **미규명**. Q-only 대조는 FAIL, 고정 `+32` 상위 load는 PASS | 원인이 불명인 변형은 일반화하지 않는다. 원래 vector pointer를 유지하고 *없는* upper band만 uniform native branch로 skip하는 `*_exact_v2`가 PASS. 옛 `>256 *_exact`는 폐기 |
| 4 | OpenCL C의 outer SV skip guard가 seq 257에서 NaN/Inf 45개 (`raw_h80_vactive`), seq256은 PASS | barrier/SLM 주변의 컴파일러-제어 control flow 재배치 | 동기화가 있는 구간의 guard는 **asm 안의 native uniform branch**로 둔다 (`raw_h80_vactive_native`). 한 길이 PASS로 후보를 되살리지 않는다 |
| 5 | KQ early-exit(skip)을 켜면 NaN (tiny64, clamped wholeSV) | 이 lowering에서 KQ skip이 unsafe | `KQ skip` 금지. mask를 유지하고 모든 raw load를 clamp한다. 같은 이유로 `RAW_KQ_TAIL`은 OFF |
| 6 | key8로 KQ SG 폭을 줄인 후보에서 SV가 NaN/Inf (1008개, seq255) | WGK 256→128로 줄었는데 SV helper가 `causal_k-k0>128`이면 cp8+를 읽어 **쓰이지 않은 P-SLM**을 읽음 | producer와 consumer의 tile bound를 같은 값으로 전달: `SV_limit = min(WGK, causal_k-k0)` |
| 7 | 같은 asm 블록을 두 번 인라인하면 컴파일 FAIL | vISA verifier가 **중복 라벨**을 거부(라벨 이름을 바꿔도 같은 증상이 있었음) | 라벨이 있는 asm은 호출 위치당 한 번만 둔다. 라벨 없이 컴파일되게(tail jump 제거) 구성하거나 호출을 하나로 모은다 |
| 8 | `lsc_load.slm d32x128t` / `d64x64t`(micro native ISA에 존재하는 512 B load) 컴파일 FAIL: "this message accesses more than 8 registers" | **inline asm verifier의 허용 집합이 micro의 nGEN emitter보다 좁음** | micro final ISA에 있다고 inline asm에서 되는 것은 아니다. 허용 메시지를 먼저 컴파일해 확인한다 |
| 9 | `gather d64x8` 거부(8 GRF message limit), ExecSize 8 transposed gather(`requires 1`), strided mov stride 8 거부(허용 0/1/2/4) | 같은 verifier 제약 | 에러 로그는 `*_inline_errors.txt`로 보존하고 같은 형태를 재시도하지 않는다 |
| 10 | raw payload operand가 GRF 정렬이 아니라 오답 | scalar raw-address payload는 GRF aligned여야 함 | payload는 `align=wordx32` decl 또는 alias로 만든다 |
| 11 | `dpas.w.hf...` 문법 FAIL | 올바른 split-matrix 문법은 `dpasw.hf.hf.8.8` | 새 명령은 한 줄짜리 최소 커널로 문법부터 확인 |
| 12 | 첫 DPAS의 accumulator 초기화 `mov`가 64개 | zero-init을 별도 mov로 | 첫 depth의 `src0`에 `%null.0`(C 문자열에서는 `%%null.0`)을 쓰면 mov를 제거(142.7→140.6 µs, 같은 누산 순서) |
| 13 | 처음 만든 nativepipe가 오답 + `memory` clobber가 효과 없음 | 위 #1(uniform 선언) | clobber를 고친 약으로 쓰지 않는다. 오답 시간은 모두 무효 |
| 14 | partial tile에서 `min(k0+sg*8, k-8)`로 8행 묶음의 *원점을 이동*하면 seq 129/511/513 FAIL | 끝 8행 묶음 원점 이동이 유효 key 위치까지 바꿈 | full tile만 native로, partial tile은 검증된 다른 경로로 처리하는 `k16_native_safe` |
| 15 | h48에서 tiny V32의 "group start clamp"가 유효 channel을 shift (seq36 maxabs 3.18) | group start clamp는 head가 V tile 폭에 정렬될 때만 안전 | h48은 V16 tiny(4 SG, 1개 padded value group)로 |
| 16 | `K` operand 소스 mapping을 잘못 지정(첫 DPAS operand) | 수작업 asm의 가장 흔한 실수 | 초기 파일은 폐기하고 correct 파일만 유효. DPAS operand 매핑은 01장을 먼저 확인 |

## 8.5 asm 변경의 검증 프로토콜

1. **최소 단위로 나눠 컴파일**: 새 명령/메시지/region은 한 개짜리 커널에서 문법과 verifier를 통과시킨 뒤 큰 블록에 넣는다.
2. **정확도 먼저**: NaN poison + CPU double all-rows. 통과 후 이전 OCL 변형과 **full-output bit-identical**을 확인한다(수학/누산 순서를 안 바꾼 변경은 bit-identical이어야 한다). 순서가 달라지는 변경(key8)은 halfword 몇 개가 달라질 수 있고, 이를 허용치 안에서만 받아들인다(반드시 기록).
3. **경계 길이를 먼저 친다**: partial tile 길이(예: 1, 16, 31, 32, 33, 64, 65, 97, 127, 129, 255, 257, 511, 513, 1025, 1033), 마지막 head band, 홀수 depth(head80의 5 depth는 `DKS_ACTIVE/2` floor에서 마지막 tile을 놓쳐 FAIL했다), 여러 subsequence.
4. **spill=0 / ISA diff**: asm 블록을 바꾸면 IGC 할당이 흔들린다. 항상 spill과 send/dpas 개수를 본다.
5. **한 변경씩**: asm 묶음을 크게 바꾼 결과를 단일 원인으로 해석하지 않는다. 구조 변경을 많이 결합한 후보의 시간은 "총효과"로만 기록한다.
6. **fixed policy를 먼저 고정하고 측정**: 측정 뒤에 잘 나온 후보를 고르는 cherry-pick을 막기 위해 policy/소스 SHA256을 측정 전에 동결한다(09장 §9.5).
7. **IGC 버전·할당 의존성**: 물리 GRF 할당과 스케줄은 IGC가 정한다. 같은 asm도 IGC 버전이 바뀌면 시간이 달라질 수 있다. 통합 후 리뷰 항목: virtual register alias의 byte pitch/operand shape, subgroup mask(`M1_NM`), barrier 순서, partial head/unused band/multi-iteration 경로, clobber/liveness.

## 8.6 asm을 쓰지 않아야 하는 경우

- 병목이 대역폭/알고리즘(예: 불필요한 pre-pass, 잘못된 tile)일 때. 먼저 06장의 root cause 절차로 구조를 고친다.
- 한 번의 예쁜 성능만 보고 결정할 때. asm·native 후보 상당수(`v/` 약 224개 변형과 `raw_stage/` 후보 포함)가 정확도 또는 성능으로 폐기됐다.
- 다른 아키텍처(SG16/Xe2)와 byte-identical을 유지해야 하는 공유 소스일 때. 해당 asm은 `#if SG8` 안에 격리하고 `python3 test/sdpa_ocl_ab.py l0 --base <commit>`로 SG16 ISA 동일성을 확인한다.
