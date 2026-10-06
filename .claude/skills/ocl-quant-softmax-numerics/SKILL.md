---
name: ocl-quant-softmax-numerics
description: Playbook for numerics in Intel Xe DPAS OpenCL attention kernels - online softmax (exp2, alpha, -inf/NaN guards), causal/window/bidir/sink masks, int8/u4 to f16 dequant via the 0x6480 XOR trick, scale/zp factoring, bf16, accuracy verification. Use for sdpa_ocl, KV-cache quantization, WWB/accuracy regressions.
---

# OpenCL 어텐션 커널 수치 설계 / 검증 플레이북

깊이 있는 설명과 근거(path:line, 측정값, 하드웨어)는 `src/plugins/intel_gpu/docs/ocl_perf_guide/03-numerics-softmax-quantization.md`.
DPAS operand 역할은 `01-dpas-and-tiling.md`, ISA/spill 분석은 `04-spill-isa-profiling.md`, 방법론은 `05-methodology-and-pitfalls.md`.
저장소 규칙(AGENTS.md): 수치 정확성 > 성능, 측정 없는 성능 주장 금지, 빌드/실행은 사용자가 한다(명령 블록만 제공), NaN/Inf/dynamic shape/accumulator/양자화를 항상 점검.

## 1. 작업 시작 전 분류 (먼저 답할 것)

1. 이 변경은 **bit-preserving** 인가 **order-changing**(누적/타일 순서 변경)인가? 후자는 end-to-end 지표(WWB)를 움직여도 버그 증거가 아니다.
2. 어떤 operand 에 head dim 이 lane 인가 (prefill KQ: A=K,B=Q / decode KQ: A=Q,B=K)? dequant 항등식은 **자기 mapping 으로 다시 유도**한다 (다른 커널 복사 금지).
3. 데이터 경로: plain SDPA(zp 정수 i8) / PA 캐시(zp 비정수, writer 가 1/scale 저장) / u4 / bf16 중 무엇인가?

## 2. Online softmax 체크리스트

- [ ] acc 는 f32 (DPAS `float8`, max/sum f32). P 만 f16/bf16 으로 SLM 저장, 분모는 반올림 전 f32 합.
- [ ] `scale' = scale * LOG2E` 를 한 번 접고 `exp2(S*scale' - m*scale')`. S 와 max 는 raw 도메인 유지 → 마스크/sink 는 `* iscale` (= 1/scale) 로 얹는다.
- [ ] `ok = isfinite(m_new)`; `!ok` 이면 exp_tile=0, alpha=1, max 상태 유지 (`-inf - -inf = NaN` 방지).
- [ ] epilogue: `inv_l = (l > 0) ? recip(l) : 0` (fully-masked 행 = 0, NaN 아님).
- [ ] 루프 경계(causal_k, window_k0_begin)를 바꾸면 `first`/`last` 도 같이. window 하한은 k0 타일 경계로 **내림**.
- [ ] 런타임 인덱스 private 배열(alpha[])은 scratch 로 간다 → 컴파일 타임 select chain.
- [ ] decode: 점수에 scale 미리 곱함(Q pre-scaled), mask 값은 `VAL_MIN`(−INF 아님: partition 전체 마스크 시 `MIN-MIN=0`), `exp_sums/max_logits` 는 partition 자기 max 기준. partition 수/직접 출력 분기는 finalization 의 `effective_seq_len` 과 정확히 상보.

## 3. 마스크 체크리스트

- [ ] causal 상한 `causal_k` + block skip(`causal_block_clear`) + per-element 술어가 **같은 offset**(lower-right: `max(0,k-q)`, PA: `past_len`)을 쓰는가.
- [ ] 동적 마스크의 `MASK_KIND` 는 호스트 추정이다 → 런타임 `MSK_D2==1`/`MSK_D3==1` clamp 가 있는가 (없으면 OOB: Xe2 에서 `CL_OUT_OF_RESOURCES`).
- [ ] bidir(`token_type_ids`): 합집합 `(causal∩window) ∪ 자기 그룹`, query 쪽 `[gb,ge)` 만. 좌표: `token_type_ids` = LOCAL, 나머지 = KEY(`+query_position_offset`). 빈 텐서(`count==0`)는 접근 금지.
- [ ] sink: prefill 은 online 상태를 seed (`S_max=sink_raw`, `S_max_tile=sink_raw*scale`, `S_sum` 은 `sg_i_kq==0` 한 곳만 1). decode 는 partition 0 만. 도메인 차이 주의(prefill raw / decode pre-scaled). 테스트의 sink 값은 `log(kv_len)` 수준으로.
- [ ] 미기록 캐시 슬롯: `NaN + -inf = NaN` 이므로 로드에서 clamp/0 처리. V 의 `0*NaN` 도 주의(scale 과 zp 둘 다 0 강제).
- [ ] jit 매크로 / 커널 시그니처 / 호스트 arg push 3자 일치 (HAS_SINK_INPUT, HAS_QQ_BIAS 에서 arg shift 사고 전례).

## 4. int8/u4 → f16 widen 체크리스트

공식: `as_half(0x6480 ^ (ushort)(byte & 0xFF)) == (float)s + 1152` (s=signed int8).
유도: `0x6400` = 2^10(E=25), ulp=1 in [1024,2048); `0x6480` = 1024+128; `u ^ 0x80` = s+128 (부호→오프셋 이진); 합쳐서 1024+s+128.
예: s=−3 → 0xFD ^ 0x6480 = 0x647D → mantissa 125 → 1149 = 1152−3. s=−128 → 0x6400 = 1024, s=127 → 0x64FF = 1279. u4: `as_half(0x6400 | n) == 1024 + n`. u8: `0x6400 | u`.

- [ ] byte 추출은 shift+mask (`as_char4` 는 `:b` deinterleave 를 만든다). 4 byte → VNNI dword 2개를 dword 산술로 (`... ^ 0x64806480u`).
- [ ] bias 는 **zp 에 접어** `(wide - (zp+1152)) * scale` (분리된 bias 감산은 add 폭증).
- [ ] **정수 zp 에서만 exact** (f16 ulp@1152 = 1.0). PA 캐시/u4/int4 의 비정수 zp 에는 금지 → `(q - zp)*scale` half 명시 또는 decode 처럼 bias 를 **float 점수 보정**에서 제거.
- [ ] bias 를 되돌리는 항(`k_corr`)은 DPAS 가 본 **f16 반올림된** `Q*sc` 로 계산 (float 로 하면 오차 2.3x, 6e-3 임계 테스트로 안 걸림).
- [ ] bf16: 트릭 없음(ulp=1 구간이 [128,256) 뿐). float dequant → bf16 encode. `as_char((uchar)b)` 사용 (`(char)(uchar)b` 는 127 로 saturate). scale/zp 텐서는 f16 유지. 런타임 scale dtype 은 layout 에서 읽어 `SCALE_TO_FLOAT`(bf16 = `as_float((uint)x << 16)`).
- [ ] u4: K adjacent 패킹(writer WG 분할 때문에 강제), V split 패킹(lane==dim 유지). sdpa_ocl MIXED 의 K 는 depth 축 permute `win+2L+par` 를 Q staging 에 한 번만 지불. nibble 선택은 lane-uniform 이라 shift 량에 접힘.

## 5. dequant 항등식 (인수분해 위치)

| 항 | 접는 곳 |
|---|---|
| BY_TOKEN K sc/zp | 점수 후처리 per-lane (`sc*(S - (zp+bias)*sum_d Q)`), `sum_d Q` head 당 1회 reduce |
| BY_CHANNEL K sc | Q(A operand)에 곱 (lane==head dim) |
| BY_CHANNEL K zp | 점수에서 상수 `k_corr` 하나 감산 (페이지·head 당) |
| V sc | P(A operand)에 per-lane 곱. **분모는 sum(P) 그대로** |
| V zp | V B operand 에서 감산 (key 축을 따라 변해 유일하게 broadcast; 상수 lane 이면 source region 에 접혀 공짜) |

per-key scale/zp 는 깊이/키 루프 밖으로 hoist (루프 안에서는 SIMD-1 load 가 k0 당 128~256개).

## 6. "exact" MIXED (압축 캐시 + 현재 chunk)

캐시 `[0,past_len)` = 양자화 dequant, 현재 `[past_len,k)` = raw f16 Kc/Vc. k0 타일이 `past_len` 을 가로지르면 `k_chunk` 로 타일을 잘라 한 iteration 이 한 소스만 읽고, `k_chunk` 이후 행은 mask, `k0 += k_chunk`. 페이지 단위(GRAN=0) 분할은 오답.

## 7. 검증 플레이북 (실행은 사용자 — 명령 블록 제공, `--device_suffix=1` 필수)

1. **테스트 데이터부터 의심**: N(0,0.1) 은 softmax 가 거의 uniform → Q/K 오독이 숨는다. `logit_scale_gain=128`(상수 scale) / `runtime_scale_multiplier`(Xe2+) sharp 복사 케이스 추가.
   LCG: `InputGenerateData(start, range, res, seed)` 에서 `range*res` 가 2의 거듭제곱이고 ≤ 행 길이이면 모든 행이 동일 → res 를 31/37/1000 등으로, **LCG 를 파이썬으로 재생**해 logit spread 와 버그 효과를 먼저 확인. Q/K/V 같은 seed 금지.
2. **음성 대조를 옛 코드에서 먼저** 돌려 예측과 관측을 비교 (새 테스트가 버그 코드에서 FAIL 해야 의미 있음).
3. **layer-0 동일 입력 oracle**: 연쇄 네트워크는 첫 레이어에서만 비교. 출력 dtype ULP(f16 ≈ 4.9e-4 상대)와 대조: max ≈ 1 ULP 이면 정확. 이후 레이어는 카오스 증폭(1600x, plateau)이라 정상·버그 구분 불가. causal 격리 영역(토큰 `[0,tile)`)이 bit-identical 인지로 덤프 비교 가능성 자체검증.
4. **WWB 유사도는 커널 게이트가 아니다** (소형 모델 ±0.05 민감, minicpm4-0.5b −0.070 은 버그 아님으로 판정).
5. 토글 결과 해석 전 bit-preserving/order-changing 분류. 한쪽 대조만으로 귀속 금지 (예: head-512 실패는 sdpa_ocl 이 아니라 `dynamic_quantize` 256 상한 — `TEST_USE_SDPA_OCL=0` 대조를 먼저).
6. **실제 실행된 커널 확인**(cliloader / `get_kernels_dump_info()`): 게이트가 거절하면 정답인 다른 backend 로 조용히 빠져 초록이 증명이 아님. set-equality(FAIL→PASS 집합 = 게이트 허용 집합).
7. 오프셋/nibble/page 일치는 파이썬 사전 검증 (`test/check_by_channel_offsets.py`, `test/check_u4_offsets.py`, `test/check_u4_page_read.py`; 음성 대조 포함).
8. 정적 ISA(instCount/mov 수)는 시간이 아니다. 성능은 사용자 측정(cliloader per-kernel ns), B580/B70/DG2 구분 표기, 측정값 vs 가설 구분.

## 8. 자주 쓰는 함정 한 줄 요약

`-inf-(-inf)=NaN` / partition 전체 마스크는 VAL_MIN / 미기록 슬롯 NaN 은 마스크로 안 지워짐 / 비정수 zp 에 +1152 접기 금지 / k_corr f16 반올림 / `(char)(uchar)` saturate /
sink 중복 계상 / LOCAL vs KEY 좌표 / 동적 마스크 kind OOB / bf16 scale 오독 / dynamic_quantize 256 상한 / i8 BY_CHANNEL requantize 가 in_data_pitch 무시(미수정, `pa_kv_cache_update_ref.cl:297`).
전체 표(20항): 03 장 §10.
