---
name: xe-visa-inline-asm
description: Playbook for writing and validating inline vISA assembly (__asm__ with .decl/alias, lsc_load, dpas/dpasw, predicates/labels) inside Intel Xe-HPG OpenCL kernels when IGC sinks loads or scatters packing - syntax, verified patterns, verifier limits, 16 failure modes found on DG2, and the accuracy protocol. Use when considering or reviewing native/asm in sdpa_ocl experiments.
---

# Xe 인라인 vISA 어셈블리

상세: `src/plugins/intel_gpu/docs/ocl_perf_guide/08-inline-visa-asm.md`. 소스 예: `test/sdpa_ocl_xe_hpg/s7a/perf/opt/short_diag_20261003/raw_stage/gate3_asm_first/raw_h256_qslmahead.cl` (untracked, standalone 실험용; 제품 `sdpa_ocl.cl`에는 미통합).

## 언제
병목이 **스케줄/메시지 모양/패킹 명령**임을 ablation과 ISA로 확인한 뒤에만. (IGC가 load를 `dpas` 직전으로 sink, VNNI pack이 operand당 ~80 mov, 같은 소스가 ±40% 변동.) 병목이 알고리즘·대역폭·pre-pass면 asm은 도움이 안 된다. 사용자 정책: 성능 목표를 먼저 맞추고 유지보수 리뷰에서 OpenCL C 전환 판단.

## 문법 요약
```c
__asm__ volatile("{\n"
  ".decl PW v_type=G type=uw num_elts=256 alias=<%0,0>\n"     // C 변수 %0의 byte offset 0을 uw로 재해석
  ".decl P0 v_type=G type=ud num_elts=64 align=wordx32\n"      // GRF 정렬 가상 레지스터
  ".decl PRED v_type=P num_elts=1\n"
  "mov (M1_NM,8) PW(0,0)<2> XW(0,0)<16;8,2>\n"                 // region <vstride;width,hstride>
  "lsc_load.ugm.ca.ca (M1,8) %9:d32x8 flat[%13]:a64\n"          // global per-lane 주소
  "lsc_load.slm (M1_NM,1) P0:d32x64t flat[%12+0]:a32\n"         // SLM block transposed
  "dpas.hf.hf.8.8 (M1,8) %0.0 %0.0 P0.0 PD0(0,0)\n"             // dpasw는 dpasw.hf.hf.8.8
  "cmp.ne (M1_NM,1) PRED %7(0,0)<0;1,0> 0:ud\n(PRED) jmp (M1_NM,1) LBL\n" "LBL:\n"
  "}\n" : "=rw"(out) : "rw"(in) /* 누산기는 "+rw" */);
```
첫 depth의 accumulator는 `%null.0`(C 문자열에서 `%%null.0`)로 zero-init mov 제거.

## 깨졌던 것 (재발 방지)
1. subgroup-varying 값(SLM 주소)을 `rw.u`(uniform)로 → 오답. varying은 `rw`. 오답 시 asm을 기능별로 쪼개 하나씩 빼며 분리. `memory` clobber는 고치지 못함.
2. 64-bit `uq` ADD → low word만 scalar로 내려 carry 누락(오답). `addc` 또는 C에서 선계산.
3. 원인 불명의 payload-lowering 오답(h272 상위 band): 일반화 금지, *없는* band를 uniform native branch로 skip하는 우회만 채택.
4. OpenCL C의 barrier/SLM 주변 guard가 컴파일러 재배치로 NaN(seq257에서만). guard는 **asm 안 native uniform branch**.
5. KQ early-exit/skip이 일부 lowering에서 NaN → 금지, mask 유지 + 모든 raw load clamp.
6. producer(WGK)와 consumer(SV 한계) bound 불일치 → 쓰이지 않은 SLM 읽기. `min(WGK, causal_k-k0)`.
7. 라벨 있는 asm을 두 번 인라인 → verifier 중복 라벨 오류(라벨명 변경으로도 해결 안 된 사례). 호출당 1회.
8. **inline verifier는 micro의 nGEN보다 좁다**: `d32x128t`/`d64x64t`(512 B) "more than 8 registers", gather d64x8, ExecSize8 transposed gather, mov stride 8(허용 0/1/2/4) 거부. micro ISA에 있어도 inline에서 안 될 수 있음 → 한 줄짜리 커널로 먼저 컴파일.
9. raw payload는 GRF 정렬(alias/`align=wordx32`). 문법은 최소 커널로 먼저 확인(`dpas.w.`는 오류).
10. partial tile에서 8행 묶음 원점 이동(`min(k0+sg*8,k-8)`)은 유효 key 위치까지 바꿈 → full tile만 native.
11. head 정렬 의존: group-start clamp는 head가 V tile 폭에 정렬될 때만 안전(h48 FAIL). 홀수 depth(h80 5 depth)는 `/2` floor로 마지막 tile 누락.
12. 첫 operand 소스 매핑 실수가 가장 흔함 → 폐기, correct 파일만 유효.

## 효과가 있었던 패턴
d64×4 gather + uq register transpose(native grouping이 lowering을 바꿈, 138.6→129.3 µs), whole-KQ/whole-SV 단일 asm, clamped whole-SV(분기 제거), V 주소 대수 단순화(+4.06→+1.70%), null-src0, native uniform skip(padded value group).

## 검증 프로토콜
1. 새 메시지/region은 최소 커널에서 verifier 통과 확인.  2. NaN poison + CPU double **all-rows**, tolerance 불변(`maxabs<1e-2`).  3. 수학 불변 변경은 이전 변형과 **full-output bit-identical**.  4. 경계 길이(1,16,31–33,63–65,127–129,255–257,511–513,1025,1033)·partial band·홀수 depth·다중 subsequence·sharp 입력.  5. spill=0, ISA send/dpas 개수 확인.  6. 한 번에 한 가지.  7. 정책/소스 SHA256을 측정 전에 동결.  8. 실패 시간은 폐기하고 로그만 보존.  9. 통합 리뷰: virtual register alias byte pitch/operand shape, `M1_NM` mask, barrier 순서, clobber/liveness, IGC 버전 의존. SG16/Xe2 소스는 `#if SG8`로 격리하고 `sdpa_ocl_ab.py l0`로 byte-identity 확인.
