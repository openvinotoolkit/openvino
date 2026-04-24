# Инструкция для агента: AArch64 INT8 JIT kernels для OpenVINO PR #33951

## Контекст

Работа ведётся над OpenVINO PR:

<https://github.com/openvinotoolkit/openvino/pull/33951>

Текущая ветка:

```text
allnes/openvino:an/conv-int8-jit
```

PR добавляет:

- AArch64 INT8 BRGEMM convolution executor;
- `Xbyak_aarch64` JIT INT8 kernels;
- packing paths;
- K-padding для MMLA-style execution;
- unit tests;
- `tools/int8_microbench`.

Основная задача агента — не просто “дописать JIT kernel”, а привести PR к архитектурно чистому, безопасному и максимально быстрому решению для **CPU-only AArch64 INT8 inference**.

---

## Главная цель

```text
Turn OpenVINO PR #33951 into a safe, ISA-dispatched, benchmark-proven AArch64 INT8 CPU implementation that can compete with the best existing ARM CPU libraries on clearly defined convolution/BRGEMM/GEMM shapes.
```

Решение должно быть:

- безопасным по ISA;
- без device-name hacks;
- корректным относительно reference C++ path;
- воспроизводимо benchmark-driven;
- reviewable для OpenVINO;
- разбитым на понятные коммиты;
- ориентированным на реальные CPU: Raspberry Pi 5, Graviton2/3/4, Apple M2, Samsung S25.

---

## Важная терминология

```text
Regular Xbyak is for x86/x64.
For AArch64 JIT code generation use Xbyak_aarch64.
```

Нельзя использовать обычный `Xbyak` для генерации ARM/AArch64 кода.

---

## Целевые устройства и CPU-paths

| Устройство / класс | CPU / ISA | Ожидаемый path |
|---|---|---|
| Raspberry Pi 5 | Cortex-A76, NEON + dotprod | `neon_dotprod`, `SDOT/UDOT` |
| AWS Graviton2 | Neoverse N1, NEON + dotprod | `neon_dotprod`, `SDOT/UDOT` |
| Apple M2 CPU | NEON + dotprod + I8MM | `neon_i8mm`, `SMMLA/UMMLA/USMMLA` |
| Samsung Galaxy S25 CPU | Snapdragon 8 Elite / Oryon, NEON + dotprod + I8MM | `neon_i8mm`, `SMMLA/UMMLA/USMMLA` |
| AWS Graviton3 / Graviton3E | SVE-capable server CPU | `sve_int8` / oneDNN BRGEMM |
| AWS Graviton4 | SVE2/SVE INT8-capable server CPU | `sve2_int8` / oneDNN BRGEMM |

Важно:

```text
Do not dispatch by device name.
Dispatch only by runtime ISA capabilities.
```

---

## Жёсткие границы задачи

### Нельзя

```text
Do not:
- rewrite the OpenVINO CPU plugin architecture;
- touch graph-level transformations;
- change public OpenVINO APIs;
- rewrite the ACL executor;
- add GPU/NPU/CoreML/Metal/Vulkan paths;
- add device-name-specific hacks;
- emit unsupported instructions without runtime feature checks;
- claim performance wins without reproducible benchmark data;
- expand convolution semantics while optimizing kernels;
- mix correctness refactoring, new ISA kernels, and benchmark tooling into one unreadable commit;
- use SIGILL probing as normal CPU feature detection;
- silently fall back without logging/diagnostics;
- make int8_microbench always built by default;
- remove existing safe fallback paths.
```

### Можно

```text
Allowed scope:
- AArch64 CPU plugin executor/kernels;
- INT8 convolution / BRGEMM / GEMM microkernel path;
- runtime ISA detection;
- kernel dispatch;
- packing;
- tail handling;
- microbenchmarks;
- comparison against state-of-the-art CPU libraries.
```

---

## Библиотеки-соперники для сравнения

Агент должен сравнивать результат не только с текущим OpenVINO fallback, но и с передовыми ARM CPU библиотеками, где это технически сопоставимо.

| Библиотека | Зачем сравнивать |
|---|---|
| Existing OpenVINO CPU path | Главный baseline: новый PR обязан доказать пользу относительно текущего поведения |
| oneDNN AArch64 / BRGEMM / ACL-integrated paths | Самый близкий baseline внутри экосистемы OpenVINO/oneDNN |
| Arm Compute Library | Оптимизированные Arm CPU kernels для Cortex-A/Neoverse |
| KleidiAI | Современные Arm AI micro-kernels |
| XNNPACK | Сильный mobile/server CPU inference baseline |

Важно:

```text
Do not claim “we beat library X” unless:
- same shape;
- same datatype;
- same layout or packing cost is accounted;
- same thread count;
- same batch size;
- same post-processing;
- same correctness target;
- reproducible benchmark numbers are provided.
```

---

## Требуемая архитектура

### 1. Единый слой AArch64 ISA detection

Нужно добавить или переиспользовать централизованный helper, который предоставляет ISA-фичи для всего AArch64 INT8 path.

Пример enum:

```cpp
enum class Aarch64Isa {
    scalar_reference,
    neon,
    neon_dotprod,
    neon_i8mm,
    sve,
    sve_i8mm,
    sve2_i8mm
};
```

Минимальные feature checks:

```text
- NEON / ASIMD
- ASIMDDP / dotprod
- I8MM
- SVE
- SVE2
- SVE INT8 / SVE I8MM if available
```

Linux/Android:

```text
Use getauxval(AT_HWCAP / AT_HWCAP2) where available.
```

macOS Apple Silicon:

```text
Use sysctlbyname for:
- hw.optional.arm.FEAT_DotProd
- hw.optional.arm.FEAT_I8MM
```

Правило безопасности:

```text
Never execute dotprod/I8MM/SVE instructions unless the corresponding runtime feature is detected.
```

---

### 2. Разделение kernel families

Нужны отдельные kernel families, а не один класс с кучей boolean-флагов.

```text
A. reference_cpp
   - exact int32 accumulation;
   - used for correctness and safe fallback only.

B. neon_dotprod
   - uses SDOT/UDOT;
   - target: Raspberry Pi 5, Graviton2;
   - must not emit I8MM instructions.

C. neon_i8mm
   - uses SMMLA/UMMLA/USMMLA as appropriate;
   - target: Apple M2, Samsung S25, I8MM-capable CPUs;
   - must not be a dotprod kernel renamed as I8MM.

D. sve_or_sve2_int8
   - target: Graviton3/Graviton4;
   - either use oneDNN BRGEMM path safely or implement vector-length-agnostic SVE/SVE2 kernel;
   - must not hardcode SVE vector length.
```

---

### 3. Runtime dispatch priority

```cpp
if (has_sve2_int8()) {
    use_sve2_int8_path();
} else if (has_sve_int8()) {
    use_sve_int8_path();
} else if (has_i8mm()) {
    use_neon_i8mm_path();
} else if (has_dotprod()) {
    use_neon_dotprod_path();
} else {
    use_reference_or_existing_safe_fallback();
}
```

Обязательное требование:

```text
Every selected path must be visible in debug/verbose benchmark output.
```

---

## Что изменить в текущем PR #33951

### 1. ISA detection

Сейчас локальные feature checks не должны оставаться главным механизмом выбора kernel.

Нужно:

```text
1. Move local has_asimd_dotprod() from brgemm_int8_kernel.cpp into a reusable AArch64 CPU feature detection helper.
2. Add detection for dotprod, I8MM, SVE, SVE2, SVE INT8 where available.
3. Use Linux/Android getauxval.
4. Use macOS sysctlbyname.
5. Add selected ISA logging.
6. Add ISA-safe tests where possible.
```

---

### 2. BRGEMM ISA selection

Нельзя делать implicit SVE fallback.

Нужно:

```text
- Only use oneDNN AArch64 BRGEMM with ISA that mayiuse() confirms.
- If SVE BRGEMM is unavailable, fall back to custom NEON dotprod/I8MM kernels.
- Do not choose sve_128/sve_256/sve_512 unless SVE is explicitly available.
```

---

### 3. Dotprod path

Существующие dotprod kernels нужно привести к явной kernel family:

```text
- aarch64_neon_dotprod_s8s8
- aarch64_neon_dotprod_u8s8
```

Требования:

```text
- keep this as the first production path;
- target Raspberry Pi 5 and Graviton2;
- add correctness tests;
- add microbench coverage;
- do not mix I8MM instructions into this path.
```

---

### 4. I8MM path

Добавить отдельную kernel family:

```text
- aarch64_neon_i8mm_s8s8
- aarch64_neon_i8mm_u8s8 if needed/supported
```

Требования:

```text
- use SMMLA / UMMLA / USMMLA as appropriate;
- target Apple M2, Samsung S25 and other I8MM-capable CPUs;
- do not emulate I8MM with dotprod and call it I8MM;
- dotprod path must remain fallback.
```

---

### 5. SVE/SVE2 path

Для Graviton3/4:

```text
- clean up SVE/SVE2/oneDNN BRGEMM path;
- do not hardcode SVE vector length;
- use vector-length-agnostic SVE style if implementing custom SVE;
- otherwise safely call oneDNN BRGEMM only when ISA is available.
```

---

### 6. Microbench

`tools/int8_microbench` не должен собираться всегда.

Нужно добавить явную CMake-опцию:

```cmake
option(ENABLE_INT8_AARCH64_MICROBENCH "Build AArch64 INT8 microbench tool" OFF)
```

Benchmark tool должен печатать:

```text
- selected ISA path;
- kernel family;
- shape;
- thread count;
- packing time;
- compute-only time;
- end-to-end executor time;
- median latency;
- p90/p95 latency;
- throughput;
- correctness status.
```

---

## Benchmark protocol

### Главный принцип

```text
Do not optimize blindly.
Use a benchmark-first loop.
```

Для каждого kernel path:

```text
1. Compare with C++ reference.
2. Compare with existing OpenVINO path.
3. Compare with oneDNN/ACL/KleidiAI/XNNPACK where comparable.
4. Find the best baseline per shape.
5. Optimize until the new path is:
   - faster than the best comparable baseline, or
   - within 90-95% of the best baseline with a clear bottleneck explanation, or
   - blocked by missing ISA/library support with evidence.
```

---

### Правила честного сравнения

Speedup считается валидным только если совпадают:

```text
- shape;
- datatype;
- layout or packing cost is accounted;
- batch size;
- thread count;
- affinity policy;
- input/output post-processing;
- correctness target;
- warmup policy;
- measurement policy.
```

Benchmark должен включать:

```text
- warmup iterations;
- repeated measurements;
- median latency;
- p90 latency;
- p95 latency;
- throughput;
- selected ISA printout;
- thread count printout;
- CPU frequency/governor notes;
- thermal throttling notes for phones.
```

Для Samsung S25 отдельно:

```text
Phone benchmarks must mention thermal state/throttling.
Short burst numbers and sustained numbers must not be mixed.
```

---

## Обязательные benchmark shapes

### 1. 1x1 convolution / GEMM-like

Первый и самый важный fast path.

```text
Optimize 1x1 first.
Do not start by over-optimizing 3x3/5x5 before 1x1 is correct and fast.
```

---

### 2. 3x3 convolution

```text
Include stride 1 and stride 2 if supported by current executor.
Do not expand semantics only for benchmark convenience.
```

---

### 3. 5x5 convolution

```text
Benchmark only if already supported by current PR.
Do not add new broad semantics just for benchmark coverage.
```

---

### 4. Mobile CNN shapes

Для Raspberry Pi 5 и Samsung S25:

```text
- small C;
- small spatial;
- batch 1;
- tail-heavy channels;
- realistic mobile inference shapes.
```

---

### 5. Server throughput shapes

Для Graviton2/3/4:

```text
- larger batch;
- larger output channels;
- larger spatial;
- multi-thread scaling;
- memory bandwidth sensitivity.
```

---

### 6. Tail-heavy shapes

Обязательно:

```text
- OC not multiple of block;
- IC not multiple of dotprod/I8MM block;
- OH/OW tails;
- K not divisible by 4/8/16/32.
```

---

### 7. Pathological correctness shapes

```text
- M/N/K = 1;
- unaligned pointers;
- signed/unsigned combinations;
- zero-points;
- bias;
- clamp;
- dst i32;
- dst i8/u8 if supported.
```

---

## Kernel optimization rules

### 1. Сначала 1x1

```text
Start with 1x1 convolution.
This is the cleanest BRGEMM/GEMM-like path and the best place to prove the kernel.
```

---

### 2. Packing отдельно от compute

Нужно отдельно измерять:

```text
- compute-only time;
- packing time;
- end-to-end executor time.
```

---

### 3. Weights/B packing

```text
B/weights packing is mandatory.
Document the packed layout explicitly.
Do not silently transpose inside the hot kernel.
```

---

### 4. Accumulators

```text
Keep accumulators in vector registers.
Avoid C loads/stores inside the hot K loop unless accumulation semantics require it.
```

---

### 5. Register blocking

Требуется ISA-specific blocking:

```text
dotprod path:
- tune around SDOT/UDOT throughput;
- avoid excessive register pressure;
- optimize for Cortex-A76 and Neoverse N1 class cores.

I8MM path:
- tune around SMMLA/UMMLA tile shape;
- optimize for Apple M2 and Oryon-class cores;
- keep dotprod fallback.

SVE/SVE2 path:
- vector-length agnostic loops;
- predicated tails;
- no hardcoded vector length.
```

---

### 6. Tail handling

```text
No scalar slow tail unless benchmark proves it is acceptable.
Prefer masked/predicated or small specialized tails where reasonable.
```

---

### 7. Prefetch

```text
Do not add prefetch by intuition.
Add prefetch only after benchmark proves benefit.
Keep prefetch distance tunable.
```

---

### 8. Threading

```text
Do not mix microkernel optimization with thread scheduling first.
First win single-thread.
Then optimize multi-thread partitioning.
```

---

### 9. Tuning tables

```text
Tuning tables are allowed only after benchmark data.
They must be keyed by ISA/features and shape class, not device name.
```

---

## Correctness requirements

### Главное правило

```text
Every optimized path must compare against reference_cpp.
No optimized path is accepted without correctness tests.
```

---

### Обязательные тесты

```text
- src u8, weights i8;
- src i8, weights i8;
- dst i32;
- dst i8/u8 if post-op path is supported;
- bias i32;
- zero-points;
- per-channel scales if supported;
- clamp/relu if supported;
- K tails;
- N tails;
- M tails;
- unaligned pointers;
- batch/BRGEMM accumulation > 1;
- 1x1 convolution;
- 3x3 convolution if supported;
- 5x5 convolution if supported;
- unsupported ISA tests skip safely.
```

---

### Правила корректности

```text
- Unsupported ISA tests must skip, not crash.
- No illegal instruction on unsupported hardware.
- No tolerance for int32 accumulation mismatch.
- Post-processing mismatch must be explained and bounded only if quantization rounding policy differs.
```

---

## Структура коммитов

Агент должен работать не одним гигантским коммитом, а в reviewable порядке.

```text
Commit 1:
AArch64 ISA detection and dispatch enum.
No kernel math changes.

Commit 2:
Reference INT8 BRGEMM/GEMM correctness harness.
No optimization.

Commit 3:
Refactor current PR dispatch to use centralized ISA detection.
No performance changes expected.

Commit 4:
Clean up existing NEON dotprod kernel family.
Add tests and benchmark output.

Commit 5:
Add NEON I8MM kernel family.
Add tests and benchmark output.

Commit 6:
Clean up SVE/SVE2/oneDNN BRGEMM path.
No silent SVE fallback.

Commit 7:
Add/guard microbench tooling.
Benchmark tool must be behind explicit CMake option.

Commit 8:
Packing/blocking/tuning based on benchmark data only.
```

Правило:

```text
Each commit must build independently if possible.
Do not create one giant mixed commit.
```

---

## Первый узкий шаг для агента

Начинать нужно не с оптимизации, а с безопасного dispatch.

```text
First task only:

Refactor PR #33951 so that AArch64 INT8 kernel selection is centralized and ISA-safe.

Specifically:
1. Move local has_asimd_dotprod() from brgemm_int8_kernel.cpp into a reusable AArch64 CPU feature detection helper.
2. Add detection for dotprod, I8MM, SVE, SVE2, SVE INT8 where available.
3. Use Linux/Android getauxval and macOS sysctlbyname.
4. Replace implicit SVE fallback in BrgemmInt8Kernel with explicit mayiuse checks.
5. Add selected-kernel debug logging.
6. Add unit tests for detection/dispatch where possible.
7. Do not change math kernels in this step.
8. Do not add I8MM implementation in this step.
9. Keep behavior identical except safer dispatch and clearer fallback.
```

---

## Acceptance criteria

### Safety

```text
- No illegal instruction on unsupported CPUs.
- Dispatch is ISA-based, not device-name-based.
- Unsupported optimized paths skip safely in tests.
```

---

### Correctness

```text
- All optimized paths are bit-exact against reference int32 accumulation.
- All tail cases pass.
- All supported src/weight/dst combinations pass.
```

---

### Performance

```text
- New path beats existing OpenVINO path on target shapes.
- New path is benchmarked against oneDNN/ACL/KleidiAI/XNNPACK where comparable.
- For every shape, report:
  - selected ISA;
  - latency median;
  - p90/p95;
  - throughput;
  - packing cost;
  - compute-only cost;
  - total executor cost;
  - thread count.
```

Если новый path не быстрее лучшего внешнего baseline:

```text
Do not hide it.
Explain the exact bottleneck:
- ISA choice;
- packing;
- memory bandwidth;
- register blocking;
- tail handling;
- post-processing;
- threading;
- cache behavior.
```

---

### Reviewability

```text
- No public API changes.
- No graph-level transformation changes.
- No unrelated refactoring.
- Microbench is optional and not built by default unless project convention requires it.
- PR can be reviewed commit-by-commit.
```

---

## Финальная команда агенту

```text
Your job is not to merely add an AArch64 INT8 JIT kernel.

Your job is to produce a safe, ISA-dispatched, benchmark-proven implementation that can compete with the best existing ARM CPU libraries.

If the new kernel is faster, prove it with reproducible data.

If it is slower, do not hide it. Identify the exact bottleneck:
- ISA choice;
- packing;
- memory bandwidth;
- register blocking;
- tail handling;
- post-processing;
- threading;
- cache behavior.

No performance claim is accepted without benchmark evidence.
No optimized path is accepted without correctness tests.
No device-specific hack is accepted.
```

---

## Короткий вариант для вставки в Codex первым сообщением

```text
Work on OpenVINO PR #33951.

Refactor and optimize the AArch64 INT8 BRGEMM convolution/JIT work into a safe, ISA-dispatched, benchmark-proven CPU-only implementation.

Use Xbyak_aarch64, not regular Xbyak.

Do not dispatch by device name. Dispatch only by runtime ISA features:
- NEON
- dotprod
- I8MM
- SVE
- SVE2
- SVE INT8 where available

Target paths:
- Raspberry Pi 5 / Graviton2: NEON dotprod / SDOT/UDOT
- Apple M2 / Samsung S25: NEON I8MM / SMMLA-family
- Graviton3/4: SVE/SVE2 INT8 or safe oneDNN BRGEMM path

First step:
Only centralize ISA detection and dispatch.
Do not change math kernels in the first step.
Do not add I8MM in the first step.

Then proceed commit-by-commit:
1. ISA detection
2. reference correctness harness
3. dispatch refactor
4. dotprod cleanup
5. I8MM kernels
6. SVE/SVE2 cleanup
7. optional microbench
8. packing/blocking/tuning based on benchmark data

Hard rules:
- no device-name hacks
- no unsupported instructions without feature checks
- no public API changes
- no graph transformation changes
- no giant mixed commit
- no performance claims without reproducible benchmarks
- no optimized path without correctness tests

Compare against:
- existing OpenVINO path
- oneDNN/ACL where applicable
- Arm Compute Library where applicable
- KleidiAI where applicable
- XNNPACK where applicable

For every benchmark report:
- selected ISA
- shape
- thread count
- packing time
- compute-only time
- end-to-end time
- median latency
- p90/p95
- throughput
- correctness status

If the new path is slower than the best comparable baseline, explain the bottleneck instead of hiding it.
```
