# JIT INT8 Convolution (AArch64) via BRGEMM: инструкция/чеклист

Цель: реализовать и сопровождать INT8 свертки на AArch64 через BRGEMM-подход в CPU-плагине OpenVINO так, чтобы:
- по функциональности было **полное парити** с x86/x64 веткой (oneDNN BRGEMM INT8),
- по производительности было **быстрее, чем ACL**, а **KleidiAI** использовался **только как перф-референс**.

## Базовые правила

1) JIT-ядра: **только Xbyak_aarch64** (генерация кода).  
   Никаких intrinsics в compute-ядрах. Допускается обычный C++ для glue/pack/раскладки.

2) Максимально переиспользуем существующую инфраструктуру CPU-плагина и oneDNN.  
   Чего нет (или нет для AArch64 INT8) - дописываем у себя.

3) Структурно ориентируемся на **oneDNN BRGEMM INT8 для x64**:
- такие же режимы/флаги,
- такие же варианты квантования/компенсаций,
- такая же логика prepare/update vs execute,
- поведенческая совместимость (edge cases, хвосты, паддинг/дилейшн/группы, post-ops).

4) Executor:
- обязателен отдельный AArch64 executor для вызова наших JIT-ядрер,
- регистрация через `src/plugins/intel_cpu/src/nodes/executors/convolution_implementations.cpp`,
- **update()/prepare()** выполняется при инициализации/подготовке и **не должен дергаться из execute()**.

5) ACL INT8 conv:
- если доступен наш AArch64 BRGEMM executor - **ACL INT8 conv path не должен выбираться**.

## Где лежит код

- Executor (AArch64 INT8 conv):
  - `src/plugins/intel_cpu/src/nodes/executors/aarch64/jit_int8_conv.cpp`
  - `src/plugins/intel_cpu/src/nodes/executors/aarch64/jit_int8_conv.hpp`

- JIT BRGEMM микроядра (AArch64):
  - `src/plugins/intel_cpu/src/nodes/kernels/aarch64/brgemm_kernels/`

Нотация файлов/ядрер: придерживаемся “kleidiai-like” схемы: один файл = одно микроядро/тайл/ISA-вариант,
с явным отражением типа данных/аккумуляции/пакетинга/ISA в имени.

## Покрытие (делаем по этапам, но цель - все)

Минимальный “боевой” набор:
- 1x1 (GEMM-путь)
- kxk (3x3, 5x5) со stride/pad/dilation
- layouts: как минимум NHWC (nspc) и NCHW (ncsp), далее blocked (nCsp8c/nCsp16c)
- квантование: u8/s8 вход, s8 веса, s32 аккум; output: s8/u8 (и/или f32 где нужно по API)

Дальше:
- groups, depthwise
- post-ops parity (bias, sum, eltwise/activation, clamp, scales, per-oc scales)
- zero-points (src/dst), weights compensation, слияние с LPT/FQ где применимо
- остатки/хвосты, неполные блоки по M/N/K

## Перф-методика (сначала базовый перф, потом оптимизации)

Релизная сборка обязательна для измерений:
- `cmake --build build-release-ninja`

Microbench (INT8 GEMM/CONV) - основной быстрый сигнал:
- таблицы генерируются скриптом:
  - `python3 src/plugins/intel_cpu/tools/int8_microbench/collect_perf_tables.py --conv-only`
  - `python3 src/plugins/intel_cpu/tools/int8_microbench/collect_perf_tables.py --gemm-only`

Сравнение:
- ACL (в репо):
  - `src/plugins/intel_cpu/thirdparty/ComputeLibrary`
- KleidiAI:
  - **только как референс производительности** (не интегрируем как зависимость).

Важно про fairness:
- многие внешние бенчи считают “compute-only”, исключая pack/im2col из тайминга.
- наши таблицы должны показывать:
  - `Our ms` (end-to-end),
  - `Our pack ms`,
  - `Our compute ms = Our ms - Our pack ms`.

Бенч на реальных сетках:
- прогоняем INT8 модели через `benchmark_app`, снимаем послойную статистику.
- модели берем из PTS/Phoronix (только INT8), baseline сравниваем с OpenVINO master.

## Обязательный цикл после каждой итерации

1) Сборка RelWithDebInfo:
- `cmake --build build-relwithdebinfo-ninja`

2) Тесты по сверткам:
- `bin/aarch64/RelWithDebInfo/ov_cpu_func_tests --gtest_filter='*Convolution*'`

3) Сборка Release:
- `cmake --build build-release-ninja`

4) Перф-таблицы microbench:
- `python3 src/plugins/intel_cpu/tools/int8_microbench/collect_perf_tables.py --conv-only`
- `python3 src/plugins/intel_cpu/tools/int8_microbench/collect_perf_tables.py --gemm-only`

## Gap-анализ (перед крупными оптимизациями)

Обязательное сравнение “что у них есть, чего у нас нет”:
- oneDNN x64 BRGEMM INT8 (вендорная структура/фичи/пост-опы/compensation),
- ACL int8 conv/gemm kernels (packing/prepare/update, indirect/direct подходы),
- KleidiAI (как reference: что именно делает быстрее - tile shapes, pointer-indirect, prefetch, scheduling).

Вывод gap-анализа должен приводить к 2-3 пунктам, которые:
- дают максимальный ожидаемый выигрыш,
- измеряются microbench + `benchmark_app`,
- не ломают функциональность (парити с x64).

