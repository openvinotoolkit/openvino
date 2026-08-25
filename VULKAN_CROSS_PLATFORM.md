# ov::core::vulkan::cross_platform — план объединения бэкендов в единое Vulkan-ядро

> План зафиксирован в файле, чтобы не теряться между сессиями.
> Статус обновляется после каждого шага.

## Цель

`openvino/src/plugins/intel_cpu + intel_gpu + intel_npu` → единое Vulkan-ядро
`ov::core::vulkan::cross_platform`.

- Ядро = Vulkan-рантайм (движок, потоки, память, ядра, сеть), не зависит от
  openvino core-графа (`ov::Model`, `ov::Node`) и не требует его.
- openvino core больше не обязателен: модели загружаются конвертерами
  напрямую в `vk::ir_graph` (Float Binary, Parallel Binary и другие форматы).
- Плагины intel_cpu / intel_gpu / intel_npu — **не нужны и не делаются обёртками**:
  ядро само кросплатформенное (GPU/CPU/NPU = одно и то же Vulkan-ядро,
  устройство выбирается по доступному Vulkan-драйверу). Обёртки отменены.
- Без сложных абстракций: платформенная разница — только флаги инициализации
  `vk_engine` (никаких интерфейсных слоёв).

## Матрица платформ (актуально, 2026)

| Платформа | Механизм | Текущий статус экосистемы |
|---|---|---|
| Desktop GPU/CPU (Intel/AMD/NVIDIA/ARM) | Vulkan 1.2+ (нативный) | Работает |
| macOS / iOS / tvOS | MoltenVK v1.4.2 (июль 2026), macOS 12+/iOS 15+, portability (`VK_KHR_portability_enumeration` + `VK_KHR_portability_subset`), SPIR-V → Metal через SPIRV-Cross | Флаги заложены в `vk_engine` (шаг 3): включаются только при наличии расширений, конфиг `moltenvk` |
| Embedded / RISC-V / телефоны | Vulkan SC (на базе Vulkan 1.2, SPIR-V 1.5), оффлайн-компиляция пайплайнов (vksc SDK), статическая память; имплементации: CoreAVI VkCore SC, NVIDIA DRIVE/Jetson/desktop, Arm Mali-G78AE, Imagination; dev-слой эмуляции поверх обычного Vulkan | Задел готов (шаг 3): конфиг `vulkan_sc` + экспорт оффлайн-артефактов (`.spv` + `VkPipelineCache`); полный путь — при наличии `vksc_core.h` |
| NVIDIA multi-GPU | NCCL 2.2x (коллективы NVLink/PCIe) через CUDA; мост: `VK_KHR_external_memory_*` / `VK_KHR_external_semaphore_*` + `cudaImportExternalMemory` / `cudaImportExternalSemaphore` | Отдельный опциональный модуль |

## Шаги (каждый — со сборкой)

1. [x] **Выделить ядро** из `intel_gpu/src/vulkan` в standalone-модуль
       `src/plugins/vulkan_core/`, namespace `ov::core::vulkan::cross_platform`.
       Плагин intel_gpu переводится на линковку ядра.
       *(сделано: ядро собрано как `openvino_vulkan_core`, плагин линкует его
       и исполняет инференс целиком на Vulkan-рантайме)*
2. [x] **Форматные конвертеры в ядре**: FB (Float Binary) / PB (Parallel
       Binary) → `vk::ir_graph` без `ov::Model`. `VkModelConverter` остаётся
       только как адаптер OV-плагина.
       *(сделано: `vk_graph_format.hpp/cpp` — standalone-модуль без зависимостей
       от openvino core (только `vk_ir.hpp`); FB = один граф (magic
       `VKFB0001`, версия u32, little-endian секции: тензоры, константы, ноды
       с pool-параметрами и флагом transpose_b, порты; границы проверяются —
       битые/обрезанные блобы падают с ошибкой), PB = контейнер графов
       (magic `VKPB0001`, вложенные FB-блобы). Плагин intel_gpu использует FB
       как формат compiled blob: `Graph::export_model` пишет
       `[VKFB magic][u32 size][FB]`, `Graph(BinaryInputBuffer&)` читает его и
       строит программу через общий `build_from_ir` — импорт не требует
       `ov::Model`. Проверено: standalone round-trip FB/PB (cl-сборка без
       openvino.dll) + сквозной экспорт/импорт блоба через публичный API
       (smoke9).)*
3. [x] **Платформенные точки входа `vk_engine`**: desktop, MoltenVK
       (portability-флаги), Vulkan SC (оффлайн-пайплайны), опционально
       NCCL-мост.
       *(сделано: `vk_platform_config` в `vk_platform.hpp` (enum
       `vk_platform`: desktop/moltenvk/vulkan_sc + оффлайн-директория +
       зарезервированный флаг `nccl_bridge`); конфиг по умолчанию читается
       из окружения (`OV_GPU_VK_PLATFORM`, `OV_GPU_VK_OFFLINE_DIR`), можно
       задать программно через `create_vk_engine(device, runtime_type,
       config)`. desktop — Vulkan 1.3; moltenvk — apiVersion 1.2 +
       `VK_KHR_portability_enumeration` + `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR`
       (включаются только если расширение реально есть — на десктопе
       грациозный фолбэк) и `VK_KHR_portability_subset` при создании
       устройства (обязателен на macOS, иначе vkCreateDevice падает);
       vulkan_sc — apiVersion 1.2 + экспорт оффлайн-артефактов: каждый
       нативный кернел пишется как `<kernel_id>.spv`, все пайплайны строятся
       через общий `VkPipelineCache`, который сбрасывается в
       `vk_pipeline_cache.bin` при разрушении билдера. Полный SC-путь
       (замена `vkCreateComputePipelines` на оффлайн-объекты) требует
       `vksc_core.h` — задел готов, заголовки отсутствуют. Проверено
       smoke10: desktop, moltenvk (Windows-фолбэк) и vulkan_sc +
       оффлайн-экспорт (артефакты на месте, инференс PASS).)*
4. [x] **Маршрутизация по именам устройств (автономная)**: единая точка входа
       ядра сама выбирает исполнителя по имени (`device_name`), роутинг
       прежнего OV-плагина перенесён внутрь ядра.
       *(сделано в standalone-архитектуре: `vk_dispatch.hpp` — `vk_execute(
       graph, inputs, "GPU[.N]" | "CPU", platform_config)`: CPU → нативный
       `cpu_execute`, всё остальное → Vulkan-путь (vk_engine → vk_program →
       vk_network, readback через host-visible override); `vk_available_devices()`
       перечисляет устройства как маршрутизируемые имена ("GPU.0", "CPU.1"...);
       в `init_vk_engine_data` явная классификация по типу физического
       устройства: GPU = discrete/integrated/virtual, CPU = CPU-тип драйвера,
       NPU = класса не существует → всегда чистая ошибка «No Vulkan device
       matching ...» (раньше NPU ошибочно матчился с GPU — поймано смоуком);
       суффикс `.N` теперь реально выбирает N-й кандидат: stable_sort по score,
       ничьи сохраняют порядок перечисления. Проверено ops_smoke 22/22:
       dispatch CPU/GPU/GPU.0 vs эталон, NPU — чистая ошибка; полный блок
       операций + FB/PB round-trip PASS.)*

5. [x] **Легаси удалён накорню**: `src/plugins/intel_cpu` (~1900 файлов) и
       `src/plugins/intel_npu` (~700 файлов), `src/plugins/intel_gpu` выпилены
       из дерева вместе с их сабмодулями (из `.gitmodules` убраны 7 записей:
       onednn, ComputeLibrary, mlas, yaml-cpp, libxsmm, kleidiai, xbyak_riscv;
       остались `onednn` и `xbyak`). Из `cmake/features.cmake` удалены опции
       `ENABLE_INTEL_CPU` / `ENABLE_INTEL_NPU` / `ENABLE_INTEL_GPU` и их производные
       (ARM_COMPUTE_CMAKE, CPU/NPU_DEBUG_CAPS, SNIPPETS_LIBXSMM_TPP,
       NPU_INTERNAL), `ENABLE_OV_ZERO_LOADER` теперь зависит только от
       `GPU_RT_TYPE=ZE`. Конфиг-лист фич больше не знает CPU/NPU-плагинов;
       один плагин `openvino_intel_gpu_plugin.dll` обслуживает GPU/CPU/NPU.
       oneDNN для следующего шага — уже выкачанный сабмодуль
       `src/thirdparty/onednn` (oneapi-src/oneDNN, ~4.8k файлов) — новых скачиваний
       не требуется. *(проверено: полная переконфигурация + сборка плагина + регресс
       smoke–smoke11, fmt_test — все PASS)*
6. [x] **CPU-бэкенд ядра (кросплатформенный исполнитель)**: `cpu_engine.hpp/cpp`
       — исполняет тот же `vk::ir_graph` целиком на CPU (f32), без Vulkan.
       Ops: relu/add/matmul (+transpose_b)/maxpool/avgpool/conv2d + нативный
       девант квантованных констант (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) теми же
       формулами, что в `matmul_q_f32.comp`. Выбор устройства: GPU → Vulkan,
       CPU → cpu_engine (одно ядро, два исполнителя). *(сделано: `gguf_cross.exe`
       — f32-графы против эталонов + Q4_0 против референса + CPU↔GPU parity —
       ALL PASS; smoke и gguf_fe_q регресс PASS.)* Бенчмарки на моделях
       (ResNet/YOLO/BERT, ~1GB скачиваний) — отложены по решению
       пользователя по причине, того что пользователь не хочет тратить
        один гигабайт интернета их пятнадцити на месяц.
7. [ ] **NPU / TPU-бэкенды (после oneDNN)** — см. раздел
       «Дорожная карта бэкендов» ниже. Один `ir_graph` → много исполнителей,
       устройство выбирается по имени (`device_name`). CPU-бэкенд (шаг 6) —
       рабочий шаблон нового исполнителя.
8. [x] **Автономные конвертеры моделей в ядро** (загрузка напрямую в
       `ir_graph` без `ov::Model`): `paddle_reader.hpp/cpp` — PaddlePaddle
       `__model__` (ProgramDesc, protobuf wire-формат без protobuf-библиотеки)
       + вес-файлы `LoDTensor`; маппинг feed/fetch, relu, elementwise_add,
       matmul(+transpose_Y), conv2d (groups=1, NCHW), max/avg_pool2d (floor,
       avg exclude_pad=true); неподдерживаемое → чистая ошибка. Проверено
       `paddle_fe.exe`: conv→relu→maxpool (pad1) и matmul→add на CPU и Vulkan —
       ALL PASS. *(Ранее в той же серии: `gguf_reader` — GGUF-веса напрямую в
       quant_constants, нативный девант в шейдере.)*

> **Дорожная карта бэкендов (после oneDNN)** — NPU и TPU добавляются как новые
> исполнители того же `ir_graph` (как `cpu_engine`/`vk_network`), без
> переделки ядра. Формат загрузки уже вендор-агностичен (FB/PB/GGUF → ir_graph).
>
> **NPU — два пути:**
> - **Vulkan SC-путь** (задел уже есть в `vk_platform.hpp`: конфиг `vulkan_sc`,
>   оффлайн-экспорт `.spv` + `vk_pipeline_cache.bin`, статическая память):
>   NPU с Vulkan SC-драйвером (CoreAVI VkCore SC, NVIDIA DRIVE/Jetson,
>   Arm Mali-G78AE, Imagination) = ещё один режим `vk_engine`, тот же рантайм.
>   Нужно: `vksc_core.h` + железо. Это не-Intel экосистема — вендор-агностично.
> - **Вендорный рантайм** (Rockchip/RKNN, Raspberry Pi NPU, Hailo, CVFlow и пр.):
>   новый исполнитель по образцу `cpu_engine`, который конвертит `ir_graph` в
>   API вендора (RKNN-компилятор, HailoRT, TFLite-делегат) и исполняет. Плюс
>   маппинг операций — конвертер в ir_graph уже есть, FB/PB уже сериализуются.
>
> **TPU:**
> - **Edge TPU / Coral**: маршрут через TFLite-делегат — конвертер `ir_graph` →
>   TFLite flatbuffer + официальный `edgetpu` delegate. Отдельный исполнитель.
> - **Cloud TPU**: только XLA — для автономного ядра нерелевантно, не берём.
>
> **Как ложится в код:**
> - Маршрутизация по именам уже есть (`device_name` + выбор физического
>   устройства) — расширяется на «имя NPU/TPU → свой исполнитель».
> - Единый интерфейс исполнителя: `execute(ir_graph, inputs) → outputs`
>   (как `cpu_execute` / `vk_network::execute`).
> - Добавление бэкенда = новый исполнитель + (для неродных рантаймов) конвертер
>   в API вендора. Архитектурных переделок не требуется.
>
> **Важно (позиция):** NPU «не работает» сегодня не из-за Intel, а потому что
> нет Vulkan SC-железа и `vksc_core.h`; где Vulkan-драйвера нет вовсе (часть
> NPU/TPU) — либо вендорный рантайм, либо фолбэк на `cpu_engine`.


> **Жизнеспособность ядра:** набор операций расширяется батчами (каждый новый
> op = `ir_op` + `.comp`-кернел + случай в `vk_program`/`cpu_engine` +
> `u8_to_op` + смоук; границы проверяются на этапе конвертации). Готово:
> батч 1 — mul/sub/div, sigmoid/tanh/leaky_relu (+alpha, FB v2); батч 2 —
> transpose, reshape (алиас), concat, softmax, reduce mean/sum/max (FB v4);
> батч 3 — батчированный matmul 3D(+tb), gelu/swiglu, контракт bias для conv;
> батч 4 — quick_gelu, rms_norm, pad, квантованный batched matmul (FB v5);
> батч 5 — softmax по произвольной оси, crop, попарно-батчированный matmul,
> attention-композиция (scale → Q·Kᵀ → softmax → ·V) проверена тестом;
> батч 6 — causal_softmax, rope, функциональный cache_write, argmax, GQA
> (Bb=1), полный decoder-шаг в тесте.
> Дальше: батч 7 (скорость: tiled matmul, float4, фьюжн bias+act, FP16),
> батч 8 (пул памяти, графовые пассы dce/fold, Safetensors), батч 9
> (витрина: док формата, примеры).

## Риски

- Пункт 4 — самый объёмный: `engine/stream/memory/kernel` тянут OV runtime API.
  Делается последним, после того как ядро заработает автономно через OV-плагин.

## Журнал

- 2026-08-26 (ночь): **батч 9 + бенчмарк №1**. Витрина: `src/core/README.md`
  (обзор ядра) и `FORMAT.md` (байтовая спека FB/PB v5 + таблица op-кодов).
  Бенч (`test_perf.exe`, сборка один раз, чистый execute): tiled matmul
  256³ = **182 GFLOP/s**, 512³ = **274–378 GFLOP/s**; decoder-step
  B=1 L=128 D=256 (attention + KV-cache append + causal softmax + argmax,
  целиком ops ядра) = **2.75 мс/шаг ≈ 46 500 tokens/s**. Пул памяти и
  FP16-тензоры — в следующий батч с дизайном: пулу нужны оффсетные
  дескрипторы (VkDescriptorBufferInfo.offset в стриме + view-offset в
  vk_memory), FP16 — dtype в IR и FB v6. Всё закоммичено и запушено на
  fork (`afe2df5462` core batches, `93604b5215` repo slimming); рабочее
  дерево чистое.
- 2026-08-26 (вечер): **батч 8 (инженерия рантайма, часть 1)** —
  **Safetensors-ридер** (`safetensors_reader.hpp/cpp`, namespace `st_r`):
  u64-заголовок + мини-JSON-сканер фиксированной схемы (dtype/shape/
  data_offsets, skip `__metadata__`), F32/F16/BF16 → f32, остальное — чистая
  ошибка; тест пишет файл из кода и гоняет matmul с весами из файла на CPU+GPU.
  **Графовые пассы** (`vk_pass.hpp/cpp`, namespace `pass`, чистые функции над
  ir_graph): `dce` (недостижимые ветки; осиротевшие constant-ноды удаляются
  ВМЕСТЕ с payload — первая версия оставляла ноды без данных), 
  `fold_constants` (каскадный фолдинг через cpu_execute мини-графа; два
  грабли: синтетическая result-нода не создаёт тензор → outputs.at() падал,
  константы обязаны идти раньше потребителя в мини-графе),
  `peephole` (transpose∘transpose → композиция/аннигиляция, relu∘relu,
  sigmoid∘sigmoid), `optimize` = всё до фикспоинта. Тесты `optim` + 
  `safetensors`. Пул памяти и FP16-тензоры перенесены в батч 9: пулу нужны
  оффсетные дескрипторы в стриме, FP16 — поле dtype в IR (FB v6).
- 2026-08-26 (день): **батч 7 (скорость)** — shared-memory tiling 16×16 для
  трёх горячих GEMM: `matmul_tiled_f32` (2D), `matmul_batched_tiled_f32`
  (shared-батч), `matmul_bb_tiled_f32` (попарный/GQA). Пуш-раскладка
  идентична наивным кернелам — drop-in; билдер выбирает тайл автоматически
  при M,N,K ≥ 16 (f32, не transpose_b); грид для тайлов — 2D {N,M,B} вместо
  линейного (первый запуск дал частичный расчёт — поймано паритетом на
  17³/33×48). Замер (`test_perf.exe`, сборка один раз, тайминг чистого
  execute): **129 GFLOP/s @ 256³, 162 GFLOP/s @ 512³** (0.52 мс / 1.65 мс)
  на пользовательском GPU; первый замер «0.2 GFLOP/s» мерял пересборку
  программы в каждом вызове, не GEMM. FP16-тензоры и фьюжн-пассы перенесены
  в батч 8 (нужно решение по памяти и инфраструктура пассов). Инцидент:
  vk_program.cpp был обнулён неудачной записью редактора — восстановлен
  целиком по памяти сессии, все 6 exe зелёные.
- 2026-08-26: **батч 6 (LLM-словарь)** — ядро теперь 28 ops, и в `test_llm.cpp`
  собран полный decoder-шаг из ops ядра: scale → Q·Kᵀ (попарный батч) →
  causal softmax → P·V → KV-cache append → crop чтение истории. Новые op:
  **causal_softmax** (последняя ось [...,L,L], хвост обнуляется),
  **rope** (halves-конвенция; cos/sin = x_dims[:-2]+[D/2]; `half` —
  зарезервированное слово GLSL, член пуша назван hf), **cache_write**
  (функциональный append: out=cache с заменой rows [pos,pos+L); первая
  версия была in-place и сломалась об override-индирекцию — писатель целил
  override-буфер, читатель брал программный; функциональная семантика
  композируется идеально), **argmax** (первая ничья). **GQA**: pairwise
  matmul принимает Bb=1 (матрица шарится на батч, push b_batch + bt%bb).
  Регрессия: 5 exe ALL PASS (~75 проверок). Дважды виноват референс теста,
  не движок: sc не обнулялся между батчами; P·V суммировал по всем j вместо
  каузального треугольника.
- 2026-08-25 (утро #2): **батч 5** — ядро теперь 24 ops, и оно впервые
  собирает настоящий трансформерный блок. (1) **Softmax по произвольной
  оси**: кернел переписан на [outer,len,inner]-декомпозицию (нить на строку,
  стабильный max-вычит); last-axis-ограничение снято, shape_ops теперь
  проверяет корректность средней оси вместо отказа. (2) **Crop** — окно по
  осям, begin-офсеты в pads_begin, до 8D. (3) **Попарно-батчированный MatMul**
  `matmul_bb_f32`: A [B,M,K] x B [B,K,N] — per-head матрицы; ветка
  выбирается по рангу B (3D → pairwise, 2D → shared). (4) **Attention как
  композиция** (`test_attention.cpp`): scale→Q·Kᵀ→softmax(axis=2)→·V —
  целиком из ops ядра, CPU и GPU против референса; crop там же.
  Найдено в ходе: K попарного GEMM берётся из b_shape[1]; lines softmax = 
  outer*inner (не outer). Регрессия: 4 exe ALL PASS.
- 2026-08-25 (ночь #2): **батч 4** — ядро теперь 23 ops. **QuickGELU**
  (`x·sigmoid(1.702x)`), **RMSNorm** по последней оси (weight [axis], eps =
  alpha; alpha теперь документирован как общий скалярный атрибут op:
  slope/eps/fill), **Pad** с constant fill (pads_begin/pads_end по осям, до
  8D), **квантованный batched MatMul** (`matmul_q_batched_f32`, Q4_0/Q4_1/
  Q5_0/Q5_1/Q8_0, блоки вдоль N общей матрицы). FB → **v5** (pads_end).
  Найдено в ходе: (1) push-бюджет 128Б жёсткий — pad с четырьмя 8-тuples не
  влезал (140Б); in_dims выведены в шейдере из out−pb−pe (108Б); стрим
  поймал перерасход чистой ошибкой — защита работает; (2) softmax/rms_norm
  корректны только на последней оси (контiguous lines) — добавлена явная
  валидация вместо молчаливо неверных чисел на средней оси; (3) тестовый
  heap-corruption: референс-вектор короче формы графа. Тесты: новый
  `test_shape_ops.cpp` (12 проверок) — регрессия всех трёх exe ALL PASS.
- 2026-08-25 (ночь): **батч 3** — ядро теперь 20 ops. (1) **Батчированный
  MatMul**: A [B,M,K] x общая B [K,N|N,K] → [B,M,N], кернелы
  `matmul_batched(+transpose_b)_f32`; ветка выбирается по рангу входа A в
  `build()` (ir_op тот же, IR/FB без изменений); квантование пока только на
  2D-пути — чистая ошибка на 3D+quant; fc = reshape-алиас + обычный 2D matmul.
  (2) **GELU** (tanh-аппроксимация) и **SwiGLU** (`silu(a)*b`) — elementwise,
  swiglu входит в broadcast-группу бинарников. (3) **Контракт conv**: ровно
  3 входа (data, weights, bias); отсутствие bias раньше читало мусорную
  память в GPU-кернеле (binding B0 безусловен) и падало с std::out_of_range
  на CPU — теперь чистая ошибка «materialize a zero bias upstream» на обоих
  исполнителях. Тесты: новый `test_nn_ops.cpp` (10 проверок — заодно первая
  проверка мульти-тестового CMake: два exe из одной папки), регрессия
  ops_smoke ALL PASS.
- 2026-08-25 (вечер): смоки переехали в репо — `src/core/tests/`, и переведены
  с ручных `cl`/`link`-команд на чистый CMake: `CMakeLists.txt` собирает ядро
  статически + SPIR-V через `gen_spirv.cmake`, каждый `test_*.cpp` → exe +
  регистрация в CTest; `build_tests.bat` = configure/build/run тремя
  командами cmake. Vulkan: SDK или vktools-bootstrap (fallback). Проверено:
  полный прогон ALL PASS.
- 2026-08-25: **батч 2** — 7 новых ops (ядро теперь 18): transpose
  (1..8D-перестановка, gather по out_dims/in_strides/perm), reshape
  (**алиас памяти**, без кернела — плоские f32 переинтерпретируются,
  producer форвардится для выходов модели), concat (цепочка slab_copy в
  общий буфер + финальная identity-копия как registered producer — readback
  захватывает полный буфер), softmax (нить на строку, стабильный max-вычит),
  reduce mean/sum/max (общий кернел с mode-push, ось удаляется).
  `ir_node` + поля `transpose_order`/`axis`; FB → **v4** (сериализация axis +
  perm). Найдено в ходе: (1) push_constant блоки БЕЗ явного std430 получают
  std140 — массивы выравниваются на 16 байт; transpose переписан на 24
  отдельных скаляра (вообще без массивов), новые кернелы помечены
  `layout(push_constant, std430)`; (2) декомпозиция линейного индекса должна
  идти от быстрой оси (первый вариант шёл от медленной → транспонировал
  column-major); (3) concat нельзя собрать gather-кернелом со статическими
  биндингами (стрим биндит фиксированный слот на вход ноды) — потому цепочка
  копий. Смоук ops_smoke: 37 проверок ALL PASS (все ops × CPU/GPU, dispatch,
  FB/PB round-trip новых атрибутов). Набор покрывает базовые цепочки ResNet/
  BERT-класса по ширине (остаются: GEMM 4D, bias conv, активации CLIP-like,
  pad/tiling для больших моделей).
- 2026-08-24 (ночь): шаг 4 завершён в автономной архитектуре — маршрутизация
  по именам устройств перенесена из эпохи плагинов в ядро. Новый header-only
  фасад `src/core/vk_dispatch.hpp`: `vk_execute(graph, inputs, device_name,
  platform_config)` — CPU → `cpu_execute`, иначе Vulkan-путь с readback через
  host-visible override; `vk_available_devices()` — перечисление как
  маршрутизируемых имён. В `init_vk_engine_data`: явная классификация
  (GPU = discrete/integrated/virtual, CPU = CPU-драйвер, NPU — класса нет →
  чистая ошибка; найден и исправлен баг: старый фильтр `is_cpu != want_cpu`
  пропускал GPU под именем NPU) + суффикс `.N` теперь выбирает N-й кандидат
  (stable_sort по score, ничьи — порядок перечисления). Смоук расширен до
  22 проверок (dispatch CPU/GPU/GPU.0/NPU-error/available_devices) — ALL PASS.
- 2026-08-24 (вечер): закрыты оба пробела из утреннего батча.
  (1) **FB-формат v3**: `quant_constants` сериализуются в FB/PB (id, тип u32,
  длина + сырые байты блока) — экспортированный блоб квантованного графа
  больше не теряет нативный девант; проверено round-trip'ом Q4_0-matmul
  структурно (байты идентичны) и численно (CPU и GPU совпали с эталоном).
  (2) **Broadcast в ядре**: constant-broadcast входы elementwise
  (add/mul/sub/div) теперь материализуются самим ядром — `vk_program_builder`
  разворачивает константу до полного размера в host-visible буфер (`id#bcast#node`),
  `cpu_execute` делает то же самое; динамический вход несовпадающего размера →
  чистая ошибка вместо тихо неверных чисел. Проверено bias [1,2] против [2,2]
  на CPU и GPU. Смоук расширен до 17 проверок — ALL PASS. Попутно два бага
  в самом смоуке: беззнаковый underflow `(n%16+1)-8` в референсе (size_t) и
  переполнение ниббла q=16 в генераторе Q4_0 (движок в обоих случаях считал
  верно).
- 2026-08-24: батч 1 расширения операций — 6 новых ops в ядре:
  elementwise mul/sub/div (клоны `eltwise_add_f32`) и активации
  sigmoid/tanh/leaky_relu (шаблон relu; у leaky_relu push-константа
  `{uint total, float alpha}` — `scalar_t::FLOAT32`). `ir_node` получил поле
  `float alpha`; FB-формат поднят до **v2** (в ноду добавлено f32 alpha,
  старые v1-блобы отсекаются проверкой версии). Все точки обновлены:
  `vk_ir.hpp` (enum, новые коды в конце — старые u8 не сдвинуты),
  `vk_graph_format.cpp` (`u8_to_op`, put/get alpha), `vk_program.cpp`
  (имена кернелов + группированные случаи build), `cpu_engine.cpp`.
  Тулчейн восстановлен после чистки temp: glslang 16.5.0
  (%LOCALAPPDATA%\vktools\glslang, новый бинарь `glslang.exe` вместо
  `glslangValidator.exe`) + Vulkan-Headers 1.3.290 + `vulkan-1.lib`,
  сгенерированный из системного `C:\Windows\System32\vulkan-1.dll`
  (dumpbin /EXPORTS → .def → lib /def). Смоук `ops_smoke.exe`
  (vksmoke): цепочка leaky_relu(0.1)→sigmoid→tanh и mul→sub→div против
  аналитических эталонов на CPU и GPU; FB/PB round-trip (alpha
  сериализуется); регрессия legacy relu→matmul→add. Нюанс: readback
  device-local выхода через lock() невозможен — используется
  host-visible override по `output_port_to_id` (set_output_memory).
  Broadcast-bias в тесте сначала дал «неправильные» числа — это контракт
  ядра (broadcast материализуется загрузчиком), тест исправлен.
  Итог: ALL PASS 10/10. Зафиксированы пробелы на следующие батчи:
  quant_constants не пишутся в FB/PB; broadcast-материализация —
  ответственность загрузчика.
- 2026-08-16: автономный Paddle-конвертер в ядре (`openvino/src/core/paddle_reader.hpp/cpp`,
  namespace `...::paddle_r`): читает PaddlePaddle `__model__` (ProgramDesc, proto2
  wire-формат, мини-парсер без protobuf-либы: varint/length-delimited/skip,
  строгие границы) и вес-файлы `LoDTensor` (данные из поля 3). Маппинг:
  feed/fetch → I/O, relu → relu, elementwise_add (same shape) → add,
  matmul (+transpose_Y → matmul_transpose_b) → matmul, conv2d (groups=1,
  NCHW, kernel из формы фильтра) → convolution, max/avg_pool2d (floor,
  avg exclude_pad=true) → pool; ksize/strides/paddings из attr (2/4-элементный
  формат), неподдерживаемое (ceil_mode, exclusive=false, groups>1, не-FP32
  веса, transpose_X) → чистая ошибка. `paddle_fe.exe` (в vksmoke): мини
  protobuf-писатель строит 2 модели — conv(pad1)→relu→maxpool (ожидание
  13.5/7.5/17.5/27.5) и matmul_tb→add (50.5/67.5/122.5/166.5) — обе проходят
  на CPU и на Vulkan, ALL PASS. Найдены и исправлены в ходе: (1) вес-файл
  целиком читался как данные (это protobuf LoDTensor — нужен парсер поля 3),
  (2) `tensor_shapes` не заполнялись из dims переменных, (3) тестовая модель
  давала 1x1 после maxpool 2x2 из 2x2 (conv pad0) — переведена на pad1 → 4x4.
  Регресс: smoke/gguf_cross/gguf_fe_q/gguf_fe — ALL PASS.
- 2026-08-15: CPU-бэкенд ядра (`openvino/src/core/cpu_engine.hpp/cpp`,
  namespace `ov::core::vulkan::cross_platform`): исполняет тот же
  `ir_graph` целиком на CPU (f32) без Vulkan — relu/add/matmul(+transpose_b)/
  maxpool/avgpool/conv2d + нативный девант квантованных констант (Q4_0/Q4_1/
  Q5_0/Q5_1/Q8_0) теми же формулами, что в `matmul_q_f32.comp` (f16→f32,
  блоки вдоль N, 32 веса/блок). `gguf_cross.exe` (тест в vksmoke): f32-графы
  против эталонов (pipeline 58.5/63.5/139.5/153.5, matmul_tb 50/68/122/167,
  maxpool 6/8/14/16, avgpool 3.5/5.5/11.5/13.5, conv 6.5/7.5/10.5/11.5) +
  Q4_0 CPU vs референс + CPU↔GPU parity — ALL PASS; регресс smoke и gguf_fe_q
  PASS. Исправлены 2 бага в cpu_engine: пулинг ошибочно редуцировал каналы
  (окно должно идти по тому же каналу) и конв-кернел читал `pool.kernel`
  вместо формы весов [O,C,KH,KW].
- 2026-08-15: автономная сборка ядра без openvino (14 .cpp → .obj, C++23
  `/std:c++latest /O2 /DNDEBUG`, локальный Vulkan-стек вместо SDK) + сквозной
  смоук `vk_core_smoke.cpp`: ir_graph → serialize_fb/deserialize_fb →
  vk_engine::create → vk_program_builder::build → vk_network::execute →
  readback → числовая проверка. Исправлены 2 бага жизнеспособности ядра:
  (1) `vk_engine` не удерживал `vk_device_ptr` — `vk_device` (и `VkDevice`)
  уничтожался при выходе из конструктора, все аллокации шли по висячему
  хендлу → краш 0xC0000005; добавлено поле `_device` + хендлы берутся из
  `_device->`. (2) парсер OpExecutionMode LocalSize в `vk_spirv_reflection.cpp`
  требовал word_count>=7, а LocalSize занимает 6 слов → `has_local_size`
  никогда не выставлялся, диспатчи шли бы с локальным размером 1,1,1; фикс
  word_count>=6. Третий «краш» в `vk_program_builder::build()` оказался
  следствием залипания в сборке (нулевой/устаревший `vk_program.obj`) — после
  пересборки пайплайн-граф (matmul+add+relu) строится и исполняется.
  Итог: все 7 нативных ядер (relu/add/matmul/matmul_transpose_b/maxpool/
  avgpool/conv2d) — `ALL PASS`, числа сходятся с эталоном.
- 2026-08-14: легаси intel_cpu (~1900 файлов) и intel_npu (~700 файлов)
  удалены из дерева накорню: директории плагинов, 7 записей .gitmodules,
  опции ENABLE_INTEL_CPU/ENABLE_INTEL_NPU + производные из features.cmake.
  Затем удалён и hetero (отдельный плагин, ENABLE_HETERO убрана) — ссылки
  только строковые. Конфиг пересобран без них, регресс smoke–smoke11 +
  fmt_test — все PASS. Для oneDNN-фазы используется уже выкачанный
  `intel_gpu/thirdparty/onednn_gpu` (oneapi-src/oneDNN) — без новых
  скачиваний.
- 2026-08-14: шаг 6 (маршрутизация по именам устройств) — каркас готов:
  `device_name` в конфиге ядра, фильтр физических устройств в
  `init_vk_engine_data` (GPU → не-CPU, CPU → CPU-тип, NPU → ошибка «No Vulkan
  device matching the device name '...'»), проброс имени через
  `engine::create(..., device_name)` ← `RemoteContextImpl::m_device_name`;
  `Plugin::device_matches_plugin_name` применён в available_devices /
  get_default_contexts / get_metric; убран хардкод `set_device_name("GPU")`
  в ctor (имя назначает рантайм — иначе core-проверка `device_name.find(
  plugin_name)` режет CPU/NPU); plugins.xml (GPU+CPU+NPU → один dll) пишется
  POST_BUILD-шагом CMake. Проверено smoke11: GPU.0/GPU.1 в списке устройств,
  GPU infer PASS, CPU/NPU — чистые ошибки; регресс smoke–smoke10 + fmt_test
  PASS. Дальше: регистрация CPU/NPU работает, реальные CPU-инференсы — на
  машинах с CPU-тип Vulkan-драйвером; NPU-класс (Vulkan SC) — по мере
  появления железа.
- 2026-08-14: шаг 3 завершён — платформенные точки входа `vk_engine`
  (desktop/moltenvk/vulkan_sc): `vk_platform_config` + `platform_config_from_env`
  в `vk_platform.hpp`, factory `create_vk_engine(device, runtime_type, config)`;
  MoltenVK-флаги включаются только при наличии расширений (на Windows
  фолбэк), portability subset включается при создании устройства; SC —
  оффлайн-экспорт `<kernel_id>.spv` + `vk_pipeline_cache.bin` через общий
  `VkPipelineCache`. Проверено smoke10 (3 конфигурации) + полный регресс
  smoke–smoke9, fmt_test — все PASS. В `vk_platform.hpp` зарезервирован
  флаг `nccl_bridge` (NCCL-мост, не реализован).
- 2026-08-14: ядро переведено на C++23: стандарт зафиксирован явно в
  `vulkan_core/CMakeLists.txt` (MSVC идёт на `-std:c++latest`); форматный
  модуль и пайплайн модернизированы (std::span/std::byte API у
  vk_graph_format, string_view в конвертере/программе, std::ranges,
  std::fold_left, constexpr-константы); смоуки собираются с /std:c++latest.
  Регресс: все прогоны PASS.
- 2026-08-14: шаг 2 завершён — FB/PB-конвертеры в ядре (vk_graph_format,
  zero-OV-зависимость), compiled blob intel_gpu переведён на FB
  (export/import без ov::Model); smoke9 + standalone fmt_test PASS.
  В план добавлена заметка о расширении набора операций для жизнеспособности.
- 2026-08-14: шаг 1 завершён — ядро работает end-to-end: relu/add/matmul
  (+transpose_b, +broadcast-констант)/maxpool/avgpool/conv2d (pad, stride,
  batch) проверены 13 смоук-прогонами.
- 2026-08-14: исправлен порядок I/O-портов (get_parameters/get_results);
  выводные буферы нод заполнялись в vk_program build (node->outputs);
  константы переведены на host-visible.
- 2026-08-14: сделан IR-декаплинг (vk_ir.hpp, VkModelConverter, сборка ok).
- 2026-08-14: зафиксирован план; старт шага 1.

## Проверка бенчмарков с oneDNN на Vulkan

- ResNet
- YOLO
- BERT

## Отсоеденить core в отдельный модуль openvino_core, а биндинги в openvino сделать в продакшене

## Текущий статус ядра (август 2026)

Плагин собирается, инференс идёт целиком по Vulkan-рантайму ядра
(`vk_engine`/`vk_stream`/`vk_program`/`vk_network`, кернелы компилируются
glslangValidator → clspv SPIR-V, зашиты при сборке в `spirv_kernels.inc`).

### Поддерживаемые операции (f32, NCHW/2D)

| Операция | Кернел | Ограничения | Проверено |
|---|---|---|---|
| Relu | `relu_f32` | — | smoke, ops_smoke |
| Add | `eltwise_add_f32` | constant-broadcast входы материализуются ядром; динамические — ошибка | smoke, smoke2, ops_smoke |
| Mul / Sub / Div | `eltwise_mul/sub/div_f32` | same-shape или constant-broadcast (разворачивает ядро) | ops_smoke |
| Sigmoid / Tanh | `sigmoid_f32` / `tanh_f32` | — | ops_smoke |
| LeakyRelu | `leaky_relu_f32` | alpha — push-константа (FB v2) | ops_smoke |
| GELU | `gelu_f32` | tanh-аппроксимация | nn_ops |
| QuickGELU | `quick_gelu_f32` | x·sigmoid(1.702x) | shape_ops |
| SwiGLU | `swiglu_f32` | silu(a)*b; constant-broadcast допустим | nn_ops |
| RMSNorm | `rms_norm_f32` | последняя ось (контiguous lines); weight [axis]; eps = alpha | shape_ops |
| Pad | `pad_f32` | constant fill (alpha), pads_begin/pads_end по осям, до 8D | shape_ops |
| Transpose | `transpose_f32` | 1..8D, произвольная перестановка осей; push = 24 скаляра (dims/strides/perm) | ops_smoke |
| Reshape | — (алиас памяти) | плоские f32-буферы переинтерпретируются без копирования; reshape от непроизводимого тензора не может быть выходом модели | ops_smoke |
| Concat | `slab_copy_f32` ×k + identity | произвольная ось; собирается цепочкой слэб-копий + финальная полная копия (readback) | ops_smoke |
| Softmax | `softmax_f32` | произвольная ось ([outer,len,inner], нить на строку), стабильный max-вычит | ops_smoke, shape_ops, attention |
| Reduce Mean/Sum/Max | `reduce_f32` (mode-push) | ось удаляется (keep_dims=false) | ops_smoke |
| Crop | `crop_f32` | окно по осям; begin = pads_begin; до 8D | attention |
| CausalSoftmax | `causal_softmax_f32` | последняя ось [...,L,L]; j>i → 0 | llm |
| RoPE | `rope_f32` | halves-конвенция; x [...,H,D], cos/sin x_dims[:-2]+[D/2] | llm |
| CacheWrite | `cache_write_f32` | функциональный append KV: out=cache с rows [pos,pos+L); pos = axis | llm |
| ArgMax | `argmax_f32` | последняя ось, индекс как f32, первая ничья | llm |
| MatMul | `matmul_f32` / `matmul_transpose_b_f32` / `matmul_q_f32` / `matmul_batched(+tb)_f32` / `matmul_q_batched_f32` / `matmul_bb_f32` / `*(+tiled)` | 2D `[M,K]x[K,N]`, 3D-батч shared, попарный батч + **GQA** (`Bb=1`); f32 ≥16³ автоматически идёт в tiled-кернелы (shared memory 16×16; 129–162 GFLOP/s на тестовом GPU); Q4_0…Q8_0 — девант в шейдере (2D/3D-shared); fc = reshape-алиас + matmul | все смоки + perf |
| Convolution | `conv2d_f32` | 2D, stride/pad произвольные, dilation=1, batch произвольный; **ровно 3 входа** (data, weights, bias — нулевой bias материализует загрузчик) | smoke5–smoke8, nn_ops |

- I/O-порты модели маппятся строго по порядку `get_parameters()` /
  `get_results()` (порядок `get_ordered_ops()` не гарантирован — была ошибка
  перепутанных входов).
- Входные тензоры пользователя: zero-copy (shared USM host buffer) при
  совместимом layout, иначе копия. Константы — host-visible, записываются
  `mem_lock` на этапе build.
- Не поддерживается (явная ошибка `unsupported op` вместо неверных чисел):
  `transpose_a` в MatMul, dilation≠1 в conv, `exclude_pad=false` в AvgPool.

### Смоук-тесты

**Живут в репо:** `src/core/tests/` (`build_tests.bat` собирает все
`test_*.cpp` → `build/*.exe`; bootstrap тулчейна — `tools/setup_vktools.ps1`,
подробности в `src/core/tests/README.md`).

- `test_ops_smoke.exe` — все 18 `ir_op` против эталонов на CPU и GPU,
  dispatch CPU/GPU/GPU.0/чистая ошибка NPU, FB/PB round-trip
  (alpha/axis/perm/quant_constants), broadcast-bias — **ALL PASS 37**.
- Ранние прогоны (smoke–smoke11, gguf_fe/gguf_fe_q/gguf_cross/paddle_fe,
  fmt_test) выполнялись из временной папки `vksmoke` и не пережили чистку
  temp; их покрытие поглощено `test_ops_smoke`, кроме тестов самих ридеров
  (GGUF/Paddle) — восстановить как `test_gguf_fe.cpp` / `test_paddle_fe.cpp`
  с генерацией мини-моделей в коде.
