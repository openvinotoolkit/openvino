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
4. [~] **Маршрутизация по именам устройств**: один кросплатформенный плагин
       регистрируется как `GPU`/`CPU`/`NPU` — ядро само выбирает физическое
       устройство по классу (GPU → любой не-CPU Vulkan-девайс, CPU → CPU-тип
       драйвера (SwiftShader/Lavapipe), NPU → класс пока не существует, запрос
       падает с чистой ошибкой). Регресс: intel_cpu/intel_npu снимаются,
       остаётся только openvino_crossplatform.dll.
       *(сделано: `vk_platform_config.device_name` + фильтр физических устройств
       в `init_vk_engine_data` (throw вместо fallback, когда нет совпадения);
       `engine::create(..., device_name)` и проброс `m_device_name` из
       `RemoteContextImpl::initialize`; фильтр в `Plugin`: `available_devices`,
       `get_default_contexts`, `get_metric` (device_matches_plugin_name —
       dynamic_cast к `vk_device`, `is_cpu_device()`); контекст без устройств
       кидает «No Vulkan device matching ...»; ctor плагина больше не
       хардкодит `set_device_name("GPU")` — имя назначает рантайм, иначе
       core-проверка «plugin name ⊂ device name» режет CPU/NPU; plugins.xml
       пишется POST_BUILD-шагом с тремя записями на один dll. Проверено
       smoke11: GPU PASS, CPU/NPU — чистые ошибки без устройств; полный
       регресс smoke–smoke10 + fmt_test PASS.)*

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


> **Жизнеспособность ядра:** текущий набор операций (relu/add/matmul/
> maxpool/avgpool/conv2d) покрывает только демо-цепочки. Для реальных
> моделей (ResNet/BERT и т.п.) набор необходимо расширять — в первую
> очередь reshape/transpose (без OV-пасса), concat,
> softmax, reductions (mean/sum/max), elementwise (mul/sub/div),
> activation (sigmoid/tanh/leaky_relu), GEMM-обёртки fc/matmul 4D,
> раздельные bias-константы для conv. Каждый новый op = `ir_op` +
> матчинг в конвертере + (как правило) один `.comp`-кернел + смоук-тест;
> границы проверяются на этапе конвертации, не на инференсе.

## Риски

- Пункт 4 — самый объёмный: `engine/stream/memory/kernel` тянут OV runtime API.
  Делается последним, после того как ядро заработает автономно через OV-плагин.

## Журнал

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

## Текущий статус ядра (август 2026)

Плагин собирается, инференс идёт целиком по Vulkan-рантайму ядра
(`vk_engine`/`vk_stream`/`vk_program`/`vk_network`, кернелы компилируются
glslangValidator → clspv SPIR-V, зашиты при сборке в `spirv_kernels.inc`).

### Поддерживаемые операции (f32, NCHW/2D)

| Операция | Кернел | Ограничения | Проверено |
|---|---|---|---|
| Relu | `relu_f32` | — | smoke |
| Add | `eltwise_add_f32` | broadcast-входы материализуются константой на этапе конвертера | smoke, smoke2 |
| MatMul | `matmul_f32` / `matmul_transpose_b_f32` | 2D, `transpose_a` не поддержан; `transpose_b` — отдельный кернел (веса `[N,K]`) | smoke2, smoke3 |
| MaxPool | `maxpool_f32` | 2D, stride/pad произвольные, batch произвольный | smoke4, smoke6, smoke7 |
| AvgPool | `avgpool_f32` | 2D, только `exclude_pad=true` (деление на число валидных ячеек) | smoke8 |
| Convolution | `conv2d_f32` | 2D, stride/pad произвольные, dilation=1, batch произвольный | smoke5–smoke8 |

- I/O-порты модели маппятся строго по порядку `get_parameters()` /
  `get_results()` (порядок `get_ordered_ops()` не гарантирован — была ошибка
  перепутанных входов).
- Входные тензоры пользователя: zero-copy (shared USM host buffer) при
  совместимом layout, иначе копия. Константы — host-visible, записываются
  `mem_lock` на этапе build.
- Не поддерживается (явная ошибка `unsupported op` вместо неверных чисел):
  `transpose_a` в MatMul, dilation≠1 в conv, `exclude_pad=false` в AvgPool.

### Смоук-тесты

`C:\Users\selem\AppData\Local\Temp\opencode\vksmoke\` (smoke–smoke10) — все
прогоны PASS: 2 инференса подряд, MatMul(+Add, bias-бродкаст), константные
веса через `Transpose`→`transpose_b`, conv/maxpool/avgpool с паддингом и
batch=2, цепочка Conv→Relu→MaxPool, экспорт/импорт compiled blob (smoke9),
standalone FB/PB round-trip (fmt_test, собирается без openvino),
платформенные конфигурации (smoke10: desktop / moltenvk-фолбэк на Windows /
vulkan_sc + оффлайн-экспорт артефактов).
