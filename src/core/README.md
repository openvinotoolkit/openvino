# openvino-core: standalone Vulkan compute core

`ov::core::vulkan::cross_platform` — автономное ядро инференса: свой IR
(`vk_ir.hpp`), Vulkan-рантайм, CPU-исполнитель того же IR, конвертеры моделей
(GGUF / Paddle / Safetensors) и бинарный формат (FB/PB v5). Ни `ov::Model`,
ни вендорских SDK.

```
модель (GGUF | Paddle | Safetensors) ─┐
                                      ├→ ir_graph → пассы → GPU (Vulkan)
обученный граф (FB/PB blob) ──────────┘          ↘ CPU (cpu_engine)
```

## Точки входа

| Хедер | Что даёт |
|---|---|
| `vk_dispatch.hpp` | `vk_execute(graph, inputs, "GPU[.N]"\|"CPU")`, `vk_available_devices()` |
| `vk_pass.hpp` | `optimize(graph)` — dce + constant folding + peepholes |
| `gguf_reader.hpp` / `safetensors_reader.hpp` / `paddle_reader.hpp` | веса/графы → `ir_graph` |
| `vk_graph_format.hpp` | FB/PB сериализация (спека: `FORMAT.md`) |
| `vk_platform.hpp` | desktop / MoltenVK / Vulkan SC конфиги |

28 операций (f32): elementwise, активации GPT-семейства (gelu/swiglu/
quick_gelu/rms_norm), shape-ops (transpose/reshape-алиас/concat/pad/crop),
редукции, softmax (любая ось, causal-вариант), matmul-семейство (2D /
batched-shared / pairwise+GQA / quant Q4_0…Q8_0 с девантом в шейдере,
shared-memory tiling 16×16 — 129–162 GFLOP/s на референс-GPU), RoPE,
KV-cache append, argmax.

## Сборка

CMake (`src/core/CMakeLists.txt`) требует только glslangValidator и Vulkan;
без SDK — bootstrap из `tests/tools/setup_vktools.ps1`.

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DGLSLANG=...
cmake --build build --config Release
ctest --test-dir build -C Release
```

Тесты и примеры: `tests/README.md`. Спецификация формата: `FORMAT.md`.
Дорожная карта проекта: `../VULKAN_CROSS_PLATFORM.md`.
