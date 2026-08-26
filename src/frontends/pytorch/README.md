# PyTorch frontend (standalone core)

Фронтенд переписан под автономное ядро: **никакого `ov::Model`, никакой
линковки с openvino**. Это python-экспортёр + ридер на стороне ядра.

```
torch.nn.Module
  └─ tools/torch_export.py   (torch.jit.trace, torch >= 2.13)
       ├─ <model>.graph.vktorch       — OP/CONST/OUT построчно
       └─ <model>.weights.safetensors — все параметры F32
  └─ src/core/pytorch_reader.hpp (pt_r::load_export)
       └─ bridge::graph → lower() → ir_graph → vk_execute("GPU"|"CPU")
```

## Использование

```python
import torch, torch_export
from my_model import MyModel

m = MyModel().eval()
torch_export.export_with_reference(
    m, (torch.randn(1, 3, 224, 224),),
    "model.graph.vktorch", "model.weights.safetensors", "expected.txt")
```

```cpp
auto g = pt_r::load_export("model.graph.vktorch", "model.weights.safetensors");
auto out = vk_execute(g, {{"in_0", x}}, "GPU");
```

## Покрытие

25 aten-киндов → 28 ops ядра (см. `_KIND_MAP` в экспортёре): elementwise,
активации GPT-семейства, shape-ops, matmul-семейство, conv2d, pooling,
softmax/reduce/argmax/pad/narrow. `nn.Linear` раскладывается в
`matmul(transpose_b) + bias-add` (broadcast-расширение ядра).

Непокрытое: экспортёр падает с точным именем op — расширяется одной строкой
в `_KIND_MAP` + (при необходимости) кернел по дорожной карте ядра.
Control-flow (if/loop) не поддерживается — трассируйте без ветвлений.

## Тесты

- `src/core/tests/test_torch.cpp` — бандл как из экспортёра, CPU+GPU parity
- `src/core/tests/test_torch_live.cpp` — **живая** модель (TinyMLP, torch
  2.13): вывод ядра == torch eager до 6 знаков на CPU и GPU

Старый OV-фронтенд (409 конвертеров, op/transforms/helper_ops) удалён —
история в git.
