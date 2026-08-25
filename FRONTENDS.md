# Frontends → standalone core: план перевода

Принцип: ядро (`src/core`) не знает про `ov::Model`. Каждый фронтенд получает
**lowering-слой** `ov::Model → vk::ir_graph` (по образцу `VkModelConverter`,
вынесенный рядом с фронтендом). Конвертеры файлов (GGUF/Safetensors/Paddle)
остаются в ядре — они не зависят от OV.

## Порядок работ

| Фаза | Фронтенд | Стратегия |
|---|---|---|
| 1 | **pytorch** (409 aten ops) | lowering поверх существующего графа; покрыть топ-30 ops, остальное — чистая ошибка со списком недостающих |
| 2 | tensorflow / tensorflow_common | тот же lowering переиспользуется: TF-фронтенд тоже даёт ov::Model |
| 3 | tensorflow_lite | после TF (делится common) |
| 4 | jax | самый молодой фронтенд, последним |

Ключ: **один общий lowering** (`src/frontends/core_bridge/`) обслуживает все
четыре фронтенда — различается только набор узлов, приходящий от каждого.

## Покрытие aten → core (топ-операции инференса)

| aten | core op | статус |
|---|---|---|
| relu_ / sigmoid_ / tanh_ | relu/sigmoid/tanh | ✅ |
| gelu (tanh approx) / silu_and_mul? | gelu / swiglu | ✅ |
| mul / add / sub / div_ | mul/sub/div/add | ✅ |
| leaky_relu_ | leaky_relu | ✅ |
| transpose / permute | transpose | ✅ |
| view / reshape / flatten | reshape (алиас) | ✅ |
| cat / narrow / slice(step=1) | concat / crop | ✅ concat; slice-step ❌ |
| softmax(dim) | softmax | ✅ |
| mean/sum/max(dim, keepdim=False) | reduce_* | ✅ keepdim=True ❌ |
| mm / matmul(2D,3D-shared,3D-pair,GQA) | matmul семейство | ✅ |
| linear (flatten+mm+bias) | reshape+matmul+bcast-add | ✅ |
| conv2d (+bias) | convolution | ✅ (dilation=1, groups=1) |
| max_pool2d / avg_pool2d(excl) | max_pool / avg_pool | ✅ |
| pad (constant) | pad | ✅ |
| rms_norm (LLM) | rms_norm | ✅ |
| apply_rotary_pos_emb | rope | ✅ (halves) |
| kv-cache append | cache_write | ✅ |
| argmax | argmax | ✅ |
| scaled_dot_product_attention | композиция (scale→bb-mm→softmax→mm) | ✅ |
| pow/exp/neg/clamp/min/max-eltwise | ❌ батч 10 | элементарные кернелы |
| where / masked_fill | ❌ батч 10 | select-кернел |
| layer_norm (mean/var) | ❌ через composition или кернел | батч 10 |
| embedding (gather по строкам) | ❌ батч 10 | gather-кернел |
| cumsum / sort / topk | ❌ батч 11 | редукции-сканы |

## Дорожная карта снижения риска

1. `core_bridge` принимает `ov::Model`, идёт по топологическому порядку,
   маппит тип узла → ir_op, константы → constant_data, формы берёт из
   рантайм-инфо узла; неподдержанный тип → исключение со списком
   «top-N отсутствующих» за прогон.
2. Приёмочный тест каждой фазы: resnet18-подобная и mini-GPT-подобная
   цепочки, parity CPU(core) vs CPU(OV-legacy путь удалён → vs эталон).
3. Числа бенчей (#2) снимаем на тех же графах до/после фронтендов.
