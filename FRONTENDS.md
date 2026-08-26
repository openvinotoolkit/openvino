# Frontends в†’ standalone core: РїР»Р°РЅ РїРµСЂРµРІРѕРґР°

РџСЂРёРЅС†РёРї: СЏРґСЂРѕ (`src/core`) РЅРµ Р·РЅР°РµС‚ РїСЂРѕ `ov::Model`. РљР°Р¶РґС‹Р№ С„СЂРѕРЅС‚РµРЅРґ РїРѕР»СѓС‡Р°РµС‚
**lowering-СЃР»РѕР№** `ov::Model в†’ vk::ir_graph` (РїРѕ РѕР±СЂР°Р·С†Сѓ `VkModelConverter`,
РІС‹РЅРµСЃРµРЅРЅС‹Р№ СЂСЏРґРѕРј СЃ С„СЂРѕРЅС‚РµРЅРґРѕРј). РљРѕРЅРІРµСЂС‚РµСЂС‹ С„Р°Р№Р»РѕРІ (GGUF/Safetensors/Paddle)
РѕСЃС‚Р°СЋС‚СЃСЏ РІ СЏРґСЂРµ вЂ” РѕРЅРё РЅРµ Р·Р°РІРёСЃСЏС‚ РѕС‚ OV.

## РџРѕСЂСЏРґРѕРє СЂР°Р±РѕС‚

| Р¤Р°Р·Р° | Р¤СЂРѕРЅС‚РµРЅРґ | РЎС‚СЂР°С‚РµРіРёСЏ |
|---|---|---|
| 1 | **pytorch** (409 aten ops) | lowering РїРѕРІРµСЂС… СЃСѓС‰РµСЃС‚РІСѓСЋС‰РµРіРѕ РіСЂР°С„Р°; РїРѕРєСЂС‹С‚СЊ С‚РѕРї-30 ops, РѕСЃС‚Р°Р»СЊРЅРѕРµ вЂ” С‡РёСЃС‚Р°СЏ РѕС€РёР±РєР° СЃРѕ СЃРїРёСЃРєРѕРј РЅРµРґРѕСЃС‚Р°СЋС‰РёС… |
| 2 | tensorflow / tensorflow_common | С‚РѕС‚ Р¶Рµ lowering РїРµСЂРµРёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ: TF-С„СЂРѕРЅС‚РµРЅРґ С‚РѕР¶Рµ РґР°С‘С‚ ov::Model |
| 3 | tensorflow_lite | РїРѕСЃР»Рµ TF (РґРµР»РёС‚СЃСЏ common) |
| 4 | jax | СЃР°РјС‹Р№ РјРѕР»РѕРґРѕР№ С„СЂРѕРЅС‚РµРЅРґ, РїРѕСЃР»РµРґРЅРёРј |

РљР»СЋС‡: **РѕРґРёРЅ РѕР±С‰РёР№ lowering** (`src/frontends/core_bridge/`) РѕР±СЃР»СѓР¶РёРІР°РµС‚ РІСЃРµ
С‡РµС‚С‹СЂРµ С„СЂРѕРЅС‚РµРЅРґР° вЂ” СЂР°Р·Р»РёС‡Р°РµС‚СЃСЏ С‚РѕР»СЊРєРѕ РЅР°Р±РѕСЂ СѓР·Р»РѕРІ, РїСЂРёС…РѕРґСЏС‰РёР№ РѕС‚ РєР°Р¶РґРѕРіРѕ.

## РџРѕРєСЂС‹С‚РёРµ aten в†’ core (С‚РѕРї-РѕРїРµСЂР°С†РёРё РёРЅС„РµСЂРµРЅСЃР°)

| aten | core op | СЃС‚Р°С‚СѓСЃ |
|---|---|---|
| relu_ / sigmoid_ / tanh_ | relu/sigmoid/tanh | вњ… |
| gelu (tanh approx) / silu_and_mul? | gelu / swiglu | вњ… |
| mul / add / sub / div_ | mul/sub/div/add | вњ… |
| leaky_relu_ | leaky_relu | вњ… |
| transpose / permute | transpose | вњ… |
| view / reshape / flatten | reshape (Р°Р»РёР°СЃ) | вњ… |
| cat / narrow / slice(step=1) | concat / crop | вњ… concat; slice-step вќЊ |
| softmax(dim) | softmax | вњ… |
| mean/sum/max(dim, keepdim=False) | reduce_* | вњ… keepdim=True вќЊ |
| mm / matmul(2D,3D-shared,3D-pair,GQA) | matmul СЃРµРјРµР№СЃС‚РІРѕ | вњ… |
| linear (flatten+mm+bias) | reshape+matmul+bcast-add | вњ… |
| conv2d (+bias) | convolution | вњ… (dilation=1, groups=1) |
| max_pool2d / avg_pool2d(excl) | max_pool / avg_pool | вњ… |
| pad (constant) | pad | вњ… |
| rms_norm (LLM) | rms_norm | вњ… |
| apply_rotary_pos_emb | rope | вњ… (halves) |
| kv-cache append | cache_write | вњ… |
| argmax | argmax | вњ… |
| scaled_dot_product_attention | РєРѕРјРїРѕР·РёС†РёСЏ (scaleв†’bb-mmв†’softmaxв†’mm) | вњ… |
| pow/exp/neg/clamp/min/max-eltwise | вќЊ Р±Р°С‚С‡ 10 | СЌР»РµРјРµРЅС‚Р°СЂРЅС‹Рµ РєРµСЂРЅРµР»С‹ |
| where / masked_fill | вќЊ Р±Р°С‚С‡ 10 | select-РєРµСЂРЅРµР» |
| layer_norm (mean/var) | вќЊ С‡РµСЂРµР· composition РёР»Рё РєРµСЂРЅРµР» | Р±Р°С‚С‡ 10 |
| embedding (gather РїРѕ СЃС‚СЂРѕРєР°Рј) | вќЊ Р±Р°С‚С‡ 10 | gather-РєРµСЂРЅРµР» |
| cumsum / sort / topk | вќЊ Р±Р°С‚С‡ 11 | СЂРµРґСѓРєС†РёРё-СЃРєР°РЅС‹ |

## Р”РѕСЂРѕР¶РЅР°СЏ РєР°СЂС‚Р° СЃРЅРёР¶РµРЅРёСЏ СЂРёСЃРєР°

1. `core_bridge` РїСЂРёРЅРёРјР°РµС‚ `ov::Model`, РёРґС‘С‚ РїРѕ С‚РѕРїРѕР»РѕРіРёС‡РµСЃРєРѕРјСѓ РїРѕСЂСЏРґРєСѓ,
   РјР°РїРїРёС‚ С‚РёРї СѓР·Р»Р° в†’ ir_op, РєРѕРЅСЃС‚Р°РЅС‚С‹ в†’ constant_data, С„РѕСЂРјС‹ Р±РµСЂС‘С‚ РёР·
   СЂР°РЅС‚Р°Р№Рј-РёРЅС„Рѕ СѓР·Р»Р°; РЅРµРїРѕРґРґРµСЂР¶Р°РЅРЅС‹Р№ С‚РёРї в†’ РёСЃРєР»СЋС‡РµРЅРёРµ СЃРѕ СЃРїРёСЃРєРѕРј
   В«top-N РѕС‚СЃСѓС‚СЃС‚РІСѓСЋС‰РёС…В» Р·Р° РїСЂРѕРіРѕРЅ.
2. РџСЂРёС‘РјРѕС‡РЅС‹Р№ С‚РµСЃС‚ РєР°Р¶РґРѕР№ С„Р°Р·С‹: resnet18-РїРѕРґРѕР±РЅР°СЏ Рё mini-GPT-РїРѕРґРѕР±РЅР°СЏ
   С†РµРїРѕС‡РєРё, parity CPU(core) vs CPU(OV-legacy РїСѓС‚СЊ СѓРґР°Р»С‘РЅ в†’ vs СЌС‚Р°Р»РѕРЅ).
3. Р§РёСЃР»Р° Р±РµРЅС‡РµР№ (#2) СЃРЅРёРјР°РµРј РЅР° С‚РµС… Р¶Рµ РіСЂР°С„Р°С… РґРѕ/РїРѕСЃР»Рµ С„СЂРѕРЅС‚РµРЅРґРѕРІ.

