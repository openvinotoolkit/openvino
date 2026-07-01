// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node_context.h"

namespace ov {
namespace frontend {
namespace gguf {

namespace op {

#define GGUF_OP_CONVERTER(op) OutputVector op(const NodeContext& context)

GGUF_OP_CONVERTER(translate_add);
GGUF_OP_CONVERTER(translate_cont);
GGUF_OP_CONVERTER(translate_get_rows);
GGUF_OP_CONVERTER(translate_mul);
GGUF_OP_CONVERTER(translate_mulmat);
GGUF_OP_CONVERTER(translate_permute);
GGUF_OP_CONVERTER(translate_reshape);
GGUF_OP_CONVERTER(translate_rms_norm);
GGUF_OP_CONVERTER(translate_rope);
GGUF_OP_CONVERTER(translate_scale);
GGUF_OP_CONVERTER(translate_unary_silu);
GGUF_OP_CONVERTER(translate_unary_gelu);
GGUF_OP_CONVERTER(translate_soft_max);
GGUF_OP_CONVERTER(translate_transpose);
GGUF_OP_CONVERTER(translate_view);
GGUF_OP_CONVERTER(translate_glu_swiglu);
GGUF_OP_CONVERTER(translate_glu_geglu);
GGUF_OP_CONVERTER(translate_set_rows);
GGUF_OP_CONVERTER(translate_cpy);
GGUF_OP_CONVERTER(translate_flash_attn_ext);
GGUF_OP_CONVERTER(translate_weight);

}  // namespace op

std::unordered_map<std::string, CreatorFunction> get_supported_ops();

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
