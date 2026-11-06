// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include "intel_npu/ops/flash_attention_tile.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "ov_ops/dynamic_quantize.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

#define OP_EXTENSION(NAME) std::make_shared<ov::OpExtension<NAME>>(),

#define NPU_SUPPORTED_EXTENSIONS                        \
    OP_EXTENSION(ov::op::internal::RMS)                 \
    OP_EXTENSION(ov::op::internal::RoPE)                \
    OP_EXTENSION(ov::op::internal::GroupQueryAttention) \
    OP_EXTENSION(ov::op::internal::DynamicQuantize)     \
    OP_EXTENSION(ov::op::internal::MOECompressed)       \
    OP_EXTENSION(ov::intel_npu::op::FlashAttentionTile)

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>({NPU_SUPPORTED_EXTENSIONS}));
