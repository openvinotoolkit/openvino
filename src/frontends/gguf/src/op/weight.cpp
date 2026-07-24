// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <memory>
#include <numeric>
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "quant/weights.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// A GGUF weight surfaced as a node. A weight is a ggml leaf (op type "GGML_OP_NONE") that the
// decoder marks by exposing a "data" attribute (the raw weight bytes), alongside the ggml quant
// type name and the logical shape. The frontend does all dequant / repacking here, so the
// decoder never builds OV nodes itself. (Model-input leaves are also GGML_OP_NONE, but they are
// resolved to Parameters before the graph walk and never reach this translator.)
OutputVector translate_weight(const NodeContext& context) {
    auto data = context.get_attribute<ov::Tensor>("data");
    FRONT_END_OP_CONVERSION_CHECK(data, "GGML_OP_NONE node has no 'data' attribute; not a weight");
    auto quant_type = context.get_attribute<std::string>("quant_type");
    auto shape = context.get_output_shape().to_shape();

    // MoE MXFP4 expert weights stay PACKED: MUL_MAT_ID gathers the selected expert and dequantizes
    // on-graph, so materializing all experts to f32 here would waste memory. Surface the raw bytes
    // as a rank-5 u8 constant [1, n_expert, m, k_blocks, 17] (17 = 1 e8m0 scale byte + 16 nibble
    // bytes per 32-element block), which the mul_mat_id MXFP4 path recognizes.
    if (shape.size() > 2 && quant_type == "MXFP4") {
        constexpr size_t kQk = 32, kBlockBytes = 17;
        const size_t n_expert = shape[shape.size() - 3];
        const size_t m = shape[shape.size() - 2];
        const size_t k = shape.back();
        FRONT_END_OP_CONVERSION_CHECK(k % kQk == 0, "MXFP4 MoE expert cols must be a multiple of 32");
        const size_t k_blocks = k / kQk;
        ov::Shape packed_shape{1, n_expert, m, k_blocks, kBlockBytes};
        FRONT_END_OP_CONVERSION_CHECK(data.get_byte_size() == n_expert * m * k_blocks * kBlockBytes,
                                      "MXFP4 MoE packed byte size mismatch");
        auto packed = std::make_shared<ov::op::v0::Constant>(ov::element::u8, packed_shape, data.data());
        return rename_outputs_with_suffix({packed}, context.get_name());
    }

    // MoE expert weights are rank > 2 ([1, n_expert, m, k]). The dequant path works on a 2D
    // [rows, cols] tensor, so flatten the leading dims to rows, dequantize, then reshape the f32
    // result back to the full expert shape for MUL_MAT_ID. (Regular weights are already 2D.)
    if (shape.size() > 2) {
        const size_t cols = shape.back();
        const size_t rows = std::accumulate(shape.begin(), shape.end() - 1, size_t{1}, std::multiplies<size_t>());
        auto node = make_weight_node(data, quant_type, ov::Shape{rows, cols}, context.get_name());
        std::vector<int64_t> full(shape.begin(), shape.end());
        auto target = ov::op::v0::Constant::create(ov::element::i64, {full.size()}, full);
        auto reshaped = std::make_shared<ov::op::v1::Reshape>(node, target, false);
        return rename_outputs_with_suffix({reshaped}, context.get_name());
    }

    auto node = make_weight_node(data, quant_type, shape, context.get_name());
    return rename_outputs_with_suffix({node}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
