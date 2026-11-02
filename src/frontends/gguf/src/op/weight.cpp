// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "helper_ops/gguf_constant.hpp"
#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "quant/deferred_weights.hpp"
#include "quant/weights.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

namespace {

std::shared_ptr<ov::Node> materialize_raw_weight(const ov::Tensor& data,
                                                 const std::string& quant_type,
                                                 const ov::Shape& shape,
                                                 const std::string& name) {
    if (shape.size() > 2 && quant_type == "MXFP4") {
        const ov::Shape packed_shape{1, shape[shape.size() - 3], shape[shape.size() - 2], shape.back() / 32, 17};
        return std::make_shared<ov::op::v0::Constant>(ov::element::u8, packed_shape, data.data());
    }
    if (shape.size() > 2) {
        const size_t cols = shape.back();
        const size_t rows = std::accumulate(shape.begin(), shape.end() - 1, size_t{1}, std::multiplies<size_t>());
        auto node = make_weight_node(data, quant_type, ov::Shape{rows, cols}, name);
        std::vector<int64_t> full(shape.begin(), shape.end());
        auto target = ov::op::v0::Constant::create(ov::element::i64, {full.size()}, full);
        return std::make_shared<ov::op::v1::Reshape>(node, target, false);
    }
    return make_weight_node(data, quant_type, shape, name);
}

}  // namespace

// Keep quantized payloads deferred until weight normalization.
OutputVector translate_weight(const NodeContext& context) {
    if (context.get_attribute<bool>("gguf_weight", false)) {
        WeightTensors tensors{
            context.get_attribute<ov::Tensor>("gguf.blob.weight", ov::Tensor()),
            context.get_attribute<ov::Tensor>("gguf.blob.scales", ov::Tensor()),
            context.get_attribute<ov::Tensor>("gguf.blob.zp", ov::Tensor()),
        };
        FRONT_END_OP_CONVERSION_CHECK(tensors.weight, "GGML_OP_NONE weight leaf has no 'gguf.blob.weight' attribute");
        auto qtype = static_cast<GgufTensorType>(context.get_attribute<int>("gguf_qtype"));
        if (is_deferred(tensors) || tensors.scales) {
            auto node = std::make_shared<GGUFConstant>(tensors, qtype, context.get_name());
            return rename_outputs_with_suffix({node}, context.get_name());
        }
        auto node = make_weight_node(tensors, qtype, context.get_name());
        return rename_outputs_with_suffix({std::move(node)}, context.get_name());
    }

    auto data = context.get_attribute<ov::Tensor>("data");
    FRONT_END_OP_CONVERSION_CHECK(data, "GGML_OP_NONE node has no weight payload; not a weight");
    auto quant_type = context.get_attribute<std::string>("quant_type");
    auto shape = context.get_output_shape().to_shape();
    auto logical_shape = shape;
    auto type = ov::element::f32;

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
        logical_shape = packed_shape;
        type = ov::element::u8;
    }

    auto materialize = [data, quant_type, shape, name = context.get_name()] {
        return materialize_raw_weight(data, quant_type, shape, name);
    };
    auto node = std::make_shared<GGUFConstant>(logical_shape,
                                               type,
                                               gguf_type_from_name(quant_type),
                                               context.get_name(),
                                               std::move(materialize));
    return rename_outputs_with_suffix({std::move(node)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
