// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_attention.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "intel_gpu/op/sparse_sdpa.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/broadcast.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/gather.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/shape_of.hpp"
#include "intel_gpu/primitives/sparse_sdpa.hpp"
#include "openvino/op/constant.hpp"

// Lowers ov::op::v17::SparseAttention (dense K/V history, no paging) to cldnn::sparse_sdpa, like
// CreateScaledDotProductAttentionOp lowers SDPA: the head geometry comes from the input shapes, a constant scale is
// folded into the primitive and the index inputs are converted to i32 when they are not.
//
// sparse_sdpa reads one key_length value per batch row, so the other forms the op allows are expanded here: an absent
// key_length is the key sequence length of every row, and a scalar or [1] key_length is broadcast over the batch.
//
// Supported forms: query_block_size == 1, static head counts / head sizes (see SparseSDPA::validate_config), f16
// query / key / value, a Constant scale and no sinks. SparseAttention that does not fit is reported.
// A quantized KV cache is handled by KVCacheCompression, which replaces the op with the GPU SparseSDPA op.

namespace ov::intel_gpu {

namespace {

// Inputs of ov::op::v17::SparseAttention.
struct SparseAttentionIdx {
    static constexpr size_t QUERY = 0;
    static constexpr size_t KEY = 1;
    static constexpr size_t VALUE = 2;
    static constexpr size_t BLOCK_INDICES = 3;
    static constexpr size_t BLOCK_COUNT = 4;
    static constexpr size_t KEY_LENGTH = 5;
    static constexpr size_t SCALE = 6;
    static constexpr size_t SINKS = 7;
};

size_t static_dim(const std::shared_ptr<ov::op::v17::SparseAttention>& op, size_t input, size_t axis, const char* what) {
    const auto& shape = op->get_input_partial_shape(input);
    OPENVINO_ASSERT(shape.rank().is_static() && shape.rank().get_length() == 4 && shape[axis].is_static(),
                    "[GPU] SparseAttention '",
                    op->get_friendly_name(),
                    "': ",
                    what,
                    " must be static");
    return static_cast<size_t>(shape[axis].get_length());
}

cldnn::input_info add_i32_data(ProgramBuilder& p, const ov::Node& op, const cldnn::primitive_id& name, const std::vector<int32_t>& values) {
    const cldnn::layout layout{ov::PartialShape{static_cast<int64_t>(values.size())}, cldnn::data_types::i32, cldnn::format::bfyx};
    auto mem = p.get_engine().allocate_memory(layout, false);
    {
        cldnn::mem_lock<int32_t> host{mem, p.get_engine().get_service_stream()};
        std::copy(values.begin(), values.end(), host.data());
    }
    p.add_primitive(op, cldnn::data(name, mem));
    return cldnn::input_info(name);
}

// [1] i32 tensor holding dimension `axis` of `input` (ShapeOf + Gather).
cldnn::input_info add_dim_of(ProgramBuilder& p, const ov::Node& op, const cldnn::input_info& input, int64_t axis, const std::string& name) {
    p.add_primitive(op, cldnn::shape_of(name + "_shape", input, cldnn::data_types::i32));
    const auto index = add_i32_data(p, op, name + "_axis", {static_cast<int32_t>(axis)});
    p.add_primitive(op, cldnn::gather(name, cldnn::input_info(name + "_shape"), index, 0, 1, ov::Shape{}));
    return cldnn::input_info(name);
}

// The i32 [B] key_length sparse_sdpa reads, see the file description. `key_length` is the converted op input, if any.
cldnn::input_info per_row_key_length(ProgramBuilder& p,
                                     const std::shared_ptr<ov::op::v17::SparseAttention>& op,
                                     const std::vector<cldnn::input_info>& inputs,
                                     const std::string& layer_name) {
    using Idx = SparseAttentionIdx;
    const auto& batch = op->get_input_partial_shape(Idx::QUERY)[0];
    const bool has_key_length = op->get_input_size() > Idx::KEY_LENGTH;
    if (has_key_length) {
        const auto& shape = op->get_input_partial_shape(Idx::KEY_LENGTH);
        if (shape.rank().is_static() && shape.size() == 1 && shape[0].is_static() && batch.is_static() && shape[0] == batch) {
            return inputs[Idx::KEY_LENGTH];
        }
    }

    const auto& key_shape = op->get_input_partial_shape(Idx::KEY);
    if (!has_key_length && key_shape[2].is_static() && batch.is_static()) {
        // Fully static: every row holds the whole key sequence.
        return add_i32_data(p,
                            *op,
                            layer_name + "_key_length",
                            std::vector<int32_t>(static_cast<size_t>(batch.get_length()), static_cast<int32_t>(key_shape[2].get_length())));
    }

    const auto source = has_key_length ? inputs[Idx::KEY_LENGTH] : add_dim_of(p, *op, inputs[Idx::KEY], 2, layer_name + "_key_seq_len");
    const auto name = layer_name + "_key_length";
    std::shared_ptr<cldnn::broadcast> broadcast;
    if (batch.is_static()) {
        broadcast =
            std::make_shared<cldnn::broadcast>(name, source, ov::Shape{static_cast<size_t>(batch.get_length())}, ov::AxisSet{}, ov::op::BroadcastType::NUMPY);
    } else {
        const auto target = add_dim_of(p, *op, inputs[Idx::QUERY], 0, layer_name + "_batch");
        broadcast = std::make_shared<cldnn::broadcast>(name, source, target, ov::AxisSet{}, ov::op::BroadcastType::NUMPY);
    }
    broadcast->output_pshape = ov::PartialShape{batch};
    p.add_primitive(*op, *broadcast);
    return cldnn::input_info(name);
}

}  // namespace

static void CreateSparseAttentionOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v17::SparseAttention>& op) {
    using SparseSDPA = ov::intel_gpu::op::SparseSDPA;
    using Idx = SparseAttentionIdx;
    validate_inputs_count(op, {Idx::BLOCK_COUNT + 1, Idx::KEY_LENGTH + 1, Idx::SCALE + 1, Idx::SINKS + 1});
    const auto layer_name = layer_type_name_ID(op);

    OPENVINO_ASSERT(op->get_query_block_size() == 1, "[GPU] SparseAttention '", layer_name, "': only query_block_size == 1 is supported");
    OPENVINO_ASSERT(op->get_input_size() <= Idx::SINKS, "[GPU] SparseAttention '", layer_name, "': the sinks input is not supported by the GPU implementation");

    SparseSDPA::Config config;
    config.num_heads = static_dim(op, Idx::QUERY, 1, "the query head count");
    config.k_head_size = static_dim(op, Idx::QUERY, 3, "the query head size");
    config.num_kv_heads = static_dim(op, Idx::KEY, 1, "the key head count");
    config.v_head_size = static_dim(op, Idx::VALUE, 3, "the value head size");
    config.num_selection_heads = static_dim(op, Idx::BLOCK_INDICES, 1, "the selection head count");
    config.key_block_size = static_cast<size_t>(op->get_key_block_size());
    config.causal = op->get_causal();
    std::string reason;
    OPENVINO_ASSERT(SparseSDPA::validate_config(config, reason), "[GPU] Cannot lower SparseAttention '", layer_name, "': ", reason);
    OPENVINO_ASSERT(config.num_selection_heads == 1,
                    "[GPU] SparseAttention '",
                    layer_name,
                    "': only num_selection_heads == 1 is supported, got ",
                    config.num_selection_heads);
    for (const size_t idx : {Idx::QUERY, Idx::KEY, Idx::VALUE}) {
        OPENVINO_ASSERT(op->get_input_element_type(idx) == ov::element::f16,
                        "[GPU] SparseAttention '",
                        layer_name,
                        "': input ",
                        idx,
                        " must be f16, got ",
                        op->get_input_element_type(idx));
    }

    float scale = 1.0F / std::sqrt(static_cast<float>(config.k_head_size));
    if (op->get_input_size() > Idx::SCALE) {
        const auto scale_const = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(Idx::SCALE));
        OPENVINO_ASSERT(scale_const && ov::shape_size(scale_const->get_shape()) == 1,
                        "[GPU] SparseAttention '",
                        layer_name,
                        "': the scale input must be a one-element Constant");
        scale = scale_const->cast_vector<float>()[0];
    }

    auto inputs = p.GetInputInfo(op);
    for (const size_t idx : {Idx::BLOCK_INDICES, Idx::BLOCK_COUNT, Idx::KEY_LENGTH}) {
        if (idx >= op->get_input_size() || op->get_input_element_type(idx) == ov::element::i32) {
            continue;
        }
        const auto convert_name = layer_name + "_i32_input_" + std::to_string(idx);
        p.add_primitive(*op, cldnn::reorder(convert_name, inputs[idx], cldnn::format::any, cldnn::data_types::i32));
        inputs[idx] = cldnn::input_info(convert_name);
    }

    // The primitive takes the six core inputs; the scale is folded into it.
    std::vector<cldnn::input_info> prim_inputs(inputs.begin(), inputs.begin() + Idx::KEY_LENGTH);
    prim_inputs.push_back(per_row_key_length(p, op, inputs, layer_name));

    cldnn::sparse_sdpa prim(layer_name, prim_inputs, config, scale);
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v17, SparseAttention);

}  // namespace ov::intel_gpu
