// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_sparse_attention_to_sparse_sdpa.hpp"

#include <memory>
#include <optional>
#include <string>

#include "intel_gpu/op/sparse_sdpa.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/pass/node_registry.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"

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

std::optional<size_t> static_dim(const ov::PartialShape& shape, int64_t rank, size_t axis) {
    if (shape.rank().is_dynamic() || shape.rank().get_length() != rank || shape[axis].is_dynamic()) {
        return std::nullopt;
    }
    return static_cast<size_t>(shape[axis].get_length());
}

}  // namespace

ConvertSparseAttentionToSparseSDPA::ConvertSparseAttentionToSparseSDPA() {
    using namespace ov::pass::pattern;
    using SparseSDPA = ov::intel_gpu::op::SparseSDPA;

    const auto p_attention = wrap_type<ov::op::v17::SparseAttention>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto attention = ov::as_type_ptr<ov::op::v17::SparseAttention>(m.get_match_root());
        if (!attention || transformation_callback(attention) || attention->get_query_block_size() != 1) {
            return false;
        }

        const auto num_heads = static_dim(attention->get_input_partial_shape(SparseAttentionIdx::QUERY), 4, 1);
        const auto k_head_size = static_dim(attention->get_input_partial_shape(SparseAttentionIdx::QUERY), 4, 3);
        const auto num_kv_heads = static_dim(attention->get_input_partial_shape(SparseAttentionIdx::KEY), 4, 1);
        const auto v_head_size = static_dim(attention->get_input_partial_shape(SparseAttentionIdx::VALUE), 4, 3);
        const auto selection_heads = static_dim(attention->get_input_partial_shape(SparseAttentionIdx::BLOCK_INDICES), 4, 1);
        if (!num_heads || !k_head_size || !num_kv_heads || !v_head_size || !selection_heads) {
            return false;
        }

        SparseSDPA::Config config;
        config.num_heads = *num_heads;
        config.num_kv_heads = *num_kv_heads;
        config.num_selection_heads = *selection_heads;
        config.k_head_size = *k_head_size;
        config.v_head_size = *v_head_size;
        config.key_block_size = static_cast<size_t>(attention->get_key_block_size());
        config.causal = attention->get_causal();
        std::string reason;
        if (!SparseSDPA::validate_config(config, reason)) {
            return false;
        }

        const auto input_count = attention->get_input_size();
        // SparseSDPA takes a scalar or 1-element scale and a [num_heads] sinks input.
        if (input_count > SparseAttentionIdx::SCALE) {
            const auto& scale_shape = attention->get_input_partial_shape(SparseAttentionIdx::SCALE);
            if (scale_shape.is_static() && ov::shape_size(scale_shape.to_shape()) != 1) {
                return false;
            }
        }
        if (input_count > SparseAttentionIdx::SINKS &&
            !attention->get_input_partial_shape(SparseAttentionIdx::SINKS).compatible(ov::PartialShape{static_cast<int64_t>(config.num_heads)})) {
            return false;
        }

        ov::pass::NodeRegistry rg;
        const auto to_i32 = [&rg](const ov::Output<ov::Node>& value) -> ov::Output<ov::Node> {
            return value.get_element_type() == ov::element::i32 ? value : rg.make<ov::op::v0::Convert>(value, ov::element::i32);
        };
        const auto dim_of = [&rg](const ov::Output<ov::Node>& data, int64_t axis) -> ov::Output<ov::Node> {
            const auto shape = rg.make<ov::op::v3::ShapeOf>(data, ov::element::i32);
            return rg.make<ov::op::v8::Gather>(shape,
                                               ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {axis}),
                                               ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0}));
        };

        const auto& query = attention->input_value(SparseAttentionIdx::QUERY);
        const auto query_batch = query.get_partial_shape()[0];
        ov::Output<ov::Node> key_length;
        if (input_count > SparseAttentionIdx::KEY_LENGTH) {
            key_length = to_i32(attention->input_value(SparseAttentionIdx::KEY_LENGTH));
        } else {
            // Without key_length every row holds the whole key sequence.
            key_length = dim_of(attention->input_value(SparseAttentionIdx::KEY), 2);
        }
        // key_length may be [1]; SparseSDPA takes one value per batch row.
        const auto& key_length_shape = key_length.get_partial_shape();
        const bool per_row = key_length_shape.rank().is_static() && key_length_shape.size() == 1 && key_length_shape[0].is_static() &&
                             query_batch.is_static() && key_length_shape[0] == query_batch;
        if (!per_row) {
            key_length = rg.make<ov::op::v3::Broadcast>(key_length, dim_of(query, 0));
        }

        ov::OutputVector inputs{query,
                                attention->input_value(SparseAttentionIdx::KEY),
                                attention->input_value(SparseAttentionIdx::VALUE),
                                to_i32(attention->input_value(SparseAttentionIdx::BLOCK_INDICES)),
                                to_i32(attention->input_value(SparseAttentionIdx::BLOCK_COUNT)),
                                key_length};
        for (size_t idx = SparseAttentionIdx::SCALE; idx < input_count; ++idx) {
            inputs.push_back(attention->input_value(idx));
        }

        const auto sparse_sdpa = rg.make<SparseSDPA>(inputs, config);
        sparse_sdpa->set_friendly_name(attention->get_friendly_name());
        ov::copy_runtime_info(attention, rg.get());
        ov::replace_node(attention, sparse_sdpa);
        return true;
    };

    register_matcher(std::make_shared<Matcher>(p_attention, "ConvertSparseAttentionToSparseSDPA"), callback);
}

}  // namespace ov::intel_gpu
