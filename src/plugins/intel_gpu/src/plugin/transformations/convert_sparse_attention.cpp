// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_sparse_attention.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/node_registry.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

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

}  // namespace

std::optional<op::SparseSDPA::Config> ConvertSparseAttentionToInternal::make_config(const ov::op::v17::SparseAttention& op, std::string& reason) {
    using Idx = SparseAttentionIdx;
    const auto static_dim = [&op](size_t input, size_t axis) -> std::optional<size_t> {
        const auto& shape = op.get_input_partial_shape(input);
        if (shape.rank().is_dynamic() || shape.size() != 4 || shape[axis].is_dynamic()) {
            return std::nullopt;
        }
        return static_cast<size_t>(shape[axis].get_length());
    };
    const auto num_heads = static_dim(Idx::QUERY, 1);
    const auto k_head_size = static_dim(Idx::QUERY, 3);
    const auto num_kv_heads = static_dim(Idx::KEY, 1);
    const auto v_head_size = static_dim(Idx::VALUE, 3);
    const auto num_selection_heads = static_dim(Idx::BLOCK_INDICES, 1);
    if (!num_heads || !k_head_size || !num_kv_heads || !v_head_size || !num_selection_heads) {
        reason = "the query / key / value head counts and head sizes and the selection head count must be static";
        return std::nullopt;
    }

    op::SparseSDPA::Config config;
    config.num_heads = *num_heads;
    config.k_head_size = *k_head_size;
    config.num_kv_heads = *num_kv_heads;
    config.v_head_size = *v_head_size;
    config.num_selection_heads = *num_selection_heads;
    config.key_block_size = static_cast<size_t>(op.get_key_block_size());
    config.query_block_size = static_cast<size_t>(op.get_query_block_size());
    config.causal = op.get_causal();
    if (!op::SparseSDPA::validate_config(config, reason)) {
        return std::nullopt;
    }
    return config;
}

ConvertSparseAttentionToInternal::ConvertSparseAttentionToInternal() {
    using namespace ov::pass::pattern;
    using Idx = SparseAttentionIdx;

    auto sparse_attention = wrap_type<ov::op::v17::SparseAttention>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto attention = ov::as_type_ptr<ov::op::v17::SparseAttention>(m.get_match_root());
        if (!attention || transformation_callback(attention)) {
            return false;
        }
        std::string reason;
        const auto config = make_config(*attention, reason);
        if (!config) {
            return false;
        }

        ov::pass::NodeRegistry rg;
        const auto to_i32 = [&rg](const ov::Output<ov::Node>& value) -> ov::Output<ov::Node> {
            return value.get_element_type() == ov::element::i32 ? value : rg.make<ov::op::v0::Convert>(value, ov::element::i32);
        };
        // [1] i32 tensor holding dimension `axis` of `data`, a Constant when it is static.
        const auto dim_of = [&rg](const ov::Output<ov::Node>& data, int64_t axis) -> ov::Output<ov::Node> {
            const auto shape = rg.add(ov::op::util::make_try_fold<ov::op::v3::ShapeOf>(data, ov::element::i32));
            const auto indices = rg.add(ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {axis}));
            const auto gather_axis = rg.add(ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0}));
            return rg.add(ov::op::util::make_try_fold<ov::op::v8::Gather>(shape, indices, gather_axis));
        };

        const auto& query = attention->input_value(Idx::QUERY);
        const auto& batch = query.get_partial_shape()[0];
        ov::Output<ov::Node> key_length =
            attention->get_input_size() > Idx::KEY_LENGTH ? to_i32(attention->input_value(Idx::KEY_LENGTH)) : dim_of(attention->input_value(Idx::KEY), 2);
        const auto& key_length_shape = key_length.get_partial_shape();
        const bool per_row = key_length_shape.rank().is_static() && key_length_shape.size() == 1 && key_length_shape[0].is_static() && batch.is_static() &&
                             key_length_shape[0] == batch;
        if (!per_row) {
            key_length = rg.add(ov::op::util::make_try_fold<ov::op::v3::Broadcast>(key_length, dim_of(query, 0)));
        }

        ov::OutputVector inputs{query,
                                attention->input_value(Idx::KEY),
                                attention->input_value(Idx::VALUE),
                                to_i32(attention->input_value(Idx::BLOCK_INDICES)),
                                to_i32(attention->input_value(Idx::BLOCK_COUNT)),
                                key_length};
        for (const size_t idx : {Idx::SCALE, Idx::SINKS}) {
            if (attention->get_input_size() > idx) {
                inputs.push_back(attention->input_value(idx));
            }
        }
        const auto sparse_sdpa = std::make_shared<op::SparseSDPA>(inputs, *config);
        sparse_sdpa->set_friendly_name(attention->get_friendly_name());
        ov::copy_runtime_info(attention, rg.get());
        ov::copy_runtime_info(attention, sparse_sdpa);
        ov::replace_node(attention, sparse_sdpa);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sparse_attention, "ConvertSparseAttentionToInternal");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
