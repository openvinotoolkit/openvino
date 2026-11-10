// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_sparse_attention_fusion.hpp"

#include <cmath>
#include <string>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/pass/node_registry.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/paged_qsa_indexer.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::pass {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v17 = ov::op::v17;
namespace op_internal = ov::op::internal;
using PagedQSAIndexerInputIdx = op_internal::PagedQSAIndexer::InputIdx;
using PagedQSAIndexerOutputIdx = op_internal::PagedQSAIndexer::OutputIdx;

PagedSparseAttentionFusion::PagedSparseAttentionFusion(paged_attention::PaParams& pa_params,
                                                       std::unordered_set<std::string>& var_ids_to_remove) {
    MATCHER_SCOPE(PagedSparseAttentionFusion);
    using namespace ov::pass::pattern;

    constexpr size_t batch_axis = 0;
    constexpr size_t heads_axis = 1;
    constexpr size_t sequence_axis = 2;
    constexpr size_t head_dim_axis = 3;
    constexpr size_t selected_blocks_axis = 3;
    constexpr size_t scale_input_index = 6;
    constexpr size_t sinks_input_index = 7;

    const auto activation_shape = rank_equals(4) && has_static_dims({heads_axis, head_dim_axis});
    const auto p_query = any_input(activation_shape && type_matches_any({element::f32, element::f16, element::bf16}));
    const auto p_key = any_input(activation_shape);
    const auto p_value = any_input(activation_shape);
    const auto p_key_read = wrap_type<ov::op::util::ReadValueBase>();
    const auto p_value_read = wrap_type<ov::op::util::ReadValueBase>();
    const auto read_state = [](const std::shared_ptr<Node>& p_read) {
        return optional<ov::op::v8::Gather>(
            {p_read, any_input(rank_equals(1)), wrap_type<v0::Constant>(value_matches("0"))},
            {{"batch_dims", 0}});
    };
    const auto p_full_key =
        wrap_type<v0::Concat>({read_state(p_key_read), p_key}, {{"axis", static_cast<int64_t>(sequence_axis)}});
    const auto p_full_value =
        wrap_type<v0::Concat>({read_state(p_value_read), p_value}, {{"axis", static_cast<int64_t>(sequence_axis)}});

    const auto p_indexer = wrap_type_strict_index<op_internal::PagedQSAIndexer>();
    const auto p_indices = wrap_type<v1::Transpose>(
        {wrap_type<v1::Reshape>({p_indexer->output(PagedQSAIndexerOutputIdx::SELECTED_BLOCK_INDICES), any_input()}),
         wrap_type<v0::Constant>(value_matches("0, 2, 1, 3"))},
        has_static_dims({heads_axis, selected_blocks_axis}));
    const auto p_count = wrap_type<v1::Transpose>(
        {wrap_type<v1::Reshape>({p_indexer->output(PagedQSAIndexerOutputIdx::SELECTED_BLOCK_COUNT), any_input()}),
         wrap_type<v0::Constant>(value_matches("0, 2, 1"))},
        has_static_dims({heads_axis}));
    const auto p_scale = any_input(rank_equals(0) || shape_matches("[1]"));
    const auto p_sinks = any_input(rank_equals(1));
    const OutputVector inputs{p_query, p_full_key, p_full_value, p_indices, p_count};
    const auto sparse_with = [&](const OutputVector& optional_inputs) {
        auto args = inputs;
        args.insert(args.end(), optional_inputs.begin(), optional_inputs.end());
        return wrap_type<v17::SparseAttention>(args, {{"causal", true}, {"query_block_size", 1}});
    };
    const auto p_attention =
        std::make_shared<pattern::op::Or>(OutputVector{sparse_with({}),
                                                       sparse_with({any_input()}),
                                                       sparse_with({any_input(), p_scale}),
                                                       sparse_with({any_input(), p_scale, p_sinks})});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS, &pa_params, &var_ids_to_remove](Matcher& m) {
        const auto attention = ov::as_type_ptr<v17::SparseAttention>(m.get_match_root());
        OPENVINO_ASSERT(attention, "PagedSparseAttentionFusion: expected SparseAttention");
        if (transformation_callback(attention)) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        const auto indexer =
            ov::as_type_ptr<op_internal::PagedQSAIndexer>(pattern_map.at(p_indexer).get_node_shared_ptr());
        const auto& query = pattern_map.at(p_query);
        const auto& key = pattern_map.at(p_key);
        const auto& value = pattern_map.at(p_value);
        const auto& query_shape = query.get_partial_shape();
        const auto& key_shape = key.get_partial_shape();
        const auto& value_shape = value.get_partial_shape();
        const auto& indices_shape = pattern_map.at(p_indices).get_partial_shape();
        const auto& count_shape = pattern_map.at(p_count).get_partial_shape();
        const auto& config = indexer->get_config();
        if (attention->get_key_block_size() != static_cast<int64_t>(config.key_block_size) ||
            indices_shape[heads_axis].get_length() != static_cast<int64_t>(config.num_selection_heads) ||
            indices_shape[selected_blocks_axis].get_length() != static_cast<int64_t>(config.block_topk) ||
            count_shape[heads_axis].get_length() != static_cast<int64_t>(config.num_selection_heads) ||
            !query_shape[batch_axis].compatible(key_shape[batch_axis]) ||
            !query_shape[batch_axis].compatible(value_shape[batch_axis]) ||
            !query_shape[sequence_axis].compatible(key_shape[sequence_axis]) ||
            !query_shape[sequence_axis].compatible(value_shape[sequence_axis])) {
            return false;
        }
        while (pa_params.exists("key_cache." + std::to_string(m_layer_index)) ||
               pa_params.exists("value_cache." + std::to_string(m_layer_index))) {
            ++m_layer_index;
        }
        const auto suffix = "." + std::to_string(m_layer_index++);
        const auto key_cache = pa_params.add("key_cache" + suffix, key.get_element_type(), PartialShape::dynamic(4));
        const auto value_cache =
            pa_params.add("value_cache" + suffix, value.get_element_type(), PartialShape::dynamic(4));

        NodeRegistry rg;
        const auto i64 = [&rg](const std::vector<int64_t>& values) {
            return rg.make<v0::Constant>(element::i64, Shape{values.size()}, values);
        };
        const auto order = i64({0, 2, 1, 3});
        const auto flatten = [&](const Output<Node>& data) {
            const auto& shape = data.get_partial_shape();
            const auto token_major = rg.make<v1::Transpose>(data, order);
            return rg.make<v1::Reshape>(token_major,
                                        i64({-1, shape[heads_axis].get_length() * shape[head_dim_axis].get_length()}),
                                        false);
        };
        const auto default_scale = 1.0f / std::sqrt(static_cast<float>(query_shape[head_dim_axis].get_length()));
        Output<Node> scale =
            attention->get_input_size() > scale_input_index
                ? pattern_map.at(p_scale)
                : rg.make<v0::Constant>(query.get_element_type(), Shape{}, std::vector<float>{default_scale});
        if (scale.get_partial_shape().rank() == 1) {
            scale = rg.make<v0::Squeeze>(scale, i64({0}));
        }
        const Output<Node> sinks =
            attention->get_input_size() > sinks_input_index
                ? pattern_map.at(p_sinks)
                : rg.make<v0::Constant>(query.get_element_type(), Shape{0}, std::vector<float>{});
        const auto paged = rg.make<op_internal::PagedSparseAttention>(
            OutputVector{flatten(query),
                         flatten(key),
                         flatten(value),
                         key_cache,
                         value_cache,
                         indexer->input_value(PagedQSAIndexerInputIdx::PAST_LENS),
                         indexer->input_value(PagedQSAIndexerInputIdx::SUBSEQUENCE_BEGINS),
                         indexer->input_value(PagedQSAIndexerInputIdx::BLOCK_INDICES),
                         indexer->input_value(PagedQSAIndexerInputIdx::BLOCK_INDICES_BEGINS),
                         indexer->output(PagedQSAIndexerOutputIdx::SELECTED_BLOCK_INDICES),
                         indexer->output(PagedQSAIndexerOutputIdx::SELECTED_BLOCK_COUNT),
                         scale,
                         sinks},
            attention->get_key_block_size());
        const auto batch_and_length =
            rg.add(ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(query,
                                                                                      {batch_axis, sequence_axis},
                                                                                      {attention}));
        const auto shape = rg.make<v0::Concat>(
            OutputVector{batch_and_length,
                         i64({query_shape[heads_axis].get_length(), value_shape[head_dim_axis].get_length()})},
            0);
        const auto restored = rg.make<v1::Reshape>(paged, shape, false);
        const auto output = rg.make<v1::Transpose>(restored, order);
        output->set_friendly_name(attention->get_friendly_name());
        rg.add(key_cache);
        rg.add(value_cache);
        ov::copy_runtime_info(attention, rg.get());
        enable_keep_const_precision(key_cache);
        enable_keep_const_precision(value_cache);
        paged->get_rt_info()["num_k_heads"] = key_shape[heads_axis].get_length();
        paged->get_rt_info()["k_head_size"] = key_shape[head_dim_axis].get_length();
        paged->get_rt_info()["num_v_heads"] = value_shape[heads_axis].get_length();
        paged->get_rt_info()["v_head_size"] = value_shape[head_dim_axis].get_length();
        for (const auto& p_read : {p_key_read, p_value_read}) {
            const auto read =
                ov::as_type_ptr<ov::op::util::ReadValueBase>(pattern_map.at(p_read).get_node_shared_ptr());
            var_ids_to_remove.insert(read->get_variable_id());
        }
        ov::replace_node(attention, output);
        return true;
    };

    register_matcher(std::make_shared<Matcher>(p_attention, matcher_name), callback);
}

}  // namespace ov::pass
