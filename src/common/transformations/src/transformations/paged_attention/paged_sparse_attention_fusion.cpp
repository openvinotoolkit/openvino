// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_sparse_attention_fusion.hpp"

#include <memory>
#include <string>
#include <vector>

#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

namespace {
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v8 = ov::op::v8;
namespace v15 = ov::op::v15;
namespace pattern = ov::pass::pattern;

ov::Output<ov::Node> pack_heads(const ov::Output<ov::Node>& input) {
    // Query:         [B_token, H, 1, S]   -> [B_token, 1, H, S]   -> [B_token, H * S].
    // Current key:   [B_token, Hk, 1, S]  -> [B_token, 1, Hk, S]  -> [B_token, Hk * S].
    // Current value: [B_token, Hk, 1, Sv] -> [B_token, 1, Hk, Sv] -> [B_token, Hk * Sv].
    const auto order = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
    const auto transposed = std::make_shared<v1::Transpose>(input, order);
    const auto shape = v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, -1});
    const auto packed = std::make_shared<v1::Reshape>(transposed, shape, true);
    return packed;
}

ov::Output<ov::Node> pack_selection(const ov::Output<ov::Node>& input) {
    // Indices: [B_token, Hs, 1, KB] -> [B_token, Hs, KB].
    // Counts:  [B_token, Hs, 1]     -> [B_token, Hs].
    // Hs is the number of selection heads; KB is the maximum selected sparse blocks.
    const auto axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    const auto packed = std::make_shared<v15::Squeeze>(input, axis);
    if (packed->get_output_element_type(0) == ov::element::i32) {
        return packed;
    }
    return std::make_shared<v0::Convert>(packed, ov::element::i32);
}

struct CachePattern {
    std::shared_ptr<ov::Node> read_value;
    std::shared_ptr<ov::Node> current;
    std::shared_ptr<ov::Node> concat;
};

CachePattern cache_pattern() {
    auto read_value = pattern::wrap_type<ov::op::util::ReadValueBase>(pattern::rank_equals(4));
    auto past = pattern::optional<v8::Gather>({read_value, pattern::any_input(), pattern::any_input()});
    auto current = pattern::any_input(pattern::has_static_dims({1, 3}));
    auto concat = pattern::wrap_type<v0::Concat>({past, current});
    return {read_value, current, concat};
}
}  // namespace

ov::pass::PagedSparseAttentionFusion::PagedSparseAttentionFusion(ov::pass::paged_attention::PaParams& pa_params,
                                                                 const ov::pass::paged_attention::Options& options,
                                                                 std::unordered_set<std::string>& var_ids_to_remove) {
    MATCHER_SCOPE(PagedSparseAttentionFusion);

    const auto p_query = pattern::any_input(pattern::rank_equals(4) && pattern::has_static_dims({1, 3}));
    const auto p_key = cache_pattern();
    const auto p_value = cache_pattern();
    const auto index_type_pred = pattern::type_matches_any({ov::element::i32, ov::element::i64});
    const auto p_block_indices = pattern::any_input(pattern::rank_equals(4) && index_type_pred);
    const auto p_block_count = pattern::any_input(pattern::rank_equals(3) && index_type_pred);
    const auto p_key_length = pattern::any_input(pattern::rank_equals(1) && index_type_pred);
    const auto p_scale = pattern::any_input(pattern::has_static_rank());
    const auto p_sinks = pattern::any_input(pattern::rank_equals(1));

    const auto attrs = pattern::attrs_match({{"causal", true}});
    auto sparse_attention =
        pattern::wrap_type<ov::op::v17::SparseAttention>(
            {p_query, p_key.concat, p_value.concat, p_block_indices, p_block_count},
            attrs) |
        pattern::wrap_type<ov::op::v17::SparseAttention>(
            {p_query, p_key.concat, p_value.concat, p_block_indices, p_block_count, p_key_length},
            attrs) |
        pattern::wrap_type<ov::op::v17::SparseAttention>(
            {p_query, p_key.concat, p_value.concat, p_block_indices, p_block_count, p_key_length, p_scale},
            attrs) |
        pattern::wrap_type<ov::op::v17::SparseAttention>(
            {p_query, p_key.concat, p_value.concat, p_block_indices, p_block_count, p_key_length, p_scale, p_sinks},
            attrs);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS, &pa_params, &var_ids_to_remove](
                                             pattern::Matcher& m) {
        std::cout << matcher_name << " STARTED" << std::endl;
        const auto& pm = m.get_pattern_value_map();

        const auto current_key = pm.at(p_key.current);
        const auto current_value = pm.at(p_value.current);
        const auto key_read =
            ov::as_type_ptr<ov::op::util::ReadValueBase>(pm.at(p_key.read_value).get_node_shared_ptr());
        const auto value_read =
            ov::as_type_ptr<ov::op::util::ReadValueBase>(pm.at(p_value.read_value).get_node_shared_ptr());
        const auto sparse_node = ov::as_type_ptr<ov::op::v17::SparseAttention>(m.get_match_root());
        OPENVINO_ASSERT(key_read && value_read && sparse_node);

        var_ids_to_remove.insert(key_read->get_variable_id());
        var_ids_to_remove.insert(value_read->get_variable_id());

        // Share the regular paged-attention input namespace without overwriting caches
        // already allocated by another attention conversion.
        while (pa_params.exists("key_cache." + std::to_string(m_layer_index)) ||
               pa_params.exists("value_cache." + std::to_string(m_layer_index)) ||
               (options.use_per_layer_block_indices_inputs &&
                pa_params.exists("block_indices." + std::to_string(m_layer_index)))) {
            ++m_layer_index;
        }
        const auto layer = std::to_string(m_layer_index);
        const auto& key_shape = current_key.get_partial_shape();
        const auto& value_shape = current_value.get_partial_shape();
        const auto key_cache = pa_params.add("key_cache." + layer, ov::element::dynamic, PartialShape::dynamic(4));
        const auto value_cache = pa_params.add("value_cache." + layer, ov::element::dynamic, PartialShape::dynamic(4));
        enable_keep_const_precision(key_cache);
        enable_keep_const_precision(value_cache);

        const auto past_lens = pa_params.add("past_lens", ov::element::i32, ov::PartialShape{-1});
        const auto subsequence_begins = pa_params.add("subsequence_begins", ov::element::i32, ov::PartialShape{-1});
        const auto cache_block_indices =
            pa_params.add(options.use_per_layer_block_indices_inputs ? "block_indices." + layer : "block_indices",
                          ov::element::i32,
                          ov::PartialShape{-1});
        const auto block_indices_begins = pa_params.add("block_indices_begins", ov::element::i32, ov::PartialShape{-1});

        const auto packed_query = pack_heads(pm.at(p_query));
        const auto packed_key = pack_heads(current_key);
        const auto packed_value = pack_heads(current_value);

        const auto packed_indices = pack_selection(pm.at(p_block_indices));
        const auto packed_count = pack_selection(pm.at(p_block_count));

        ov::Output<ov::Node> scale;
        if (pm.count(p_scale)) {
            scale = pm.at(p_scale);
            if (scale.get_partial_shape().rank() == 1) {
                const auto squeezed_scale = std::make_shared<v15::Squeeze>(scale);
                scale = squeezed_scale;
            }
        } else {
            scale = v0::Constant::create(packed_query.get_element_type(),
                                         ov::Shape{},
                                         {1 / std::sqrt(pm.at(p_query).get_partial_shape()[3].get_length())});
        }

        auto attention_sinks = pm.count(p_sinks)
                                   ? pm.at(p_sinks)
                                   : v0::Constant::create(packed_query.get_element_type(), ov::Shape{0}, {});

        const auto paged_sparse =
            std::make_shared<ov::op::internal::PagedSparseAttention>(ov::OutputVector{packed_query,
                                                                                      packed_key,
                                                                                      packed_value,
                                                                                      key_cache,
                                                                                      value_cache,
                                                                                      past_lens,
                                                                                      subsequence_begins,
                                                                                      cache_block_indices,
                                                                                      block_indices_begins,
                                                                                      packed_indices,
                                                                                      packed_count,
                                                                                      scale,
                                                                                      attention_sinks},
                                                                     sparse_node->get_key_block_size());
        paged_sparse->set_friendly_name(sparse_node->get_friendly_name() + "/PagedSparseAttention");
        paged_sparse->get_rt_info()["num_k_heads"] = key_shape[1].get_length();
        paged_sparse->get_rt_info()["k_head_size"] = key_shape[3].get_length();
        paged_sparse->get_rt_info()["num_v_heads"] = value_shape[1].get_length();
        paged_sparse->get_rt_info()["v_head_size"] = value_shape[3].get_length();

        // [B_token, H * Sv] -> [B_token, 1, H, Sv] -> [B_token, H, 1, Sv].
        auto output_shape =
            v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{4}, {0, 1, -1, value_shape[3].get_length()});
        const auto reshaped = std::make_shared<v1::Reshape>(paged_sparse, output_shape, true);
        const auto output =
            std::make_shared<v1::Transpose>(reshaped,
                                            v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3}));

        output->set_friendly_name(sparse_node->get_friendly_name());
        ov::copy_runtime_info(sparse_node, paged_sparse);
        ov::replace_node(sparse_node, output);
        ++m_layer_index;
        std::cout << matcher_name << " FINISHED" << std::endl;
        return true;
    };

    register_matcher(std::make_shared<pattern::Matcher>(sparse_attention, matcher_name), callback);
}
