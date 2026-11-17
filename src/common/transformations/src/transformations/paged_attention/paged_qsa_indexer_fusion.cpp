// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_qsa_indexer_fusion.hpp"

#include <limits>
#include <utility>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/pass/node_registry.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::pass {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace op_internal = ov::op::internal;
using QSAIndexerInputIdx = op_internal::QSAIndexer::InputIdx;
using QSAIndexerOutputIdx = op_internal::QSAIndexer::OutputIdx;
using PagedQSAIndexerInputIdx = op_internal::PagedQSAIndexer::InputIdx;

namespace {

constexpr size_t heads_axis = 2;

op_internal::PagedQSAIndexer::Config paged_config_from(const op_internal::QSAIndexer::Config& config) {
    op_internal::PagedQSAIndexer::Config result;
    result.key_block_size = config.key_block_size;
    result.block_topk = config.block_topk;
    result.num_selection_heads = config.num_selection_heads;
    result.indexer_rotary_dim = config.indexer_rotary_dim;
    result.indexer_head_dim = config.indexer_head_dim;
    result.epsilon = config.epsilon;
    result.scale = op_internal::QSAIndexer::effective_scale(config);
    result.query_norm = config.query_norm;
    result.query_rope = config.query_rope;
    return result;
}

}  // namespace

PagedQSAIndexerFusion::PagedQSAIndexerFusion(paged_attention::PaParams& pa_params,
                                             std::unordered_set<std::string>& var_ids_to_remove) {
    MATCHER_SCOPE(PagedQSAIndexerFusion);
    using namespace ov::pass::pattern;

    // QSAIndexer activations are [B, L, H, D], cos / sin are [B, L, rotary_dim].
    const auto p_query = any_input();
    const auto p_raw_keys = any_input();
    const auto p_rope_cos = any_input();
    const auto p_rope_sin = any_input();

    // The dense states are read from variables, directly or through the beam Gather, in the FuseQSAIndexer layout.
    const auto read_state = [](const std::shared_ptr<ov::Node>& p_read_value) {
        return optional<ov::op::v8::Gather>({p_read_value, any_input(), any_input()});
    };
    const auto p_block_keys_read = wrap_type<ov::op::util::ReadValueBase>();
    const auto p_raw_keys_read = wrap_type<ov::op::util::ReadValueBase>();
    const auto p_rope_read = wrap_type<ov::op::util::ReadValueBase>();
    // With FuseQSAIndexer padding, the block-key cache is grown by a Concat.
    const auto p_block_keys_cache =
        optional<v0::Concat>({wrap_type<v0::Unsqueeze>({read_state(p_block_keys_read), any_input()}), any_input()});
    const auto p_incomplete_raw_keys = wrap_type<v0::Unsqueeze>({read_state(p_raw_keys_read), any_input()});

    const auto flattenable_inputs = [](const ov::Output<ov::Node>& output) {
        const auto indexer = ov::as_type_ptr<op_internal::QSAIndexer>(output.get_node_shared_ptr());
        OPENVINO_ASSERT(indexer, "PagedQSAIndexerFusion: matched node is expected to be QSAIndexer");
        const auto& config = indexer->get_config();
        // The [-1, width] adapters require nonzero widths that fit in int64_t.
        if (config.indexer_rotary_dim == 0) {
            return false;
        }
        const auto max_heads = static_cast<size_t>(std::numeric_limits<int64_t>::max()) / config.indexer_head_dim;
        for (const auto input : {QSAIndexerInputIdx::INDEX_QUERY, QSAIndexerInputIdx::RAW_INDEX_KEYS}) {
            const auto& shape = indexer->get_input_partial_shape(input);
            if (shape.rank().is_dynamic() || shape[heads_axis].is_dynamic()) {
                return false;
            }
            const auto heads = shape[heads_axis].get_length();
            if (heads == 0 || static_cast<size_t>(heads) > max_heads) {
                return false;
            }
        }
        return true;
    };
    const auto p_indexer = wrap_type<op_internal::QSAIndexer>({p_query,
                                                               p_raw_keys,
                                                               p_rope_cos,
                                                               p_rope_sin,
                                                               any_input(),
                                                               any_input(),
                                                               any_input(),
                                                               p_block_keys_cache,
                                                               p_incomplete_raw_keys,
                                                               read_state(p_rope_read)},
                                                              flattenable_inputs);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS, &pa_params, &var_ids_to_remove](Matcher& m) {
        const auto indexer = ov::as_type_ptr<op_internal::QSAIndexer>(m.get_match_root());
        OPENVINO_ASSERT(indexer, "PagedQSAIndexerFusion: matched node is expected to be QSAIndexer");
        if (transformation_callback(indexer)) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& query = pattern_map.at(p_query);
        const auto& raw_keys = pattern_map.at(p_raw_keys);
        const auto& config = indexer->get_config();
        const auto data_type = query.get_element_type();
        const auto& query_shape = query.get_partial_shape();
        const auto& keys_shape = raw_keys.get_partial_shape();
        const auto query_heads = query_shape[heads_axis].get_length();
        const auto key_heads = keys_shape[heads_axis].get_length();
        const auto paged_config = paged_config_from(config);
        const auto head_dim = static_cast<int64_t>(config.indexer_head_dim);
        const auto rotary_dim = static_cast<int64_t>(config.indexer_rotary_dim);
        const auto query_width = query_heads * head_dim;
        const auto key_width = key_heads * head_dim;

        pa_params.add("past_lens", ov::element::i32, ov::PartialShape{-1});
        pa_params.add("subsequence_begins", ov::element::i32, ov::PartialShape{-1});
        pa_params.add("block_indices", ov::element::i32, ov::PartialShape{-1});
        pa_params.add("block_indices_begins", ov::element::i32, ov::PartialShape{-1});

        // The caches use the main KV page table, so the page dimensions are dynamic.
        const auto layer = std::to_string(m_layer_index++);
        const auto add_cache = [&](const std::string& name, const ov::PartialShape& shape) {
            const auto cache = pa_params.add(name + "." + layer, data_type, shape);
            enable_keep_const_precision(cache);
            return cache;
        };
        const auto index_key_cache = add_cache("index_key_cache", {-1, key_heads, -1, head_dim});
        const auto index_block_rope_cache = add_cache("index_block_rope_cache", {-1, -1, 2 * rotary_dim});
        const auto block_index_keys_cache = add_cache("block_index_keys_cache", {-1, key_heads, -1, head_dim});
        for (const auto& p_read_value : {p_block_keys_read, p_raw_keys_read, p_rope_read}) {
            const auto read_value =
                ov::as_type_ptr<ov::op::util::ReadValueBase>(pattern_map.at(p_read_value).get_node_shared_ptr());
            var_ids_to_remove.insert(read_value->get_variable_id());
        }

        ov::pass::NodeRegistry rg;
        const auto flatten = [&](const ov::Output<ov::Node>& data, int64_t width) {
            return rg.make<v1::Reshape>(data,
                                        v0::Constant::create(ov::element::i64, ov::Shape{2}, {int64_t{-1}, width}),
                                        false);
        };
        ov::OutputVector inputs(PagedQSAIndexerInputIdx::COUNT);
        inputs[PagedQSAIndexerInputIdx::INDEX_QUERY] = flatten(query, query_width);
        inputs[PagedQSAIndexerInputIdx::RAW_INDEX_KEYS] = flatten(raw_keys, key_width);
        inputs[PagedQSAIndexerInputIdx::ROPE_COS] = flatten(pattern_map.at(p_rope_cos), rotary_dim);
        inputs[PagedQSAIndexerInputIdx::ROPE_SIN] = flatten(pattern_map.at(p_rope_sin), rotary_dim);
        inputs[PagedQSAIndexerInputIdx::Q_NORM_WEIGHT] = indexer->input_value(QSAIndexerInputIdx::Q_NORM_WEIGHT);
        inputs[PagedQSAIndexerInputIdx::K_NORM_WEIGHT] = indexer->input_value(QSAIndexerInputIdx::K_NORM_WEIGHT);
        inputs[PagedQSAIndexerInputIdx::PAST_LENS] = pa_params["past_lens"];
        inputs[PagedQSAIndexerInputIdx::SUBSEQUENCE_BEGINS] = pa_params["subsequence_begins"];
        inputs[PagedQSAIndexerInputIdx::BLOCK_INDICES] = pa_params["block_indices"];
        inputs[PagedQSAIndexerInputIdx::BLOCK_INDICES_BEGINS] = pa_params["block_indices_begins"];
        inputs[PagedQSAIndexerInputIdx::INDEX_KEY_CACHE] = index_key_cache;
        inputs[PagedQSAIndexerInputIdx::INDEX_BLOCK_ROPE_CACHE] = index_block_rope_cache;
        inputs[PagedQSAIndexerInputIdx::BLOCK_INDEX_KEYS_CACHE] = block_index_keys_cache;

        const auto paged = rg.make<op_internal::PagedQSAIndexer>(inputs, paged_config);
        paged->set_friendly_name(indexer->get_friendly_name());

        // The token-major selections [B * L, S, ...] are restored to the QSAIndexer layout [B, S, L, ...].
        const auto batch_and_length = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(query, {0, 1});
        const auto selection_heads = static_cast<int64_t>(config.num_selection_heads);
        const auto restore = [&](size_t output, const std::vector<int64_t>& tail, const std::vector<int64_t>& order) {
            const auto shape = rg.make<v0::Concat>(
                ov::OutputVector{batch_and_length,
                                 v0::Constant::create(ov::element::i64, ov::Shape{tail.size()}, tail)},
                0);
            const auto token_major = rg.make<v1::Reshape>(paged->output(output), shape, false);
            return rg.make<v1::Transpose>(token_major,
                                          v0::Constant::create(ov::element::i64, ov::Shape{order.size()}, order));
        };
        const auto indices = restore(0, {selection_heads, static_cast<int64_t>(config.block_topk)}, {0, 2, 1, 3});
        const auto count = restore(1, {selection_heads}, {0, 2, 1});
        rg.add(batch_and_length);
        ov::copy_runtime_info(indexer, rg.get());
        register_new_node(paged);

        indexer->output(QSAIndexerOutputIdx::SELECTED_BLOCK_INDICES).replace(indices);
        indexer->output(QSAIndexerOutputIdx::SELECTED_BLOCK_COUNT).replace(count);
        // The dense states pass through unchanged; the caller must retire their variable sinks.
        const std::vector<std::pair<size_t, size_t>> states{
            {QSAIndexerOutputIdx::BLOCK_INDEX_KEYS_CACHE, QSAIndexerInputIdx::BLOCK_INDEX_KEYS_CACHE},
            {QSAIndexerOutputIdx::PAST_INCOMPLETE_RAW_K, QSAIndexerInputIdx::PAST_INCOMPLETE_RAW_K},
            {QSAIndexerOutputIdx::PAST_INCOMPLETE_ROPE, QSAIndexerInputIdx::PAST_INCOMPLETE_ROPE}};
        for (const auto& [output, input] : states) {
            for (const auto& consumer : indexer->output(output).get_target_inputs()) {
                consumer.replace_source_output(indexer->input_value(input));
            }
        }
        return true;
    };

    register_matcher(std::make_shared<Matcher>(p_indexer, matcher_name), callback);
}

}  // namespace ov::pass
