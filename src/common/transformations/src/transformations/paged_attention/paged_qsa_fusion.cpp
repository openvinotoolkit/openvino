// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_qsa_fusion.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/pattern_blocks/qsa_index_query_block.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

using ov::pass::pattern::any_input;
using ov::pass::pattern::optional;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v8 = ov::op::v8;
namespace v17 = ov::op::v17;

namespace {

// QSAPrepare input ports.
constexpr size_t RAW_INDEX_KEYS = 0;
constexpr size_t PAST_BLOCK_INDEX_KEYS = 1;
constexpr size_t ROPE_COS = 3;
constexpr size_t ROPE_SIN = 4;
constexpr size_t K_NORM_WEIGHT = 5;
constexpr size_t PAST_INCOMPLETE_RAW_KEYS = 6;
constexpr size_t PAST_INCOMPLETE_ROPE = 7;

// SparseAttention input ports.
constexpr size_t SCALE = 6;
constexpr size_t SINKS = 7;

// The ReadValue a state input comes from, through the beam search Gather and precision conversions.
std::shared_ptr<ov::op::util::ReadValueBase> get_read_value(ov::Output<ov::Node> output) {
    while (ov::is_type<v0::Convert>(output.get_node()) || ov::is_type<v8::Gather>(output.get_node())) {
        output = output.get_node()->input_value(0);
    }
    return ov::as_type_ptr<ov::op::util::ReadValueBase>(output.get_node_shared_ptr());
}

bool static_dim(const ov::PartialShape& shape, size_t rank, size_t axis, int64_t& value) {
    if (shape.rank().is_dynamic() || shape.size() != rank || shape[axis].is_dynamic()) {
        return false;
    }
    value = shape[axis].get_length();
    return true;
}

// The first free layer index: QSA layers are numbered after the PagedAttention layers.
size_t get_free_layer_index(const ov::pass::paged_attention::PaParams& pa_params) {
    size_t index = 0;
    while (pa_params.exists("key_cache." + std::to_string(index))) {
        ++index;
    }
    return index;
}

struct Builder {
    ov::NodeVector nodes;

    template <class T, class... Args>
    std::shared_ptr<T> make(Args&&... args) {
        auto node = std::make_shared<T>(std::forward<Args>(args)...);
        nodes.push_back(node);
        return node;
    }

    std::shared_ptr<v0::Constant> i64(const std::vector<int64_t>& values) {
        return make<v0::Constant>(ov::element::i64, ov::Shape{values.size()}, values);
    }

    // [N, A, B, D] -> [N, B, A, D]
    ov::Output<ov::Node> transpose_0213(const ov::Output<ov::Node>& input) {
        return make<v1::Transpose>(input, i64({0, 2, 1, 3}));
    }

    // [...] -> [-1, width]
    ov::Output<ov::Node> flatten(const ov::Output<ov::Node>& input, int64_t width) {
        return make<v1::Reshape>(input, i64({-1, width}), false);
    }
};

}  // namespace

ov::pass::PagedQSAFusion::PagedQSAFusion(ov::pass::paged_attention::PaParams& pa_params,
                                         std::unordered_set<std::string>& var_ids_to_remove) {
    MATCHER_SCOPE(PagedQSAFusion);

    auto index_query = std::make_shared<ov::pass::pattern::op::QSAIndexQueryBlock>();
    auto indexer = wrap_type<v17::SparseAttentionIndexer>({index_query, wrap_type<v17::QSAPrepare>(), any_input()});

    // past key / value: ReadValue -> [Gather(beam_idx)] -> Concat with the current ones along the length axis
    auto past_key = optional<v8::Gather>({wrap_type<ov::op::util::ReadValueBase>(), any_input(), any_input()});
    auto past_value = optional<v8::Gather>({wrap_type<ov::op::util::ReadValueBase>(), any_input(), any_input()});
    auto current_key = any_input();
    auto current_value = any_input();
    auto key = wrap_type<v0::Concat>({past_key, current_key});
    auto value = wrap_type<v0::Concat>({past_value, current_value});
    auto query = any_input();
    auto key_length = any_input();
    auto attention = wrap_type<v17::SparseAttention>({query, key, value, indexer, indexer, key_length}) |
                     wrap_type<v17::SparseAttention>({query, key, value, indexer, indexer, key_length, any_input()}) |
                     wrap_type<v17::SparseAttention>(
                         {query, key, value, indexer, indexer, key_length, any_input(), any_input()});

    matcher_pass_callback callback = [=, &pa_params, &var_ids_to_remove](pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& pm = m.get_pattern_value_map();
        const auto sparse_attention = ov::as_type_ptr<v17::SparseAttention>(m.get_match_root());
        const auto indexer_node = ov::as_type_ptr<v17::SparseAttentionIndexer>(pm.at(indexer).get_node_shared_ptr());
        if (!sparse_attention || !indexer_node || !sparse_attention->get_causal() ||
            sparse_attention->get_query_block_size() != 1 ||
            sparse_attention->input_value(3) != indexer_node->output(0) ||
            sparse_attention->input_value(4) != indexer_node->output(1) ||
            sparse_attention->input_value(5) != indexer_node->input_value(2)) {
            return false;
        }
        const auto match = index_query->get_match(pm, indexer_node);
        if (!match) {
            return false;
        }
        const auto& prepare = match->prepare;

        // ---- the five states ----
        const auto key_concat = pm.at(key).get_node_shared_ptr();
        const auto value_concat = pm.at(value).get_node_shared_ptr();
        const std::vector<std::shared_ptr<ov::op::util::ReadValueBase>> states{
            get_read_value(key_concat->input_value(0)),
            get_read_value(value_concat->input_value(0)),
            get_read_value(prepare->input_value(PAST_BLOCK_INDEX_KEYS)),
            get_read_value(prepare->input_value(PAST_INCOMPLETE_RAW_KEYS)),
            get_read_value(prepare->input_value(PAST_INCOMPLETE_ROPE))};
        for (const auto& state : states) {
            if (!state) {
                return false;
            }
        }
        const auto concat_axis = ov::as_type_ptr<v0::Concat>(key_concat)->get_axis();
        if ((concat_axis != 2 && concat_axis != -2) ||
            ov::as_type_ptr<v0::Concat>(value_concat)->get_axis() != concat_axis) {
            return false;
        }

        // ---- head geometry: [N, H, L, E] attention inputs ----
        int64_t num_heads = 0, head_size = 0, num_kv_heads = 0, key_head_size = 0, value_num_heads = 0,
                value_head_size = 0;
        const auto& current_key_value = pm.at(current_key);
        const auto& current_value_value = pm.at(current_value);
        if (!static_dim(pm.at(query).get_partial_shape(), 4, 1, num_heads) ||
            !static_dim(pm.at(query).get_partial_shape(), 4, 3, head_size) ||
            !static_dim(current_key_value.get_partial_shape(), 4, 1, num_kv_heads) ||
            !static_dim(current_key_value.get_partial_shape(), 4, 3, key_head_size) ||
            !static_dim(current_value_value.get_partial_shape(), 4, 1, value_num_heads) ||
            !static_dim(current_value_value.get_partial_shape(), 4, 3, value_head_size) || key_head_size != head_size ||
            value_num_heads != num_kv_heads) {
            return false;
        }
        const auto index_heads = static_cast<int64_t>(match->num_heads);
        const auto index_dim = static_cast<int64_t>(match->head_dim);
        const auto rotary_dim = static_cast<int64_t>(match->rotary_dim);

        // ---- page table and caches ----
        const auto layer = std::to_string(get_free_layer_index(pa_params));
        auto add_cache = [&](const std::string& name, const ov::PartialShape& shape) {
            auto cache = pa_params.add(name + "." + layer, ov::element::dynamic, shape);
            enable_keep_const_precision(cache);
            return cache;
        };
        const auto key_cache = add_cache("key_cache", {-1, num_kv_heads, -1, head_size});
        const auto value_cache = add_cache("value_cache", {-1, num_kv_heads, -1, value_head_size});
        const auto index_key_cache = add_cache("index_key_cache", {-1, 1, -1, index_dim});
        const auto index_block_rope_cache = add_cache("index_block_rope_cache", {-1, -1, 2 * rotary_dim});
        const auto block_index_keys_cache = add_cache("block_index_keys_cache", {-1, 1, -1, index_dim});
        const auto past_lens = pa_params["past_lens"];
        const auto subsequence_begins = pa_params["subsequence_begins"];
        const auto block_indices = pa_params["block_indices"];
        const auto block_indices_begins = pa_params["block_indices_begins"];

        Builder b;
        // ---- PagedQSAIndexer: the raw index query, the per-token index keys and RoPE tables ----
        const auto raw_query = match->raw_query_blhd ? match->raw_query : b.transpose_0213(match->raw_query);
        ov::Output<ov::Node> q_norm_weight =
            match->q_norm_weight.get_node()
                ? b.make<v1::Reshape>(match->q_norm_weight, b.i64({index_dim}), false)->output(0)
                : b.make<v0::Constant>(raw_query.get_element_type(),
                                       ov::Shape{static_cast<size_t>(index_dim)},
                                       std::vector<float>(index_dim, 1.0f))
                      ->output(0);
        ov::op::internal::PagedQSAIndexer::Config config;
        config.key_block_size = match->key_block_size;
        config.block_topk = match->block_topk;
        config.num_selection_heads = 1;
        config.indexer_rotary_dim = match->rotary_dim;
        config.indexer_head_dim = match->head_dim;
        config.epsilon = match->epsilon;
        const auto paged_indexer = b.make<ov::op::internal::PagedQSAIndexer>(
            ov::OutputVector{b.flatten(raw_query, index_heads * index_dim),
                             b.flatten(prepare->input_value(RAW_INDEX_KEYS), index_dim),
                             b.flatten(prepare->input_value(ROPE_COS), rotary_dim),
                             b.flatten(prepare->input_value(ROPE_SIN), rotary_dim),
                             q_norm_weight,
                             b.make<v1::Reshape>(prepare->input_value(K_NORM_WEIGHT), b.i64({index_dim}), false),
                             past_lens,
                             subsequence_begins,
                             block_indices,
                             block_indices_begins,
                             index_key_cache,
                             index_block_rope_cache,
                             block_index_keys_cache},
            config);
        paged_indexer->set_friendly_name(indexer_node->get_friendly_name() + "/PagedQSAIndexer");

        // ---- PagedSparseAttention on the selected blocks ----
        const auto& query_value = pm.at(query);
        const auto data_type = query_value.get_element_type();
        const auto scale =
            sparse_attention->get_input_size() > SCALE
                ? sparse_attention->input_value(SCALE)
                : b.make<v0::Constant>(data_type, ov::Shape{}, 1.0 / std::sqrt(static_cast<double>(head_size)))
                      ->output(0);
        const auto sinks = sparse_attention->get_input_size() > SINKS
                               ? sparse_attention->input_value(SINKS)
                               : b.make<v0::Constant>(data_type, ov::Shape{0}, std::vector<float>{})->output(0);
        const auto paged_attention = b.make<ov::op::internal::PagedSparseAttention>(
            ov::OutputVector{b.flatten(b.transpose_0213(query_value), num_heads * head_size),
                             b.flatten(b.transpose_0213(current_key_value), num_kv_heads * head_size),
                             b.flatten(b.transpose_0213(current_value_value), num_kv_heads * value_head_size),
                             key_cache,
                             value_cache,
                             past_lens,
                             subsequence_begins,
                             block_indices,
                             block_indices_begins,
                             paged_indexer->output(0),
                             paged_indexer->output(1),
                             scale,
                             sinks},
            sparse_attention->get_key_block_size());
        paged_attention->set_friendly_name(sparse_attention->get_friendly_name() + "/PagedSparseAttention");

        // [N * L, H * Ev] -> [N, L, H, Ev] -> [N, H, L, Ev], the SparseAttention output layout
        const auto query_shape = b.make<v3::ShapeOf>(query_value, ov::element::i64);
        const auto batch_and_length =
            b.make<v8::Gather>(query_shape, b.i64({0, 2}), b.make<v0::Constant>(ov::element::i64, ov::Shape{}, 0));
        const auto output_shape =
            b.make<v0::Concat>(ov::OutputVector{batch_and_length, b.i64({num_heads, value_head_size})}, 0);
        const auto output = b.transpose_0213(b.make<v1::Reshape>(paged_attention, output_shape, false));

        auto matched_nodes = match->nodes;
        for (const std::shared_ptr<ov::Node>& node :
             {std::shared_ptr<ov::Node>(sparse_attention), std::shared_ptr<ov::Node>(indexer_node),
              std::shared_ptr<ov::Node>(prepare), key_concat, value_concat}) {
            matched_nodes.push_back(node);
        }
        ov::copy_runtime_info(matched_nodes, b.nodes);
        output.get_node_shared_ptr()->set_friendly_name(sparse_attention->get_friendly_name());
        ov::replace_node(sparse_attention, output.get_node_shared_ptr());

        // The QSAPrepare state outputs and the key / value concatenations only feed the Assigns of these states.
        for (const auto& state : states) {
            var_ids_to_remove.insert(state->get_variable_id());
        }
        return true;
    };

    register_matcher(std::make_shared<pattern::Matcher>(attention, matcher_name), callback);
}
