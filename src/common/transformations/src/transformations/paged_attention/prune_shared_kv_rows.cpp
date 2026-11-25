// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/prune_shared_kv_rows.hpp"

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace ov;
using namespace ov::op;

namespace {

using NodeSet = std::unordered_set<Node*>;

NodeSet ancestors(const Output<Node>& output) {
    NodeSet visited;
    std::vector<Node*> pending{output.get_node()};
    while (!pending.empty()) {
        auto* node = pending.back();
        pending.pop_back();
        if (!visited.insert(node).second)
            continue;
        for (const auto& input : node->input_values())
            pending.push_back(input.get_node());
    }
    return visited;
}

Node* row_selection_source(Node* node) {
    while (is_type<v1::Reshape>(node) || is_type<v0::Squeeze>(node) || is_type<v0::Unsqueeze>(node) ||
           is_type<v0::Convert>(node)) {
        node = node->input_value(0).get_node();
    }
    return node;
}

std::shared_ptr<v1::Reshape> row_selection_indices(Node* node) {
    const auto unsqueeze = as_type<v0::Unsqueeze>(node);
    const auto gather = unsqueeze ? as_type<v8::Gather>(unsqueeze->get_input_node_ptr(0)) : nullptr;
    if (!gather || gather->get_batch_dims() != 0)
        return nullptr;
    const auto axis = as_type_ptr<v0::Constant>(gather->get_input_node_shared_ptr(2));
    if (!axis || axis->cast_vector<int64_t>() != std::vector<int64_t>{0})
        return nullptr;
    return as_type_ptr<v1::Reshape>(gather->get_input_node_shared_ptr(1));
}

}  // namespace

bool ov::pass::PruneSharedKVRows::run_on_model(const std::shared_ptr<Model>& model) {
    RUN_ON_MODEL_SCOPE(PruneSharedKVRows);

    if (!model->get_rt_info().count(shared_kv_row_selection_key))
        return false;

    for (const auto* name : {"token_type_ids", "qq_bias", "shared_kv_tokens_indices"}) {
        if (m_params.exists(name))
            return false;
    }

    std::vector<std::shared_ptr<PagedAttentionExtension>> shared_layers;
    NodeSet sources;
    for (const auto& node : model->get_ordered_ops()) {
        const auto pa = as_type_ptr<PagedAttentionExtension>(node);
        if (!pa || pa->get_write_kv_cache())
            continue;
        if (pa->get_input_partial_shape(26) != PartialShape{0})
            return false;
        NodeSet layer_sources;
        for (auto* ancestor : ancestors(pa->input_value(0))) {
            const auto indices = row_selection_indices(ancestor);
            if (!indices)
                continue;
            const auto gather = ancestor->get_input_node_ptr(0);
            const auto data_ancestors = ancestors(gather->input_value(0));
            const bool after_attention = std::any_of(data_ancestors.begin(), data_ancestors.end(), [](Node* item) {
                return is_type<PagedAttentionExtension>(item);
            });
            if (after_attention)
                layer_sources.insert(row_selection_source(indices->get_input_node_ptr(0)));
        }
        if (layer_sources.empty())
            continue;
        if (layer_sources.size() != 1)
            return false;
        sources.insert(layer_sources.begin(), layer_sources.end());
        shared_layers.push_back(pa);
    }
    if (sources.size() != 1 || shared_layers.empty())
        return false;

    std::vector<std::shared_ptr<v1::Reshape>> selections;
    for (const auto& node : model->get_ordered_ops()) {
        const auto indices = row_selection_indices(node.get());
        if (indices && sources.count(row_selection_source(indices->get_input_node_ptr(0))))
            selections.push_back(indices);
    }
    OPENVINO_ASSERT(!selections.empty(), "Shared-KV pruning has no row selections");

    auto tokens = m_params.add("shared_kv_tokens_indices", element::i64, PartialShape{-1});
    auto past_lens = m_params.add("shared_kv_past_lens", element::i32, PartialShape{-1});
    auto subsequence_begins = m_params.add("shared_kv_subsequence_begins", element::i32, PartialShape{-1});
    auto selected_ids =
        std::make_shared<v1::Reshape>(tokens, v0::Constant::create(element::i64, Shape{3}, {1, -1, 1}), false);
    copy_runtime_info(selections.front(), {tokens, selected_ids});
    copy_runtime_info(shared_layers.front()->get_input_node_shared_ptr(5), past_lens);
    copy_runtime_info(shared_layers.front()->get_input_node_shared_ptr(6), subsequence_begins);
    for (const auto& indices : selections)
        indices->input(0).replace_source_output(selected_ids);

    const auto axis = v0::Constant::create(element::i64, Shape{}, {0});
    for (const auto& pa : shared_layers) {
        for (const auto index : {1, 2}) {
            auto input = pa->input_value(index);
            auto gather = std::make_shared<v8::Gather>(input, tokens, axis);
            copy_runtime_info(input.get_node_shared_ptr(), gather);
            pa->input(index).replace_source_output(gather);
        }
        pa->input(5).replace_source_output(past_lens);
        pa->input(6).replace_source_output(subsequence_begins);
    }
    return true;
}
