// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/prune_shared_kv_rows.hpp"

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bucketize.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace ov;
using namespace ov::op;

namespace {

using NodeSet = std::unordered_set<Node*>;

// Barrier nodes are included but not traversed.
NodeSet ancestors(const Output<Node>& output, const NodeSet& barriers = {}) {
    NodeSet visited;
    std::vector<Node*> pending{output.get_node()};
    while (!pending.empty()) {
        auto* node = pending.back();
        pending.pop_back();
        if (!visited.insert(node).second || barriers.count(node))
            continue;
        for (const auto& input : node->input_values())
            pending.push_back(input.get_node());
    }
    return visited;
}

bool reaches_any(const NodeSet& nodes, const NodeSet& targets) {
    return std::any_of(nodes.begin(), nodes.end(), [&](Node* node) {
        return targets.count(node);
    });
}

Node* row_selection_source(Node* node) {
    while (is_type<v1::Reshape>(node) || is_type<v0::Squeeze>(node) || is_type<v0::Unsqueeze>(node) ||
           is_type<v0::Convert>(node)) {
        node = node->input_value(0).get_node();
    }
    return node;
}

std::shared_ptr<v1::Reshape> row_selection_indices(Node* node) {
    const auto gather = as_type<v8::Gather>(node);
    if (!gather || gather->get_batch_dims() != 0)
        return nullptr;
    const auto axis = as_type_ptr<v0::Constant>(gather->get_input_node_shared_ptr(2));
    if (!axis || axis->cast_vector<int64_t>() != std::vector<int64_t>{0})
        return nullptr;
    return as_type_ptr<v1::Reshape>(gather->get_input_node_shared_ptr(1));
}

std::shared_ptr<v0::Parameter> find_parameter(const std::shared_ptr<Model>& model, const std::string& name) {
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == name || parameter->output(0).get_names().count(name))
            return parameter;
    }
    return nullptr;
}

Output<Node> to_i64(const Output<Node>& value) {
    return value.get_element_type() == element::i64 ? value : std::make_shared<v0::Convert>(value, element::i64);
}

// Keeps each sequence's rows from its first sampled row to its end, or its last row when none is sampled.
struct KeptRows {
    Output<Node> rows;
    Output<Node> past_lens;
    Output<Node> subsequence_begins;
    Output<Node> sampled_rows;
};

KeptRows keep_sampled_rows(const Output<Node>& sampled, const Output<Node>& past_lens, const Output<Node>& begins) {
    const auto scalar = [](int64_t value) {
        return v0::Constant::create(element::i64, Shape{}, {value});
    };
    const auto vector = [](int64_t value) {
        return v0::Constant::create(element::i64, Shape{1}, {value});
    };
    const auto gather = [&](const Output<Node>& data, const Output<Node>& indices) {
        return std::make_shared<v8::Gather>(data, indices, scalar(0));
    };
    const auto all_begins = to_i64(begins);
    const auto starts = std::make_shared<v8::Slice>(all_begins, vector(0), vector(-1), vector(1));
    const auto ends =
        std::make_shared<v8::Slice>(all_begins, vector(1), vector(std::numeric_limits<int64_t>::max()), vector(1));
    const auto sampled_i64 = to_i64(sampled);
    const auto sampled_sequence = std::make_shared<v3::Bucketize>(sampled_i64, ends, element::i64, false);

    const auto sequences = std::make_shared<v0::Squeeze>(std::make_shared<v3::ShapeOf>(ends, element::i64));
    const auto sequence_ids = std::make_shared<v4::Range>(scalar(0), sequences, scalar(1), element::i64);
    const auto matches =
        std::make_shared<v1::Equal>(std::make_shared<v0::Unsqueeze>(sequence_ids, scalar(1)), sampled_sequence);
    // The end column keeps the reduction non-empty when a sequence has no sampled rows.
    const auto end_column = std::make_shared<v0::Unsqueeze>(ends, scalar(1));
    const auto candidates = std::make_shared<v0::Concat>(
        OutputVector{std::make_shared<v1::Select>(matches, sampled_i64, end_column), end_column},
        1);
    const auto first_sampled = std::make_shared<v1::ReduceMin>(candidates, scalar(1));
    const auto lengths = std::make_shared<v1::Subtract>(ends, starts);
    const auto counts = std::make_shared<v1::Minimum>(
        std::make_shared<v1::Maximum>(std::make_shared<v1::Subtract>(ends, first_sampled), scalar(1)),
        lengths);
    const auto kept_ends = std::make_shared<v0::CumSum>(counts, scalar(0));

    const auto total = std::make_shared<v1::ReduceSum>(counts, scalar(0));
    const auto kept_ids = std::make_shared<v4::Range>(scalar(0), total, scalar(1), element::i64);
    const auto kept_sequence = std::make_shared<v3::Bucketize>(kept_ids, kept_ends, element::i64, false);
    // Row i of a sequence's kept suffix maps to end - (kept_end - i).
    const auto rows =
        std::make_shared<v1::Subtract>(gather(ends, kept_sequence),
                                       std::make_shared<v1::Subtract>(gather(kept_ends, kept_sequence), kept_ids));
    const auto sampled_rows =
        std::make_shared<v1::Subtract>(gather(kept_ends, sampled_sequence),
                                       std::make_shared<v1::Subtract>(gather(ends, sampled_sequence), sampled_i64));

    const auto skipped = std::make_shared<v1::Subtract>(lengths, counts);
    const auto kept_past = std::make_shared<v1::Add>(to_i64(past_lens), skipped);
    const auto kept_begins = std::make_shared<v0::Concat>(OutputVector{vector(0), kept_ends}, 0);
    return {rows,
            std::make_shared<v0::Convert>(kept_past, past_lens.get_element_type()),
            std::make_shared<v0::Convert>(kept_begins, begins.get_element_type()),
            std::make_shared<v0::Convert>(sampled_rows, sampled.get_element_type())};
}

}  // namespace

bool ov::pass::PruneSharedKVRows::run_on_model(const std::shared_ptr<Model>& model) {
    RUN_ON_MODEL_SCOPE(PruneSharedKVRows);

    if (!model->get_rt_info().count(shared_kv_row_selection_key))
        return false;

    const auto sampled = find_parameter(model, "sampled_tokens_indices");
    if (!sampled || find_parameter(model, "token_type_ids") || find_parameter(model, "qq_bias"))
        return false;

    std::vector<std::shared_ptr<PagedAttentionExtension>> shared_layers;
    NodeSet sources;
    for (const auto& node : model->get_ordered_ops()) {
        const auto pa = as_type_ptr<PagedAttentionExtension>(node);
        if (!pa || pa->get_write_kv_cache())
            continue;
        if (pa->get_input_partial_shape(26) != PartialShape{0} || !is_type<v0::Parameter>(pa->get_input_node_ptr(5)) ||
            !is_type<v0::Parameter>(pa->get_input_node_ptr(6)))
            return false;
        NodeSet layer_sources;
        for (auto* ancestor : ancestors(pa->input_value(0))) {
            const auto indices = row_selection_indices(ancestor);
            if (!indices)
                continue;
            const auto data_ancestors = ancestors(ancestor->input_value(0));
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

    NodeSet shared_nodes;
    for (const auto& pa : shared_layers) {
        if (pa->input_value(5) != shared_layers.front()->input_value(5) ||
            pa->input_value(6) != shared_layers.front()->input_value(6))
            return false;
        for (size_t i = 1; i < pa->get_output_size(); ++i) {
            if (!pa->output(i).get_target_inputs().empty())
                return false;
        }
        shared_nodes.insert(pa.get());
    }
    // Shared-layer rows may reach outputs only through the sampled-row gathers.
    NodeSet sampled_gathers;
    for (const auto& consumer : sampled->output(0).get_target_inputs()) {
        auto* gather = as_type<v8::Gather>(consumer.get_node());
        if (!gather || consumer.get_index() != 1 || !reaches_any(ancestors(gather->input_value(0)), shared_nodes))
            return false;
        sampled_gathers.insert(gather);
    }
    for (const auto& result : model->get_results()) {
        if (reaches_any(ancestors(result->input_value(0), sampled_gathers), shared_nodes))
            return false;
    }

    const auto sampled_consumers = sampled->output(0).get_target_inputs();
    const auto kept =
        keep_sampled_rows(sampled, shared_layers.front()->input_value(5), shared_layers.front()->input_value(6));
    for (const auto& consumer : sampled_consumers)
        consumer.replace_source_output(kept.sampled_rows);
    auto selected_ids =
        std::make_shared<v1::Reshape>(kept.rows, v0::Constant::create(element::i64, Shape{3}, {1, -1, 1}), false);
    copy_runtime_info(selections.front(), selected_ids);
    for (const auto& indices : selections)
        indices->input(0).replace_source_output(selected_ids);

    const auto axis = v0::Constant::create(element::i64, Shape{}, {0});
    for (const auto& pa : shared_layers) {
        for (const auto index : {1, 2}) {
            auto input = pa->input_value(index);
            auto gather = std::make_shared<v8::Gather>(input, kept.rows, axis);
            copy_runtime_info(input.get_node_shared_ptr(), gather);
            pa->input(index).replace_source_output(gather);
        }
        pa->input(5).replace_source_output(kept.past_lens);
        pa->input(6).replace_source_output(kept.subsequence_begins);
    }
    model->validate_nodes_and_infer_types();
    return true;
}
