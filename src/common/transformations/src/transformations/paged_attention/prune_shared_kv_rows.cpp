// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/prune_shared_kv_rows.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <unordered_set>
#include <vector>

#include "itt.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bucketize.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mvn.hpp"
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
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/binary_elementwise_comparison.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"
#include "openvino/op/util/symbolic_info.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/variadic_split.hpp"
#include "ov_ops/rms.hpp"

using namespace ov;
using namespace ov::op;

namespace {

using NodeSet = std::unordered_set<const Node*>;

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

// Rows are the leading axis of tensors whose first dimension carries the row symbol.
class Rows {
public:
    explicit Rows(std::shared_ptr<Symbol> symbol) : m_symbol(std::move(symbol)) {}

    bool in(const Output<Node>& output) const {
        const auto& shape = output.get_partial_shape();
        return shape.rank().is_static() && shape.size() > 0 && shape[0].get_symbol() &&
               symbol::are_equal(shape[0].get_symbol(), m_symbol);
    }

private:
    std::shared_ptr<Symbol> m_symbol;
};

std::optional<std::vector<int64_t>> constant_axes(const Output<Node>& axes, const PartialShape& data_shape) {
    const auto constant = ov::as_type_ptr<v0::Constant>(axes.get_node_shared_ptr());
    if (!constant || data_shape.rank().is_dynamic())
        return std::nullopt;
    auto values = constant->cast_vector<int64_t>();
    for (auto& value : values)
        value = value < 0 ? value + data_shape.rank().get_length() : value;
    return values;
}

bool excludes_rows(const Output<Node>& axes, const PartialShape& data_shape) {
    const auto values = constant_axes(axes, data_shape);
    return values && std::find(values->begin(), values->end(), 0) == values->end();
}

bool excludes_rows(int64_t axis, const PartialShape& data_shape) {
    return data_shape.rank().is_static() && (axis < 0 ? axis + data_shape.rank().get_length() : axis) != 0;
}

bool is_shared_attention(const Node* node) {
    const auto pa = ov::as_type<const PagedAttentionExtension>(node);
    return pa && !pa->get_write_kv_cache();
}

bool depends_on_shared_attention(const Output<Node>& output) {
    NodeSet visited;
    std::vector<const Node*> pending{output.get_node()};
    while (!pending.empty()) {
        const auto* node = pending.back();
        pending.pop_back();
        if (!visited.insert(node).second)
            continue;
        if (is_shared_attention(node))
            return true;
        for (const auto& input : node->input_values())
            pending.push_back(input.get_node());
    }
    return false;
}

// Whether each output row of the node depends only on the same input row.
bool computes_rows_independently(const std::shared_ptr<Node>& node, const Rows& rows) {
    const auto& data_shape = node->get_input_partial_shape(0);
    if (const auto pa = ov::as_type_ptr<PagedAttentionExtension>(node)) {
        if (pa->get_write_kv_cache() || pa->get_input_partial_shape(25) != PartialShape{0} ||
            pa->get_input_partial_shape(26) != PartialShape{0} || !rows.in(pa->input_value(0)))
            return false;
        for (size_t i = 1; i < pa->get_output_size(); ++i) {
            if (!pa->output(i).get_target_inputs().empty())
                return false;
        }
        return true;
    }
    if (ov::is_type_any_of<op::util::UnaryElementwiseArithmetic,
                           op::util::BinaryElementwiseArithmetic,
                           op::util::BinaryElementwiseComparison,
                           op::util::BinaryElementwiseLogical,
                           v0::Convert,
                           v0::Clamp,
                           v4::Swish,
                           v1::Select,
                           ov::op::internal::RMS>(node))
        return true;
    if (const auto matmul = ov::as_type_ptr<v0::MatMul>(node)) {
        const bool matrices = data_shape.rank().is_static() && data_shape.size() == 2;
        return !(matrices && matmul->get_transpose_a()) && !(matrices && rows.in(matmul->input_value(1)));
    }
    if (ov::is_type_any_of<v1::Reshape, v0::Squeeze, v0::Unsqueeze>(node))
        return rows.in(node->input_value(0));
    if (ov::is_type<v1::Transpose>(node)) {
        const auto order = constant_axes(node->input_value(1), data_shape);
        return order && !order->empty() && order->front() == 0;
    }
    if (const auto concat = ov::as_type_ptr<v0::Concat>(node))
        return excludes_rows(concat->get_axis(), node->get_output_partial_shape(0));
    if (ov::is_type_any_of<v1::Split, v1::VariadicSplit>(node))
        return excludes_rows(node->input_value(1), data_shape);
    if (ov::is_type<v8::Slice>(node))
        return node->get_input_size() == 5 && excludes_rows(node->input_value(4), data_shape);
    if (const auto gather = ov::as_type_ptr<op::util::GatherBase>(node)) {
        const auto axis = constant_axes(gather->input_value(2), data_shape);
        if (!axis || axis->size() != 1 || gather->get_batch_dims() != 0)
            return false;
        // Either columns of each row, or a per-row lookup into a static table.
        return axis->front() != 0 ? rows.in(gather->input_value(0)) && !rows.in(gather->input_value(1))
                                  : data_shape.rank().is_static() && data_shape.size() > 0 &&
                                        data_shape[0].is_static() && rows.in(gather->input_value(1));
    }
    if (ov::is_type<op::util::ArithmeticReductionKeepDims>(node))
        return excludes_rows(node->input_value(1), data_shape);
    if (const auto softmax = ov::as_type_ptr<v8::Softmax>(node))
        return excludes_rows(softmax->get_axis(), data_shape);
    if (const auto softmax = ov::as_type_ptr<v5::LogSoftmax>(node))
        return excludes_rows(softmax->get_axis(), data_shape);
    if (ov::is_type<v6::MVN>(node))
        return excludes_rows(node->input_value(1), data_shape);
    return false;
}

// Assigns symbols like SymbolicPropagation and also equates the rows of token-major PagedAttention
// inputs, and the leading dimensions of reshapes that keep the other sizes.
void propagate_row_symbols(const std::vector<std::shared_ptr<Node>>& ops) {
    const auto inner_size = [](const PartialShape& shape) -> std::optional<int64_t> {
        if (shape.rank().is_dynamic() || shape.size() == 0 || !shape[0].get_symbol())
            return std::nullopt;
        int64_t size = 1;
        for (size_t i = 1; i < shape.size(); ++i) {
            if (shape[i].is_dynamic())
                return std::nullopt;
            size *= shape[i].get_length();
        }
        return size;
    };
    std::shared_ptr<Symbol> tokens;
    for (const auto& node : ops) {
        node->invalidate_values();
        for (auto& output : node->outputs())
            ov::skip_invalidation(output);
        node->revalidate_and_infer_types();
        for (auto& output : node->outputs()) {
            auto shape = output.get_partial_shape();
            if (shape.rank().is_static()) {
                for (auto& dimension : shape) {
                    if (dimension.is_static())
                        dimension.set_symbol(nullptr);
                    else if (!dimension.get_symbol())
                        dimension.set_symbol(std::make_shared<Symbol>());
                }
            }
            ov::descriptor::set_tensor_type(output.get_tensor(), output.get_element_type(), shape);
        }
        const auto& output = node->get_output_partial_shape(0);
        if (ov::is_type<v0::Parameter>(node)) {
            const auto& names = node->output(0).get_names();
            const bool token_major = names.count("input_ids") || names.count("inputs_embeds") ||
                                     (names.count("position_ids") && output.rank().get_length() == 1);
            if (token_major && output.rank().is_static() && output.size() > 0 && output[0].get_symbol()) {
                if (tokens)
                    symbol::set_equal(tokens, output[0].get_symbol());
                tokens = output[0].get_symbol();
            }
        } else if (ov::is_type_any_of<v1::Reshape, v0::Squeeze, v0::Unsqueeze>(node)) {
            const auto& input = node->get_input_partial_shape(0);
            const auto input_size = inner_size(input), output_size = inner_size(output);
            if (input_size && output_size && *input_size == *output_size)
                symbol::set_equal(input[0].get_symbol(), output[0].get_symbol());
        }
    }
}

void clear_symbols(const std::shared_ptr<Model>& model) {
    ov::remove_skip_invalidation_rti(model);
    for (const auto& parameter : model->get_parameters()) {
        auto shape = parameter->get_partial_shape();
        for (auto& dimension : shape)
            dimension.set_symbol(nullptr);
        parameter->set_partial_shape(shape);
    }
    model->validate_nodes_and_infer_types();
}

}  // namespace

bool ov::pass::PruneSharedKVRows::run_on_model(const std::shared_ptr<Model>& model) {
    RUN_ON_MODEL_SCOPE(PruneSharedKVRows);

    const auto ops = model->get_ordered_ops();
    if (std::none_of(ops.begin(), ops.end(), [](const std::shared_ptr<Node>& node) {
            return is_shared_attention(node.get());
        }))
        return false;
    // The row selection feeding outputs, such as GenAI's sampled_tokens_indices before the LM head.
    std::vector<std::shared_ptr<v8::Gather>> selections;
    for (const auto& node : ops) {
        const auto gather = ov::as_type_ptr<v8::Gather>(node);
        const auto axis = gather ? ov::as_type_ptr<v0::Constant>(gather->get_input_node_shared_ptr(2)) : nullptr;
        if (gather && gather->get_batch_dims() == 0 && axis &&
            axis->cast_vector<int64_t>() == std::vector<int64_t>{0} &&
            ov::is_type<v0::Parameter>(gather->get_input_node_ptr(1)) &&
            gather->get_input_partial_shape(1).rank() == 1 && depends_on_shared_attention(gather->input_value(0)))
            selections.push_back(gather);
    }
    if (selections.empty())
        return false;

    propagate_row_symbols(ops);
    const auto symbol = selections.front()->get_input_partial_shape(0)[0].get_symbol();
    const Rows rows(symbol);
    NodeSet selection_nodes;
    for (const auto& selection : selections) {
        if (!symbol || !rows.in(selection->input_value(0)) ||
            selection->input_value(1) != selections.front()->input_value(1)) {
            clear_symbols(model);
            return false;
        }
        selection_nodes.insert(selection.get());
    }

    // Values computed only from shapes and constants.
    NodeSet shape_values;
    for (const auto& node : ops) {
        if (ov::is_type_any_of<v0::ShapeOf, v3::ShapeOf>(node)) {
            shape_values.insert(node.get());
            continue;
        }
        bool from_shapes = false, only_shapes = node->get_input_size() > 0 && !ov::is_type<v0::Result>(node);
        for (const auto& input : node->input_values()) {
            from_shapes |= shape_values.count(input.get_node()) > 0;
            only_shapes &= shape_values.count(input.get_node()) || ov::is_type<v0::Constant>(input.get_node());
        }
        if (from_shapes && only_shapes)
            shape_values.insert(node.get());
    }

    // Grow the region backwards: a node joins when all its consumers are in it or select its rows.
    NodeSet region;
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
        const auto& node = *it;
        if (ov::is_type_any_of<v0::Parameter, v0::Constant, v0::Result>(node) || selection_nodes.count(node.get()))
            continue;
        bool consumed = false, inside = true;
        for (const auto& output : node->outputs()) {
            for (const auto& consumer : output.get_target_inputs()) {
                consumed = true;
                inside &= region.count(consumer.get_node()) ||
                          (selection_nodes.count(consumer.get_node()) && consumer.get_index() == 0);
            }
        }
        if (!consumed || !inside)
            continue;
        bool row_outputs = true;
        for (const auto& output : node->outputs())
            row_outputs &= output.get_target_inputs().empty() || rows.in(output);
        if (shape_values.count(node.get()) || (row_outputs && computes_rows_independently(node, rows)))
            region.insert(node.get());
    }

    std::vector<PagedAttentionExtension*> attentions;
    std::map<Input<Node>, Output<Node>> cuts;
    for (const auto& node : ops) {
        if (!region.count(node.get()))
            continue;
        const auto pa = ov::as_type<PagedAttentionExtension>(node.get());
        if (pa)
            attentions.push_back(pa);
        for (const auto& input : node->inputs()) {
            const auto source = input.get_source_output();
            if (region.count(source.get_node()))
                continue;
            const auto& shape = source.get_partial_shape();
            if (rows.in(source)) {
                cuts.emplace(input, source);
            } else if (shape_values.count(source.get_node()) ||
                       (!pa && shape.rank().is_static() && shape.size() > 0 && shape[0].is_dynamic()) ||
                       (pa && input.get_index() < 3)) {
                // Shapes of unpruned tensors, or a tensor with unknown leading rows.
                clear_symbols(model);
                return false;
            }
        }
    }
    const bool consistent = !attentions.empty() && std::all_of(attentions.begin(), attentions.end(), [&](auto* pa) {
        return pa->input_value(5) == attentions.front()->input_value(5) &&
               pa->input_value(6) == attentions.front()->input_value(6);
    });
    if (!consistent) {
        clear_symbols(model);
        return false;
    }

    const auto indices = selections.front()->input_value(1);
    const auto kept =
        keep_sampled_rows(indices, attentions.front()->input_value(5), attentions.front()->input_value(6));
    const auto axis = v0::Constant::create(element::i64, Shape{}, {0});
    std::map<Output<Node>, Output<Node>> gathered;
    for (auto& [input, source] : cuts) {
        auto& replacement = gathered[source];
        if (!replacement.get_node()) {
            replacement = std::make_shared<v8::Gather>(source, kept.rows, axis);
            copy_runtime_info(source.get_node_shared_ptr(), replacement.get_node_shared_ptr());
        }
        Input<Node>(input).replace_source_output(replacement);
    }
    for (auto* pa : attentions) {
        pa->input(5).replace_source_output(kept.past_lens);
        pa->input(6).replace_source_output(kept.subsequence_begins);
    }
    for (const auto& selection : selections)
        selection->input(1).replace_source_output(kept.sampled_rows);
    clear_symbols(model);
    return true;
}
