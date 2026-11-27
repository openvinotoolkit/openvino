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
#include "openvino/op/broadcast.hpp"
#include "openvino/op/bucketize.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/logical_and.hpp"
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

// PagedAttentionExtension inputs.
constexpr size_t past_lens_input = 5;
constexpr size_t subsequence_begins_input = 6;
constexpr size_t xattention_threshold_input = 17;
constexpr size_t token_type_ids_input = 25;
constexpr size_t qq_bias_input = 26;

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
    const auto all_rows =
        std::make_shared<v8::Slice>(all_begins, vector(-1), vector(std::numeric_limits<int64_t>::max()), vector(1));
    const auto raw_sampled = to_i64(sampled);
    const auto normalized = std::make_shared<v1::Select>(std::make_shared<v1::Less>(raw_sampled, scalar(0)),
                                                         std::make_shared<v1::Add>(raw_sampled, all_rows),
                                                         raw_sampled);
    // Out-of-range indices select zeros; as the row count they belong to no sequence.
    const auto in_range = std::make_shared<v1::LogicalAnd>(std::make_shared<v1::GreaterEqual>(normalized, scalar(0)),
                                                           std::make_shared<v1::Less>(normalized, all_rows));
    const auto sampled_i64 = std::make_shared<v1::Select>(in_range, normalized, all_rows);
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
    const auto sampled_lookup =
        std::make_shared<v1::Minimum>(sampled_sequence, std::make_shared<v1::Subtract>(sequences, scalar(1)));
    const auto sampled_rows = std::make_shared<v1::Select>(
        in_range,
        std::make_shared<v1::Subtract>(gather(kept_ends, sampled_lookup),
                                       std::make_shared<v1::Subtract>(gather(ends, sampled_lookup), sampled_i64)),
        total);

    const auto skipped = std::make_shared<v1::Subtract>(lengths, counts);
    const auto kept_past = std::make_shared<v1::Add>(to_i64(past_lens), skipped);
    const auto kept_begins = std::make_shared<v0::Concat>(OutputVector{vector(0), kept_ends}, 0);
    return {rows,
            std::make_shared<v0::Convert>(kept_past, past_lens.get_element_type()),
            std::make_shared<v0::Convert>(kept_begins, begins.get_element_type()),
            sampled_rows};
}

// Rows are the leading axis of tensors whose first dimension carries the row symbol.
class Rows {
public:
    explicit Rows(std::shared_ptr<Symbol> symbol) : m_symbol(std::move(symbol)) {}

    bool in(const Output<Node>& output) const {
        const auto& shape = output.get_partial_shape();
        return shape.rank().is_static() && shape.size() > 0 && is_row(shape[0]);
    }

    bool mentioned(const Output<Node>& output) const {
        const auto& shape = output.get_partial_shape();
        return shape.rank().is_dynamic() || std::any_of(shape.begin(), shape.end(), [&](const Dimension& d) {
                   return is_row(d);
               });
    }

    // Rows on the leading axis and nowhere else.
    bool only_leading(const Output<Node>& output) const {
        const auto& shape = output.get_partial_shape();
        return in(output) && std::none_of(shape.begin() + 1, shape.end(), [&](const Dimension& d) {
                   return is_row(d);
               });
    }

private:
    bool is_row(const Dimension& dimension) const {
        return dimension.get_symbol() && symbol::are_equal(dimension.get_symbol(), m_symbol);
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
        if (pa->get_write_kv_cache() || pa->get_input_partial_shape(token_type_ids_input) != PartialShape{0} ||
            pa->get_input_partial_shape(qq_bias_input) != PartialShape{0} ||
            pa->get_input_partial_shape(xattention_threshold_input) != PartialShape{0} || !rows.in(pa->input_value(0)))
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
                           v1::Select>(node)) {
        // Lower-rank operands could broadcast their rows onto another axis.
        const auto rank = node->get_output_partial_shape(0).rank();
        for (const auto& input : node->input_values()) {
            if (rows.mentioned(input) && input.get_partial_shape().rank() != rank)
                return false;
        }
        return true;
    }
    if (ov::is_type<ov::op::internal::RMS>(node))
        return data_shape.rank().is_static() && data_shape.size() > 1;
    if (const auto matmul = ov::as_type_ptr<v0::MatMul>(node)) {
        const auto& weights = node->input_value(1);
        const auto rank = data_shape.rank();
        if (rank.is_dynamic() || rank.get_length() < 2 || (rank.get_length() == 2 && matmul->get_transpose_a()))
            return false;
        // Either rows times weights, or matrices batched over rows.
        return !rows.mentioned(weights) ||
               (rank.get_length() > 2 && rows.only_leading(weights) && weights.get_partial_shape().rank() == rank);
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
                    else if (!dimension.get_symbol() || ov::is_type<v0::Parameter>(node))
                        dimension.set_symbol(std::make_shared<Symbol>());
                }
            }
            ov::descriptor::set_tensor_type(output.get_tensor(), output.get_element_type(), shape);
        }
        const auto& output = node->get_output_partial_shape(0);
        if (ov::is_type<v0::Parameter>(node)) {
            const auto& names = node->output(0).get_names();
            const bool token_major = names.count("input_ids") || names.count("inputs_embeds") ||
                                     (names.count("position_ids") && output.rank().is_static() && output.size() == 1);
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

// Restores the parameter shapes and the shapes inferred from them.
void restore_symbols(const std::shared_ptr<Model>& model, const std::vector<PartialShape>& shapes) {
    ov::remove_skip_invalidation_rti(model);
    const auto& parameters = model->get_parameters();
    for (size_t i = 0; i < parameters.size(); ++i)
        parameters[i]->set_partial_shape(shapes[i]);
    model->validate_nodes_and_infer_types();
}

bool feeds_only_shapes(const Node* node, const NodeSet& shape_values) {
    for (const auto& output : node->outputs()) {
        for (const auto& consumer : output.get_target_inputs()) {
            const auto* target = consumer.get_node();
            if (!shape_values.count(target) &&
                !(ov::is_type_any_of<v1::Reshape, v1::Broadcast, v3::Broadcast>(target) && consumer.get_index() == 1))
                return false;
        }
    }
    return true;
}

// Copies runtime info to the nodes between `inputs` and `outputs`.
void copy_runtime_info_to_subgraph(const NodeVector& from, const OutputVector& outputs, const OutputVector& inputs) {
    NodeSet stop;
    for (const auto& input : inputs)
        stop.insert(input.get_node());
    NodeVector created;
    NodeSet visited;
    std::vector<std::shared_ptr<Node>> pending;
    for (const auto& output : outputs)
        pending.push_back(output.get_node_shared_ptr());
    while (!pending.empty()) {
        const auto node = pending.back();
        pending.pop_back();
        if (stop.count(node.get()) || !visited.insert(node.get()).second)
            continue;
        created.push_back(node);
        for (const auto& input : node->input_values())
            pending.push_back(input.get_node_shared_ptr());
    }
    copy_runtime_info(from, created);
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

    std::vector<PartialShape> parameter_shapes;
    for (const auto& parameter : model->get_parameters())
        parameter_shapes.push_back(parameter->get_partial_shape());
    propagate_row_symbols(ops);
    const auto& seed = selections.front()->get_input_partial_shape(0);
    const auto symbol = seed.rank().is_static() && seed.size() > 0 ? seed[0].get_symbol() : nullptr;
    const Rows rows(symbol);
    NodeSet selection_nodes;
    for (const auto& selection : selections) {
        if (!symbol || !rows.only_leading(selection->input_value(0)) ||
            selection->input_value(1) != selections.front()->input_value(1)) {
            restore_symbols(model, parameter_shapes);
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
            row_outputs &= output.get_target_inputs().empty() || rows.only_leading(output);
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
        // Shapes of pruned tensors may only size other pruned tensors.
        if (shape_values.count(node.get()) && !feeds_only_shapes(node.get(), shape_values)) {
            restore_symbols(model, parameter_shapes);
            return false;
        }
        for (const auto& input : node->inputs()) {
            const auto source = input.get_source_output();
            if (region.count(source.get_node()))
                continue;
            const auto& shape = source.get_partial_shape();
            if (rows.only_leading(source)) {
                cuts.emplace(input, source);
            } else if (rows.mentioned(source) || shape_values.count(source.get_node()) ||
                       (!pa && shape.rank().is_static() && shape.size() > 0 && shape[0].is_dynamic()) ||
                       (pa && input.get_index() < 3)) {
                // Shapes of unpruned tensors, or a tensor with unknown leading rows.
                restore_symbols(model, parameter_shapes);
                return false;
            }
        }
    }
    const bool selects_region = std::all_of(selections.begin(), selections.end(), [&](const auto& selection) {
        return region.count(selection->get_input_node_ptr(0)) > 0;
    });
    const bool consistent =
        selects_region && !attentions.empty() && std::all_of(attentions.begin(), attentions.end(), [&](auto* pa) {
            return pa->input_value(past_lens_input) == attentions.front()->input_value(past_lens_input) &&
                   pa->input_value(subsequence_begins_input) ==
                       attentions.front()->input_value(subsequence_begins_input);
        });
    if (!consistent) {
        restore_symbols(model, parameter_shapes);
        return false;
    }

    const auto indices = selections.front()->input_value(1);
    const auto kept = keep_sampled_rows(indices,
                                        attentions.front()->input_value(past_lens_input),
                                        attentions.front()->input_value(subsequence_begins_input));
    copy_runtime_info_to_subgraph(NodeVector(selections.begin(), selections.end()),
                                  {kept.rows, kept.past_lens, kept.subsequence_begins, kept.sampled_rows},
                                  {indices,
                                   attentions.front()->input_value(past_lens_input),
                                   attentions.front()->input_value(subsequence_begins_input)});
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
        pa->input(past_lens_input).replace_source_output(kept.past_lens);
        pa->input(subsequence_begins_input).replace_source_output(kept.subsequence_begins);
    }
    for (const auto& selection : selections)
        selection->input(1).replace_source_output(kept.sampled_rows);
    restore_symbols(model, parameter_shapes);
    return true;
}
