// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "materialize_weights.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <numeric>
#include <optional>

#include "helper_ops/gguf_constant.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/variadic_split.hpp"
#include "quant/weights.hpp"

namespace ov::frontend::gguf::pass {
namespace {

struct SourceRow {
    std::shared_ptr<const GgufWeight> source;
    size_t row;
};

struct WeightPlan {
    ov::Shape shape;
    std::vector<SourceRow> rows;
    std::vector<int64_t> columns;
};

bool compatible(const WeightPlan& lhs, const WeightPlan& rhs) {
    if (lhs.shape.size() != rhs.shape.size() || lhs.shape.empty() || lhs.rows.empty() || rhs.rows.empty() ||
        !std::equal(lhs.shape.begin() + 1, lhs.shape.end(), rhs.shape.begin() + 1) || lhs.columns != rhs.columns)
        return false;
    const auto& a = *lhs.rows.front().source;
    const auto& b = *rhs.rows.front().source;
    return a.qtype == b.qtype && a.native == b.native && same_weight_conversion_policy(a.source_name, b.source_name) &&
           (a.qtype != GGUF_TYPE_Q5_K ||
            native_q5_k_is_symmetric(a.source_name) == native_q5_k_is_symmetric(b.source_name));
}

using Plans = std::map<ov::Node*, std::optional<WeightPlan>>;

std::optional<std::vector<int64_t>> constant_values(const std::shared_ptr<ov::Node>& node) {
    const auto value = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!value || !value->get_element_type().is_integral_number())
        return std::nullopt;
    return value->cast_vector<int64_t>();
}

std::optional<WeightPlan> plan_weight(const std::shared_ptr<ov::Node>& node, Plans& plans) {
    if (const auto found = plans.find(node.get()); found != plans.end())
        return found->second;
    std::optional<WeightPlan> result;
    if (const auto leaf = ov::as_type_ptr<GGUFConstant>(node)) {
        const auto& source = leaf->source();
        if (source.native && !source.shape.empty() && source.shape.size() <= 2 && ov::shape_size(source.shape) != 0 &&
            (source.shape.size() == 2 || gguf_weight_group_size(source) == 1)) {
            WeightPlan plan;
            plan.shape = source.shape;
            auto shared = std::make_shared<GgufWeight>(source);
            for (size_t row = 0; row < source.shape[0]; ++row)
                plan.rows.push_back({shared, row});
            result = std::move(plan);
        }
    } else if (const auto concat = ov::as_type_ptr<ov::op::v0::Concat>(node)) {
        if (concat->get_axis() == 0 && concat->get_input_size() >= 2) {
            result = plan_weight(concat->get_input_node_shared_ptr(0), plans);
            for (size_t i = 1; result && i < concat->get_input_size(); ++i) {
                const auto next = plan_weight(concat->get_input_node_shared_ptr(i), plans);
                if (!next || !compatible(*result, *next)) {
                    result.reset();
                    break;
                }
                result->rows.insert(result->rows.end(), next->rows.begin(), next->rows.end());
                result->shape[0] += next->shape[0];
            }
        }
    } else if (ov::is_type<ov::op::v8::Gather>(node) || ov::is_type<ov::op::v7::Gather>(node) ||
               ov::is_type<ov::op::v1::Gather>(node) || ov::is_type<ov::op::v8::Slice>(node)) {
        auto input = plan_weight(node->get_input_node_shared_ptr(0), plans);
        std::vector<int64_t> selection;
        int64_t axis = -1;
        bool valid = false;
        if (input && ov::is_type<ov::op::v8::Slice>(node) && node->get_input_size() == 5) {
            const auto start = constant_values(node->get_input_node_shared_ptr(1));
            const auto stop = constant_values(node->get_input_node_shared_ptr(2));
            const auto step = constant_values(node->get_input_node_shared_ptr(3));
            const auto axes = constant_values(node->get_input_node_shared_ptr(4));
            if (start && stop && step && axes && start->size() == 1 && stop->size() == 1 && step->size() == 1 &&
                axes->size() == 1 && (*step)[0] > 0) {
                axis = (*axes)[0];
                if (axis < 0)
                    axis += input->shape.size();
                if (axis >= 0 && static_cast<size_t>(axis) < input->shape.size()) {
                    const auto extent = static_cast<int64_t>(input->shape[axis]);
                    auto begin = (*start)[0], end = (*stop)[0];
                    if (begin < 0)
                        begin += extent;
                    if (end < 0)
                        end += extent;
                    begin = std::clamp(begin, int64_t{0}, extent);
                    end = std::clamp(end, int64_t{0}, extent);
                    for (auto i = begin; i < end;) {
                        selection.push_back(i);
                        if ((*step)[0] >= end - i)
                            break;
                        i += (*step)[0];
                    }
                    valid = true;
                }
            }
        } else if (input) {
            const auto indices = constant_values(node->get_input_node_shared_ptr(1));
            const auto axes = constant_values(node->get_input_node_shared_ptr(2));
            const auto gather = ov::as_type_ptr<ov::op::util::GatherBase>(node);
            if (indices && axes && axes->size() == 1 && gather && gather->get_batch_dims() == 0 &&
                node->get_input_shape(1).size() == 1) {
                axis = (*axes)[0];
                if (axis < 0)
                    axis += input->shape.size();
                selection = *indices;
                valid = axis >= 0 && static_cast<size_t>(axis) < input->shape.size();
                if (valid) {
                    const auto extent = static_cast<int64_t>(input->shape[axis]);
                    for (auto& index : selection) {
                        if (index < 0)
                            index += extent;
                        valid &= index >= 0 && index < extent;
                    }
                }
            }
        }
        if (valid && !selection.empty()) {
            auto plan = *input;
            if (axis == 0) {
                plan.rows.clear();
                for (const auto index : selection)
                    plan.rows.push_back(input->rows[index]);
                plan.shape[0] = selection.size();
                result = std::move(plan);
            } else if (axis == 1 && input->shape.size() == 2) {
                plan.columns.clear();
                for (const auto index : selection)
                    plan.columns.push_back(input->columns.empty() ? index : input->columns[index]);
                if (std::all_of(plan.rows.begin(), plan.rows.end(), [&](const auto& row) {
                        return gguf_column_groups_are_aligned(*row.source, plan.columns);
                    })) {
                    plan.shape[1] = selection.size();
                    result = std::move(plan);
                }
            }
        }
    } else if (ov::is_type<ov::op::v1::Reshape>(node) || ov::is_type<ov::op::v0::Convert>(node) ||
               ov::is_type<ov::op::v1::Transpose>(node)) {
        const auto input = plan_weight(node->get_input_node_shared_ptr(0), plans);
        if (input && node->get_output_partial_shape(0).is_static() && node->get_output_shape(0) == input->shape &&
            node->get_output_element_type(0) == ov::element::f32) {
            if (ov::is_type<ov::op::v1::Transpose>(node)) {
                const auto order = constant_values(node->get_input_node_shared_ptr(1));
                bool identity = order && order->size() == input->shape.size();
                for (size_t i = 0; identity && i < order->size(); ++i)
                    identity &= (*order)[i] == static_cast<int64_t>(i);
                if (identity)
                    result = *input;
            } else
                result = *input;
        }
    }
    plans.emplace(node.get(), result);
    return result;
}

std::array<ov::Tensor*, 3> components(WeightTensors& tensors) {
    return {&tensors.weight, &tensors.scales, &tensors.zero_point};
}

GgufWeight source_rows(const GgufWeight& source, size_t first, size_t count) {
    auto result = source;
    const size_t row_bytes = source.data.get_byte_size() / source.shape[0];
    result.shape[0] = count;
    auto* pointer = const_cast<uint8_t*>(static_cast<const uint8_t*>(source.data.data())) + first * row_bytes;
    const auto data_shape =
        source.data.get_element_type() == ov::element::u8 ? ov::Shape{count * row_bytes} : result.shape;
    auto view = ov::Tensor(source.data.get_element_type(), data_shape, pointer);
    result.data = ov::Tensor(view, std::make_shared<ov::Tensor>(source.data));
    result.mapped.reset();
    return result;
}

std::shared_ptr<ov::Node> materialize_plan(const WeightPlan& plan) {
    const auto& first_source = *plan.rows.front().source;
    auto output = allocate_gguf_weight(first_source, plan.shape);
    const auto outputs = components(output);
    for (size_t begin = 0; begin < plan.rows.size();) {
        const auto& row = plan.rows[begin];
        size_t end = begin + 1;
        while (end < plan.rows.size() && plan.rows[end].source == row.source &&
               plan.rows[end].row == row.row + end - begin)
            ++end;
        auto source = source_rows(*row.source, row.row, end - begin);
        WeightTensors views;
        const auto destinations = components(views);
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (!*outputs[i])
                continue;
            auto shape = outputs[i]->get_shape();
            const auto row_bytes = outputs[i]->get_byte_size() / shape[0];
            shape[0] = end - begin;
            *destinations[i] = ov::Tensor(outputs[i]->get_element_type(),
                                          shape,
                                          static_cast<uint8_t*>(outputs[i]->data()) + begin * row_bytes);
        }
        if (plan.columns.empty()) {
            unpack_gguf_weight_into(source, views);
        } else {
            auto unpacked = unpack_gguf_weight(source);
            const auto inputs = components(unpacked);
            const auto group = gguf_weight_group_size(source);
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (!*inputs[i])
                    continue;
                const auto input_row_bytes = inputs[i]->get_byte_size() / (end - begin);
                const auto output_row_bytes = destinations[i]->get_byte_size() / (end - begin);
                const auto block_bytes = input_row_bytes / (source.shape.back() / group);
                const auto* in = static_cast<const uint8_t*>(inputs[i]->data());
                auto* out = static_cast<uint8_t*>(destinations[i]->data());
                for (size_t r = 0; r < end - begin; ++r)
                    for (size_t col = 0; col < plan.columns.size(); col += group)
                        std::memcpy(out + r * output_row_bytes + col / group * block_bytes,
                                    in + r * input_row_bytes + plan.columns[col] / group * block_bytes,
                                    block_bytes);
            }
        }
        if (row.source->mapped) {
            const auto* base = reinterpret_cast<const uint8_t*>(row.source->mapped->data());
            row.source->mapped->hint_evict(static_cast<const uint8_t*>(source.data.data()) - base,
                                           source.data.get_byte_size());
        }
        begin = end;
    }
    return make_weight_node(output, first_source.qtype, first_source.source_name);
}

void fuse_projections(const std::shared_ptr<ov::Model>& model) {
    using MatMul = ov::op::v0::MatMul;
    std::map<std::pair<ov::Node*, size_t>, std::vector<std::shared_ptr<MatMul>>> groups;
    Plans plans;
    for (const auto& node : model->get_ordered_ops()) {
        const auto matmul = ov::as_type_ptr<MatMul>(node);
        if (!matmul || matmul->get_transpose_a() || !matmul->get_transpose_b())
            continue;
        const auto plan = plan_weight(matmul->get_input_node_shared_ptr(1), plans);
        if (!plan || plan->shape.size() != 2)
            continue;
        const auto type = plan->rows.front().source->qtype;
        if (type != GGUF_TYPE_Q2_0 && type != GGUF_TYPE_PQ2_0 && type != GGUF_TYPE_PTQ1_0)
            continue;
        const auto input = matmul->input_value(0);
        groups[{input.get_node(), input.get_index()}].push_back(matmul);
    }
    for (const auto& [input, candidates] : groups) {
        std::vector<bool> used(candidates.size());
        for (size_t first = 0; first < candidates.size(); ++first) {
            if (used[first])
                continue;
            const auto first_plan = plan_weight(candidates[first]->get_input_node_shared_ptr(1), plans);
            std::vector<std::shared_ptr<MatMul>> projections{candidates[first]};
            ov::OutputVector weights{candidates[first]->input_value(1)};
            std::vector<int64_t> lengths{static_cast<int64_t>(first_plan->shape[0])};
            for (size_t i = first + 1; i < candidates.size(); ++i) {
                if (used[i])
                    continue;
                const auto plan = plan_weight(candidates[i]->get_input_node_shared_ptr(1), plans);
                if (!compatible(*first_plan, *plan))
                    continue;
                projections.push_back(candidates[i]);
                weights.push_back(candidates[i]->input_value(1));
                lengths.push_back(plan->shape[0]);
                used[i] = true;
            }
            if (projections.size() == 1)
                continue;
            auto concat = std::make_shared<ov::op::v0::Concat>(weights, 0);
            auto merged = std::make_shared<MatMul>(projections.front()->input_value(0), concat, false, true);
            auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {-1});
            auto sizes = ov::op::v0::Constant::create(ov::element::i64, {lengths.size()}, lengths);
            auto split = std::make_shared<ov::op::v1::VariadicSplit>(merged, axis, sizes);
            ov::copy_runtime_info(ov::NodeVector(projections.begin(), projections.end()), {concat, merged, split});
            for (size_t i = 0; i < projections.size(); ++i)
                ov::replace_node(projections[i], ov::OutputVector{split->output(i)});
        }
    }
}

bool collect_projection_weights(const std::shared_ptr<ov::Node>& node, Plans& plans, ov::OutputVector& weights) {
    if (const auto plan = plan_weight(node, plans)) {
        if (plan->shape.size() != 2)
            return false;
        weights.push_back(node);
        return true;
    }
    const auto concat = ov::as_type_ptr<ov::op::v0::Concat>(node);
    if (!concat || concat->get_axis() != 0)
        return false;
    for (const auto& input : concat->input_values())
        if (!collect_projection_weights(input.get_node_shared_ptr(), plans, weights))
            return false;
    return true;
}

void separate_incompatible_projections(const std::shared_ptr<ov::Model>& model) {
    Plans plans;
    for (const auto& node : model->get_ordered_ops()) {
        const auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(node);
        if (!matmul || matmul->get_transpose_a() || !matmul->get_transpose_b())
            continue;
        const auto concat = ov::as_type_ptr<ov::op::v0::Concat>(matmul->get_input_node_shared_ptr(1));
        if (!concat || concat->get_axis() != 0 || plan_weight(concat, plans))
            continue;
        ov::OutputVector weights;
        if (!collect_projection_weights(concat, plans, weights))
            continue;
        ov::OutputVector outputs;
        for (const auto& weight : weights) {
            auto projection = std::make_shared<ov::op::v0::MatMul>(matmul->input_value(0), weight, false, true);
            ov::copy_runtime_info(matmul, projection);
            outputs.push_back(projection);
        }
        auto replacement = std::make_shared<ov::op::v0::Concat>(outputs, -1);
        replacement->set_friendly_name(matmul->get_friendly_name());
        ov::copy_runtime_info(matmul, replacement);
        ov::replace_node(matmul, replacement);
    }
}

}  // namespace

bool MaterializeGGUFWeights::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    for (const auto& node : model->get_ordered_ops())
        if (const auto subgraph = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node))
            for (const auto& body : subgraph->get_functions())
                changed |= run_on_model(body);
    separate_incompatible_projections(model);
    fuse_projections(model);
    Plans plans;
    const auto nodes = model->get_ordered_ops();
    std::vector<std::pair<std::shared_ptr<ov::Node>, WeightPlan>> roots;
    for (const auto& node : nodes) {
        if (ov::is_type<GGUFConstant>(node))
            continue;
        const auto plan = plan_weight(node, plans);
        if (!plan)
            continue;
        bool root = false;
        for (const auto& consumer : node->output(0).get_target_inputs())
            root |= !plan_weight(consumer.get_node()->shared_from_this(), plans).has_value();
        if (root)
            roots.emplace_back(node, *plan);
    }
    for (const auto& [node, plan] : roots) {
        auto materialized = materialize_plan(plan);
        OPENVINO_ASSERT(materialized->get_output_shape(0) == node->get_output_shape(0) &&
                            materialized->get_output_element_type(0) == node->get_output_element_type(0),
                        "Folded GGUF weight has unexpected shape or type");
        materialized->set_friendly_name(node->get_friendly_name());
        ov::copy_runtime_info(node, materialized);
        ov::replace_node(node, materialized);
        changed = true;
    }
    for (const auto& node : model->get_ordered_ops())
        if (const auto weight = ov::as_type_ptr<GGUFConstant>(node)) {
            auto materialized = weight->materialize();
            OPENVINO_ASSERT(materialized->get_output_partial_shape(0) == weight->get_output_partial_shape(0) &&
                                materialized->get_output_element_type(0) == weight->get_output_element_type(0),
                            "Materialized GGUF weight has unexpected shape or type: ",
                            weight->source().source_name);
            materialized->set_friendly_name(weight->get_friendly_name());
            ov::copy_runtime_info(weight, materialized);
            ov::replace_node(weight, materialized);
            changed = true;
        }
    return changed;
}

}  // namespace ov::frontend::gguf::pass
