// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>
#include <ngraph/rt_info.hpp>

#include "transformations/gather_sinking_transpose_reshape.hpp"

#include "transformations/utils/transformation_helper.hpp"

#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::intel_gna::pass;
using namespace ov;
using namespace ov::opset9;
using namespace ov::pass::pattern;

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using NodePair = std::pair<NodePtr, NodePtr>;

size_t FindEndOfSlice(const Shape& transpose_order, size_t start_idx) {
    size_t slice_end = start_idx;
    for (size_t i = start_idx + 1; i < transpose_order.size(); ++i) {
        if (transpose_order[i] != transpose_order[slice_end] + 1)
            break;
        slice_end = i;
    }
    return slice_end;
}

std::vector<int64_t> CreateGatherIndices(const Shape& transpose_input_shape,
                                                       const Shape& reshape_output_shape,
                                                       const Shape& transpose_order) {

    const size_t slice_0_end = FindEndOfSlice(transpose_order, 0);
    const size_t slice_1_start = slice_0_end + 1;
    const size_t slice_1_end = FindEndOfSlice(transpose_order, slice_1_start);
    const size_t slice_2_start = slice_1_end + 1;

    if (slice_0_end >= transpose_input_shape.size() ||
        slice_1_start >= transpose_input_shape.size() ||
        slice_1_end >= transpose_input_shape.size() ||
        slice_2_start >= transpose_input_shape.size()) {
            return {};
        }

    const int64_t transpose_part_0 = std::accumulate(transpose_order.begin() + slice_1_start,
                                                     transpose_order.begin() + slice_1_end + 1, 1,
                                 [&transpose_input_shape](int64_t result, int64_t order_value) {
                                    return result *= transpose_input_shape[order_value];
                                 });
    const int64_t transpose_part_1 = std::accumulate(transpose_order.begin() + slice_2_start,
                                                     transpose_order.end(), 1,
                                 [&transpose_input_shape](int64_t result, int64_t order_value) {
                                    return result *= transpose_input_shape[order_value];
                                 });

    std::vector<int64_t> gather_indices_value(reshape_output_shape.back());
    for (size_t i = 0; i < gather_indices_value.size(); ++i) {
        gather_indices_value[i] = transpose_part_0 * (i % transpose_part_1) + i / transpose_part_1;
    }

    return gather_indices_value;
}

NodePair SinkForward(NodePtr transpose, std::shared_ptr<Constant> transpose_constant, NodePtr reshape) {
    const auto gather_indices_value = CreateGatherIndices(transpose->get_input_shape(0),
                                                                        reshape->get_output_shape(0),
                                                                        transpose_constant->get_axis_vector_val());
    const int64_t gather_axis_value = reshape->get_output_shape(0).size() - 1;

    auto reshape_new = reshape->clone_with_new_inputs({transpose->input_value(0), reshape->input_value(1)});

    auto gather_axis = std::make_shared<Constant>(element::i64, Shape{}, gather_axis_value);
    auto gather_indices = std::make_shared<Constant>(element::i64, Shape{gather_indices_value.size()}, gather_indices_value);
    auto gather = std::make_shared<Gather>(reshape_new, gather_indices, gather_axis);

    ov::replace_node(reshape, gather);

    ov::copy_runtime_info({reshape}, {gather, gather_indices, gather_axis, reshape_new});
    gather->set_friendly_name(reshape->get_friendly_name());

    return std::make_pair(reshape_new, gather);
}

Shape TransposeShape(const Shape& shape, AxisVector transpose_axis) {
    Shape transposed(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        transposed[i] = shape[transpose_axis[i]];
    }
    return transposed;
}

NodePair SinkBackward(NodePtr transpose, std::shared_ptr<Constant> transpose_constant, NodePtr reshape) {
    const int64_t gather_axis_value = reshape->get_input_shape(0).size() - 1;

    const auto gather_indices_value = CreateGatherIndices(TransposeShape(transpose->get_output_shape(0),
                                                                                       transpose_constant->get_axis_vector_val()),
                                                                        reshape->get_input_shape(0),
                                                                        transpose_constant->get_axis_vector_val());

    auto gather_axis = std::make_shared<Constant>(element::i64, Shape{}, gather_axis_value);
    auto gather_indices = std::make_shared<Constant>(element::i64, Shape{gather_indices_value.size()}, gather_indices_value);
    auto gather = std::make_shared<Gather>(reshape->input_value(0), gather_indices, gather_axis);

    auto reshape_const_new = std::make_shared<Constant>(element::i64, Shape{transpose->get_output_shape(0).size()}, transpose->get_output_shape(0));
    auto reshape_new = std::make_shared<Reshape>(gather, reshape_const_new, false);

    ov::replace_node(transpose, reshape_new);

    ov::copy_runtime_info({transpose}, {gather, gather_indices, gather_axis, reshape_new, reshape_const_new});
    reshape_new->set_friendly_name(transpose->get_friendly_name());

    return std::make_pair(transpose, reshape_new);
}
#if 0
bool IsFlatten2D(const Output<Node>& output) {
    std::shared_ptr<ov::Node> reshape_node = output.get_node_shared_ptr();
    if (reshape_node->get_output_partial_shape(0).rank().is_dynamic() ||
        reshape_node->get_input_partial_shape(0).rank().is_dynamic())
        return false;
    const Shape& input_shape = reshape_node->get_input_shape(0);
    const Shape& output_shape = reshape_node->get_output_shape(0);
    return (input_shape.size() == 3 &&
            output_shape.size() == 2 &&
            input_shape[0] == output_shape[0] &&
            output_shape[1] == input_shape[1] * input_shape[2]);
}

bool IsUnflatten2D(const Output<Node>& output) {
    std::shared_ptr<ov::Node> reshape_node = output.get_node_shared_ptr();
    if (reshape_node->get_output_partial_shape(0).rank().is_dynamic() ||
        reshape_node->get_input_partial_shape(0).rank().is_dynamic())
        return false;
    const Shape& input_shape = reshape_node->get_input_shape(0);
    const Shape& output_shape = reshape_node->get_output_shape(0);
    return (input_shape.size() == 2 &&
            output_shape.size() == 3 &&
            output_shape[0] == input_shape[0] &&
            input_shape[1] == output_shape[1] * output_shape[2]);
}
#endif
} // namespace

// working with situation when we transpose dims that are flatten/unflatten
// consider only if flatten/unflatten are last dimensions
GatherSinkingTransposeReshapeForward::GatherSinkingTransposeReshapeForward() {
    MATCHER_SCOPE(GatherSinkingTransposeReshapeForward);

    auto transpose_const_label = wrap_type<Constant>();
    auto transpose_label = wrap_type<Transpose>({any_input(), transpose_const_label});
    auto reshape_label = wrap_type<Reshape>({transpose_label, any_input()}/*, IsFlatten2D*/); // TODO

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto transpose_const = as_type_ptr<Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto reshape = pattern_to_output.at(reshape_label).get_node_shared_ptr();

        const NodePair new_nodes = SinkForward(transpose, transpose_const, reshape);

        register_new_node(new_nodes.first);
        register_new_node(new_nodes.second);

        // UpdateForwardSinkingAbility(new_nodes.second); TODO
        return true;
    };

    auto m = std::make_shared<Matcher>(reshape_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

GatherSinkingTransposeReshapeBackward::GatherSinkingTransposeReshapeBackward() {
    MATCHER_SCOPE(GatherSinkingTransposeReshapeBackward);

    auto reshape_label = wrap_type<Reshape>({any_input(), any_input()}/*, IsUnflatten2D*//*check if it is sinkable */); // TODO: IsUnflatten2D
    auto transpose_const_label = wrap_type<Constant>();
    auto transpose_label = wrap_type<Transpose>({reshape_label, transpose_const_label});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto transpose_const = as_type_ptr<Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto reshape = pattern_to_output.at(reshape_label).get_node_shared_ptr();

        const NodePair new_nodes = SinkBackward(transpose, transpose_const, reshape);
        register_new_node(new_nodes.first);
        register_new_node(new_nodes.second);

        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
