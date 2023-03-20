// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_transpose_reshape.hpp"

#include <ngraph/rt_info.hpp>
#include <openvino/cc/ngraph/itt.hpp>

#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/gather_sinking_utils.hpp"
#include "transformations/utils/transformation_helper.hpp"

using namespace ov::intel_gna::pass;
using namespace ov;
using namespace ov::opset9;
using namespace ov::pass::pattern;
using namespace gather_sinking;

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using NodePair = std::pair<NodePtr, NodePtr>;

/*
 * Finds the end of the slice transpose order. Slice here is the subvector
 * with increasing values transpose_order[i + 1] == transpose_order[i] + 1
 */
size_t FindEndOfSlice(const Shape& transpose_order, size_t start_idx) {
    size_t slice_end = start_idx;
    for (size_t i = start_idx + 1; i < transpose_order.size(); ++i) {
        if (transpose_order[i] != transpose_order[slice_end] + 1)
            break;
        slice_end = i;
    }
    return slice_end;
}

size_t GetSliceNum(const Shape& transpose_order) {
    size_t slice_count = 0;
    for (size_t i = 0; i < transpose_order.size(); ++i) {
        if (!i || transpose_order[i] != transpose_order[i - 1] + 1) {
            ++slice_count;
        }
    }
    return slice_count;
}

std::vector<int64_t> CreateGatherIndices(const Shape& transpose_input_shape,
                                         const Shape& reshape_output_shape,
                                         const Shape& transpose_order) {
    const size_t slice_0_end = FindEndOfSlice(transpose_order, 0);
    const size_t slice_1_start = slice_0_end + 1;
    const size_t slice_1_end = FindEndOfSlice(transpose_order, slice_1_start);
    const size_t slice_2_start = slice_1_end + 1;

    if (slice_0_end >= transpose_input_shape.size() || slice_1_start >= transpose_input_shape.size() ||
        slice_1_end >= transpose_input_shape.size() || slice_2_start >= transpose_input_shape.size()) {
        return {};
    }

    const int64_t transpose_part_0 = std::accumulate(transpose_order.begin() + slice_1_start,
                                                     transpose_order.begin() + slice_1_end + 1,
                                                     1,
                                                     [&transpose_input_shape](int64_t result, int64_t order_value) {
                                                         return result *= transpose_input_shape[order_value];
                                                     });
    const int64_t transpose_part_1 = std::accumulate(transpose_order.begin() + slice_2_start,
                                                     transpose_order.end(),
                                                     1,
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
    auto gather_indices =
        std::make_shared<Constant>(element::i64, Shape{gather_indices_value.size()}, gather_indices_value);
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

    const auto gather_indices_value =
        CreateGatherIndices(TransposeShape(transpose->get_output_shape(0), transpose_constant->get_axis_vector_val()),
                            reshape->get_input_shape(0),
                            transpose_constant->get_axis_vector_val());

    auto gather_axis = std::make_shared<Constant>(element::i64, Shape{}, gather_axis_value);
    auto gather_indices =
        std::make_shared<Constant>(element::i64, Shape{gather_indices_value.size()}, gather_indices_value);
    auto gather = std::make_shared<Gather>(reshape->input_value(0), gather_indices, gather_axis);

    auto reshape_const_new = std::make_shared<Constant>(element::i64,
                                                        Shape{transpose->get_output_shape(0).size()},
                                                        transpose->get_output_shape(0));
    auto reshape_new = std::make_shared<Reshape>(gather, reshape_const_new, false);

    ov::replace_node(transpose, reshape_new);

    ov::copy_runtime_info({transpose}, {gather, gather_indices, gather_axis, reshape_new, reshape_const_new});
    reshape_new->set_friendly_name(transpose->get_friendly_name());

    return std::make_pair(transpose, reshape_new);
}

bool AreFlattenShapes(const Shape& shape1, const Shape& shape2) {
    size_t i = 0;
    // find non-equal parts
    while (shape1[i] == shape2[i]) {
        ++i;
    }
    // consider only last dimension to be flatten/unflatten
    if (shape1.size() - 1 != i && shape2.size() - 1 != i)
        return false;
    // min_shape.back() == MULTIPLY(max_shape.begin() + i, max_shape.end())
    const size_t mult1 = std::accumulate(shape1.begin() + i, shape1.end(), 1, std::multiplies<size_t>());
    const size_t mult2 = std::accumulate(shape2.begin() + i, shape2.end(), 1, std::multiplies<size_t>());
    return mult1 == mult2;
}

bool Is2DTransposeConstant(const Output<Node>& output) {
    std::shared_ptr<Constant> transpose_constant = as_type_ptr<Constant>(output.get_node_shared_ptr());
    if (!transpose_constant)
        return false;
    return GetSliceNum(transpose_constant->get_axis_vector_val()) == 3;
}

bool IsTailFlatten(const Output<Node>& output) {
    std::shared_ptr<ov::Node> reshape_node = output.get_node_shared_ptr();
    if (reshape_node->get_output_partial_shape(0).rank().is_dynamic() ||
        reshape_node->get_input_partial_shape(0).rank().is_dynamic())
        return false;
    const Shape& input_shape = helper::SqueezeShape(reshape_node->get_input_shape(0));
    const Shape& output_shape = helper::SqueezeShape(reshape_node->get_output_shape(0));
    return output_shape.size() < input_shape.size() && AreFlattenShapes(input_shape, output_shape);
}

bool IsTailUnflatten(const Output<Node>& output) {
    std::shared_ptr<ov::Node> reshape_node = output.get_node_shared_ptr();
    if (reshape_node->get_output_partial_shape(0).rank().is_dynamic() ||
        reshape_node->get_input_partial_shape(0).rank().is_dynamic())
        return false;
    const Shape& input_shape = helper::SqueezeShape(reshape_node->get_input_shape(0));
    const Shape& output_shape = helper::SqueezeShape(reshape_node->get_output_shape(0));
    return output_shape.size() > input_shape.size() && AreFlattenShapes(input_shape, output_shape);
}
}  // namespace

// working with situation when we transpose dims that are flatten/unflatten
// consider only if flatten/unflatten are last dimensions
GatherSinkingTransposeReshapeForward::GatherSinkingTransposeReshapeForward() {
    MATCHER_SCOPE(GatherSinkingTransposeReshapeForward);

    auto transpose_const_label = wrap_type<Constant>(Is2DTransposeConstant);
    auto transpose_label = wrap_type<Transpose>({any_input(), transpose_const_label});
    auto reshape_label = wrap_type<Reshape>({transpose_label, any_input()}, IsTailFlatten);

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto transpose_const = as_type_ptr<Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto reshape = pattern_to_output.at(reshape_label).get_node_shared_ptr();

        const ov::Shape reshape_shape = pass::helper::SqueezeShape(reshape->get_shape());
        const ov::Shape transpose_shape = pass::helper::SqueezeShape(transpose->get_shape());
        if (reshape_shape == transpose_shape) {
            pass::helper::RemoveSingleInputNodeFromFunction(transpose);
            return true;
        }

        const NodePair new_nodes = SinkForward(transpose, transpose_const, reshape);

        register_new_node(new_nodes.first);
        register_new_node(new_nodes.second);

        UpdateForwardGatherSinkingAbility(new_nodes.second);
        return true;
    };

    auto m = std::make_shared<Matcher>(reshape_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

GatherSinkingTransposeReshapeBackward::GatherSinkingTransposeReshapeBackward() {
    MATCHER_SCOPE(GatherSinkingTransposeReshapeBackward);

    auto reshape_label = wrap_type<Reshape>({any_input(), any_input()}, IsTailUnflatten);
    auto transpose_const_label = wrap_type<Constant>(Is2DTransposeConstant);
    auto transpose_label = wrap_type<Transpose>({reshape_label, transpose_const_label});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto transpose_const = as_type_ptr<Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto reshape = pattern_to_output.at(reshape_label).get_node_shared_ptr();

        const ov::Shape reshape_shape = pass::helper::SqueezeShape(reshape->get_shape());
        const ov::Shape transpose_shape = pass::helper::SqueezeShape(transpose->get_shape());
        if (reshape_shape == transpose_shape) {
            pass::helper::RemoveSingleInputNodeFromFunction(transpose);
            return true;
        }

        const NodePair new_nodes = SinkBackward(transpose, transpose_const, reshape);
        register_new_node(new_nodes.first);
        register_new_node(new_nodes.second);

        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
