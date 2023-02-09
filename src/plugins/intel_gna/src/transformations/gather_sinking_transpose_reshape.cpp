// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

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

bool IsFlatUnflat(const Shape& shape1, const Shape& shape2) {
    const Shape *max_shape = nullptr, *min_shape = nullptr;
    if (shape1.size() >= shape2.size()) {
        max_shape = &shape1;
        min_shape = &shape2;
    } else {
        max_shape = &shape2;
        min_shape = &shape1;
    }

    size_t flat_part_mult = 1;
    for (size_t i = 0; i < max_shape->size(); ++i) {
        if (i < min_shape->size()) {
            if ((*max_shape)[i] != (*min_shape)[i])
                return false;
        } else {
            flat_part_mult *= (*max_shape)[i];
        }
    }

    return min_shape->back() == flat_part_mult;
}

bool IsUnflattenReshape(const Output<Node>& output) {
    const Shape& out_shape = output.get_shape();
    const Shape& input_shape = output.get_node()->get_input_shape(0);
    return (out_shape.size() > input_shape.size() && IsFlatUnflat(out_shape, input_shape));
}

bool IsFlattenReshape(const Output<Node>& output) {
    const Shape& out_shape = output.get_shape();
    const Shape& input_shape = output.get_node()->get_input_shape(0);
    return (input_shape.size() > out_shape.size() && IsFlatUnflat(out_shape, input_shape));
}

std::vector<size_t> GetTransposeOrder(const std::vector<size_t>& order, const Shape& shape1, const Shape& shape2) {
    const size_t new_order_size = std::abs(static_cast<int>(shape1.size()) - static_cast<int>(shape2.size()));
    std::vector<size_t> new_order(order.begin() + (order.size() - new_order_size), order.end());

    const size_t delta = order.size() - new_order_size;
    for (size_t i = 0; i < new_order.size(); ++i) {
        if (new_order[i] < delta)
            return {};
        new_order[i] -= delta;
    }

    return new_order;
}

bool Next(Shape& coordinate, const Shape& shape) {
    for (size_t i = shape.size() - 1; i >= 0; --i) {
        if (coordinate[i] >= shape[i] - 1) {
            coordinate[i] = 0;
            continue;
        }
        ++coordinate[i];
        return true;
    }

    return false;
}

Shape TransposeShape(const Shape& coordinate, const std::vector<size_t>& transpose_order) {
    if (coordinate.size() != transpose_order.size())
        return {};

    Shape transposed(coordinate.size(), 0);
    for (size_t i = 0; i < coordinate.size(); ++i) {
        transposed[i] = coordinate[transpose_order[i]];
    }

    return transposed;
}

size_t GetIndexByCoordinate(const Shape& coordinate, const Shape& shape) {
    size_t index = 0;
    size_t dims_mult = 1;
    for (size_t i = coordinate.size() - 1; i >= 0 ; --i) {
        index += dims_mult * coordinate[i];
        dims_mult *= shape[i];
    }

    return index;
}

std::vector<int64_t> MakeGatherIndices(size_t size, const std::vector<size_t>& transpose_order, const Shape& transposed_shape) {
    std::vector<int64_t> indices;
    std::iota(indices.begin(), indices.end(), 0);

    Shape coordinate(transpose_order.size(), 0);
    do {
        Shape transposed_coordinate = TransposeShape(coordinate, transpose_order);
        size_t transposed_index = GetIndexByCoordinate(transposed_coordinate, transposed_shape);
        size_t orig_index = GetIndexByCoordinate(coordinate, transposed_shape); // FIXME: orig_shape ?
        indices[transposed_index] = indices[orig_index];
    } while (Next(coordinate, transposed_shape));

    return indices;
}

Shape FindTransposedShapePart(const Shape& first, const Shape& second) {
    const Shape *max_shape = nullptr, *min_shape = nullptr;
    if (first.size() >= second.size()) {
        max_shape = &first;
        min_shape = &second;
    } else {
        max_shape = &second;
        min_shape = &first;
    }

    return Shape(max_shape->begin() + (min_shape->size() - 1), max_shape->end());
}

NodePair SinkForward(NodePtr transpose, std::shared_ptr<Constant> transpose_constant, NodePtr reshape) {
    const Shape& pattern_input_shape = transpose->get_input_shape(0);
    const Shape& pattern_output_shape = reshape->get_output_shape(0);
    const int64_t gather_axis_value = pattern_output_shape.size() - 1;

    const Shape transposed_shape_part = FindTransposedShapePart(pattern_input_shape, pattern_output_shape);

    const std::vector<size_t> transpose_order = GetTransposeOrder(transpose_constant->cast_vector<size_t>(),
                                                                  pattern_input_shape,
                                                                  pattern_output_shape);
    const std::vector<int64_t> gather_indices_value = MakeGatherIndices(pattern_output_shape.back(), transpose_order, transposed_shape_part);

    auto reshape_new = reshape->clone_with_new_inputs({transpose->input_value(0), reshape->input_value(1)});

    auto gather_axis = std::make_shared<Constant>(element::i64, Shape{}, gather_axis_value);
    auto gather_indices = std::make_shared<Constant>(element::i64, Shape{1}, gather_indices_value);
    auto gather = std::make_shared<Gather>(reshape_new, gather_indices, gather_axis);

    return std::make_pair(reshape_new, gather);
}

NodePair SinkBackward(NodePtr transpose, std::shared_ptr<Constant> transpose_constant, NodePtr reshape) {
    const Shape& pattern_input_shape = transpose->get_input_shape(0);
    const Shape& pattern_output_shape = reshape->get_output_shape(0);
    const int64_t gather_axis_value = pattern_output_shape.size() - 1;

    const Shape transposed_shape_part = FindTransposedShapePart(pattern_input_shape, pattern_output_shape);

    const std::vector<size_t> transpose_order = GetTransposeOrder(transpose_constant->cast_vector<size_t>(),
                                                                  pattern_input_shape,
                                                                  pattern_output_shape);
    const std::vector<int64_t> gather_indices_value = MakeGatherIndices(pattern_output_shape.back(), transpose_order, transposed_shape_part);

    auto gather_axis = std::make_shared<Constant>(element::i64, Shape{}, gather_axis_value);
    auto gather_indices = std::make_shared<Constant>(element::i64, Shape{1}, gather_indices_value);
    auto gather = std::make_shared<Gather>(reshape->input_value(0), gather_indices, gather_axis);

    auto reshape_new = reshape->clone_with_new_inputs({gather, reshape->input_value(1)});

    return std::make_pair(reshape_new, gather);
}

} // namespace

// working with situation when we transpose dims that are flatten/unflatten
// consider only if flatten/unflatten are last dimensions
GatherSinkingTransposeReshapeForward::GatherSinkingTransposeReshapeForward() {
    MATCHER_SCOPE(GatherSinkingTransposeReshapeForward);

    auto transpose_constant_label = wrap_type<Constant>();
    auto transpose_label = wrap_type<Transpose>({any_input(), transpose_constant_label});
    auto reshape_label = wrap_type<Reshape>({transpose_label, any_input()}, IsUnflattenReshape);

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_constant = as_type_ptr<Constant>(pattern_to_output.at(transpose_constant_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto reshape = pattern_to_output.at(reshape_label).get_node_shared_ptr();

        const NodePair new_nodes = SinkForward(transpose, transpose_constant, reshape);

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

    auto reshape_label = wrap_type<Reshape>({any_input(), any_input()}, IsFlattenReshape);
    auto transpose_constant_label = wrap_type<Constant>();
    auto transpose_label = wrap_type<Transpose>({reshape_label, transpose_constant_label});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_constant = as_type_ptr<Constant>(pattern_to_output.at(transpose_constant_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto reshape = pattern_to_output.at(reshape_label).get_node_shared_ptr();

        const NodePair new_nodes = SinkBackward(transpose, transpose_constant, reshape);
        register_new_node(new_nodes.first);
        register_new_node(new_nodes.second);

        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
