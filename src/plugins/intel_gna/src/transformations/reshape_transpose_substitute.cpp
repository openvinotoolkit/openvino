// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/reshape_transpose_substitute.hpp"

#include <openvino/cc/ngraph/itt.hpp>
#include <transformations/utils/utils.hpp>
#include <utility>

#include "common/graph_utils.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov;
using namespace ov::opset12;
using namespace ov::pass::pattern;
using namespace ov::op::util;
using namespace ov::intel_gna;
using namespace ov::intel_gna::pass;

namespace {

/**
 * @brief Checks if output_shape is just permutation of input_shape dimensions
 */
bool is_shape_permutation(const Shape& input_shape, const Shape& output_shape) {
    std::unordered_map<Shape::value_type, size_t> values_count;
    for (const auto& dim : input_shape) {
        auto it = values_count.find(dim);
        if (it != values_count.end()) {
            ++it->second;
            continue;
        }
        values_count[dim] = 1;
    }
    for (const auto& dim : output_shape) {
        auto it = values_count.find(dim);
        if (it != values_count.end()) {
            --it->second;
            if (!it->second) {
                values_count.erase(it);
            }
            continue;
        }
        return false;
    }
    return values_count.empty();
}

/**
 * @brief Checks if all shape dimensions have different values
 */
bool are_dims_unique(const Shape& shape) {
    std::unordered_set<Shape::value_type> dims;
    for (const auto& dim : shape) {
        if (dims.find(dim) != dims.end())
            return false;
        dims.insert(dim);
    }
    return true;
}

AxisVector get_unique_shapes_transpose_order(const Shape& input_shape, const Shape& output_shape) {
    std::unordered_map<Shape::value_type, size_t> input_shape_items;
    for (size_t i = 0; i < input_shape.size(); ++i) {
        input_shape_items[input_shape[i]] = i;
    }

    AxisVector order;
    order.reserve(output_shape.size());
    for (const auto& output_dim : output_shape) {
        order.push_back(input_shape_items.find(output_dim)->second);
    }

    return order;
}

/**
 * @brief Changes shape according to specified order
 */
Shape apply_permutation(const Shape& shape, const AxisVector& order) {
    Shape transposed_shape;
    transposed_shape.reserve(shape.size());
    for (const auto& position : order) {
        transposed_shape.push_back(shape[position]);
    }
    return transposed_shape;
}

AxisVector find_suitable_transpose_order(const Shape& input_shape,
                                         const Shape& output_shape,
                                         const std::vector<AxisVector>& orders) {
    for (const auto& order : orders) {
        const Shape transposed_shape = apply_permutation(input_shape, order);
        if (graph_utils::are_shapes_equal(transposed_shape, output_shape))
            return order;
    }

    return {};
}

AxisVector find_suitable_transpose_order(const Shape& input_shape, const Shape& output_shape) {
    static std::vector<AxisVector> orders_4d = {AxisVector{0, 2, 3, 1}, AxisVector{0, 3, 1, 2}};
    static std::vector<AxisVector> orders_3d = {AxisVector{1, 2, 0}, AxisVector{2, 0, 1}};

    switch (input_shape.size()) {
    case 4:
        return find_suitable_transpose_order(input_shape, output_shape, orders_4d);
    case 3:
        return find_suitable_transpose_order(input_shape, output_shape, orders_3d);
    case 2:
        return AxisVector{1, 0};
    default:
        return {};
    }
    return {};
}

/**
 * @brief Finds applicable transpose order to get output_shape from input_shape if
 * it is possible
 * Returns empty AxisVector on unsupported case
 */
AxisVector get_transpose_order(const Shape& input_shape, const Shape& output_shape) {
    if (are_dims_unique(input_shape))
        return get_unique_shapes_transpose_order(input_shape, output_shape);
    return find_suitable_transpose_order(input_shape, output_shape);
}

}  // namespace

ReshapeTransposeSubstitute::ReshapeTransposeSubstitute() {
    MATCHER_SCOPE(ReshapeTransposeSubstitute);

    auto reshape_label = wrap_type<Reshape>({any_input(), any_input()}, [](const Output<Node>& output) {
        if (!has_static_shape()(output))
            return false;
        const Node* node = output.get_node();
        return is_shape_permutation(node->get_input_shape(0), node->get_output_shape(0));
    });

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto reshape = as_type_ptr<Reshape>(pattern_to_output.at(reshape_label).get_node_shared_ptr());

        const AxisVector transpose_order =
            get_transpose_order(reshape->get_input_shape(0), reshape->get_output_shape(0));
        if (transpose_order.empty())
            return false;

        auto new_transpose_const =
            std::make_shared<Constant>(element::i64, Shape{transpose_order.size()}, transpose_order);
        auto new_transpose = std::make_shared<Transpose>(reshape->input_value(0), new_transpose_const);

        replace_node_update_name(reshape, new_transpose);

        return true;
    };

    auto m = std::make_shared<Matcher>(reshape_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
