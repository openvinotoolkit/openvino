// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/utils/utils.hpp"

#include <assert.h>

#include <functional>
#include <memory>
#include <ngraph/op/broadcast.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/reshape.hpp>
#include <ngraph/op/gather.hpp>
#include <ngraph/op/util/op_annotations.hpp>

namespace ngraph {
namespace op {
namespace util {

bool get_single_value(const std::shared_ptr<op::Constant>& const_node, float& value) {
    switch (const_node->get_element_type()) {
    case element::Type_t::f16:
        return util::normalize_single_value(const_node->get_vector<float16>(), value);
    case element::Type_t::f32:
        return util::normalize_single_value(const_node->get_vector<float>(), value);
    case element::Type_t::bf16:
        return util::normalize_single_value(const_node->get_vector<bfloat16>(), value);
    case element::Type_t::f64:
        return util::normalize_single_value(const_node->get_vector<double>(), value);
    case element::Type_t::i8:
        return util::normalize_single_value(const_node->get_vector<int8_t>(), value);
    case element::Type_t::i16:
        return util::normalize_single_value(const_node->get_vector<int16_t>(), value);
    case element::Type_t::i32:
        return util::normalize_single_value(const_node->get_vector<int32_t>(), value);
    case element::Type_t::i64:
        return util::normalize_single_value(const_node->get_vector<int64_t>(), value);
    case element::Type_t::u8:
        return util::normalize_single_value(const_node->get_vector<uint8_t>(), value);
    case element::Type_t::u16:
        return util::normalize_single_value(const_node->get_vector<uint16_t>(), value);
    case element::Type_t::u32:
        return util::normalize_single_value(const_node->get_vector<uint32_t>(), value);
    case element::Type_t::u64:
        return util::normalize_single_value(const_node->get_vector<uint64_t>(), value);
    default:
        throw ngraph_error("Unsupported precision for const operation: " + const_node->get_friendly_name());
    }
}

std::shared_ptr<Node> normalize_constant(const std::shared_ptr<op::Constant>& constant,
                                         const PartialShape& shape) {
    auto const_shape = constant->get_shape();
    if (static_cast<int64_t>(const_shape.size()) == shape.rank().get_length()) {
        return constant;
    }
    int64_t cnt = shape.rank().get_length() - const_shape.size();
    for (int i = 0; i < cnt; ++i) {
        const_shape.insert(const_shape.begin(), 1);
    }

    return reshapeTo(constant, const_shape);
}

std::shared_ptr<Node> broadcastTo(const Output<Node>& input, const ngraph::Shape& shape) {
    return std::make_shared<op::v1::Broadcast>(input, op::Constant::create(ngraph::element::i64, Shape {shape.size()}, shape));
}

std::shared_ptr<ngraph::Node> reshapeTo(const Output<Node> & input, const Shape& shape) {
    return std::make_shared<op::v1::Reshape>(input, op::Constant::create(element::i64, Shape{shape.size()}, shape), true);
}

bool constantIsEqualTo(const std::shared_ptr<ngraph::op::Constant>& const_node, float value, float eps) {
    float res(0);
    if (!get_single_value(const_node, res)) {
        return false;
    }

    return std::abs(res - value) < eps;
}

bool has_f16_constants(const std::shared_ptr<const ngraph::Function> &function) {
    for (auto & layer : function->get_ops()) {
        if (std::dynamic_pointer_cast<ngraph::op::Constant>(layer) && layer->output(0).get_element_type() == ngraph::element::f16) {
            return true;
        }
    }
    return false;
}

bool check_for_broadcast(const ngraph::Shape &ref_shape, const ngraph::Shape &other_shape) {
    // Check that other_shape doesn't broadcast ref_shape
    if (other_shape.size() > ref_shape.size()) {
        return true;
    }
    auto ref_it = ref_shape.rbegin();
    auto other_it = other_shape.rbegin();
    // Check that other_shape dims are equal to ref_shape dims
    // In case if other_shape rank is less than ref_shape rank
    // we stop comparision and return true
    while (other_it != other_shape.rend()) {
        if (*other_it != *ref_it && *other_it != 1) {
            return true;
        }
        ++other_it;
        ++ref_it;
    }
    return false;
}

std::shared_ptr<ngraph::Node> activation(const std::string& activation_name, const ngraph::Output<ngraph::Node>& apply_to) {
    if (activation_name == "relu") {
        return std::make_shared<ngraph::opset4::Relu>(apply_to);
    } else if (activation_name == "sigmoid") {
        return std::make_shared<ngraph::opset4::Sigmoid>(apply_to);
    } else if (activation_name == "tanh") {
        return std::make_shared<ngraph::opset4::Tanh>(apply_to);
    } else {
        throw ngraph_error("Unsupported activation function");
    }
}

bool is_seq_len_provided(const std::shared_ptr<Node> &seq_len_input, int64_t max_seq_len) {
    if (const auto &seq_len_const = std::dynamic_pointer_cast<ngraph::op::Constant>(seq_len_input)) {
        const auto &seq_len_values = seq_len_const->cast_vector<int64_t>();
        return std::any_of(seq_len_values.begin(), seq_len_values.end(), [max_seq_len](const int64_t val) {
            return val != max_seq_len;
        });
    }
    return true;
}

std::shared_ptr<Node> try_fold_unary_output(const std::shared_ptr<Node>& node) {
    const auto& num_outputs = node->get_output_size();
    NGRAPH_CHECK(num_outputs == 1, "Unary has unexpected number of outputs:" + std::to_string(num_outputs));
    OutputVector output(num_outputs);
    return node->constant_fold(output, node->input_values()) ? output[0].get_node_shared_ptr() : node;
}

std::shared_ptr<Node> clone_try_fold(const std::shared_ptr<Node>& node, const OutputVector& inputs) {
    auto unary_output_node = node->clone_with_new_inputs(inputs);
    return try_fold_unary_output(unary_output_node);
}

std::vector<Input<Node>> get_node_target_inputs(const std::shared_ptr<Node>& node) {
    std::vector<Input<Node>> result;
    for (auto output : node->outputs()) {
        for (auto input : output.get_target_inputs()) {
            result.push_back(input);
        }
    }
    return result;
}

std::shared_ptr<ngraph::Node> node_to_get_shape_value_of_indices_from_shape_node(const std::shared_ptr<ngraph::Node>& shape_node,
                                                                                 const std::vector<size_t>& indices) {
    return make_try_fold<v7::Gather>(
            shape_node,
            v0::Constant::create(ngraph::element::i64, {indices.size()}, indices),
            v0::Constant::create(ngraph::element::i64, {}, {0}));
}

std::shared_ptr<ngraph::Node> node_to_get_shape_value_of_indices_from_shape_source(const ngraph::Output<ngraph::Node>& shape_source,
    const std::vector<size_t>& indices) {
    const auto& shape_node = make_try_fold<v3::ShapeOf>(shape_source);
    return node_to_get_shape_value_of_indices_from_shape_node(shape_node, indices);
}

bool shapes_equal_except_dynamic_expected_batch(const ngraph::PartialShape& expected, const ngraph::PartialShape& actual) {
    if (expected[0].is_static()) {
        return actual == expected;
    } else {
        auto actual_with_dynamic_batch = actual;
        actual_with_dynamic_batch[0] = expected[0];
        return actual_with_dynamic_batch == expected;
    }
}

}  // namespace util
}  // namespace op
}  // namespace ngraph
