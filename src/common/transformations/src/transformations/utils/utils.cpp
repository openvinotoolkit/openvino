// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/utils/utils.hpp"

#include <assert.h>

#include <functional>
#include <memory>

#include "openvino/core/validation_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"

namespace ov {
namespace op {
namespace util {

bool get_single_value(const std::shared_ptr<op::v0::Constant>& const_node, float& value, bool check_value_range) {
    switch (const_node->get_element_type()) {
    case element::Type_t::f16:
        return util::normalize_single_value(const_node->get_vector<float16>(), value, check_value_range);
    case element::Type_t::f32:
        return util::normalize_single_value(const_node->get_vector<float>(), value, check_value_range);
    case element::Type_t::bf16:
        return util::normalize_single_value(const_node->get_vector<bfloat16>(), value, check_value_range);
    case element::Type_t::f64:
        return util::normalize_single_value(const_node->get_vector<double>(), value, check_value_range);
    case element::Type_t::i4:
        return util::normalize_single_value(const_node->cast_vector<int8_t>(), value, check_value_range);
    case element::Type_t::i8:
        return util::normalize_single_value(const_node->get_vector<int8_t>(), value, check_value_range);
    case element::Type_t::i16:
        return util::normalize_single_value(const_node->get_vector<int16_t>(), value, check_value_range);
    case element::Type_t::i32:
        return util::normalize_single_value(const_node->get_vector<int32_t>(), value, check_value_range);
    case element::Type_t::i64:
        return util::normalize_single_value(const_node->get_vector<int64_t>(), value, check_value_range);
    case element::Type_t::u4:
        return util::normalize_single_value(const_node->cast_vector<int8_t>(), value, check_value_range);
    case element::Type_t::u8:
        return util::normalize_single_value(const_node->get_vector<uint8_t>(), value, check_value_range);
    case element::Type_t::u16:
        return util::normalize_single_value(const_node->get_vector<uint16_t>(), value, check_value_range);
    case element::Type_t::u32:
        return util::normalize_single_value(const_node->get_vector<uint32_t>(), value, check_value_range);
    case element::Type_t::u64:
        return util::normalize_single_value(const_node->get_vector<uint64_t>(), value, check_value_range);
    default:
        OPENVINO_THROW("Unsupported precision for const operation: ", const_node->get_friendly_name());
    }
}

std::shared_ptr<Node> normalize_constant(const std::shared_ptr<op::v0::Constant>& constant, const PartialShape& shape) {
    auto const_shape = constant->get_output_partial_shape(0).to_shape();
    if (static_cast<int64_t>(const_shape.size()) == shape.rank().get_length()) {
        return constant;
    }
    int64_t cnt = shape.rank().get_length() - const_shape.size();
    for (int i = 0; i < cnt; ++i) {
        const_shape.insert(const_shape.begin(), 1);
    }

    return reshapeTo(constant, const_shape);
}

std::shared_ptr<Node> broadcastTo(const Output<Node>& input, const ov::Shape& shape) {
    return std::make_shared<op::v1::Broadcast>(input,
                                               op::v0::Constant::create(ov::element::i64, Shape{shape.size()}, shape));
}

std::shared_ptr<ov::Node> reshapeTo(const Output<Node>& input, const Shape& shape) {
    return std::make_shared<op::v1::Reshape>(input,
                                             op::v0::Constant::create(element::i64, Shape{shape.size()}, shape),
                                             true);
}

bool constantIsEqualTo(const std::shared_ptr<op::v0::Constant>& const_node, float value, float eps) {
    float res(0);
    if (!get_single_value(const_node, res)) {
        return false;
    }

    return std::abs(res - value) < eps;
}

bool has_f16_constants(const std::shared_ptr<const ov::Model>& function) {
    for (auto& layer : function->get_ops()) {
        if (std::dynamic_pointer_cast<op::v0::Constant>(layer) &&
            layer->output(0).get_element_type() == ov::element::f16) {
            return true;
        }
    }
    return false;
}

bool check_for_broadcast(const ov::PartialShape& ref_shape, const ov::PartialShape& other_shape) {
    // Check that other_shape doesn't broadcast ref_shape
    if (ref_shape.rank().is_dynamic() || other_shape.rank().is_dynamic() || other_shape.size() > ref_shape.size()) {
        return true;
    }
    auto ref_it = ref_shape.rbegin();
    auto other_it = other_shape.rbegin();
    // Check that other_shape dims are equal to ref_shape dims
    // In case if other_shape rank is less than ref_shape rank
    // we stop comparison and return true
    while (other_it != other_shape.rend()) {
        if ((other_it->is_dynamic() || other_it->get_length() != 1) &&
            (ref_it->is_dynamic() || ref_it->get_length() == 1)) {
            return true;
        }
        ++other_it;
        ++ref_it;
    }
    return false;
}

std::shared_ptr<ov::Node> activation(const std::string& activation_name, const ov::Output<ov::Node>& apply_to) {
    if (activation_name == "relu") {
        return std::make_shared<opset4::Relu>(apply_to);
    } else if (activation_name == "sigmoid") {
        return std::make_shared<opset4::Sigmoid>(apply_to);
    } else if (activation_name == "tanh") {
        return std::make_shared<opset4::Tanh>(apply_to);
    } else {
        OPENVINO_THROW("Unsupported activation function");
    }
}

bool is_seq_len_provided(const std::shared_ptr<Node>& seq_len_input, int64_t max_seq_len) {
    if (const auto& seq_len_const = std::dynamic_pointer_cast<op::v0::Constant>(seq_len_input)) {
        const auto& seq_len_values = seq_len_const->cast_vector<int64_t>();
        return std::any_of(seq_len_values.begin(), seq_len_values.end(), [max_seq_len](const int64_t val) {
            return val != max_seq_len;
        });
    }
    return true;
}

std::shared_ptr<Node> try_fold_unary_output(const std::shared_ptr<Node>& node) {
    const auto& num_outputs = node->get_output_size();
    OPENVINO_ASSERT(num_outputs == 1, "Unary has unexpected number of outputs:" + std::to_string(num_outputs));
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

std::shared_ptr<ov::Node> node_to_get_shape_value_of_indices_from_shape_node(
    const std::shared_ptr<ov::Node>& shape_node,
    const std::vector<size_t>& indices,
    const std::vector<std::shared_ptr<ov::Node>>& copy_rt_info_from) {
    const auto& indices_op = v0::Constant::create(ov::element::i64, {indices.size()}, indices);
    const auto& axis_op = v0::Constant::create(ov::element::i64, {}, {0});
    auto op = make_try_fold<v7::Gather>(shape_node, indices_op, axis_op);
    if (!copy_rt_info_from.empty())
        ov::copy_runtime_info(copy_rt_info_from, {op, indices_op, axis_op});
    return op;
}

std::shared_ptr<ov::Node> node_to_get_shape_value_of_indices_from_shape_source(
    const ov::Output<ov::Node>& shape_source,
    const std::vector<size_t>& indices,
    const std::vector<std::shared_ptr<ov::Node>>& copy_rt_info_from) {
    const auto& shape_node = make_try_fold<v3::ShapeOf>(shape_source);
    if (!copy_rt_info_from.empty())
        ov::copy_runtime_info(copy_rt_info_from, shape_node);
    return node_to_get_shape_value_of_indices_from_shape_node(shape_node, indices, copy_rt_info_from);
}

bool shapes_equal_except_dynamic_expected_batch(const ov::PartialShape& expected, const ov::PartialShape& actual) {
    if (expected[0].is_static()) {
        return actual == expected;
    } else {
        auto actual_with_dynamic_batch = actual;
        actual_with_dynamic_batch[0] = expected[0];
        return actual_with_dynamic_batch == expected;
    }
}

void visit_shape_path(Node* node, std::unordered_set<ov::Node*>& visited, std::function<void(ov::Node*)> func) {
    if (!node)
        return;
    visited.insert(node);
    std::deque<ov::Node*> nodes{node};
    while (!nodes.empty()) {
        auto curr_node = nodes.front();
        nodes.pop_front();
        // Do not check if already visited
        if (ov::is_type<opset1::ShapeOf>(curr_node) || ov::is_type<opset3::ShapeOf>(curr_node)) {
            continue;
        }

        func(curr_node);
        for (auto& input_value : curr_node->input_values()) {
            // continue searching
            const auto& input_node = input_value.get_node();
            if (visited.count(input_node))
                continue;
            nodes.push_front(input_node);
            visited.insert(input_node);
        }
    }
}

bool is_dequantization_subgraph(const Output<Node>& node) {
    if (!is_type<opset8::Multiply>(node.get_node())) {
        return false;
    }

    auto mul_inputs = node.get_node()->input_values();
    Node* sub = nullptr;
    Node* convert = nullptr;

    if (is_type<opset8::Subtract>(mul_inputs[0].get_node())) {
        sub = mul_inputs[0].get_node();
    } else if (is_type<opset8::Convert>(mul_inputs[0].get_node())) {
        convert = mul_inputs[0].get_node();
    } else {
        return false;
    }

    if (sub) {
        auto sub_inputs = sub->input_values();
        if (is_type<opset8::Convert>(sub_inputs[0].get_node())) {
            convert = sub_inputs[0].get_node();
        }
    }

    if (!convert) {
        return false;
    }

    auto input_type = convert->get_input_element_type(0);
    auto output_type = convert->get_output_element_type(0);
    return input_type.is_integral() && output_type.is_real();
}

bool can_eliminate_eltwise_node(const std::shared_ptr<Node>& eltwise,
                                const Output<Node>& constant,
                                const Output<Node>& non_constant_input) {
    if (!is_type<opset8::Add>(eltwise) && !is_type<opset8::Subtract>(eltwise) && !is_type<opset8::Multiply>(eltwise) &&
        !is_type<opset8::Divide>(eltwise)) {
        return false;
    }

    if (is_dequantization_subgraph(eltwise)) {
        return false;
    }

    // check if constant has a single value with either 0 (for Add, Subtract) or 1 (for Multiply, Divide)
    auto constant_ptr = std::dynamic_pointer_cast<opset8::Constant>(constant.get_node_shared_ptr());
    if (!constant_ptr) {
        return false;
    }
    if (!constant_ptr->get_all_data_elements_bitwise_identical()) {
        return false;
    }
    float actual_const = 0;
    const void* data_ptr = constant_ptr->get_data_ptr();
    switch (constant_ptr->get_element_type()) {
    case element::f32:
        actual_const = reinterpret_cast<const float*>(data_ptr)[0];
        break;
    case element::f16:
        actual_const = reinterpret_cast<const ov::float16*>(data_ptr)[0];
        break;
    case element::i32:
        actual_const = static_cast<float>(reinterpret_cast<const int32_t*>(data_ptr)[0]);
        break;
    case element::u32:
        actual_const = static_cast<float>(reinterpret_cast<const uint32_t*>(data_ptr)[0]);
        break;
    case element::i64:
        actual_const = static_cast<float>(reinterpret_cast<const int64_t*>(data_ptr)[0]);
        break;
    case element::u64:
        actual_const = static_cast<float>(reinterpret_cast<const uint64_t*>(data_ptr)[0]);
        break;
    case element::i8:
        actual_const = static_cast<float>(reinterpret_cast<const int8_t*>(data_ptr)[0]);
        break;
    case element::u8:
        actual_const = static_cast<float>(reinterpret_cast<const uint8_t*>(data_ptr)[0]);
        break;
    case element::i16:
        actual_const = static_cast<float>(reinterpret_cast<const int16_t*>(data_ptr)[0]);
        break;
    case element::u16:
        actual_const = static_cast<float>(reinterpret_cast<const uint16_t*>(data_ptr)[0]);
        break;
    case element::f64:
        actual_const = static_cast<float>(reinterpret_cast<const double*>(data_ptr)[0]);
        break;
    default:
        return false;
    }
    float expected_const = 0;
    if (is_type<opset8::Multiply>(eltwise) || is_type<opset8::Divide>(eltwise)) {
        expected_const = 1;
    }
    if (actual_const != expected_const) {
        return false;
    }

    // fuse uncoditionally if constant is a scalar
    const auto& constant_shape = constant.get_partial_shape().to_shape();
    if (ov::is_scalar(constant_shape)) {
        return true;
    }

    const auto& input_shape = non_constant_input.get_partial_shape();
    if (input_shape.rank().is_dynamic()) {
        return false;
    }

    // cannot fuse if constant extends input's rank
    auto input_rank = static_cast<size_t>(input_shape.rank().get_length());
    auto constant_rank = constant_shape.size();
    if (input_rank < constant_rank) {
        return false;
    }

    // cannot fuse if constant makes input to be broadcasted, e.g.
    // Multiply(input{2, 1, 5}, constant{1, 5, 1}) -> {2, 5, 5}
    for (size_t i = 0; i < constant_rank; i++) {
        auto constant_dim = constant_shape[constant_rank - i - 1];
        if (constant_dim != 1 && input_shape[input_rank - i - 1] != constant_dim) {
            return false;
        }
    }
    return true;
}

float cast_eps_to_float(double eps_d) {
    auto eps_f = static_cast<float>(eps_d);
    if (eps_d > 0.) {  // zero is fine; negative values have no sense
        if (std::nextafter(eps_d, 0) < static_cast<double>(std::numeric_limits<float>::min()))
            eps_f = std::numeric_limits<float>::min();
        else if (std::nextafter(eps_d, std::numeric_limits<double>::max()) >
                 static_cast<double>(std::numeric_limits<float>::max()))
            eps_f = std::numeric_limits<float>::max();
    }
    return eps_f;
}

bool is_constant_and_all_values_equal_int(const Output<Node>& output, const int64_t& v) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    if (const auto& constant = ov::get_constant_from_source(output)) {
        OPENVINO_SUPPRESS_DEPRECATED_END
        const auto& values = constant->cast_vector<int64_t>();
        return std::all_of(values.begin(), values.end(), [&](const int64_t& i) {
            return i == v;
        });
    }
    return false;
}

bool is_on_constant_path(const ov::Output<ov::Node>& output) {
    auto status = true;
    std::deque<ov::Node*> nodes_to_calculate = {output.get_node()};

    while (status && !nodes_to_calculate.empty()) {
        auto current_node = nodes_to_calculate.front();
        nodes_to_calculate.pop_front();

        if (current_node->get_input_size() == 0 && !ov::is_type<ov::op::v0::Constant>(current_node)) {
            status = false;
        } else {
            // not a leaf - continue to search
            for (const auto& input_value : current_node->input_values()) {
                const auto& input_node = input_value.get_node();
                nodes_to_calculate.push_front(input_node);
            }
        }
    }
    return status;
}

}  // namespace util
}  // namespace op
}  // namespace ov
