// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bound_evaluate.hpp"

#include "ngraph/validation_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset10.hpp"

namespace {
using namespace ov;

void propagate_rt_info(Node* node, const Output<Node>& final_port) {
    auto node_outputs = node->outputs();
    bool same_outputs = std::all_of(node_outputs.begin(), node_outputs.end(), [](const Output<Node>& output) {
        return output.get_tensor().has_and_set_bound();
    });
    if (same_outputs && op::util::is_constant(node))  // constant should not propagate it's rt_info
    {
        std::unordered_set<Node*> stop_nodes;
        for (const auto& in : final_port.get_target_inputs())
            stop_nodes.insert(in.get_node());

        auto curr_node = node->shared_from_this();
        for (const auto& output : node_outputs) {
            if (output == final_port)
                continue;
            for (auto& in : output.get_target_inputs()) {
                if (stop_nodes.count(in.get_node()))
                    continue;
                try {
                    auto consumer = in.get_node()->shared_from_this();
                    copy_runtime_info({curr_node, consumer}, consumer);
                } catch (const std::bad_weak_ptr&) {
                    // Exception can be thrown, if `shared_from_this()` was called during node creation.
                    // Continue propagation for other nodes.
                }
            }
        }
    }
}

bool are_same_tensor(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    return (lhs && rhs) && (lhs.get_element_type() == rhs.get_element_type()) && (lhs.get_shape() == rhs.get_shape()) &&
           (lhs.data() == rhs.data());
}

bool are_equal(const ov::Tensor& lhs, const ov::Tensor& rhs, size_t element_limit = 10) {
    if (!lhs || !rhs) {
        return false;
    }
    const auto& lhs_shape = lhs.get_shape();
    const auto& rhs_shape = rhs.get_shape();

    const auto& lhs_et = lhs.get_element_type();
    const auto& rhs_et = rhs.get_element_type();

    auto are_eq = (lhs_et == rhs_et) && (lhs_shape == rhs_shape) && shape_size(lhs_shape) <= element_limit;

    if (are_eq) {
        are_eq = memcmp(lhs.data(), rhs.data(), lhs.get_byte_size()) == 0;
    }
    return are_eq;
}

ov::Tensor evaluate_bound(const Output<Node>& output, bool is_upper, bool invalidate_all_unused_values = true) {
    if (is_upper && output.get_tensor().get_upper_value()) {
        return output.get_tensor().get_upper_value();
    }

    if (!is_upper && output.get_tensor().get_lower_value()) {
        return output.get_tensor().get_lower_value();
    }

    std::vector<Node*> order;
    if (could_propagate(output, order)) {
        reverse(order.begin(), order.end());
        for (const auto& node : order) {
            ov::TensorVector outputs;
            for (const auto& out : node->outputs()) {
                const auto& out_shape = out.get_partial_shape();
                const auto& out_et = out.get_element_type();

                if (out_et.is_dynamic()) {
                    outputs.emplace_back();
                } else if (out_shape.is_static()) {
                    outputs.emplace_back(out_et, out_shape.to_shape());
                } else if (out_shape.rank().is_static()) {
                    outputs.emplace_back(out_et, Shape(out_shape.rank().get_length()));
                } else {
                    outputs.emplace_back(out_et, Shape{0});
                }
            }

            if (is_upper ? node->evaluate_upper(outputs) : node->evaluate_lower(outputs)) {
                const auto& input_values = node->input_values();
                TensorLabelVector output_labels(outputs.size());

                bool same_inputs = std::all_of(input_values.begin(), input_values.end(), [](const Output<Node>& input) {
                    auto& t = input.get_tensor();
                    return t.has_and_set_bound() || are_equal(t.get_lower_value(), t.get_upper_value());
                });

                for (size_t i = 0; i < outputs.size(); ++i) {
                    if ((same_inputs || is_upper) && !node->get_output_tensor(i).get_upper_value() && outputs[i]) {
                        node->get_output_tensor(i).set_upper_value(outputs[i]);
                    }

                    if ((same_inputs || !is_upper) && !node->get_output_tensor(i).get_lower_value() && outputs[i]) {
                        node->get_output_tensor(i).set_lower_value(outputs[i]);
                    }

                    if (are_equal(node->get_output_tensor(i).get_lower_value(),
                                  node->get_output_tensor(i).get_upper_value())) {
                        node->get_output_tensor(i).set_lower_value(node->get_output_tensor(i).get_upper_value());
                    }
                }

                if (node->evaluate_label(output_labels))
                    for (size_t i = 0; i < outputs.size(); ++i)
                        node->get_output_tensor(i).set_value_label(output_labels[i]);

                for (const auto& input : input_values) {
                    auto& tensor = input.get_tensor();
                    bool should_invalidate = invalidate_all_unused_values;
                    if (tensor.get_lower_value() && shape_size(tensor.get_lower_value().get_shape()) > 10)
                        should_invalidate |= true;

                    if (tensor.get_upper_value() && shape_size(tensor.get_upper_value().get_shape()) > 10)
                        should_invalidate |= true;

                    if (should_invalidate && input.get_target_inputs().size() == 1)
                        tensor.invalidate_values();
                }
                propagate_rt_info(node, output);
            } else {
                break;
            }
        }
    }

    if (is_upper)
        return output.get_tensor().get_upper_value();
    else
        return output.get_tensor().get_lower_value();
}

bool default_bound_evaluator(const ov::Node* node,
                             const ov::Tensor& (ov::descriptor::Tensor::*get_bound)() const,
                             ov::TensorVector& output_values) {
    const auto size = node->get_input_size();

    ov::TensorVector inputs;
    inputs.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        if (auto bound = (node->get_input_tensor(i).*get_bound)()) {
            inputs.push_back(bound);
        } else {
            return false;
        }
    }
    return node->evaluate(output_values, inputs);
}

ov::Tensor equality_mask(const ov::Tensor& tensor, const std::shared_ptr<op::v0::Constant>& constant) {
    auto mask_out = ov::TensorVector{{element::boolean, tensor.get_shape()}};

    auto c_tensor = ov::Tensor(constant->get_element_type(), constant->get_shape());
    memcpy(c_tensor.data(), constant->get_data_ptr(), c_tensor.get_byte_size());

    const auto& param = std::make_shared<op::v0::Parameter>(tensor.get_element_type(), tensor.get_shape());
    op::v1::Equal(param, constant).evaluate(mask_out, ov::TensorVector{tensor, c_tensor});
    return mask_out.front();
}

ov::Tensor or_tensor(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    auto outs = ov::TensorVector{{lhs.get_element_type(), Shape{0}}};
    op::v1::LogicalOr(std::make_shared<op::v0::Parameter>(lhs.get_element_type(), lhs.get_shape()),
                      std::make_shared<op::v0::Parameter>(rhs.get_element_type(), rhs.get_shape()),
                      ngraph::op::AutoBroadcastType::NUMPY)
        .evaluate(outs, ov::TensorVector{lhs, rhs});
    return outs.front();
}

struct TensorVectorCmp {
    bool operator()(const ov::TensorVector& lhs, const ov::TensorVector& rhs) const {
        auto rhs_it = rhs.begin();
        return std::any_of(lhs.begin(), lhs.end(), [&rhs_it](const ov::Tensor& lhs) {
            bool is_less =
                (lhs && *rhs_it) ? lhs.data() < rhs_it->data() : static_cast<bool>(lhs) < static_cast<bool>(*rhs_it);
            ++rhs_it;
            return is_less;
        });
    }
};

ov::Tensor make_tensor_max_of_type(ov::element::Type_t t) {
#define OV_TYPE_TO_MAX_CONST(ET, TENSOR)                                                                   \
    case ET:                                                                                               \
        *(TENSOR.data<fundamental_type_for<ET>>()) = std::numeric_limits<fundamental_type_for<ET>>::max(); \
        break

    auto tensor = ov::Tensor(t, Shape{});

    switch (t) {
        OV_TYPE_TO_MAX_CONST(element::boolean, tensor);
        OV_TYPE_TO_MAX_CONST(element::bf16, tensor);
        OV_TYPE_TO_MAX_CONST(element::f16, tensor);
        OV_TYPE_TO_MAX_CONST(element::f32, tensor);
        OV_TYPE_TO_MAX_CONST(element::f64, tensor);
        OV_TYPE_TO_MAX_CONST(element::i8, tensor);
        OV_TYPE_TO_MAX_CONST(element::i16, tensor);
        OV_TYPE_TO_MAX_CONST(element::i32, tensor);
        OV_TYPE_TO_MAX_CONST(element::i64, tensor);
        OV_TYPE_TO_MAX_CONST(element::u1, tensor);
        OV_TYPE_TO_MAX_CONST(element::u8, tensor);
        OV_TYPE_TO_MAX_CONST(element::u16, tensor);
        OV_TYPE_TO_MAX_CONST(element::u32, tensor);
        OV_TYPE_TO_MAX_CONST(element::u64, tensor);
    default:
        break;
    }

#undef OV_TYPE_TO_MAX_CONST
    return tensor;
}

}  // namespace

bool ov::could_propagate(const Output<Node>& output, std::vector<Node*>& order) {
    auto status = true;

    std::deque<Node*> nodes_to_calculate = {output.get_node()};
    order.push_back(output.get_node());

    while (status && !nodes_to_calculate.empty()) {
        auto current_node = nodes_to_calculate.front();
        nodes_to_calculate.pop_front();

        if (current_node->inputs().empty() && !is_type<op::v0::Constant>(current_node)) {
            status = false;
        } else if (!is_type<op::v0::ShapeOf>(current_node) && !is_type<op::v3::ShapeOf>(current_node)) {
            // not a leaf, not a shape_of -- continue to search
            for (const auto& input_value : current_node->input_values()) {
                const auto& input_node = input_value.get_node();
                order.push_back(input_node);
                nodes_to_calculate.push_front(input_node);
            }
        }
    }
    return status;
}

ov::Tensor ov::evaluate_lower_bound(const Output<Node>& output) {
    return evaluate_bound(output, false);
}

ov::Tensor ov::evaluate_upper_bound(const Output<Node>& output) {
    return evaluate_bound(output, true);
}

std::pair<ov::Tensor, ov::Tensor> ov::evaluate_both_bounds(const Output<Node>& output) {
    evaluate_bound(output, false, false);
    evaluate_bound(output, true);
    return {output.get_tensor_ptr()->get_lower_value(), output.get_tensor_ptr()->get_upper_value()};
}

bool ov::default_lower_bound_evaluator(const Node* node, TensorVector& output_values) {
    return default_bound_evaluator(node, &descriptor::Tensor::get_lower_value, output_values);
}

bool ov::default_upper_bound_evaluator(const Node* node, TensorVector& output_values) {
    return default_bound_evaluator(node, &descriptor::Tensor::get_upper_value, output_values);
}

bool ov::interval_bound_evaluator(const Node* node,
                                  TensorVector& lower_output_values,
                                  TensorVector& upper_output_values) {
    // TODO: relax for n inputs ?
    OPENVINO_ASSERT(lower_output_values.size() == upper_output_values.size());
    OPENVINO_ASSERT(node->get_input_size() == 2);

    const auto num_of_outputs = node->get_output_size();
    auto low_0 = ov::evaluate_lower_bound(node->get_input_source_output(0));
    auto low_1 = ov::evaluate_lower_bound(node->get_input_source_output(1));
    auto up_0 = ov::evaluate_upper_bound(node->get_input_source_output(0));
    auto up_1 = ov::evaluate_upper_bound(node->get_input_source_output(1));

    std::set<TensorVector, TensorVectorCmp> input_variants = {{low_0, low_1},
                                                              {low_0, up_1},
                                                              {up_0, low_1},
                                                              {up_0, up_1}};

    for (const auto& variant_of_input_vector : input_variants)
        for (const auto& input_tensor : variant_of_input_vector)
            if (!input_tensor)
                return false;

    if (input_variants.size() == 1)
        return node->evaluate(upper_output_values, *input_variants.begin()) &&
               node->evaluate(lower_output_values, *input_variants.begin());

    auto zero = op::v0::Constant::create(element::i64, {1}, {0});
    const auto zero_t = ov::Tensor(element::i64, Shape{1});
    *zero_t.data<int64_t>() = 0;

    std::vector<TensorVector> unsqueezed_output_variants;
    for (auto& input_variant : input_variants) {
        TensorVector vector_of_output_variants;
        for (const auto& output : lower_output_values) {
            vector_of_output_variants.emplace_back(output.get_element_type(), output.get_shape());
        }

        node->evaluate(vector_of_output_variants, input_variant);

        TensorVector vector_of_unsqueezed_output_variants;
        for (const auto& output : vector_of_output_variants) {
            if (!output) {
                return false;
            }

            auto unsqueezed_shape = output.get_shape();
            unsqueezed_shape.insert(unsqueezed_shape.begin(), 1);

            auto unsqueezed_outputs = TensorVector{{output.get_element_type(), unsqueezed_shape}};
            auto& unsqueezed = unsqueezed_outputs.front();

            op::v0::Unsqueeze().evaluate(unsqueezed_outputs, TensorVector{output, zero_t});
            vector_of_unsqueezed_output_variants.push_back(unsqueezed);
        }
        unsqueezed_output_variants.push_back(vector_of_unsqueezed_output_variants);
    }

    auto input_0_maximum_value = ngraph::get_constant_max_of_type(low_0.get_element_type());
    auto input_1_maximum_value = ngraph::get_constant_max_of_type(low_1.get_element_type());
    if (input_0_maximum_value == nullptr || input_1_maximum_value == nullptr)
        return false;

    auto input_0_low_dyn_mask = equality_mask(low_0, input_0_maximum_value);
    auto input_0_up_dyn_mask = equality_mask(up_0, input_0_maximum_value);
    auto input_1_low_dyn_mask = equality_mask(low_1, input_1_maximum_value);
    auto input_1_up_dyn_mask = equality_mask(up_1, input_1_maximum_value);
    auto final_input_dyn_mask = or_tensor(or_tensor(input_0_low_dyn_mask, input_0_up_dyn_mask),
                                          or_tensor(input_1_low_dyn_mask, input_1_up_dyn_mask));

    bool fully_defined = true;
    for (size_t i = 0; i < num_of_outputs; ++i) {
        TensorVector all_variants_for_ith_output;
        for (const auto& unsqueezed_output_variant : unsqueezed_output_variants)
            all_variants_for_ith_output.push_back(unsqueezed_output_variant[i]);

        auto concated_shape = all_variants_for_ith_output[0].get_shape();
        concated_shape[0] = all_variants_for_ith_output.size();
        auto concat = Tensor(all_variants_for_ith_output[0].get_element_type(), concated_shape);
        auto concat_out = TensorVector{concat};
        auto c = op::v0::Concat();
        c.set_axis(0);
        c.evaluate(concat_out, all_variants_for_ith_output);

        auto fake_param =
            std::make_shared<op::v0::Parameter>(all_variants_for_ith_output[0].get_element_type(), concated_shape);
        auto reduce_min_op = op::v1::ReduceMin(fake_param, zero, false);
        auto lower_out = ov::TensorVector{lower_output_values[i]};
        reduce_min_op.evaluate(lower_out, {concat, zero_t});

        auto reduce_max_op = op::v1::ReduceMax(fake_param, zero, false);
        auto upper_out = ov::TensorVector{upper_output_values[i]};
        reduce_max_op.evaluate(upper_out, {concat, zero_t});

        if (!upper_output_values[i]) {
            fully_defined = false;
        } else {
            const auto output_maximum_value = make_tensor_max_of_type(upper_output_values[i].get_element_type());

            op::v1::Select().evaluate(upper_out, {final_input_dyn_mask, output_maximum_value, upper_output_values[i]});
            node->get_output_tensor(i).set_upper_value(upper_output_values[i]);
        }

        if (!lower_output_values[i]) {
            fully_defined = false;
        } else {
            // Can not set to get_constant_min_of_type(lower_output_values[i]->get_element_type())
            // yet
            op::v1::Select().evaluate(lower_out, {final_input_dyn_mask, zero_t, lower_output_values[i]});
            node->get_output_tensor(i).set_lower_value(lower_output_values[i]);
        }
    }
    return fully_defined;
}

bool ov::tensor_is_positive(const Tensor& bound) {
    const auto bound_constant =
        std::make_shared<op::v0::Constant>(bound.get_element_type(), bound.get_shape(), bound.data());
    const auto zero_constant = op::v0::Constant::create(bound.get_element_type(), {1}, {0});
    OutputVector greater(1);

    bool folded = std::make_shared<op::v1::Greater>(bound_constant, zero_constant)
                      ->constant_fold(greater, {bound_constant, zero_constant});
    OPENVINO_ASSERT(folded);

    auto axes_vector = std::vector<int64_t>(greater[0].get_shape().size());
    std::iota(axes_vector.begin(), axes_vector.end(), 0);
    const auto axes = op::v0::Constant::create(element::i64, {axes_vector.size()}, axes_vector);

    OutputVector all(1);
    folded = std::make_shared<op::v1::ReduceLogicalAnd>(greater[0], axes)->constant_fold(all, {greater[0], axes});
    OPENVINO_ASSERT(folded && ov::is_type<op::v0::Constant>(all[0].get_node_shared_ptr()));
    OPENVINO_ASSERT(all[0].get_shape() == Shape{});
    return std::dynamic_pointer_cast<op::v0::Constant>(all[0].get_node_shared_ptr())->cast_vector<bool>()[0];
}

bool ov::has_and_set_equal_bounds(const Output<Node>& source) {
    if (op::util::is_constant(source.get_node_shared_ptr()))
        return true;

    auto bounds = ov::evaluate_both_bounds(source);
    return are_same_tensor(bounds.first, bounds.second);
}

bool ov::have_node_inputs_bounds_set(const Node* const node, const size_t first_idx, const size_t last_idx) {
    bool have_bound_set = last_idx <= node->get_input_size();
    for (size_t i = first_idx; have_bound_set && (i <= last_idx); ++i) {
        have_bound_set = node->get_input_tensor(i).has_and_set_bound();
    }
    return have_bound_set;
}
