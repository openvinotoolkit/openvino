// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bound_evaluate.hpp"

#include "compare.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/tensor_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/symbolic_info.hpp"
#include "openvino/opsets/opset10.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/is_shape_subgraph.hpp"

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

bool are_equal(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    if (!lhs || !rhs) {
        return false;
    }
    const auto& lhs_shape = lhs.get_shape();
    const auto& rhs_shape = rhs.get_shape();

    const auto& lhs_et = lhs.get_element_type();
    const auto& rhs_et = rhs.get_element_type();

    auto are_eq = (lhs_et == rhs_et) && (lhs_shape == rhs_shape);

    if (are_eq) {
        are_eq = memcmp(lhs.data(), rhs.data(), lhs.get_byte_size()) == 0;
    }
    return are_eq;
}

bool is_type_allocable(const element::Type& type) {
    return type != element::undefined && type.is_static();
}

/**
 * @brief Checks if node's inputs has bound set and are equal.
 *
 * @param node  Node
 * @return true if all inputs have both bounds set and are equal, otherwise false.
 */
bool node_has_same_inputs(const Node& node) {
    const auto& values = node.input_values();

    return std::all_of(values.begin(), values.end(), [](const Output<Node>& value) {
        const auto& t_desc = value.get_tensor();
        return t_desc.has_and_set_bound() || are_equal(t_desc.get_lower_value(), t_desc.get_upper_value());
    });
};

/**
 * @brief Invalidates unused tensor values for outputs (except Constants)
 *
 * @param outputs  Outputs to apply invalidation.
 */
void invalidate_unused_values(const ov::OutputVector& outputs) {
    for (const auto& output : outputs) {
        auto& tensor = output.get_tensor();
        const auto& lower = tensor.get_lower_value();
        const auto& upper = tensor.get_upper_value();
        const auto should_invalidate =
            (lower && shape_size(lower.get_shape()) > 10) || (upper && shape_size(upper.get_shape()) > 10);
        if (should_invalidate && output.get_target_inputs().size() == 1) {
            tensor.invalidate_values();
        }
    }
}

/** @brief Common base for single bound evaluation. */
struct SingleBound {
    void init(const Node& node) {
        bounds.resize(0);
        for (const auto& output : node.outputs()) {
            if (is_type_allocable(output.get_element_type())) {
                bounds.emplace_back(output);
            } else {
                bounds.emplace_back();
            }
        }
    }

    void set_same_bound_at_port(descriptor::Tensor& tensor_desc, const size_t port) const {
        tensor_desc.set_lower_value(bounds[port]);
        tensor_desc.set_upper_value(bounds[port]);
    }

    TensorVector bounds;
};

/** @brief Lower bound evaluation specific actions. */
struct LowerBound : SingleBound {
    static bool requires_evaluation(const descriptor::Tensor& tensor_desc) {
        return !tensor_desc.get_lower_value();
    }

protected:
    bool evaluate(const Node& node) {
        return node.evaluate_lower(bounds);
    }

    void set_bound_at_port(descriptor::Tensor& tensor_desc, const size_t port) const {
        tensor_desc.set_lower_value(bounds[port]);
        if (are_equal(bounds[port], tensor_desc.get_upper_value())) {
            tensor_desc.set_upper_value(bounds[port]);
        }
    }
};

/** @brief Upper bound evaluation specific actions. */
struct UpperBound : SingleBound {
    static bool requires_evaluation(const ov::descriptor::Tensor& tensor_desc) {
        return !tensor_desc.get_upper_value();
    }

protected:
    bool evaluate(const ov::Node& node) {
        return node.evaluate_upper(bounds);
    }

    void set_bound_at_port(descriptor::Tensor& tensor_desc, const size_t port) const {
        tensor_desc.set_upper_value(bounds[port]);
        if (are_equal(bounds[port], tensor_desc.get_lower_value())) {
            tensor_desc.set_lower_value(bounds[port]);
        }
    }
};

/** @brief Both bounds evaluation specific actions. */
struct BothBounds {
    static bool requires_evaluation(const descriptor::Tensor& tensor_desc) {
        return !tensor_desc.get_lower_value() || !tensor_desc.get_upper_value();
    }

protected:
    void init(const Node& node) {
        lowers.resize(0);
        uppers.resize(0);

        for (const auto& output : node.outputs()) {
            if (is_type_allocable(output.get_element_type())) {
                lowers.emplace_back(output);
                uppers.emplace_back(output);
            } else {
                lowers.emplace_back();
                uppers.emplace_back();
            }
        }
    }

    void set_same_bound_at_port(descriptor::Tensor& tensor_desc, const size_t port) const {
        tensor_desc.set_lower_value(lowers[port]);
        tensor_desc.set_upper_value(lowers[port]);
    }

    void set_bound_at_port(descriptor::Tensor& tensor_desc, const size_t port) const {
        tensor_desc.set_lower_value(lowers[port]);
        if (are_equal(lowers[port], uppers[port])) {
            tensor_desc.set_upper_value(lowers[port]);
        } else {
            tensor_desc.set_upper_value(uppers[port]);
        }
    }

    bool evaluate(const Node& node) {
        return node.evaluate_lower(lowers) && node.evaluate_upper(uppers);
    }

    ov::TensorVector lowers, uppers;
};

/** @brief Evaluates and sets symbols. */
struct SymbolEvaluator {
    void init() {
        symbols.resize(0);
    }

    void evaluate(const Node& node) {
        symbols.resize(node.get_output_size());
        if (!node.evaluate_symbol(symbols)) {
            symbols.resize(0);
        }
    }

    void set_symbols_at_port(descriptor::Tensor& tensor_desc, const size_t port) const {
        if (!symbols.empty() && tensor_desc.get_value_symbol().empty()) {
            tensor_desc.set_value_symbol(symbols[port]);
            ov::populate_tensor_with_missing_symbols(tensor_desc);
        }
    }

    TensorSymbolVector symbols;
};

/** @brief Evaluator evaluate and set bound depends specified BoundType. */
template <class BoundType>
struct Evaluator : BoundType {
    void init(const Node* n) {
        node = n;
        symbol_evaluator.init();
        BoundType::init(*node);
    }

    void set_bounds_and_symbols() const {
        const auto set_bound =
            node_has_same_inputs(*node) ? &Evaluator::set_same_bound_at_port : &Evaluator::set_bound_at_port;

        for (size_t port = 0; port < node->get_output_size(); ++port) {
            auto& output_tensor_desc = node->get_output_tensor(port);

            (this->*set_bound)(output_tensor_desc, port);
            symbol_evaluator.set_symbols_at_port(output_tensor_desc, port);
        }
    }

    bool evaluate() {
        if (BoundType::evaluate(*node)) {
            symbol_evaluator.evaluate(*node);
            return true;
        } else {
            return false;
        }
    }

    const Node* node;
    SymbolEvaluator symbol_evaluator;
};

/**
 * @brief Evaluate bound(s) algorithm.
 *
 * Calculate required bounds for Node's output.
 *
 * @tparam        Evaluator Specify the bound type evaluator.
 * @param output  Node's output for which bounds will be calculated.
 */
template <class Evaluator>
void evaluate_bound(const Output<Node>& output) {
    std::vector<Node*> ordered_nodes;
    if (Evaluator::requires_evaluation(output.get_tensor()) && could_propagate(output, ordered_nodes)) {
        auto bound_evaluator = Evaluator{};
        for (const auto& node : ordered_nodes) {
            bound_evaluator.init(node);
            if (!bound_evaluator.evaluate()) {
                break;
            }
            bound_evaluator.set_bounds_and_symbols();
            invalidate_unused_values(node->input_values());
            propagate_rt_info(node, output);
        }
    }
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

ov::Tensor equality_mask(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    auto mask_out = ov::TensorVector{{element::boolean, lhs.get_shape()}};
    const auto l_param = std::make_shared<op::v0::Parameter>(lhs.get_element_type(), lhs.get_shape());
    const auto r_param = std::make_shared<op::v0::Parameter>(rhs.get_element_type(), rhs.get_shape());
    op::v1::Equal(l_param, r_param).evaluate(mask_out, ov::TensorVector{lhs, rhs});
    return mask_out.front();
}

ov::Tensor or_tensor(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    auto logical_or = op::v1::LogicalOr(std::make_shared<op::v0::Parameter>(lhs.get_element_type(), lhs.get_shape()),
                                        std::make_shared<op::v0::Parameter>(rhs.get_element_type(), rhs.get_shape()),
                                        op::AutoBroadcastType::NUMPY);

    auto outs = ov::TensorVector{{lhs.get_element_type(), logical_or.get_output_shape(0)}};
    logical_or.evaluate(outs, ov::TensorVector{lhs, rhs});
    return outs.front();
}

struct TensorVectorCmp {
    // Comparing Tensor vectors as numbers composed with pointers as digits.
    // Indexed loop used to preserve order of comparison.
    bool operator()(const ov::TensorVector& lhs, const ov::TensorVector& rhs) const {
        const auto lhs_size = lhs.size();
        const auto rhs_size = rhs.size();

        if (lhs_size < rhs_size)
            return true;
        if (lhs_size > rhs_size)
            return false;

        for (size_t i = 0; i < lhs_size; ++i) {
            if (lhs[i].data() < rhs[i].data())
                return true;
            if (lhs[i].data() > rhs[i].data())
                return false;
        }

        // if all equals
        return false;
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

bool ov::could_propagate(const Output<Node>& output, std::vector<Node*>& result) {
    auto status = true;

    std::stack<Node*, std::vector<Node*>> nodes_to_do;
    nodes_to_do.push(output.get_node());
    std::unordered_set<Node*> nodes_done;

    while (status && nodes_to_do.size() > 0) {
        Node* node = nodes_to_do.top();
        if (nodes_done.count(node) == 0) {
            bool can_add = true;
            size_t arg_count = node->get_input_size();

            auto node_shared_ptr = node->shared_from_this();
            bool is_decompress_data_path = is_decompression(node_shared_ptr) && !is_shape_subgraph(node_shared_ptr);
            if ((arg_count == 0 && !is_type<op::v0::Constant>(node)) || is_decompress_data_path) {
                status = false;
                continue;
            } else if (is_type<op::v0::ShapeOf>(node) || is_type<op::v3::ShapeOf>(node)) {
                result.push_back(node);
                nodes_to_do.pop();
                nodes_done.insert(node);
                continue;
            }

            for (size_t i = 0; i < arg_count; ++i) {
                Node* dep = node->get_input_node_ptr(arg_count - i - 1);
                if (nodes_done.count(dep) == 0) {
                    can_add = false;
                    nodes_to_do.push(dep);
                }
            }
            for (auto& depptr : node->get_control_dependencies()) {
                Node* dep = depptr.get();
                if (nodes_done.count(dep) == 0) {
                    can_add = false;
                    nodes_to_do.push(dep);
                }
            }
            if (can_add) {
                result.push_back(node);
                nodes_to_do.pop();
                nodes_done.insert(node);
            }
        } else {
            nodes_to_do.pop();
        }
    }
    return status;
}

ov::Tensor ov::util::evaluate_lower_bound(const Output<Node>& output) {
    evaluate_bound<Evaluator<LowerBound>>(output);
    return output.get_tensor().get_lower_value();
}

ov::Tensor ov::util::evaluate_upper_bound(const Output<Node>& output) {
    evaluate_bound<Evaluator<UpperBound>>(output);
    return output.get_tensor().get_upper_value();
}

std::pair<ov::Tensor, ov::Tensor> ov::util::evaluate_both_bounds(const Output<Node>& output) {
    evaluate_bound<Evaluator<BothBounds>>(output);
    const auto& output_tensor_desc = output.get_tensor();
    return {output_tensor_desc.get_lower_value(), output_tensor_desc.get_upper_value()};
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
    auto low_0 = ov::util::evaluate_lower_bound(node->get_input_source_output(0));
    auto low_1 = ov::util::evaluate_lower_bound(node->get_input_source_output(1));
    auto up_0 = ov::util::evaluate_upper_bound(node->get_input_source_output(0));
    auto up_1 = ov::util::evaluate_upper_bound(node->get_input_source_output(1));
    if (!low_0 || !low_1 || !up_0 || !up_1)
        return false;

    std::set<TensorVector, TensorVectorCmp> input_variants = {{low_0, low_1},
                                                              {low_0, up_1},
                                                              {up_0, low_1},
                                                              {up_0, up_1}};

    if (input_variants.size() == 1)
        return node->evaluate(upper_output_values, *input_variants.begin()) &&
               node->evaluate(lower_output_values, *input_variants.begin());

    auto zero = op::v0::Constant::create(element::i64, {1}, {0});
    const auto zero_t = ov::Tensor(element::i64, Shape{});
    *zero_t.data<int64_t>() = 0;

    std::vector<TensorVector> unsqueezed_output_variants;
    for (auto& input_variant : input_variants) {
        TensorVector vector_of_output_variants;
        for (const auto& output : lower_output_values) {
            vector_of_output_variants.emplace_back(output.get_element_type(), output.get_shape());
        }

        if (!node->evaluate(vector_of_output_variants, input_variant)) {
            return false;
        };

        TensorVector vector_of_unsqueezed_output_variants;
        for (const auto& output : vector_of_output_variants) {
            auto unsqueezed_shape = output.get_shape();
            unsqueezed_shape.insert(unsqueezed_shape.begin(), 1);

            auto unsqueezed_outputs = TensorVector{{output.get_element_type(), unsqueezed_shape}};
            auto& unsqueezed = unsqueezed_outputs.front();

            op::v0::Unsqueeze().evaluate(unsqueezed_outputs, TensorVector{output, zero_t});
            vector_of_unsqueezed_output_variants.push_back(unsqueezed);
        }
        unsqueezed_output_variants.push_back(vector_of_unsqueezed_output_variants);
    }
    const auto input_0_maximum_value = ov::util::make_tensor_of_max_value(low_0.get_element_type());
    const auto input_1_maximum_value = ov::util::make_tensor_of_max_value(low_1.get_element_type());
    if (!input_0_maximum_value || !input_1_maximum_value)
        return false;

    const auto input_0_low_dyn_mask = equality_mask(low_0, input_0_maximum_value);
    const auto input_0_up_dyn_mask = equality_mask(up_0, input_0_maximum_value);
    const auto input_1_low_dyn_mask = equality_mask(low_1, input_1_maximum_value);
    const auto input_1_up_dyn_mask = equality_mask(up_1, input_1_maximum_value);
    const auto final_input_dyn_mask = or_tensor(or_tensor(input_0_low_dyn_mask, input_0_up_dyn_mask),
                                                or_tensor(input_1_low_dyn_mask, input_1_up_dyn_mask));

    bool fully_defined = true;
    for (size_t i = 0; i < num_of_outputs; ++i) {
        TensorVector all_variants_for_ith_output;
        for (const auto& unsqueezed_output_variant : unsqueezed_output_variants)
            all_variants_for_ith_output.push_back(unsqueezed_output_variant[i]);

        auto concated_shape = all_variants_for_ith_output[0].get_shape();
        concated_shape[0] = all_variants_for_ith_output.size();
        auto concat = TensorVector{Tensor(all_variants_for_ith_output[0].get_element_type(), concated_shape)};
        auto c = op::v0::Concat();
        c.set_axis(0);
        c.evaluate(concat, all_variants_for_ith_output);

        auto fake_param =
            std::make_shared<op::v0::Parameter>(all_variants_for_ith_output[0].get_element_type(), concated_shape);
        auto reduce_min_op = op::v1::ReduceMin(fake_param, zero, false);
        auto lower_out = ov::TensorVector{lower_output_values[i]};
        concat.push_back(zero_t);
        reduce_min_op.evaluate(lower_out, concat);

        auto reduce_max_op = op::v1::ReduceMax(fake_param, zero, false);
        auto upper_out = ov::TensorVector{upper_output_values[i]};
        reduce_max_op.evaluate(upper_out, concat);

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
            // Can not set to make_tensor_of_min_value(lower_output_values[i]->get_element_type()) yet
            const auto then = Tensor{lower_out[0].get_element_type(), Shape{}};
            const auto then_data = static_cast<char*>(then.data());
            std::memset(then_data, 0, then.get_byte_size());
            op::v1::Select().evaluate(lower_out, {final_input_dyn_mask, then, lower_out[0]});
            node->get_output_tensor(i).set_lower_value(lower_out[0]);
        }
    }
    return fully_defined;
}

bool ov::tensor_has_max_value(const Tensor& bound) {
    const auto bound_constant =
        std::make_shared<op::v0::Constant>(bound.get_element_type(), bound.get_shape(), bound.data());

    const auto max_values = ov::util::make_tensor_of_max_value(bound.get_element_type());
    const auto max_constant = std::make_shared<ov::op::v0::Constant>(max_values);
    OutputVector equal(1);

    bool folded = std::make_shared<op::v1::Equal>(bound_constant, max_constant)
                      ->constant_fold(equal, {bound_constant, max_constant});
    OPENVINO_ASSERT(folded);

    auto axes_vector = std::vector<int64_t>(equal[0].get_shape().size());
    std::iota(axes_vector.begin(), axes_vector.end(), 0);
    const auto axes = op::v0::Constant::create(element::i64, {axes_vector.size()}, axes_vector);

    OutputVector all(1);
    folded = std::make_shared<op::v1::ReduceLogicalOr>(equal[0], axes)->constant_fold(all, {equal[0], axes});
    OPENVINO_ASSERT(folded && ov::is_type<op::v0::Constant>(all[0].get_node_shared_ptr()));
    OPENVINO_ASSERT(all[0].get_shape() == Shape{});
    return std::dynamic_pointer_cast<op::v0::Constant>(all[0].get_node_shared_ptr())->cast_vector<bool>()[0];
}

bool ov::tensor_has_zero_value(const Tensor& bound) {
    const auto bound_constant =
        std::make_shared<op::v0::Constant>(bound.get_element_type(), bound.get_shape(), bound.data());
    const auto zero_constant = op::v0::Constant::create(bound.get_element_type(), {1}, {0});
    OutputVector equal(1);

    bool folded = std::make_shared<op::v1::Equal>(bound_constant, zero_constant)
                      ->constant_fold(equal, {bound_constant, zero_constant});
    OPENVINO_ASSERT(folded);

    auto axes_vector = std::vector<int64_t>(equal[0].get_shape().size());
    std::iota(axes_vector.begin(), axes_vector.end(), 0);
    const auto axes = op::v0::Constant::create(element::i64, {axes_vector.size()}, axes_vector);

    OutputVector all(1);
    folded = std::make_shared<op::v1::ReduceLogicalOr>(equal[0], axes)->constant_fold(all, {equal[0], axes});
    OPENVINO_ASSERT(folded && ov::is_type<op::v0::Constant>(all[0].get_node_shared_ptr()));
    OPENVINO_ASSERT(all[0].get_shape() == Shape{});
    return std::dynamic_pointer_cast<op::v0::Constant>(all[0].get_node_shared_ptr())->cast_vector<bool>()[0];
}

bool ov::has_and_set_equal_bounds(const Output<Node>& source) {
    if (op::util::is_constant(source.get_node_shared_ptr()))
        return true;

    auto bounds = ov::util::evaluate_both_bounds(source);
    return are_same_tensor(bounds.first, bounds.second);
}

bool ov::have_node_inputs_bounds_set(const Node* const node, const size_t first_idx, const size_t last_idx) {
    bool have_bound_set = last_idx < node->get_input_size();
    for (size_t i = first_idx; have_bound_set && (i <= last_idx); ++i) {
        have_bound_set = node->get_input_tensor(i).has_and_set_bound();
    }
    return have_bound_set;
}

namespace {
/// \brief Encodes tensor symbol vector as tensor integer vector for the purpose of evaluation. Provides the key for
/// decoding back.
///
/// \param symbols All symbols that are planned for evaluation
/// \param integer_representations Container representing resulting encodings
/// \param key Map representing resulting key for decoding
void symbols_to_integer_and_key(const TensorSymbolVector& symbols,
                                std::vector<std::vector<int32_t>>& integer_representations,
                                std::unordered_map<int32_t, std::shared_ptr<Symbol>>& key) {
    int32_t x = 0;
    std::unordered_map<std::shared_ptr<Symbol>, int32_t> key_for_encoding;

    key_for_encoding[nullptr] = 0;
    key[0] = nullptr;

    for (const auto& container : symbols) {
        for (const auto& symbol : container) {
            if (symbol == nullptr)
                continue;
            const auto& root = symbol::ancestor_of(symbol);
            if (key_for_encoding.find(root) == key_for_encoding.end()) {
                x += 1;
                key_for_encoding[root] = x;
                key[x] = root;
            }
        }
    }
    integer_representations.resize(symbols.size());
    for (size_t i = 0; i < symbols.size(); ++i) {
        integer_representations[i].resize(symbols[i].size());
        for (size_t j = 0; j < symbols[i].size(); ++j) {
            const auto& symbol = symbols[i][j];
            const auto& root = (symbol ? symbol::ancestor_of(symbol) : nullptr);
            integer_representations[i][j] = key_for_encoding[root];
        }
    }
}

/// \brief Decodes tensor integer vector to tensor symbol vector after the evaluation. Uses provided key for decoding.
///
/// \param integer_representations Container representing encodings
/// \param key Map representing key for decoding
/// \param symbols Tensor symbol vector representing resulting symbols after evaluation
void integer_and_key_to_symbols(const std::vector<int32_t>& integer_representations,
                                const std::unordered_map<int32_t, std::shared_ptr<Symbol>>& key,
                                TensorSymbol& symbols) {
    symbols.resize(integer_representations.size());
    for (size_t i = 0; i < integer_representations.size(); ++i) {
        if (key.count(integer_representations[i]))
            symbols[i] = key.at(integer_representations[i]);
        else
            symbols[i] = nullptr;
    }
}
}  // namespace

bool ov::default_symbol_evaluator(const Node* node,
                                  std::initializer_list<size_t> symbol_inputs,
                                  TensorSymbolVector& output_symbols) {
    TensorSymbolVector input_symbols;
    for (const auto& input : node->input_values())
        input_symbols.push_back(input.get_tensor().get_value_symbol());

    /// turn Symbol objects to int32 to put them through evaluate
    std::vector<std::vector<int32_t>> integer_representation;
    std::unordered_map<int32_t, std::shared_ptr<Symbol>> key;
    symbols_to_integer_and_key(input_symbols, integer_representation, key);

    bool has_any_input_symbols = false;
    const auto& inputs_count = node->get_input_size();

    TensorVector inputs;
    inputs.reserve(inputs_count);

    for (size_t i = 0; i < inputs_count; ++i) {
        if (!symbol_inputs.size() || std::find(symbol_inputs.begin(), symbol_inputs.end(), i) != symbol_inputs.end()) {
            const auto& pshape = node->get_input_partial_shape(i);
            if (pshape.is_dynamic())
                return false;

            auto& representation = integer_representation[i];
            if (std::any_of(representation.begin(), representation.end(), [](int32_t& s) {
                    return s > 0;
                }))
                has_any_input_symbols = true;

            representation.resize(shape_size(pshape.to_shape()), 0);
            inputs.emplace_back(element::from<int32_t>(), node->get_input_shape(i));
            std::copy(representation.begin(), representation.end(), inputs.back().data<int32_t>());
        } else {
            if (node->get_input_tensor(i).has_and_set_bound()) {
                inputs.push_back(node->get_input_tensor(i).get_lower_value());
            } else {
                return false;
            }
        }
    }

    if (has_any_input_symbols) {
        const auto& outputs_count = node->get_output_size();
        TensorVector outputs;
        outputs.reserve(outputs_count);

        for (size_t i = 0; i < outputs_count; ++i) {
            const auto& partial_shape = node->get_output_partial_shape(i);
            // Set shape for static or special dynamic if partial shape is dynamic.
            const auto& shape = partial_shape.is_static() ? partial_shape.to_shape() : Shape{0};
            outputs.emplace_back(element::from<int32_t>(), shape);
        }

        if (node->evaluate(outputs, inputs)) {
            std::transform(outputs.cbegin(), outputs.cend(), output_symbols.begin(), [&](const Tensor& t) {
                // Return empty symbol tensor if input tensor not valid (can have Shape{0})
                if (t) {
                    TensorSymbol output_symbol;
                    std::vector<int32_t> integer_output_data(t.data<int32_t>(), t.data<int32_t>() + t.get_size());
                    integer_and_key_to_symbols(integer_output_data, key, output_symbol);
                    return output_symbol;
                } else {
                    return TensorSymbol();
                }
            });
            return true;
        }
    }
    return false;
}
