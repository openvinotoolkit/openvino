// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>

#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/rt_info/attributes.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace util {

template <class T>
bool normalize_single_value(std::vector<T> vec, float& value, bool check_value_range = true) {
    for (const auto& val : vec) {
        if (val != *vec.begin())
            return false;
    }

    float ref_val = static_cast<float>(*vec.begin());

    if (check_value_range &&
        (ref_val < std::numeric_limits<float>::lowest() || ref_val > std::numeric_limits<float>::max())) {
        return false;
    }

    value = ref_val;
    return true;
}

template <class T>
bool has_op_with_type(const std::shared_ptr<const ov::Model>& function) {
    for (const auto& op : function->get_ops()) {
        if (ov::as_type_ptr<T>(op)) {
            return true;
        }
    }
    return false;
}

inline bool has_decompression_converts(const std::shared_ptr<const ov::Model>& function) {
    for (const auto& op : function->get_ops()) {
        if (ov::as_type_ptr<ov::op::v0::Convert>(op)) {
            if (ov::is_decompression(op))
                return true;
        }
    }
    return false;
}

OPENVINO_DEPRECATED("Plugins should use ov::ISyncInferRequest::find_port")
inline std::string create_ie_output_name(const Output<const Node>& output) {
    const auto& prev_layer = output.get_node_shared_ptr();
    auto out_name = prev_layer->get_friendly_name();
    if (prev_layer->get_output_size() != 1) {
        out_name += "." + std::to_string(output.get_index());
    }
    return out_name;
}

OPENVINO_DEPRECATED("Plugins should use ov::ISyncInferRequest::find_port")
inline std::string create_ie_output_name(const Output<Node>& output) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    return create_ie_output_name(ov::Output<const Node>(output.get_node(), output.get_index()));
    OPENVINO_SUPPRESS_DEPRECATED_END
}

OPENVINO_DEPRECATED("Plugins should use ov::ISyncInferRequest::find_port")
inline std::string get_ie_output_name(const Output<const Node>& output) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    return create_ie_output_name(output);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

OPENVINO_DEPRECATED("Plugins should use ov::ISyncInferRequest::find_port")
inline std::string get_ie_output_name(const Output<Node>& output) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    return get_ie_output_name(ov::Output<const Node>(output.get_node(), output.get_index()));
    OPENVINO_SUPPRESS_DEPRECATED_END
}

/**
 * \brief Convert epsilon value from double to float type.
 *
 * If the value is too large, the epsilon is converted to std::numeric_limits<float>::min() or
 * std::numeric_limits<float>::min(), otherwise static cast to float is called.
 * The adjustment is made for positive values only, for negative it works as static cast.
 *
 * \param eps  Original value of the epsilon (double).
 *
 * \return Epsilon value as float.
 */
float cast_eps_to_float(double eps_d);

template <typename T>
bool get_constant_value(const std::shared_ptr<ov::Node>& node, T& value) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant)
        return false;
    if (shape_size(constant->get_shape()) != 1)
        return false;
    value = constant->cast_vector<T>()[0];
    return true;
}

template <typename T>
bool has_constant_value(const std::shared_ptr<Node>& node,
                        const T value,
                        T epsilon = std::numeric_limits<T>::epsilon()) {
    if (!node) {
        return false;
    }

    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant) {
        return false;
    }

    const bool is_scalar_or_single_elem = is_scalar(constant->get_shape()) || shape_size(constant->get_shape()) == 1;
    if (!is_scalar_or_single_elem) {
        return false;
    }

    if (constant->get_element_type() == element::f16 || constant->get_element_type() == element::f32 ||
        constant->get_element_type() == element::f64 || constant->get_element_type() == element::bf16) {
        const auto data = constant->cast_vector<T>();
        if (std::fabs(data[0] - value) > epsilon) {
            return false;
        }
    } else {
        const auto data = constant->cast_vector<T>();
        if (data[0] != value) {
            return false;
        }
    }

    return true;
}

template <typename T>
bool has_constant_value(const std::shared_ptr<Node>& node,
                        const std::vector<T> values,
                        T epsilon = std::numeric_limits<T>::epsilon()) {
    if (!node) {
        return false;
    }

    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant) {
        return false;
    }

    const auto const_values = constant->cast_vector<T>();

    if (constant->get_element_type() == element::f16 || constant->get_element_type() == element::f32 ||
        constant->get_element_type() == element::f64 || constant->get_element_type() == element::bf16) {
        return std::equal(const_values.cbegin(), const_values.cend(), values.cbegin(), [&](T lhs, T rhs) {
            return std::fabs(lhs - rhs) < epsilon;
        });
    }

    return const_values == values;
}

TRANSFORMATIONS_API bool get_single_value(const std::shared_ptr<ov::op::v0::Constant>& const_node,
                                          float& value,
                                          bool check_value_range = true);

TRANSFORMATIONS_API std::shared_ptr<Node> normalize_constant(const std::shared_ptr<ov::op::v0::Constant>& constant,
                                                             const PartialShape& shape);

TRANSFORMATIONS_API std::shared_ptr<Node> broadcastTo(const Output<Node>& input, const Shape& shape);

TRANSFORMATIONS_API std::shared_ptr<Node> reshapeTo(const Output<Node>& input, const Shape& shape);

TRANSFORMATIONS_API bool constantIsEqualTo(const std::shared_ptr<ov::op::v0::Constant>& const_node,
                                           float value,
                                           float eps = 1e-5);

TRANSFORMATIONS_API bool has_f16_constants(const std::shared_ptr<const ov::Model>& function);

TRANSFORMATIONS_API bool is_large_language_model(const ov::Model& model);

/**
 * \brief Check if 'other_shape' can be broadcasted to 'ref_shape'
 *
 * \param ref_shape  The target shape we use as reference we are trying to broadcast to.
 * \param other_shape  The shape we use to check if it can be broadcasted to 'ref_shape'.
 */
TRANSFORMATIONS_API bool check_for_broadcast(const PartialShape& ref_shape, const PartialShape& other_shape);

TRANSFORMATIONS_API std::shared_ptr<Node> activation(const std::string& activation_name, const Output<Node>& apply_to);

TRANSFORMATIONS_API bool is_seq_len_provided(const std::shared_ptr<Node>& X,
                                             const std::shared_ptr<Node>& seq_len_input);

TRANSFORMATIONS_API std::shared_ptr<Node> try_fold_unary_output(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API std::shared_ptr<Node> clone_try_fold(const std::shared_ptr<Node>& node, const OutputVector& inputs);

TRANSFORMATIONS_API bool shapes_equal_except_dynamic_expected_batch(const PartialShape& expected,
                                                                    const PartialShape& actual);

/**
 * \brief Traverses path starting from `node`, and calls "func" for each ov::Node.
 *
 * \param node  The node from which path is started.
 * \param visited  Set of nodes which were visited.
 * \param func  The function which is called for each visited node.
 * \param skip_node_predicate  predicte to skip nodes.
 */
TRANSFORMATIONS_API void visit_path(ov::Node* node,
                                    std::unordered_set<ov::Node*>& visited,
                                    std::function<void(ov::Node*)> func,
                                    std::function<bool(ov::Node*)> skip_node_predicate);

/**
 * \brief Traverses a shapeOf subgraph starting from the node and not including the ShapeOf nodes,
 * and calls "func" for each ov::Node.
 *
 * \param node  The node from which constant path is started.
 * \param visited  Set of nodes which were visited.
 * \param func  The function which is called for each visited node.
 */
TRANSFORMATIONS_API void visit_shape_path(ov::Node* node,
                                          std::unordered_set<ov::Node*>& visited,
                                          std::function<void(ov::Node*)> func);

/**
 * \brief Traverses a constant path starting from "node", and calls "func" for each ov::Node.
 * If the function was called for non-constant subgraph, exception is thrown.
 *
 * \param node  The node from which constant path is started.
 * \param visited  Set of nodes which were visited.
 * \param func  The function which is called for each visited node.
 */
TRANSFORMATIONS_API void visit_constant_path(ov::Node* node,
                                             std::unordered_set<ov::Node*>& visited,
                                             std::function<void(ov::Node*)> func);

template <typename T, typename... Args>
std::shared_ptr<Node> make_try_fold(Args&&... args) {
    auto unary_output_node = std::make_shared<T>(std::forward<Args>(args)...);
    return try_fold_unary_output(unary_output_node);
}

TRANSFORMATIONS_API std::vector<Input<Node>> get_node_target_inputs(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API std::shared_ptr<Node> node_to_get_shape_value_of_indices_from_shape_node(
    const std::shared_ptr<Node>& shape_node,
    const std::vector<size_t>& indices,
    const std::vector<std::shared_ptr<Node>>& copy_rt_info_from = {},
    const ov::element::Type& shape_path_precision = ov::element::i64);

TRANSFORMATIONS_API std::shared_ptr<Node> node_to_get_shape_value_of_indices_from_shape_source(
    const Output<Node>& shape_source,
    const std::vector<size_t>& indices,
    const std::vector<std::shared_ptr<Node>>& copy_rt_info_from = {},
    const ov::element::Type& shape_path_precision = ov::element::i64);

TRANSFORMATIONS_API bool is_dequantization_subgraph(const Output<Node>& node);

TRANSFORMATIONS_API bool can_eliminate_eltwise_node(const std::shared_ptr<Node>& eltwise,
                                                    const Output<Node>& constant,
                                                    const Output<Node>& non_constant_input);

TRANSFORMATIONS_API bool is_constant_and_all_values_equal_int(const Output<Node>& output, const int64_t& v);

TRANSFORMATIONS_API bool is_on_constant_path(const ov::Output<ov::Node>& output);

TRANSFORMATIONS_API bool process_subgraph(ov::pass::ModelPass& model_pass, const std::shared_ptr<Node>& node);

template <typename T>
ov::pass::pattern::op::Predicate constant_predicate(std::function<bool(const std::vector<T>&)> predicate) {
    return ov::pass::pattern::op::Predicate([=](std::shared_ptr<Node> n) -> bool {
        if (auto constant = as_type_ptr<v0::Constant>(n)) {
            auto values = constant->cast_vector<T>();
            return predicate(values);
        }
        return false;
    });
}
}  // namespace util
}  // namespace op
}  // namespace ov

#define INT_CONSTANT_WITH_PREDICATE(expression)                                                   \
    pattern::wrap_type<op::v0::Constant>(                                                         \
        ov::op::util::constant_predicate<int64_t>([](const std::vector<int64_t>& value) -> bool { \
            return expression;                                                                    \
        }))

#define FLOAT_CONSTANT_WITH_PREDICATE(expression)                                             \
    pattern::wrap_type<op::v0::Constant>(                                                     \
        ov::op::util::constant_predicate<float>([](const std::vector<float>& value) -> bool { \
            return expression;                                                                \
        }))
