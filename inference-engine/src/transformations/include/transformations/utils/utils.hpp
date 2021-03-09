// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <assert.h>
#include <vector>
#include <limits>

#include <transformations_visibility.hpp>
#include <ngraph/op/util/op_annotations.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>

namespace ngraph {
namespace op {
namespace util {

template <class T>
bool normalize_single_value(std::vector<T> vec, float & value) {
    for (const auto & val : vec) {
        if (val != *vec.begin()) return false;
    }

    float ref_val = static_cast<float>(*vec.begin());

    if (ref_val < std::numeric_limits<float>::lowest() || ref_val > std::numeric_limits<float>::max()) {
        return false;
    }

    value = ref_val;
    return true;
}

template <class T>
bool has_op_with_type(const std::shared_ptr<const ngraph::Function> &function) {
    for (const auto & op : function->get_ops()) {
        if (std::dynamic_pointer_cast<T>(op)) {
            return true;
        }
    }
    return false;
}

inline std::string create_ie_output_name(const ngraph::Output<ngraph::Node>& output) {
    const auto& prev_layer = output.get_node_shared_ptr();
    std::string out_name = prev_layer->get_friendly_name();
    if (prev_layer->get_output_size() != 1)
        out_name += "." + std::to_string(output.get_index());
    return out_name;
}

template <typename T>
bool has_constant_value(const std::shared_ptr<ngraph::opset4::Constant>& constant,
                        const T value,
                        T epsilon = std::numeric_limits<T>::epsilon()) {
    if (!constant) {
        return false;
    }

    const bool is_scalar_or_single_elem = is_scalar(constant->get_shape()) ||
                                          shape_size(constant->get_shape()) == 1;
    if (!is_scalar_or_single_elem) {
        return false;
    }

    if (constant->get_element_type() == ngraph::element::f16 ||
        constant->get_element_type() == ngraph::element::f32 ||
        constant->get_element_type() == ngraph::element::f64 ||
        constant->get_element_type() == ngraph::element::bf16) {
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

TRANSFORMATIONS_API bool get_single_value(const std::shared_ptr<op::Constant> & const_node, float & value);

TRANSFORMATIONS_API std::shared_ptr<ngraph::Node> normalize_constant(const std::shared_ptr<op::Constant> & constant,
                                                                           const PartialShape & shape);

TRANSFORMATIONS_API std::shared_ptr<ngraph::Node> broadcastTo(const Output<Node>& input, const Shape& shape);

TRANSFORMATIONS_API std::shared_ptr<ngraph::Node> reshapeTo(const Output<Node> & input, const Shape& shape);

TRANSFORMATIONS_API bool constantIsEqualTo(const std::shared_ptr<ngraph::op::Constant>& const_node, float value, float eps = 1e-5);

TRANSFORMATIONS_API bool has_f16_constants(const std::shared_ptr<const ngraph::Function> &function);

TRANSFORMATIONS_API bool check_for_broadcast(const ngraph::Shape &ref_shape, const ngraph::Shape &other_shape);

TRANSFORMATIONS_API std::shared_ptr<ngraph::Node> activation(const std::string& activation_name,
                                                             const ngraph::Output<ngraph::Node>& apply_to);

template <class T>
Output<Node> eltwise_fold(const Output<Node> & input0, const Output<Node> & input1) {
    auto eltwise = std::make_shared<T>(input0, input1);
    OutputVector output(eltwise->get_output_size());
    if (!eltwise->constant_fold(output, {input0, input1})) {
        throw ngraph_error("Can not constant fold eltwise node");
    }
    if (output.size() != 1) {
        throw ngraph_error("Eltwise constant fold has unexpected number of outputs: " + std::to_string(output.size()));
    }
    return output[0];
}
}  // namespace util
}  // namespace op
}  // namespace ngraph
