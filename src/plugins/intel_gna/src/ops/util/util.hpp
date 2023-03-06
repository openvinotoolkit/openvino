// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>
#include <vector>

#include "backend/gna_limitations.hpp"
#include "gna_plugin_config.hpp"
#include "layers/gna_convolution_layer.hpp"
#include "layers/gna_permute.hpp"
#include "legacy/ngraph_ops/convolution_ie.hpp"
#include "legacy/ngraph_ops/crop_ie.hpp"
#include "legacy/ngraph_ops/eltwise.hpp"
#include "legacy/ngraph_ops/fully_connected.hpp"
#include "legacy/ngraph_ops/power.hpp"
#include "legacy/ngraph_ops/relu_ie.hpp"
#include "legacy/ngraph_ops/scaleshift.hpp"
#include "ngraph/opsets/opset7.hpp"
#include "ngraph/opsets/opset8.hpp"
#include "ngraph/opsets/opset9.hpp"
#include "ops/copy.hpp"
#include "ops/identity.hpp"
#include "ops/pwl.hpp"
#include "transformations/rt_info/gna_transpose_fusable.hpp"
#include "transformations/utils/transformation_helper.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gna {
namespace ngraph_util {

template <typename T>
inline bool get_constant_value(const std::shared_ptr<ngraph::opset8::Constant>& constant, std::vector<double>& values) {
    using A = typename ov::element_type_traits<T::value>::value_type;
    const auto& v = constant->get_vector<A>();
    std::copy(v.begin(), v.end(), std::back_inserter(values));
    return true;
}

inline bool get_constant_value(std::tuple<>&&, const std::shared_ptr<ngraph::opset8::Constant>&, std::vector<double>&) {
    return false;
}

template <typename T, typename... Types>
inline bool get_constant_value(std::tuple<T, Types...>&&,
                               const std::shared_ptr<ngraph::opset8::Constant>& constant,
                               std::vector<double>& values) {
    return (constant->get_element_type() == T::value && get_constant_value<T>(constant, values)) ||
           get_constant_value(std::tuple<Types...>(), constant, values);
}

inline bool get_constant_value(const std::shared_ptr<ngraph::opset8::Constant>& constant, std::vector<double>& values) {
    return get_constant_value(std::tuple<std::integral_constant<ov::element::Type_t, ov::element::i32>,
                                         std::integral_constant<ov::element::Type_t, ov::element::i64>,
                                         std::integral_constant<ov::element::Type_t, ov::element::u32>,
                                         std::integral_constant<ov::element::Type_t, ov::element::u64>,
                                         std::integral_constant<ov::element::Type_t, ov::element::f16>,
                                         std::integral_constant<ov::element::Type_t, ov::element::f32>,
                                         std::integral_constant<ov::element::Type_t, ov::element::f64>>(),
                              constant,
                              values);
}

inline bool get_constant_value(const std::shared_ptr<ngraph::opset8::Constant>& constant, double& value) {
    std::vector<double> values;
    if (!get_constant_value(constant, values)) {
        return false;
    }

    if (values.empty() || values.size() > 1) {
        throw std::runtime_error("The size of values is more than 1.");
    }

    value = values[0];
    return true;
}

inline bool is_aligned_split(const std::shared_ptr<ngraph::Node> input_op, size_t input_op_out_index) {
    size_t offset = 0;

    if (std::dynamic_pointer_cast<ngraph::opset8::Split>(input_op) ||
        std::dynamic_pointer_cast<ngraph::opset8::VariadicSplit>(input_op)) {
        for (size_t index = 0; index < input_op_out_index; index++) {
            size_t outputSize = ngraph::shape_size(input_op->get_output_shape(index));
            offset += outputSize * limitations::bytesPerSplitElement;
        }
    }
    return (offset == ALIGN64(offset));
}

inline bool is_crop_affined(std::shared_ptr<ngraph::Node> node) {
    auto crop = std::dynamic_pointer_cast<ngraph::op::CropIE>(node);
    if (crop != nullptr && !crop->offset.empty()) {
        return limitations::isCropAffinedOffset(crop->offset.back());
    }
    return false;
}

// this not only mathematically trivial
inline bool is_trivial_transpose(std::shared_ptr<ngraph::Node> node) {
    auto transpose = std::dynamic_pointer_cast<ngraph::opset8::Transpose>(node);
    if (!transpose)
        return false;

    if (transpose->get_input_size() == 0)
        return false;  // unsupported case

    if (ov::intel_gna::rt_info::is_transpose_fusable(transpose))
        return true;

    auto transpose_const =
        std::dynamic_pointer_cast<ngraph::op::Constant>(transpose->input_value(1).get_node_shared_ptr());
    if (!transpose_const)
        return false;

    auto node_order = transpose_const->cast_vector<int64_t>();

    auto input = transpose->input(0).get_source_output().get_node_shared_ptr();
    auto input_order = transpose->get_input_shape(0);

    return permute::isTrivialPermute(node_order, input_order);
}

inline std::shared_ptr<ov::Node> get_prev_node_skipping_certain(
    const std::shared_ptr<ngraph::Node>& node,
    const std::function<bool(std::shared_ptr<ngraph::Node>)>& skip) {
    auto current_node = node;
    while (skip(current_node)) {
        current_node = current_node->get_input_node_shared_ptr(0);
    }
    return current_node;
}

inline std::shared_ptr<ov::Node> get_next_node_skipping_certain(
    const std::shared_ptr<ngraph::Node>& node,
    const std::function<bool(std::shared_ptr<ngraph::Node>)>& skip) {
    auto current_node = node;
    while (skip(current_node)) {
        current_node = current_node->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
    }
    return current_node;
}

inline bool is_gna_non_functional_node(const std::shared_ptr<ngraph::Node>& node) {
    return std::dynamic_pointer_cast<ngraph::opset8::Reshape>(node) ||
           std::dynamic_pointer_cast<ngraph::opset8::Squeeze>(node) ||
           std::dynamic_pointer_cast<ngraph::opset8::Unsqueeze>(node) || is_trivial_transpose(node);
}

inline bool is_one_dim_shape(const ov::Shape& dims) {
    return std::count_if(std::begin(dims), std::end(dims), [](size_t dim) {
               return dim != 1;
           }) <= 1;
}

inline bool is_one_dim_shapes(const ov::Shape& in_dims, const ov::Shape& out_dims) {
    return is_one_dim_shape(in_dims) && is_one_dim_shape(out_dims);
}

inline bool is_power_activation(const ov::Node* node) noexcept {
    if (auto power_op = dynamic_cast<const ngraph::opset9::Power*>(node)) {
        auto const_node = std::dynamic_pointer_cast<ngraph::opset9::Constant>(power_op->get_input_node_shared_ptr(1));
        if (!const_node)
            return false;
        float value;
        if (!ov::op::util::get_single_value(const_node, value)) {
            return true;
        }
        return (1.0f != value);
    } else if (auto power_op = dynamic_cast<const ngraph::op::PowerIE*>(node)) {
        return (1.0f != power_op->power);
    }
    return false;
}

inline bool is_power_activation(const std::shared_ptr<ngraph::Node>& node) noexcept {
    return is_power_activation(node.get());
}

inline bool is_eltwise_mul(const std::shared_ptr<ngraph::Node>& node) {
    auto eltwise = std::dynamic_pointer_cast<ngraph::op::Eltwise>(node);
    if (!eltwise)
        return false;
    return eltwise->eltwise_type == ELTWISE_TYPE::Prod;
}

inline bool is_eltwise_add(const std::shared_ptr<ngraph::Node>& node) {
    auto eltwise = std::dynamic_pointer_cast<ngraph::op::Eltwise>(node);
    if (!eltwise)
        return false;
    return eltwise->eltwise_type == ELTWISE_TYPE::Sum;
}

inline bool is_pooling(const std::shared_ptr<ngraph::Node>& node) {
    return (std::dynamic_pointer_cast<ngraph::opset7::MaxPool>(node) != nullptr);
}

template <typename T>
inline bool is_Tbit_fq(const std::shared_ptr<ngraph::Node>& node) {
    auto fq_node = std::dynamic_pointer_cast<ngraph::opset9::FakeQuantize>(node);
    if (!fq_node)
        return false;
    auto levels = fq_node->get_levels();
    return (std::numeric_limits<T>::max() == levels) || (std::numeric_limits<T>::max() == levels - 1);
}

inline bool is_32bit_fq(const std::shared_ptr<ngraph::Node>& node) {
    return is_Tbit_fq<uint32_t>(node);
}

inline bool is_16bit_fq(const std::shared_ptr<ngraph::Node>& node) {
    return is_Tbit_fq<uint16_t>(node);
}

inline bool is_8bit_fq(const std::shared_ptr<ngraph::Node>& node) {
    return is_Tbit_fq<uint8_t>(node);
}

inline bool is_activation(const ov::Node* node) noexcept {
    return ((dynamic_cast<const ngraph::opset9::Clamp*>(node) != nullptr) ||
            (dynamic_cast<const ngraph::opset9::Sigmoid*>(node) != nullptr) ||
            (dynamic_cast<const ngraph::opset9::Relu*>(node) != nullptr) ||
            (dynamic_cast<const ngraph::op::ReLUIE*>(node) != nullptr) ||
            (dynamic_cast<const ngraph::opset9::Tanh*>(node) != nullptr) ||
            (dynamic_cast<const ngraph::opset9::PRelu*>(node) != nullptr) ||
            (dynamic_cast<const ngraph::opset9::Exp*>(node) != nullptr) ||
            (dynamic_cast<const ngraph::opset9::Log*>(node) != nullptr) ||
            (dynamic_cast<const ngraph::opset9::Sign*>(node) != nullptr) ||
            (dynamic_cast<const ngraph::opset9::Abs*>(node) != nullptr) ||
            (dynamic_cast<const ngraph::opset9::SoftSign*>(node) != nullptr) || is_power_activation(node) ||
            (dynamic_cast<const ngraph::opset9::FakeQuantize*>(node) != nullptr) ||
            (dynamic_cast<const ov::intel_gna::op::Pwl*>(node) != nullptr) ||
            (dynamic_cast<const ov::intel_gna::op::Identity*>(node) != nullptr));
}

inline bool is_activation(const std::shared_ptr<ngraph::Node>& node) noexcept {
    return is_activation(node.get());
}

inline bool is_gna_precision_agnostic(std::shared_ptr<ngraph::Node> node) {
    return ((std::dynamic_pointer_cast<ngraph::opset9::VariadicSplit>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ngraph::opset9::Split>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ngraph::opset9::Slice>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ngraph::opset9::Concat>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ngraph::opset9::Reshape>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ngraph::opset9::Squeeze>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ngraph::opset9::Unsqueeze>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ngraph::opset9::Transpose>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ov::intel_gna::op::Copy>(node) != nullptr) ||
            ((std::dynamic_pointer_cast<ngraph::op::CropIE>(node) != nullptr) && !is_crop_affined(node)));
}

inline bool has_8bit_or_16_bit_output(const std::shared_ptr<ngraph::Node>& node) noexcept {
    return ((ngraph::op::is_parameter(node)) || (ngraph::op::is_constant(node)) ||
            (std::dynamic_pointer_cast<ngraph::opset9::ReadValue>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ngraph::opset9::Assign>(node) != nullptr) ||
            (is_activation(node) && (!is_32bit_fq(node))) || (is_8bit_fq(node) || (is_16bit_fq(node))) ||
            is_gna_precision_agnostic(node));
}

inline bool has_32bit_output(const std::shared_ptr<ngraph::Node>& node) {
    return ((std::dynamic_pointer_cast<ngraph::op::FullyConnected>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ngraph::opset9::MatMul>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ngraph::opset9::Convolution>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ngraph::op::ConvolutionIE>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ngraph::opset9::Add>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ngraph::opset9::Multiply>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ngraph::op::Eltwise>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ngraph::op::ScaleShiftIE>(node) != nullptr) || is_pooling(node) ||
            ((std::dynamic_pointer_cast<ngraph::opset9::Power>(node) != nullptr) && !is_power_activation(node)) ||
            ((std::dynamic_pointer_cast<ngraph::op::PowerIE>(node) != nullptr) && !is_power_activation(node)) ||
            is_crop_affined(node) || is_32bit_fq(node));
}

inline bool has_32bit_input(const std::shared_ptr<ngraph::Node>& node) {
    return is_activation(node) || is_pooling(node);
}
}  // namespace ngraph_util
}  // namespace intel_gna
}  // namespace ov
