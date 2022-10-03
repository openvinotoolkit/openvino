// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ngraph_ops/crop_ie.hpp>
#include "gna_lib_ver_selector.hpp"
#include "backend/gna_limitations.hpp"
#include "layers/gna_permute.hpp"
#include <transformations/utils/utils.hpp>
#include <transformations/rt_info/gna_transpose_fusable.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <vector>
#include <memory>

namespace ov {
namespace intel_gna {
namespace ngraph_util {

template <typename T>
static bool get_constant_value(const std::shared_ptr<ngraph::opset8::Constant>& constant, std::vector<double>& values) {
    using A = typename ov::element_type_traits<T::value>::value_type;
    const auto& v = constant->get_vector<A>();
    std::copy(v.begin(), v.end(), std::back_inserter(values));
    return true;
}

static bool get_constant_value(std::tuple<>&&, const std::shared_ptr<ngraph::opset8::Constant>&, std::vector<double>&) {
    return false;
}

template<typename T, typename ...Types>
static bool get_constant_value(std::tuple<T, Types...>&&,
                  const std::shared_ptr<ngraph::opset8::Constant>& constant, std::vector<double>& values) {
    return constant->get_element_type() == T::value &&
           get_constant_value<T>(constant, values) ||
           get_constant_value(std::tuple<Types...>(), constant, values);
}

static bool get_constant_value(const std::shared_ptr<ngraph::opset8::Constant>& constant, std::vector<double>& values) {
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

static bool get_constant_value(const std::shared_ptr<ngraph::opset8::Constant>& constant, double& value) {
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

static bool is_aligned_split(const std::shared_ptr<ngraph::Node> input_op, size_t input_op_out_index) {
    size_t offset = 0;

    if (std::dynamic_pointer_cast<ngraph::opset8::Split>(input_op) || std::dynamic_pointer_cast<ngraph::opset8::VariadicSplit>(input_op)) {
        for (size_t index = 0; index < input_op_out_index; index++) {
            size_t outputSize = ngraph::shape_size(input_op->get_output_shape(index));
            offset += outputSize * GNAPluginNS::GNALimitations::bytesPerSplitElement;
        }
    }
    return (offset == ALIGN64(offset));
}

static bool is_crop_affined(std::shared_ptr<ngraph::Node> node) {
    auto crop = std::dynamic_pointer_cast<ngraph::op::CropIE>(node);
    if (crop != nullptr && !crop->offset.empty()) {
        return GNAPluginNS::GNALimitations::isCropAffinedOffset(crop->offset.back());
    }
    return false;
}

// this not only mathematically trivial
static bool is_trivial_transpose(std::shared_ptr<ngraph::Node> node) {
    auto transpose = std::dynamic_pointer_cast<ngraph::opset8::Transpose>(node);
    if (!transpose) return false;

    if (transpose->get_input_size() == 0)
        return false; // unsupported case

    if (ov::intel_gna::rt_info::is_transpose_fusable(transpose))
        return true;

    auto transpose_const = std::dynamic_pointer_cast<ngraph::op::Constant>(transpose->input_value(1).get_node_shared_ptr());
    if (!transpose_const) return false;

    auto node_order = transpose_const->cast_vector<int64_t>();

    auto input = transpose->input(0).get_source_output().get_node_shared_ptr();
    auto input_order = transpose->get_input_shape(0);

    return GNAPluginNS::isTrivialPermute(node_order, input_order);
}

inline std::shared_ptr<ov::Node> get_prev_node_skipping_certain(const std::shared_ptr<ngraph::Node>& node,
                                                                const std::function<bool(std::shared_ptr<ngraph::Node>)>& skip) {
    auto current_node = node;
    while (skip(current_node)) {
        current_node = current_node->get_input_node_shared_ptr(0);
    }
    return current_node;
}

inline std::shared_ptr<ov::Node> get_next_node_skipping_certain(const std::shared_ptr<ngraph::Node>& node,
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
            std::dynamic_pointer_cast<ngraph::opset8::Unsqueeze>(node) ||
            is_trivial_transpose(node);
}

inline bool is_one_dim_shape(const ov::Shape& dims) {
    return std::count_if(std::begin(dims), std::end(dims), [](size_t dim) { return dim != 1; }) <= 1;
}

inline bool is_one_dim_shapes(const ov::Shape& in_dims, const ov::Shape& out_dims) {
    return is_one_dim_shape(in_dims) && is_one_dim_shape(out_dims);
}

} // namespace ngraph_util
} // namespace intel_gna
} // namespace ov
