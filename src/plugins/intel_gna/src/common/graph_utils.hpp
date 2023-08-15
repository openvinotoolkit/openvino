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
#include "openvino/opsets/opset10.hpp"
#include "openvino/opsets/opset12.hpp"
#include "ops/copy.hpp"
#include "ops/identity.hpp"
#include "ops/pwl.hpp"
#include "transformations/rt_info/gna_transpose_fusable.hpp"
#include "transformations/utils/transformation_helper.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gna {
namespace graph_utils {

template <typename T>
inline bool get_constant_value(const std::shared_ptr<ngraph::opset8::Constant>& constant, std::vector<double>& values) {
    using A = typename ov::element_type_traits<T::value>::value_type;
    const auto& v = constant->get_vector<A>();
    std::transform(v.begin(), v.end(), std::back_inserter(values), [](A value) {
        return static_cast<double>(value);
    });
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

/**
 * @brief Checks if 2 shapes are the same
 */
inline bool are_shapes_equal(const ov::Shape& shape_1, const ov::Shape& shape_2) {
    return (shape_1.size() == shape_2.size()) && std::equal(shape_1.begin(), shape_1.end(), shape_2.begin());
}

inline bool is_aligned_split(const std::shared_ptr<ngraph::Node> input_op, size_t input_op_out_index) {
    size_t offset = 0;

    if (std::dynamic_pointer_cast<ngraph::opset8::Split>(input_op) ||
        std::dynamic_pointer_cast<ngraph::opset8::VariadicSplit>(input_op)) {
        for (size_t index = 0; index < input_op_out_index; index++) {
            size_t outputSize = ngraph::shape_size(input_op->get_output_shape(index));
            offset += outputSize * limitations::Limitations::kBytesPerSplitElement;
        }
    }
    return limitations::Limitations::get_instance()->is_aligned(offset);
}

inline bool is_crop_affined(std::shared_ptr<ngraph::Node> node) {
    auto crop = std::dynamic_pointer_cast<ngraph::op::CropIE>(node);
    if (crop != nullptr && !crop->offset.empty()) {
        return limitations::Limitations::get_instance()->is_crop_affined_offset(crop->offset.back());
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
    return ((std::dynamic_pointer_cast<ngraph::opset7::MaxPool>(node) != nullptr) ||
            std::dynamic_pointer_cast<ov::intel_gna::op::GNAMaxPool>(node) != nullptr);
}

inline bool is_concat(const std::shared_ptr<ngraph::Node>& node) {
    return (std::dynamic_pointer_cast<ov::opset12::Concat>(node) != nullptr);
}

inline bool is_fake_quantize(const std::shared_ptr<ngraph::Node>& node) {
    return (std::dynamic_pointer_cast<ov::opset12::FakeQuantize>(node) != nullptr);
}

inline bool is_read_value(const std::shared_ptr<ngraph::Node>& node) {
    return (std::dynamic_pointer_cast<ov::opset12::ReadValue>(node) != nullptr);
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

inline bool is_non_functional(const std::shared_ptr<ov::Node>& node) {
    return std::dynamic_pointer_cast<ov::opset12::Reshape>(node) != nullptr ||
           std::dynamic_pointer_cast<ov::opset12::Squeeze>(node) != nullptr ||
           std::dynamic_pointer_cast<ov::opset12::Unsqueeze>(node) != nullptr ||
           std::dynamic_pointer_cast<ov::opset12::FakeQuantize>(node) != nullptr;
}

inline bool is_copy(const std::shared_ptr<ov::Node>& node) {
    return std::dynamic_pointer_cast<ov::intel_gna::op::Copy>(node) != nullptr;
}

inline bool is_matmul(const std::shared_ptr<ov::Node>& node) {
    return std::dynamic_pointer_cast<ov::opset12::MatMul>(node) != nullptr;
}

inline bool is_fully_connected(const std::shared_ptr<ov::Node>& node) {
    return std::dynamic_pointer_cast<ngraph::op::FullyConnected>(node) != nullptr;
}

inline bool is_split(const std::shared_ptr<ov::Node>& node) {
    return std::dynamic_pointer_cast<ov::opset12::Split>(node) != nullptr ||
           std::dynamic_pointer_cast<ov::opset12::VariadicSplit>(node) != nullptr;
}

inline bool is_interleaved(const std::shared_ptr<ov::Node>& node) {
    return is_matmul(node) || is_fully_connected(node);
}

inline bool is_gna_precision_agnostic(std::shared_ptr<ngraph::Node> node) {
    return ((std::dynamic_pointer_cast<ngraph::opset9::VariadicSplit>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ngraph::opset9::Split>(node) != nullptr) ||
            (std::dynamic_pointer_cast<ngraph::opset9::Slice>(node) != nullptr) || is_concat(node) ||
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
            (std::dynamic_pointer_cast<ov::intel_gna::op::GNAConvolution>(node) != nullptr) ||
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

/**
 * @brief Remove all dimensions equal to 1 from the tensor shape vector
 * @param shape original tensor shape vector
 * @return modified shape
 */
inline ov::Shape squeeze_shape(const ov::Shape& shape) {
    ov::Shape squeezed_shape;
    squeezed_shape.reserve(shape.size());

    auto if_not_eq_1 = [](ov::Shape::value_type value) {
        return value != 1;
    };
    std::copy_if(shape.begin(), shape.end(), std::back_inserter(squeezed_shape), if_not_eq_1);

    return squeezed_shape;
}

/**
 * @brief Remove all dimensions equal to 1 from the left and right of the tensor shape vector
 * @param shape original tensor shape vector
 * @return modified shape
 */
inline ov::Shape trim_shape(const ov::Shape& shape) {
    auto comp = [](size_t x) {
        return x != 1;
    };

    auto start_it = std::find_if(shape.begin(), shape.end(), comp);
    auto end_it = std::find_if(shape.rbegin(), shape.rend(), comp);
    if (start_it == shape.end() || end_it == shape.rend()) {
        return ov::Shape(shape.begin(), shape.end());
    }
    return ov::Shape(start_it, end_it.base());
}

/**
 * @brief Transpose shape
 * @param shape the shape to be transposed
 * @param order the permutation array to apply to the input shape
 * @return transposed shape
 */
inline ov::Shape transpose_shape(const ov::Shape& shape, std::vector<size_t> order) {
    if (shape.size() != order.size()) {
        THROW_GNA_EXCEPTION << "Sizes of the shape " << shape.size() << " and transpose axis " << order.size()
                            << " are different";
    }
    ov::Shape transposed(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        transposed[i] = shape[order[i]];
    }
    return transposed;
}

/**
 * @brief Create gather indexes using transpose axes.
 * @param input_shape the shape to be transposed as gather
 * @param order the permutation array to apply to the input shape
 * @return vector with indexes to gather
 */
inline std::vector<size_t> make_gather_indexes_from_transpose_axes(const Shape& input_shape, const AxisVector& order) {
    // Supported shape ranks: 2d, 3d, 4d
    if (input_shape.size() < 2 || input_shape.size() > 4) {
        THROW_GNA_EXCEPTION << "Usupported shape size: " << input_shape.size();
    }

    ov::Shape input_shape_4d = input_shape;
    ov::AxisVector order_4d = order;
    // Just to simplify the code we transform all shapes to 4d by adding dimension(s) equal to 1 at the end
    while (input_shape_4d.size() < 4) {
        input_shape_4d.push_back(1);
        order_4d.push_back(order_4d.size());
    }
    ov::Shape output_shape_4d = transpose_shape(input_shape_4d, order_4d);

    // common case when shape is 4d
    std::vector<size_t> xyz_4d = {input_shape_4d[3] * input_shape_4d[2] * input_shape_4d[1],
                                  input_shape_4d[3] * input_shape_4d[2],
                                  input_shape_4d[3],
                                  1};

    std::vector<size_t> xyz = transpose_shape(xyz_4d, order_4d);
    std::vector<size_t> gather_order;

    for (size_t n = 0; n < output_shape_4d[0]; ++n) {
        for (size_t i = 0; i < output_shape_4d[1]; ++i) {
            for (size_t j = 0; j < output_shape_4d[2]; ++j) {
                for (size_t k = 0; k < output_shape_4d[3]; ++k) {
                    gather_order.push_back(n * xyz[0] + i * xyz[1] + j * xyz[2] + k * xyz[3]);
                }
            }
        }
    }

    return gather_order;
}

inline int64_t get_first_valuable_dim_id(const ov::Shape& shape) {
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] != 1) {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Converts Gather indexes into positive form
 */
template <typename T>
std::vector<T> normalize_gather_indices(const std::vector<T>& indices) {
    std::vector<T> normalized(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        T index = indices[i];
        if (index < 0)
            index += indices.size();
        normalized[i] = index;
    }
    return normalized;
}

/**
 * @brief Gets Gather indexes from Constant and converts them into positive form
 */
inline std::vector<int64_t> get_normalized_gather_indices(const std::shared_ptr<ov::opset12::Constant>& indices) {
    return normalize_gather_indices(indices->cast_vector<int64_t>());
}

/**
 * @brief Checks if node has dynamic rank inputs
 */
inline bool has_dynamic_rank_input(const std::shared_ptr<ov::Node>& node) {
    for (const auto& input_node : node->input_values()) {
        const Rank output_rank = input_node.get_partial_shape().rank();
        if (output_rank.is_dynamic())
            return true;
    }
    return false;
}

/**
 * @brief Gets maximum rank of all input nodes
 */
inline Rank::value_type get_max_input_rank(const std::shared_ptr<ov::Node>& node) {
    Rank::value_type max_input_rank = 0;
    for (auto& input_node : node->input_values()) {
        const Rank output_rank = input_node.get_partial_shape().rank();
        if (output_rank.is_dynamic())
            return -1;
        const Rank::value_type output_rank_len = output_rank.get_length();
        if (output_rank_len > max_input_rank)
            max_input_rank = output_rank_len;
    }
    return max_input_rank;
}

/**
 * @brief Gets dimension value by axis (works if axis could be < 0)
 */
inline Shape::value_type get_dim_by_axis(const Shape& shape, int64_t axis) {
    if (axis < 0)
        axis += static_cast<int64_t>(shape.size());
    if (axis < 0 || axis >= static_cast<int64_t>(shape.size()))
        throw std::runtime_error("get_dim_by_axis invalid axis");
    return shape[axis];
}

/**
 * @brief unsqueezes shape to rank
 */
inline Shape unsqueeze_shape(const Shape& shape, ov::Rank::value_type rank) {
    const ov::Rank::value_type rank_delta = rank - static_cast<ov::Rank::value_type>(shape.size());

    if (rank_delta <= 0)
        return shape;

    Shape broadcasted(rank);
    for (int i = 0; i < rank_delta; ++i) {
        broadcasted[i] = 1;
    }
    std::copy(shape.begin(), shape.end(), broadcasted.begin() + rank_delta);

    return broadcasted;
}

/**
 * @brief Converts axis to positive form
 */
inline int64_t convert_axis_to_positive(int64_t axis, ov::Rank rank) {
    const auto rank_val = rank.get_length();
    if (axis < 0)
        axis += rank_val;
    if (axis < 0 || axis >= rank_val)
        throw std::runtime_error("convert_axis_to_positive invalid axis");
    return axis;
}

/**
 * @brief Reverts gather indices in such a way that reverted and initial gather will do nothing if
 *   they stay one after another. Works only with positive form (no negative indices).
 */
inline std::vector<int64_t> reverse_gather_indexes(const std::vector<int64_t>& indexes) {
    std::vector<int64_t> out(indexes.size());
    for (size_t i = 0; i < indexes.size(); i++) {
        out.at(indexes[i]) = i;
    }
    return out;
}

/**
 * @brief Finds first consumer node
 */
inline Node* find_first_consumer(const std::shared_ptr<ov::Node>& node) {
    for (const auto& output : node->outputs()) {
        auto inputs = output.get_target_inputs();
        if (inputs.empty())
            continue;
        return inputs.begin()->get_node();
    }
    return nullptr;
}

/**
 * @brief Finds first input node with type NodeT
 */
template <typename NodeT>
std::shared_ptr<NodeT> find_first_input_node(const std::shared_ptr<ov::Node>& node) {
    for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        std::shared_ptr<ov::Node> input_node = node->get_input_node_shared_ptr(input_idx);
        auto target_node = ov::as_type_ptr<NodeT>(input_node);
        if (target_node)
            return target_node;
    }
    return {};
}

/**
 * @brief Gets split axis from Constant converting it to positive form
 */
inline bool get_split_axis(const std::shared_ptr<ov::opset12::Constant>& split_axis,
                           const ov::Rank& rank,
                           int64_t& axis) {
    auto split_axis_val = split_axis->cast_vector<int64_t>();
    if (split_axis_val.empty()) {
        return false;
    }
    axis = convert_axis_to_positive(split_axis_val[0], rank);
    return true;
}

/**
 * @brief Checks if has 2D input shape inputs
 */
inline bool has_2d_inputs(const ov::Output<ov::Node>& output) {
    auto node = output.get_node_shared_ptr();
    auto input_left_rank = node->get_input_partial_shape(0).rank();
    auto input_right_rank = node->get_input_partial_shape(0).rank();
    return (input_left_rank.is_static() && input_right_rank.is_static() && input_left_rank.get_length() == 2 &&
            input_right_rank.get_length() == 2);
}

/**
 * @brief Checks if the permutation does nothing
 */
inline bool is_pointless_permutation(const std::vector<int64_t>& indices) {
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] != static_cast<int64_t>(i))
            return false;
    }
    return true;
}

/**
 * @brief Checks if MatMul input with @arg input_idx input is transposed
 */
inline bool is_matmul_input_transposed(const std::shared_ptr<ov::opset12::MatMul>& matmul, size_t input_idx) {
    if (!input_idx)
        return matmul->get_transpose_a();
    return matmul->get_transpose_b();
}

/**
 * @brief Checks if Reshape node is Unsqueeze
 */
inline bool is_reshape_unsqueeze(const ov::Output<ov::Node>& output) {
    auto reshape = output.get_node_shared_ptr();
    const ov::Shape input_shape = trim_shape(reshape->get_input_shape(0));
    const ov::Shape output_shape = trim_shape(reshape->get_output_shape(0));
    return are_shapes_equal(input_shape, output_shape);
}

/**
 * @brief Checks if output has rank not more than expected
 */
inline std::function<bool(Output<Node>)> rank_not_more_than(const ov::Rank::value_type expected_rank) {
    return [=](Output<Node> output) -> bool {
        const Rank rank = output.get_partial_shape().rank();
        return (rank.is_static() && (rank.get_length() <= expected_rank));
    };
}

/**
 * @brief Checks if output has rank not more than expected
 */
inline bool constant_has_rank_not_more_than(const std::shared_ptr<ov::opset12::Constant>& node,
                                            const ov::Rank::value_type expected_rank) {
    const ov::Rank rank = node->get_output_partial_shape(0).rank();
    return (rank.is_static() && (rank.get_length() <= expected_rank));
}

/**
 * @brief Checks if output is Constant with rank 1
 */
inline bool is_constant_1d(const Output<Node>& output) {
    return ov::pass::pattern::rank_equals(0)(output) || ov::pass::pattern::rank_equals(1)(output);
}

/**
 * @brief Checks if node has parent node with type T
 */
template <typename T>
bool has_parent_node(std::shared_ptr<ov::Node> node) {
    for (const auto& parent : node->input_values()) {
        if (dynamic_cast<const T*>(parent.get_node()))
            return true;
    }
    return false;
}

/**
 * @brief Checks if node has child node with type T
 */
template <typename T>
bool has_child_node(std::shared_ptr<ov::Node> node) {
    for (size_t output_idx = 0; output_idx < node->get_output_size(); ++output_idx) {
        for (auto& input : node->get_output_target_inputs(output_idx)) {
            if (dynamic_cast<const T*>(input.get_node()))
                return true;
        }
    }
    return false;
}

/**
 * @brief Checks if shape without dimensions == 1 is 2D
 */
inline bool is_shape_2d(const ov::Shape& shape) {
    return graph_utils::squeeze_shape(shape).size() == 2;
}

/**
 * @brief Checks if node has N consumers
 */
inline bool has_n_consumers(const std::shared_ptr<ov::Node>& node, size_t n_consumers) {
    return node->output(0).get_target_inputs().size() == n_consumers;
}

/**
 * @brief Merge gather indexes.
 * @param ids_in vector with indexes to 1st gather
 * @param ids_out vector with indexes to 2nd gather
 * @return vector with indexes to merged gather
 */
inline std::vector<size_t> combine_gather_indexes(const std::vector<size_t>& ids_in,
                                                  const std::vector<size_t>& ids_out) {
    if (ids_in.size() != ids_out.size())
        return {};
    std::vector<size_t> result(ids_in.size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = ids_in[ids_out[i]];
    }
    return result;
}

}  // namespace graph_utils
}  // namespace intel_gna
}  // namespace ov
