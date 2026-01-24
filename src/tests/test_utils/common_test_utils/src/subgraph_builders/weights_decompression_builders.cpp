// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"

#include <algorithm>
#include <limits>

#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/opsets/opset10_decl.hpp"
#include "transformations/rt_info/decompression.hpp"

namespace ov {
namespace test {
namespace utils {

std::ostream& operator<<(std::ostream& os, DecompressionType type) {
    switch (type) {
    case DecompressionType::empty:
        os << "empty";
        break;
    case DecompressionType::scalar:
        os << "scalar";
        break;
    case DecompressionType::full:
        os << "full";
        break;
    default:
        OPENVINO_THROW("Not supported DecompressionType");
    }
    return os;
}

std::shared_ptr<ov::Node> initMatMulDecompressionSubgraph(const ov::Shape& weights_shape,
                                                          const int group_size,
                                                          const ov::element::Type data_precision,
                                                          const ov::element::Type weights_precision,
                                                          const ov::element::Type decompression_precision,
                                                          const ov::element::Type scale_precision,
                                                          const bool transpose_weights,
                                                          const DecompressionType decompression_multiply_type,
                                                          const DecompressionType decompression_subtract_type,
                                                          const bool reshape_on_decompression_constant,
                                                          const std::optional<bool>& insert_transpose_node,
                                                          const size_t seed) {
    auto transpose_if_necessary = [&](const ov::Shape& shape) {
        auto result_shape = shape;
        if (transpose_weights)
            std::swap(*result_shape.rbegin(), *(result_shape.rbegin() + 1));
        return result_shape;
    };

    const bool group_decompression = group_size != -1;
    // Weights has shape [I, O], where
    // I - input channels
    // O - output channels
    // In case of group decompression, input channels dimension is split into 2: I -> [N, G], where
    // N - number of groups
    // G - group size
    auto transformed_weights_shape = transpose_if_necessary(weights_shape);
    const auto IC = *(weights_shape.rbegin() + 1);
    if (group_decompression) {
        OPENVINO_ASSERT(IC % group_size == 0,
                        "Weights output channels count (",
                        IC,
                        ") must be divisible by decompression group size (",
                        group_size,
                        ").");
        auto in_channel_idx =
            transpose_weights ? transformed_weights_shape.size() - 1 : transformed_weights_shape.size() - 2;
        transformed_weights_shape[in_channel_idx] = IC / group_size;
        transformed_weights_shape.insert(transformed_weights_shape.begin() + in_channel_idx + 1, group_size);
    }

    auto up_to = weights_precision == ov::element::u2 ? 3 : weights_precision == ov::element::i4 ? 7 : 15;
    auto start_from = weights_precision == ov::element::u2 ? 0 : 1;
    auto weights_tensor =
        ov::test::utils::create_and_fill_tensor(weights_precision,
                                                transformed_weights_shape,
                                                ov::test::utils::InputGenerateData(start_from, up_to, 1, seed));
    auto weights = std::make_shared<ov::op::v0::Constant>(weights_tensor);

    auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, decompression_precision);

    std::shared_ptr<ov::Node> mul_parent = weights_convert;

    const auto OC = weights_shape.back();
    // Decompression constants shape:
    // Ordinary decompression: [O, 1]
    // Group decompression: [O, N, 1]
    ov::Shape scaleshift_target_shape{OC};
    scaleshift_target_shape.insert(scaleshift_target_shape.begin(), group_decompression ? IC / group_size : 1);
    scaleshift_target_shape = transpose_if_necessary(scaleshift_target_shape);
    if (group_decompression) {
        auto in_channel_idx =
            transpose_weights ? scaleshift_target_shape.size() - 1 : scaleshift_target_shape.size() - 2;
        scaleshift_target_shape.insert(scaleshift_target_shape.begin() + in_channel_idx + 1, 1);
    }
    const bool with_batch = weights_shape.size() == 3;
    if (with_batch) {
        scaleshift_target_shape.insert(scaleshift_target_shape.begin(), weights_shape[0]);
    }

    auto scaleshift_const_shape = scaleshift_target_shape;
    if (reshape_on_decompression_constant)
        scaleshift_const_shape.erase(std::remove(scaleshift_const_shape.begin(), scaleshift_const_shape.end(), 1),
                                     scaleshift_const_shape.end());
    if (decompression_subtract_type != DecompressionType::empty) {
        auto subtract_shape =
            decompression_subtract_type == DecompressionType::full ? scaleshift_const_shape : ov::Shape({});
        auto shift_const_tensor =
            ov::test::utils::create_and_fill_tensor(weights_precision,
                                                    subtract_shape,
                                                    ov::test::utils::InputGenerateData(start_from, up_to, 1, seed));
        auto shift_const = std::make_shared<ov::op::v0::Constant>(shift_const_tensor);

        std::shared_ptr<ov::Node> shift_convert =
            std::make_shared<ov::op::v0::Convert>(shift_const, decompression_precision);
        if (reshape_on_decompression_constant) {
            auto subtract_target_shape = decompression_subtract_type == DecompressionType::full
                                             ? scaleshift_target_shape
                                             : ov::Shape(scaleshift_const_shape.size(), 1);
            auto shift_reshape_const =
                ov::opset10::Constant::create(ov::element::i32, {subtract_target_shape.size()}, subtract_target_shape);
            auto shift_reshape = std::make_shared<ov::opset10::Reshape>(shift_convert, shift_reshape_const, false);
            shift_convert = shift_reshape;
        }
        mul_parent = std::make_shared<ov::opset10::Subtract>(weights_convert, shift_convert);
    }

    std::shared_ptr<ov::Node> last_node = mul_parent;
    const auto& scale_prc = scale_precision == ov::element::dynamic ? decompression_precision : scale_precision;
    if (decompression_multiply_type != DecompressionType::empty) {
        auto multiply_shape =
            decompression_multiply_type == DecompressionType::full ? scaleshift_const_shape : ov::Shape({});
        auto scale_const_tensor =
            ov::test::utils::create_and_fill_tensor_real_distribution(scale_prc, multiply_shape, 0.001f, 0.01f, seed);
        std::shared_ptr<ov::Node> scale_const = std::make_shared<ov::op::v0::Constant>(scale_const_tensor);

        if (scale_prc != decompression_precision) {
            const auto scale_convert = std::make_shared<ov::op::v0::Convert>(scale_const, decompression_precision);
            ov::mark_as_decompression(scale_convert);
            scale_const = scale_convert;
        }

        if (reshape_on_decompression_constant) {
            auto multiply_target_shape = decompression_multiply_type == DecompressionType::full
                                             ? scaleshift_target_shape
                                             : ov::Shape(scaleshift_const_shape.size(), 1);
            auto reshape_const =
                ov::opset10::Constant::create(ov::element::i32, {multiply_target_shape.size()}, multiply_target_shape);
            auto scale_reshape = std::make_shared<ov::opset10::Reshape>(scale_const, reshape_const, false);
            scale_const = scale_reshape;
        }
        last_node = std::make_shared<ov::opset10::Multiply>(mul_parent, scale_const);
    }

    if (group_decompression) {
        auto reshape_target_shape =
            transpose_weights ? std::vector<int>{-1, static_cast<int>(IC)} : std::vector<int>{static_cast<int>(IC), -1};
        if (with_batch)
            reshape_target_shape.insert(reshape_target_shape.begin(), weights_shape[0]);
        auto target_shape_node =
            ov::opset10::Constant::create(ov::element::i32, {reshape_target_shape.size()}, reshape_target_shape);
        last_node = std::make_shared<ov::opset10::Reshape>(last_node, target_shape_node, false);
    }
    if (decompression_precision != data_precision) {
        last_node = std::make_shared<ov::opset10::Convert>(last_node, data_precision);
        ov::mark_as_decompression(last_node);
    }
    const bool insert_transpose = insert_transpose_node.value_or(transpose_weights);
    if (insert_transpose) {
        const size_t rank = last_node->get_output_partial_shape(0).size();
        std::vector<int> order(rank);
        std::iota(order.begin(), order.end(), 0);
        std::swap(*order.rbegin(), *(order.rbegin() + 1));
        auto transpose_constant = ov::opset10::Constant::create(ov::element::i32, {rank}, order);
        last_node = std::make_shared<ov::opset10::Transpose>(last_node, transpose_constant);
    }
    return last_node;
}

std::shared_ptr<ov::Node> initMatMulDecompressionSubgraphQuantization(
    const ov::Shape& weights_shape,
    const int group_size,
    const ov::element::Type data_precision,
    const ov::element::Type weights_precision,
    const ov::element::Type decompression_precision,
    const ov::element::Type scale_precision,
    const bool transpose_weights,
    const DecompressionType decompression_multiply_type,
    const DecompressionType decompression_subtract_type,
    const bool reshape_on_decompression_constant,
    const std::optional<bool>& insert_transpose_node,
    const size_t seed) {
    // Validate that the precision is supported for real quantization
    OPENVINO_ASSERT(weights_precision == ov::element::u2 || weights_precision == ov::element::u4 ||
                        weights_precision == ov::element::i4 || weights_precision == ov::element::u8 ||
                        weights_precision == ov::element::i8,
                    "initMatMulDecompressionSubgraphQuantization only supports u2, u4, i4, u8, i8 precisions. Got: ",
                    weights_precision);

    auto transpose_if_necessary = [&](const ov::Shape& shape) {
        auto result_shape = shape;
        if (transpose_weights)
            std::swap(*result_shape.rbegin(), *(result_shape.rbegin() + 1));
        return result_shape;
    };

    const bool group_decompression = group_size != -1;
    // Weights has shape [I, O], where
    // I - input channels
    // O - output channels
    // In case of group decompression, input channels dimension is split into 2: I -> [N, G], where
    // N - number of groups
    // G - group size
    auto transformed_weights_shape = transpose_if_necessary(weights_shape);
    const auto IC = *(weights_shape.rbegin() + 1);
    if (group_decompression) {
        OPENVINO_ASSERT(IC % group_size == 0,
                        "Weights output channels count (",
                        IC,
                        ") must be divisible by decompression group size (",
                        group_size,
                        ").");
        auto in_channel_idx =
            transpose_weights ? transformed_weights_shape.size() - 1 : transformed_weights_shape.size() - 2;
        transformed_weights_shape[in_channel_idx] = IC / group_size;
        transformed_weights_shape.insert(transformed_weights_shape.begin() + in_channel_idx + 1, group_size);
    }

    // Step 1: Generate FP32 weights for real quantization
    auto fp32_weights_tensor =
        ov::test::utils::create_and_fill_tensor(ov::element::f32,
                                                transformed_weights_shape,
                                                ov::test::utils::InputGenerateData(1, 6, 8, seed));

    // Calculate quantization parameters
    const auto qmin = weights_precision == ov::element::u2   ? 0.0f
                      : weights_precision == ov::element::i4 ? -8.0f
                      : weights_precision == ov::element::u4 ? 0.0f
                      : weights_precision.is_signed()        ? -128.0f
                                                             : 0.0f;
    const auto qmax = weights_precision == ov::element::u2   ? 3.0f
                      : weights_precision == ov::element::i4 ? 7.0f
                      : weights_precision == ov::element::u4 ? 15.0f
                      : weights_precision.is_signed()        ? 127.0f
                                                             : 255.0f;

    auto* fp32_data = fp32_weights_tensor.data<float>();
    const size_t total_size = ov::shape_size(transformed_weights_shape);

    const auto OC = weights_shape.back();
    // Decompression constants shape:
    // Ordinary decompression: [O, 1]
    // Group decompression: [O, N, 1]
    ov::Shape scaleshift_target_shape{OC};
    scaleshift_target_shape.insert(scaleshift_target_shape.begin(), group_decompression ? IC / group_size : 1);
    scaleshift_target_shape = transpose_if_necessary(scaleshift_target_shape);
    if (group_decompression) {
        auto in_channel_idx =
            transpose_weights ? scaleshift_target_shape.size() - 1 : scaleshift_target_shape.size() - 2;
        scaleshift_target_shape.insert(scaleshift_target_shape.begin() + in_channel_idx + 1, 1);
    }
    const bool with_batch = weights_shape.size() == 3;
    if (with_batch) {
        scaleshift_target_shape.insert(scaleshift_target_shape.begin(), weights_shape[0]);
    }

    auto scaleshift_const_shape = scaleshift_target_shape;
    if (reshape_on_decompression_constant)
        scaleshift_const_shape.erase(std::remove(scaleshift_const_shape.begin(), scaleshift_const_shape.end(), 1),
                                     scaleshift_const_shape.end());

    // Step 2: Compute quantization parameters (scales and zero points) based on actual FP32 data
    const auto& scale_prc = scale_precision == ov::element::dynamic ? decompression_precision : scale_precision;

    // Determine number of quantization groups
    size_t num_groups = 1;
    if (decompression_multiply_type == DecompressionType::full ||
        decompression_subtract_type == DecompressionType::full) {
        num_groups = ov::shape_size(scaleshift_const_shape);
    }

    // Calculate group size for quantization
    size_t elements_per_group = total_size / num_groups;

    // Compute scales and zero points per group
    std::vector<float> scales(num_groups);
    std::vector<float> zero_points(num_groups);

    for (size_t g = 0; g < num_groups; ++g) {
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();

        // Find min/max in the current group
        for (size_t i = 0; i < elements_per_group; ++i) {
            float val = fp32_data[g * elements_per_group + i];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }

        // Compute scale and zero point
        float range = max_val - min_val;
        if (range < 1e-6f) {
            range = 1e-6f;  // Avoid division by zero
        }

        scales[g] = range / (qmax - qmin);
        float zp = std::round(-min_val / scales[g] + qmin);
        // Clamp zero point to valid quantization range
        zero_points[g] = std::max(qmin, std::min(qmax, zp));
    }

    // Step 3: Quantize the weights
    ov::Tensor quantized_weights_tensor(weights_precision, transformed_weights_shape);

    auto quantize_value = [&](float val, float scale, float zp) -> float {
        float quantized = std::round(val / scale + zp);
        return std::max(qmin, std::min(qmax, quantized));
    };

    // Quantize based on weights_precision type
    if (weights_precision == ov::element::u2 || weights_precision == ov::element::u4) {
        // For sub-byte unsigned types, use element::iterator
        std::vector<uint8_t> quantized_buffer(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            size_t group_idx = (num_groups == 1) ? 0 : (i / elements_per_group);
            quantized_buffer[i] =
                static_cast<uint8_t>(quantize_value(fp32_data[i], scales[group_idx], zero_points[group_idx]));
        }
        if (weights_precision == ov::element::u4) {
            auto iter = ov::element::iterator<ov::element::u4>(quantized_weights_tensor.data());
            std::copy(quantized_buffer.begin(), quantized_buffer.end(), iter);
        } else {
            auto iter = ov::element::iterator<ov::element::u2>(quantized_weights_tensor.data());
            std::copy(quantized_buffer.begin(), quantized_buffer.end(), iter);
        }
    } else if (weights_precision == ov::element::i4) {
        // For sub-byte signed types, use element::iterator
        std::vector<int8_t> quantized_buffer(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            size_t group_idx = (num_groups == 1) ? 0 : (i / elements_per_group);
            quantized_buffer[i] =
                static_cast<int8_t>(quantize_value(fp32_data[i], scales[group_idx], zero_points[group_idx]));
        }
        auto iter = ov::element::iterator<ov::element::i4>(quantized_weights_tensor.data());
        std::copy(quantized_buffer.begin(), quantized_buffer.end(), iter);
    } else if (weights_precision == ov::element::u8) {
        auto* q_data = quantized_weights_tensor.data<uint8_t>();
        for (size_t i = 0; i < total_size; ++i) {
            size_t group_idx = (num_groups == 1) ? 0 : (i / elements_per_group);
            q_data[i] = static_cast<uint8_t>(quantize_value(fp32_data[i], scales[group_idx], zero_points[group_idx]));
        }
    } else if (weights_precision == ov::element::i8) {
        auto* q_data = quantized_weights_tensor.data<int8_t>();
        for (size_t i = 0; i < total_size; ++i) {
            size_t group_idx = (num_groups == 1) ? 0 : (i / elements_per_group);
            q_data[i] = static_cast<int8_t>(quantize_value(fp32_data[i], scales[group_idx], zero_points[group_idx]));
        }
    }

    // Step 4: Create constants with quantized data
    auto weights = std::make_shared<ov::op::v0::Constant>(quantized_weights_tensor);
    auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, decompression_precision);

    std::shared_ptr<ov::Node> mul_parent = weights_convert;

    // Create zero point (shift) constants if needed
    if (decompression_subtract_type != DecompressionType::empty) {
        auto subtract_shape =
            decompression_subtract_type == DecompressionType::full ? scaleshift_const_shape : ov::Shape({});

        ov::Tensor shift_tensor(weights_precision, subtract_shape);
        if (decompression_subtract_type == DecompressionType::scalar) {
            // Use average zero point for scalar
            float avg_zp = 0.0f;
            for (size_t g = 0; g < num_groups; ++g) {
                avg_zp += zero_points[g];
            }
            avg_zp /= num_groups;

            // Use element iterator for sub-byte types
            if (weights_precision == ov::element::u4) {
                auto iter = ov::element::iterator<ov::element::u4>(shift_tensor.data());
                *iter = static_cast<uint8_t>(std::round(avg_zp));
            } else if (weights_precision == ov::element::u2) {
                auto iter = ov::element::iterator<ov::element::u2>(shift_tensor.data());
                *iter = static_cast<uint8_t>(std::round(avg_zp));
            } else if (weights_precision == ov::element::i4) {
                auto iter = ov::element::iterator<ov::element::i4>(shift_tensor.data());
                *iter = static_cast<int8_t>(std::round(avg_zp));
            } else if (weights_precision.is_signed()) {
                shift_tensor.data<int8_t>()[0] = static_cast<int8_t>(std::round(avg_zp));
            } else {
                shift_tensor.data<uint8_t>()[0] = static_cast<uint8_t>(std::round(avg_zp));
            }
        } else {
            // Full zero points
            if (weights_precision == ov::element::u4) {
                std::vector<uint8_t> zp_buffer(num_groups);
                for (size_t i = 0; i < num_groups; ++i) {
                    zp_buffer[i] = static_cast<uint8_t>(std::round(zero_points[i]));
                }
                auto iter = ov::element::iterator<ov::element::u4>(shift_tensor.data());
                std::copy(zp_buffer.begin(), zp_buffer.end(), iter);
            } else if (weights_precision == ov::element::u2) {
                std::vector<uint8_t> zp_buffer(num_groups);
                for (size_t i = 0; i < num_groups; ++i) {
                    zp_buffer[i] = static_cast<uint8_t>(std::round(zero_points[i]));
                }
                auto iter = ov::element::iterator<ov::element::u2>(shift_tensor.data());
                std::copy(zp_buffer.begin(), zp_buffer.end(), iter);
            } else if (weights_precision == ov::element::i4) {
                std::vector<int8_t> zp_buffer(num_groups);
                for (size_t i = 0; i < num_groups; ++i) {
                    zp_buffer[i] = static_cast<int8_t>(std::round(zero_points[i]));
                }
                auto iter = ov::element::iterator<ov::element::i4>(shift_tensor.data());
                std::copy(zp_buffer.begin(), zp_buffer.end(), iter);
            } else if (weights_precision.is_signed()) {
                auto* shift_data = shift_tensor.data<int8_t>();
                for (size_t i = 0; i < num_groups; ++i) {
                    shift_data[i] = static_cast<int8_t>(std::round(zero_points[i]));
                }
            } else {
                auto* shift_data = shift_tensor.data<uint8_t>();
                for (size_t i = 0; i < num_groups; ++i) {
                    shift_data[i] = static_cast<uint8_t>(std::round(zero_points[i]));
                }
            }
        }

        auto shift_const = std::make_shared<ov::op::v0::Constant>(shift_tensor);
        std::shared_ptr<ov::Node> shift_convert =
            std::make_shared<ov::op::v0::Convert>(shift_const, decompression_precision);
        if (reshape_on_decompression_constant) {
            auto subtract_target_shape = decompression_subtract_type == DecompressionType::full
                                             ? scaleshift_target_shape
                                             : ov::Shape(scaleshift_const_shape.size(), 1);
            auto shift_reshape_const =
                ov::opset10::Constant::create(ov::element::i32, {subtract_target_shape.size()}, subtract_target_shape);
            auto shift_reshape = std::make_shared<ov::opset10::Reshape>(shift_convert, shift_reshape_const, false);
            shift_convert = shift_reshape;
        }
        mul_parent = std::make_shared<ov::opset10::Subtract>(weights_convert, shift_convert);
    }

    // Create scale constants
    if (decompression_multiply_type != DecompressionType::empty) {
        auto multiply_shape =
            decompression_multiply_type == DecompressionType::full ? scaleshift_const_shape : ov::Shape({});

        ov::Tensor scale_tensor;
        if (decompression_multiply_type == DecompressionType::scalar) {
            scale_tensor = ov::Tensor(scale_prc, multiply_shape);
            // Use average scale for scalar
            float avg_scale = 0.0f;
            for (size_t g = 0; g < num_groups; ++g) {
                avg_scale += scales[g];
            }
            avg_scale /= num_groups;

            if (scale_prc == ov::element::f16) {
                scale_tensor.data<ov::float16>()[0] = ov::float16(avg_scale);
            } else {
                scale_tensor.data<float>()[0] = avg_scale;
            }
        } else {
            // Full scales
            scale_tensor = ov::Tensor(scale_prc, multiply_shape);
            if (scale_prc == ov::element::f16) {
                auto* scale_data = scale_tensor.data<ov::float16>();
                for (size_t i = 0; i < num_groups; ++i) {
                    scale_data[i] = ov::float16(scales[i]);
                }
            } else {
                auto* scale_data = scale_tensor.data<float>();
                for (size_t i = 0; i < num_groups; ++i) {
                    scale_data[i] = scales[i];
                }
            }
        }

        std::shared_ptr<ov::Node> scale_const = std::make_shared<ov::op::v0::Constant>(scale_tensor);

        if (scale_prc != decompression_precision) {
            const auto scale_convert = std::make_shared<ov::op::v0::Convert>(scale_const, decompression_precision);
            ov::mark_as_decompression(scale_convert);
            scale_const = scale_convert;
        }

        if (reshape_on_decompression_constant) {
            auto multiply_target_shape = decompression_multiply_type == DecompressionType::full
                                             ? scaleshift_target_shape
                                             : ov::Shape(scaleshift_const_shape.size(), 1);
            auto reshape_const =
                ov::opset10::Constant::create(ov::element::i32, {multiply_target_shape.size()}, multiply_target_shape);
            auto scale_reshape = std::make_shared<ov::opset10::Reshape>(scale_const, reshape_const, false);
            scale_const = scale_reshape;
        }
        mul_parent = std::make_shared<ov::opset10::Multiply>(mul_parent, scale_const);
    }

    std::shared_ptr<ov::Node> last_node = mul_parent;

    if (group_decompression) {
        auto reshape_target_shape =
            transpose_weights ? std::vector<int>{-1, static_cast<int>(IC)} : std::vector<int>{static_cast<int>(IC), -1};
        if (with_batch)
            reshape_target_shape.insert(reshape_target_shape.begin(), weights_shape[0]);
        auto target_shape_node =
            ov::opset10::Constant::create(ov::element::i32, {reshape_target_shape.size()}, reshape_target_shape);
        last_node = std::make_shared<ov::opset10::Reshape>(last_node, target_shape_node, false);
    }
    if (decompression_precision != data_precision) {
        last_node = std::make_shared<ov::opset10::Convert>(last_node, data_precision);
        ov::mark_as_decompression(last_node);
    }
    const bool insert_transpose = insert_transpose_node.value_or(transpose_weights);
    if (insert_transpose) {
        const size_t rank = last_node->get_output_partial_shape(0).size();
        std::vector<int> order(rank);
        std::iota(order.begin(), order.end(), 0);
        std::swap(*order.rbegin(), *(order.rbegin() + 1));
        auto transpose_constant = ov::opset10::Constant::create(ov::element::i32, {rank}, order);
        last_node = std::make_shared<ov::opset10::Transpose>(last_node, transpose_constant);
    }
    return last_node;
}

std::shared_ptr<ov::Node> initGatherDecompressionSubgraph(const ov::Shape& data_shape,
                                                          const int group_size,
                                                          const ov::element::Type data_precision,
                                                          const ov::element::Type output_precision,
                                                          const bool add_subtract,
                                                          const bool reshape_on_decompression_constant,
                                                          const bool per_tensor_zp,
                                                          const bool per_tensor_scale) {
    const bool group_decompression = group_size != -1;
    // Weights has shape [I, D], where
    // I - index
    // D - data
    // In case of group decompression, data dimension is split into 2: I -> [N, G], where
    // N - number of groups
    // G - group size
    auto original_data_shape = data_shape;
    if (group_decompression) {
        OPENVINO_ASSERT(data_shape[1] % group_size == 0,
                        "The last data dimension (",
                        data_shape[1],
                        ") must be divisible by decompression group size (",
                        group_size,
                        ").");
        auto data_idx = data_shape.size() - 1;
        original_data_shape[data_idx] = data_shape[1] / group_size;
        original_data_shape.insert(original_data_shape.begin() + data_idx + 1, group_size);
    }

    const auto up_to = data_precision == ov::element::i4 ? 7 : 15;
    ov::test::utils::InputGenerateData generate_data(0, up_to);
    if (data_precision.is_signed())
        generate_data.start_from = -1;
    auto weights_tensor = ov::test::utils::create_and_fill_tensor(data_precision, original_data_shape, generate_data);
    auto weights = std::make_shared<ov::op::v0::Constant>(weights_tensor);
    weights->set_friendly_name("Compressed_weights");
    auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, output_precision);

    std::shared_ptr<ov::Node> mul_parent = weights_convert;

    // Decompression constants shape:
    // Ordinary decompression: [I, 1]
    // Group decompression: [I, N, 1]
    ov::Shape scaleshift_target_shape{data_shape[0]};
    scaleshift_target_shape.insert(scaleshift_target_shape.end(), group_decompression ? data_shape[1] / group_size : 1);
    if (group_decompression || scaleshift_target_shape.size() < original_data_shape.size()) {
        auto data_idx = scaleshift_target_shape.size() - 1;
        scaleshift_target_shape.insert(scaleshift_target_shape.begin() + data_idx + 1, 1);
    }

    auto scaleshift_const_shape = scaleshift_target_shape;
    if (reshape_on_decompression_constant)
        scaleshift_const_shape.erase(std::remove(scaleshift_const_shape.begin(), scaleshift_const_shape.end(), 1),
                                     scaleshift_const_shape.end());
    if (add_subtract) {
        auto shift_tensor_shape = per_tensor_zp ? ov::Shape{1} : scaleshift_const_shape;
        auto shift_tensor = ov::test::utils::create_and_fill_tensor(data_precision,
                                                                    shift_tensor_shape,
                                                                    ov::test::utils::InputGenerateData(0, up_to));
        if (per_tensor_zp && data_precision.bitwidth() == 4) {
            static_cast<uint8_t*>(shift_tensor.data())[0] = 0x88;
        }
        auto shift_const = std::make_shared<ov::op::v0::Constant>(shift_tensor);
        std::shared_ptr<ov::Node> shift_convert = std::make_shared<ov::op::v0::Convert>(shift_const, output_precision);
        if (reshape_on_decompression_constant && !per_tensor_zp) {
            auto shift_reshape_const = ov::op::v0::Constant::create(ov::element::i32,
                                                                    {scaleshift_target_shape.size()},
                                                                    scaleshift_target_shape);
            auto shift_reshape = std::make_shared<ov::op::v1::Reshape>(shift_convert, shift_reshape_const, false);
            shift_convert = shift_reshape;
        }
        mul_parent = std::make_shared<ov::op::v1::Subtract>(weights_convert, shift_convert);
    }

    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = -0.5;
    in_data.range = 1;
    in_data.resolution = 30000;
    auto scale_tensor_shape = per_tensor_scale ? ov::Shape{1} : scaleshift_const_shape;
    auto scale_tensor = ov::test::utils::create_and_fill_tensor(output_precision, scale_tensor_shape, in_data);
    for (size_t i = 0; i < scale_tensor.get_size(); i++) {
        if (output_precision == ov::element::f16)
            scale_tensor.data<ov::float16>()[i] /= ov::float16(16.f);
        else if (output_precision == ov::element::f32)
            scale_tensor.data<float>()[i] /= 16.f;
    }
    std::shared_ptr<ov::Node> scale_const = std::make_shared<ov::op::v0::Constant>(scale_tensor);
    if (reshape_on_decompression_constant && !per_tensor_scale) {
        auto scale_reshape_const =
            ov::op::v0::Constant::create(ov::element::i32, {scaleshift_target_shape.size()}, scaleshift_target_shape);
        auto scale_reshape = std::make_shared<ov::op::v1::Reshape>(scale_const, scale_reshape_const, false);
        scale_const = scale_reshape;
    }
    std::shared_ptr<ov::Node> last_node = std::make_shared<ov::op::v1::Multiply>(mul_parent, scale_const);

    if (group_decompression) {
        auto reshape_target_shape = std::vector<int>{static_cast<int>(data_shape[0]), -1};
        auto target_shape_node =
            ov::op::v0::Constant::create(ov::element::i32, {reshape_target_shape.size()}, reshape_target_shape);
        last_node = std::make_shared<ov::op::v1::Reshape>(last_node, target_shape_node, false);
    }
    return last_node;
}

}  // namespace utils
}  // namespace test
}  // namespace ov