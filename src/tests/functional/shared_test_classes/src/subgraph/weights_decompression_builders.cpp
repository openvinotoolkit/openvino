// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/weights_decompression_builders.hpp"

#include "openvino/opsets/opset10.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "transformations/rt_info/decompression.hpp"

namespace ov {
namespace test {
std::ostream& operator<<(std::ostream& os, MatMulDecompressionShapeParams shape_params) {
    os << "data_shape=" << shape_params.data_shape << "_weights_shape=" << shape_params.weights_shape;
    if (shape_params.decompression_group_size != -1)
        os << "_group_size=" << shape_params.decompression_group_size;
    return os;
}

std::ostream& operator<<(std::ostream& os, GatherDecompressionShapeParams shape_params) {
    os << "data_shape=" << shape_params.data_shape << "_indices_shape=" << shape_params.indices_shape;
    if (shape_params.decompression_group_size != -1)
        os << "_group_size=" << shape_params.decompression_group_size;
    os << "_axis=" << shape_params.axis << "_batch_dims=" << shape_params.batch_dims;
    return os;
}

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

std::shared_ptr<ov::Node> initMatMulDecompressionSubgraph(
    const ov::Shape& weights_shape,
    const int group_size,
    const ov::element::Type data_precision,
    const ov::element::Type weights_precision,
    const ov::element::Type decompression_precision,
    const ov::element::Type scale_precision,
    const bool transpose_weights,
    const DecompressionType decompression_multiply_type,
    const DecompressionType decompression_subtract_type,
    const bool reshape_on_decompression_constant) {
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
    if (group_decompression) {
        OPENVINO_ASSERT(weights_shape[0] % group_size == 0,
                        "Weights output channels count (",
                        weights_shape[0],
                        ") must be divisible by decompression group size (",
                        group_size,
                        ").");
        auto in_channel_idx =
            transpose_weights ? transformed_weights_shape.size() - 1 : transformed_weights_shape.size() - 2;
        transformed_weights_shape[in_channel_idx] = weights_shape[0] / group_size;
        transformed_weights_shape.insert(transformed_weights_shape.begin() + in_channel_idx + 1, group_size);
    }

    auto up_to = weights_precision == ov::element::i4 ? 7 : 15;
    auto weights_tensor = ov::test::utils::create_and_fill_tensor(weights_precision,
                                                                  transformed_weights_shape,
                                                                  ov::test::utils::InputGenerateData(1, up_to));
    auto weights = std::make_shared<ov::op::v0::Constant>(weights_tensor);
    auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, decompression_precision);

    std::shared_ptr<ov::Node> mul_parent = weights_convert;
    auto output_channels = *weights_shape.rbegin();

    // Decompression constants shape:
    // Ordinary decompression: [O, 1]
    // Group decompression: [O, N, 1]
    ov::Shape scaleshift_target_shape{output_channels};
    scaleshift_target_shape.insert(scaleshift_target_shape.begin(),
                                    group_decompression ? weights_shape[0] / group_size : 1);
    scaleshift_target_shape = transpose_if_necessary(scaleshift_target_shape);
    if (group_decompression) {
        auto in_channel_idx =
            transpose_weights ? scaleshift_target_shape.size() - 1 : scaleshift_target_shape.size() - 2;
        scaleshift_target_shape.insert(scaleshift_target_shape.begin() + in_channel_idx + 1, 1);
    }

    auto scaleshift_const_shape = scaleshift_target_shape;
    if (reshape_on_decompression_constant)
        scaleshift_const_shape.erase(std::remove(scaleshift_const_shape.begin(), scaleshift_const_shape.end(), 1),
                                        scaleshift_const_shape.end());
    if (decompression_subtract_type != DecompressionType::empty) {
        auto subtract_shape =
            decompression_subtract_type == DecompressionType::full ? scaleshift_const_shape : ov::Shape({});
        auto shift_const_tensor = ov::test::utils::create_and_fill_tensor(weights_precision,
                                                                          subtract_shape,
                                                                          ov::test::utils::InputGenerateData(1, up_to));
        auto shift_const = std::make_shared<ov::op::v0::Constant>(shift_const_tensor);

        std::shared_ptr<ov::Node> shift_convert =
            std::make_shared<ov::op::v0::Convert>(shift_const, decompression_precision);
        if (reshape_on_decompression_constant) {
            auto subtract_target_shape = decompression_subtract_type == DecompressionType::full
                                                ? scaleshift_target_shape
                                                : ov::Shape(scaleshift_const_shape.size(), 1);
            auto shift_reshape_const = ov::opset10::Constant::create(ov::element::i32,
                                                                        {subtract_target_shape.size()},
                                                                        subtract_target_shape);
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
        auto scale_const_tensor = ov::test::utils::create_and_fill_tensor_real_distribution(scale_prc,
                                                                                            multiply_shape,
                                                                                            0.001f,
                                                                                            0.01f,
                                                                                            1);
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
            auto reshape_const = ov::opset10::Constant::create(ov::element::i32, {multiply_target_shape.size()}, multiply_target_shape);
            auto scale_reshape = std::make_shared<ov::opset10::Reshape>(scale_const, reshape_const, false);
            scale_const = scale_reshape;
        }
        last_node = std::make_shared<ov::opset10::Multiply>(mul_parent, scale_const);
    }

    if (group_decompression) {
        auto reshape_target_shape = transpose_weights ? std::vector<int>{-1, static_cast<int>(weights_shape[0])}
                                                        : std::vector<int>{static_cast<int>(weights_shape[0]), -1};
        auto target_shape_node =
            ov::opset10::Constant::create(ov::element::i32, {reshape_target_shape.size()}, reshape_target_shape);
        last_node = std::make_shared<ov::opset10::Reshape>(last_node, target_shape_node, false);
    }
    if (decompression_precision != data_precision) {
        last_node = std::make_shared<ov::opset10::Convert>(last_node, data_precision);
        ov::mark_as_decompression(last_node);
    }
    if (transpose_weights) {
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
        auto shift_tensor = ov::test::utils::create_and_fill_tensor(data_precision, shift_tensor_shape, ov::test::utils::InputGenerateData(0, up_to));
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

} // namespace test
} // namespace ov
