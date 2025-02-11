// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/decompression.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"

namespace {
using ov::test::InputShape;

struct ShapeParams {
    ShapeParams() = default;
    ShapeParams(InputShape data_shape, ov::Shape weights_shape, int weights_group_size = -1)
        : data_shape(std::move(data_shape)),
          weights_shape(std::move(weights_shape)),
          weights_group_size(weights_group_size) {}

    InputShape data_shape;
    ov::Shape weights_shape;
    // Decompression group size. If the value is equal to -1, ordinary decompression is used
    int weights_group_size;
};

const std::vector<ov::element::Type> activations_precisions = {ov::element::f32, ov::element::f16};
const std::vector<ov::element::Type> weights_precisions = {ov::element::u8, ov::element::u4, ov::element::i4};
const std::vector<bool> transpose_weights = {true, false};

const std::vector<ShapeParams> input_shapes_basic = {
    {{{-1, -1, -1}, {{1, 8, 4}, {1, 8, 4}}}, {4, 4}},
};

const std::vector<bool> add_decompression_sub = {true, false};
const std::vector<bool> reshape_on_decompression = {true, false};
const std::vector<bool> per_tensor_zp = {true, false};

using MatmulWeightsDecompressionQuantizeConvolutionParams = std::tuple<ShapeParams,   // input shapes
                                                            ov::element::Type,        // weights type
                                                            ov::element::Type,        // activations type
                                                            bool,                     // transpose on weights
                                                            bool,                     // decompression subtract
                                                            bool,                     // reshape on decompression constants
                                                            bool,                     // extra multiply
                                                            bool,                     // per-tensor zero-point
                                                            uint64_t                  // dynamic_quantization_group_size
                                                            >;

class MatmulWeightsDecompressionQuantizeConvolution : public testing::WithParamInterface<MatmulWeightsDecompressionQuantizeConvolutionParams>,
                                                      virtual public ov::test::SubgraphBaseTest {
public:
    static std::string get_test_case_name(testing::TestParamInfo<MatmulWeightsDecompressionQuantizeConvolutionParams> obj) {
        ShapeParams shape_params;
        ov::element::Type weights_precision;
        ov::element::Type activations_precision;
        bool transpose;
        bool decompression_sub;
        bool reshape_on_decompression;
        bool extra_multiply;
        bool per_tensor_zp;
        uint64_t dyn_quan_group_size;

        std::tie(shape_params,
                 weights_precision,
                 activations_precision,
                 transpose,
                 decompression_sub,
                 reshape_on_decompression,
                 extra_multiply,
                 per_tensor_zp,
                 dyn_quan_group_size) = obj.param;

        std::ostringstream result;
        result << "data_shape=";
        result << ov::test::utils::partialShape2str({shape_params.data_shape.first}) << "_";
        for (const auto& actual_shape : shape_params.data_shape.second) {
            result << ov::test::utils::partialShape2str({actual_shape}) << "_";
        }
        result << "_" << "weights_shape=" << shape_params.weights_shape << "_";
        result << "group_size=" << (shape_params.weights_group_size == -1 ? 1111 : shape_params.weights_group_size) << "_";
        result << "weights_precision=" << weights_precision << "_";
        result << "activations_precision=" << activations_precision << "_";
        result << "transpose_weights=" << transpose << "_";
        result << "decompression_subtract=" << decompression_sub << "_";
        result << "reshape_on_decompression=" << reshape_on_decompression << "_";
        result << "extra_multiply=" << extra_multiply << "_";
        result << "per_tensor_zp=" << per_tensor_zp << "_";
        result << "dyn_quan_group_size=" << dyn_quan_group_size << "_";
        result << "quantize_conv";

        return result.str();
    }
protected:
    std::shared_ptr<ov::Model> init_subgraph(const ov::PartialShape& data_shape,
                                              const ov::Shape& weights_shape,
                                              const int group_size,
                                              const ov::element::Type data_precision,
                                              const ov::element::Type weights_precision,
                                              const bool transpose_weights,
                                              const bool add_subtract,
                                              const bool reshape_on_decompression,
                                              const bool extra_multiply,
                                              const bool per_tensor_zp) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(data_precision, data_shape)};
        const auto weights_subgraph = init_compressed_weights_subgraph(weights_shape,
                                                                       group_size,
                                                                       data_precision,
                                                                       weights_precision,
                                                                       transpose_weights,
                                                                       add_subtract,
                                                                       reshape_on_decompression,
                                                                       extra_multiply,
                                                                       per_tensor_zp);

        auto mat_mul = std::make_shared<ov::op::v0::MatMul>(params[0], weights_subgraph);

        std::vector<int> shape_pattern = {1, -1, 2, 4};
        auto shapePatternsNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape({4}), shape_pattern);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(mat_mul, shapePatternsNode, false);
        reshape->set_friendly_name("reshape");

        auto conv = init_quantized_convolution_subgraph(reshape);
        return std::make_shared<ov::Model>(ov::NodeVector{conv}, params, "MatmulWeightsDecompressionQuantizeConvolution");
    }

    std::shared_ptr<ov::Node> init_compressed_weights_subgraph(const ov::Shape& weights_shape,
                                                               const int group_size,
                                                               const ov::element::Type data_precision,
                                                               const ov::element::Type weights_precision,
                                                               const bool transpose_weights,
                                                               const bool add_subtract,
                                                               const bool reshape_on_decompression_constant,
                                                               const bool extra_multiply,
                                                               const bool per_tensor_zp) {
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
            auto in_channel_idx = transpose_weights ? transformed_weights_shape.size() - 1 : transformed_weights_shape.size() - 2;
            transformed_weights_shape[in_channel_idx] = weights_shape[0] / group_size;
            transformed_weights_shape.insert(transformed_weights_shape.begin() + in_channel_idx + 1, group_size);
        }
        auto weights_tensor = ov::test::utils::create_and_fill_tensor(weights_precision, transformed_weights_shape);
        auto weights = std::make_shared<ov::op::v0::Constant>(weights_tensor);
        weights->set_friendly_name("Compressed_weights");
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, data_precision);

        std::shared_ptr<ov::Node> mul_parent = weights_convert;
        auto output_channels = *weights_shape.rbegin();

        // Decompression constants shape:
        // Ordinary decompression: [O, 1]
        // Group decompression: [O, N, 1]
        ov::Shape scaleshift_target_shape{output_channels};
        scaleshift_target_shape.insert(scaleshift_target_shape.begin(), group_decompression ? weights_shape[0] / group_size : 1);
        scaleshift_target_shape = transpose_if_necessary(scaleshift_target_shape);
        if (group_decompression) {
            auto in_channel_idx = transpose_weights ? scaleshift_target_shape.size() - 1 : scaleshift_target_shape.size() - 2;
            scaleshift_target_shape.insert(scaleshift_target_shape.begin() + in_channel_idx + 1, 1);
        }

        auto scaleshift_const_shape = scaleshift_target_shape;
        if (reshape_on_decompression_constant)
            scaleshift_const_shape.erase(std::remove(scaleshift_const_shape.begin(), scaleshift_const_shape.end(), 1), scaleshift_const_shape.end());
        if (add_subtract) {
            auto shift_tensor_shape = per_tensor_zp ? ov::Shape{1} : scaleshift_const_shape;
            auto shift_tensor = ov::test::utils::create_and_fill_tensor(weights_precision, shift_tensor_shape);
            if (per_tensor_zp && weights_precision.bitwidth() == 4) {
                static_cast<uint8_t*>(shift_tensor.data())[0] = 0x88;
            }
            auto shift_const = std::make_shared<ov::op::v0::Constant>(shift_tensor);
            std::shared_ptr<ov::Node> shift_convert = std::make_shared<ov::op::v0::Convert>(shift_const, data_precision);
            if (reshape_on_decompression_constant && !per_tensor_zp) {
                auto shift_reshape_const = ov::op::v0::Constant::create(ov::element::i32, {scaleshift_target_shape.size()}, scaleshift_target_shape);
                auto shift_reshape = std::make_shared<ov::op::v1::Reshape>(shift_convert, shift_reshape_const, false);
                shift_convert = shift_reshape;
            }
            mul_parent = std::make_shared<ov::op::v1::Subtract>(weights_convert, shift_convert);
        }

        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = -0.5;
        in_data.range = 1;
        in_data.resolution = 30000;
        auto scale_tensor = ov::test::utils::create_and_fill_tensor(data_precision, scaleshift_const_shape, in_data);
        for (size_t i = 0; i < scale_tensor.get_size(); i++) {
            if (data_precision == ov::element::f16)
                scale_tensor.data<ov::float16>()[i] /= ov::float16(16.f);
            else if (data_precision == ov::element::f32)
                scale_tensor.data<float>()[i] /= 16.f;
        }
        std::shared_ptr<ov::Node> scale_const = std::make_shared<ov::op::v0::Constant>(scale_tensor);
        if (reshape_on_decompression_constant) {
            auto scale_reshape_const = ov::op::v0::Constant::create(ov::element::i32, {scaleshift_target_shape.size()}, scaleshift_target_shape);
            auto scale_reshape = std::make_shared<ov::op::v1::Reshape>(scale_const, scale_reshape_const, false);
            scale_const = scale_reshape;
        }
        std::shared_ptr<ov::Node> last_node = std::make_shared<ov::op::v1::Multiply>(mul_parent, scale_const);

        if (group_decompression) {
            auto reshape_target_shape = transpose_weights ? std::vector<int>{-1, static_cast<int>(weights_shape[0])}
                                                          : std::vector<int>{static_cast<int>(weights_shape[0]), -1};
            auto target_shape_node = ov::op::v0::Constant::create(ov::element::i32, {reshape_target_shape.size()}, reshape_target_shape);
            last_node = std::make_shared<ov::op::v1::Reshape>(last_node, target_shape_node, false);
        }
        if (transpose_weights) {
            const size_t rank = last_node->get_output_partial_shape(0).size();
            std::vector<int> order(rank);
            std::iota(order.begin(), order.end(), 0);
            std::swap(*order.rbegin(), *(order.rbegin() + 1));
            auto transpose_constant = ov::op::v0::Constant::create(ov::element::i32, {rank}, order);
            last_node = std::make_shared<ov::op::v1::Transpose>(last_node, transpose_constant);
        } else if (extra_multiply) {
            last_node = std::make_shared<ov::op::v1::Multiply>(last_node, scale_const);
        }
        return last_node;
    }

    std::shared_ptr<ov::Node> init_quantized_convolution_subgraph(std::shared_ptr<ov::Node> data) {
        size_t input_channels = 4;
        size_t output_channels = 4;
        ov::Shape input_intervals_shape;
        ov::Shape weights_intervals_shape;
        ov::Shape weights_shape1{output_channels, input_channels, 1, 1};

        auto low_act = ov::op::v0::Constant::create(ov::element::f32, input_intervals_shape, {0});
        auto high_act = ov::op::v0::Constant::create(ov::element::f32, input_intervals_shape, {20});
        auto low_weights = ov::op::v0::Constant::create(ov::element::f32, weights_intervals_shape, {-0.72519057});
        auto high_weights = ov::op::v0::Constant::create(ov::element::f32, weights_intervals_shape, {0.72519057});
        std::shared_ptr<ov::Node> activations = nullptr;
        auto weights_tensor = ov::test::utils::create_and_fill_tensor_real_distribution(ov::element::f32, weights_shape1, -0.5f, 0.5f, 1);
        std::shared_ptr<ov::Node> weights = std::make_shared<ov::op::v0::Constant>(weights_tensor);

        auto output_low_act =  ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0});
        auto output_high_act =  ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {255});
        activations = std::make_shared< ov::op::v0::FakeQuantize>(data, low_act, high_act, output_low_act, output_high_act, 256);
        activations = std::make_shared< ov::op::v0::Convert>(activations, ov::element::u8);
        activations = std::make_shared< ov::op::v0::Convert>(activations, ov::element::f32);
        std::shared_ptr<ov::Node> shift_act =  ov::op::v0::Constant::create(ov::element::u8, input_intervals_shape, {0});
        shift_act = std::make_shared< ov::op::v0::Convert>(shift_act, ov::element::f32);
        activations = std::make_shared<ov::op::v1::Subtract>(activations, shift_act);
        auto scale_act = ov::op::v0::Constant::create(ov::element::f32, input_intervals_shape, {20.0 / 255.0});
        activations = std::make_shared<ov::op::v1::Multiply>(activations, scale_act);

        auto output_low_weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0});
        auto output_high_weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {254});
        weights = std::make_shared<ov::op::v0::FakeQuantize>(weights, low_weights, high_weights, output_low_weights, output_high_weights, 255);
        weights = std::make_shared<ov::op::v0::Convert>(weights, ov::element::i8);
        weights = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
        std::shared_ptr<ov::Node> shift_weights = ov::op::v0::Constant::create(ov::element::i8, weights_intervals_shape, {0});
        shift_weights = std::make_shared<ov::op::v0::Convert>(shift_weights, ov::element::f32);
        weights = std::make_shared<ov::op::v1::Subtract>(weights, shift_weights);
        auto scale_weights = ov::op::v0::Constant::create(ov::element::f32, weights_intervals_shape, {2.0 / 255.0});
        weights = std::make_shared<ov::op::v1::Multiply>(weights, scale_weights);

        std::shared_ptr<ov::Node> conv;
        conv = std::make_shared<ov::op::v1::Convolution>(activations, weights, ov::Strides{1, 1},
                    ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
        return conv;
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        ShapeParams shape_params;
        ov::element::Type weights_precision;
        ov::element::Type activations_precision;
        bool transpose_weights;
        bool decompression_sub;
        bool reshape_on_decompression;
        bool extra_multiply;
        bool per_tensor_zp;
        uint64_t dyn_quan_group_size;

        std::tie(shape_params,
                 weights_precision,
                 activations_precision,
                 transpose_weights,
                 decompression_sub,
                 reshape_on_decompression,
                 extra_multiply,
                 per_tensor_zp,
                 dyn_quan_group_size) = GetParam();

        init_input_shapes({shape_params.data_shape, {{}, {{shape_params.weights_shape}}}});

        inType = outType = activations_precision;

        function = init_subgraph(inputDynamicShapes[0],
                                 shape_params.weights_shape,
                                 shape_params.weights_group_size,
                                 activations_precision,
                                 weights_precision,
                                 transpose_weights,
                                 decompression_sub,
                                 reshape_on_decompression,
                                 extra_multiply,
                                 per_tensor_zp);


        if (activations_precision == ov::element::f16) {
            abs_threshold = 1.0f;
        } else {
            abs_threshold = 1e-4f;
        }

        this->configuration.insert({ov::hint::dynamic_quantization_group_size(dyn_quan_group_size)});
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
          inputs.clear();
          const auto& model_inputs = function->inputs();
          for (size_t i = 0; i < model_inputs.size(); ++i) {
                const auto& model_input = model_inputs[i];
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = -1;
                in_data.range = 2;
                in_data.resolution = 10000;
                ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(), target_input_static_shapes[i], in_data);
                inputs.insert({model_input.get_node_shared_ptr(), tensor});
          }
    }

    void check_results() {
        const auto& test_param = GetParam();
        ov::element::Type weights_precision = std::get<1>(test_param);
        for (const auto& n : compiledModel.get_runtime_model()->get_ordered_ops()) {
            if (n->get_friendly_name() == "Compressed_weights") {
                ASSERT_EQ(n->get_output_element_type(0), weights_precision);
            }
        }
    }
};

TEST_P(MatmulWeightsDecompressionQuantizeConvolution, Inference) {
    run();
    check_results();
}

INSTANTIATE_TEST_SUITE_P(smoke_MatmulWeightsDecompressionQuantizeConvolution_basic,
                         MatmulWeightsDecompressionQuantizeConvolution,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(activations_precisions),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(0)),
                         MatmulWeightsDecompressionQuantizeConvolution::get_test_case_name);

} // namespace
