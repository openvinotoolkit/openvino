// Copyright (C) 2024 Intel Corporation
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
    ShapeParams(InputShape data_shape, std::vector<ov::Shape> weights_shapes, int weights_group_size = -1)
        : data_shape(std::move(data_shape)),
          weights_shapes(std::move(weights_shapes)),
          weights_group_size(weights_group_size) {}

    InputShape data_shape;
    std::vector<ov::Shape> weights_shapes;
    // Decompression group size. If the value is equal to -1, ordinary decompression is used
    int weights_group_size;
};

using FullyConnectedHorizontalFusionParams = std::tuple<ShapeParams,              // input shapes
                                                    ov::element::Type,        // weights type
                                                    ov::element::Type,        // activations type
                                                    bool,                     // transpose on weights
                                                    bool,                     // decompression subtract
                                                    bool,                     // reshape on decompression constants
                                                    bool,                     // per-tensor zero-point
                                                    bool,                     // has bias
                                                    uint64_t                  // dynamic_quantization_group_size
                                                    >;


class FullyConnectedHorizontalFusion : public testing::WithParamInterface<FullyConnectedHorizontalFusionParams>,
                                       virtual public ov::test::SubgraphBaseTest {
public:
    static std::string get_test_case_name(testing::TestParamInfo<FullyConnectedHorizontalFusionParams> obj) {
        ShapeParams shape_params;
        ov::element::Type weights_precision;
        ov::element::Type activations_precision;
        bool transpose;
        bool decompression_sub;
        bool reshape_on_decompression;
        bool per_tensor_zp;
        bool has_bias;
        uint64_t dyn_quan_group_size;

        std::tie(shape_params,
                 weights_precision,
                 activations_precision,
                 transpose,
                 decompression_sub,
                 reshape_on_decompression,
                 per_tensor_zp,
                 has_bias,
                 dyn_quan_group_size) = obj.param;

        std::ostringstream result;
        result << "data_shape=";
        result << ov::test::utils::partialShape2str({shape_params.data_shape.first}) << "_";
        for (const auto& actual_shape : shape_params.data_shape.second) {
            result << ov::test::utils::partialShape2str({actual_shape}) << "_";
        }
        result << "_" << "weights1_shape=" << shape_params.weights_shapes[0] << "_";
        result << "_" << "weights2_shape=" << shape_params.weights_shapes[1] << "_";
        result << "_" << "weights3_shape=" << shape_params.weights_shapes[2] << "_";
        auto weights_group_size = shape_params.weights_group_size;
        weights_group_size = weights_group_size == -1 ? 111 : weights_group_size;
        result << "group_size=" << weights_group_size << "_";
        result << "weights_precision=" << weights_precision << "_";
        result << "activations_precision=" << activations_precision << "_";
        result << "transpose_weights=" << transpose << "_";
        result << "decompression_subtract=" << decompression_sub << "_";
        result << "reshape_on_decompression=" << reshape_on_decompression << "_";
        result << "per_tensor_zp=" << per_tensor_zp;
        result << "has_bias=" << has_bias;
        result << "dyn_quan_group_size=" << dyn_quan_group_size;

        return result.str();
    }

protected:
    std::shared_ptr<ov::Node> init_compressed_weights_subgraph(const ov::Shape& weights_shape,
                                                               const int group_size,
                                                               const ov::element::Type data_precision,
                                                               const ov::element::Type weights_precision,
                                                               const bool transpose_weights,
                                                               const bool add_subtract,
                                                               const bool reshape_on_decompression_constant,
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
            auto in_channel_idx =
                transpose_weights ? transformed_weights_shape.size() - 1 : transformed_weights_shape.size() - 2;
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
        if (add_subtract) {
            auto shift_tensor_shape = per_tensor_zp ? ov::Shape{1} : scaleshift_const_shape;
            auto shift_tensor = ov::test::utils::create_and_fill_tensor(weights_precision, shift_tensor_shape);
            if (per_tensor_zp && weights_precision.bitwidth() == 4) {
                static_cast<uint8_t*>(shift_tensor.data())[0] = 0x88;
            }
            auto shift_const = std::make_shared<ov::op::v0::Constant>(shift_tensor);
            shift_const->set_friendly_name("shift_const");
            std::shared_ptr<ov::Node> shift_convert =
                std::make_shared<ov::op::v0::Convert>(shift_const, data_precision);
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
        auto scale_tensor = ov::test::utils::create_and_fill_tensor(data_precision, scaleshift_const_shape, in_data);
        for (size_t i = 0; i < scale_tensor.get_size(); i++) {
            if (data_precision == ov::element::f16)
                scale_tensor.data<ov::float16>()[i] /= ov::float16(16.f);
            else if (data_precision == ov::element::f32)
                scale_tensor.data<float>()[i] /= 16.f;
        }
        std::shared_ptr<ov::Node> scale_const = std::make_shared<ov::op::v0::Constant>(scale_tensor);
        scale_const->set_friendly_name("scale_const");
        if (reshape_on_decompression_constant) {
            auto scale_reshape_const = ov::op::v0::Constant::create(ov::element::i32,
                                                                    {scaleshift_target_shape.size()},
                                                                    scaleshift_target_shape);
            auto scale_reshape = std::make_shared<ov::op::v1::Reshape>(scale_const, scale_reshape_const, false);
            scale_const = scale_reshape;
        }
        std::shared_ptr<ov::Node> last_node = std::make_shared<ov::op::v1::Multiply>(mul_parent, scale_const);

        if (group_decompression) {
            auto reshape_target_shape = transpose_weights ? std::vector<int>{-1, static_cast<int>(weights_shape[0])}
                                                          : std::vector<int>{static_cast<int>(weights_shape[0]), -1};
            auto target_shape_node =
                ov::op::v0::Constant::create(ov::element::i32, {reshape_target_shape.size()}, reshape_target_shape);
            last_node = std::make_shared<ov::op::v1::Reshape>(last_node, target_shape_node, false);
        }
        return last_node;
    }
    std::shared_ptr<ov::Model> init_subgraph(const ov::PartialShape& data_shape,
                                             const std::vector<ov::Shape>& weights_shapes,
                                             const int group_size,
                                             const ov::element::Type data_precision,
                                             const ov::element::Type weights_precision,
                                             const bool transpose_weights,
                                             const bool add_subtract,
                                             const bool reshape_on_decompression,
                                             const bool per_tensor_zp,
                                             const bool has_bias) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(data_precision, data_shape)};
        const auto weight1 = init_compressed_weights_subgraph(weights_shapes[0],
                                                              group_size,
                                                              data_precision,
                                                              weights_precision,
                                                              transpose_weights,
                                                              add_subtract,
                                                              reshape_on_decompression,
                                                              per_tensor_zp);
        const auto weight2 = init_compressed_weights_subgraph(weights_shapes[1],
                                                              group_size,
                                                              data_precision,
                                                              weights_precision,
                                                              transpose_weights,
                                                              add_subtract,
                                                              reshape_on_decompression,
                                                              per_tensor_zp);

        const auto weight3 = init_compressed_weights_subgraph(weights_shapes[2],
                                                              group_size,
                                                              data_precision,
                                                              weights_precision,
                                                              transpose_weights,
                                                              add_subtract,
                                                              reshape_on_decompression,
                                                              per_tensor_zp);

        auto matmul1 = std::make_shared<ov::op::v0::MatMul>(params[0], weight1, false, transpose_weights);
        matmul1->set_friendly_name("fully_connected_1");
        auto matmul2 = std::make_shared<ov::op::v0::MatMul>(params[0], weight2, false, transpose_weights);
        matmul2->set_friendly_name("fully_connected_2");
        auto matmul3 = std::make_shared<ov::op::v0::MatMul>(params[0], weight3, false, transpose_weights);
        matmul3->set_friendly_name("fully_connected_3");
        if (!has_bias) {
            auto matmul4 = std::make_shared<ov::op::v0::MatMul>(matmul1, matmul2, true, false);
            matmul4->set_friendly_name("gemm1");
            auto matmul5 = std::make_shared<ov::op::v0::MatMul>(matmul4, matmul3, true, true);
            matmul5->set_friendly_name("gemm2");
            return std::make_shared<ov::Model>(ov::NodeVector{matmul5}, params, "FCHorizontalFusion");
        } else {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -0.5;
            in_data.range = 1;
            in_data.resolution = 30000;

            auto bias1_shape = ov::Shape{1, weights_shapes[0].back()};
            auto bias1_tensor = ov::test::utils::create_and_fill_tensor(data_precision, bias1_shape, in_data);
            auto bias1_const = std::make_shared<ov::op::v0::Constant>(bias1_tensor);
            auto bias_add1 = std::make_shared<ov::op::v1::Add>(matmul1, bias1_const);
            bias_add1->set_friendly_name("add1");
            auto bias2_shape = ov::Shape{1, weights_shapes[1].back()};
            auto bias2_tensor = ov::test::utils::create_and_fill_tensor(data_precision, bias2_shape, in_data);
            auto bias2_const = std::make_shared<ov::op::v0::Constant>(bias2_tensor);
            auto bias_add2 = std::make_shared<ov::op::v1::Add>(matmul2, bias2_const);
            bias_add2->set_friendly_name("add2");
            auto bias3_shape = ov::Shape{1, weights_shapes[2].back()};
            auto bias3_tensor = ov::test::utils::create_and_fill_tensor(data_precision, bias3_shape, in_data);
            auto bias3_const = std::make_shared<ov::op::v0::Constant>(bias3_tensor);
            auto bias_add3 = std::make_shared<ov::op::v1::Add>(matmul3, bias3_const);
            bias_add3->set_friendly_name("add3");

            auto matmul4 = std::make_shared<ov::op::v0::MatMul>(bias_add1, bias_add2, true, false);
            matmul4->set_friendly_name("gemm1");
            auto matmul5 = std::make_shared<ov::op::v0::MatMul>(matmul4, bias_add3, true, true);
            matmul5->set_friendly_name("gemm2");
            return std::make_shared<ov::Model>(ov::NodeVector{matmul5}, params, "FCHorizontalFusion");
        }
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        ShapeParams shape_params;
        ov::element::Type weights_precision;
        ov::element::Type activations_precision;
        bool transpose_weights;
        bool decompression_sub;
        bool reshape_on_decompression;
        bool per_tensor_zp;
        bool has_bias;
        uint64_t dyn_quan_group_size;

        std::tie(shape_params,
                 weights_precision,
                 activations_precision,
                 transpose_weights,
                 decompression_sub,
                 reshape_on_decompression,
                 per_tensor_zp,
                 has_bias,
                 dyn_quan_group_size) = GetParam();

        init_input_shapes({shape_params.data_shape, {{}, shape_params.weights_shapes}});

        inType = outType = activations_precision;
        function = init_subgraph(inputDynamicShapes[0],
                                 shape_params.weights_shapes,
                                 shape_params.weights_group_size,
                                 activations_precision,
                                 weights_precision,
                                 transpose_weights,
                                 decompression_sub,
                                 reshape_on_decompression,
                                 per_tensor_zp,
                                 has_bias);

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
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(),
                                                                        target_input_static_shapes[i],
                                                                        in_data);
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

TEST_P(FullyConnectedHorizontalFusion, Inference) {
    run();
    check_results();
}

const std::vector<ov::element::Type> activations_precisions = {ov::element::f32, ov::element::f16};
const std::vector<ov::element::Type> weights_precisions = {ov::element::u8, ov::element::u4, ov::element::i4};
const std::vector<bool> per_tensor_zp = {true, false};
const std::vector<bool> transpose_weights = {true, false};

std::vector<ov::Shape> weights1 = {{1, 16, 32}, {1, 16, 4}, {1, 16, 32}};
std::vector<ov::Shape> weights2 = {{16, 32}, {16, 4}, {16, 32}};
std::vector<ov::Shape> weights3 = {{28, 24}, {28, 18}, {28, 24}};
std::vector<ov::Shape> weights4 = {{1, 16, 24}, {1, 16, 24}, {1, 16, 24}};

const std::vector<ShapeParams> input_shapes = {
    {{{-1, -1, -1}, {{1, 4, 16}}}, weights1},
    {{{-1, -1, 16}, {{1, 4, 16}}}, weights2, 4},
    {{{-1, 28}, {{16, 28}}}, weights3, 4},
    {{{-1, -1, -1}, {{1, 4, 16}}}, weights4},
};

INSTANTIATE_TEST_SUITE_P(smoke_FCHorizontalFusion_no_bias,
                         FullyConnectedHorizontalFusion,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(activations_precisions),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(per_tensor_zp),
                                            ::testing::Values(false),
                                            ::testing::Values(0) /* no dyn_quan */),
                         FullyConnectedHorizontalFusion::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_FCHorizontalFusion_with_bias,
                         FullyConnectedHorizontalFusion,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(activations_precisions),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(0) /* no dyn_quan */),
                         FullyConnectedHorizontalFusion::get_test_case_name);

std::vector<ov::Shape> dyn_quan_weights = {{1, 128, 32}, {1, 128, 4}, {1, 128, 32}};

const ShapeParams dyn_quan_input_shape =
    {{{-1, -1, 128}, {{1, 4, 128}}}, dyn_quan_weights};

INSTANTIATE_TEST_SUITE_P(smoke_FCHorizontalFusion_no_bias_dyn_quan,
                         FullyConnectedHorizontalFusion,
                         ::testing::Combine(::testing::Values(dyn_quan_input_shape),
                                            ::testing::Values(weights_precisions[0]),
                                            ::testing::Values(activations_precisions[0]),
                                            ::testing::Values(transpose_weights[0]),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::Values(UINT64_MAX) /* dyn_quan */),
                         FullyConnectedHorizontalFusion::get_test_case_name);


}  // namespace
