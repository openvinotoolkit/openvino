// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace ov;
using namespace ov::test;

namespace SubgraphTestsDefinitions {
/*
 *                        Subtract_const(U8/NF4/U4/I4)
 *                             /
 *    Weights(U8/NF4/U4/I4)  Convert(F32)
 *       |                 /
 *    Convert(F32)   Reshape(optional)
 *            \        /       Multiply_const(F32)
 *            Subtract(optional)     /
 *                  \       Reshape(optional)
 *                   \       /
 *                   Multiply
 *                      |
 *      Data(F32)   Transpose(optional)
 *            \     /
 *             Matmul
 *               |
 *              Bias
 */

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
using MatmulWeightsDecompressionParams = std::tuple<ShapeParams,              // input shapes
                                                    ov::test::ElementType,    // weights precision
                                                    ov::test::ElementType,    // activations precision
                                                    bool,                     // transpose on weights
                                                    bool,                     // decompression subtract
                                                    bool,                     // reshape on decompression constants
                                                    bool,                     // per-tensor zero-point
                                                    std::map<std::string, std::string>>;  // additional config

class MatmulWeightsDecompression : public testing::WithParamInterface<MatmulWeightsDecompressionParams>, public SubgraphBaseTest {
public:
    static std::string get_test_case_name(testing::TestParamInfo<MatmulWeightsDecompressionParams> obj) {
        ShapeParams shape_params;
        ov::test::ElementType weights_precision;
        ov::test::ElementType activations_precision;
        bool transpose;
        bool decompression_sub;
        bool reshape_on_decompression;
        bool per_tensor_zp;
        std::map<std::string, std::string> additional_config;

        std::tie(shape_params,
                 weights_precision,
                 activations_precision,
                 transpose,
                 decompression_sub,
                 reshape_on_decompression,
                 per_tensor_zp,
                 additional_config) = obj.param;

        std::ostringstream result;
        result << "data_shape=" << shape_params.data_shape << "_";
        result << "weights_shape=" << shape_params.weights_shape << "_";
        result << "group_size=" << shape_params.weights_group_size << "_";
        result << "weights_precision=" << weights_precision << "_";
        result << "activations_precision=" << activations_precision << "_";
        result << "transpose_weights=" << transpose << "_";
        result << "decompression_subtract=" << decompression_sub << "_";
        result << "reshape_on_decompression=" << reshape_on_decompression << "_";
        result << "per_tensor_zp=" << per_tensor_zp << "_";

        result << "config=(";
        for (const auto& configEntry : additional_config) {
            result << configEntry.first << ", " << configEntry.second << ":";
        }
        result << ")";

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
                                              const bool per_tensor_zp) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(data_precision, data_shape)};
        const auto weights_subgraph = init_compressed_weights_subgraph(weights_shape,
                                                                       group_size,
                                                                       data_precision,
                                                                       weights_precision,
                                                                       transpose_weights,
                                                                       add_subtract,
                                                                       reshape_on_decompression,
                                                                       per_tensor_zp);

        auto mat_mul = std::make_shared<ov::op::v0::MatMul>(params[0], weights_subgraph);
        return std::make_shared<ov::Model>(NodeVector{mat_mul}, params, "MatmulWeightsDecompression");
    }

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
            auto in_channel_idx = transpose_weights ? transformed_weights_shape.size() - 1 : transformed_weights_shape.size() - 2;
            transformed_weights_shape[in_channel_idx] = weights_shape[0] / group_size;
            transformed_weights_shape.insert(transformed_weights_shape.begin() + in_channel_idx + 1, group_size);
        }
        auto weights_tensor = ov::test::utils::create_and_fill_tensor(weights_precision, transformed_weights_shape);
        auto weights = std::make_shared<ov::op::v0::Constant>(weights_tensor);
        weights->set_friendly_name("Compressed_weights");
        auto weights_convert = std::make_shared<ngraph::opset1::Convert>(weights, data_precision);

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
            std::shared_ptr<ov::Node> shift_convert = std::make_shared<ngraph::opset1::Convert>(shift_const, data_precision);
            if (reshape_on_decompression_constant && !per_tensor_zp) {
                auto shift_reshape_const = ov::opset10::Constant::create(ov::element::i32, {scaleshift_target_shape.size()}, scaleshift_target_shape);
                auto shift_reshape = std::make_shared<ov::opset10::Reshape>(shift_convert, shift_reshape_const, false);
                shift_convert = shift_reshape;
            }
            mul_parent = std::make_shared<ov::opset10::Subtract>(weights_convert, shift_convert);
        }

        auto scale_tensor = ov::test::utils::create_and_fill_tensor(data_precision, scaleshift_const_shape, 1, -0.5, 30000);
        for (size_t i = 0; i < scale_tensor.get_size(); i++) {
            if (data_precision == ov::element::f16)
                scale_tensor.data<ov::float16>()[i] /= ov::float16(16.f);
            else if (data_precision == ov::element::f32)
                scale_tensor.data<float>()[i] /= 16.f;
        }
        std::shared_ptr<ov::Node> scale_const = std::make_shared<ov::op::v0::Constant>(scale_tensor);
        if (reshape_on_decompression_constant) {
            auto scale_reshape_const = ov::opset10::Constant::create(ov::element::i32, {scaleshift_target_shape.size()}, scaleshift_target_shape);
            auto scale_reshape = std::make_shared<ov::opset10::Reshape>(scale_const, scale_reshape_const, false);
            scale_const = scale_reshape;
        }
        std::shared_ptr<ov::Node> last_node = std::make_shared<ov::opset10::Multiply>(mul_parent, scale_const);

        if (group_decompression) {
            auto reshape_target_shape = transpose_weights ? std::vector<int>{-1, static_cast<int>(weights_shape[0])}
                                                          : std::vector<int>{static_cast<int>(weights_shape[0]), -1};
            auto target_shape_node = ov::opset10::Constant::create(ov::element::i32, {reshape_target_shape.size()}, reshape_target_shape);
            last_node = std::make_shared<ov::opset10::Reshape>(last_node, target_shape_node, false);
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

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        ShapeParams shape_params;
        ov::test::ElementType weights_precision;
        ov::test::ElementType activations_precision;
        bool transpose_weights;
        bool decompression_sub;
        bool reshape_on_decompression;
        bool per_tensor_zp;
        std::map<std::string, std::string> additional_config;

        std::tie(shape_params,
                 weights_precision,
                 activations_precision,
                 transpose_weights,
                 decompression_sub,
                 reshape_on_decompression,
                 per_tensor_zp,
                 additional_config) = GetParam();

        configuration.insert(additional_config.begin(), additional_config.end());
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
                                 per_tensor_zp);


        if (activations_precision == ov::element::f16) {
            abs_threshold = 1.0f;
        } else {
            abs_threshold = 1e-4f;
        }
    }

    void generate_inputs(const std::vector<ngraph::Shape>& target_input_static_shapes) override {
          inputs.clear();
          const auto& model_inputs = function->inputs();
          for (size_t i = 0; i < model_inputs.size(); ++i) {
                const auto& model_input = model_inputs[i];
                ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(),
                                                                            target_input_static_shapes[i],
                                                                            2,
                                                                            -1,
                                                                            10000);
                inputs.insert({model_input.get_node_shared_ptr(), tensor});
          }
    }

    void check_results() {
        const auto& test_param = GetParam();
        ov::test::ElementType weights_precision = std::get<1>(test_param);
        for (const auto& n : compiledModel.get_runtime_model()->get_ordered_ops()) {
            if (n->get_friendly_name() == "Compressed_weights") {
                ASSERT_EQ(n->get_output_element_type(0), weights_precision);
            }
        }
    }
};

TEST_P(MatmulWeightsDecompression, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    check_results();
}

namespace {

const std::vector<ov::test::ElementType> activations_precisions = {ov::element::f32, ov::element::f16};
const std::vector<ov::test::ElementType> weights_precisions = {ov::element::u8, ov::element::u4, ov::element::i4};
const std::vector<bool> transpose_weights = {true, false};
const std::vector<ShapeParams> input_shapes_basic = {
    {{{-1, -1, -1}, {{1, 4, 16}, {10, 16, 16}}}, {16, 32}},
    {{{}, {{1, 4, 16}}}, {16, 32}, 2ul},
    {{{}, {{1, 4, 16}}}, {1, 16, 32}},
    {{{}, {{1, 4, 48}}}, {48, 256}},
    {{{}, {{11, 339, 377}}}, {377, 335}}
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_basic,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(activations_precisions),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::Values(std::map<std::string, std::string>())),
                         MatmulWeightsDecompression::get_test_case_name);

const std::vector<ShapeParams> input_shapes_corner_cases_basic = {
    {{{-1, -1, -1}, {{1, 4, 16}}}, {1, 16, 32}},
    {{{-1, -1, -1}, {{1, 4, 16}}}, {16, 32}},
    {{{-1, -1, 16}, {{1, 4, 16}}}, {16, 32}, 4},
    {{{-1, 16}, {{4, 16}}}, {16, 32}, 4},
};
const std::vector<ShapeParams> input_shapes_corner_cases_big = {
    {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {1, 480, 256}},
    {{{-1, -1, -1}, {{1, 1, 4096}}}, {4096, 4096}, 128},
    {{{-1, -1, -1}, {{1, 1, 4096}}}, {4096, 4096}},
    {{{-1, 4096}, {{1, 4096}}}, {4096, 4096}, 128},
};

const std::vector<bool> add_decompression_sub = {true, false};
const std::vector<bool> reshape_on_decompression = {true, false};
const std::vector<bool> per_tensor_zp = {true, false};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases_basic,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_basic),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(activations_precisions),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::ValuesIn(add_decompression_sub),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::ValuesIn(per_tensor_zp),
                                            ::testing::Values(std::map<std::string, std::string>{})),
                         MatmulWeightsDecompression::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(MatMulCompressedWeights_corner_cases_big,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_big),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(activations_precisions),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::ValuesIn(add_decompression_sub),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::ValuesIn(per_tensor_zp),
                                            ::testing::Values(std::map<std::string, std::string>{})),
                         MatmulWeightsDecompression::get_test_case_name);
} // namespace

} // namespace SubgraphTestsDefinitions
