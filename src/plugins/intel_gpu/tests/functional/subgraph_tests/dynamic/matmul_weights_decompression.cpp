// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace ngraph;
using namespace ov::test;

namespace SubgraphTestsDefinitions {
/*
 *                        Subtract_const(U8)
 *                           /
 *    Weights(U8)       Convert(F32)
 *       |               /
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
using MatmulWeightsDecompressionParams = std::tuple<std::vector<InputShape>,  // input shapes
                                                    ov::test::ElementType,    // weights precision
                                                    ov::test::ElementType,    // activations precision
                                                    bool,                     // transpose on weights
                                                    bool,                     // decompression subtract
                                                    bool,                     // reshape on decompression constants
                                                    std::map<std::string, std::string>>;  // additional config

class MatmulWeightsDecompression : public testing::WithParamInterface<MatmulWeightsDecompressionParams>, public SubgraphBaseTest {
public:
    static std::string get_test_case_name(testing::TestParamInfo<MatmulWeightsDecompressionParams> obj) {
        std::vector<InputShape> inputShapes;
        ov::test::ElementType weights_precision;
        ov::test::ElementType activations_precision;
        bool transpose;
        bool decompression_sub;
        bool reshape_on_decompression;
        std::map<std::string, std::string> additional_config;

        std::tie(inputShapes,
                 weights_precision,
                 activations_precision,
                 transpose,
                 decompression_sub,
                 reshape_on_decompression,
                 additional_config) = obj.param;

        std::ostringstream result;
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "weights_precision=" << weights_precision << "_";
        result << "activations_precision=" << activations_precision << "_";
        result << "transpose_weights=" << transpose << "_";
        result << "decompression_subtract=" << decompression_sub << "_";
        result << "reshape_on_decompression=" << reshape_on_decompression << "_";

        result << "config=(";
        for (const auto& configEntry : additional_config) {
            result << configEntry.first << ", " << configEntry.second << ":";
        }
        result << ")";

        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> init_subgraph(std::vector<ov::PartialShape>& inputShapes,
                                             const ov::element::Type data_precision,
                                             const ov::element::Type weights_precision,
                                             const bool transpose_weights,
                                             const bool add_subtract,
                                             const bool reshape_on_decompression) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(data_precision, inputShapes[0])};
        auto transpose_if_necessary = [&](const ov::Shape& shape) {
            if (!transpose_weights)
                return shape;
            auto transposed_shape = shape;
            std::swap(*transposed_shape.rbegin(), *(transposed_shape.rbegin() + 1));
            return transposed_shape;
        };

        auto weights_shape = transpose_if_necessary(inputShapes[1].to_shape());
        auto weights = ngraph::builder::makeConstant<uint8_t>(weights_precision, weights_shape, {}, true);
        weights->set_friendly_name("Compressed_weights");
        auto weights_convert = std::make_shared<ngraph::opset1::Convert>(weights, data_precision);

        std::shared_ptr<ov::Node> mul_parent = weights_convert;
        auto output_channels = transpose_weights ? *(weights_shape.rbegin() + 1) : *weights_shape.rbegin();
        auto scaleshift_target_shape = transpose_if_necessary(ov::Shape{1, output_channels});
        auto scaleshift_const_shape = reshape_on_decompression ? ov::Shape{output_channels} : scaleshift_target_shape;
        if (add_subtract) {
            auto shift_const = ngraph::builder::makeConstant<uint8_t>(weights_precision, scaleshift_const_shape, {}, true);
            std::shared_ptr<ov::Node> shift_convert = std::make_shared<ngraph::opset1::Convert>(shift_const, data_precision);
            if (reshape_on_decompression) {
                auto shift_reshape_const = ov::opset10::Constant::create(ov::element::i32, {scaleshift_target_shape.size()}, scaleshift_target_shape);
                auto shift_reshape = std::make_shared<ov::opset10::Reshape>(shift_convert, shift_reshape_const, false);
                shift_convert = shift_reshape;
            }
            mul_parent = std::make_shared<ov::opset10::Subtract>(weights_convert, shift_convert);
        }

        std::shared_ptr<ov::Node> scale_const = ngraph::builder::makeConstant<float>(data_precision, scaleshift_const_shape, {}, true);
        if (reshape_on_decompression) {
            auto scale_reshape_const = ov::opset10::Constant::create(ov::element::i32, {scaleshift_target_shape.size()}, scaleshift_target_shape);
            auto scale_reshape = std::make_shared<ov::opset10::Reshape>(scale_const, scale_reshape_const, false);
            scale_const = scale_reshape;
        }
        auto multiply = std::make_shared<ov::opset10::Multiply>(mul_parent, scale_const);

        std::shared_ptr<ov::Node> matmul_weights = multiply;
        if (transpose_weights) {
            const size_t rank = matmul_weights->get_output_partial_shape(0).size();
            std::vector<int> order(rank);
            std::iota(order.begin(), order.end(), 0);
            std::swap(*order.rbegin(), *(order.rbegin() + 1));
            auto transpose_constant = ov::opset10::Constant::create(ov::element::i32, {rank}, order);
            auto transpose = std::make_shared<ov::opset10::Transpose>(matmul_weights, transpose_constant);
            matmul_weights = transpose;
        }
        auto matMul = builder::makeMatMul(params[0], matmul_weights);
        return std::make_shared<ov::Model>(NodeVector{matMul}, params, "MatmulWeightsDecompression");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        std::vector<InputShape> inputShapes;
        ov::test::ElementType weights_precision;
        ov::test::ElementType activations_precision;
        bool transpose_weights;
        bool decompression_sub;
        bool reshape_on_decompression;
        std::map<std::string, std::string> additional_config;

        std::tie(inputShapes,
                 weights_precision,
                 activations_precision,
                 transpose_weights,
                 decompression_sub,
                 reshape_on_decompression,
                 additional_config) = GetParam();

        configuration.insert(additional_config.begin(), additional_config.end());
        init_input_shapes(inputShapes);

        inType = outType = activations_precision;

        function = init_subgraph(inputDynamicShapes, activations_precision, weights_precision, transpose_weights, decompression_sub, reshape_on_decompression);
    }

    void checkResults() {
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
    checkResults();
}

namespace {

const std::vector<ov::test::ElementType> activations_precisions = {ov::element::f32, ov::element::f16};
const std::vector<ov::test::ElementType> weights_precisions = {ov::element::u8};
const std::vector<std::vector<InputShape>> input_shapes_basic = {
    {{{-1, -1, -1}, {{1, 4, 16}, {10, 16, 16}}}, {{}, {{16, 32}}}},
    {{{}, {{10, 40, 496}}}, {{}, {{1, 496, 240}}}},
    {{{}, {{1, 4, 48}}}, {{}, {{48, 256}}}},
    {{{}, {{11, 339, 377}}}, {{}, {{377, 335}}}},
    {{{}, {{1, 4, 32}}}, {{}, {{32, 256}}}},
    {{{}, {{1, 4, 512}}}, {{}, {{512, 256}}}},
    {{{}, {{1, 16, 32}}}, {{}, {{32, 64}}}},
    {{{}, {{2, 4, 32}}}, {{}, {{32, 65}}}},
    {{{}, {{3, 12, 768}}}, {{}, {{768, 1024}}}},
    {{{}, {{11, 339, 577}}}, {{}, {{577, 335}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_basic,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(activations_precisions),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(std::map<std::string, std::string>())),
                         MatmulWeightsDecompression::get_test_case_name);

const std::vector<std::vector<InputShape>> input_shapes_corner_cases_basic = {
    {{{-1, -1, -1}, {{1, 4, 16}}}, {{}, {{1, 16, 32}}}},
    {{{}, {{1, 4, 16}}}, {{}, {{1, 16, 32}}}},
    {{{-1, -1, -1}, {{1, 4, 16}}}, {{}, {{16, 32}}}},
    {{{-1, -1, -1, -1}, {{1, 1, 4, 16}}}, {{}, {{1, 1, 16, 32}}}},
    {{{}, {{1, 1, 4, 16}}}, {{}, {{1, 1, 16, 32}}}},
};
const std::vector<std::vector<InputShape>> input_shapes_corner_cases_big = {
    {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {{}, {{1, 480, 256}}}},
};

const std::vector<bool> transpose_weights = {true, false};
const std::vector<bool> add_decompression_sub = {true, false};
const std::vector<bool> reshape_on_decompression = {true, false};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases_basic,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_basic),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(activations_precisions),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::ValuesIn(add_decompression_sub),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::Values(std::map<std::string, std::string>{})),
                         MatmulWeightsDecompression::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases_big,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_big),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(activations_precisions),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::ValuesIn(add_decompression_sub),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::Values(std::map<std::string, std::string>{})),
                         MatmulWeightsDecompression::get_test_case_name);
} // namespace

} // namespace SubgraphTestsDefinitions
