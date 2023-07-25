// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/fusing_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace SubgraphTestsDefinitions {
/*
 *                        Subtract_const(U8)
 *                           /
 *    Weights(U8)       Convert(F32)
 *       |               /
 *    Convert(F32)   Reshape
 *            \        /       Multiply_const(F32)
 *            Subtract(opt)     /
 *                  \       Reshape
 *                   \       /
 *                   Multiply
 *                      |
 *      Data(F32)   Transpose(opt)
 *            \     /
 *             Matmul
 *               |
 *              Bias
 */
static std::shared_ptr<ov::Model> initSubgraph(std::vector<ov::PartialShape>& inputShapes,
                                               const ov::element::Type data_precision,
                                               const ov::element::Type weights_precision,
                                               const bool transpose_weights,
                                               const bool add_subtract) {
    auto params = builder::makeDynamicParams(data_precision, {inputShapes[0]});

        auto transpose_if_necessary = [&](const ov::Shape& shape) {
            if (!transpose_weights)
                return shape;
            auto transposed_shape = shape;
            std::swap(*transposed_shape.rbegin(), *(transposed_shape.rbegin() + 1));
            return transposed_shape;
        };

        auto weights_shape = transpose_if_necessary(inputShapes[1].to_shape());
        auto weights = ngraph::builder::makeConstant<uint8_t>(weights_precision, weights_shape, {}, true);
        auto weights_convert = std::make_shared<ngraph::opset1::Convert>(weights, data_precision);

        std::shared_ptr<ov::Node> mul_parent = weights_convert;
        auto output_channels = transpose_weights ? *(weights_shape.rbegin() + 1) : *weights_shape.rbegin();
        auto scaleshift_target_shape = transpose_if_necessary(ov::Shape{1, output_channels});
        if (add_subtract) {
            auto shift_const = ngraph::builder::makeConstant<uint8_t>(weights_precision, ov::Shape{output_channels}, {}, true);
            auto shift_convert = std::make_shared<ngraph::opset1::Convert>(shift_const, data_precision);
            auto shift_reshape_const = ov::opset10::Constant::create(ov::element::i32, {scaleshift_target_shape.size()}, scaleshift_target_shape);
            auto shift_reshape = std::make_shared<ov::opset10::Reshape>(shift_convert, shift_reshape_const, false);
            mul_parent = std::make_shared<ov::opset10::Subtract>(weights_convert, shift_reshape);
        }

        auto scale_const = ngraph::builder::makeConstant<float>(data_precision, ov::Shape{output_channels}, {}, true);
        auto scale_reshape_const = ov::opset10::Constant::create(ov::element::i32, {scaleshift_target_shape.size()}, scaleshift_target_shape);
        auto scale_reshape = std::make_shared<ov::opset10::Reshape>(scale_const, scale_reshape_const, false);
        auto multiply = std::make_shared<ov::opset10::Multiply>(mul_parent, scale_reshape);

        std::shared_ptr<ov::Node> matmul_weights = multiply;
        if (transpose_weights) {
            const size_t rank = matmul_weights->get_output_partial_shape(0).size();
            std::vector<int> order(rank);
            std::iota(order.begin(), order.end(), 0);
            std::swap(*order.rbegin(), *(order.rbegin() + 1));
            const auto transpose_constant = ov::opset10::Constant::create(ov::element::i32, {rank}, order);
            const auto transpose = std::make_shared<ov::opset10::Transpose>(matmul_weights, transpose_constant);
            matmul_weights = transpose;
        }
        auto matMul = builder::makeMatMul(params[0], matmul_weights);
        ov::Shape bias_const_shape{static_cast<size_t>(matMul->get_output_partial_shape(0).rbegin()->get_length())};
        auto bias_const = ngraph::builder::makeConstant<float>(data_precision, bias_const_shape, {}, true);
        auto bias = std::make_shared<ov::opset10::Add>(matMul, bias_const);

        return std::make_shared<ov::Model>(matMul, params, "MatmulWeightsDecompression");
}

using MatmulWeightsDecompressionParams = std::tuple<std::vector<InputShape>,            // input shapes
                                                    bool,                               // transpose on weights
                                                    bool,                               // decompression subtract
                                                    std::map<std::string, std::string>  // additional config
                                                    >;

class MatmulWeightsDecompression : public testing::WithParamInterface<MatmulWeightsDecompressionParams>,
                                  virtual public SubgraphBaseTest,
                                  public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatmulWeightsDecompressionParams> obj) {
        std::vector<InputShape> inputShapes;
        bool transpose;
        bool decompression_sub;
        std::map<std::string, std::string> additional_config;

        std::tie(inputShapes, transpose, decompression_sub, additional_config) = obj.param;

        std::ostringstream result;
        for (const auto& shape : inputShapes) {
            result << CommonTestUtils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << CommonTestUtils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "transpose_weights=" << transpose << "_";
        result << "decompression_subtract=" << decompression_sub << "_";

        result << "config=(";
        for (const auto& configEntry : additional_config) {
            result << configEntry.first << ", " << configEntry.second << ":";
        }
        result << ")";

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        std::vector<InputShape> inputShapes;
        bool transpose_weights;
        bool decompression_sub;
        std::map<std::string, std::string> additional_config;

        std::tie(inputShapes, transpose_weights, decompression_sub, additional_config) = this->GetParam();

        configuration.insert(additional_config.begin(), additional_config.end());
        init_input_shapes(inputShapes);

        ElementType netType = element::f32;
        if (additional_config[PluginConfigParams::KEY_ENFORCE_BF16] == PluginConfigParams::YES)
            netType = ElementType::bf16;
        inType = outType = netType;

        function = initSubgraph(inputDynamicShapes, netType, ov::element::u8, transpose_weights, decompression_sub);
    }
};

TEST_P(MatmulWeightsDecompression, CompareWithRefs) {
    run();
    std::map<std::string, std::string> additional_config = std::get<3>(GetParam());
    const size_t expected_count = additional_config[PluginConfigParams::KEY_ENFORCE_BF16] == PluginConfigParams::YES ? 1 : 0;
    CheckNumberOfNodesWithType(compiledModel, "Convert", expected_count);
    CheckNumberOfNodesWithType(compiledModel, "Eltwise", expected_count);
}

namespace {

std::vector<std::map<std::string, std::string>> filterAdditionalConfig() {
    std::vector<std::map<std::string, std::string>> additional_config{CPUTestUtils::cpuEmptyPluginConfig};
    if (with_cpu_x86_avx512_core())
        additional_config.push_back({{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}});
    return additional_config;
}

const std::vector<std::vector<InputShape>> input_shapes_basic = {
    {{{-1, -1, -1}, {{1, 4, 16}, {10, 16, 16}}}, {{}, {{16, 32}}}},
    {{{}, {{1, 4, 16}}}, {{}, {{1, 16, 32}}}},
    {{{}, {{10, 40, 496}}}, {{}, {{1, 496, 240}}}},
    {{{}, {{1, 4, 32}}}, {{}, {{32, 256}}}},
    {{{}, {{1, 4, 48}}}, {{}, {{48, 256}}}},
    {{{}, {{1, 4, 512}}}, {{}, {{512, 256}}}},
    {{{}, {{1, 16, 32}}}, {{}, {{32, 64}}}},
};
INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedU8Weights_basic,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(filterAdditionalConfig())),
                         MatmulWeightsDecompression::getTestCaseName);

const std::vector<std::vector<InputShape>> input_shapes_corner_cases = {
    {{{-1, -1, -1}, {{1, 4, 16}}}, {{}, {{1, 16, 32}}}},
    {{{-1, -1, -1}, {{1, 4, 16}}}, {{}, {{16, 32}}}},
};
const std::vector<bool> transpose_weights = {true, false};
const std::vector<bool> add_decompression_sub = {true, false};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedU8Weights_corner_cases,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::ValuesIn(add_decompression_sub),
                                            ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
                         MatmulWeightsDecompression::getTestCaseName);
} // namespace

} // namespace SubgraphTestsDefinitions
