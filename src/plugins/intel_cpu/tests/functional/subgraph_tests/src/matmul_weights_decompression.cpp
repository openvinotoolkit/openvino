// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/fusing_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace SubgraphTestsDefinitions {
/*
 * WP - weights precision
 * DP - decompression precision
 * IP - input precision
 * Opt - optional
 *                        Subtract_const(WP)
 *                           /
 *    Weights(WP)     Convert(DP)
 *       |               /
 *    Convert(DP)   Reshape (Opt)
 *            \        /          Multiply_const(DP)
 *            Subtract(Opt)       /
 *                  \         Reshape (Opt)
 *                   \         /
 *                    Multiply
 *                      |
 *                   Reshape (in case of group decompression)
 *                      |
 *                   Convert (if IP != DP)
 *                      |
 *      Data(IP)   Transpose(Opt)
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
using MatmulWeightsDecompressionParams = std::tuple<ShapeParams,
                                                    ov::test::ElementType,  // weights precision
                                                    ov::test::ElementType,  // decompression precision
                                                    bool,                   // transpose on weights
                                                    bool,                   // decompression subtract
                                                    bool,                   // reshape on decompression constants
                                                    std::map<std::string, std::string>,  // additional config
                                                    fusingSpecificParams,
                                                    bool>;  // should use decompression implementation

class MatmulWeightsDecompression : public testing::WithParamInterface<MatmulWeightsDecompressionParams>,
                                  virtual public SubgraphBaseTest,
                                  public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatmulWeightsDecompressionParams> obj) {
        ShapeParams shape_params;
        ov::test::ElementType weights_precision;
        ov::test::ElementType decompression_precision;
        bool transpose;
        bool decompression_sub;
        bool reshape_on_decompression;
        std::map<std::string, std::string> additional_config;
        fusingSpecificParams fusing_params;
        bool should_fuse;

        std::tie(shape_params,
                 weights_precision,
                 decompression_precision,
                 transpose,
                 decompression_sub,
                 reshape_on_decompression,
                 additional_config,
                 fusing_params,
                 should_fuse) = obj.param;

        std::ostringstream result;
        result << "data_shape=" << shape_params.data_shape << "_";
        result << "weights_shape=" << shape_params.weights_shape << "_";
        result << "group_size=" << shape_params.weights_group_size << "_";
        result << "weights_precision=" << weights_precision << "_";
        result << "decompression_precision=" << decompression_precision << "_";
        result << "transpose_weights=" << transpose << "_";
        result << "decompression_subtract=" << decompression_sub << "_";
        result << "reshape_on_decompression=" << reshape_on_decompression << "_";

        result << "config=(";
        for (const auto& configEntry : additional_config) {
            result << configEntry.first << ", " << configEntry.second << ":";
        }
        result << ")";
        result << CpuTestWithFusing::getTestCaseName(fusing_params);

        return result.str();
    }

protected:
    std::shared_ptr<ov::Node> initDecompressionWeights(const ov::Shape& weights_shape,
                                                       const int group_size,
                                                       const ov::element::Type data_precision,
                                                       const ov::element::Type weights_precision,
                                                       const ov::element::Type decompression_precision,
                                                       const bool transpose_weights,
                                                       const bool add_subtract,
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
            auto in_channel_idx = transpose_weights ? transformed_weights_shape.size() - 1 : transformed_weights_shape.size() - 2;
            transformed_weights_shape[in_channel_idx] = weights_shape[0] / group_size;
            transformed_weights_shape.insert(transformed_weights_shape.begin() + in_channel_idx + 1, group_size);
        }

        auto weights = ngraph::builder::makeConstant<int8_t>(weights_precision, transformed_weights_shape, {}, true, 7);
        weights->set_friendly_name("Compressed_weights");
        auto weights_convert = std::make_shared<ngraph::opset1::Convert>(weights, decompression_precision);

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
            auto shift_const = ngraph::builder::makeConstant<uint8_t>(weights_precision, scaleshift_const_shape, {}, true, 7);
            std::shared_ptr<ov::Node> shift_convert = std::make_shared<ngraph::opset1::Convert>(shift_const, decompression_precision);
            if (reshape_on_decompression_constant) {
                auto shift_reshape_const = ov::opset10::Constant::create(ov::element::i32, {scaleshift_target_shape.size()}, scaleshift_target_shape);
                auto shift_reshape = std::make_shared<ov::opset10::Reshape>(shift_convert, shift_reshape_const, false);
                shift_convert = shift_reshape;
            }
            mul_parent = std::make_shared<ov::opset10::Subtract>(weights_convert, shift_convert);
        }

        std::shared_ptr<ov::Node> scale_const = ngraph::builder::makeConstant<float>(decompression_precision, scaleshift_const_shape, {}, true);
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
        if (decompression_precision != data_precision) {
            last_node = std::make_shared<ov::opset10::Convert>(last_node, data_precision);
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

    std::shared_ptr<ov::Model> initSubgraph(const ov::PartialShape& data_shape,
                                            const ov::Shape& weights_shape,
                                            const int group_size,
                                            const ov::element::Type data_precision,
                                            const ov::element::Type weights_precision,
                                            const ov::element::Type decompression_precision,
                                            const bool transpose_weights,
                                            const bool add_subtract,
                                            const bool reshape_on_decompression) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(data_precision, data_shape)};
        const auto weights_subgraph = initDecompressionWeights(weights_shape,
                                                               group_size,
                                                               data_precision,
                                                               weights_precision,
                                                               decompression_precision,
                                                               transpose_weights,
                                                               add_subtract,
                                                               reshape_on_decompression);
        auto matMul = std::make_shared<ov::op::v0::MatMul>(params[0], weights_subgraph);
        return makeNgraphFunction(data_precision, params, matMul, "MatmulWeightsDecompression");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        ShapeParams shape_params;
        ov::test::ElementType weights_precision;
        ov::test::ElementType decompression_precision;
        bool transpose_weights;
        bool decompression_sub;
        bool reshape_on_decompression;
        std::map<std::string, std::string> additional_config;
        fusingSpecificParams fusing_params;
        bool should_fuse;

        std::tie(shape_params,
                 weights_precision,
                 decompression_precision,
                 transpose_weights,
                 decompression_sub,
                 reshape_on_decompression,
                 additional_config,
                 fusing_params,
                 should_fuse) = GetParam();

        configuration.insert(additional_config.begin(), additional_config.end());
        std::tie(postOpMgrPtr, fusedOps) = fusing_params;
        init_input_shapes({shape_params.data_shape, {{}, {{shape_params.weights_shape}}}});

        ElementType netType = element::f32;
        inType = outType = netType;

        function = initSubgraph(inputDynamicShapes[0],
                                shape_params.weights_shape,
                                shape_params.weights_group_size,
                                netType,
                                weights_precision,
                                decompression_precision,
                                transpose_weights,
                                decompression_sub,
                                reshape_on_decompression);
    }

    void checkResults() {
        const auto& test_param = GetParam();
        const auto& weights_precision = std::get<1>(test_param);

        bool weights_found = false;
        for (const auto& n : compiledModel.get_runtime_model()->get_ordered_ops()) {
            if (n->get_friendly_name() == "Compressed_weights") {
                ASSERT_EQ(n->get_output_element_type(0), weights_precision);
                weights_found = true;
            }
        }
        ASSERT_TRUE(weights_found);

        const bool should_fuse = std::get<8>(test_param);
        const size_t expected_count = should_fuse ? 0 : 1;
        CheckNumberOfNodesWithType(compiledModel, "Convert", expected_count);
        CheckNumberOfNodesWithType(compiledModel, "Eltwise", expected_count);
        CheckNumberOfNodesWithType(compiledModel, "Subgraph", 0);
    }
};

TEST_P(MatmulWeightsDecompression, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    checkResults();
}
using MatmulWeightsDecompression_FP16 = MatmulWeightsDecompression;
TEST_P(MatmulWeightsDecompression_FP16, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!(ov::with_cpu_x86_avx512_core_fp16() || ov::with_cpu_x86_avx512_core_amx_fp16())) {
        GTEST_SKIP() << "Skipping test, platform don't support precision f16";
    }
    configuration.insert({ov::hint::inference_precision.name(), "f16"});

    run();
    checkResults();
}

namespace {

std::vector<std::map<std::string, std::string>> filterAdditionalConfigBasic() {
    std::vector<std::map<std::string, std::string>> additional_config = {CPUTestUtils::cpuEmptyPluginConfig};
    return additional_config;
}
std::vector<std::map<std::string, std::string>> filterAdditionalConfigAMX() {
    std::vector<std::map<std::string, std::string>> additional_config = {};
    if (with_cpu_x86_avx512_core_amx())
        additional_config.push_back({{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}});
    return additional_config;
}

std::vector<std::map<std::string, std::string>> filterAdditionalConfig_FP16() {
    std::vector<std::map<std::string, std::string>> additional_config;
    additional_config.push_back({{ov::hint::inference_precision.name(), "f16"}});
    return additional_config;
}

const std::vector<ov::test::ElementType> decompression_precisions = {ov::element::f32};
const std::vector<ov::test::ElementType> weights_precisions_basic = {ov::element::u8,
                                                                     ov::element::u4,
                                                                     ov::element::i4,
                                                                     ov::element::nf4};
const std::vector<ov::test::ElementType> weights_precisions_amx = {ov::element::u8};

const std::vector<ShapeParams> input_shapes_basic = {
    {{{-1, -1, -1}, {{1, 4, 16}, {10, 16, 16}}}, {16, 32}},
    {{{}, {{1, 8, 16}}}, {16, 32}, 4ul},
    {{{}, {{1, 4, 16}}}, {1, 16, 32}},
    {{{}, {{10, 40, 496}}}, {1, 496, 240}},
    {{{}, {{1, 4, 48}}}, {48, 256}},
    {{{}, {{1, 11, 154}}}, {154, 77}, 154ul},
    {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {1, 480, 256}},
};
const std::vector<ShapeParams> input_shapes_amx = {
    {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {1, 480, 256}},
    {{{}, {{1, 4, 32}}}, {32, 256}},
    {{{}, {{1, 16, 32}}}, {32, 64}},
    {{{}, {{2, 4, 32}}}, {32, 65}},
    {{{}, {{3, 12, 768}}}, {768, 1024}},
    {{{}, {{11, 339, 577}}}, {577, 335}},
    {{{}, {{1, 1, 256}}}, {256, 128}, 64ul},
};
const std::vector<fusingSpecificParams> fusing_params {
    emptyFusingSpec,
    fusingBias,
    fusingFakeQuantizePerTensorRelu
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_basic,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(weights_precisions_basic),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(filterAdditionalConfigBasic()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_FP16,
                         MatmulWeightsDecompression_FP16,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(weights_precisions_basic),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(filterAdditionalConfig_FP16()),
                                            ::testing::ValuesIn(fusing_params),
                                            //only check if the tese case pass, not check the graph convert logic
                                            ::testing::Values(false)),
                         MatmulWeightsDecompression::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_amx,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_amx),
                                            ::testing::ValuesIn(weights_precisions_amx),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(filterAdditionalConfigAMX()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

const std::vector<ShapeParams> input_shapes_corner_cases_basic = {
    {{{-1, -1, -1}, {{1, 4, 16}}}, {1, 16, 32}},
    {{{-1, -1, -1}, {{1, 4, 16}}}, {16, 32}},
    {{{-1, -1, -1}, {{1, 4, 16}}}, {16, 32}, 4ul},
    {{{-1, -1, -1}, {{1, 1, 4096}}}, {4096, 4096}, 128ul},
};
const std::vector<ShapeParams> input_shapes_corner_cases_amx = {
    {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {1, 480, 256}},
    {{{-1, -1, -1}, {{1, 1, 4096}}}, {4096, 4096}, 128ul},
};

const std::vector<bool> transpose_weights = {true, false};
const std::vector<bool> add_decompression_sub = {true, false};
const std::vector<bool> reshape_on_decompression = {true, false};
const std::vector<ov::test::ElementType> decompression_precisions_corner_cases = {ov::element::f16, ov::element::f32};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases_basic,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_basic),
                                            ::testing::ValuesIn(weights_precisions_basic),
                                            ::testing::ValuesIn(decompression_precisions_corner_cases),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::ValuesIn(add_decompression_sub),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::ValuesIn(filterAdditionalConfigBasic()),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases_FP16,
                         MatmulWeightsDecompression_FP16,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_basic),
                                            ::testing::ValuesIn(weights_precisions_basic),
                                            ::testing::ValuesIn(decompression_precisions_corner_cases),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::ValuesIn(add_decompression_sub),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::ValuesIn(filterAdditionalConfig_FP16()),
                                            ::testing::Values(emptyFusingSpec),
                                            //only check if the tese case pass, not check the graph convert logic
                                            ::testing::Values(false)),
                         MatmulWeightsDecompression::getTestCaseName);



INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases_amx,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_amx),
                                            ::testing::ValuesIn(weights_precisions_amx),
                                            ::testing::ValuesIn(decompression_precisions_corner_cases),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::ValuesIn(add_decompression_sub),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::ValuesIn(filterAdditionalConfigAMX()),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

} // namespace
} // namespace SubgraphTestsDefinitions
