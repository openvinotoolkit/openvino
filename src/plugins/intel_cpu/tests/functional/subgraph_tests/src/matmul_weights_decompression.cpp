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
 *                        Subtract_const(U8/NF4)
 *                           /
 *    Weights(U8/NF4)     Convert(F32)
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
        bool transpose;
        bool decompression_sub;
        bool reshape_on_decompression;
        std::map<std::string, std::string> additional_config;
        fusingSpecificParams fusing_params;
        bool should_fuse;

        std::tie(shape_params,
                 weights_precision,
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
        auto weights = ngraph::builder::makeConstant<uint8_t>(weights_precision, transformed_weights_shape, {}, true);
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
            auto shift_const = ngraph::builder::makeConstant<uint8_t>(weights_precision, scaleshift_const_shape, {}, true);
            std::shared_ptr<ov::Node> shift_convert = std::make_shared<ngraph::opset1::Convert>(shift_const, data_precision);
            if (reshape_on_decompression_constant) {
                auto shift_reshape_const = ov::opset10::Constant::create(ov::element::i32, {scaleshift_target_shape.size()}, scaleshift_target_shape);
                auto shift_reshape = std::make_shared<ov::opset10::Reshape>(shift_convert, shift_reshape_const, false);
                shift_convert = shift_reshape;
            }
            mul_parent = std::make_shared<ov::opset10::Subtract>(weights_convert, shift_convert);
        }

        std::shared_ptr<ov::Node> scale_const = ngraph::builder::makeConstant<float>(data_precision, scaleshift_const_shape, {}, true);
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

    std::shared_ptr<ov::Model> initSubgraph(const ov::PartialShape& data_shape,
                                            const ov::Shape& weights_shape,
                                            const int group_size,
                                            const ov::element::Type data_precision,
                                            const ov::element::Type weights_precision,
                                            const bool transpose_weights,
                                            const bool add_subtract,
                                            const bool reshape_on_decompression) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(data_precision, data_shape)};
        const auto weights_subgraph = initDecompressionWeights(weights_shape,
                                                               group_size,
                                                               data_precision,
                                                               weights_precision,
                                                               transpose_weights,
                                                               add_subtract,
                                                               reshape_on_decompression);
        auto matMul = builder::makeMatMul(params[0], weights_subgraph);
        return makeNgraphFunction(data_precision, params, matMul, "MatmulWeightsDecompression");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        ShapeParams shape_params;
        ov::test::ElementType weights_precision;
        bool transpose_weights;
        bool decompression_sub;
        bool reshape_on_decompression;
        std::map<std::string, std::string> additional_config;
        fusingSpecificParams fusing_params;
        bool should_fuse;

        std::tie(shape_params,
                 weights_precision,
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
                                transpose_weights,
                                decompression_sub,
                                reshape_on_decompression);
    }

    void checkResults() {
        const auto& test_param = GetParam();
        const auto& weights_precision = std::get<1>(test_param);
        // TODO: remove this condition when group decompression is supported
        if (weights_precision == ov::element::nf4 || std::get<0>(test_param).weights_group_size != -1) {
            return;
        }
        bool weights_found = false;
        for (const auto& n : compiledModel.get_runtime_model()->get_ordered_ops()) {
            if (n->get_friendly_name() == "Compressed_weights") {
                ASSERT_EQ(n->get_output_element_type(0), weights_precision);
                weights_found = true;
            }
        }
        ASSERT_TRUE(weights_found);

        const bool should_fuse = std::get<7>(test_param);
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

namespace {

std::vector<std::map<std::string, std::string>> filterAdditionalConfigBasic() {
    std::vector<std::map<std::string, std::string>> additional_config = {CPUTestUtils::cpuEmptyPluginConfig};
    return additional_config;
}
std::vector<std::map<std::string, std::string>> filterAdditionalConfigBig() {
    std::vector<std::map<std::string, std::string>> additional_config = {CPUTestUtils::cpuEmptyPluginConfig};
    if (with_cpu_x86_avx512_core_amx())
        additional_config.push_back({{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}});
    return additional_config;
}

bool shouldUseDecompressionKernelBig() {
    // No decompression support on non-avx systems
    if (!with_cpu_x86_avx2())
        return false;

    return true;
}

bool shouldUseDecompressionKernelBasic() {
    // AMX decompression support has shape limitations
    if (with_cpu_x86_avx512_core_amx())
        return false;

    return shouldUseDecompressionKernelBig();
}

const std::vector<ov::test::ElementType> weights_precisions = {ov::element::u8, ov::element::nf4};
const std::vector<ShapeParams> input_shapes_basic = {
    {{{-1, -1, -1}, {{1, 4, 16}, {10, 16, 16}}}, {16, 32}},
    {{{}, {{1, 4, 16}}}, {16, 32}, 2ul},
    {{{}, {{1, 4, 16}}}, {1, 16, 32}},
    {{{}, {{10, 40, 496}}}, {1, 496, 240}},
    {{{}, {{1, 4, 48}}}, {48, 256}},
    {{{}, {{11, 339, 377}}}, {377, 335}},
};
const std::vector<ShapeParams> input_shapes_big = {
    {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {1, 480, 256}},
    {{{-1, 1, 4096}, {{1, 1, 4096}}}, {4096, 3840}, 128ul},
    {{{}, {{1, 4, 32}}}, {32, 256}},
    {{{}, {{1, 4, 512}}}, {512, 256}},
    {{{}, {{1, 16, 32}}}, {32, 64}},
    {{{}, {{2, 4, 32}}}, {32, 65}},
    {{{}, {{3, 12, 768}}}, {768, 1024}},
    {{{}, {{11, 339, 577}}}, {577, 335}},
};
const std::vector<fusingSpecificParams> fusingParamsSet {
    emptyFusingSpec,
    fusingBias,
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_basic,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(filterAdditionalConfigBasic()),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(shouldUseDecompressionKernelBasic())),
                         MatmulWeightsDecompression::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_big,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_big),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(filterAdditionalConfigBig()),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(shouldUseDecompressionKernelBig())),
                         MatmulWeightsDecompression::getTestCaseName);

const std::vector<ShapeParams> input_shapes_corner_cases_basic = {
    {{{-1, -1, -1}, {{1, 4, 16}}}, {1, 16, 32}},
    {{{-1, -1, -1}, {{1, 4, 16}}}, {16, 32}},
    {{{-1, -1, -1}, {{1, 4, 16}}}, {16, 32}, 4ul},
};
const std::vector<ShapeParams> input_shapes_corner_cases_big = {
    {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {1, 480, 256}},
    {{{-1, -1, -1}, {{1, 1, 4096}}}, {4096, 4096}, 128ul},
};

const std::vector<bool> transpose_weights = {true, false};
const std::vector<bool> add_decompression_sub = {true, false};
const std::vector<bool> reshape_on_decompression = {true, false};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases_basic,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_basic),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::ValuesIn(add_decompression_sub),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::ValuesIn(filterAdditionalConfigBasic()),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(shouldUseDecompressionKernelBasic())),
                         MatmulWeightsDecompression::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases_big,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_big),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::ValuesIn(add_decompression_sub),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::ValuesIn(filterAdditionalConfigBig()),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(shouldUseDecompressionKernelBig())),
                         MatmulWeightsDecompression::getTestCaseName);
} // namespace
} // namespace SubgraphTestsDefinitions
