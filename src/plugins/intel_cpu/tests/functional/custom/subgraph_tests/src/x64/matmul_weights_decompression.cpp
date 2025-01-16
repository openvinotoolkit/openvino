// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/fusing_test_utils.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "shared_test_classes/subgraph/weights_decompression_builders.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
/*
 * WP - weights precision
 * DP - decompression precision
 * IP - input precision
 * SP - scale precision
 * Opt - optional
 *                        Subtract_const(WP)
 *                           /
 *    Weights(WP)     Convert(DP)
 *       |               /           Multiply_const(SP)
 *    Convert(DP)   Reshape (Opt)      /
 *            \        /          Convert(if SP != DP)
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
using MatmulWeightsDecompressionParams = std::tuple<MatMulDecompressionShapeParams,
                                                    ov::test::ElementType,      // weights precision
                                                    ov::test::ElementType,      // decompression precision
                                                    ov::test::ElementType,      // scale precision
                                                    bool,                       // transpose on weights
                                                    DecompressionSubtractType,  // decompression subtract type
                                                    bool,                       // reshape on decompression constants
                                                    ov::AnyMap,                 // additional config
                                                    fusingSpecificParams,
                                                    bool>;                      // should use decompression implementation

class MatmulWeightsDecompression : public testing::WithParamInterface<MatmulWeightsDecompressionParams>,
                                   virtual public SubgraphBaseTest,
                                   public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatmulWeightsDecompressionParams> obj) {
        MatMulDecompressionShapeParams shape_params;
        ov::test::ElementType weights_precision;
        ov::test::ElementType decompression_precision;
        ov::test::ElementType scale_precision;
        bool transpose;
        DecompressionSubtractType decompression_subtract_type;
        bool reshape_on_decompression;
        ov::AnyMap additional_config;
        fusingSpecificParams fusing_params;
        bool should_fuse;

        std::tie(shape_params,
                 weights_precision,
                 decompression_precision,
                 scale_precision,
                 transpose,
                 decompression_subtract_type,
                 reshape_on_decompression,
                 additional_config,
                 fusing_params,
                 should_fuse) = obj.param;

        std::ostringstream result;
        result << shape_params << "_";
        result << "weights_precision=" << weights_precision << "_";
        result << "decompression_precision=" << decompression_precision << "_";
        result << "scale_precision=" << scale_precision << "_";
        result << "transpose_weights=" << transpose << "_";
        result << "decompression_subtract=" << decompression_subtract_type << "_";
        result << "reshape_on_decompression=" << reshape_on_decompression << "_";

        result << "config=(";
        for (const auto& configEntry : additional_config) {
            result << configEntry.first << ", " << configEntry.second.as<std::string>() << ":";
        }
        result << ")";
        result << CpuTestWithFusing::getTestCaseName(fusing_params);

        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> initSubgraph(const ov::PartialShape& data_shape,
                                            const ov::Shape& weights_shape,
                                            const int group_size,
                                            const ov::element::Type data_precision,
                                            const ov::element::Type weights_precision,
                                            const ov::element::Type decompression_precision,
                                            const ov::element::Type scale_precision,
                                            const bool transpose_weights,
                                            const DecompressionSubtractType decompression_subtract_type,
                                            const bool reshape_on_decompression) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(data_precision, data_shape)};
        const auto weights_subgraph = initMatMulDecompressionSubgraph(weights_shape,
                                                                      group_size,
                                                                      data_precision,
                                                                      weights_precision,
                                                                      decompression_precision,
                                                                      scale_precision,
                                                                      transpose_weights,
                                                                      decompression_subtract_type,
                                                                      reshape_on_decompression);
        auto matMul = std::make_shared<ov::op::v0::MatMul>(params[0], weights_subgraph);
        return makeNgraphFunction(data_precision, params, matMul, "MatmulWeightsDecompression");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        MatMulDecompressionShapeParams shape_params;
        ov::test::ElementType weights_precision;
        ov::test::ElementType decompression_precision;
        ov::test::ElementType scale_precision;
        bool transpose_weights;
        DecompressionSubtractType decompression_subtract_type;
        bool reshape_on_decompression;
        ov::AnyMap additional_config;
        fusingSpecificParams fusing_params;
        bool should_fuse;

        std::tie(shape_params,
                 weights_precision,
                 decompression_precision,
                 scale_precision,
                 transpose_weights,
                 decompression_subtract_type,
                 reshape_on_decompression,
                 additional_config,
                 fusing_params,
                 should_fuse) = GetParam();

        configuration.insert(additional_config.begin(), additional_config.end());
        std::tie(postOpMgrPtr, fusedOps) = fusing_params;
        init_input_shapes({shape_params.data_shape});

        // if dynamic quantization is enabled
        if (configuration.count(ov::hint::dynamic_quantization_group_size.name()) &&
            configuration.at(ov::hint::dynamic_quantization_group_size.name()) != 0) {
            abs_threshold = 0.1;
        } else if (!configuration.count(ov::hint::dynamic_quantization_group_size.name())) {
            abs_threshold = 5e-3;
        }

        ElementType netType = ov::element::f32;
        inType = outType = netType;

        function = initSubgraph(inputDynamicShapes[0],
                                shape_params.weights_shape,
                                shape_params.decompression_group_size,
                                netType,
                                weights_precision,
                                decompression_precision,
                                scale_precision,
                                transpose_weights,
                                decompression_subtract_type,
                                reshape_on_decompression);
    }

    void check_results() {
        const auto& test_param = GetParam();
        const ov::element::Type compressed_weights_precision = std::get<1>(test_param);
        const bool use_matmul_decompression_impl = std::get<9>(test_param);

        const auto runtime_model = compiledModel.get_runtime_model();
        const auto result = runtime_model->get_result();
        const auto fc = result->get_input_node_shared_ptr(0);
        const auto type = fc->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
        EXPECT_EQ(type, "FullyConnected");

        const auto& expected_weights_precision = use_matmul_decompression_impl
                                                     ? compressed_weights_precision
                                                     : fc->get_input_element_type(0);
        EXPECT_EQ(fc->get_input_element_type(1), expected_weights_precision);
    }
};

TEST_P(MatmulWeightsDecompression, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    check_results();
}

namespace {

std::vector<ov::AnyMap> filter_additional_config_basic() {
    std::vector<ov::AnyMap> additional_config = {{ov::hint::dynamic_quantization_group_size(0)}};
    return additional_config;
}
std::vector<ov::AnyMap> filter_additional_config_amx() {
    std::vector<ov::AnyMap> additional_config = {};
    if (ov::with_cpu_x86_avx512_core_amx())
        additional_config.push_back({{ov::hint::dynamic_quantization_group_size(0), ov::hint::inference_precision(ov::element::bf16)}});
    return additional_config;
}

const std::vector<ov::test::ElementType> decompression_precisions = {ov::element::f32};
const std::vector<ov::test::ElementType> weights_precisions = {ov::element::u8,
                                                               ov::element::u4,
                                                               ov::element::i4,
                                                               ov::element::nf4};

const std::vector<MatMulDecompressionShapeParams> input_shapes_basic = {
    {{{-1, -1, -1}, {{1, 4, 16}, {10, 16, 16}}}, {16, 32}},
    {{{}, {{1, 8, 16}}}, {16, 32}, 4ul},
    {{{}, {{1, 4, 16}}}, {1, 16, 32}},
    {{{}, {{5, 40, 496}}}, {1, 496, 240}},
    {{{}, {{1, 4, 48}}}, {48, 256}},
    {{{}, {{1, 11, 154}}}, {154, 77}, 154ul},
    {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {1, 480, 256}},
};
const std::vector<MatMulDecompressionShapeParams> input_shapes_amx = {
    {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {1, 480, 256}},
    {{{}, {{1, 4, 32}}}, {32, 256}},
    {{{}, {{1, 16, 32}}}, {32, 64}},
    {{{}, {{2, 4, 32}}}, {32, 65}},
    {{{}, {{3, 12, 768}}}, {768, 1024}},
    {{{}, {{3, 339, 577}}}, {577, 335}},
    {{{}, {{1, 1, 256}}}, {256, 128}, 64ul},
};
const std::vector<fusingSpecificParams> fusing_params{emptyFusingSpec, fusingBias};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_basic,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::undefined),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionSubtractType::full),
                                            // todo: zero points converted to fp32 for reshape == true case
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_amx,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_amx),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::undefined),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionSubtractType::full),
                                            // todo: zero points converted to fp32 for reshape == true case
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_amx()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

// symmetric weight compression : i4/i8 with no/empty DecompressionSubtract
const std::vector<ov::test::ElementType> sym_weights_precisions = {ov::element::i8, ov::element::i4};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_sym,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(sym_weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::undefined),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionSubtractType::empty),
                                            // todo: zero points converted to fp32 for reshape == true case
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_sym_amx,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_amx),
                                            ::testing::ValuesIn(sym_weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::undefined),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionSubtractType::empty),
                                            // todo: zero points converted to fp32 for reshape == true case
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_amx()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

const std::vector<MatMulDecompressionShapeParams> input_shapes_corner_cases_basic = {
    {{{-1, -1, -1}, {{1, 4, 16}}}, {1, 16, 32}},
    {{{-1, -1, -1}, {{1, 4, 16}}}, {16, 32}},
    {{{-1, -1, -1}, {{1, 5, 16}}}, {16, 32}, 4ul},
    {{{-1, -1, -1}, {{1, 1, 4096}}}, {4096, 4096}, 128ul},
};
const std::vector<MatMulDecompressionShapeParams> input_shapes_corner_cases_amx = {
    {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {1, 480, 256}},
    {{{-1, -1, -1}, {{1, 1, 4096}}}, {4096, 4096}, 128ul},
};

const std::vector<bool> transpose_weights = {true, false};
const std::vector<DecompressionSubtractType> decompression_subtract_type = {
    DecompressionSubtractType::full, DecompressionSubtractType::scalar, DecompressionSubtractType::empty};
const std::vector<bool> reshape_on_decompression = {true, false};
const std::vector<ov::test::ElementType> decompression_precisions_corner_cases = {ov::element::f16, ov::element::f32};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases_basic,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_basic),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions_corner_cases),
                                            ::testing::Values(ov::element::undefined),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::ValuesIn(decompression_subtract_type),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

const std::vector<MatMulDecompressionShapeParams> input_shapes_f32_decompression_f16_scale = {
    {{{}, {{1, 8, 16}}}, {16, 32}},
    {{{}, {{1, 8, 16}}}, {16, 32}, 4ul},
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_f32_decompression_f16_scale,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_f32_decompression_f16_scale),
                                            ::testing::Values(ov::element::u8),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::Values(DecompressionSubtractType::full),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

const std::vector<MatMulDecompressionShapeParams> input_shapes_corner_cases_negative = {
    {{{-1, -1, -1}, {{1, 512, 512}}}, {512, 1}},
    {{{-1, -1, -1}, {{1, 5, 32}}}, {32, 64}, 2ul},
};
INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases_negative,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_negative),
                                            ::testing::Values(ov::element::u8),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::undefined),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionSubtractType::empty),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(false)),
                         MatmulWeightsDecompression::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases_amx,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_amx),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions_corner_cases),
                                            ::testing::Values(ov::element::undefined),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::ValuesIn(decompression_subtract_type),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::ValuesIn(filter_additional_config_amx()),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

const std::vector<MatMulDecompressionShapeParams> input_shapes_basic_dyn_quant = {
    {{{}, {{1, 7, 256}}}, {256, 128}, 32lu},
    {{{}, {{1, 1, 128}}}, {128, 32}},
    {{{}, {{1, 3, 144}}}, {144, 64}, 16lu},
    {{{}, {{1, 1, 1728}}}, {1728, 128}, 64lu},
};

const std::vector<ov::test::ElementType> weights_precisions_dyn_quant = {ov::element::u8,
                                                                         ov::element::u4};

std::vector<ov::AnyMap> filter_additional_config_dyn_quant() {
    std::vector<ov::AnyMap> additional_config = {
        {{ov::hint::dynamic_quantization_group_size(0)}}, // dynamic quantization is disabled
        {{ov::hint::dynamic_quantization_group_size(16)}},
        {{ov::hint::dynamic_quantization_group_size(128)}},
    };
    return additional_config;
}

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_non_default_dyn_quant_group_sizes,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic_dyn_quant),
                                            ::testing::ValuesIn(weights_precisions_dyn_quant),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::undefined),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(decompression_subtract_type),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_dyn_quant()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

const std::vector<ov::test::ElementType> sym_weights_precisions_dyn_quant = {ov::element::i8, ov::element::i4};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_sym_non_default_dyn_quant_group_sizes,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic_dyn_quant),
                                            ::testing::ValuesIn(sym_weights_precisions_dyn_quant),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::undefined),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionSubtractType::empty),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_dyn_quant()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_mxfp4,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic_dyn_quant),
                                            ::testing::Values(ov::element::f4e2m1),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f8e8m0),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionSubtractType::empty),
                                            // todo: zero points converted to fp32 for reshape == true case
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
