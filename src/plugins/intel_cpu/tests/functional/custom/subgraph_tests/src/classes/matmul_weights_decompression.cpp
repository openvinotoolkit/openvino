// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul_weights_decompression.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

std::string MatmulWeightsDecompression::getTestCaseName(testing::TestParamInfo<MatmulWeightsDecompressionParams> obj) {
    MatMulDecompressionShapeParams shape_params;
    ov::test::ElementType weights_precision;
    ov::test::ElementType decompression_precision;
    ov::test::ElementType scale_precision;
    bool transpose;
    DecompressionType decompression_multiply_type;
    DecompressionType decompression_subtract_type;
    bool reshape_on_decompression;
    ov::AnyMap additional_config;
    fusingSpecificParams fusing_params;
    bool should_fuse;

    std::tie(shape_params,
             weights_precision,
             decompression_precision,
             scale_precision,
             transpose,
             decompression_multiply_type,
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
    result << "decompression_multiply=" << decompression_multiply_type << "_";
    result << "decompression_subtract=" << decompression_subtract_type << "_";
    result << "reshape_on_decompression=" << reshape_on_decompression << "_";

    result << "config=(";
    for (const auto& configEntry : additional_config) {
        result << configEntry.first << ", " << configEntry.second.as<std::string>() << "_";
    }
    result << ")";
    result << CpuTestWithFusing::getTestCaseName(fusing_params);

    return result.str();
}

std::shared_ptr<ov::Model> MatmulWeightsDecompression::initSubgraph(const ov::PartialShape& data_shape,
                                                                    const ov::Shape& weights_shape,
                                                                    const int group_size,
                                                                    const ov::element::Type data_precision,
                                                                    const ov::element::Type weights_precision,
                                                                    const ov::element::Type decompression_precision,
                                                                    const ov::element::Type scale_precision,
                                                                    const bool transpose_weights,
                                                                    const DecompressionType decompression_multiply_type,
                                                                    const DecompressionType decompression_subtract_type,
                                                                    const bool reshape_on_decompression) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(data_precision, data_shape)};
    const auto weights_subgraph = initMatMulDecompressionSubgraph(weights_shape,
                                                                    group_size,
                                                                    data_precision,
                                                                    weights_precision,
                                                                    decompression_precision,
                                                                    scale_precision,
                                                                    transpose_weights,
                                                                    decompression_multiply_type,
                                                                    decompression_subtract_type,
                                                                    reshape_on_decompression);
    auto matMul = std::make_shared<ov::op::v0::MatMul>(params[0], weights_subgraph);
    return makeNgraphFunction(data_precision, params, matMul, "MatmulWeightsDecompression");
}

void MatmulWeightsDecompression::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    MatMulDecompressionShapeParams shape_params;
    ov::test::ElementType weights_precision;
    ov::test::ElementType decompression_precision;
    ov::test::ElementType scale_precision;
    bool transpose_weights;
    DecompressionType decompression_multiply_type;
    DecompressionType decompression_subtract_type;
    bool reshape_on_decompression;
    ov::AnyMap additional_config;
    fusingSpecificParams fusing_params;
    bool should_fuse;

    std::tie(shape_params,
                weights_precision,
                decompression_precision,
                scale_precision,
                transpose_weights,
                decompression_multiply_type,
                decompression_subtract_type,
                reshape_on_decompression,
                additional_config,
                fusing_params,
                should_fuse) = GetParam();

    configuration.insert(additional_config.begin(), additional_config.end());
    std::tie(postOpMgrPtr, fusedOps) = fusing_params;
    init_input_shapes({shape_params.data_shape});

    if (!configuration.count(ov::hint::dynamic_quantization_group_size.name())) {
        abs_threshold = 5e-3;
    }

    // if dynamic quantization is enabled
    if (configuration.count(ov::hint::dynamic_quantization_group_size.name()) &&
        configuration.at(ov::hint::dynamic_quantization_group_size.name()) != 0) {
        abs_threshold = 0.1;
    }

    if (configuration.count(ov::hint::inference_precision.name()) &&
        configuration.at(ov::hint::inference_precision.name()) == ov::element::f16) {
        abs_threshold = 0.2;
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
                            decompression_multiply_type,
                            decompression_subtract_type,
                            reshape_on_decompression);
}

void MatmulWeightsDecompression::check_results() {
    const auto& test_param = GetParam();
    const ov::element::Type compressed_weights_precision = std::get<1>(test_param);
    const bool use_matmul_decompression_impl = std::get<10>(test_param);

    const auto runtime_model = compiledModel.get_runtime_model();
    const auto result = runtime_model->get_result();
    auto fc = result->get_input_node_shared_ptr(0);
    // Handle precision conversion before output
    auto type = fc->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
    if (type == "Reorder" || type == "Convert" || type == "Subgraph")
        fc = fc->get_input_node_shared_ptr(0);

    type = fc->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
    EXPECT_EQ(type, "FullyConnected");

    const auto& expected_weights_precision = use_matmul_decompression_impl
                                                    ? compressed_weights_precision
                                                    : fc->get_input_element_type(0);
    EXPECT_EQ(fc->get_input_element_type(1), expected_weights_precision);
}

TEST_P(MatmulWeightsDecompression, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    check_results();
}

}  // namespace test
}  // namespace ov
