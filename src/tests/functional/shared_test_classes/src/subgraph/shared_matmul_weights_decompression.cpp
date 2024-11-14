// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/shared_matmul_weights_decompression.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/subgraph/weights_decompression_builders.hpp"

namespace ov {
namespace test {
std::string SharedMatmulWeightsDecompression::getTestCaseName(testing::TestParamInfo<MatmulSharedWeightsDecompressionParams> obj) {
    std::string target_device;
    MatMulDecompressionShapeParams shape_params;
    ov::test::ElementType weights_precision;
    ov::test::ElementType decompression_precision;
    bool transpose;
    DecompressionSubtractType decompression_subtract_type;
    bool use_decompression_impl;

    std::tie(target_device,
             shape_params,
             weights_precision,
             decompression_precision,
             transpose,
             decompression_subtract_type,
             use_decompression_impl) = obj.param;

    std::ostringstream result;
    result << "device=" << target_device << "_";
    result << shape_params << "_";
    result << "weights_precision=" << weights_precision << "_";
    result << "decompression_precision=" << decompression_precision << "_";
    result << "transpose_weights=" << transpose << "_";
    result << "decompression_subtract=" << decompression_subtract_type << "_";
    result << "use_decompression_impl=" << use_decompression_impl;
    return result.str();
}

std::shared_ptr<ov::Model> SharedMatmulWeightsDecompression::initSubgraph(
    const ov::PartialShape& data_shape,
    const ov::Shape& weights_shape,
    const int group_size,
    const ov::element::Type data_precision,
    const ov::element::Type weights_precision,
    const ov::element::Type decompression_precision,
    const bool transpose_weights,
    const DecompressionSubtractType decompression_subtract_type) {
    const auto weights_subgraph = initMatMulDecompressionSubgraph(weights_shape,
                                                                  group_size,
                                                                  data_precision,
                                                                  weights_precision,
                                                                  decompression_precision,
                                                                  ov::element::undefined,
                                                                  transpose_weights,
                                                                  decompression_subtract_type,
                                                                  false);
    ov::ParameterVector params;
    ov::OutputVector last_layers;
    for (size_t i = 0; i < 2; ++i) {
        const auto param = std::make_shared<ov::op::v0::Parameter>(data_precision, data_shape);
        auto shared_weights_input = weights_subgraph;
        // In real cases, transpose is not shared between MatMuls,
        // so we recreate the own copy of transpose for each matmul
        if (transpose_weights) {
            OPENVINO_ASSERT(ov::is_type<ov::opset10::Transpose>(shared_weights_input));
            shared_weights_input = weights_subgraph->clone_with_new_inputs(weights_subgraph->input_values());
        }
        const auto matMul = std::make_shared<ov::op::v0::MatMul>(param, shared_weights_input);
        params.push_back(param);
        last_layers.push_back(matMul);
    }

    // if dynamic quantization is enabled
    if (group_size != 0) {
        abs_threshold = 0.1;
    }

    return std::make_shared<ov::Model>(last_layers, params, "SharedMatmulWeightsDecompression");
}

void SharedMatmulWeightsDecompression::SetUp() {
    MatMulDecompressionShapeParams shape_params;
    ov::test::ElementType weights_precision;
    ov::test::ElementType decompression_precision;
    bool transpose_weights;
    DecompressionSubtractType decompression_subtract_type;
    bool use_decompression_impl;

    std::tie(targetDevice,
             shape_params,
             weights_precision,
             decompression_precision,
             transpose_weights,
             decompression_subtract_type,
             use_decompression_impl) = GetParam();
    init_input_shapes({shape_params.data_shape, shape_params.data_shape});

    ElementType netType = ov::element::f32;
    inType = outType = netType;

    function = initSubgraph(inputDynamicShapes[0],
                            shape_params.weights_shape,
                            shape_params.decompression_group_size,
                            netType,
                            weights_precision,
                            decompression_precision,
                            transpose_weights,
                            decompression_subtract_type);
}

void SharedMatmulWeightsDecompression::check_results() {
    const auto& test_param = GetParam();
    const ov::element::Type compressed_weights_precision = std::get<2>(test_param);
    const auto use_matmul_decompression_impl = std::get<6>(test_param);

    const auto results = compiledModel.get_runtime_model()->get_results();
    for (const auto& result : results) {
        const auto last_layer = result->get_input_node_shared_ptr(0);
        const auto& expected_weights_precision = use_matmul_decompression_impl
                                                     ? compressed_weights_precision
                                                     : last_layer->get_input_element_type(0);
        EXPECT_EQ(last_layer->get_input_element_type(1), expected_weights_precision);
    }
}

}  // namespace test
}  // namespace ov