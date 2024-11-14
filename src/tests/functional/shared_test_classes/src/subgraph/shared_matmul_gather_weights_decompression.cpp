// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/shared_matmul_gather_weights_decompression.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/subgraph/weights_decompression_builders.hpp"

namespace ov {
namespace test {
std::string SharedMatmulAndGatherWeightsDecompression::getTestCaseName(testing::TestParamInfo<SharedMatmulAndGatherWeightsDecompressionParams> obj) {
    std::string target_device;
    GatherDecompressionShapeParams shape_params;
    ov::test::ElementType weights_precision;
    ov::test::ElementType decompression_precision;
    bool decompression_subtract;
    bool use_decompression_impl;

    std::tie(target_device,
             shape_params,
             weights_precision,
             decompression_precision,
             decompression_subtract,
             use_decompression_impl) = obj.param;

    std::ostringstream result;
    result << "device=" << target_device << "_";
    result << shape_params << "_";
    result << "weights_precision=" << weights_precision << "_";
    result << "decompression_precision=" << decompression_precision << "_";
    result << "decompression_subtract=" << decompression_subtract << "_";
    result << "use_decompression_impl=" << use_decompression_impl;
    return result.str();
}

std::shared_ptr<ov::Model> SharedMatmulAndGatherWeightsDecompression::initSubgraph(const ov::Shape& data_shape,
                                                                                   const ov::PartialShape& indices_shape,
                                                                                   const int axis,
                                                                                   const int64_t batch_dims,
                                                                                   const int group_size,
                                                                                   const ov::element::Type data_precision,
                                                                                   const ov::element::Type output_precision,
                                                                                   const bool add_subtract) {
    const auto indices_data = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, indices_shape);
    const auto axis_const = ov::op::v0::Constant::create(ov::element::i32, {1}, {axis});
    const auto decompression_subgraph = initGatherDecompressionSubgraph(data_shape,
                                                                        group_size,
                                                                        data_precision,
                                                                        output_precision,
                                                                        add_subtract,
                                                                        false,
                                                                        false,
                                                                        false);
    const auto gather = std::make_shared<ov::op::v8::Gather>(decompression_subgraph, indices_data, axis_const, batch_dims);

    const auto fc_data = std::make_shared<ov::op::v0::Parameter>(output_precision, indices_shape);
    const auto matmul = std::make_shared<ov::op::v0::MatMul>(fc_data, decompression_subgraph, false, true);
    const ov::OutputVector last_nodes{gather, matmul};
    const ov::ParameterVector params{indices_data, fc_data};

    // if dynamic quantization is enabled
    if (group_size != 0) {
        abs_threshold = 0.15;
    }

    return std::make_shared<ov::Model>(last_nodes, params, "SharedMatmulAndGatherWeightsDecompression");
}

void SharedMatmulAndGatherWeightsDecompression::SetUp() {
    GatherDecompressionShapeParams shape_params;
    ov::test::ElementType weights_precision;
    ov::test::ElementType decompression_precision;
    bool decompression_subtract;
    bool use_decompression_impl;

    std::tie(targetDevice,
             shape_params,
             weights_precision,
             decompression_precision,
             decompression_subtract,
             use_decompression_impl) = GetParam();

    init_input_shapes({shape_params.indices_shape, shape_params.indices_shape});

    function = initSubgraph(shape_params.data_shape,
                            inputDynamicShapes[0],
                            shape_params.axis,
                            shape_params.batch_dims,
                            shape_params.decompression_group_size,
                            weights_precision,
                            decompression_precision,
                            decompression_subtract);
}

void SharedMatmulAndGatherWeightsDecompression::check_results() {
    const auto& test_param = GetParam();
    const ov::element::Type compressed_weights_precision = std::get<2>(test_param);
    const auto use_matmul_decompression_impl = std::get<5>(test_param);

    const auto results = compiledModel.get_runtime_model()->get_results();
    EXPECT_EQ(results.size(), 2);
    const auto gather_node = results[0]->get_input_node_shared_ptr(0);
    EXPECT_EQ(gather_node->get_input_element_type(0), compressed_weights_precision);

    const auto matmul_node = results[1]->get_input_node_shared_ptr(0);
    const auto& expected_mm_weights_precision = use_matmul_decompression_impl
                                                    ? compressed_weights_precision
                                                    : matmul_node->get_input_element_type(0);
    EXPECT_EQ(matmul_node->get_input_element_type(1), expected_mm_weights_precision);
}

}  // namespace test
}  // namespace ov
