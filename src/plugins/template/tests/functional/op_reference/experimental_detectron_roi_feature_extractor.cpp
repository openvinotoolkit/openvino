// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/experimental_detectron_roi_feature.hpp"

using namespace ov;
using namespace reference_tests;

struct ExperimentalROIParams {
    ExperimentalROIParams(const std::vector<reference_tests::Tensor>& experimental_detectron_roi_feature_inputs,
                          const std::vector<reference_tests::Tensor>& expected_results,
                          const std::string& test_case_name)
        : inputs{experimental_detectron_roi_feature_inputs},
          expected_results{expected_results},
          test_case_name{test_case_name} {}

    std::vector<reference_tests::Tensor> inputs;
    std::vector<reference_tests::Tensor> expected_results;
    std::string test_case_name;
};

class ReferenceExperimentalROILayerTest : public testing::TestWithParam<ExperimentalROIParams>,
                                          public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = create_function(params.inputs);
        inputData.reserve(params.inputs.size());
        refOutData.reserve(params.expected_results.size());
        for (const auto& input_tensor : params.inputs) {
            inputData.push_back(input_tensor.data);
        }
        for (const auto& expected_tensor : params.expected_results) {
            refOutData.push_back(expected_tensor.data);
        }
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ExperimentalROIParams>& obj) {
        auto param = obj.param;
        return param.test_case_name;
    }

private:
    std::shared_ptr<Model> create_function(const std::vector<reference_tests::Tensor>& inputs) {
        op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes attrs;
        attrs.aligned = false;
        attrs.output_size = 3;
        attrs.sampling_ratio = 2;
        attrs.pyramid_scales = {4};

        const size_t num_of_inputs = inputs.size();
        NodeVector node_vector(num_of_inputs);
        ParameterVector parameter_vector(num_of_inputs);
        for (size_t i = 0; i < num_of_inputs; ++i) {
            const auto& current_input = inputs[i];
            auto current_parameter = std::make_shared<op::v0::Parameter>(current_input.type, current_input.shape);
            node_vector[i] = current_parameter;
            parameter_vector[i] = current_parameter;
        }

        auto roi = std::make_shared<op::v6::ExperimentalDetectronROIFeatureExtractor>(node_vector, attrs);
        auto fun = std::make_shared<ov::Model>(OutputVector{roi->output(0), roi->output(1)}, parameter_vector);
        return fun;
    }
};

TEST_P(ReferenceExperimentalROILayerTest, ExperimentalROIWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_ExperimentalROI_With_Hardcoded_Refs,
    ReferenceExperimentalROILayerTest,
    ::testing::Values(
        ExperimentalROIParams(
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(Shape{2, 4},
                                        ov::element::f32,
                                        std::vector<float>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}),
                reference_tests::Tensor(
                    Shape{1, 2, 2, 3},
                    ov::element::f32,
                    std::vector<float>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0})},
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(
                    Shape{2, 2, 3, 3},
                    ov::element::f32,
                    std::vector<float>{1.416667, 1.75, 2.083333, 2.416667, 2.75, 3.083333, 3.166667, 3.5,  3.833333,
                                       7.416667, 7.75, 8.083333, 8.416667, 8.75, 9.083334, 9.166666, 9.5,  9.833334,
                                       4.166667, 4.5,  4.833333, 4.166667, 4.5,  4.833333, 2.083333, 2.25, 2.416667,
                                       10.16667, 10.5, 10.83333, 10.16667, 10.5, 10.83333, 5.083333, 5.25, 5.416667}),
                reference_tests::Tensor(Shape{2, 4},
                                        ov::element::f32,
                                        std::vector<float>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0})},
            "experimental_detectron_roi_feature_eval_f32"),
        ExperimentalROIParams(
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(Shape{2, 4},
                                        ov::element::f16,
                                        std::vector<ov::float16>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}),
                reference_tests::Tensor(
                    Shape{1, 2, 2, 3},
                    ov::element::f16,
                    std::vector<ov::float16>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0})},
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(Shape{2, 2, 3, 3},
                                        ov::element::f16,
                                        std::vector<ov::float16>{1.416667, 1.75, 2.083333, 2.416667, 2.75, 3.083333,
                                                                 3.166667, 3.5,  3.833333, 7.416667, 7.75, 8.083333,
                                                                 8.416667, 8.75, 9.083334, 9.166666, 9.5,  9.833334,
                                                                 4.166667, 4.5,  4.833333, 4.166667, 4.5,  4.833333,
                                                                 2.083333, 2.25, 2.416667, 10.16667, 10.5, 10.83333,
                                                                 10.16667, 10.5, 10.83333, 5.083333, 5.25, 5.416667}),
                reference_tests::Tensor(Shape{2, 4},
                                        ov::element::f16,
                                        std::vector<ov::float16>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0})},
            "experimental_detectron_roi_feature_eval_f16"),
        ExperimentalROIParams(
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(Shape{2, 4},
                                        ov::element::bf16,
                                        std::vector<ov::bfloat16>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}),
                reference_tests::Tensor(
                    Shape{1, 2, 2, 3},
                    ov::element::bf16,
                    std::vector<ov::bfloat16>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0})},
            std::vector<reference_tests::Tensor>{
                reference_tests::Tensor(Shape{2, 2, 3, 3},
                                        ov::element::bf16,
                                        std::vector<ov::bfloat16>{1.416667, 1.75, 2.083333, 2.416667, 2.75, 3.083333,
                                                                  3.166667, 3.5,  3.833333, 7.416667, 7.75, 8.083333,
                                                                  8.416667, 8.75, 9.083334, 9.166666, 9.5,  9.833334,
                                                                  4.166667, 4.5,  4.833333, 4.166667, 4.5,  4.833333,
                                                                  2.083333, 2.25, 2.416667, 10.16667, 10.5, 10.83333,
                                                                  10.16667, 10.5, 10.83333, 5.083333, 5.25, 5.416667}),
                reference_tests::Tensor(Shape{2, 4},
                                        ov::element::bf16,
                                        std::vector<ov::bfloat16>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0})},
            "experimental_detectron_roi_feature_eval_bf16")));
