// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <tuple>

#include "base_reference_test.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace reference_tests;

struct ExperimentalROIFunctional {
    std::shared_ptr<Function> create_function(const std::vector<Tensor>& ed_inputs,
                                              const std::vector<Tensor>& results,
                                              const std::vector<Shape>& input_shapes,
                                              const ngraph::element::Type& et) {
        op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes attrs;
        attrs.aligned = false;
        attrs.output_size = 3;
        attrs.sampling_ratio = 2;
        attrs.pyramid_scales = {4};

        const size_t num_of_inputs = input_shapes.size();
        NodeVector node_vector(num_of_inputs);
        ParameterVector parameter_vector(num_of_inputs);
        for (size_t i = 0; i < num_of_inputs; ++i) {
            auto current_parameter = std::make_shared<op::Parameter>(et, input_shapes[i]);
            node_vector[i] = current_parameter;
            parameter_vector[i] = current_parameter;
        }

        auto roi = std::make_shared<op::v6::ExperimentalDetectronROIFeatureExtractor>(node_vector, attrs);
        auto fun = std::make_shared<Function>(OutputVector{roi->output(0), roi->output(1)}, parameter_vector);
        return fun;
    }
};

struct ExperimentalROIParams {
    ExperimentalROIParams(const std::shared_ptr<ExperimentalROIFunctional>& functional,
                          const std::vector<Shape>& input_shapes,
                          const ngraph::element::Type& et,
                          const std::vector<Tensor>& experimental_detectron_roi_feature_inputs,
                          const std::vector<Tensor>& expected_results,
                          const std::string& test_case_name)
        : function{functional},
          input_shapes{input_shapes},
          et{et},
          inputs{experimental_detectron_roi_feature_inputs},
          expected_results{expected_results},
          test_case_name{test_case_name} {}

    std::shared_ptr<ExperimentalROIFunctional> function;
    std::vector<Shape> input_shapes;
    ngraph::element::Type et;
    std::vector<Tensor> inputs;
    std::vector<Tensor> expected_results;
    std::string test_case_name;
};

class ReferenceExperimentalROILayerTest : public testing::TestWithParam<ExperimentalROIParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = params.function->create_function(params.inputs, params.expected_results, params.input_shapes, params.et);
        inputData.reserve(params.inputs.size());
        refOutData.reserve(params.expected_results.size());
        for (auto& input_tensor : params.inputs) {
            inputData.push_back(input_tensor.data);
        }
        for (auto& expected_tensor : params.expected_results) {
            refOutData.push_back(expected_tensor.data);
        }
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ExperimentalROIParams>& obj) {
        auto param = obj.param;
        return param.test_case_name;
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
            std::make_shared<ExperimentalROIFunctional>(),
            std::vector<Shape>{Shape{2, 4}, Shape{1, 2, 2, 3}},
            ngraph::element::f32,
            std::vector<Tensor>{Tensor(Shape{2, 4},
                                       ngraph::element::f32,
                                       std::vector<float>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}),
                                Tensor(Shape{1, 2, 2, 3},
                                       ngraph::element::f32,
                                       std::vector<float>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0})},
            std::vector<Tensor>{Tensor(Shape{2, 2, 3, 3},
                                       ngraph::element::f32,
                                       std::vector<float>{1.416667,
                                                          1.75,
                                                          2.083333,
                                                          2.416667,
                                                          2.75,
                                                          3.083333,
                                                          3.166667,
                                                          3.5,
                                                          3.833333,
                                                          7.416667,
                                                          7.75,
                                                          8.083333,
                                                          8.416667,
                                                          8.75,
                                                          9.083334,
                                                          9.166666,
                                                          9.5,
                                                          9.833334,
                                                          4.166667,
                                                          4.5,
                                                          4.833333,
                                                          4.166667,
                                                          4.5,
                                                          4.833333,
                                                          2.083333,
                                                          2.25,
                                                          2.416667,
                                                          10.16667,
                                                          10.5,
                                                          10.83333,
                                                          10.16667,
                                                          10.5,
                                                          10.83333,
                                                          5.083333,
                                                          5.25,
                                                          5.416667}),
                                Tensor(Shape{2, 4},
                                       ngraph::element::f32,
                                       std::vector<float>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0})},
            "experimental_detectron_roi_feature_eval_f32"),
        ExperimentalROIParams(
            std::make_shared<ExperimentalROIFunctional>(),
            std::vector<Shape>{Shape{2, 4}, Shape{1, 2, 2, 3}},
            ngraph::element::f32,
            std::vector<Tensor>{Tensor(Shape{2, 4},
                                       ngraph::element::f16,
                                       std::vector<ngraph::float16>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}),
                                Tensor(Shape{1, 2, 2, 3},
                                       ngraph::element::f16,
                                       std::vector<ngraph::float16>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0})},
            std::vector<Tensor>{Tensor(Shape{2, 2, 3, 3},
                                       ngraph::element::f16,
                                       std::vector<ngraph::float16>{1.416667,
                                                                    1.75,
                                                                    2.083333,
                                                                    2.416667,
                                                                    2.75,
                                                                    3.083333,
                                                                    3.166667,
                                                                    3.5,
                                                                    3.833333,
                                                                    7.416667,
                                                                    7.75,
                                                                    8.083333,
                                                                    8.416667,
                                                                    8.75,
                                                                    9.083334,
                                                                    9.166666,
                                                                    9.5,
                                                                    9.833334,
                                                                    4.166667,
                                                                    4.5,
                                                                    4.833333,
                                                                    4.166667,
                                                                    4.5,
                                                                    4.833333,
                                                                    2.083333,
                                                                    2.25,
                                                                    2.416667,
                                                                    10.16667,
                                                                    10.5,
                                                                    10.83333,
                                                                    10.16667,
                                                                    10.5,
                                                                    10.83333,
                                                                    5.083333,
                                                                    5.25,
                                                                    5.416667}),
                                Tensor(Shape{2, 4},
                                       ngraph::element::f16,
                                       std::vector<ngraph::float16>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0})},
            "experimental_detectron_roi_feature_eval_f16")));
