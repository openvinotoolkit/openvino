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

using Attrs = op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes;
using ExperimentalROI = op::v6::ExperimentalDetectronROIFeatureExtractor;

struct ExperimentalROIFunctional {
    std::shared_ptr<Function> create_function(const std::vector<Tensor>& ed_inputs,
                                              const std::vector<Tensor>& results) {
        Attrs attrs;
        attrs.aligned = false;
        attrs.output_size = 3;
        attrs.sampling_ratio = 2;
        attrs.pyramid_scales = {4};

        auto input = std::make_shared<op::Parameter>(element::f32, Shape{2, 4});
        auto pyramid_layer0 = std::make_shared<op::Parameter>(element::f32, Shape{1, 2, 2, 3});

        auto roi = std::make_shared<ExperimentalROI>(NodeVector{input, pyramid_layer0}, attrs);

        auto fun = std::make_shared<Function>(OutputVector{roi->output(0), roi->output(1)},
                                              ParameterVector{input, pyramid_layer0});
        return fun;
    }
};

struct ExperimentalROIParams {
    ExperimentalROIParams(const std::shared_ptr<ExperimentalROIFunctional>& functional,
                          const std::vector<Tensor>& experimental_detectron_roi_feature_inputs,
                          const std::vector<Tensor>& expected_results,
                          const std::string& test_case_name)
        : function{functional},
          inputs{experimental_detectron_roi_feature_inputs},
          expected_results{expected_results},
          test_case_name{test_case_name} {}

    std::shared_ptr<ExperimentalROIFunctional> function;
    std::vector<Tensor> inputs;
    std::vector<Tensor> expected_results;
    std::string test_case_name;
};

class ReferenceExperimentalROILayerTest : public testing::TestWithParam<ExperimentalROIParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = params.function->create_function(params.inputs, params.expected_results);
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
            std::vector<Tensor>{Tensor(Shape{2, 4},
                                       ngraph::element::f32,
                                       std::vector<float>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}),
                                Tensor(Shape{1, 2, 2, 3},
                                       ngraph::element::f32,
                                       std::vector<float>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0})},
            std::vector<Tensor>{Tensor(Shape{2, 2, 3, 3},
                                       ngraph::element::f32,
                                       std::vector<float>{1.41666675,
                                                          1.75000012,
                                                          2.08333349,
                                                          2.41666675,
                                                          2.75,
                                                          3.083333492279052734,
                                                          3.166666507720947266,
                                                          3.5,
                                                          3.833333492279052734,
                                                          7.416666507720947266,
                                                          7.75,
                                                          8.083333015441894531,
                                                          8.416666984558105469,
                                                          8.75,
                                                          9.083333969116210938,
                                                          9.166666030883789062,
                                                          9.5,
                                                          9.833333969116210938,
                                                          4.166666984558105469,
                                                          4.5,
                                                          4.833333492279052734,
                                                          4.166666984558105469,
                                                          4.5,
                                                          4.833333492279052734,
                                                          2.083333492279052734,
                                                          2.25,
                                                          2.416666746139526367,
                                                          10.16666603088378906,
                                                          10.5,
                                                          10.83333206176757812,
                                                          10.16666603088378906,
                                                          10.5,
                                                          10.83333206176757812,
                                                          5.083333015441894531,
                                                          5.25,
                                                          5.416666507720947266}),
                                Tensor(Shape{2, 4},
                                       ngraph::element::f32,
                                       std::vector<float>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0})},
            "experimental_detectron_roi_feature_eval")));
