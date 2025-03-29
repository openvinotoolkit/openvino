// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental_detectron_topkrois.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ExperimentalDetectronTopKROIsParams {
    ExperimentalDetectronTopKROIsParams(const reference_tests::Tensor& dataTensor,
                                        const reference_tests::Tensor& probsTensor,
                                        const int32_t numRois,
                                        const reference_tests::Tensor& expectedTensor,
                                        const std::string& testcaseName = "")
        : dataTensor(dataTensor),
          probsTensor(probsTensor),
          numRois(numRois),
          expectedTensor(expectedTensor),
          testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor probsTensor;
    int32_t numRois;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
};

class ReferenceExperimentalDetectronTopKROIsTest : public testing::TestWithParam<ExperimentalDetectronTopKROIsParams>,
                                                   public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data, params.probsTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ExperimentalDetectronTopKROIsParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_pType=" << param.probsTensor.type;
        result << "_pShape=" << param.probsTensor.shape;
        result << "_numRois=" << param.numRois;
        result << "_eType=" << param.expectedTensor.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expectedTensor.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_eShape=" << param.expectedTensor.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ExperimentalDetectronTopKROIsParams& params) {
        std::shared_ptr<Model> function;
        const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto probs = std::make_shared<op::v0::Parameter>(params.probsTensor.type, params.probsTensor.shape);
        const auto topkRois = std::make_shared<op::v6::ExperimentalDetectronTopKROIs>(data, probs, params.numRois);
        function = std::make_shared<ov::Model>(topkRois, ParameterVector{data, probs});
        return function;
    }
};

TEST_P(ReferenceExperimentalDetectronTopKROIsTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ExperimentalDetectronTopKROIsParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ExperimentalDetectronTopKROIsParams> params{
        ExperimentalDetectronTopKROIsParams(
            reference_tests::Tensor(ET, {2, 4}, std::vector<T>{1.0f, 1.0f, 3.0f, 4.0f, 2.0f, 1.0f, 5.0f, 7.0f}),
            reference_tests::Tensor(ET, {2}, std::vector<T>{0.5f, 0.3f}),
            1,
            reference_tests::Tensor(ET, {1, 4}, std::vector<T>{1.0, 1.0, 3.0, 4.0}),
            "experimental_detectron_topk_rois_eval"),
        ExperimentalDetectronTopKROIsParams(
            reference_tests::Tensor(ET,
                                    {4, 4},
                                    std::vector<T>{1.0f,
                                                   1.0f,
                                                   4.0f,
                                                   5.0f,
                                                   3.0f,
                                                   2.0f,
                                                   7.0f,
                                                   9.0f,
                                                   10.0f,
                                                   15.0f,
                                                   13.0f,
                                                   17.0f,
                                                   13.0f,
                                                   10.0f,
                                                   18.0f,
                                                   15.0f}),
            reference_tests::Tensor(ET, {4}, std::vector<T>{0.1f, 0.7f, 0.5f, 0.9f}),
            2,
            reference_tests::Tensor(ET, {2, 4}, std::vector<T>{13.0f, 10.0f, 18.0f, 15.0f, 3.0f, 2.0f, 7.0f, 9.0f}),
            "experimental_detectron_topk_rois_eval"),
    };
    return params;
}

std::vector<ExperimentalDetectronTopKROIsParams> generateCombinedParams() {
    const std::vector<std::vector<ExperimentalDetectronTopKROIsParams>> generatedParams{
        generateParams<element::Type_t::bf16>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f64>(),
    };
    std::vector<ExperimentalDetectronTopKROIsParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalDetectronTopKROIs_With_Hardcoded_Refs,
                         ReferenceExperimentalDetectronTopKROIsTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceExperimentalDetectronTopKROIsTest::getTestCaseName);
}  // namespace
