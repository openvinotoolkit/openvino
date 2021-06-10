// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

using concatParamsTuple = typename std::tuple<
        //TODO: according to specification axis have to be int, negative values are allowed
        size_t,                            // Concat axis
        std::vector<size_t>,               // Input shapes
        size_t,                            // Concat inputs number
        size_t,                            // Concats nymber
        InferenceEngine::Precision,        // Network precision
        InferenceEngine::Precision,        // Input precision
        InferenceEngine::Precision,        // Output precision
        InferenceEngine::Layout,           // Input layout
        InferenceEngine::Layout,           // Output layout
        std::string>;                      // Device name

namespace LayerTestsDefinitions {

class InsertCopyBeforeSelfConcatTest : public testing::WithParamInterface<concatParamsTuple>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<concatParamsTuple> obj) {
        int axis;
        std::vector<size_t> inputShape;
        size_t inputsNum, concatsNum;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        std::string targetName;
        std::tie(axis, inputShape, inputsNum, concatsNum, netPrecision, inPrc, outPrc, inLayout, outLayout, targetName) = obj.param;
        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "IN=" << inputsNum << "_";
        result << "CN=" << concatsNum << "_";
        result << "axis=" << axis << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "inPRC=" << inPrc.name() << "_";
        result << "outPRC=" << outPrc.name() << "_";
        result << "inL=" << inLayout << "_";
        result << "outL=" << outLayout << "_";
        result << "trgDev=" << targetName;
        return result.str();
    }

protected:
    void SetUp() override {
        int axis;
        std::vector<size_t> inputShape;
        size_t inputsNum, concatsNum;
        InferenceEngine::Precision netPrecision;
        std::tie(axis, inputShape, inputsNum, concatsNum, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        ngraph::OutputVector concatInputs;
        for (int i = 0; i < inputsNum; ++i) {
            concatInputs.push_back(params[0]);
        }
        ngraph::ResultVector results;
        for (int i = 0; i < concatsNum; ++i) {
            auto concat = std::make_shared<ngraph::opset1::Concat>(concatInputs, axis);
            auto relu = std::make_shared<ngraph::opset1::Relu>(concat);
            results.push_back(std::make_shared<ngraph::opset1::Result>(relu));
        }
        function = std::make_shared<ngraph::Function>(results, params, "InsertCopyBeforeSelfConcat");
    }
};

TEST_P(InsertCopyBeforeSelfConcatTest, CompareWithRefs) {
    Run();
};

std::vector<size_t > axes = {1};

std::vector<std::vector<size_t>> inShapes = {
        {1, 32},
        {1, 128},
        {8, 64},
        {1, 16}
};

std::vector<size_t> inputsNum = {2, 3, 5};

std::vector<size_t> concatsNum = {1, 2, 3, 4};

std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                         InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_CASE_P(smoke_InsertCopy, InsertCopyBeforeSelfConcatTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(axes),
                                ::testing::ValuesIn(inShapes),
                                ::testing::ValuesIn(inputsNum),
                                ::testing::ValuesIn(concatsNum),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        InsertCopyBeforeSelfConcatTest::getTestCaseName);

} // namespace LayerTestsDefinitions
