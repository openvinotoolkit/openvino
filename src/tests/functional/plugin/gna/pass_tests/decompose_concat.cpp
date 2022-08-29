// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include "transformations/init_node_info.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"


using namespace ngraph;
using namespace opset8;


namespace LayerTestsDefinitions {

typedef std::tuple<
        std::vector<std::vector<size_t>>,   // Input shapes
        int64_t,                            // Concat axis
        InferenceEngine::Precision,         // Network Precision
        std::string,                        // Target Device
        std::map<std::string, std::string>  // Configuration
> decomposeConcatParams;

class DecomposeConcatTest : public testing::WithParamInterface<decomposeConcatParams>,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<decomposeConcatParams> obj) {
        std::vector<std::vector<size_t>> inputShapes;
        int64_t concatAxis;
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(inputShapes, concatAxis, netPrecision, targetDevice, configuration) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        result << "concatAxis=" << std::to_string(concatAxis) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        return result.str();
    }

protected:
    void SetUp() override {
        std::vector<std::vector<size_t>> inputShapes;
        int64_t concatAxis;
        InferenceEngine::Precision netPrecision;
        std::tie(inputShapes, concatAxis, netPrecision, targetDevice, configuration) = this->GetParam();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = builder::makeParams(ngPrc, inputShapes);
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(params));

        std::vector<int64_t> startOffset = {0, 0, 0, 0};
        std::vector<int64_t> endOffset(startOffset);
        OutputVector concatInput;
        std::vector<std::shared_ptr<opset9::Relu>> ssArray;

        for (size_t i = 0; i < inputShapes.size(); ++i) {
            std::vector<size_t> shape(inputShapes[i]);
            std::transform(shape.begin(), shape.end(), startOffset.begin(), endOffset.begin(), std::plus<int>());
            auto begin = std::make_shared<op::Constant>(element::i64, Shape{4}, startOffset);
            auto end = std::make_shared<op::Constant>(element::i64, Shape{4}, endOffset);
            auto stride = std::make_shared<op::Constant>(element::i64, Shape{4}, std::vector<int64_t>{1, 1, 1, 1});
            auto ss = std::make_shared<opset9::StridedSlice>(paramOuts[0], begin, end, stride,
                                                             std::vector<int64_t>{0, 0, 0, 0},
                                                             std::vector<int64_t>{0, 0, 0, 0});
            auto relu = std::make_shared<opset9::Relu>(paramOuts[i]);
            ssArray.push_back(relu);
            concatInput.push_back(ssArray[i]);
            startOffset[concatAxis] += shape[concatAxis];
        }

        auto concat = std::make_shared<opset9::Concat>(concatInput, concatAxis);
        auto relu = std::make_shared<opset9::Relu>(concat);

        ResultVector results{std::make_shared<opset9::Result>(relu)};
        function = std::make_shared<Function>(results, params, "DecomposeConcat");
    }
};

TEST_P(DecomposeConcatTest, CompareWithRefs) {
    Run();
}

const std::vector<std::vector<std::vector<size_t>>> inShapes = {
    {{1, 2, 4, 64}, {1, 2, 4, 64}},
};

const std::vector<int64_t> concatAxis = {
    1,
    2
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_DecomposeConcat, DecomposeConcatTest,
    ::testing::Combine(
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(concatAxis),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs)),
    DecomposeConcatTest::getTestCaseName);

} // namespace LayerTestsDefinitions
