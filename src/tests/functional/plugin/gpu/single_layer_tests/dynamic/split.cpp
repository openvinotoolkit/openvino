// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/select.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "ngraph_functions/builders.hpp"
#include <string>

using namespace ngraph;
using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

typedef std::tuple<
        size_t,                    // Num splits
        size_t,                    // Axis
        ElementType,               // Net precision
        InputShape,                // Input shapes
        std::vector<size_t>        // Used outputs indices
> splitDynamicGPUTestParams;

class SplitLayerGPUDynamicTest : public testing::WithParamInterface<splitDynamicGPUTestParams>,
                          virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<splitDynamicGPUTestParams> obj) {
        std::ostringstream result;
        size_t numSplits;
        int64_t axis;
        ElementType netPrecision;
        InputShape inputShape;
        std::vector<size_t> outIndices;
        std::tie(numSplits, axis, netPrecision, inputShape, outIndices) = obj.param;

        result << "IS=";
        result << CommonTestUtils::partialShape2str({inputShape.first}) << "_";
        result << "TS=";
        for (const auto& shape : inputShape.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }
        result << "numSplits=" << numSplits << "_";
        result << "axis=" << axis << "_";
        if (!outIndices.empty()) {
            result << "outIndices" << CommonTestUtils::vec2str(outIndices) << "_";
        }
        result << "netPRC=" << netPrecision << "_";
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_GPU;
        size_t axis, numSplits;
        InputShape inputShape;
        std::vector<size_t> outIndices;
        ElementType netPrecision;
        splitDynamicGPUTestParams params = this->GetParam();
        std::tie(numSplits, axis, netPrecision, inputShape, outIndices) = params;
        if (outIndices.empty()) {
            for (int i = 0; i < numSplits; ++i) {
                outIndices.push_back(i);
            }
        }
        init_input_shapes({inputShape});
        auto dyn_params = ngraph::builder::makeDynamicParams(netPrecision, {inputDynamicShapes[0]});
        auto paramOuts =
            ngraph::helpers::convert2OutputVector(helpers::castOps2Nodes<opset1::Parameter>(dyn_params));
        auto split = std::dynamic_pointer_cast<ngraph::opset5::Split>(
                     ngraph::builder::makeSplit(paramOuts[0], netPrecision, numSplits, axis));
        ngraph::ResultVector results;
        for (int i = 0; i < outIndices.size(); i++) {
            results.push_back(std::make_shared<ngraph::opset1::Result>(split->output(outIndices[i])));
        }
        function = std::make_shared<ngraph::Function>(results, dyn_params, "split");
    }
};

TEST_P(SplitLayerGPUDynamicTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

const std::vector<InputShape> inputShapes4d = {
        {
            {-1, -1, -1, -1}, {{1, 4, 5, 7}, {3, 8, 5, 9}, {5, 16, 1, 8}}
        }
};

const std::vector<InputShape> inputShapes5d = {
        {
            {-1, -1, -1, -1, -1}, {{10, 20, 30, 40, 10}, {5, 18, 3, 10, 10}, {3, 10, 6, 2, 4}}
        }
};

const std::vector<InputShape> inputShapes6d = {
        {
            {-1, -1, -1, -1, -1, -1}, {{10, 32, 3, 4, 12, 6}, {5, 2, 3, 1, 30, 12}, {3, 1, 6, 2, 6, 18}}
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_NumSplitsCheck4Dtaylor, SplitLayerGPUDynamicTest,
                        ::testing::Combine(
                                ::testing::Values(2),                                       // nSplits
                                ::testing::Values(1),                                       // axes
                                ::testing::Values(ElementType::f16),                         // netPrec
                                ::testing::ValuesIn(inputShapes4d),                         // inShapes
                                ::testing::Values(std::vector<size_t>({}))),                // outIndices
                        SplitLayerGPUDynamicTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NumSplitsCheck5Dtaylor, SplitLayerGPUDynamicTest,
                        ::testing::Combine(
                                ::testing::Values(3),                                       // nSplits
                                ::testing::Values(2),                                       // axes
                                ::testing::Values(ElementType::f32),                         // netPrec
                                ::testing::ValuesIn(inputShapes5d),                         // inShapes
                                ::testing::Values(std::vector<size_t>({}))),                // outIndices
                        SplitLayerGPUDynamicTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NumSplitsCheck6Dtaylor, SplitLayerGPUDynamicTest,
                        ::testing::Combine(
                                ::testing::Values(6),                                       // nSplits
                                ::testing::Values(4),                                       // axes
                                ::testing::Values(ElementType::f16),                         // netPrec
                                ::testing::ValuesIn(inputShapes6d),                         // inShapes
                                ::testing::Values(std::vector<size_t>({}))),                // outIndices
                        SplitLayerGPUDynamicTest::getTestCaseName);

} // namespace GPULayerTestsDefinitions

