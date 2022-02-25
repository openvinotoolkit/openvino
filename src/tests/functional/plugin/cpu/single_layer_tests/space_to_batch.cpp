// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::opset3;
using namespace ov::test;

namespace CPULayerTestsDefinitions  {

using SpaceToBatchLayerTestCPUParams = std::tuple<
        InputShape,            // Input shapes
        std::vector<int64_t>,               // block shape
        std::vector<int64_t>,               // pads begin
        std::vector<int64_t>,               // pads end
        Precision ,                         // Network precision
        CPUSpecificParams>;

class SpaceToBatchCPULayerTest : public testing::WithParamInterface<SpaceToBatchLayerTestCPUParams>,
                                 virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SpaceToBatchLayerTestCPUParams> &obj) {
        InputShape inputShapes;
        std::vector<int64_t> blockShape, padsBegin, padsEnd;
        Precision netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, blockShape, padsBegin, padsEnd, netPrecision, cpuParams) = obj.param;
        std::ostringstream result;
        if (inputShapes.first.size() != 0) {
            result << "IS=(";
            result << CommonTestUtils::partialShape2str(std::vector<ngraph::PartialShape>{inputShapes.first}) << "_";
            result.seekp(-1, result.cur);
            result << ")_";
        }
        result << "TS=";
        for (const auto &item : inputShapes.second) {
            result << CommonTestUtils::vec2str(item) << "_";
        }
        result << "blockShape=" << CommonTestUtils::vec2str(blockShape) << "_";
        result << "padsBegin=" << CommonTestUtils::vec2str(padsBegin) << "_";
        result << "padsEnd=" << CommonTestUtils::vec2str(padsEnd) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        InputShape  inputShapes;
        std::vector<int64_t> blockShape, padsBegin, padsEnd;
        Precision netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, blockShape, padsBegin, padsEnd, netPrecision, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        auto ngPrec = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        inType = outType[0] = ngPrec;
        const std::vector<InputShape> inputShapesVec{inputShapes};
        init_input_shapes(inputShapesVec);

        if (strcmp(netPrecision.name(), "U8") == 0)
            selectedType = std::string("ref_any_") + "I8";
        else
            selectedType = std::string("ref_any_") + netPrecision.name();

        auto params = ngraph::builder::makeDynamicParams(ngPrec, {inputDynamicShapes.front()});
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto s2b = ngraph::builder::makeSpaceToBatch(paramOuts[0], ngPrec, blockShape, padsBegin, padsEnd);
        function = makeNgraphFunction(inType, params, s2b, "SpaceToBatchCPU");
    }
};

TEST_P(SpaceToBatchCPULayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
    CPUTestsBase::CheckPluginRelatedResults(compiledModel, "SpaceToBatch");
};

namespace {

const std::vector<Precision> netPrecision = {
        Precision::U8,
        Precision::I8,
        Precision::I32,
        Precision::FP32,
        Precision::BF16
};

const std::vector<std::vector<int64_t>> blockShape4D1 = {{1, 2, 1, 2}, {1, 1, 2, 2}, {1, 2, 2, 2}};
const std::vector<std::vector<int64_t>> padsBegin4D1 = {{0, 0, 0, 1}, {0, 0, 2, 1}, {0, 0, 4, 3}};
const std::vector<std::vector<int64_t>> padsEnd4D1   = {{0, 0, 0, 1}, {0, 0, 4, 1}, {0, 0, 2, 3}};

std::vector<ov::Shape> staticInputShapes4D1 = {{1, 16, 8, 12}, {1, 32, 8, 8}};

std::vector<InputShape> dynamicInputShapes4D1 = {
                {{-1, -1, -1, -1}, {{1, 6, 4, 8}, {2, 4, 8, 10}, {1, 8, 4, 10}}},
                {{{1, 4}, {2, 16}, 6, -1}, {{4, 8, 6, 4}, {1, 6, 6, 8}, {2, 12, 6, 4}}}
};

std::vector<InputShape> dynamicInputShapes4D1Blocked = {
                {{-1, 16, -1, -1}, {{1, 16, 4, 6}, {2, 16, 6, 6}, {4, 16, 4, 8}}}
};

const std::vector<std::vector<int64_t>> blockShape4D2 = { {1, 2, 4, 3}, {1, 4, 4, 1}};
const std::vector<std::vector<int64_t>> padsBegin4D2 = {{0, 0, 0, 0}, {0, 0, 4, 3}};
const std::vector<std::vector<int64_t>> padsEnd4D2   = {{0, 0, 4, 0}, {0, 0, 4, 3}};

std::vector<ov::Shape> staticInputShapes4D2 = {{1, 16, 12, 12}, {1, 32, 12, 15}};
std::vector<InputShape> dynamicInputShapes4D2 = {
                {{-1, -1, -1, -1}, {{1, 4, 8, 9}, {2, 8, 12, 9}, {6, 12, 4, 12}}},
                 {{2, {4, 16}, -1, -1}, {{2, 8, 4, 9}, {2, 4, 8, 6}, {2, 12, 12, 3}}}
};

std::vector<InputShape> dynamicInputShapes4D2Blocked = {
                 {{-1, 16, -1, -1}, {{2, 16, 4, 15}, {2, 16, 8, 12}, {3, 16, 12, 9}}}
};

const std::vector<CPUSpecificParams> cpuParamsWithBlock_4D = {
        CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
        CPUSpecificParams({nChw8c}, {nChw8c}, {}, {}),
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw}, {nchw}, {}, {})
};

const std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw}, {nchw}, {}, {})
};

const auto staticSpaceToBatchParamsSet4D1 = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes4D1)),
        ::testing::ValuesIn(blockShape4D1),
        ::testing::ValuesIn(padsBegin4D1),
        ::testing::ValuesIn(padsEnd4D1),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParams_4D));

const auto dynamicSpaceToBatchParamsSet4D1 = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes4D1),
        ::testing::ValuesIn(blockShape4D1),
        ::testing::ValuesIn(padsBegin4D1),
        ::testing::ValuesIn(padsEnd4D1),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParams_4D));

const auto dynamicSpaceToBatchParamsWithBlockedSet4D1 = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes4D1Blocked),
        ::testing::ValuesIn(blockShape4D1),
        ::testing::ValuesIn(padsBegin4D1),
        ::testing::ValuesIn(padsEnd4D1),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParamsWithBlock_4D));

const auto staticSpaceToBatchParamsSet4D2 = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes4D2)),
        ::testing::ValuesIn(blockShape4D2),
        ::testing::ValuesIn(padsBegin4D2),
        ::testing::ValuesIn(padsEnd4D2),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParams_4D));

const auto dynamicSpaceToBatchParamsSet4D2 = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes4D2),
        ::testing::ValuesIn(blockShape4D2),
        ::testing::ValuesIn(padsBegin4D2),
        ::testing::ValuesIn(padsEnd4D2),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParams_4D));

const auto dynamicSpaceToBatchParamsWithBlockedSet4D2 = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes4D2Blocked),
        ::testing::ValuesIn(blockShape4D2),
        ::testing::ValuesIn(padsBegin4D2),
        ::testing::ValuesIn(padsEnd4D2),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParamsWithBlock_4D));

INSTANTIATE_TEST_SUITE_P(smoke_StaticSpaceToBatchCPULayerTestCase1_4D, SpaceToBatchCPULayerTest,
                         staticSpaceToBatchParamsSet4D1, SpaceToBatchCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicSpaceToBatchCPULayerTestCase1_4D, SpaceToBatchCPULayerTest,
                         dynamicSpaceToBatchParamsSet4D1, SpaceToBatchCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicSpaceToBatchCPULayerTestCaseWithBlocked1_4D, SpaceToBatchCPULayerTest,
                         dynamicSpaceToBatchParamsWithBlockedSet4D1, SpaceToBatchCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticSpaceToBatchCPULayerTestCase2_4D, SpaceToBatchCPULayerTest,
                         staticSpaceToBatchParamsSet4D2, SpaceToBatchCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicSpaceToBatchCPULayerTestCase2_4D, SpaceToBatchCPULayerTest,
                         dynamicSpaceToBatchParamsSet4D2, SpaceToBatchCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicSpaceToBatchCPULayerTestCaseWithBlocked2_4D, SpaceToBatchCPULayerTest,
                         dynamicSpaceToBatchParamsWithBlockedSet4D2, SpaceToBatchCPULayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> blockShape5D = {{1, 1, 2, 2, 1}, {1, 2, 4, 1, 3}};
const std::vector<std::vector<int64_t>> padsBegin5D = {{0, 0, 0, 0, 0}, {0, 0, 4, 0, 0}, {0, 0, 0, 2, 3}};
const std::vector<std::vector<int64_t>> padsEnd5D   = {{0, 0, 0, 0, 0}, {0, 0, 0, 4, 3}, {0, 0, 4, 2, 3}};

std::vector<ov::Shape> staticInputShapes5D = {{2, 16, 4, 6, 12}, {1, 32, 8, 8, 6}, {1, 16, 4, 12, 12}};

std::vector<InputShape> dynamicInputShapes5D = {
                {{-1, -1, -1, -1, -1}, {{2, 2, 12, 4, 15}, {4, 4, 8, 6, 9}, {3, 6, 4, 2, 12}}},
                {{{1, 10}, {2, 20}, {4, 50}, -1, -1}, {{3, 12, 8, 6, 9}, {5, 10, 4, 8, 15}, {6, 8, 20, 4, 12}}}
};

std::vector<InputShape> dynamicInputShapes5DBlocked = {
                {{-1, 16, -1, -1, -1}, {{2, 16, 4, 6, 9}, {5, 16, 16, 4, 6}, {7, 16, 8, 2, 3}}}
};

const std::vector<CPUSpecificParams> cpuParamsWithBlock_5D = {
        CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {}),
        CPUSpecificParams({nCdhw8c}, {nCdhw8c}, {}, {}),
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {})
};

const std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {})
};

const auto staticSpaceToBatchParamsSet5D = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes5D)),
        ::testing::ValuesIn(blockShape5D),
        ::testing::ValuesIn(padsBegin5D),
        ::testing::ValuesIn(padsEnd5D),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParamsWithBlock_5D));

const auto dynamicSpaceToBatchParamsSet5D = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes5D),
        ::testing::ValuesIn(blockShape5D),
        ::testing::ValuesIn(padsBegin5D),
        ::testing::ValuesIn(padsEnd5D),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParams_5D));

const auto dynamicSpaceToBatchParamsWithBlockedSet5D = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes5DBlocked),
        ::testing::ValuesIn(blockShape5D),
        ::testing::ValuesIn(padsBegin5D),
        ::testing::ValuesIn(padsEnd5D),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParamsWithBlock_5D));

INSTANTIATE_TEST_SUITE_P(smoke_StaticSpaceToBatchCPULayerTestCase_5D, SpaceToBatchCPULayerTest,
                         staticSpaceToBatchParamsSet5D, SpaceToBatchCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicSpaceToBatchCPULayerTestCase_5D, SpaceToBatchCPULayerTest,
                         dynamicSpaceToBatchParamsSet5D, SpaceToBatchCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicSpaceToBatchCPULayerTestCaseWithBlocked_5D, SpaceToBatchCPULayerTest,
                         dynamicSpaceToBatchParamsWithBlockedSet5D, SpaceToBatchCPULayerTest::getTestCaseName);


}  // namespace
}  // namespace CPULayerTestsDefinitions
