// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/batch_to_space.hpp>
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::opset3;

namespace CPULayerTestsDefinitions  {

using inputShapesPair = std::pair<std::vector<ov::PartialShape>, std::vector<std::vector<ov::Shape>>>;

using BatchToSpaceLayerTestCPUParams = std::tuple<
        inputShapesPair,                    // Input shapes
        std::vector<int64_t>,               // block shape
        std::vector<int64_t>,               // crops begin
        std::vector<int64_t>,               // crops end
        InferenceEngine::Precision,         // Network precision
        CPUSpecificParams>;

class BatchToSpaceCPULayerTest : public testing::WithParamInterface<BatchToSpaceLayerTestCPUParams>,
                                 virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BatchToSpaceLayerTestCPUParams> &obj) {
        inputShapesPair inputShapes;
        std::vector<int64_t> blockShape, cropsBegin, cropsEnd;
        Precision netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, blockShape, cropsBegin, cropsEnd, netPrecision, cpuParams) = obj.param;
        std::ostringstream result;
        if (!inputShapes.first.empty()) {
            result << "IS=" << CommonTestUtils::partialShape2str(inputShapes.first) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes.second) {
            result << "(";
            for (const auto& item : shape) {
                result << CommonTestUtils::vec2str(item) << "_";
            }
            result << ")_";
        }
        result << "blockShape=" << CommonTestUtils::vec2str(blockShape) << "_";
        result << "cropsBegin=" << CommonTestUtils::vec2str(cropsBegin) << "_";
        result << "cropsEnd=" << CommonTestUtils::vec2str(cropsEnd) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        inputShapesPair inputShapes;
        std::vector<int64_t> blockShape, cropsBegin, cropsEnd;
        Precision netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, blockShape, cropsBegin, cropsEnd, netPrecision, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        inPrc = outPrc = netPrecision;

        targetStaticShapes.reserve(inputShapes.second.size());
        for (size_t i = 0; i < inputShapes.second.size(); i++) {
            targetStaticShapes.push_back(inputShapes.second[i]);
        }
        inputDynamicShapes = inputShapes.first;

        if (strcmp(netPrecision.name(), "U8") == 0)
            selectedType = std::string("ref_any_") + "I8";
        else
            selectedType = std::string("ref_any_") + netPrecision.name();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {targetStaticShapes.front().front()});
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto b2s = ngraph::builder::makeBatchToSpace(paramOuts[0], ngPrc, blockShape, cropsBegin, cropsEnd);
        b2s->get_rt_info() = getCPUInfo();
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(b2s)};
        function = std::make_shared<ngraph::Function>(results, params, "BatchToSpace");
    }
};

TEST_P(BatchToSpaceCPULayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "BatchToSpace");
};

namespace {

const std::vector<Precision> netPrecision = {
        Precision::U8,
        Precision::I8,
        Precision::I32,
        Precision::FP32,
        Precision::BF16
};

const std::vector<std::vector<int64_t>> blockShape4D1  = {{1, 1, 1, 2}, {1, 2, 2, 1}};
const std::vector<std::vector<int64_t>> cropsBegin4D1  = {{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 2, 0}};
const std::vector<std::vector<int64_t>> cropsEnd4D1    = {{0, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 1, 1}};

const std::vector<inputShapesPair> staticInputShapes4D1 = {
        {
                {},
                // Static shapes
                {
                        {{8, 16, 10, 10}}
                }
        }
};

const std::vector<inputShapesPair> dynamicInputShapes4D1 = {
        {
                // dynamic
                {
                        {-1, -1, -1, -1},
                },
                // target
                {
                        {{8, 8, 6, 7}},
                        {{4, 10, 5, 5}},
                        {{12, 9, 7, 5}}
                }
        },
        {
                // dynamic
                {
                        {{4, 12}, {8, 16}, {6}, -1},
                },
                // target
                {
                        {{8, 8, 6, 7}},
                        {{4, 10, 6, 5}},
                        {{12, 9, 6, 5}},
                }
        }
};

const std::vector<inputShapesPair> dynamicInputShapes4D1Blocked = {
        {
                // dynamic
                {
                        {-1, {16}, -1, -1},
                },
                // target
                {
                        {{4, 16, 5, 8}},
                        {{8, 16, 7, 6}},
                        {{12, 16, 4, 5}}
                }
        }
};

const std::vector<std::vector<int64_t>> blockShape4D2  = {{1, 2, 3, 4}, {1, 3, 4, 2}};
const std::vector<std::vector<int64_t>> cropsBegin4D2  = {{0, 0, 0, 1}, {0, 0, 1, 2}};
const std::vector<std::vector<int64_t>> cropsEnd4D2    = {{0, 0, 1, 0}, {0, 0, 3, 1}};

const std::vector<inputShapesPair> staticInputShapes4D2 = {
        {
                {},
                // Static shapes
                {
                        {{24, 16, 7, 8}}
                }
        }
};

const std::vector<inputShapesPair> dynamicInputShapes4D2 = {
        {
                // dynamic
                {
                        {-1, -1, -1, -1},
                },
                // target
                {
                        {{48, 4, 7, 8}},
                        {{24, 8, 6, 7}},
                        {{24, 16, 5, 5}}
                }
        },
        {
                // dynamic
                {
                        {{24}, {4, 10}, -1, -1},
                },
                // target
                {
                        {{24, 8, 6, 7}},
                        {{24, 6, 7, 5}},
                        {{24, 4, 5, 5}}
                }
        }
};

const std::vector<inputShapesPair> dynamicInputShapes4D2Blocked = {
        {
                // dynamic
                {
                        {-1, {16}, -1, -1},
                },
                // target
                {
                        {{24, 16, 5, 5}},
                        {{24, 16, 6, 7}},
                        {{48, 16, 4, 4}}
                }
        }
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

const auto staticBatchToSpaceParamsSet4D1 = ::testing::Combine(
        ::testing::ValuesIn(staticInputShapes4D1),
        ::testing::ValuesIn(blockShape4D1),
        ::testing::ValuesIn(cropsBegin4D1),
        ::testing::ValuesIn(cropsEnd4D1),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParamsWithBlock_4D));

const auto dynamicBatchToSpaceParamsSet4D1 = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes4D1),
        ::testing::ValuesIn(blockShape4D1),
        ::testing::ValuesIn(cropsBegin4D1),
        ::testing::ValuesIn(cropsEnd4D1),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParams_4D));

const auto dynamicBatchToSpaceParamsWithBlockedSet4D1 = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes4D1Blocked),
        ::testing::ValuesIn(blockShape4D1),
        ::testing::ValuesIn(cropsBegin4D1),
        ::testing::ValuesIn(cropsEnd4D1),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParamsWithBlock_4D));

const auto staticBatchToSpaceParamsSet4D2 = ::testing::Combine(
        ::testing::ValuesIn(staticInputShapes4D2),
        ::testing::ValuesIn(blockShape4D2),
        ::testing::ValuesIn(cropsBegin4D2),
        ::testing::ValuesIn(cropsEnd4D2),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParamsWithBlock_4D));

const auto dynamicBatchToSpaceParamsSet4D2 = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes4D2),
        ::testing::ValuesIn(blockShape4D2),
        ::testing::ValuesIn(cropsBegin4D2),
        ::testing::ValuesIn(cropsEnd4D2),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParams_4D));

const auto dynamicBatchToSpaceParamsWithBlockedSet4D2 = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes4D2Blocked),
        ::testing::ValuesIn(blockShape4D2),
        ::testing::ValuesIn(cropsBegin4D2),
        ::testing::ValuesIn(cropsEnd4D2),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParamsWithBlock_4D));

INSTANTIATE_TEST_SUITE_P(smoke_StaticBatchToSpaceCPULayerTestCase1_4D, BatchToSpaceCPULayerTest,
                         staticBatchToSpaceParamsSet4D1, BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCase1_4D, BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsSet4D1, BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCaseWithBlocked1_4D, BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsWithBlockedSet4D1, BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticBatchToSpaceCPULayerTestCase2_4D, BatchToSpaceCPULayerTest,
                         staticBatchToSpaceParamsSet4D2, BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCase2_4D, BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsSet4D2, BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCaseWithBlocked2_4D, BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsWithBlockedSet4D2, BatchToSpaceCPULayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> blockShape5D1  = {{1, 1, 2, 2, 1}, {1, 2, 1, 2, 2}};
const std::vector<std::vector<int64_t>> cropsBegin5D1  = {{0, 0, 0, 0, 0}, {0, 0, 0, 3, 3}};
const std::vector<std::vector<int64_t>> cropsEnd5D1    = {{0, 0, 0, 0, 0}, {0, 0, 1, 0, 1}};

const  std::vector<inputShapesPair> staticInputShapes5D1 = {
        {
                {},
                // Static shapes
                {
                        {{8, 16, 4, 10, 10}}
                }
        }
};

const std::vector<inputShapesPair> dynamicInputShapes5D1 = {
        {
                // dynamic
                {
                        {-1, -1, -1, -1, -1},
                },
                // target
                {
                        {{8, 16, 4, 10, 10}},
                        {{16, 10, 5, 11, 9}},
                        {{24, 6, 6, 8, 8}},
                }
        },
        {
                // dynamic
                {
                        {{8, 16}, {8, 16}, {2, 7}, -1, -1},
                },
                // target
                {
                        {{8, 16, 2, 6, 8}},
                        {{8, 10, 4, 7, 5}},
                        {{16, 8, 7, 5, 10}}
                }
        }
};

const std::vector<inputShapesPair> dynamicInputShapes5D1Blocked = {
        {
                // dynamic
                {
                        {-1, {16}, {3, 5}, -1, -1},
                },
                // target
                {
                        {{24, 16, 3, 6, 7}},
                        {{48, 16, 4, 5, 5}},
                        {{24, 16, 5, 8, 5}},
                }
        }
};

const std::vector<std::vector<int64_t>> blockShape5D2  = {{1, 2, 4, 3, 1}, {1, 1, 2, 4, 3}};
const std::vector<std::vector<int64_t>> cropsBegin5D2  = {{0, 0, 1, 2, 0}, {0, 0, 1, 0, 1}};
const std::vector<std::vector<int64_t>> cropsEnd5D2    = {{0, 0, 1, 0, 1}, {0, 0, 1, 1, 1}};

const  std::vector<inputShapesPair> staticInputShapes5D2 = {
        {
                {},
                // Static shapes
                {
                        {{48, 16, 3, 3, 3}}
                }
        }
};

const std::vector<inputShapesPair> dynamicInputShapes5D2 = {
        {
                // dynamic
                {
                        {-1, -1, -1, -1, -1},
                },
                // target
                {
                        {{48, 4, 3, 3, 3}},
                        {{24, 16, 5, 3, 5}},
                        {{24, 8, 7, 5, 5}}
                }
        },
        {
                // dynamic
                {
                        {{24}, {8, 16}, {3, 5}, -1, -1},
                },
                // target
                {
                        {{24, 16, 3, 4, 3}},
                        {{24, 12, 5, 3, 5}},
                        {{24, 8, 4, 5, 5}}
                }
        }
};

const std::vector<inputShapesPair> dynamicInputShapes5D2Blocked = {
        {
                // dynamic
                {
                        {-1, {16}, -1, -1, -1},
                },
                // target
                {
                        {{24, 16, 4, 5, 5}},
                        {{48, 16, 3, 4, 3}},
                        {{24, 16, 5, 3, 5}}
                }
        }
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

const auto staticBatchToSpaceParamsSet5D1 = ::testing::Combine(
        ::testing::ValuesIn(staticInputShapes5D1),
        ::testing::ValuesIn(blockShape5D1),
        ::testing::ValuesIn(cropsBegin5D1),
        ::testing::ValuesIn(cropsEnd5D1),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParamsWithBlock_5D));

const auto dynamicBatchToSpaceParamsSet5D1 = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes5D1),
        ::testing::ValuesIn(blockShape5D1),
        ::testing::ValuesIn(cropsBegin5D1),
        ::testing::ValuesIn(cropsEnd5D1),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParams_5D));

const auto dynamicBatchToSpaceParamsWithBlockedSet5D1 = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes5D1Blocked),
        ::testing::ValuesIn(blockShape5D1),
        ::testing::ValuesIn(cropsBegin5D1),
        ::testing::ValuesIn(cropsEnd5D1),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParamsWithBlock_5D));

const auto staticBatchToSpaceParamsSet5D2 = ::testing::Combine(
        ::testing::ValuesIn(staticInputShapes5D2),
        ::testing::ValuesIn(blockShape5D2),
        ::testing::ValuesIn(cropsBegin5D2),
        ::testing::ValuesIn(cropsEnd5D2),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParamsWithBlock_5D));

const auto dynamicBatchToSpaceParamsSet5D2 = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes5D2),
        ::testing::ValuesIn(blockShape5D2),
        ::testing::ValuesIn(cropsBegin5D2),
        ::testing::ValuesIn(cropsEnd5D2),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParams_5D));

const auto dynamicBatchToSpaceParamsWithBlockedSet5D2 = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes5D2Blocked),
        ::testing::ValuesIn(blockShape5D2),
        ::testing::ValuesIn(cropsBegin5D2),
        ::testing::ValuesIn(cropsEnd5D2),
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(cpuParamsWithBlock_5D));

INSTANTIATE_TEST_SUITE_P(smoke_StaticBatchToSpaceCPULayerTestCase1_5D, BatchToSpaceCPULayerTest,
                         staticBatchToSpaceParamsSet5D1, BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCase1_5D, BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsSet5D1, BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCaseWithBlocked1_5D, BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsWithBlockedSet5D1, BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticBatchToSpaceCPULayerTestCase2_5D, BatchToSpaceCPULayerTest,
                         staticBatchToSpaceParamsSet5D2, BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCase2_5D, BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsSet5D2, BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCaseWithBlocked2_5D, BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsWithBlockedSet5D2, BatchToSpaceCPULayerTest::getTestCaseName);

}  // namespace
}  // namespace CPULayerTestsDefinitions
