// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::opset3;
using namespace ov::test;

namespace CPULayerTestsDefinitions  {

namespace {
    std::vector<int64_t> blockShape, cropsBegin, cropsEnd;
    ngraph::Shape paramShape;
}  // namespace

using BatchToSpaceLayerTestCPUParams = std::tuple<
        std::vector<InputShape>,            // Input shapes
        std::vector<int64_t>,               // block shape
        std::vector<int64_t>,               // crops begin
        std::vector<int64_t>,               // crops end
        Precision ,                         // Network precision
        CPUSpecificParams>;

class BatchToSpaceCPULayerTest : public testing::WithParamInterface<BatchToSpaceLayerTestCPUParams>,
                                 virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BatchToSpaceLayerTestCPUParams> &obj) {
        std::vector<InputShape> inputShapes;
        Precision netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, blockShape, cropsBegin, cropsEnd, netPrecision, cpuParams) = obj.param;
        std::ostringstream result;
        if (inputShapes.front().first.size() != 0) {
            result << "IS=(";
            for (const auto &shape : inputShapes) {
                result << ov::test::utils::partialShape2str({shape.first}) << "_";
            }
            result.seekp(-1, result.cur);
            result << ")_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            for (const auto& item : shape.second) {
                result << ov::test::utils::vec2str(item) << "_";
            }
        }
        result << "blockShape=" << ov::test::utils::vec2str(blockShape) << "_";
        result << "cropsBegin=" << ov::test::utils::vec2str(cropsBegin) << "_";
        result << "cropsEnd=" << ov::test::utils::vec2str(cropsEnd) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); i++) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (i == 0U) {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 2560, 0, 256);
            } else if (i == 1U) {
                tensor = ov::Tensor(funcInput.get_element_type(), paramShape);
                auto *dataPtr = tensor.data<int64_t>();
                for (size_t j = 0; j < blockShape.size(); j++) {
                    dataPtr[j] = blockShape[j];
                }
            } else if (i == 2U) {
                tensor = ov::Tensor(funcInput.get_element_type(), paramShape);
                auto *dataPtr = tensor.data<int64_t>();
                for (size_t j = 0; j < cropsBegin.size(); j++) {
                    dataPtr[j] = cropsBegin[j];
                }
            } else if (i == 3U) {
                tensor = ov::Tensor(funcInput.get_element_type(), paramShape);
                auto *dataPtr = tensor.data<int64_t>();
                for (size_t j = 0; j < cropsEnd.size(); j++) {
                    dataPtr[j] = cropsEnd[j];
                }
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        std::vector<InputShape>  inputShapes;
        Precision netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, blockShape, cropsBegin, cropsEnd, netPrecision, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        auto ngPrec = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        const std::vector<InputShape> inputShapesVec{inputShapes};
        init_input_shapes(inputShapesVec);

        if (strcmp(netPrecision.name(), "U8") == 0)
            selectedType = std::string("ref_any_") + "I8";
        else
            selectedType = std::string("ref_any_") + netPrecision.name();

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrec, inputDynamicShapes.front())};
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        paramShape = {paramOuts[0].get_partial_shape().size()};

        std::shared_ptr<ov::Node> in2, in3, in4;
        auto blockShapeParam = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i64, paramShape);
        in2 = blockShapeParam;
        params.push_back(blockShapeParam);
        auto cropsBeginParam = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i64, paramShape);
        params.push_back(cropsBeginParam);
        in3 = cropsBeginParam;
        auto cropsEndParam = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i64, paramShape);
        params.push_back(cropsEndParam);
        in4 = cropsEndParam;
        auto btsNode = std::make_shared<ngraph::opset2::BatchToSpace>(paramOuts[0], in2, in3, in4);
        btsNode->get_rt_info() = getCPUInfo();
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(btsNode)};
        function = std::make_shared<ngraph::Function>(results, params, "BatchToSpace");
    }
};

TEST_P(BatchToSpaceCPULayerTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "BatchToSpace");
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

std::vector<std::vector<ov::Shape>> staticInputShapes4D1 = {
        {{8, 16, 10, 10}, {4}, {4}, {4}}
};

std::vector<std::vector<InputShape>> dynamicInputShapes4D1 = {
    {
        {{-1, -1, -1, -1}, {{8, 8, 6, 7}, {4, 10, 5, 5}, {12, 9, 7, 5}}},
        {{4}, {{4}, {4}, {4}}},
        {{4}, {{4}, {4}, {4}}},
        {{4}, {{4}, {4}, {4}}}
    },
    {
        {{{4, 12}, {8, 16}, 6, -1}, {{8, 8, 6, 7}, {4, 10, 6, 5}, {12, 9, 6, 5}}},
        {{4}, {{4}, {4}, {4}}},
        {{4}, {{4}, {4}, {4}}},
        {{4}, {{4}, {4}, {4}}}
    }
};

std::vector<std::vector<InputShape>> dynamicInputShapes4D1Blocked = {
    {
        {{-1, 16, -1, -1}, {{4, 16, 5, 8}, {8, 16, 7, 6}, {12, 16, 4, 5}}},
        {{4}, {{4}, {4}, {4}}},
        {{4}, {{4}, {4}, {4}}},
        {{4}, {{4}, {4}, {4}}}
    }
};

const std::vector<std::vector<int64_t>> blockShape4D2  = {{1, 2, 3, 4}, {1, 3, 4, 2}};
const std::vector<std::vector<int64_t>> cropsBegin4D2  = {{0, 0, 0, 1}, {0, 0, 1, 2}};
const std::vector<std::vector<int64_t>> cropsEnd4D2    = {{0, 0, 1, 0}, {0, 0, 3, 1}};

std::vector<std::vector<ov::Shape>> staticInputShapes4D2 = {
        {{24, 16, 7, 8}, {4}, {4}, {4}}
};

std::vector<std::vector<InputShape>> dynamicInputShapes4D2 = {
    {
        {{-1, -1, -1, -1}, {{48, 4, 7, 8}, {24, 8, 6, 7}, {24, 16, 5, 5}}},
        {{4}, {{4}, {4}, {4}}},
        {{4}, {{4}, {4}, {4}}},
        {{4}, {{4}, {4}, {4}}}
    },
    {
        {{24, {4, 10}, -1, -1}, {{24, 8, 6, 7}, {24, 6, 7, 5}, {24, 4, 5, 5}}},
        {{4}, {{4}, {4}, {4}}},
        {{4}, {{4}, {4}, {4}}},
        {{4}, {{4}, {4}, {4}}}
    }
};

std::vector<std::vector<InputShape>> dynamicInputShapes4D2Blocked = {
    {
        {{-1, 16, -1, -1}, {{24, 16, 5, 5}, {24, 16, 6, 7}, {48, 16, 4, 4}}},
        {{4}, {{4}, {4}, {4}}},
        {{4}, {{4}, {4}, {4}}},
        {{4}, {{4}, {4}, {4}}}
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
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes4D1)),
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
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes4D2)),
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

std::vector<std::vector<ov::Shape>> staticInputShapes5D1 = {
    {{8, 16, 4, 10, 10}, {5}, {5}, {5}}
};


std::vector<std::vector<InputShape>> dynamicInputShapes5D1 = {
    {
        {{-1, -1, -1, -1, -1}, {{8, 16, 4, 10, 10}, {16, 10, 5, 11, 9}, {24, 6, 6, 8, 8}}},
        {{5}, {{5}, {5}, {5}}},
        {{5}, {{5}, {5}, {5}}},
        {{5}, {{5}, {5}, {5}}}
    },
    {
        {{{8, 16}, {8, 16}, {2, 7}, -1, -1}, {{8, 16, 2, 6, 8}, {8, 10, 4, 7, 5}, {16, 8, 7, 5, 10}}},
        {{5}, {{5}, {5}, {5}}},
        {{5}, {{5}, {5}, {5}}},
        {{5}, {{5}, {5}, {5}}}
    }
};

std::vector<std::vector<InputShape>> dynamicInputShapes5D1Blocked = {
    {
        {{-1, 16, -1, -1, -1}, {{24, 16, 3, 6, 7}, {48, 16, 4, 5, 5}, {24, 16, 5, 8, 5}}},
        {{5}, {{5}, {5}, {5}}},
        {{5}, {{5}, {5}, {5}}},
        {{5}, {{5}, {5}, {5}}}
    }
};

const std::vector<std::vector<int64_t>> blockShape5D2  = {{1, 2, 4, 3, 1}, {1, 1, 2, 4, 3}};
const std::vector<std::vector<int64_t>> cropsBegin5D2  = {{0, 0, 1, 2, 0}, {0, 0, 1, 0, 1}};
const std::vector<std::vector<int64_t>> cropsEnd5D2    = {{0, 0, 1, 0, 1}, {0, 0, 1, 1, 1}};

std::vector<std::vector<ov::Shape>> staticInputShapes5D2 = {
    {{48, 16, 3, 3, 3}, {5}, {5}, {5}}
};

std::vector<std::vector<InputShape>> dynamicInputShapes5D2 = {
    {
        {{-1, -1, -1, -1, -1}, {{48, 4, 3, 3, 3}, {24, 16, 5, 3, 5}, {24, 8, 7, 5, 5}}},
        {{5}, {{5}, {5}, {5}}},
        {{5}, {{5}, {5}, {5}}},
        {{5}, {{5}, {5}, {5}}}
    },
    {
        {{24, {8, 16}, {3, 5}, -1, -1}, {{24, 16, 3, 4, 3}, {24, 12, 5, 3, 5}, {24, 8, 4, 5, 5}}},
        {{5}, {{5}, {5}, {5}}},
        {{5}, {{5}, {5}, {5}}},
        {{5}, {{5}, {5}, {5}}}
    },
    {
        // special case
        {{{1, 24}, {1, 16}, {1, 10}, {1, 10}, {1, 10}},
        {
            {24, 16, 5, 3, 5},
            {24, 16, 5, 3, 5},
            {24, 16, 7, 5, 5}
        }},
        {{5}, {{5}, {5}, {5}}},
        {{5}, {{5}, {5}, {5}}},
        {{5}, {{5}, {5}, {5}}}
    }
};

std::vector<std::vector<InputShape>> dynamicInputShapes5D2Blocked = {
    {
        {{-1, 16, -1, -1, -1}, {{24, 16, 4, 5, 5}, {48, 16, 3, 4, 3}, {24, 16, 5, 3, 5}}},
        {{5}, {{5}, {5}, {5}}},
        {{5}, {{5}, {5}, {5}}},
        {{5}, {{5}, {5}, {5}}}
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
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes5D1)),
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
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes5D2)),
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
