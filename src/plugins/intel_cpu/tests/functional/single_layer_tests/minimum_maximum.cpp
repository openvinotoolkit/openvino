// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "test_utils/fusing_test_utils.hpp"
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>


using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<InputShape>,         // Input shapes
        ngraph::helpers::MinMaxOpType,   // Operation type
        ElementType,                     // Net precision
        ngraph::helpers::InputLayerType, // Second input type: Parameter or Constant
        ov::AnyMap                       // Additional network configuration
> basicMinMaxParams;

typedef std::tuple<
        basicMinMaxParams,
        CPUSpecificParams> MinMaxLayerCPUTestParamSet;

class MinMaxCPULayerTest : public testing::WithParamInterface<MinMaxLayerCPUTestParamSet>,
                           virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MinMaxLayerCPUTestParamSet>& obj) {
        basicMinMaxParams basicParams;
        CPUSpecificParams cpuParams;
        std::tie(basicParams, cpuParams) = obj.param;

        std::vector<InputShape> inputShapes;
        ngraph::helpers::MinMaxOpType opType;
        ElementType netPrecision;
        ngraph::helpers::InputLayerType layerType;
        ov::AnyMap config;

        std::tie(inputShapes, opType, netPrecision, layerType, config) = basicParams;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : inputShapes) {
            result << CommonTestUtils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=(";
        for (const auto& shape : inputShapes) {
            for (const auto& item : shape.second) {
                result << CommonTestUtils::vec2str(item) << "_";
            }
        }
        if (opType == ngraph::helpers::MinMaxOpType::MINIMUM) {
            result << "opType=MIN_";
        } else {
            result << "opType=MAX_";
        }
        result << "netPRC=" << netPrecision << "_";
        result << "type=" << layerType;
        for (auto const& configItem : config) {
            result << "_configItem=" << configItem.first << "_";
            configItem.second.print(result);
        }

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        basicMinMaxParams basicParams;
        CPUSpecificParams cpuParams;
        std::tie(basicParams, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        std::vector<InputShape> inputShapes;
        ngraph::helpers::MinMaxOpType opType;
        ElementType netPrecision;
        ngraph::helpers::InputLayerType layerType;

        std::tie(inputShapes, opType, netPrecision, layerType, configuration) = basicParams;

        init_input_shapes(inputShapes);

        auto params = ngraph::builder::makeDynamicParams(netPrecision, inputDynamicShapes);
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(params));

        auto maxMinNode = ngraph::builder::makeMinMax(paramOuts[0], paramOuts[1], opType);

        if (netPrecision == ElementType::i64 || netPrecision == ElementType::u64) {
            auto i64It = configuration.find(InferenceEngine::PluginConfigInternalParams::KEY_CPU_NATIVE_I64);
            if (i64It == configuration.end() || i64It->second == InferenceEngine::PluginConfigParams::NO) {
                selectedType = makeSelectedTypeStr(getPrimitiveType(), ElementType::i32);
            } else {
                selectedType = makeSelectedTypeStr(getPrimitiveType(), ElementType::i64);
            }
        } else if (netPrecision == ElementType::boolean) {
            selectedType = makeSelectedTypeStr(getPrimitiveType(), ElementType::i8);
        } else {
            selectedType = makeSelectedTypeStr(getPrimitiveType(), netPrecision);
        }

        function = makeNgraphFunction(netPrecision, params, maxMinNode, "MinMax");
    }
};

TEST_P(MinMaxCPULayerTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<ElementType> netPrecisions = { ElementType::f32, ElementType::i32 };

std::vector<std::vector<InputShape>> inShapesStatic = {
    { {{}, {{2}}}, {{}, {{1}}} },
    { {{}, {{1, 1, 1, 3}}}, {{}, {{1}}} },
    { {{}, {{1, 2, 4}}}, {{}, {{1}}} },
    { {{}, {{1, 4, 4}}}, {{}, {{1}}} },
    { {{}, {{1, 4, 4, 1}}}, {{}, {{1}}} },
    { {{}, {{256, 56}}}, {{}, {{256, 56}}} },
    { {{}, {{8, 1, 6, 1}}}, {{}, {{7, 1, 5}}} }
};

const std::vector<ngraph::helpers::MinMaxOpType> opType = {
        ngraph::helpers::MinMaxOpType::MINIMUM,
        ngraph::helpers::MinMaxOpType::MAXIMUM,
};

const std::vector<ngraph::helpers::InputLayerType> inputType = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

ov::AnyMap config = {};
ov::AnyMap config_i64 = {{InferenceEngine::PluginConfigInternalParams::KEY_CPU_NATIVE_I64, InferenceEngine::PluginConfigParams::YES}};

INSTANTIATE_TEST_SUITE_P(smoke_MinMax, MinMaxCPULayerTest,
                ::testing::Combine(
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapesStatic),
                                ::testing::ValuesIn(opType),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(inputType),
                                ::testing::Values(config)),
                        testing::Values(emptyCPUSpec)),
                MinMaxCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MinMax_I64, MinMaxCPULayerTest,
                ::testing::Combine(
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapesStatic),
                                ::testing::ValuesIn(opType),
                                ::testing::Values(ElementType::i64),
                                ::testing::ValuesIn(inputType),
                                ::testing::Values(config_i64)),
                        testing::Values(emptyCPUSpec)),
                MinMaxCPULayerTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
