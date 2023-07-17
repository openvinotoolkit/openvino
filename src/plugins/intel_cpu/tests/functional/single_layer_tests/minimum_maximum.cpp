// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"


using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<InputShape>,         // Input shapes
        ngraph::helpers::MinMaxOpType,   // Operation type
        ElementType,                     // Net precision
        ngraph::helpers::InputLayerType, // Second input type: Parameter or Constant
        ov::AnyMap                       // Additional plugin configuration
> basicMinMaxParams;

typedef std::tuple<
        basicMinMaxParams,
        CPUSpecificParams> MinMaxLayerCPUTestParamSet;

class MinMaxCPULayerTest : public testing::WithParamInterface<MinMaxLayerCPUTestParamSet>,
                           virtual public SubgraphBaseTest, public CPUTestsBase {
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
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=(";
        for (const auto& shape : inputShapes) {
            for (const auto& item : shape.second) {
                result << ov::test::utils::vec2str(item) << "_";
            }
        }
        if (opType == ngraph::helpers::MinMaxOpType::MINIMUM) {
            result << "opType=MIN_";
        } else {
            result << "opType=MAX_";
        }
        result << "netPRC=" << netPrecision << "_";
        result << "type=" << layerType;

        if (!config.empty()) {
            result << "_PluginConf{";
            for (const auto& configItem : config) {
                result << "_" << configItem.first << "=";
                configItem.second.print(result);
            }
            result << "}";
        }

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        basicMinMaxParams basicParams;
        CPUSpecificParams cpuParams;
        std::tie(basicParams, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        std::vector<InputShape> inputShapes;
        ngraph::helpers::MinMaxOpType opType;
        ngraph::helpers::InputLayerType layerType;

        std::tie(inputShapes, opType, inType, layerType, configuration) = basicParams;

        init_input_shapes(inputShapes);
        selectedType = makeSelectedTypeStr(getPrimitiveType(), inType, configuration);

        auto params = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);
        auto opMinMax = ngraph::builder::makeMinMax(params[0], params[1], opType);

        function = makeNgraphFunction(inType, params, opMinMax, "MinMax");
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

INSTANTIATE_TEST_SUITE_P(smoke_MinMax, MinMaxCPULayerTest,
                ::testing::Combine(
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapesStatic),
                                ::testing::ValuesIn(opType),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(inputType),
                                ::testing::Values(emptyPluginConfig)),
                        testing::Values(emptyCPUSpec)),
                MinMaxCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_I64, MinMaxCPULayerTest,
                ::testing::Combine(
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapesStatic),
                                ::testing::ValuesIn(opType),
                                ::testing::Values(ElementType::i64),
                                ::testing::ValuesIn(inputType),
                                ::testing::Values(cpuI64PluginConfig)),
                        testing::Values(emptyCPUSpec)),
                MinMaxCPULayerTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
