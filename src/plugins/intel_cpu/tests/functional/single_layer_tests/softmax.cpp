// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph_functions/builders.hpp>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

struct SoftMaxConfig {
    ov::test::InputShape inputShape;
    size_t axis;
};

typedef std::tuple<ElementType,    // netPrecision
                   SoftMaxConfig,  // softmaxTestConfig
                   std::string,    // targetDevice
                   CPUSpecificParams>
    softmaxCPUTestParams;

class SoftMaxLayerCPUTest : public testing::WithParamInterface<softmaxCPUTestParams>,
                            virtual public SubgraphBaseTest,
                            public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<softmaxCPUTestParams>& obj) {
        CPUSpecificParams cpuParams;
        ElementType inType;
        SoftMaxConfig config;
        std::string targetDevice;
        std::tie(inType, config, targetDevice, cpuParams) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << inType << "_";
        result << "IS=" << CommonTestUtils::partialShape2str({config.inputShape.first}) << "_";
        result << "TS=";
        for (const auto& shape : config.inputShape.second) {
            result << "(";
            result << CommonTestUtils::vec2str(shape);
            result << ")_";
        }
        result << "axis=" << config.axis << "_";
        result << "trgDev=" << targetDevice;
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() override {
        ElementType inType;
        SoftMaxConfig config;
        CPUSpecificParams cpuParams;
        std::tie(inType, config, targetDevice, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        if (selectedType.empty()) {
            selectedType = getPrimitiveType();
        }

        if (inType == ElementType::bf16) {
            rel_threshold = 1e-2f;
        }
        selectedType = makeSelectedTypeStr(selectedType, inType);
        init_input_shapes({config.inputShape});
        auto params = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);

        const auto paramOuts =
            ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const auto softMax = std::make_shared<ngraph::opset1::Softmax>(paramOuts.at(0), config.axis);

        function = makeNgraphFunction(inType, params, softMax, "SoftMax");
    }
};

TEST_P(SoftMaxLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    CheckPluginRelatedResults(compiledModel, "Softmax");
}

namespace {
// not optimized cpu spec
const auto notOptimizedCPUSpec = CPUSpecificParams{{}, {}, {"ref_any"}, "ref_any"};

const std::vector<SoftMaxConfig> optimizedConfigsFP32 = {
    // Static shapes
    {ov::test::InputShape{ov::PartialShape{1, 100}, {ov::Shape{1, 100}}}, 1},
    {ov::test::InputShape{ov::PartialShape{10, 10}, {ov::Shape{10, 10}}}, 1},
    {ov::test::InputShape{ov::PartialShape{100, 1}, {ov::Shape{100, 1}}}, 0},
    {ov::test::InputShape{ov::PartialShape{100, 1}, {ov::Shape{100, 1}}}, 1},
    {ov::test::InputShape{ov::PartialShape{5, 5, 1}, {ov::Shape{5, 5, 1}}}, 1},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5}, {ov::Shape{5, 5, 5}}}, 2},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5}}}, 0},
    {ov::test::InputShape{ov::PartialShape{5, 5, 1, 1}, {ov::Shape{5, 5, 1, 1}}}, 1},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5}}}, 1},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 1}, {ov::Shape{5, 5, 5, 1}}}, 2},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5}}}, 2},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5}}}, 3},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5}}}, 0},
    {ov::test::InputShape{ov::PartialShape{5, 5, 1, 1, 1}, {ov::Shape{5, 5, 1, 1, 1}}}, 1},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5}}}, 1},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 1, 1}, {ov::Shape{5, 5, 5, 1, 1}}}, 2},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5}}}, 2},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 1, 1}, {ov::Shape{5, 5, 5, 1, 1}}}, 3},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5}}}, 3},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 1}, {ov::Shape{5, 5, 5, 5, 1}}}, 4},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5}}}, 4},
    // Dynamic shapes
    {ov::test::InputShape{// dynamic shape
                          ov::PartialShape{-1, -1},
                          {// target static shapes
                           ov::Shape{10, 10},
                           ov::Shape{15, 15},
                           ov::Shape{10, 10},
                           ov::Shape{10, 5}}},
     1},
    {ov::test::InputShape{// dynamic shape
                          ov::PartialShape{-1, -1, 1, 1, 1},
                          {// target static shapes
                           ov::Shape{5, 5, 1, 1, 1},
                           ov::Shape{10, 7, 1, 1, 1},
                           ov::Shape{5, 5, 1, 1, 1}}},
     1},
    {ov::test::InputShape{// dynamic shape
                          ov::PartialShape{{1, 10}, 10},
                          {// target static shapes
                           ov::Shape{10, 10},
                           ov::Shape{5, 10}}},
     1},
};

const std::vector<SoftMaxConfig> notOptimizedConfigsFP32{
    // Static shapes
    {ov::test::InputShape{ov::PartialShape{1, 100}, {ov::Shape{1, 100}}}, 0},
    {ov::test::InputShape{ov::PartialShape{10, 10}, {ov::Shape{10, 10}}}, 0},
    {ov::test::InputShape{ov::PartialShape{10, 10, 10}, {ov::Shape{10, 10, 10}}}, 0},
    {ov::test::InputShape{ov::PartialShape{10, 10, 10}, {ov::Shape{10, 10, 10}}}, 1},
    // Dynamic shapes
    {ov::test::InputShape{// dynamic shape
                          ov::PartialShape{-1, -1},
                          {// target static shapes
                           ov::Shape{10, 1},
                           ov::Shape{15, 15},
                           ov::Shape{10, 5},
                           ov::Shape{15, 15}}},
     0},
    {ov::test::InputShape{// dynamic shape
                          ov::PartialShape{ov::Dimension{1, 100}, ov::Dimension{1, 100}, -1},
                          {// target static shapes
                           ov::Shape{10, 10, 10},
                           ov::Shape{10, 10, 1},
                           ov::Shape{10, 5, 10},
                           ov::Shape{10, 10, 1}}},
     1},
    {ov::test::InputShape{// dynamic shape
                          ov::PartialShape{{1, 10}, 10},
                          {// target static shapes
                           ov::Shape{7, 10},
                           ov::Shape{10, 10}}},
     0},
};

const std::vector<SoftMaxConfig> unsupportedConfigsFP32{
    // Static shapes
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5, 5}}}, 0},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5, 5}}}, 1},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5, 5}}}, 2},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5, 5}}}, 3},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5, 5}}}, 4},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5, 5}}}, 5},
    // Dynamic shapes
    {ov::test::InputShape{// dynamic shape
                          ov::PartialShape{-1, -1, -1, -1, -1, -1},
                          {// target static shapes
                           ov::Shape{5, 5, 5, 5, 5, 5},
                           ov::Shape{7, 7, 7, 7, 7, 7},
                           ov::Shape{5, 5, 5, 5, 5, 5}}},
     4},
    {ov::test::InputShape{// dynamic shape
                          ov::PartialShape{{1, 10}, 5, 5, 5, 5, 5},
                          {// target static shapes
                           ov::Shape{5, 5, 5, 5, 5, 5},
                           ov::Shape{7, 5, 5, 5, 5, 5}}},
     4},
};

const auto avx512 = CPUSpecificParams{{}, {}, {"jit_avx512"}, "jit_avx512"};
const auto avx2 = CPUSpecificParams{{}, {}, {"jit_avx2"}, "jit_avx2"};
const auto sse42 = CPUSpecificParams{{}, {}, {"jit_sse42"}, "jit_sse42"};
const auto ref = CPUSpecificParams{{}, {}, {"ref_any"}, "ref_any"};

const std::vector<CPUSpecificParams> vecCpuConfigs = {ref, sse42, avx2, avx512};
const auto OptimizedParams = testing::Combine(testing::Values(ElementType::f32, ElementType::bf16),
                                              testing::ValuesIn(optimizedConfigsFP32),
                                              testing::Values(CommonTestUtils::DEVICE_CPU),
                                              testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax_Optimized_CPU,
                         SoftMaxLayerCPUTest,
                         OptimizedParams,
                         SoftMaxLayerCPUTest::getTestCaseName);

const auto NotOptimizedParams = testing::Combine(testing::Values(ElementType::f32, ElementType::bf16),
                                                 testing::ValuesIn(notOptimizedConfigsFP32),
                                                 testing::Values(CommonTestUtils::DEVICE_CPU),
                                                 testing::Values(notOptimizedCPUSpec));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax_CPU,
                         SoftMaxLayerCPUTest,
                         NotOptimizedParams,
                         SoftMaxLayerCPUTest::getTestCaseName);

const auto UnsupportedParams = testing::Combine(testing::Values(ElementType::f32, ElementType::bf16),
                                                testing::ValuesIn(unsupportedConfigsFP32),
                                                testing::Values(CommonTestUtils::DEVICE_CPU),
                                                testing::Values(notOptimizedCPUSpec));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax_Unsupported_CPU,
                         SoftMaxLayerCPUTest,
                         UnsupportedParams,
                         SoftMaxLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace CPULayerTestsDefinitions
