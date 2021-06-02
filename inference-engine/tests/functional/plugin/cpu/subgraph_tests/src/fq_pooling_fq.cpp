// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/include/fq_pooling_fq.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph;

namespace SubgraphTestsDefinitions {

std::string FQPoolingFQSubgraphTest::getTestCaseName(testing::TestParamInfo<FQPoolingFQCpuTestParamsSet> obj) {
    std::ostringstream result;
    CPUSpecificParams cpuParams;
    SizeVector inputShape;
    PoolingParams poolParams;
    std::vector<size_t> kernel;
    std::vector<size_t> strides;
    std::vector<size_t> padsBegin, padsEnd;
    std::tie(inputShape, poolParams, cpuParams) = obj.param;
    std::tie(kernel, strides, padsBegin, padsEnd) = poolParams;

    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "K=" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S=" << CommonTestUtils::vec2str(strides) << "_";
    result << "PB=" << CommonTestUtils::vec2str(padsBegin) << "_";
    result << "PE=" << CommonTestUtils::vec2str(padsEnd) << "_";
    result << CPUTestsBase::getTestCaseName(cpuParams);

    return result.str();
}

/*  FQPoolingFQ graph
      ---------
      |Input  |
      ---------
          |
    -------------
    | --------- |
    |   |FQ|    |
    | --------- |
    |     |     |
    | --------- |
    | |AvgPool| |
    | --------- |
    |     |     |
    | --------- |
    |   |FQ|    |
    |-----------|
          |
      ---------
      |Output |
      ---------
*/

void FQPoolingFQSubgraphTest::SetUp() {
    CPUSpecificParams cpuParams;
    PoolingParams poolParams;
    std::vector<size_t> kernel;
    std::vector<size_t> strides;
    std::vector<size_t> padsBegin, padsEnd;
    std::tie(inputShape, poolParams, cpuParams) = this->GetParam();
    std::tie(kernel, strides, padsBegin, padsEnd) = poolParams;
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

    element::Type prc = element::f32;
    Shape shape = Shape(inputShape.size(), 1);
    auto params = ngraph::builder::makeParams(prc, {inputShape});
    auto firstFQ = ngraph::builder::makeFakeQuantize(params[0], prc, 256, shape);
    auto pooling = ngraph::builder::makePooling(firstFQ,
                                                strides,
                                                padsBegin,
                                                padsEnd,
                                                kernel,
                                                op::RoundingType::CEIL,
                                                op::PadType::EXPLICIT,
                                                false,
                                                helpers::PoolingTypes::AVG);
    auto secondFQ = ngraph::builder::makeFakeQuantize(pooling, prc, 256, shape);

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(secondFQ)};
    function = std::make_shared<ngraph::Function>(results, params, "FQPoolingFQ");
}

void FQPoolingFQSubgraphTest::CheckFQCount(const size_t expectedFQCount) {
    InferenceEngine::CNNNetwork execGraphInfo = executableNetwork.GetExecGraphInfo();
    auto function = execGraphInfo.getFunction();
    ASSERT_NE(nullptr, function);
    size_t actualFQCount = 0;
    for (const auto &node : function->get_ops()) {
        const auto & rtInfo = node->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
            IE_ASSERT(nullptr != value);
            return value->get();
        };
        if (getExecValue(ExecGraphInfoSerialization::LAYER_TYPE) == "FakeQuantize") {
            actualFQCount++;
        }
    }
    ASSERT_EQ(expectedFQCount, actualFQCount);
}

TEST_P(FQPoolingFQSubgraphTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckFQCount(1);
}

const auto avx512 = CPUSpecificParams{{nhwc}, {nhwc}, {"jit_avx512"}, "jit_avx512"};
const auto avx2 = CPUSpecificParams{{nhwc}, {nhwc}, {"jit_avx2"}, "jit_avx2"};
const auto sse42 = CPUSpecificParams{{nhwc}, {nhwc}, {"jit_sse42"}, "jit_sse42"};

const std::vector<CPUSpecificParams> vecCpuConfigsFusing = {sse42, avx2, avx512};

const std::vector<std::vector<size_t>> inputShapes4D = {
        std::vector<size_t>{1, 1, 64, 64},
        std::vector<size_t>{1, 3, 64, 64},
        std::vector<size_t>{1, 16, 64, 64},
};

const std::vector<PoolingParams> params4D = {
        PoolingParams{ {2, 2}, {2, 2}, {0, 0}, {0, 0} },
        PoolingParams{ {2, 2}, {2, 2}, {2, 0}, {0, 2} },
        PoolingParams{ {1, 4}, {1, 4}, {0, 0}, {0, 0} },
        PoolingParams{ {4, 1}, {4, 1}, {0, 0}, {0, 0} },
};

INSTANTIATE_TEST_CASE_P(smoke_FQPoolingFQ_4D, FQPoolingFQSubgraphTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes4D),
                                ::testing::ValuesIn(params4D),
                                ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigsFusing))),
                        FQPoolingFQSubgraphTest::getTestCaseName);

 const std::vector<std::vector<size_t>> inputShapes5D = {
         std::vector<size_t>{1, 1, 32, 32, 32},
         std::vector<size_t>{1, 3, 32, 32, 32},
         std::vector<size_t>{1, 16, 32, 32, 32},
};

const std::vector<PoolingParams> params5D = {
        PoolingParams{ {2, 2, 2}, {2, 2, 2}, {}, {} },
        PoolingParams{ {2, 2, 2}, {2, 2, 2}, {2, 0, 2}, {0, 2, 0} },
        PoolingParams{ {1, 1, 4}, {1, 1, 4}, {}, {} },
        PoolingParams{ {1, 4, 1}, {1, 4, 1}, {}, {} },
        PoolingParams{ {4, 1, 1}, {4, 1, 1}, {}, {} },
};

INSTANTIATE_TEST_CASE_P(smoke_FQPoolingFQ_5D, FQPoolingFQSubgraphTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes5D),
                                ::testing::ValuesIn(params5D),
                                ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigsFusing))),
                        FQPoolingFQSubgraphTest::getTestCaseName);

}  // namespace SubgraphTestsDefinitions
