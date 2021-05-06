// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/gather_elements.hpp>
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace CPULayerTestsDefinitions  {

typedef std::tuple<
        GatherElementsParams,
        CPUSpecificParams
    > GatherElementsCPUTestParamSet;

class GatherElementsCPUTest : public testing::WithParamInterface<GatherElementsCPUTestParamSet>,
                            virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherElementsCPUTestParamSet> &obj) {
        GatherElementsParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << GatherElementsLayerTest::getTestCaseName(testing::TestParamInfo<GatherElementsParams>(basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 15, 0, 32768);
    }

protected:
    void SetUp() override {
        InferenceEngine::SizeVector dataShape, indicesShape;
        InferenceEngine::Precision dPrecision, iPrecision;
        int axis;

        GatherElementsParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        std::tie(dataShape, indicesShape, axis, dPrecision, iPrecision, targetDevice) = basicParamsSet;
        selectedType = std::string("ref_any_") + dPrecision.name();

        auto ngDPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dPrecision);
        auto ngIPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(iPrecision);

        auto params = ngraph::builder::makeParams(ngDPrc, {dataShape});
        auto activation = ngraph::builder::makeGatherElements(params[0], indicesShape, ngIPrc, axis);
        activation->get_rt_info() = getCPUInfo();
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{activation}, params, "GatherElements");
    }
};

TEST_P(GatherElementsCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "GatherElements");
}


namespace {
std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nchw}, {nchw}, {}, {})
};

INSTANTIATE_TEST_CASE_P(smoke_set1, GatherElementsCPUTest,
            ::testing::Combine(
                ::testing::Combine(
                    ::testing::Values(std::vector<size_t>({2, 3, 5, 7})),     // Data shape
                    ::testing::Values(std::vector<size_t>({2, 3, 9, 7})),     // Indices shape
                    ::testing::ValuesIn(std::vector<int>({2, -2})),           // Axis
                    ::testing::Values(Precision::BF16),
                    ::testing::Values(Precision::I32),
                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D))),
        GatherElementsCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
