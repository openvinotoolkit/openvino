// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/gather.hpp>
#include <ngraph/opsets/opset6.hpp>
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace LayerTestsDefinitions {

typedef std::tuple<
        std::vector<size_t>,               // Dictionary shape
        std::vector<size_t>,               // Indices shape
        int,                               // Axis
        InferenceEngine::Precision,        // Dictionary precision
        InferenceEngine::Precision,        // Indices precision
        InferenceEngine::Precision,        // Axis precision
        std::string,                       // Device name
        CPUSpecificParams
    > GatherCPUTestParamSet;

class GatherCpuSpecLayerTest : public testing::WithParamInterface<GatherCPUTestParamSet>,
                            virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherCPUTestParamSet>& obj) {
        std::vector<size_t> dictionaryShape, indicesShape;
        int axis;
        InferenceEngine::Precision dictPrecision, idxPrecision, axPrecision;
        std::string targetName;
        CPUSpecificParams cpuParams;
        std::tie(dictionaryShape, indicesShape, axis, dictPrecision, idxPrecision, axPrecision, targetName, cpuParams) = obj.param;

        std::ostringstream result;
        result << "dictShape=" << CommonTestUtils::vec2str(dictionaryShape) << "_";
        result << "idxShape=" << CommonTestUtils::vec2str(indicesShape) << "_";
        result << "Ax=" << axis << "_";
        result << "dictPrc=" << dictPrecision.name() << "_";
        result << "idxPrc=" << idxPrecision.name() << "_";
        result << "axPrc=" << axPrecision.name() << "_";
        result << "trgDev=" << targetName << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() override {
        std::vector<size_t> dictionaryShape, indicesShape;
        int axis;
        InferenceEngine::Precision dictPrecision, idxPrecision, axPrecision;
        CPUSpecificParams cpuParams;
        std::tie(dictionaryShape, indicesShape, axis, dictPrecision, idxPrecision, axPrecision, targetDevice, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        selectedType = std::string("unknown_") + dictPrecision.name();

        auto ngDictPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dictPrecision);
        auto ngIdxPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(idxPrecision);
        auto ngAxPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(axPrecision);

        auto functionParams = ngraph::builder::makeParams(ngDictPrc, {dictionaryShape});

        std::mt19937 gen(0);
        std::uniform_int_distribution<unsigned long> dist(0, dictionaryShape[axis] - 1);
        size_t idxSize = std::accumulate(indicesShape.begin(), indicesShape.end(), 1lu, std::multiplies<size_t>());
        std::vector<int> indicesData(idxSize, 0);
        for (int i = 0; i < idxSize; i++) {
            indicesData[i] = dist(gen);
        }
        auto indicesNode = ngraph::opset6::Constant::create(ngIdxPrc, ngraph::Shape(indicesShape), indicesData);

        auto axisNode = ngraph::opset6::Constant::create(ngAxPrc, ngraph::Shape({}), {axis});
        auto gatherNode = std::make_shared<ngraph::opset1::Gather>(functionParams[0], indicesNode, axisNode);
        gatherNode->get_rt_info() = getCPUInfo();

        ngraph::ResultVector results{std::make_shared<ngraph::opset6::Result>(gatherNode)};
        function = std::make_shared<ngraph::Function>(results, functionParams, "Gather");
    }
};

TEST_P(GatherCpuSpecLayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Gather");
}


namespace {
    std::vector<InferenceEngine::Precision> precisions = {InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::BF16, InferenceEngine::Precision::I32, InferenceEngine::Precision::I8};

    std::vector<CPUSpecificParams> cpuParams_1D = {
            CPUSpecificParams({x}, {}, {}, {})
    };
    std::vector<CPUSpecificParams> cpuParams_2D = {
            CPUSpecificParams({nc}, {}, {}, {})
    };
    std::vector<CPUSpecificParams> cpuParams_3D = {
            CPUSpecificParams({}, {}, {}, {})
    };
    std::vector<CPUSpecificParams> cpuParams_4D = {
            CPUSpecificParams({nchw}, {}, {}, {})
    };
    std::vector<CPUSpecificParams> cpuParams_5D = {
            CPUSpecificParams({ncdhw}, {}, {}, {})
    };

    INSTANTIATE_TEST_CASE_P(smoke_1D, GatherCpuSpecLayerTest,
                ::testing::Combine(
                    ::testing::ValuesIn(std::vector<std::vector<size_t>>({{3}, {8}, {16}, {33}, {47}})),  // Data shape
                    ::testing::ValuesIn(std::vector<std::vector<size_t>>({{2}, {2, 5}, {1, 3, 8}})),      // Indices shape
                    ::testing::ValuesIn(std::vector<int>({0})),                                           // Axis
                    ::testing::ValuesIn(precisions),
                    ::testing::ValuesIn(precisions),
                    ::testing::ValuesIn(precisions),
                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_1D))),
            GatherCpuSpecLayerTest::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_2D, GatherCpuSpecLayerTest,
                ::testing::Combine(
                    ::testing::ValuesIn(std::vector<std::vector<size_t>>({{3, 3}, {2, 8}, {3, 16}, {2, 33}, {2, 47}})),  // Data shape
                    ::testing::ValuesIn(std::vector<std::vector<size_t>>({{2}, {2, 5}, {1, 3, 8}, {2, 32}, /*{2, 320000}*/})),            // Indices shape
                    ::testing::ValuesIn(std::vector<int>({0, 1})),                                                       // Axis
                    ::testing::ValuesIn(precisions),
                    ::testing::ValuesIn(precisions),
                    ::testing::ValuesIn(precisions),
                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_2D))),
            GatherCpuSpecLayerTest::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_3D, GatherCpuSpecLayerTest,
                ::testing::Combine(
                        ::testing::ValuesIn(std::vector<std::vector<size_t>>({{3, 3, 3}, {2, 2, 8}, {3, 1, 16}, {2, 2, 33}, {2, 1, 47}})),  // Data shape
                        ::testing::ValuesIn(std::vector<std::vector<size_t>>({{2}, {2, 5}, {1, 3, 8}})),                                    // Indices shape
                        ::testing::ValuesIn(std::vector<int>({0, 1, 2})),                                                                   // Axis
                        ::testing::ValuesIn(precisions),
                        ::testing::ValuesIn(precisions),
                        ::testing::ValuesIn(precisions),
                        ::testing::Values(CommonTestUtils::DEVICE_CPU),
                    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_3D))),
            GatherCpuSpecLayerTest::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_4D, GatherCpuSpecLayerTest,
                ::testing::Combine(
                    ::testing::ValuesIn(std::vector<std::vector<size_t>>({{3, 3, 3, 3},
                        {2, 2, 2, 8}, {1, 3, 1, 16}, {1, 2, 2, 33}, {1, 2, 1, 47}})),                   // Data shape
                    ::testing::ValuesIn(std::vector<std::vector<size_t>>({{2}, {2, 5}, {1, 3, 8}})),    // Indices shape
                    ::testing::ValuesIn(std::vector<int>({0, 1, 3})),                                   // Axis
                    ::testing::ValuesIn(precisions),
                    ::testing::ValuesIn(precisions),
                    ::testing::ValuesIn(precisions),
                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D))),
            GatherCpuSpecLayerTest::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_5D, GatherCpuSpecLayerTest,
                ::testing::Combine(
                    ::testing::ValuesIn(std::vector<std::vector<size_t>>({{3, 3, 3, 3, 3},
                        {2, 2, 2, 2, 8}, {3, 1, 3, 1, 16}, {2, 1, 2, 2, 33}, {2, 1, 2, 1, 47}})),              // Data shape
                    ::testing::ValuesIn(std::vector<std::vector<size_t>>({{2}, {2, 5}, {1, 3, 8}, {1, 16}})),  // Indices shape
                    ::testing::ValuesIn(std::vector<int>({0, 2, 4})),                                          // Axis
                    ::testing::ValuesIn(precisions),
                    ::testing::ValuesIn(precisions),
                    ::testing::ValuesIn(precisions),
                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D))),
            GatherCpuSpecLayerTest::getTestCaseName);
} // namespace
} // namespace LayerTestsDefinitions
