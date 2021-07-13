// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/topk.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        LayerTestsDefinitions::TopKParams,
        CPUSpecificParams,
        std::map<std::string, std::string>> TopKLayerCPUTestParamsSet;

class TopKLayerCPUTest : public testing::WithParamInterface<TopKLayerCPUTestParamsSet>,
                                     virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TopKLayerCPUTestParamsSet> obj) {
        LayerTestsDefinitions::TopKParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, cpuParams, additionalConfig) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::TopKLayerTest::getTestCaseName(
                     testing::TestParamInfo<LayerTestsDefinitions::TopKParams>(basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto &item : additionalConfig) {
                if (item.second == PluginConfigParams::YES)
                    result << "_" << item.first << "=" << item.second;
            }
        }

        return result.str();
    }

protected:
    void SetUp() {
        LayerTestsDefinitions::TopKParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, cpuParams, additionalConfig) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        InferenceEngine::SizeVector inputShape;
        InferenceEngine::Precision netPrecision;
        int64_t keepK, axis;
        ngraph::opset4::TopK::Mode mode;
        ngraph::opset4::TopK::SortType sort;
        std::tie(keepK, axis, mode, sort, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) = basicParamsSet;

        if (additionalConfig[PluginConfigParams::KEY_ENFORCE_BF16] == PluginConfigParams::YES)
            inPrc = outPrc = netPrecision = Precision::BF16;
        else
            inPrc = outPrc = netPrecision;
        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        // Spec TopK_3.md allows to use unstable sorting, thus
        // a. Skip comparing of index results, because an element in actual index tensor can be different with
        //    its counterpart in expected index tensor
        // b. If SortType is SORT_INDICES or NONE, the test program still needs to apply std::sort for all pairs
        //    of 1xk value vectors in expected and actual output tensor before comparing them
        size_t axis_idx = axis < 0 ? static_cast<size_t>(axis + static_cast<int64_t>(inputShape.size())) : static_cast<size_t>(axis);
        if (sort == ngraph::opset4::TopK::SortType::SORT_VALUES)
            setCustomizedCompare(false, true, static_cast<size_t>(keepK), axis_idx, inputShape);
        else
            setCustomizedCompare(false, false, static_cast<size_t>(keepK), axis_idx, inputShape);

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramIn = ngraph::helpers::convert2OutputVector(
                            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        auto k = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{}, &keepK);
        auto topk = std::dynamic_pointer_cast<ngraph::opset4::TopK>(
                std::make_shared<ngraph::opset4::TopK>(paramIn[0], k, axis, mode, sort));
        topk->get_rt_info() = getCPUInfo();

        ngraph::ResultVector results;
        for (int i = 0; i < topk->get_output_size(); i++) {
            results.push_back(std::make_shared<ngraph::opset4::Result>(topk->output(i)));
        }
        function = std::make_shared<ngraph::Function>(results, params, "TopK");

        selectedType += "_";
        selectedType += netPrecision.name();
    }
};

TEST_P(TopKLayerCPUTest, CompareWithRefs) {
    if (FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()) {
        clearCustomizedCompare();
    }
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    clearCustomizedCompare();
    CheckPluginRelatedResults(executableNetwork, "TopK");
}

namespace {

std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw, nchw}, {"jit_avx512"}, "jit_avx512"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc, nhwc}, {"jit_avx512"}, "jit_avx512"});
        resCPUParams.push_back(CPUSpecificParams{{nChw16c, x}, {nChw16c, nChw16c}, {"jit_avx512"}, "jit_avx512"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw, nchw}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc, nhwc}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x}, {nChw8c, nChw8c}, {"jit_avx2"}, "jit_avx2"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw, nchw}, {"jit_sse42"}, "jit_sse42"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc, nhwc}, {"jit_sse42"}, "jit_sse42"});
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x}, {nChw8c, nChw8c}, {"jit_sse42"}, "jit_sse42"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw, nchw}, {"ref"}, "ref"});
    }
    return resCPUParams;
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::BF16
};

std::vector<std::map<std::string, std::string>> additionalConfig = {
    {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO}},
    {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}}
};

const std::vector<int64_t> axes = {0, 1, 2, 3};
const std::vector<int64_t> k = {1, 5, 9, 10};

const std::vector<ngraph::opset4::TopK::Mode> modes = {
    ngraph::opset4::TopK::Mode::MIN,
    ngraph::opset4::TopK::Mode::MAX
};

const std::vector<ngraph::opset4::TopK::SortType> sortTypes = {
    ngraph::opset4::TopK::SortType::SORT_VALUES,
    ngraph::opset4::TopK::SortType::SORT_INDICES,
    ngraph::opset4::TopK::SortType::NONE
};

INSTANTIATE_TEST_CASE_P(smoke_TopK, TopKLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::ValuesIn(k),
            ::testing::ValuesIn(axes),
            ::testing::ValuesIn(modes),
            ::testing::ValuesIn(sortTypes),
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t>({10, 10, 10, 10})),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice()),
        ::testing::ValuesIn(additionalConfig)),
    TopKLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
