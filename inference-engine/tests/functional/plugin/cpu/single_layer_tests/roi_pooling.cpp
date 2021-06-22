// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <shared_test_classes/single_layer/roi_pooling.hpp>
#include "ie_common.h"
#include "test_utils/cpu_test_utils.hpp"
#include "utils/bfloat16.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using ROIPoolingCPUTestParamsSet = std::tuple<LayerTestsDefinitions::roiPoolingParamsTuple,
                                              CPUSpecificParams,
                                              std::map<std::string, std::string>>;

class ROIPoolingCPULayerTest : public testing::WithParamInterface<ROIPoolingCPUTestParamsSet>,
                               virtual public LayerTestsUtils::LayerTestsCommon,
                               public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ROIPoolingCPUTestParamsSet> obj) {
        LayerTestsDefinitions::roiPoolingParamsTuple basicParamsSet;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;

        std::tie(basicParamsSet, cpuParams, additionalConfig) = obj.param;
        std::ostringstream result;

        result << LayerTestsDefinitions::ROIPoolingLayerTest::getTestCaseName(
            testing::TestParamInfo<LayerTestsDefinitions::roiPoolingParamsTuple>(basicParamsSet, 0));
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
    void GenerateInputs() override {
        auto feat_map_shape = cnnNetwork.getInputShapes().begin()->second;

        const auto is_roi_max_mode = (pool_method == ngraph::helpers::ROIPoolingTypes::ROI_MAX);

        const int height = is_roi_max_mode ? feat_map_shape[2] / spatial_scale : 1;
        const int width = is_roi_max_mode ? feat_map_shape[3] / spatial_scale : 1;

        size_t it = 0;
        for (const auto &input : cnnNetwork.getInputsInfo()) {
            const auto &info = input.second;
            InferenceEngine::Blob::Ptr blob;

            if (it == 1) {
                blob = make_blob_with_precision(info->getTensorDesc());
                blob->allocate();
                switch (inPrc) {
                case Precision::FP32: {
                    CommonTestUtils::fill_data_roi<Precision::FP32>
                        (blob, feat_map_shape[0] - 1, height, width, 1.0f, is_roi_max_mode);
                    break;
                }
                case Precision::BF16: {
                    CommonTestUtils::fill_data_roi<Precision::BF16>
                        (blob, feat_map_shape[0] - 1, height, width, 1.0f, is_roi_max_mode);
                    break;
                }
                default:
                    IE_THROW() << "roi_pooling. Unsupported precision";
                    break;
                }
            } else {
                blob = GenerateInput(*info);
            }
            inputs.push_back(blob);
            it++;
        }
    }

    void SetUp() override {
        LayerTestsDefinitions::roiPoolingParamsTuple basicParamsSet;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;

        InferenceEngine::SizeVector inputShape;
        InferenceEngine::SizeVector coordsShape;
        InferenceEngine::SizeVector poolShape;
        InferenceEngine::Precision netPrecision;

        std::tie(basicParamsSet, cpuParams, additionalConfig) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(inputShape, coordsShape, poolShape, spatial_scale, pool_method, netPrecision, targetDevice) = basicParamsSet;

        if (additionalConfig[PluginConfigParams::KEY_ENFORCE_BF16] == PluginConfigParams::YES)
            inPrc = outPrc = netPrecision = Precision::BF16;
        else
            inPrc = outPrc = netPrecision;
        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape, coordsShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        std::shared_ptr<ngraph::Node> roi_pooling = ngraph::builder::makeROIPooling(paramOuts[0], paramOuts[1], poolShape, spatial_scale, pool_method);
        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(roi_pooling)};

        function = makeNgraphFunction(ngPrc, params, roi_pooling, "roi_pooling");

        selectedType += "_";
        selectedType += netPrecision.name();
    }

private:
    ngraph::helpers::ROIPoolingTypes pool_method;
    float spatial_scale;
};

TEST_P(ROIPoolingCPULayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
    CheckPluginRelatedResults(executableNetwork, "ROIPooling");
}

namespace {

std::vector<std::map<std::string, std::string>> additionalConfig
    = {{{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO}},
       {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}}};

/* have to select particular implementation type, since currently
 * nodes always choose the best one */
std::vector<CPUSpecificParams> selectCPUInfoForDevice() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw16c, nc}, {nChw16c}, {"jit_avx512"}, "jit_avx512"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, nc}, {nChw8c}, {"jit_avx2"}, "jit_avx2"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, nc}, {nChw8c}, {"jit_sse42"}, "jit_sse42"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, nc}, {nChw8c}, {"ref"}, "ref"});
    }

    return resCPUParams;
}

const std::vector<std::vector<size_t>> inShapes = {{1, 3, 8, 8},
                                                   {3, 4, 50, 50}};

const std::vector<std::vector<size_t>> pooledShapes_max = {{1, 1},
                                                           {2, 2},
                                                           {3, 3},
                                                           {6, 6}};

const std::vector<std::vector<size_t>> pooledShapes_bilinear = {{1, 1},
                                                                {2, 2},
                                                                {3, 3},
                                                                {6, 6}};

const std::vector<std::vector<size_t>> coordShapes = {{1, 5},
                                                      {3, 5},
                                                      {5, 5}};

const std::vector<InferenceEngine::Precision> netPRCs = {InferenceEngine::Precision::FP32, InferenceEngine::Precision::BF16};

const std::vector<float> spatial_scales = {0.625f, 1.f};

const auto test_ROIPooling_max = ::testing::Combine(::testing::ValuesIn(inShapes),
                                                    ::testing::ValuesIn(coordShapes),
                                                    ::testing::ValuesIn(pooledShapes_max),
                                                    ::testing::ValuesIn(spatial_scales),
                                                    ::testing::Values(ngraph::helpers::ROIPoolingTypes::ROI_MAX),
                                                    ::testing::ValuesIn(netPRCs),
                                                    ::testing::Values(CommonTestUtils::DEVICE_CPU));

const auto test_ROIPooling_bilinear = ::testing::Combine(::testing::ValuesIn(inShapes),
                                                         ::testing::ValuesIn(coordShapes),
                                                         ::testing::ValuesIn(pooledShapes_bilinear),
                                                         ::testing::Values(spatial_scales[1]),
                                                         ::testing::Values(ngraph::helpers::ROIPoolingTypes::ROI_BILINEAR),
                                                         ::testing::ValuesIn(netPRCs),
                                                         ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_ROIPoolingCPU_max,
                        ROIPoolingCPULayerTest,
                        ::testing::Combine(test_ROIPooling_max,
                                           ::testing::ValuesIn(selectCPUInfoForDevice()),
                                           ::testing::ValuesIn(additionalConfig)),
                        ROIPoolingCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ROIPoolingCPU_bilinear,
                        ROIPoolingCPULayerTest,
                        ::testing::Combine(test_ROIPooling_bilinear,
                                           ::testing::ValuesIn(selectCPUInfoForDevice()),
                                           ::testing::ValuesIn(additionalConfig)),
                        ROIPoolingCPULayerTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
