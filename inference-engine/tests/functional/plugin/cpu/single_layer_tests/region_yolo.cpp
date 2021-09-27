// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

struct regionYoloAttributes {
    size_t classes;
    size_t coordinates;
    size_t num_regions;
    bool do_softmax;
    int start_axis;
    int end_axis;
};

using regionYoloParamsTuple = std::tuple<
        ngraph::Shape,                  // Input Shape
        regionYoloAttributes,               // Params
        std::vector<int64_t>,           // mask
        InferenceEngine::Precision,     // Network input precision
        InferenceEngine::Precision,     // Network output precision
        std::map<std::string, std::string>, // Additional network configuration
        std::string>;                   // Device name


class RegionYoloCPULayerTest : public testing::WithParamInterface<regionYoloParamsTuple>,
                               virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<regionYoloParamsTuple> obj) {
        ngraph::Shape inputShape;
        regionYoloAttributes attributes;
        std::vector<int64_t> mask;
        InferenceEngine::Precision inpPrecision;
        InferenceEngine::Precision outPrecision;
        std::string targetName;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShape, attributes, mask, inpPrecision, outPrecision, additionalConfig, targetName) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "classes=" << attributes.classes << "_";
        result << "coords=" << attributes.coordinates << "_";
        result << "num=" << attributes.num_regions << "_";
        result << "doSoftmax=" << attributes.do_softmax << "_";
        result << "axis=" << attributes.start_axis << "_";
        result << "endAxis=" << attributes.end_axis << "_";
        result << "inpPRC=" << inpPrecision.name() << "_";
        result << "outPRC=" << outPrecision.name() << "_";
        result << "targetDevice=" << targetName << "_";
        return result.str();
    }
protected:
    void SetUp() override {
        ngraph::Shape inputShape;
        regionYoloAttributes attributes;
        std::vector<int64_t> mask;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShape, attributes, mask, inPrc, outPrc, additionalConfig, targetDevice) = this->GetParam();

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        selectedType = getPrimitiveType() + "_" + inPrc.name();

        const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
        auto paramRegionYolo = ngraph::builder::makeParams(ngPrc, {inputShape});

        const auto region_yolo = std::make_shared<ngraph::op::v0::RegionYolo>(paramRegionYolo[0],
                                                                              attributes.coordinates, attributes.classes, attributes.num_regions,
                                                                              attributes.do_softmax, mask, attributes.start_axis, attributes.end_axis);

        function = makeNgraphFunction(ngPrc, paramRegionYolo, region_yolo, "RegionYolo");
    }
};

TEST_P(RegionYoloCPULayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "RegionYolo");
}

namespace {
const std::vector<Precision> inpOutPrc = {Precision::BF16, Precision::FP32};

const std::map<std::string, std::string> additional_config;

const std::vector<ngraph::Shape> inShapes_caffe = {
        {1, 125, 13, 13}
};

const std::vector<ngraph::Shape> inShapes_mxnet = {
        {1, 75, 52, 52},
        {1, 75, 32, 32},
        {1, 75, 26, 26},
        {1, 75, 16, 16},
        {1, 75, 13, 13},
        {1, 75, 8, 8},
        {1, 303, 7, 7},
        {1, 303, 14, 14},
        {1, 303, 28, 28},
};

const std::vector<ngraph::Shape> inShapes_v3 = {
        {1, 255, 52, 52},
        {1, 255, 26, 26},
        {1, 255, 13, 13}
};

const std::vector<std::vector<int64_t>> masks = {
        {0, 1, 2},
        {3, 4, 5},
        {6, 7, 8}
};

const std::vector<bool> do_softmax = {true, false};
const std::vector<size_t> classes = {80, 20};
const std::vector<size_t> num_regions = {5, 9};

const regionYoloAttributes yoloV3attr = {80, 4, 9, false, 1, 3};

const auto testCase_yolov3 = ::testing::Combine(
        ::testing::ValuesIn(inShapes_v3),
        ::testing::Values(yoloV3attr),
        ::testing::Values(masks[2]),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::Values(additional_config),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const regionYoloAttributes yoloV3mxnetAttr = {20, 4, 9, false, 1, 3};

const auto testCase_yolov3_mxnet = ::testing::Combine(
        ::testing::ValuesIn(inShapes_mxnet),
        ::testing::Values(yoloV3mxnetAttr),
        ::testing::Values(masks[1]),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::Values(additional_config),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const regionYoloAttributes yoloV2caffeAttr = {20, 4, 5, true, 1, 3};

const auto testCase_yolov2_caffe = ::testing::Combine(
        ::testing::ValuesIn(inShapes_caffe),
        ::testing::Values(yoloV2caffeAttr),
        ::testing::Values(masks[0]),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::Values(additional_config),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsRegionYolov3CPU, RegionYoloCPULayerTest, testCase_yolov3, RegionYoloCPULayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsRegionYoloMxnetCPU, RegionYoloCPULayerTest, testCase_yolov3_mxnet, RegionYoloCPULayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsRegionYoloCaffeCPU, RegionYoloCPULayerTest, testCase_yolov2_caffe, RegionYoloCPULayerTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
