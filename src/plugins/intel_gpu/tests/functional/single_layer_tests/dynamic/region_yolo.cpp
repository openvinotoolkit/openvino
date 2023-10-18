// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/region_yolo.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "ov_models/builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include <string>

using namespace ngraph;
using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

struct regionYoloAttributes {
    size_t classes;
    size_t coordinates;
    size_t num_regions;
    bool do_softmax;
    int start_axis;
    int end_axis;
};

typedef std::tuple<
        InputShape,                         // Input Shape
        regionYoloAttributes,               // Params
        std::vector<int64_t>,               // mask
        ov::test::ElementType,              // Network input precision
        ov::test::ElementType,              // Network output precision
        std::map<std::string, std::string>, // Additional network configuration
        std::string                         // Device name
> RegionYoloGPUTestParam;

class RegionYoloLayerGPUTest : public testing::WithParamInterface<RegionYoloGPUTestParam>,
                               virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RegionYoloGPUTestParam> obj) {
        InputShape inputShape;
        regionYoloAttributes attributes;
        std::vector<int64_t> mask;
        ov::test::ElementType inpPrecision;
        ov::test::ElementType outPrecision;
        std::string targetName;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShape, attributes, mask, inpPrecision, outPrecision, additionalConfig, targetName) = obj.param;

        std::ostringstream result;
        result << "IS=" << inputShape << "_";
        result << "classes=" << attributes.classes << "_";
        result << "coords=" << attributes.coordinates << "_";
        result << "num=" << attributes.num_regions << "_";
        result << "doSoftmax=" << attributes.do_softmax << "_";
        result << "axis=" << attributes.start_axis << "_";
        result << "endAxis=" << attributes.end_axis << "_";
        result << "inpPRC=" << inpPrecision << "_";
        result << "outPRC=" << outPrecision << "_";
        result << "targetDevice=" << targetName << "_";
        return result.str();
    }

protected:
    void SetUp() override {
        InputShape inputShape;
        regionYoloAttributes attributes;
        std::vector<int64_t> mask;
        ov::test::ElementType inPrc;
        ov::test::ElementType outPrc;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShape, attributes, mask, inPrc, outPrc, additionalConfig, targetDevice) = this->GetParam();

        init_input_shapes({ inputShape });

        ov::ParameterVector paramRegionYolo;
        for (auto&& shape : inputDynamicShapes) {
            paramRegionYolo.push_back(std::make_shared<ov::op::v0::Parameter>(inPrc, shape));
        }

        const auto region_yolo = std::make_shared<ngraph::op::v0::RegionYolo>(paramRegionYolo[0],
                                                                              attributes.coordinates, attributes.classes, attributes.num_regions,
                                                                              attributes.do_softmax, mask, attributes.start_axis, attributes.end_axis);

        ngraph::ResultVector results;
        for (size_t i = 0; i < region_yolo->get_output_size(); i++)
            results.push_back(std::make_shared<ngraph::opset1::Result>(region_yolo->output(i)));
        function = std::make_shared<ngraph::Function>(results, paramRegionYolo, "RegionYolo");
    }
};

TEST_P(RegionYoloLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

namespace {

std::map<std::string, std::string> emptyAdditionalConfig;

const std::vector<ov::test::ElementType> inpOutPrc = {ov::test::ElementType::f16, ov::test::ElementType::f32};

const std::vector<InputShape> inShapes_caffe_dynamic = {
        {{-1, -1, -1, -1}, {{1, 125, 13, 13}, {1, 125, 26, 26}}},
        {{{1, 2}, {100, 125}, {13, 26}, {13, 26}}, {{1, 125, 13, 13}, {1, 125, 26, 26}}}
};

const std::vector<InputShape> inShapes_mxnet_dynamic = {
        {{-1, -1, -1, -1}, {{1, 75, 52, 52}, {1, 75, 32, 32}, {1, 75, 26, 26}}},
        {{{1, 2}, {75, 80}, {26, 52}, {26, 52}}, {{1, 75, 52, 52}, {1, 75, 32, 32}, {1, 75, 26, 26}}},
};

const std::vector<InputShape> inShapes_v3_dynamic = {
        {{-1, -1, -1, -1}, {{1, 255, 52, 52}, {1, 255, 26, 26}, {1, 255, 13, 13}}},
        {{{1, 2}, {255, 256}, {13, 52}, {13, 52}}, {{1, 255, 52, 52}, {1, 255, 26, 26}, {1, 255, 13, 13}}}
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

const auto testCase_yolov3_dynamic = ::testing::Combine(
        ::testing::ValuesIn(inShapes_v3_dynamic),
        ::testing::Values(yoloV3attr),
        ::testing::Values(masks[2]),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::Values(emptyAdditionalConfig),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);

const regionYoloAttributes yoloV3mxnetAttr = {20, 4, 9, false, 1, 3};

const auto testCase_yolov3_mxnet_dynamic = ::testing::Combine(
        ::testing::ValuesIn(inShapes_mxnet_dynamic),
        ::testing::Values(yoloV3mxnetAttr),
        ::testing::Values(masks[1]),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::Values(emptyAdditionalConfig),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);

const regionYoloAttributes yoloV2caffeAttr = {20, 4, 5, true, 1, 3};

const auto testCase_yolov2_caffe_dynamic = ::testing::Combine(
        ::testing::ValuesIn(inShapes_caffe_dynamic),
        ::testing::Values(yoloV2caffeAttr),
        ::testing::Values(masks[0]),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::Values(emptyAdditionalConfig),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_GPURegionYolov3Dynamic, RegionYoloLayerGPUTest,
                         testCase_yolov3_dynamic,
                         RegionYoloLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPURegionYoloMxnetDynamic, RegionYoloLayerGPUTest,
                         testCase_yolov3_mxnet_dynamic,
                         RegionYoloLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPURegionYoloCaffeDynamic, RegionYoloLayerGPUTest,
                         testCase_yolov2_caffe_dynamic,
                         RegionYoloLayerGPUTest::getTestCaseName);

} // namespace
} // namespace GPULayerTestsDefinitions
