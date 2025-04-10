// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

struct regionYoloAttributes {
    size_t classes;
    size_t coordinates;
    size_t num_regions;
    bool do_softmax;
    int start_axis;
    int end_axis;
};

using regionYoloParamsTuple = std::tuple<InputShape,             // Input Shape
                                         regionYoloAttributes,   // Params
                                         std::vector<int64_t>,   // mask
                                         ov::test::ElementType,  // Network input precision
                                         ov::test::ElementType,  // Network output precision
                                         ov::AnyMap,             // Additional network configuration
                                         std::string>;           // Device name

class RegionYoloCPULayerTest : public testing::WithParamInterface<regionYoloParamsTuple>,
                               virtual public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<regionYoloParamsTuple> obj) {
        InputShape inputShape;
        regionYoloAttributes attributes;
        std::vector<int64_t> mask;
        ov::test::ElementType inpPrecision;
        ov::test::ElementType outPrecision;
        std::string targetName;
        ov::AnyMap additionalConfig;

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
        ov::AnyMap additionalConfig;

        std::tie(inputShape, attributes, mask, inPrc, outPrc, additionalConfig, targetDevice) = this->GetParam();

        if (inPrc == ov::test::ElementType::bf16) {
            // ticket #72342
            rel_threshold = 0.02;
        }

        init_input_shapes({ inputShape });

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        selectedType = getPrimitiveType() + "_" + ov::element::Type(inPrc).to_string();
        ov::ParameterVector paramRegionYolo;
        for (auto&& shape : inputDynamicShapes) {
            paramRegionYolo.push_back(std::make_shared<ov::op::v0::Parameter>(inPrc, shape));
        }
        const auto region_yolo = std::make_shared<ov::op::v0::RegionYolo>(paramRegionYolo[0],
                                                                          attributes.coordinates,
                                                                          attributes.classes,
                                                                          attributes.num_regions,
                                                                          attributes.do_softmax,
                                                                          mask,
                                                                          attributes.start_axis,
                                                                          attributes.end_axis);

        function = makeNgraphFunction(inPrc, paramRegionYolo, region_yolo, "RegionYolo");
    }
};

TEST_P(RegionYoloCPULayerTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "RegionYolo");
}

namespace {
const std::vector<ov::test::ElementType> inpOutPrc = {ov::test::ElementType::bf16, ov::test::ElementType::f32};

const ov::AnyMap additional_config;

/* *======================* Static Shapes *======================* */

const std::vector<ov::Shape> inShapes_caffe = {
        {1, 125, 13, 13}
};

const std::vector<ov::Shape> inShapes_mxnet = {
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

const std::vector<ov::Shape> inShapes_v3 = {
        {1, 255, 52, 52},
        {1, 255, 26, 26},
        {1, 255, 13, 13}
};

/* *======================* Dynamic Shapes *======================* */

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

const auto testCase_yolov3 = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_v3)),
        ::testing::Values(yoloV3attr),
        ::testing::Values(masks[2]),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::Values(additional_config),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto testCase_yolov3_dynamic = ::testing::Combine(
        ::testing::ValuesIn(inShapes_v3_dynamic),
        ::testing::Values(yoloV3attr),
        ::testing::Values(masks[2]),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::Values(additional_config),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const regionYoloAttributes yoloV3mxnetAttr = {20, 4, 9, false, 1, 3};

const auto testCase_yolov3_mxnet = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_mxnet)),
        ::testing::Values(yoloV3mxnetAttr),
        ::testing::Values(masks[1]),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::Values(additional_config),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto testCase_yolov3_mxnet_dynamic = ::testing::Combine(
        ::testing::ValuesIn(inShapes_mxnet_dynamic),
        ::testing::Values(yoloV3mxnetAttr),
        ::testing::Values(masks[1]),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::Values(additional_config),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const regionYoloAttributes yoloV2caffeAttr = {20, 4, 5, true, 1, 3};

const auto testCase_yolov2_caffe = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_caffe)),
        ::testing::Values(yoloV2caffeAttr),
        ::testing::Values(masks[0]),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::Values(additional_config),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto testCase_yolov2_caffe_dynamic = ::testing::Combine(
        ::testing::ValuesIn(inShapes_caffe_dynamic),
        ::testing::Values(yoloV2caffeAttr),
        ::testing::Values(masks[0]),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::ValuesIn(inpOutPrc),
        ::testing::Values(additional_config),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsRegionYolov3CPUStatic, RegionYoloCPULayerTest, testCase_yolov3, RegionYoloCPULayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsRegionYolov3CPUDynamic, RegionYoloCPULayerTest, testCase_yolov3_dynamic, RegionYoloCPULayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsRegionYoloMxnetCPUStatic, RegionYoloCPULayerTest, testCase_yolov3_mxnet, RegionYoloCPULayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsRegionYoloMxnetCPUDynamic, RegionYoloCPULayerTest, testCase_yolov3_mxnet_dynamic, RegionYoloCPULayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsRegionYoloCaffeCPUStatic, RegionYoloCPULayerTest, testCase_yolov2_caffe, RegionYoloCPULayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsRegionYoloCaffeCPUDynamic, RegionYoloCPULayerTest, testCase_yolov2_caffe_dynamic, RegionYoloCPULayerTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
