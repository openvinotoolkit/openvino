// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/region_yolo.hpp"

namespace {
using ov::test::InputShape;

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
        ov::element::Type,                  // Model type
        std::string                         // Device name
> RegionYoloGPUTestParam;

class RegionYoloLayerGPUTest : public testing::WithParamInterface<RegionYoloGPUTestParam>,
                               virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RegionYoloGPUTestParam> obj) {
        const auto& [shapes, attributes, mask, model_type, targetName] = obj.param;

        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        for (const auto& item : shapes.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
        result << "classes=" << attributes.classes << "_";
        result << "coords=" << attributes.coordinates << "_";
        result << "num=" << attributes.num_regions << "_";
        result << "doSoftmax=" << attributes.do_softmax << "_";
        result << "axis=" << attributes.start_axis << "_";
        result << "endAxis=" << attributes.end_axis << "_";
        result << "inpPRC=" << model_type << "_";
        result << "targetDevice=" << targetName << "_";
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [shapes, attributes, mask, model_type, _targetDevice] = this->GetParam();
        targetDevice = _targetDevice;

        init_input_shapes({ shapes });

        ov::ParameterVector paramRegionYolo;
        for (auto&& shape : inputDynamicShapes) {
            paramRegionYolo.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));
        }

        const auto region_yolo = std::make_shared<ov::op::v0::RegionYolo>(paramRegionYolo[0],
                                                                              attributes.coordinates, attributes.classes, attributes.num_regions,
                                                                              attributes.do_softmax, mask, attributes.start_axis, attributes.end_axis);

        ov::ResultVector results;
        for (size_t i = 0; i < region_yolo->get_output_size(); i++)
            results.push_back(std::make_shared<ov::op::v0::Result>(region_yolo->output(i)));
        function = std::make_shared<ov::Model>(results, paramRegionYolo, "RegionYolo");
    }
};

TEST_P(RegionYoloLayerGPUTest, Inference) {
    run();
}

const std::vector<ov::element::Type> model_types = {ov::element::f16, ov::element::f32};

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
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);

const regionYoloAttributes yoloV3mxnetAttr = {20, 4, 9, false, 1, 3};

const auto testCase_yolov3_mxnet_dynamic = ::testing::Combine(
        ::testing::ValuesIn(inShapes_mxnet_dynamic),
        ::testing::Values(yoloV3mxnetAttr),
        ::testing::Values(masks[1]),
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);

const regionYoloAttributes yoloV2caffeAttr = {20, 4, 5, true, 1, 3};

const auto testCase_yolov2_caffe_dynamic = ::testing::Combine(
        ::testing::ValuesIn(inShapes_caffe_dynamic),
        ::testing::Values(yoloV2caffeAttr),
        ::testing::Values(masks[0]),
        ::testing::ValuesIn(model_types),
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
