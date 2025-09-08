// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/roi_pooling.hpp"
#include "openvino/op/roi_pooling.hpp"

namespace ov {
namespace test {
class ROIPoolingLayerTestGPU : virtual public ov::test::ROIPoolingLayerTest {
protected:
    void SetUp() override;
};

void ROIPoolingLayerTestGPU::SetUp() {
    std::string target_device;
    const auto& [input_shapes, pool_shape, spatial_scale, pool_method, model_type, _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;

    abs_threshold = 0.08f;
    if (model_type == ov::element::f16) {
        abs_threshold = 0.15f;
    }

    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]);
    auto coord_param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]);
    std::string pool_method_str;
    if (pool_method == ov::test::utils::ROIPoolingTypes::ROI_MAX) {
        pool_method_str = "max";
    } else if (pool_method == ov::test::utils::ROIPoolingTypes::ROI_BILINEAR) {
        pool_method_str = "bilinear";
    } else {
        FAIL() << "Incorrect type of ROIPooling operation";
    }
    auto roi_pooling = std::make_shared<ov::op::v0::ROIPooling>(param, coord_param, pool_shape, spatial_scale, pool_method_str);
    function = std::make_shared<ov::Model>(roi_pooling->outputs(), ov::ParameterVector{param, coord_param}, "roi_pooling");
}

TEST_P(ROIPoolingLayerTestGPU, Inference) {
    run();
}

namespace {

const std::vector<ov::Shape> inShapes = {
    {{1, 3, 8, 8}},
    {{3, 4, 50, 50}},
};

const std::vector<ov::Shape> coordShapes = {
    {{1, 5}},
    {{3, 5}},
    {{5, 5}},
};

auto input_shapes = [](const std::vector<ov::Shape>& in1, const std::vector<ov::Shape>& in2) {
    std::vector<std::vector<ov::test::InputShape>> res;
    for (const auto& sh1 : in1)
        for (const auto& sh2 : in2)
            res.push_back(ov::test::static_shapes_to_test_representation({sh1, sh2}));
    return res;
}(inShapes, coordShapes);

const std::vector<ov::Shape> pooledShapes_max = {
    {{1, 1}},
    {{2, 2}},
    {{3, 3}},
    {{6, 6}},
};

const std::vector<ov::Shape> pooledShapes_bilinear = {
    {{1, 1}},
    {{2, 2}},
    {{3, 3}},
    {{6, 6}},
};

const std::vector<ov::element::Type> netPRCs = {
    ov::element::f16,
    ov::element::f32,
};

const std::vector<float> spatial_scales = {0.625f, 1.f};

const auto params_max = testing::Combine(testing::ValuesIn(input_shapes),
                                         testing::ValuesIn(pooledShapes_max),
                                         testing::ValuesIn(spatial_scales),
                                         testing::Values(ov::test::utils::ROIPoolingTypes::ROI_MAX),
                                         testing::ValuesIn(netPRCs),
                                         testing::Values(ov::test::utils::DEVICE_GPU));

const auto params_bilinear = testing::Combine(testing::ValuesIn(input_shapes),
                                              testing::ValuesIn(pooledShapes_bilinear),
                                              testing::Values(spatial_scales[1]),
                                              testing::Values(ov::test::utils::ROIPoolingTypes::ROI_BILINEAR),
                                              testing::ValuesIn(netPRCs),
                                              testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_ROIPooling_max,
                         ROIPoolingLayerTestGPU,
                         params_max,
                         ROIPoolingLayerTestGPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ROIPooling_bilinear,
                         ROIPoolingLayerTestGPU,
                         params_bilinear,
                         ROIPoolingLayerTestGPU::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
