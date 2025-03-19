// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/roi_pooling.hpp"

namespace ov {
namespace test {
std::string ROIPoolingLayerTest::getTestCaseName(const testing::TestParamInfo<roiPoolingParamsTuple>& obj) {
    std::vector<InputShape> input_shapes;
    ov::Shape pool_shape;
    float spatial_scale;
    ov::test::utils::ROIPoolingTypes pool_method;
    ov::element::Type model_type;
    std::string target_device;
    std::tie(input_shapes, pool_shape, spatial_scale, pool_method, model_type, target_device) = obj.param;

    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < input_shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({input_shapes[i].first})
               << (i < input_shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < input_shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < input_shapes.size(); j++) {
            result << ov::test::utils::vec2str(input_shapes[j].second[i]) << (j < input_shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "PS=" << ov::test::utils::vec2str(pool_shape) << "_";
    result << "Scale=" << spatial_scale << "_";
    switch (pool_method) {
        case utils::ROIPoolingTypes::ROI_MAX:
            result << "Max_";
            break;
        case utils::ROIPoolingTypes::ROI_BILINEAR:
            result << "Bilinear_";
            break;
    }
    result << "modelType=" << model_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void ROIPoolingLayerTest::SetUp() {
    std::vector<InputShape> input_shapes;
    ov::Shape pool_shape;
    float spatial_scale;
    ov::test::utils::ROIPoolingTypes pool_method;
    ov::element::Type model_type;
    std::string target_device;
    std::tie(input_shapes, pool_shape, spatial_scale, pool_method, model_type, targetDevice) = this->GetParam();

    abs_threshold = 0.08f;

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
}  // namespace test
}  // namespace ov
