// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/grid_sample.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/grid_sample.hpp"

namespace ov {
namespace test {
std::string GridSampleLayerTest::getTestCaseName(const testing::TestParamInfo<GridSampleParams>& obj) {
    ov::Shape data_shape;
    ov::Shape grid_shape;
    bool align_corners;
    ov::op::v9::GridSample::InterpolationMode mode;
    ov::op::v9::GridSample::PaddingMode padding_mode;
    ov::element::Type model_type;
    ov::element::Type grid_type;
    std::string target_device;

    std::tie(data_shape, grid_shape, align_corners, mode, padding_mode, model_type, grid_type, target_device) = obj.param;

    std::ostringstream result;
    result << "DS=" << ov::test::utils::vec2str(data_shape) << "_";
    result << "GS=" << ov::test::utils::vec2str(grid_shape) << "_";
    result << "align_corners=" << align_corners << "_";
    result << "Mode=" << ov::as_string(mode) << "_";
    result << "padding_mode=" << ov::as_string(padding_mode) << "_";
    result << "model_type=" << model_type.get_type_name() << "_";
    result << "grid_type=" << grid_type.get_type_name() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void GridSampleLayerTest::SetUp() {
    ov::Shape data_shape;
    ov::Shape grid_shape;
    bool align_corners;
    ov::op::v9::GridSample::InterpolationMode mode;
    ov::op::v9::GridSample::PaddingMode padding_mode;
    ov::element::Type model_type;
    ov::element::Type grid_type;

    std::tie(data_shape, grid_shape, align_corners, mode, padding_mode, model_type, grid_type, targetDevice) = this->GetParam();

    auto data = std::make_shared<ov::op::v0::Parameter>(model_type, data_shape);
    auto grid = std::make_shared<ov::op::v0::Parameter>(grid_type, grid_shape);
    auto gridSample = std::make_shared<ov::op::v9::GridSample>(
        data,
        grid,
        ov::op::v9::GridSample::Attributes(align_corners, mode, padding_mode));

    function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(gridSample),
                                                  ov::ParameterVector{data, grid},
                                                  "GridSample");

    if (model_type == ov::element::f16 && grid_type == ov::element::f32) {
        abs_threshold = 2e-2;
    }
}
}  // namespace test
}  // namespace ov
