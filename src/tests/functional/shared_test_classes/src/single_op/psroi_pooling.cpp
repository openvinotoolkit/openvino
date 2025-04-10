// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/psroi_pooling.hpp"

namespace ov {
namespace test {
std::string PSROIPoolingLayerTest::getTestCaseName(const testing::TestParamInfo<psroiParams>& obj) {
    std::vector<size_t> input_shape;
    std::vector<size_t> coords_shape;
    size_t output_dim;
    size_t group_size;
    float spatial_scale;
    size_t spatial_bins_x;
    size_t spatial_bins_y;
    std::string mode;
    ov::element::Type model_type;
    std::string target_device;
    std::tie(input_shape, coords_shape, output_dim, group_size, spatial_scale, spatial_bins_x, spatial_bins_y, mode, model_type, target_device) = obj.param;

    std::ostringstream result;

    result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "coord_shape=" << ov::test::utils::vec2str(coords_shape) << "_";
    result << "out_dim=" << output_dim << "_";
    result << "group_size=" << group_size << "_";
    result << "scale=" << spatial_scale << "_";
    result << "bins_x=" << spatial_bins_x << "_";
    result << "bins_y=" << spatial_bins_y << "_";
    result << "mode=" << mode << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void PSROIPoolingLayerTest::SetUp() {
    std::vector<size_t> input_shape;
    std::vector<size_t> coords_shape;
    size_t output_dim;
    size_t group_size;
    float spatial_scale;
    size_t spatial_bins_x;
    size_t spatial_bins_y;
    std::string mode;
    ov::element::Type model_type;
    std::tie(input_shape, coords_shape, output_dim, group_size, spatial_scale,
             spatial_bins_x, spatial_bins_y, mode, model_type, targetDevice) = this->GetParam();

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shape)),
                                std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(coords_shape))};
    auto psroi_pooling = std::make_shared<ov::op::v0::PSROIPooling>(params[0],
                                                                    params[1],
                                                                    output_dim,
                                                                    group_size,
                                                                    spatial_scale,
                                                                    spatial_bins_x,
                                                                    spatial_bins_y,
                                                                    mode);
    function = std::make_shared<ov::Model>(psroi_pooling->outputs(), params, "psroi_pooling");

    if (model_type == ov::element::f16) {
        abs_threshold = 8e-3;
    }
}
}  // namespace test
}  // namespace ov
