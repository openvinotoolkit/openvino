// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <random>

#include "shared_test_classes/single_op/roi_align.hpp"

#include "openvino/core/enum_names.hpp"

namespace ov {
namespace test {
std::string ROIAlignLayerTest::getTestCaseName(const testing::TestParamInfo<roialignParams>& obj) {
    std::vector<InputShape> input_shapes;
    ov::Shape coords_shape;
    int pooled_h;
    int pooled_w;
    float spatial_scale;
    int pooling_ratio;
    std::string pooling_mode;
    ov::element::Type model_type;
    std::string target_device;
    std::tie(input_shapes, coords_shape, pooled_h, pooled_w, spatial_scale,
    pooling_ratio, pooling_mode, model_type, target_device) = obj.param;

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
    result << "coordShape=" << ov::test::utils::vec2str(coords_shape) << "_";
    result << "pooledH=" << pooled_h << "_";
    result << "pooledW=" << pooled_w << "_";
    result << "spatialScale=" << spatial_scale << "_";
    result << "poolingRatio=" << pooling_ratio << "_";
    result << "poolingMode=" << pooling_mode << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

static int randInt(int low, int high) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(low, high);
    return dis(gen);
}

void ROIAlignLayerTest::fillCoordTensor(std::vector<float>& coords, int height, int width,
                                        float spatial_scale, int pooled_ratio, int pooled_h, int pooled_w) {
    int min_roi_width = pooled_w;
    int max_roi_width = width / pooled_ratio;
    int min_roi_height = pooled_h;
    int max_roi_height = height / pooled_ratio;

    for (int i = 0; i < coords.size() / 4; i++) {
        int size_x = std::min(width, randInt(min_roi_width, max_roi_width));
        int size_y = std::min(height, randInt(min_roi_height, max_roi_height));
        int start_x = randInt(0, std::max(1, width - size_x - 1));
        int start_y = randInt(0, std::max(1, height - size_y - 1));

        coords[i * 4] = start_x / spatial_scale;
        coords[i * 4 + 1] = start_y / spatial_scale;
        coords[i * 4 + 2] = (start_x + size_x - 1) / spatial_scale;
        coords[i * 4 + 3] = (start_y + size_y - 1) / spatial_scale;
    }
}
void ROIAlignLayerTest::fillIdxTensor(std::vector<int>& idx, int batch_size) {
    int batch_id = 0;
    for (int i = 0; i < idx.size(); i++) {
        idx[i] = batch_id;
        batch_id = (batch_id + 1) % batch_size;
    }
}

void ROIAlignLayerTest::SetUp() {
    std::vector<InputShape> input_shapes;
    ov::Shape coords_shape;
    int pooled_h;
    int pooled_w;
    float spatial_scale;
    int pooling_ratio;
    std::string pooling_mode;
    ov::element::Type model_type;
    std::tie(input_shapes, coords_shape, pooled_h, pooled_w, spatial_scale,
    pooling_ratio, pooling_mode, model_type, targetDevice) = this->GetParam();

    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]);
    std::vector<float> proposal_vector;
    std::vector<int> roi_idx_vector;
    proposal_vector.resize(coords_shape[0] * 4);
    roi_idx_vector.resize(coords_shape[0]);

    fillCoordTensor(proposal_vector, inputDynamicShapes[0][2].get_length(), inputDynamicShapes[0][3].get_length(),
                    spatial_scale, pooling_ratio, pooled_h, pooled_w);
    fillIdxTensor(roi_idx_vector, inputDynamicShapes[0][0].get_length());
    auto idx_shape = ov::Shape{coords_shape[0]};

    auto coords = std::make_shared<ov::op::v0::Constant>(model_type, coords_shape, proposal_vector.data());
    auto rois_Idx = std::make_shared<ov::op::v0::Constant>(ov::element::i32, idx_shape, roi_idx_vector.data());
    auto roi_align = std::make_shared<ov::op::v3::ROIAlign>(param,
                                                            coords,
                                                            rois_Idx,
                                                            pooled_h,
                                                            pooled_w,
                                                            pooling_ratio,
                                                            spatial_scale,
                                                            pooling_mode);
    function = std::make_shared<ov::Model>(roi_align->outputs(), ov::ParameterVector{param}, "roi_align");
}

std::string ROIAlignV9LayerTest::getTestCaseName(const testing::TestParamInfo<roialignV9Params>& obj) {
    std::vector<InputShape> input_shapes;
    ov::Shape coords_shape;
    int pooled_h;
    int pooled_w;
    float spatial_scale;
    int pooling_ratio;
    std::string pooling_mode;
    std::string roi_aligned_mode;
    ov::element::Type model_type;
    std::string target_device;
    std::tie(input_shapes, coords_shape, pooled_h, pooled_w, spatial_scale,
    pooling_ratio, pooling_mode, roi_aligned_mode, model_type, target_device) = obj.param;

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
    result << "coordShape=" << ov::test::utils::vec2str(coords_shape) << "_";
    result << "pooledH=" << pooled_h << "_";
    result << "pooledW=" << pooled_w << "_";
    result << "spatialScale=" << spatial_scale << "_";
    result << "poolingRatio=" << pooling_ratio << "_";
    result << "poolingMode=" << pooling_mode << "_";
    result << "ROIMode=" << roi_aligned_mode << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void ROIAlignV9LayerTest::SetUp() {
    std::vector<InputShape> input_shapes;
    ov::Shape coords_shape;
    int pooled_h;
    int pooled_w;
    float spatial_scale;
    int pooling_ratio;
    std::string pooling_mode;
    std::string roi_aligned_mode;
    ov::element::Type model_type;
    std::tie(input_shapes, coords_shape, pooled_h, pooled_w, spatial_scale,
    pooling_ratio, pooling_mode, roi_aligned_mode, model_type, targetDevice) = this->GetParam();

    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]);
    std::vector<float> proposal_vector;
    std::vector<int> roi_idx_vector;
    proposal_vector.resize(coords_shape[0] * 4);
    roi_idx_vector.resize(coords_shape[0]);

    ROIAlignLayerTest::fillCoordTensor(proposal_vector, inputDynamicShapes[0][2].get_length(), inputDynamicShapes[0][3].get_length(),
                                       spatial_scale, pooling_ratio, pooled_h, pooled_w);
    ROIAlignLayerTest::fillIdxTensor(roi_idx_vector, inputDynamicShapes[0][0].get_length());
    auto idx_shape = ov::Shape{coords_shape[0]};

    auto coords = std::make_shared<ov::op::v0::Constant>(model_type, coords_shape, proposal_vector.data());
    auto rois_Idx = std::make_shared<ov::op::v0::Constant>(ov::element::i32, idx_shape, roi_idx_vector.data());
    auto roi_align = std::make_shared<ov::op::v9::ROIAlign>(param,
                                                            coords,
                                                            rois_Idx,
                                                            pooled_h,
                                                            pooled_w,
                                                            pooling_ratio,
                                                            spatial_scale,
                                                            ov::EnumNames<ov::op::v9::ROIAlign::PoolingMode>::as_enum(pooling_mode),
                                                            ov::EnumNames<ov::op::v9::ROIAlign::AlignedMode>::as_enum(roi_aligned_mode));
    function = std::make_shared<ov::Model>(roi_align->outputs(), ov::ParameterVector{param}, "roi_align");
}
}  // namespace test
}  // namespace ov
