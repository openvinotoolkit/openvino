// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/roi_align_rotated.hpp"

#include <random>

#include "openvino/core/enum_names.hpp"

namespace ov {
namespace test {

static constexpr int ROI_DEF_SIZE = 5;
static constexpr int SEED = 7877;
static constexpr float PI = 3.14159265358979323846f;

struct TestParams {
    std::vector<InputShape> input_shapes;
    int num_rois;
    int pooled_h;
    int pooled_w;
    int sampliong_ratio;
    float spatial_scale;
    bool clockwise_mode;
    ov::element::Type model_type;
    std::string target_device;
};

static TestParams ExtractTestParams(const roialignrotatedParams& param) {
    TestParams tp;
    std::tie(tp.input_shapes,
             tp.num_rois,
             tp.pooled_h,
             tp.pooled_w,
             tp.sampliong_ratio,
             tp.spatial_scale,
             tp.clockwise_mode,
             tp.model_type,
             tp.target_device) = param;
    return tp;
}

static float RandomFloat(float low, float high) {
    static std::default_random_engine engine(SEED);
    std::uniform_real_distribution<float> dis(low, high);
    return dis(engine);
}

static std::vector<float> FillRoisTensor(int num_rois, int height, int width) {
    std::vector<float> rois;
    rois.resize(num_rois * ROI_DEF_SIZE);

    for (int i = 0; i < rois.size() / ROI_DEF_SIZE; i++) {
        // center_x, center_y, width, height, angle
        rois[i * ROI_DEF_SIZE + 0] = RandomFloat(0.0f, width);
        rois[i * ROI_DEF_SIZE + 1] = RandomFloat(0.0f, height);
        rois[i * ROI_DEF_SIZE + 2] = RandomFloat(0.0f, width);
        rois[i * ROI_DEF_SIZE + 3] = RandomFloat(0.0f, height);
        rois[i * ROI_DEF_SIZE + 4] = RandomFloat(0.0f, 2 * PI);
    }

    return rois;
}

static std::vector<int> FillBAtchIdxTensor(int num_rois, int batch_size) {
    std::vector<int> idx;
    idx.resize(num_rois);
    int batch_id = 0;
    for (int i = 0; i < idx.size(); i++) {
        idx[i] = batch_id;
        batch_id = (batch_id + 1) % batch_size;
    }

    return idx;
}

std::string ROIAlignRotatedLayerTest::getTestCaseName(const testing::TestParamInfo<roialignrotatedParams>& obj) {
    const TestParams tp = ExtractTestParams(obj.param);

    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < tp.input_shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({tp.input_shapes[i].first})
               << (i < tp.input_shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < tp.input_shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < tp.input_shapes.size(); j++) {
            result << ov::test::utils::vec2str(tp.input_shapes[j].second[i])
                   << (j < tp.input_shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "numRois=" << tp.num_rois << "_";
    result << "pooledH=" << tp.pooled_h << "_";
    result << "pooledW=" << tp.pooled_w << "_";
    result << "samplingRatio=" << tp.sampliong_ratio << "_";
    result << "spatialScale=" << tp.spatial_scale << "_";
    result << "clockwiseMode=" << tp.clockwise_mode << "_";
    result << "modelType=" << tp.model_type.to_string() << "_";
    result << "trgDev=" << tp.target_device;
    return result.str();
}

void ROIAlignRotatedLayerTest::SetUp() {
    const TestParams tp = ExtractTestParams(this->GetParam());
    targetDevice = tp.target_device;
    init_input_shapes(tp.input_shapes);

    const auto input_batch_size = inputDynamicShapes[0][0].get_length();
    const auto input_height = inputDynamicShapes[0][2].get_length();
    const auto input_width = inputDynamicShapes[0][3].get_length();

    auto input = std::make_shared<ov::op::v0::Parameter>(tp.model_type, inputDynamicShapes[0]);
    const auto rois_shape = ov::Shape{static_cast<size_t>(tp.num_rois), ROI_DEF_SIZE};
    const auto rois_idx_shape = ov::Shape{static_cast<size_t>(tp.num_rois)};

    auto rois = std::make_shared<ov::op::v0::Constant>(tp.model_type,
                                                       rois_shape,
                                                       FillRoisTensor(tp.num_rois, input_height, input_width).data());
    auto rois_idx = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                           rois_idx_shape,
                                                           FillBAtchIdxTensor(tp.num_rois, input_batch_size).data());
    auto roi_align = std::make_shared<ov::op::v15::ROIAlignRotated>(input,
                                                                    rois,
                                                                    rois_idx,
                                                                    tp.pooled_h,
                                                                    tp.pooled_w,
                                                                    tp.sampliong_ratio,
                                                                    tp.spatial_scale,
                                                                    tp.clockwise_mode);
    function = std::make_shared<ov::Model>(roi_align->outputs(), ov::ParameterVector{input}, "roi_align_rotated");
}

}  // namespace test
}  // namespace ov
