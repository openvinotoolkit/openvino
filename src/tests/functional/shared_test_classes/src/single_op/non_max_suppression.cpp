// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/non_max_suppression.hpp"

#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/multiply.hpp"

namespace ov {
namespace test {

std::string NmsLayerTest::getTestCaseName(const testing::TestParamInfo<NmsParams>& obj) {
    InputShapeParams input_shape_params;
    InputTypes input_types;
    int32_t max_out_boxes_per_class;
    float iou_thr, score_thr, soft_nms_sigma;
    op::v5::NonMaxSuppression::BoxEncodingType box_encoding;
    bool sort_res_descend;
    element::Type out_type;
    std::string target_device;
    std::tie(input_shape_params,
             input_types,
             max_out_boxes_per_class,
             iou_thr,
             score_thr,
             soft_nms_sigma,
             box_encoding,
             sort_res_descend,
             out_type,
             target_device) = obj.param;

    size_t num_batches, num_boxes, num_classes;
    std::tie(num_batches, num_boxes, num_classes) = input_shape_params;

    ov::element::Type params_type, max_box_type, thr_type;
    std::tie(params_type, max_box_type, thr_type) = input_types;

    using ov::operator<<;
    std::ostringstream result;
    result << "num_batches=" << num_batches << "_num_boxes=" << num_boxes << "_num_classes=" << num_classes << "_";
    result << "params_type=" << params_type << "_max_box_type=" << max_box_type << "_thr_type=" << thr_type << "_";
    result << "max_out_boxes_per_class=" << max_out_boxes_per_class << "_";
    result << "iou_thr=" << iou_thr << "_score_thr=" << score_thr << "_soft_nms_sigma=" << soft_nms_sigma << "_";
    result << "boxEncoding=" << box_encoding << "_sort_res_descend=" << sort_res_descend << "_out_type=" << out_type << "_";
    result << "TargetDevice=" << target_device;
    return result.str();
}

void NmsLayerTest::SetUp() {
    InputTypes input_types;
    InputShapeParams input_shape_params;
    int max_out_boxes_per_class;
    float iou_thr, score_thr, soft_nms_sigma;
    op::v5::NonMaxSuppression::BoxEncodingType box_encoding;
    bool sort_res_descend;
    element::Type out_type;
    std::tie(input_shape_params,
             input_types,
             max_out_boxes_per_class,
             iou_thr,
             score_thr,
             soft_nms_sigma,
             box_encoding,
             sort_res_descend,
             out_type,
             targetDevice) = this->GetParam();

    size_t num_batches, num_boxes, num_classes;
    std::tie(num_batches, num_boxes, num_classes) = input_shape_params;

    ov::element::Type params_type, max_box_type, thr_type;
    std::tie(params_type, max_box_type, thr_type) = input_types;

    const ov::Shape boxes_shape{num_batches, num_boxes, 4}, scores_shape{num_batches, num_classes, num_boxes};

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(params_type, ov::Shape(boxes_shape)),
                                std::make_shared<ov::op::v0::Parameter>(params_type, ov::Shape(scores_shape))};

    auto max_out_boxes_per_class_node = std::make_shared<ov::op::v0::Constant>(max_box_type, ov::Shape{}, std::vector<int>{max_out_boxes_per_class});
    auto iou_thr_node = std::make_shared<ov::op::v0::Constant>(thr_type, ov::Shape{}, std::vector<float>{iou_thr});
    auto score_thr_node = std::make_shared<ov::op::v0::Constant>(thr_type, ov::Shape{}, std::vector<float>{score_thr});
    auto soft_nms_sigma_node = std::make_shared<ov::op::v0::Constant>(thr_type, ov::Shape{}, std::vector<float>{soft_nms_sigma});

    auto nms = std::make_shared<ov::op::v5::NonMaxSuppression>(params[0],
                                                               params[1],
                                                               max_out_boxes_per_class_node,
                                                               iou_thr_node,
                                                               score_thr_node,
                                                               soft_nms_sigma_node,
                                                               box_encoding,
                                                               sort_res_descend,
                                                               out_type);

    function = std::make_shared<ov::Model>(nms, params, "NMS");
}

void Nms9LayerTest::SetUp() {
    InputTypes input_types;
    InputShapeParams input_shape_params;
    int max_out_boxes_per_class;
    float iou_thr, score_thr, soft_nms_sigma;
    op::v5::NonMaxSuppression::BoxEncodingType box_encoding;
    op::v9::NonMaxSuppression::BoxEncodingType box_encoding_v9;
    bool sort_res_descend;
    ov::element::Type out_type;
    std::tie(input_shape_params,
             input_types,
             max_out_boxes_per_class,
             iou_thr,
             score_thr,
             soft_nms_sigma,
             box_encoding,
             sort_res_descend,
             out_type,
             targetDevice) = this->GetParam();

    box_encoding_v9 = box_encoding == op::v5::NonMaxSuppression::BoxEncodingType::CENTER ?
                      op::v9::NonMaxSuppression::BoxEncodingType::CENTER :
                      op::v9::NonMaxSuppression::BoxEncodingType::CORNER;

    size_t num_batches, num_boxes, num_classes;
    std::tie(num_batches, num_boxes, num_classes) = input_shape_params;

    ov::element::Type params_type, max_box_type, thr_type;
    std::tie(params_type, max_box_type, thr_type) = input_types;

    const std::vector<size_t> boxes_shape{num_batches, num_boxes, 4}, scores_shape{num_batches, num_classes, num_boxes};

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(params_type, ov::Shape(boxes_shape)),
                                std::make_shared<ov::op::v0::Parameter>(params_type, ov::Shape(scores_shape))};

    auto max_out_boxes_per_class_node = std::make_shared<ov::op::v0::Constant>(max_box_type, ov::Shape{}, std::vector<int>{max_out_boxes_per_class});
    auto iou_thr_node = std::make_shared<ov::op::v0::Constant>(thr_type, ov::Shape{}, std::vector<float>{iou_thr});
    auto score_thr_node = std::make_shared<ov::op::v0::Constant>(thr_type, ov::Shape{}, std::vector<float>{score_thr});
    auto soft_nms_sigma_node = std::make_shared<ov::op::v0::Constant>(thr_type, ov::Shape{}, std::vector<float>{soft_nms_sigma});

    auto nms = std::make_shared<ov::op::v9::NonMaxSuppression>(params[0],
                                                               params[1],
                                                               max_out_boxes_per_class_node,
                                                               iou_thr_node,
                                                               score_thr_node,
                                                               soft_nms_sigma_node,
                                                               box_encoding_v9,
                                                               sort_res_descend,
                                                               out_type);

    function = std::make_shared<ov::Model>(nms, params, "NMS");
}

}  // namespace test
}  // namespace ov
