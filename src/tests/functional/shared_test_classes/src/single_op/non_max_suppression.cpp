// Copyright (C) 2018-2024 Intel Corporation
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


void NmsLayerTest::compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {
    CompareBBoxes(expected, actual);
}


namespace {

typedef struct Rect {
    int32_t x1;
    int32_t y1;
    int32_t x2;
    int32_t y2;
} Rect;

class Box {
public:
    Box() = default;

    Box(int32_t batchId, int32_t classId, int32_t boxId, Rect rect, float score) {
        this->batchId = batchId;
        this->classId = classId;
        this->boxId = boxId;
        this->rect = rect;
        this->score = score;
    }

    int32_t batchId;
    int32_t classId;
    int32_t boxId;
    Rect rect;
    float score;
};

template <typename fromT, typename toT>
void convert(fromT* src, toT* dst, size_t size) {
    std::transform(src, src + size, dst, [](fromT el)->toT{return static_cast<toT>(el);});
}

} // namespace
/*
 * 1: selected_indices - tensor of type T_IND and shape [number of selected boxes, 3] containing information about
 * selected boxes as triplets [batch_index, class_index, box_index]. 2: selected_scores - tensor of type T_THRESHOLDS
 * and shape [number of selected boxes, 3] containing information about scores for each selected box as triplets
 *    [batch_index, class_index, box_score].
 * 3: valid_outputs - 1D tensor with 1 element of type T_IND representing the total number of selected boxes.
 */
void NmsLayerTest::CompareBBoxes(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {
    InputTypes input_types;
    InputShapeParams input_shape_params;
    int max_out_boxes_per_class;
    float iou_thr, score_thr, soft_nms_sigma;
    op::v5::NonMaxSuppression::BoxEncodingType box_encoding;
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

    size_t num_batches, num_boxes, num_classes;
    std::tie(num_batches, num_boxes, num_classes) = input_shape_params;

    auto iouFunc = [](const Box& boxI, const Box& boxJ) {
        const Rect& rectI = boxI.rect;
        const Rect& rectJ = boxJ.rect;

        float areaI = (rectI.y2 - rectI.y1) * (rectI.x2 - rectI.x1);
        float areaJ = (rectJ.y2 - rectJ.y1) * (rectJ.x2 - rectJ.x1);

        if (areaI <= 0.0f || areaJ <= 0.0f) {
            return 0.0f;
        }

        float intersection_ymin = std::max(rectI.y1, rectJ.y1);
        float intersection_xmin = std::max(rectI.x1, rectJ.x1);
        float intersection_ymax = std::min(rectI.y2, rectJ.y2);
        float intersection_xmax = std::min(rectI.x2, rectJ.x2);

        float intersection_area = std::max(intersection_ymax - intersection_ymin, 0.0f) *
                                  std::max(intersection_xmax - intersection_xmin, 0.0f);

        return intersection_area / (areaI + areaJ - intersection_area);
    };

    // Get input bboxes' coords
    std::vector<std::vector<Rect>> coordList(num_batches, std::vector<Rect>(num_boxes));
    {
        const auto& input = inputs[function->get_parameters().at(0)];
        std::vector<double> buffer(input.get_size());
        if (input.get_element_type() == ov::element::f32) {
            convert(input.data<float>(), buffer.data(), input.get_size());
        } else if (input.get_element_type() == ov::element::f16) {
            convert(input.data<float16>(), buffer.data(), input.get_size());
        } else {
            const auto ptr = input.data<double>();
            std::copy(ptr, ptr + input.get_size(), buffer.begin());
        }

        for (size_t i = 0; i < num_batches; ++i) {
            for (size_t j = 0; j < num_boxes; ++j) {
                const int32_t y1 = static_cast<int32_t>(buffer[(i * num_boxes + j) * 4 + 0]);
                const int32_t x1 = static_cast<int32_t>(buffer[(i * num_boxes + j) * 4 + 1]);
                const int32_t y2 = static_cast<int32_t>(buffer[(i * num_boxes + j) * 4 + 2]);
                const int32_t x2 = static_cast<int32_t>(buffer[(i * num_boxes + j) * 4 + 3]);

                coordList[i][j] = {std::min(y1, y2), std::min(x1, x2), std::max(y1, y2), std::max(x1, x2)};
            }
        }
    }

    auto compareBox = [](const Box& boxA, const Box& boxB) {
        return (boxA.batchId < boxB.batchId) || (boxA.batchId == boxB.batchId && boxA.classId < boxB.classId) ||
               (boxA.batchId == boxB.batchId && boxA.classId == boxB.classId && boxA.boxId < boxB.boxId);
    };

    // Get expected bboxes' index/score
    std::vector<Box> expectedList;
    {
        size_t selected_indices_size = expected[0].get_size();
        std::vector<int64_t> selected_indices_data(expected[0].get_size());
        if (expected[0].get_element_type() == ov::element::i32) {
            convert(expected[0].data<int32_t>(), selected_indices_data.data(), expected[0].get_size());
        } else {
            const auto ptr = expected[0].data<int64_t>();
            std::copy(ptr, ptr + expected[0].get_size(), selected_indices_data.begin());
        }

        std::vector<double> selected_scores_data(expected[1].get_size());
        if (expected[1].get_element_type() == ov::element::f32) {
            convert(expected[1].data<float>(), selected_scores_data.data(), expected[1].get_size());
        } else if (expected[1].get_element_type() == ov::element::f16) {
            convert(expected[1].data<float16>(), selected_scores_data.data(), expected[1].get_size());
        } else {
            const auto ptr = expected[1].data<double>();
            std::copy(ptr, ptr + expected[1].get_size(), selected_scores_data.begin());
        }

        for (size_t i = 0; i < selected_indices_size; i += 3) {
            const int32_t batchId = selected_indices_data[i + 0];
            const int32_t classId = selected_indices_data[i + 1];
            const int32_t boxId = selected_indices_data[i + 2];
            const float score = selected_scores_data[i + 2];
            if (batchId == -1 || classId == -1 || boxId == -1)
                break;

            expectedList.emplace_back(batchId, classId, boxId, coordList[batchId][boxId], score);
        }

        std::sort(expectedList.begin(), expectedList.end(), compareBox);
    }

    // Get actual bboxes' index/score
    std::vector<Box> actualList;
    {
        size_t selected_indices_size = actual[0].get_size();
        std::vector<int64_t> selected_indices_data(actual[0].get_size());
        if (actual[0].get_element_type() == ov::element::i32) {
            convert(actual[0].data<int32_t>(), selected_indices_data.data(), actual[0].get_size());
        } else {
            const auto ptr = actual[0].data<int64_t>();
            std::copy(ptr, ptr + actual[0].get_size(), selected_indices_data.begin());
        }

        std::vector<double> selected_scores_data(actual[1].get_size());
        if (actual[1].get_element_type() == ov::element::f32) {
            convert(actual[1].data<float>(), selected_scores_data.data(), actual[1].get_size());
        } else if (actual[1].get_element_type() == ov::element::f16) {
            convert(actual[1].data<float16>(), selected_scores_data.data(), actual[1].get_size());
        } else {
            const auto ptr = actual[1].data<double>();
            std::copy(ptr, ptr + actual[1].get_size(), selected_scores_data.begin());
        }


        for (size_t i = 0; i < selected_indices_size; i += 3) {
            const int32_t batchId = selected_indices_data[i + 0];
            const int32_t classId = selected_indices_data[i + 1];
            const int32_t boxId = selected_indices_data[i + 2];
            const float score = selected_scores_data[i + 2];
            if (batchId == -1 || classId == -1 || boxId == -1)
                break;

            actualList.emplace_back(batchId, classId, boxId, coordList[batchId][boxId], score);
        }
        std::sort(actualList.begin(), actualList.end(), compareBox);
    }

    std::vector<Box> intersectionList;
    std::vector<Box> differenceList;
    {
        std::list<Box> tempExpectedList(expectedList.size()), tempActualList(actualList.size());
        std::copy(expectedList.begin(), expectedList.end(), tempExpectedList.begin());
        std::copy(actualList.begin(), actualList.end(), tempActualList.begin());
        auto sameBox = [](const Box& boxA, const Box& boxB) {
            return (boxA.batchId == boxB.batchId) && (boxA.classId == boxB.classId) && (boxA.boxId == boxB.boxId);
        };

        for (auto itA = tempActualList.begin(); itA != tempActualList.end(); ++itA) {
            bool found = false;
            for (auto itB = tempExpectedList.begin(); itB != tempExpectedList.end(); ++itB) {
                if (sameBox(*itA, *itB)) {
                    intersectionList.emplace_back(*itB);
                    tempExpectedList.erase(itB);
                    found = true;
                    break;
                }
            }

            if (!found) {
                differenceList.emplace_back(*itA);
            }
        }
        differenceList.insert(differenceList.end(), tempExpectedList.begin(), tempExpectedList.end());

        for (auto& item : differenceList) {
            if ((item.rect.x1 == item.rect.x2) || (item.rect.y1 == item.rect.y2))
                continue;

            float maxIou = 0.f;
            for (auto& refItem : intersectionList) {
                maxIou = std::max(maxIou, iouFunc(item, refItem));

                if (maxIou > 0.3f)
                    break;
            }

            ASSERT_TRUE(maxIou > 0.3f) << "MaxIOU: " << maxIou << ", expectedList.size(): " << expectedList.size()
                                       << ", actualList.size(): " << actualList.size()
                                       << ", intersectionList.size(): " << intersectionList.size()
                                       << ", diffList.size(): " << differenceList.size()
                                       << ", batchId: " << item.batchId << ", classId: " << item.classId
                                       << ", boxId: " << item.boxId << ", score: " << item.score
                                       << ", coord: " << item.rect.x1 << ", " << item.rect.y1 << ", " << item.rect.x2
                                       << ", " << item.rect.y2;
        }
    }
}


}  // namespace test
}  // namespace ov
