// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/non_max_suppression.hpp"

#include "openvino/op/multiply.hpp"
#include "openvino/op/non_max_suppression.hpp"

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
    result << "boxEncoding=" << box_encoding << "_sort_res_descend=" << sort_res_descend << "_out_type=" << out_type
           << "_";
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

    size_t num_classes;
    std::tie(m_num_batches, m_num_boxes, num_classes) = input_shape_params;

    ov::element::Type params_type, max_box_type, thr_type;
    std::tie(params_type, max_box_type, thr_type) = input_types;

    const ov::Shape boxes_shape{m_num_batches, m_num_boxes, 4}, scores_shape{m_num_batches, num_classes, m_num_boxes};

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(params_type, ov::Shape(boxes_shape)),
                               std::make_shared<ov::op::v0::Parameter>(params_type, ov::Shape(scores_shape))};

    auto max_out_boxes_per_class_node =
        std::make_shared<ov::op::v0::Constant>(max_box_type, ov::Shape{}, std::vector<int>{max_out_boxes_per_class});
    auto iou_thr_node = std::make_shared<ov::op::v0::Constant>(thr_type, ov::Shape{}, std::vector<float>{iou_thr});
    auto score_thr_node = std::make_shared<ov::op::v0::Constant>(thr_type, ov::Shape{}, std::vector<float>{score_thr});
    auto soft_nms_sigma_node =
        std::make_shared<ov::op::v0::Constant>(thr_type, ov::Shape{}, std::vector<float>{soft_nms_sigma});

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

    box_encoding_v9 = box_encoding == op::v5::NonMaxSuppression::BoxEncodingType::CENTER
                          ? op::v9::NonMaxSuppression::BoxEncodingType::CENTER
                          : op::v9::NonMaxSuppression::BoxEncodingType::CORNER;

    size_t num_classes;
    std::tie(m_num_batches, m_num_boxes, num_classes) = input_shape_params;

    ov::element::Type params_type, max_box_type, thr_type;
    std::tie(params_type, max_box_type, thr_type) = input_types;

    const std::vector<size_t> boxes_shape{m_num_batches, m_num_boxes, 4}, scores_shape{m_num_batches, num_classes, m_num_boxes};

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(params_type, ov::Shape(boxes_shape)),
                               std::make_shared<ov::op::v0::Parameter>(params_type, ov::Shape(scores_shape))};

    auto max_out_boxes_per_class_node =
        std::make_shared<ov::op::v0::Constant>(max_box_type, ov::Shape{}, std::vector<int>{max_out_boxes_per_class});
    auto iou_thr_node = std::make_shared<ov::op::v0::Constant>(thr_type, ov::Shape{}, std::vector<float>{iou_thr});
    auto score_thr_node = std::make_shared<ov::op::v0::Constant>(thr_type, ov::Shape{}, std::vector<float>{score_thr});
    auto soft_nms_sigma_node =
        std::make_shared<ov::op::v0::Constant>(thr_type, ov::Shape{}, std::vector<float>{soft_nms_sigma});

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
    compare_b_boxes(expected, actual, inputs[function->get_parameters().at(0)], m_num_batches, m_num_boxes);
}

namespace {

typedef struct Rect {
    int32_t x1;
    int32_t y1;
    int32_t x2;
    int32_t y2;
} Rect;

struct Box {
public:
    int32_t batch_id;
    int32_t class_id;
    int32_t box_id;
    Rect rect;
    float score;
};

template <typename fromT, typename toT>
void convert(fromT* src, toT* dst, size_t size) {
    std::transform(src, src + size, dst, [](fromT el) -> toT {
        return static_cast<toT>(el);
    });
}

}  // namespace

// 1: selected_indices - tensor of type T_IND and shape [number of selected boxes, 3] containing information about
// selected boxes as triplets [batch_index, class_index, box_index]. 2: selected_scores - tensor of type T_THRESHOLDS
// and shape [number of selected boxes, 3] containing information about scores for each selected box as triplets
//    [batch_index, class_index, box_score].
// 3: valid_outputs - 1D tensor with 1 element of type T_IND representing the total number of selected boxes.
void compare_b_boxes(const std::vector<ov::Tensor>& expected,
                   const std::vector<ov::Tensor>& actual,
                   const ov::Tensor& input,
                   size_t num_batches,
                   size_t num_boxes) {
    auto iou_func = [](const Box& box_i, const Box& box_j) {
        const Rect& rect_i = box_i.rect;
        const Rect& rect_j = box_j.rect;

        float area_i = (rect_i.y2 - rect_i.y1) * (rect_i.x2 - rect_i.x1);
        float area_j = (rect_j.y2 - rect_j.y1) * (rect_j.x2 - rect_j.x1);

        if (area_i <= 0.0f || area_j <= 0.0f) {
            return 0.0f;
        }

        float intersection_ymin = std::max(rect_i.y1, rect_j.y1);
        float intersection_xmin = std::max(rect_i.x1, rect_j.x1);
        float intersection_ymax = std::min(rect_i.y2, rect_j.y2);
        float intersection_xmax = std::min(rect_i.x2, rect_j.x2);

        float intersection_area = std::max(intersection_ymax - intersection_ymin, 0.0f) *
                                  std::max(intersection_xmax - intersection_xmin, 0.0f);

        return intersection_area / (area_i + area_j - intersection_area);
    };

    // Get input bboxes' coords
    std::vector<std::vector<Rect>> coord_list(num_batches, std::vector<Rect>(num_boxes));
    {
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

                coord_list[i][j] = {std::min(y1, y2), std::min(x1, x2), std::max(y1, y2), std::max(x1, x2)};
            }
        }
    }

    auto compare_box = [](const Box& box_a, const Box& box_b) {
        return (box_a.batch_id < box_b.batch_id) ||
               (box_a.batch_id == box_b.batch_id && box_a.class_id < box_b.class_id) ||
               (box_a.batch_id == box_b.batch_id && box_a.class_id == box_b.class_id && box_a.box_id < box_b.box_id);
    };

    // Get expected bboxes' index/score
    std::vector<Box> expected_list;
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
            const int32_t batch_id = selected_indices_data[i + 0];
            const int32_t class_id = selected_indices_data[i + 1];
            const int32_t box_id = selected_indices_data[i + 2];
            const float score = selected_scores_data[i + 2];
            if (batch_id == -1 || class_id == -1 || box_id == -1)
                break;

            expected_list.push_back({batch_id, class_id, box_id, coord_list[batch_id][box_id], score});
        }

        std::sort(expected_list.begin(), expected_list.end(), compare_box);
    }

    // Get actual bboxes' index/score
    std::vector<Box> actual_list;
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
            const int32_t batch_id = selected_indices_data[i + 0];
            const int32_t class_id = selected_indices_data[i + 1];
            const int32_t box_id = selected_indices_data[i + 2];
            const float score = selected_scores_data[i + 2];
            if (batch_id == -1 || class_id == -1 || box_id == -1)
                break;

            actual_list.push_back({batch_id, class_id, box_id, coord_list[batch_id][box_id], score});
        }
        std::sort(actual_list.begin(), actual_list.end(), compare_box);
    }

    std::vector<Box> intersection_list;
    std::vector<Box> difference_list;
    {
        std::list<Box> tempexpected_list(expected_list.size()), tempActualList(actual_list.size());
        std::copy(expected_list.begin(), expected_list.end(), tempexpected_list.begin());
        std::copy(actual_list.begin(), actual_list.end(), tempActualList.begin());
        auto same_box = [](const Box& box_a, const Box& box_b) {
            return (box_a.batch_id == box_b.batch_id) && (box_a.class_id == box_b.class_id) &&
                   (box_a.box_id == box_b.box_id);
        };

        for (auto itA = tempActualList.begin(); itA != tempActualList.end(); ++itA) {
            bool found = false;
            for (auto itB = tempexpected_list.begin(); itB != tempexpected_list.end(); ++itB) {
                if (same_box(*itA, *itB)) {
                    intersection_list.emplace_back(*itB);
                    tempexpected_list.erase(itB);
                    found = true;
                    break;
                }
            }

            if (!found) {
                difference_list.emplace_back(*itA);
            }
        }
        difference_list.insert(difference_list.end(), tempexpected_list.begin(), tempexpected_list.end());

        for (auto& item : difference_list) {
            if ((item.rect.x1 == item.rect.x2) || (item.rect.y1 == item.rect.y2))
                continue;

            float max_iou = 0.f;
            std::ignore = std::find_if(intersection_list.begin(), intersection_list.end(), [&](const Box& ref_item) {
                max_iou = std::max(max_iou, iou_func(item, ref_item));
                return max_iou > 0.3f;
            });

            ASSERT_TRUE(max_iou > 0.3f) << "MaxIOU: " << max_iou << ", expected_list.size(): " << expected_list.size()
                                        << ", actual_list.size(): " << actual_list.size()
                                        << ", intersection_list.size(): " << intersection_list.size()
                                        << ", diffList.size(): " << difference_list.size()
                                        << ", batch_id: " << item.batch_id << ", class_id: " << item.class_id
                                        << ", box_id: " << item.box_id << ", score: " << item.score
                                        << ", coord: " << item.rect.x1 << ", " << item.rect.y1 << ", " << item.rect.x2
                                        << ", " << item.rect.y2;
        }
    }
}

}  // namespace test
}  // namespace ov
