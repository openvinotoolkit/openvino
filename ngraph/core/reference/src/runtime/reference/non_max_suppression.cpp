//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/op/non_max_suppression.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>
#include "ngraph/runtime/reference/non_max_suppression.hpp"
#include "ngraph/shape.hpp"

using namespace ngraph;
using namespace ngraph::runtime::reference;

struct Rectangle
{
    Rectangle(float y_left, float x_left, float y_right, float x_right)
        : y1{y_left}
        , x1{x_left}
        , y2{y_right}
        , x2{x_right}
    {
    }

    Rectangle() = default;

    float y1 = 0.0f;
    float x1 = 0.0f;
    float y2 = 0.f;
    float x2 = 0.0f;
};

static float intersectionOverUnion(const Rectangle& boxI, const Rectangle& boxJ)
{
    float areaI = (boxI.y2 - boxI.y1) * (boxI.x2 - boxI.x1);
    float areaJ = (boxJ.y2 - boxJ.y1) * (boxJ.x2 - boxJ.x1);

    if (areaI <= 0.0f || areaJ <= 0.0f)
    {
        return 0.0f;
    }

    float intersection_ymin = std::max(boxI.y1, boxJ.y1);
    float intersection_xmin = std::max(boxI.x1, boxJ.x1);
    float intersection_ymax = std::min(boxI.y2, boxJ.y2);
    float intersection_xmax = std::min(boxI.x2, boxJ.x2);

    float intersection_area = std::max(intersection_ymax - intersection_ymin, 0.0f) *
                              std::max(intersection_xmax - intersection_xmin, 0.0f);

    return intersection_area / (areaI + areaJ - intersection_area);
}

struct SelectedIndex
{
    SelectedIndex(int64_t batch_idx, int64_t class_idx, int64_t box_idx)
        : batch_index(batch_idx)
        , class_index(class_idx)
        , box_index(box_idx)
    {
    }

    SelectedIndex() = default;

    int64_t batch_index = 0;
    int64_t class_index = 0;
    int64_t box_index = 0;
};

struct SelectedScore
{
    SelectedScore(float batch_idx, float class_idx, float score)
        : batch_index{batch_idx}
        , class_index{class_idx}
        , box_score{score}
    {
    }

    SelectedScore() = default;

    float batch_index = 0.0f;
    float class_index = 0.0f;
    float box_score = 0.0f;
};

struct BoxInfo
{
    BoxInfo(const Rectangle& r,
            int64_t idx,
            float sc,
            int64_t suppress_idx,
            int64_t batch_idx,
            int64_t class_idx)
        : box{r}
        , index{idx}
        , suppress_begin_index{suppress_idx}
        , batch_index{batch_idx}
        , class_index{class_idx}
        , score{sc}
    {
    }

    BoxInfo() = default;

    inline bool operator<(const BoxInfo& rhs) const
    {
        return score < rhs.score || (score == rhs.score && index > rhs.index);
    }

    Rectangle box;
    int64_t index = 0;
    int64_t suppress_begin_index = 0;
    int64_t batch_index = 0;
    int64_t class_index = 0;
    float score = 0.0f;
};

using V5BoxEncoding = op::v5::NonMaxSuppression::BoxEncodingType;

namespace
{
    constexpr size_t boxes_port = 0;
    constexpr size_t scores_port = 1;
    constexpr size_t max_output_boxes_port = 2;
    constexpr size_t iou_threshold_port = 3;
    constexpr size_t score_threshold_port = 4;
    constexpr size_t soft_nms_sigma_port = 5;

    PartialShape
        infer_selected_indices_shape(const std::vector<std::shared_ptr<HostTensor>>& inputs,
                                     int64_t max_output_boxes_per_class)
    {
        const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
        const auto scores_ps = inputs[scores_port]->get_partial_shape();

        // NonMaxSuppression produces triplets
        // that have the following format: [batch_index, class_index, box_index]
        PartialShape result = {Dimension::dynamic(), 3};

        if (boxes_ps.rank().is_static() && scores_ps.rank().is_static())
        {
            const auto num_boxes_boxes = boxes_ps[1];
            if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static())
            {
                const auto num_boxes = num_boxes_boxes.get_length();
                const auto num_classes = scores_ps[1].get_length();

                result[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes *
                            scores_ps[0].get_length();
            }
        }

        return result;
    }

    void normalize_corner(float* boxes, const Shape& boxes_shape)
    {
        size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
        for (size_t i = 0; i < total_num_of_boxes; ++i)
        {
            float* current_box = boxes + 4 * i;

            float y1 = current_box[0];
            float x1 = current_box[1];
            float y2 = current_box[2];
            float x2 = current_box[3];

            float ymin = std::min(y1, y2);
            float ymax = std::max(y1, y2);
            float xmin = std::min(x1, x2);
            float xmax = std::max(x1, x2);

            current_box[0] = ymin;
            current_box[1] = xmin;
            current_box[2] = ymax;
            current_box[3] = xmax;
        }
    }

    void normalize_center(float* boxes, const Shape& boxes_shape)
    {
        size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
        for (size_t i = 0; i < total_num_of_boxes; ++i)
        {
            float* current_box = boxes + 4 * i;

            float x_center = current_box[0];
            float y_center = current_box[1];
            float width = current_box[2];
            float height = current_box[3];

            float y1 = y_center - height / 2.0;
            float x1 = x_center - width / 2.0;
            float y2 = y_center + height / 2.0;
            float x2 = x_center + width / 2.0;

            current_box[0] = y1;
            current_box[1] = x1;
            current_box[2] = y2;
            current_box[3] = x2;
        }
    }

    void normalize_box_encoding(float* boxes,
                                const Shape& boxes_shape,
                                const V5BoxEncoding box_encoding)
    {
        if (box_encoding == V5BoxEncoding::CORNER)
        {
            normalize_corner(boxes, boxes_shape);
        }
        else
        {
            normalize_center(boxes, boxes_shape);
        }
    }

    std::vector<float> get_floats(const std::shared_ptr<HostTensor>& input, const Shape& shape)
    {
        size_t input_size = shape_size(shape);
        std::vector<float> result(input_size);

        switch (input->get_element_type())
        {
        case element::Type_t::bf16:
        {
            bfloat16* p = input->get_data_ptr<bfloat16>();
            for (size_t i = 0; i < input_size; ++i)
            {
                result[i] = float(p[i]);
            }
        }
        break;
        case element::Type_t::f16:
        {
            float16* p = input->get_data_ptr<float16>();
            for (size_t i = 0; i < input_size; ++i)
            {
                result[i] = float(p[i]);
            }
        }
        break;
        case element::Type_t::f32:
        {
            float* p = input->get_data_ptr<float>();
            memcpy(result.data(), p, input_size * sizeof(float));
        }
        break;
        default: throw std::runtime_error("Unsupported data type in op NonMaxSuppression-5"); break;
        }

        return result;
    }

    std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes,
                                          const Shape& boxes_shape,
                                          const V5BoxEncoding box_encoding)
    {
        auto result = get_floats(boxes, boxes_shape);
        normalize_box_encoding(result.data(), boxes_shape, box_encoding);
        return result;
    }

    std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores,
                                           const Shape& scores_shape)
    {
        auto result = get_floats(scores, scores_shape);
        return result;
    }
}

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            InfoForNMS5 get_info_for_nms5_evaluation(const op::v5::NonMaxSuppression* nms5,
                                                     const HostTensorVector& inputs)
            {
                InfoForNMS5 result;

                result.max_output_boxes_per_class = nms5->max_boxes_output_from_input();
                result.iou_threshold = nms5->iou_threshold_from_input();
                result.score_threshold = nms5->score_threshold_from_input();
                result.soft_nms_sigma = nms5->soft_nms_sigma_from_input();

                auto selected_indices_shape =
                    infer_selected_indices_shape(inputs, result.max_output_boxes_per_class);
                result.out_shape = selected_indices_shape.to_shape();

                result.boxes_shape = inputs[boxes_port]->get_shape();
                result.scores_shape = inputs[scores_port]->get_shape();

                result.boxes_data = prepare_boxes_data(
                    inputs[boxes_port], result.boxes_shape, nms5->get_box_encoding());
                result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

                result.out_shape_size = shape_size(result.out_shape);

                result.sort_result_descending = nms5->get_sort_result_descending();

                result.output_type = nms5->get_output_type();

                return result;
            }

            void non_max_suppression(const float* boxes_data,
                                     const Shape& boxes_data_shape,
                                     const float* scores_data,
                                     const Shape& scores_data_shape,
                                     int64_t max_output_boxes_per_class,
                                     float iou_threshold,
                                     float score_threshold,
                                     float soft_nms_sigma,
                                     int64_t* selected_indices,
                                     const Shape& selected_indices_shape,
                                     float* selected_scores,
                                     const Shape& selected_scores_shape,
                                     int64_t* valid_outputs,
                                     const bool sort_result_descending)
            {
                float scale = 0.0f;
                if (soft_nms_sigma > 0.0f)
                {
                    scale = -0.5f / soft_nms_sigma;
                }

                auto func = [iou_threshold, scale](float iou) {
                    const float weight = std::exp(scale * iou * iou);
                    return iou <= iou_threshold ? weight : 0.0f;
                };

                // boxes shape: {num_batches, num_boxes, 4}
                // scores shape: {num_batches, num_classes, num_boxes}
                int64_t num_batches = static_cast<int64_t>(scores_data_shape[0]);
                int64_t num_classes = static_cast<int64_t>(scores_data_shape[1]);
                int64_t num_boxes = static_cast<int64_t>(boxes_data_shape[1]);

                SelectedIndex* selected_indices_ptr =
                    reinterpret_cast<SelectedIndex*>(selected_indices);
                SelectedScore* selected_scores_ptr =
                    reinterpret_cast<SelectedScore*>(selected_scores);

                size_t boxes_per_class = static_cast<size_t>(max_output_boxes_per_class);

                int64_t num_of_valid_boxes = 0;

                std::vector<BoxInfo> filteredBoxes;

                for (int64_t batch = 0; batch < num_batches; batch++)
                {
                    const float* boxesPtr = boxes_data + batch * num_boxes * 4;
                    Rectangle* r = reinterpret_cast<Rectangle*>(const_cast<float*>(boxesPtr));

                    for (int64_t class_idx = 0; class_idx < num_classes; class_idx++)
                    {
                        const float* scoresPtr =
                            scores_data + batch * (num_classes * num_boxes) + class_idx * num_boxes;

                        std::vector<BoxInfo> candidate_boxes;
                        candidate_boxes.reserve(num_boxes);

                        for (size_t box_idx = 0; box_idx < num_boxes; box_idx++)
                        {
                            if (scoresPtr[box_idx] > score_threshold)
                            {
                                candidate_boxes.emplace_back(
                                    r[box_idx], box_idx, scoresPtr[box_idx], 0, batch, class_idx);
                            }
                        }

                        std::priority_queue<BoxInfo> sorted_boxes(std::less<BoxInfo>(),
                                                                  std::move(candidate_boxes));

                        std::vector<BoxInfo> selected;
                        // Get the next box with top score, filter by iou_threshold

                        BoxInfo next_candidate;
                        float original_score;

                        while (!sorted_boxes.empty() && selected.size() < boxes_per_class)
                        {
                            next_candidate = sorted_boxes.top();
                            original_score = next_candidate.score;
                            sorted_boxes.pop();

                            bool should_hard_suppress = false;
                            for (int64_t j = static_cast<int64_t>(selected.size()) - 1;
                                 j >= next_candidate.suppress_begin_index;
                                 --j)
                            {
                                float iou =
                                    intersectionOverUnion(next_candidate.box, selected[j].box);
                                next_candidate.score *= func(iou);

                                if (iou >= iou_threshold)
                                {
                                    should_hard_suppress = true;
                                    break;
                                }

                                if (next_candidate.score <= score_threshold)
                                {
                                    break;
                                }
                            }

                            next_candidate.suppress_begin_index = selected.size();

                            if (!should_hard_suppress)
                            {
                                if (next_candidate.score == original_score)
                                {
                                    selected.push_back(next_candidate);
                                    continue;
                                }
                                if (next_candidate.score > score_threshold)
                                {
                                    sorted_boxes.push(next_candidate);
                                }
                            }
                        }

                        for (const auto& box_info : selected)
                        {
                            filteredBoxes.push_back(box_info);
                        }
                    }
                }

                if (sort_result_descending)
                {
                    std::sort(filteredBoxes.begin(),
                              filteredBoxes.end(),
                              [](const BoxInfo& l, const BoxInfo& r) { return l.score > r.score; });
                }

                size_t max_num_of_selected_indices = selected_indices_shape[0];
                size_t output_size = std::min(filteredBoxes.size(), max_num_of_selected_indices);

                *valid_outputs = output_size;

                size_t idx;
                for (idx = 0; idx < output_size; idx++)
                {
                    const auto& box_info = filteredBoxes[idx];
                    SelectedIndex selected_index{
                        box_info.batch_index, box_info.class_index, box_info.index};
                    SelectedScore selected_score{static_cast<float>(box_info.batch_index),
                                                 static_cast<float>(box_info.class_index),
                                                 box_info.score};

                    selected_indices_ptr[idx] = selected_index;
                    selected_scores_ptr[idx] = selected_score;
                }

                SelectedIndex selected_index_filler{0, 0, 0};
                SelectedScore selected_score_filler{0.0f, 0.0f, 0.0f};
                for (; idx < max_num_of_selected_indices; idx++)
                {
                    selected_indices_ptr[idx] = selected_index_filler;
                    selected_scores_ptr[idx] = selected_score_filler;
                }
            }

            void nms5_postprocessing(const HostTensorVector& outputs,
                                     const ngraph::element::Type output_type,
                                     const std::vector<int64_t>& selected_indices,
                                     const std::vector<float>& selected_scores,
                                     int64_t valid_outputs,
                                     const ngraph::element::Type selected_scores_type)
            {
                outputs[0]->set_element_type(output_type);
                outputs[0]->set_shape(Shape{static_cast<size_t>(valid_outputs), 3});

                size_t num_of_outputs = outputs.size();

                if (num_of_outputs >= 2)
                {
                    outputs[1]->set_element_type(selected_scores_type);
                    outputs[1]->set_shape(Shape{static_cast<size_t>(valid_outputs), 3});
                }

                if (num_of_outputs >= 3)
                {
                    outputs[2]->set_element_type(output_type);
                    outputs[2]->set_shape(Shape{1});
                }

                size_t selected_size = valid_outputs * 3;

                if (output_type == ngraph::element::i64)
                {
                    int64_t* indices_ptr = outputs[0]->get_data_ptr<int64_t>();
                    memcpy(indices_ptr, selected_indices.data(), selected_size * sizeof(int64_t));
                }
                else
                {
                    int32_t* indices_ptr = outputs[0]->get_data_ptr<int32_t>();
                    for (size_t i = 0; i < selected_size; ++i)
                    {
                        indices_ptr[i] = static_cast<int32_t>(selected_indices[i]);
                    }
                }

                if (num_of_outputs < 2)
                {
                    return;
                }

                size_t selected_scores_size = selected_scores.size();

                switch (selected_scores_type)
                {
                case element::Type_t::bf16:
                {
                    bfloat16* scores_ptr = outputs[1]->get_data_ptr<bfloat16>();
                    for (size_t i = 0; i < selected_scores_size; ++i)
                    {
                        scores_ptr[i] = bfloat16(selected_scores[i]);
                    }
                }
                break;
                case element::Type_t::f16:
                {
                    float16* scores_ptr = outputs[1]->get_data_ptr<float16>();
                    for (size_t i = 0; i < selected_scores_size; ++i)
                    {
                        scores_ptr[i] = float16(selected_scores[i]);
                    }
                }
                break;
                case element::Type_t::f32:
                {
                    float* scores_ptr = outputs[1]->get_data_ptr<float>();
                    memcpy(scores_ptr, selected_scores.data(), selected_size * sizeof(float));
                }
                break;
                default:;
                }

                if (num_of_outputs < 3)
                {
                    return;
                }

                if (output_type == ngraph::element::i64)
                {
                    int64_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int64_t>();
                    *valid_outputs_ptr = valid_outputs;
                }
                else
                {
                    int32_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int32_t>();
                    *valid_outputs_ptr = static_cast<int32_t>(valid_outputs);
                }
            }
        }
    }
}
