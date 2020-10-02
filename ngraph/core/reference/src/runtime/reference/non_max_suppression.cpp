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
    BoxInfo(const Rectangle& r, int64_t idx, float sc, int64_t suppress_idx)
        : box{r}
        , index{idx}
        , suppress_begin_index{suppress_idx}
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
    float score = 0.0f;
};

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
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
                    scale = - 0.5f / soft_nms_sigma;
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
                                    r[box_idx], box_idx, scoresPtr[box_idx], 0);
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
                            SelectedIndex selected_index{batch, class_idx, box_info.index};
                            SelectedScore selected_score{static_cast<float>(batch),
                                                         static_cast<float>(class_idx),
                                                         box_info.score};

                            selected_indices_ptr[num_of_valid_boxes] = selected_index;
                            selected_scores_ptr[num_of_valid_boxes] = selected_score;

                            num_of_valid_boxes++;
                        }
                    }
                }


                if (sort_result_descending)
                {
                }
                else
                {
                }

            }
        }
    }
}
