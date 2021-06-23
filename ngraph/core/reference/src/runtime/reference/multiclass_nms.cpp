// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/multiclass_nms.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <vector>
#include "ngraph/runtime/reference/multiclass_nms.hpp"
#include "ngraph/shape.hpp"

using namespace ngraph;
using namespace ngraph::runtime::reference;

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace multiclass_nms_v8
            {
                struct Rectangle
                {
                    Rectangle(float x_left, float y_left, float x_right, float y_right)
                        : x1{x_left}
                        , y1{y_left}
                        , x2{x_right}
                        , y2{y_right}
                    {
                    }

                    Rectangle() = default;

                    float x1 = 0.0f;
                    float y1 = 0.0f;
                    float x2 = 0.0f;
                    float y2 = 0.0f;
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

                    float intersection_area =
                        std::max(intersection_ymax - intersection_ymin, 0.0f) *
                        std::max(intersection_xmax - intersection_xmin, 0.0f);

                    return intersection_area / (areaI + areaJ - intersection_area);
                }

                struct SelectedIndex
                {
                    SelectedIndex(int64_t batch_idx, int64_t box_idx, int64_t num_box)
                        : flattened_index(batch_idx * num_box + box_idx)
                    {
                    }

                    SelectedIndex() = default;

                    int64_t flattened_index = 0;
                };

                struct SelectedOutput
                {
                    SelectedOutput(
                        float class_idx, float score, float x1, float y1, float x2, float y2)
                        : class_index{class_idx}
                        , box_score{score}
                        , xmin{x1}
                        , ymin{y1}
                        , xmax{x2}
                        , ymax{y2}
                    {
                    }

                    SelectedOutput() = default;

                    float class_index = 0.0f;
                    float box_score = 0.0f;
                    float xmin, ymin, xmax, ymax;
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

                    inline bool operator>(const BoxInfo& rhs) const
                    {
                        return !(score < rhs.score || (score == rhs.score && index > rhs.index));
                    }

                    Rectangle box;
                    int64_t index = 0;
                    int64_t suppress_begin_index = 0;
                    int64_t batch_index = 0;
                    int64_t class_index = 0;
                    float score = 0.0f;
                };

                inline std::ostream& operator<<(std::ostream& s, const Rectangle& b)
                {
                    s << "Rectangle{";
                    s << b.x1 << ", ";
                    s << b.y1 << ", ";
                    s << b.x2 << ", ";
                    s << b.y2;
                    s << "}";
                    return s;
                }

                inline std::ostream& operator<<(std::ostream& s, const BoxInfo& b)
                {
                    s << "BoxInfo{";
                    s << b.batch_index << ", ";
                    s << b.class_index << ", ";
                    s << b.index << ", ";
                    s << b.box << ", ";
                    s << b.score;
                    s << "}";
                    return s;
                }
            } // namespace multiclass_nms_v8

            void multiclass_nms(const float* boxes_data,
                                const Shape& boxes_data_shape,
                                const float* scores_data,
                                const Shape& scores_data_shape,
                                op::util::NmsBase::SortResultType sort_result_type,
                                bool sort_result_across_batch,
                                float iou_threshold,
                                float score_threshold,
                                int nms_top_k,
                                int keep_top_k,
                                int background_class,
                                float nms_eta,
                                bool normalized,
                                float* selected_outputs,
                                const Shape& selected_outputs_shape,
                                int64_t* selected_indices,
                                const Shape& selected_indices_shape,
                                int64_t* valid_outputs)
            {
                using SelectedIndex = multiclass_nms_v8::SelectedIndex;
                using SelectedOutput = multiclass_nms_v8::SelectedOutput;
                using BoxInfo = multiclass_nms_v8::BoxInfo;
                using Rectangle = multiclass_nms_v8::Rectangle;

                auto func = [](float iou, float adaptive_threshold) {
                    return iou <= adaptive_threshold ? 1.0f : 0.0f;
                };

                // boxes shape: {num_batches, num_boxes, 4}
                // scores shape: {num_batches, num_classes, num_boxes}
                int64_t num_batches = static_cast<int64_t>(scores_data_shape[0]);
                int64_t num_classes = static_cast<int64_t>(scores_data_shape[1]);
                int64_t num_boxes = static_cast<int64_t>(boxes_data_shape[1]);

                SelectedIndex* selected_indices_ptr =
                    reinterpret_cast<SelectedIndex*>(selected_indices);
                SelectedOutput* selected_scores_ptr =
                    reinterpret_cast<SelectedOutput*>(selected_outputs);

                std::vector<BoxInfo> filteredBoxes; // container for the whole batch

                for (int64_t batch = 0; batch < num_batches; batch++)
                {
                    const float* boxesPtr = boxes_data + batch * num_boxes * 4;
                    Rectangle* r = reinterpret_cast<Rectangle*>(const_cast<float*>(boxesPtr));

                    int64_t num_dets = 0;
                    std::vector<BoxInfo> selected_boxes; // container for a batch element

                    for (int64_t class_idx = 0; class_idx < num_classes; class_idx++)
                    {
                        if (class_idx == background_class)
                            continue;

                        auto adaptive_threshold = iou_threshold;

                        const float* scoresPtr =
                            scores_data + batch * (num_classes * num_boxes) + class_idx * num_boxes;

                        std::vector<BoxInfo> candidate_boxes;

                        for (int64_t box_idx = 0; box_idx < num_boxes; box_idx++)
                        {
                            if (scoresPtr[box_idx] >=
                                score_threshold) /* NOTE: ">=" instead of ">" used in PDPD */
                            {
                                candidate_boxes.emplace_back(
                                    r[box_idx], box_idx, scoresPtr[box_idx], 0, batch, class_idx);
                            }
                        }

                        int candiate_size = candidate_boxes.size();

                        // threshold nms_top_k for each class
                        // NOTE: "nms_top_k" in PDPD not exactly equal to
                        // "max_output_boxes_per_class" in ONNX.
                        if (nms_top_k > -1 && nms_top_k < candiate_size)
                        {
                            candiate_size = nms_top_k;
                        }

                        if (candiate_size <= 0) // early drop
                        {
                            continue;
                        }

                        // sort by score
                        std::partial_sort(candidate_boxes.begin(),
                                          candidate_boxes.begin() + candiate_size,
                                          candidate_boxes.end(),
                                          std::greater<BoxInfo>());

                        std::priority_queue<BoxInfo> sorted_boxes(candidate_boxes.begin(),
                                                                  candidate_boxes.begin() +
                                                                      candiate_size,
                                                                  std::less<BoxInfo>());

                        std::vector<BoxInfo> selected; // container for a class

                        // Get the next box with top score, filter by iou_threshold
                        BoxInfo next_candidate;
                        float original_score;

                        while (!sorted_boxes.empty())
                        {
                            next_candidate = sorted_boxes.top();
                            original_score = next_candidate.score;
                            sorted_boxes.pop();

                            bool should_hard_suppress = false;
                            for (int64_t j = static_cast<int64_t>(selected.size()) - 1;
                                 j >= next_candidate.suppress_begin_index;
                                 --j)
                            {
                                float iou = multiclass_nms_v8::intersectionOverUnion(
                                    next_candidate.box, selected[j].box);
                                next_candidate.score *= func(iou, adaptive_threshold);

                                if (iou >= adaptive_threshold)
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
                                if (nms_eta < 1 && adaptive_threshold > 0.5)
                                {
                                    adaptive_threshold *= nms_eta;
                                }
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
                            selected_boxes.push_back(box_info);
                        }
                        num_dets += selected.size();
                    } // for each class

                    /* sort inside batch element */
                    if (sort_result_type == op::v8::MulticlassNms::SortResultType::SCORE)
                    {
                        std::sort(selected_boxes.begin(),
                                  selected_boxes.end(),
                                  [](const BoxInfo& l, const BoxInfo& r) {
                                      return (
                                          (l.batch_index == r.batch_index) &&
                                          ((l.score > r.score) ||
                                           ((std::fabs(l.score - r.score) < 1e-6) &&
                                            l.class_index < r.class_index) ||
                                           ((std::fabs(l.score - r.score) < 1e-6) &&
                                            l.class_index == r.class_index && l.index < r.index)));
                                  });
                    }
                    // in case of "NONE" and "CLASSID", pass through

                    // threshold keep_top_k for each batch element
                    if (keep_top_k > -1 && keep_top_k < num_dets)
                    {
                        num_dets = keep_top_k;
                        selected_boxes.resize(num_dets);
                    }

                    *valid_outputs++ = num_dets;
                    for (auto& v : selected_boxes)
                    {
                        filteredBoxes.push_back(v);
                    }
                } // for each batch element

                if (sort_result_across_batch)
                { /* sort across batch */
                    if (sort_result_type == op::v8::MulticlassNms::SortResultType::SCORE)
                    {
                        std::sort(
                            filteredBoxes.begin(),
                            filteredBoxes.end(),
                            [](const BoxInfo& l, const BoxInfo& r) {
                                return (l.score > r.score) ||
                                       (l.score == r.score && l.batch_index < r.batch_index) ||
                                       (l.score == r.score && l.batch_index == r.batch_index &&
                                        l.class_index < r.class_index) ||
                                       (l.score == r.score && l.batch_index == r.batch_index &&
                                        l.class_index == r.class_index && l.index < r.index);
                            });
                    }
                    else if (sort_result_type == op::v8::MulticlassNms::SortResultType::CLASSID)
                    {
                        std::sort(filteredBoxes.begin(),
                                  filteredBoxes.end(),
                                  [](const BoxInfo& l, const BoxInfo& r) {
                                      return (l.class_index < r.class_index) ||
                                             (l.class_index == r.class_index &&
                                              l.batch_index < r.batch_index) ||
                                             (l.class_index == r.class_index &&
                                              l.batch_index == r.batch_index &&
                                              l.score > r.score) ||
                                             (l.class_index == r.class_index &&
                                              l.batch_index == r.batch_index &&
                                              l.score == r.score && l.index < r.index);
                                  });
                    }
                }

                /* output */

                size_t max_num_of_selected_indices = selected_indices_shape[0];
                size_t output_size = std::min(filteredBoxes.size(), max_num_of_selected_indices);

                size_t idx;
                for (idx = 0; idx < output_size; idx++)
                {
                    const auto& box_info = filteredBoxes[idx];
                    SelectedIndex selected_index{box_info.batch_index, box_info.index, num_boxes};
                    SelectedOutput selected_score{static_cast<float>(box_info.class_index),
                                                  box_info.score,
                                                  box_info.box.x1,
                                                  box_info.box.y1,
                                                  box_info.box.x2,
                                                  box_info.box.y2};

                    selected_indices_ptr[idx] = selected_index;
                    selected_scores_ptr[idx] = selected_score;
                }

                SelectedIndex selected_index_filler{0, 0, 0};
                SelectedOutput selected_score_filler{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                for (; idx < max_num_of_selected_indices; idx++)
                {
                    selected_indices_ptr[idx] = selected_index_filler;
                    selected_scores_ptr[idx] = selected_score_filler;
                }
            }

            void multiclass_nms_postprocessing(const HostTensorVector& outputs,
                                               const ngraph::element::Type output_type,
                                               const std::vector<float>& selected_outputs,
                                               const std::vector<int64_t>& selected_indices,
                                               const std::vector<int64_t>& valid_outputs,
                                               const ngraph::element::Type selected_scores_type)
            {
                auto num_selected = std::accumulate(valid_outputs.begin(), valid_outputs.end(), 0);

                /* shape & type */

                outputs[0]->set_element_type(selected_scores_type); // "selected_outputs"
                outputs[0]->set_shape(Shape{static_cast<size_t>(num_selected), 6});

                size_t num_of_outputs = outputs.size();

                if (num_of_outputs >= 2)
                {
                    outputs[1]->set_element_type(output_type); // "selected_indices"
                    outputs[1]->set_shape(Shape{static_cast<size_t>(num_selected), 1});
                }

                if (num_of_outputs >= 3)
                {
                    outputs[2]->set_element_type(output_type); // "selected_num"
                    outputs[2]->set_shape(Shape{valid_outputs.size()});
                }

                /* data */
                size_t selected_outputs_size = num_selected * 6;

                switch (selected_scores_type)
                {
                case element::Type_t::bf16:
                {
                    bfloat16* scores_ptr = outputs[0]->get_data_ptr<bfloat16>();
                    for (size_t i = 0; i < selected_outputs_size; ++i)
                    {
                        scores_ptr[i] = bfloat16(selected_outputs[i]);
                    }
                }
                break;
                case element::Type_t::f16:
                {
                    float16* scores_ptr = outputs[0]->get_data_ptr<float16>();
                    for (size_t i = 0; i < selected_outputs_size; ++i)
                    {
                        scores_ptr[i] = float16(selected_outputs[i]);
                    }
                }
                break;
                case element::Type_t::f32:
                {
                    float* scores_ptr = outputs[0]->get_data_ptr<float>();
                    memcpy(
                        scores_ptr, selected_outputs.data(), selected_outputs_size * sizeof(float));
                }
                break;
                default:;
                }

                if (num_of_outputs < 2)
                {
                    return;
                }

                size_t selected_indices_size = num_selected * 1;

                if (output_type == ngraph::element::i64)
                {
                    int64_t* indices_ptr = outputs[1]->get_data_ptr<int64_t>();
                    memcpy(indices_ptr,
                           selected_indices.data(),
                           selected_indices_size * sizeof(int64_t));
                }
                else
                {
                    int32_t* indices_ptr = outputs[1]->get_data_ptr<int32_t>();
                    for (size_t i = 0; i < selected_indices_size; ++i)
                    {
                        indices_ptr[i] = static_cast<int32_t>(selected_indices[i]);
                    }
                }

                if (num_of_outputs < 3)
                {
                    return;
                }

                if (output_type == ngraph::element::i64)
                {
                    int64_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int64_t>();
                    memcpy(valid_outputs_ptr,
                           valid_outputs.data(),
                           valid_outputs.size() * sizeof(int64_t));
                }
                else
                {
                    int32_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int32_t>();
                    for (size_t i = 0; i < valid_outputs.size(); ++i)
                    {
                        valid_outputs_ptr[i] = static_cast<int32_t>(valid_outputs[i]);
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
