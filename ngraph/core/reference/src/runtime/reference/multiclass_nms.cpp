// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/multiclass_nms.hpp"
#include <algorithm>
#include <cmath>
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
            namespace
            {
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

                    float intersection_area =
                        std::max(intersection_ymax - intersection_ymin, 0.0f) *
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
            } // namespace
            void multiclass_nms(const float* boxes_data,
                                const Shape& boxes_data_shape,
                                const float* scores_data,
                                const Shape& scores_data_shape,
                                op::util::NmsBase::SortResultType sort_result_type,
                                float iou_threshold,
                                float score_threshold,
                                int nms_top_k,
                                int keep_top_k,
                                int background_class,
                                float nms_eta,
                                float* selected_outputs,
                                const Shape& selected_outputs_shape,
                                int64_t* selected_indices,
                                const Shape& selected_indices_shape,
                                int64_t* valid_outputs)
            {
                BoxInfo info;
                intersectionOverUnion(Rectangle{}, Rectangle{});
                *valid_outputs = 0;
            }

            void multiclass_nms_postprocessing(const HostTensorVector& outputs,
                                               const ngraph::element::Type output_type,
                                               const std::vector<float>& selected_outputs,
                                               const std::vector<int64_t>& selected_indices,
                                               int64_t valid_outputs)
            {
                outputs[0]->set_shape(Shape{static_cast<size_t>(valid_outputs), 6});
                float* ptr = outputs[0]->get_data_ptr<float>();
                memcpy(ptr, selected_outputs.data(), valid_outputs * sizeof(float) * 6);

                outputs[1]->set_shape(Shape{static_cast<size_t>(valid_outputs), 1});
                if (output_type == ngraph::element::i64)
                {
                    int64_t* indices_ptr = outputs[1]->get_data_ptr<int64_t>();
                    memcpy(indices_ptr, selected_indices.data(), valid_outputs * sizeof(int64_t));
                }
                else
                {
                    int32_t* indices_ptr = outputs[1]->get_data_ptr<int32_t>();
                    for (size_t i = 0; i < (size_t)valid_outputs; ++i)
                    {
                        indices_ptr[i] = static_cast<int32_t>(selected_indices[i]);
                    }
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
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
