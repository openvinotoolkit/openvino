// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/multiclass_nms.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <vector>

#include "ngraph/runtime/reference/multiclass_nms.hpp"
#include "ngraph/runtime/reference/utils/nms_common.hpp"
#include "ngraph/shape.hpp"

using namespace ngraph;
using namespace ngraph::runtime::reference;

namespace ngraph {
namespace runtime {
namespace reference {
namespace multiclass_nms_v8 {
using Rectangle = runtime::reference::nms_common::Rectangle;
using BoxInfo = runtime::reference::nms_common::BoxInfo;
static float intersectionOverUnion(const Rectangle& boxI, const Rectangle& boxJ, const bool normalized) {
    const float norm = static_cast<float>(normalized == false);

    float areaI = (boxI.y2 - boxI.y1 + norm) * (boxI.x2 - boxI.x1 + norm);
    float areaJ = (boxJ.y2 - boxJ.y1 + norm) * (boxJ.x2 - boxJ.x1 + norm);

    if (areaI <= 0.0f || areaJ <= 0.0f) {
        return 0.0f;
    }

    float intersection_ymin = std::max(boxI.y1, boxJ.y1);
    float intersection_xmin = std::max(boxI.x1, boxJ.x1);
    float intersection_ymax = std::min(boxI.y2, boxJ.y2);
    float intersection_xmax = std::min(boxI.x2, boxJ.x2);

    float intersection_area = std::max(intersection_ymax - intersection_ymin + norm, 0.0f) *
                              std::max(intersection_xmax - intersection_xmin + norm, 0.0f);

    return intersection_area / (areaI + areaJ - intersection_area);
}

struct SelectedIndex {
    SelectedIndex(int64_t batch_idx, int64_t box_idx, int64_t num_box)
        : flattened_index(batch_idx * num_box + box_idx) {}

    SelectedIndex() = default;

    int64_t flattened_index = 0;
};

struct SelectedOutput {
    SelectedOutput(float class_idx, float score, float x1, float y1, float x2, float y2)
        : class_index{class_idx},
          box_score{score},
          xmin{x1},
          ymin{y1},
          xmax{x2},
          ymax{y2} {}

    SelectedOutput() = default;

    float class_index = 0.0f;
    float box_score = 0.0f;
    float xmin, ymin, xmax, ymax;
};
}  // namespace multiclass_nms_v8

void multiclass_nms(const float* boxes_data,
                    const Shape& boxes_data_shape,
                    const float* scores_data,
                    const Shape& scores_data_shape,
                    const op::v8::MulticlassNms::Attributes& attrs,
                    float* selected_outputs,
                    const Shape& selected_outputs_shape,
                    int64_t* selected_indices,
                    const Shape& selected_indices_shape,
                    int64_t* valid_outputs) {
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

    SelectedIndex* selected_indices_ptr = reinterpret_cast<SelectedIndex*>(selected_indices);
    SelectedOutput* selected_scores_ptr = reinterpret_cast<SelectedOutput*>(selected_outputs);

    std::vector<BoxInfo> filteredBoxes;  // container for the whole batch

    for (int64_t batch = 0; batch < num_batches; batch++) {
        const float* boxesPtr = boxes_data + batch * num_boxes * 4;
        Rectangle* r = reinterpret_cast<Rectangle*>(const_cast<float*>(boxesPtr));

        int64_t num_dets = 0;
        std::vector<BoxInfo> selected_boxes;  // container for a batch element

        for (int64_t class_idx = 0; class_idx < num_classes; class_idx++) {
            if (class_idx == attrs.background_class)
                continue;

            auto adaptive_threshold = attrs.iou_threshold;

            const float* scoresPtr = scores_data + batch * (num_classes * num_boxes) + class_idx * num_boxes;

            std::vector<BoxInfo> candidate_boxes;

            for (int64_t box_idx = 0; box_idx < num_boxes; box_idx++) {
                if (scoresPtr[box_idx] >= attrs.score_threshold) /* NOTE: ">=" instead of ">" used in PDPD */
                {
                    candidate_boxes.emplace_back(r[box_idx], box_idx, scoresPtr[box_idx], 0, batch, class_idx);
                }
            }

            int candiate_size = candidate_boxes.size();

            // threshold nms_top_k for each class
            // NOTE: "nms_top_k" in PDPD not exactly equal to
            // "max_output_boxes_per_class" in ONNX.
            if (attrs.nms_top_k > -1 && attrs.nms_top_k < candiate_size) {
                candiate_size = attrs.nms_top_k;
            }

            if (candiate_size <= 0)  // early drop
            {
                continue;
            }

            // sort by score in current class
            std::partial_sort(candidate_boxes.begin(),
                              candidate_boxes.begin() + candiate_size,
                              candidate_boxes.end(),
                              std::greater<BoxInfo>());

            std::priority_queue<BoxInfo> sorted_boxes(candidate_boxes.begin(),
                                                      candidate_boxes.begin() + candiate_size,
                                                      std::less<BoxInfo>());

            std::vector<BoxInfo> selected;  // container for a class

            // Get the next box with top score, filter by iou_threshold
            BoxInfo next_candidate;
            float original_score;

            while (!sorted_boxes.empty()) {
                next_candidate = sorted_boxes.top();
                original_score = next_candidate.score;
                sorted_boxes.pop();

                bool should_hard_suppress = false;
                for (int64_t j = static_cast<int64_t>(selected.size()) - 1; j >= next_candidate.suppress_begin_index;
                     --j) {
                    float iou =
                        multiclass_nms_v8::intersectionOverUnion(next_candidate.box, selected[j].box, attrs.normalized);
                    next_candidate.score *= func(iou, adaptive_threshold);

                    if (iou >= adaptive_threshold) {
                        should_hard_suppress = true;
                        break;
                    }

                    if (next_candidate.score <= attrs.score_threshold) {
                        break;
                    }
                }

                next_candidate.suppress_begin_index = selected.size();

                if (!should_hard_suppress) {
                    if (attrs.nms_eta < 1 && adaptive_threshold > 0.5) {
                        adaptive_threshold *= attrs.nms_eta;
                    }
                    if (next_candidate.score == original_score) {
                        selected.push_back(next_candidate);
                        continue;
                    }
                    if (next_candidate.score > attrs.score_threshold) {
                        sorted_boxes.push(next_candidate);
                    }
                }
            }

            for (const auto& box_info : selected) {
                selected_boxes.push_back(box_info);
            }
            num_dets += selected.size();
        }  // for each class

        // sort inside batch element before go through keep_top_k
        std::sort(selected_boxes.begin(), selected_boxes.end(), [](const BoxInfo& l, const BoxInfo& r) {
            return ((l.batch_index == r.batch_index) &&
                    ((l.score > r.score) || ((std::fabs(l.score - r.score) < 1e-6) && l.class_index < r.class_index) ||
                     ((std::fabs(l.score - r.score) < 1e-6) && l.class_index == r.class_index && l.index < r.index)));
        });

        // threshold keep_top_k for each batch element
        if (attrs.keep_top_k > -1 && attrs.keep_top_k < num_dets) {
            num_dets = attrs.keep_top_k;
            selected_boxes.resize(num_dets);
        }

        // sort
        if (!attrs.sort_result_across_batch) {
            if (attrs.sort_result_type == op::v8::MulticlassNms::SortResultType::CLASSID) {
                std::sort(selected_boxes.begin(), selected_boxes.end(), [](const BoxInfo& l, const BoxInfo& r) {
                    return ((l.batch_index == r.batch_index) &&
                            ((l.class_index < r.class_index) ||
                             ((l.class_index == r.class_index) && l.score > r.score) ||
                             ((std::fabs(l.score - r.score) <= 1e-6) && l.class_index == r.class_index &&
                              l.index < r.index)));
                });
            }
            // in case of "SCORE", pass through, as,
            // it has already gurranteed.
        }

        *valid_outputs++ = num_dets;
        for (auto& v : selected_boxes) {
            filteredBoxes.push_back(v);
        }
    }  // for each batch element

    if (attrs.sort_result_across_batch) { /* sort across batch */
        if (attrs.sort_result_type == op::v8::MulticlassNms::SortResultType::SCORE) {
            std::sort(filteredBoxes.begin(), filteredBoxes.end(), [](const BoxInfo& l, const BoxInfo& r) {
                return (l.score > r.score) || (l.score == r.score && l.batch_index < r.batch_index) ||
                       (l.score == r.score && l.batch_index == r.batch_index && l.class_index < r.class_index) ||
                       (l.score == r.score && l.batch_index == r.batch_index && l.class_index == r.class_index &&
                        l.index < r.index);
            });
        } else if (attrs.sort_result_type == op::v8::MulticlassNms::SortResultType::CLASSID) {
            std::sort(filteredBoxes.begin(), filteredBoxes.end(), [](const BoxInfo& l, const BoxInfo& r) {
                return (l.class_index < r.class_index) ||
                       (l.class_index == r.class_index && l.batch_index < r.batch_index) ||
                       (l.class_index == r.class_index && l.batch_index == r.batch_index && l.score > r.score) ||
                       (l.class_index == r.class_index && l.batch_index == r.batch_index && l.score == r.score &&
                        l.index < r.index);
            });
        }
    }

    /* output */

    size_t max_num_of_selected_indices = selected_indices_shape[0];
    size_t output_size = std::min(filteredBoxes.size(), max_num_of_selected_indices);

    size_t idx;
    for (idx = 0; idx < output_size; idx++) {
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
    for (; idx < max_num_of_selected_indices; idx++) {
        selected_indices_ptr[idx] = selected_index_filler;
        selected_scores_ptr[idx] = selected_score_filler;
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
