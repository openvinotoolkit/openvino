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


#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>
#include "ngraph/op/non_max_suppression.hpp"
#include "ngraph/runtime/reference/non_max_suppression.hpp"
#include "ngraph/shape.hpp"

using namespace ngraph;
using namespace ngraph::runtime::reference;

struct Rectangle
{
    float y1;
    float x1;
    float y2;
    float x2;
};

static float intersectionOverUnion(float* boxesI, float* boxesJ)
{
    Rectangle boxI = *(reinterpret_cast<Rectangle*>(boxesI));
    Rectangle boxJ = *(reinterpret_cast<Rectangle*>(boxesJ));

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

struct FilteredBoxes
{
    float score;
    int batch_index;
    int class_index;
    int box_index;
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
                                     int64_t* valid_outputs)
            {
                float scale = 0.0f;
                if (soft_nms_sigma > 0.0f) {
                    scale = - 0.5f / soft_nms_sigma;
                }

                auto func = [iou_threshold, scale](float iou) {
                    const float weight = std::exp(scale * iou * iou);
                    return iou <= iou_threshold ? weight : 0.0f;
                };

                // boxes shape: {num_batches, num_boxes, 4}
                // scores shape: {num_batches, num_classes, num_boxes}
                size_t num_batches = scores_data_shape[0];
                size_t num_classes = scores_data_shape[1];

                std::vector<FilteredBoxes> fb;

                for (int batch = 0; batch < num_batches; batch++)
                {
                    for (int class_idx = 0; class_idx < num_classes; class_idx++)
                    {
                    }
                }
            }
        }
    }
}
