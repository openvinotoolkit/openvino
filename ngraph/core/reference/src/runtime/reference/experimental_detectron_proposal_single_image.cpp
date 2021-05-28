// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/experimental_detectron_generate_proposals.hpp"
#include <algorithm>
#include <cassert>
#include <utility>
#include "ngraph/runtime/reference/experimental_detectron_proposal_single_image.hpp"
#include "ngraph/shape.hpp"

namespace
{
} // namespace

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void experimental_detectron_proposals_single_image(
                const float* im_info,
                const float* anchors,
                const float* deltas,
                const float* scores,
                const op::v6::ExperimentalDetectronGenerateProposalsSingleImage::Attributes& attrs,
                float* output_rois,
                float* output_scores)
            {
            }

            void experimental_detectron_proposals_single_image_postprocessing(
                void* prois,
                void* pscores,
                const ngraph::element::Type output_type,
                const std::vector<float>& output_rois,
                const std::vector<float>& output_scores,
                const Shape& output_rois_shape,
                const Shape& output_scores_shape)
//             void experimental_detectron_detection_output_postprocessing(
//                 void* pboxes,
//                 void* pclasses,
//                 void* pscores,
//                 const ngraph::element::Type output_type,
//                 const std::vector<float>& output_boxes,
//                 const std::vector<int32_t>& output_classes,
//                 const std::vector<float>& output_scores,
//                 const Shape& output_boxes_shape,
//                 const Shape& output_classes_shape,
//                 const Shape& output_scores_shape)
            {
//                 size_t rois_num = output_boxes_shape[0];
//
//                 switch (output_type)
//                 {
//                 case element::Type_t::bf16:
//                 {
//                     bfloat16* boxes_ptr = reinterpret_cast<bfloat16*>(pboxes);
//                     bfloat16* scores_ptr = reinterpret_cast<bfloat16*>(pscores);
//                     for (size_t i = 0; i < rois_num; ++i)
//                     {
//                         boxes_ptr[4 * i + 0] = bfloat16(output_boxes[4 * i + 0]);
//                         boxes_ptr[4 * i + 1] = bfloat16(output_boxes[4 * i + 1]);
//                         boxes_ptr[4 * i + 2] = bfloat16(output_boxes[4 * i + 2]);
//                         boxes_ptr[4 * i + 3] = bfloat16(output_boxes[4 * i + 3]);
//                         scores_ptr[i] = bfloat16(output_scores[i]);
//                     }
//                 }
//                 break;
//                 case element::Type_t::f16:
//                 {
//                     float16* boxes_ptr = reinterpret_cast<float16*>(pboxes);
//                     float16* scores_ptr = reinterpret_cast<float16*>(pscores);
//                     for (size_t i = 0; i < rois_num; ++i)
//                     {
//                         boxes_ptr[4 * i + 0] = float16(output_boxes[4 * i + 0]);
//                         boxes_ptr[4 * i + 1] = float16(output_boxes[4 * i + 1]);
//                         boxes_ptr[4 * i + 2] = float16(output_boxes[4 * i + 2]);
//                         boxes_ptr[4 * i + 3] = float16(output_boxes[4 * i + 3]);
//                         scores_ptr[i] = float16(output_scores[i]);
//                     }
//                 }
//                 break;
//                 case element::Type_t::f32:
//                 {
//                     float* boxes_ptr = reinterpret_cast<float*>(pboxes);
//                     float* scores_ptr = reinterpret_cast<float*>(pscores);
//                     memcpy(boxes_ptr,
//                            output_boxes.data(),
//                            shape_size(output_boxes_shape) * sizeof(float));
//                     memcpy(scores_ptr,
//                            output_scores.data(),
//                            shape_size(output_scores_shape) * sizeof(float));
//                 }
//                 break;
//                 default:;
//                 }
//
//                 int32_t* classes_ptr = reinterpret_cast<int32_t*>(pclasses);
//                 memcpy(classes_ptr,
//                        output_classes.data(),
//                        shape_size(output_classes_shape) * sizeof(int32_t));
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
