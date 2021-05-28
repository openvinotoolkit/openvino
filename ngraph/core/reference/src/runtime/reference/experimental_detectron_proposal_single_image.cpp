// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/experimental_detectron_proposal_single_image.hpp"
#include <algorithm>
#include <cassert>
#include <utility>
#include "ngraph/op/experimental_detectron_generate_proposals.hpp"
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
                const Shape& im_info_shape,
                const Shape& anchors_shape,
                const Shape& deltas_shape,
                const Shape& scores_shape,
                float* output_rois,
                float* output_scores)
            {
                const int64_t anchors_num = static_cast<int64_t>(scores_shape[0]);

                // bottom shape: (num_anchors) x H x W
                const int64_t bottom_H = static_cast<int64_t>(deltas_shape[1]);
                const int64_t bottom_W = static_cast<int64_t>(deltas_shape[2]);

                // input image height & width
                const float img_H = im_info[0];
                const float img_W = im_info[1];

                // scale factor for height & width

                // minimum box width & height
                const float min_box_H = attrs.min_size;
                const float min_box_W = attrs.min_size;

                // number of all proposals = num_anchors * H * W
                const int64_t num_proposals = anchors_num * bottom_H * bottom_W;

                // number of top-n proposals before NMS
                const int64_t pre_nms_topn = std::min(num_proposals, attrs.pre_nms_count);

                // number of final RoIs
                int64_t num_rois = 0;

                // enumerate all proposals
                //   num_proposals = num_anchors * H * W
                //   (x1, y1, x2, y2, score) for each proposal
                // NOTE: for bottom, only foreground scores are passed
                struct ProposalBox {
                    float x0;
                    float y0;
                    float x1;
                    float y1;
                    float score;
                };
                std::vector<ProposalBox> proposals_(num_proposals);
                std::vector<float> unpacked_boxes(5 * pre_nms_topn);
                std::vector<int64_t> is_dead(pre_nms_topn);

                // Execute
                int64_t batch_size = 1;
                for (int64_t n = 0; n < batch_size; ++n)
                {
                }
            }

            void experimental_detectron_proposals_single_image_postprocessing(
                void* prois,
                void* pscores,
                const ngraph::element::Type output_type,
                const std::vector<float>& output_rois,
                const std::vector<float>& output_scores,
                const Shape& output_rois_shape,
                const Shape& output_scores_shape)
            {
                size_t rois_num = output_rois_shape[0];

                switch (output_type)
                {
                case element::Type_t::bf16:
                {
                    bfloat16* rois_ptr = reinterpret_cast<bfloat16*>(prois);
                    bfloat16* scores_ptr = reinterpret_cast<bfloat16*>(pscores);
                    for (size_t i = 0; i < rois_num; ++i)
                    {
                        rois_ptr[4 * i + 0] = bfloat16(output_rois[4 * i + 0]);
                        rois_ptr[4 * i + 1] = bfloat16(output_rois[4 * i + 1]);
                        rois_ptr[4 * i + 2] = bfloat16(output_rois[4 * i + 2]);
                        rois_ptr[4 * i + 3] = bfloat16(output_rois[4 * i + 3]);
                        scores_ptr[i] = bfloat16(output_scores[i]);
                    }
                }
                break;
                case element::Type_t::f16:
                {
                    float16* rois_ptr = reinterpret_cast<float16*>(prois);
                    float16* scores_ptr = reinterpret_cast<float16*>(pscores);
                    for (size_t i = 0; i < rois_num; ++i)
                    {
                        rois_ptr[4 * i + 0] = float16(output_rois[4 * i + 0]);
                        rois_ptr[4 * i + 1] = float16(output_rois[4 * i + 1]);
                        rois_ptr[4 * i + 2] = float16(output_rois[4 * i + 2]);
                        rois_ptr[4 * i + 3] = float16(output_rois[4 * i + 3]);
                        scores_ptr[i] = float16(output_scores[i]);
                    }
                }
                break;
                case element::Type_t::f32:
                {
                    float* rois_ptr = reinterpret_cast<float*>(prois);
                    float* scores_ptr = reinterpret_cast<float*>(pscores);
                    memcpy(rois_ptr,
                           output_rois.data(),
                           shape_size(output_rois_shape) * sizeof(float));
                    memcpy(scores_ptr,
                           output_scores.data(),
                           shape_size(output_scores_shape) * sizeof(float));
                }
                break;
                default:;
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
