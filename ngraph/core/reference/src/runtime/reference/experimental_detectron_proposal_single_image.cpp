// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/experimental_detectron_proposal_single_image.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>
#include "ngraph/op/experimental_detectron_generate_proposals.hpp"
#include "ngraph/shape.hpp"

namespace
{
    struct ProposalBox
    {
        float x0;
        float y0;
        float x1;
        float y1;
        float score;
    };

    void refine_anchors(const float* deltas,
                        const float* scores,
                        const float* anchors,
                        ProposalBox* proposals,
                        const int64_t anchors_num,
                        const int64_t bottom_H,
                        const int64_t bottom_W,
                        const float img_H,
                        const float img_W,
                        const float min_box_H,
                        const float min_box_W,
                        const float max_delta_log_wh,
                        float coordinates_offset)
    {
        int64_t bottom_area = bottom_H * bottom_W;
        for (int64_t h = 0; h < bottom_H; ++h)
        {
            for (int64_t w = 0; w < bottom_W; ++w)
            {
                for (int64_t anchor = 0; anchor < anchors_num; ++anchor)
                {
                    float* deltas_ptr = deltas + anchor * 4 * bottom_area + h * bottom_W + w;

                    float x0 = anchors[0];
                    float y0 = anchors[1];
                    float x1 = anchors[2];
                    float y1 = anchors[3];

                    const float dx = deltas_ptr[0 * bottom_area];
                    const float dy = deltas_ptr[1 * bottom_area];
                    const float d_log_w = deltas_ptr[2 * bottom_area];
                    const float d_log_h = deltas_ptr[3 * bottom_area];

                    const float score = scores[anchor * bottom_area + h * bottom_W + w];

                    // width & height of box
                    const float ww = x1 - x0 + coordinates_offset;
                    const float hh = y1 - y0 + coordinates_offset;
                    // center location of box
                    const float ctr_x = x0 + 0.5f * ww;
                    const float ctr_y = y0 + 0.5f * hh;

                    // new center location according to deltas (dx, dy)
                    const float pred_ctr_x = dx * ww + ctr_x;
                    const float pred_ctr_y = dy * hh + ctr_y;
                    // new width & height according to deltas d(log w), d(log h)
                    const float pred_w = std::exp(std::min(d_log_w, max_delta_log_wh)) * ww;
                    const float pred_h = std::exp(std::min(d_log_h, max_delta_log_wh)) * hh;

                    // update upper-left corner location
                    x0 = pred_ctr_x - 0.5f * pred_w;
                    y0 = pred_ctr_y - 0.5f * pred_h;
                    // update lower-right corner location
                    x1 = pred_ctr_x + 0.5f * pred_w - coordinates_offset;
                    y1 = pred_ctr_y + 0.5f * pred_h - coordinates_offset;

                    // adjust new corner locations to be within the image region,
                    x0 = std::max<float>(0.0f, std::min<float>(x0, img_W - coordinates_offset));
                    y0 = std::max<float>(0.0f, std::min<float>(y0, img_H - coordinates_offset));
                    x1 = std::max<float>(0.0f, std::min<float>(x1, img_W - coordinates_offset));
                    y1 = std::max<float>(0.0f, std::min<float>(y1, img_H - coordinates_offset));

                    // recompute new width & height
                    const float box_w = x1 - x0 + coordinates_offset;
                    const float box_h = y1 - y0 + coordinates_offset;

                    proposals->x0 = x0;
                    proposals->y0 = y0;
                    proposals->x1 = x1;
                    proposals->y1 = y1;
                    proposals->score = (min_box_W <= box_w) * (min_box_H <= box_h) * score;

                    anchors += 4;
                    proposals++;
                }
            }
        }
    }
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
                std::vector<ProposalBox> proposals(num_proposals);
                std::vector<float> unpacked_boxes(5 * pre_nms_topn);
                std::vector<int64_t> is_dead(pre_nms_topn);

                // Execute
                int64_t batch_size = 1; // deltas_shape[0]
                for (int64_t n = 0; n < batch_size; ++n)
                {
                    refine_anchors(deltas,
                                   scores,
                                   anchors,
                                   proposals.data(),
                                   anchors_num,
                                   bottom_H,
                                   bottom_W,
                                   img_H,
                                   img_W,
                                   min_box_H,
                                   min_box_W,
                                   static_cast<const float>(std::log(1000.0 / 16.0)),
                                   1.0f);
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
