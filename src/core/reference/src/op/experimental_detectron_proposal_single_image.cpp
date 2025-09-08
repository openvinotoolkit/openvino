// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/experimental_detectron_proposal_single_image.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <utility>

#include "openvino/core/shape.hpp"
#include "openvino/op/experimental_detectron_generate_proposals.hpp"
#include "openvino/reference/proposal.hpp"

namespace {
using ProposalBox = ov::reference::details::ProposalBox<float>;

void refine_anchors(const float* deltas,
                    const float* scores,
                    const float* anchors,
                    float* proposals,
                    const int64_t anchors_num,
                    const int64_t bottom_H,
                    const int64_t bottom_W,
                    const float img_H,
                    const float img_W,
                    const float min_box_H,
                    const float min_box_W,
                    const float max_delta_log_wh,
                    const float coordinates_offset) {
    int64_t bottom_area = bottom_H * bottom_W;

    for (int64_t h = 0; h < bottom_H; ++h) {
        for (int64_t w = 0; w < bottom_W; ++w) {
            int64_t a_idx = (h * bottom_W + w) * anchors_num * 4;
            int64_t p_idx = (h * bottom_W + w) * anchors_num * 5;
            int64_t sc_idx = h * bottom_W + w;
            int64_t d_idx = h * bottom_W + w;

            for (int64_t anchor = 0; anchor < anchors_num; ++anchor) {
                float x0 = anchors[a_idx + 0];
                float y0 = anchors[a_idx + 1];
                float x1 = anchors[a_idx + 2];
                float y1 = anchors[a_idx + 3];

                const float dx = deltas[d_idx + 0 * bottom_area];
                const float dy = deltas[d_idx + 1 * bottom_area];
                const float d_log_w = deltas[d_idx + 2 * bottom_area];
                const float d_log_h = deltas[d_idx + 3 * bottom_area];

                const float score = scores[sc_idx];

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

                proposals[p_idx + 0] = x0;
                proposals[p_idx + 1] = y0;
                proposals[p_idx + 2] = x1;
                proposals[p_idx + 3] = y1;
                proposals[p_idx + 4] = (min_box_W <= box_w) * (min_box_H <= box_h) * score;

                a_idx += 4;
                p_idx += 5;
                sc_idx += bottom_area;
                d_idx += 4 * bottom_area;
            }
        }
    }
}

void unpack_boxes(const float* p_proposals, float* unpacked_boxes, int64_t pre_nms_topn) {
    for (int64_t i = 0; i < pre_nms_topn; ++i) {
        unpacked_boxes[0 * pre_nms_topn + i] = p_proposals[5 * i + 0];
        unpacked_boxes[1 * pre_nms_topn + i] = p_proposals[5 * i + 1];
        unpacked_boxes[2 * pre_nms_topn + i] = p_proposals[5 * i + 2];
        unpacked_boxes[3 * pre_nms_topn + i] = p_proposals[5 * i + 3];
        unpacked_boxes[4 * pre_nms_topn + i] = p_proposals[5 * i + 4];
    }
}

void nms_cpu(const int64_t num_boxes,
             int64_t is_dead[],
             const float* boxes,
             int64_t index_out[],
             int64_t* const num_out,
             const int64_t base_index,
             const float nms_thresh,
             const int64_t max_num_out,
             const float coordinates_offset) {
    const int64_t num_proposals = num_boxes;
    int64_t count = 0;

    const float* x0 = boxes + 0 * num_proposals;
    const float* y0 = boxes + 1 * num_proposals;
    const float* x1 = boxes + 2 * num_proposals;
    const float* y1 = boxes + 3 * num_proposals;

    std::fill(is_dead, is_dead + num_boxes, static_cast<int64_t>(0));

    for (int64_t box = 0; box < num_boxes; ++box) {
        if (is_dead[box])
            continue;

        index_out[count++] = base_index + box;
        if (count == max_num_out)
            break;

        const float x0i = x0[box];
        const float y0i = y0[box];
        const float x1i = x1[box];
        const float y1i = y1[box];

        const float a_width = x1i - x0i;
        const float a_height = y1i - y0i;
        const float a_area = (a_width + coordinates_offset) * (a_height + coordinates_offset);

        for (int64_t tail = box + 1; tail < num_boxes; ++tail) {
            const float x0j = x0[tail];
            const float y0j = y0[tail];
            const float x1j = x1[tail];
            const float y1j = y1[tail];

            const float x0 = std::max(x0i, x0j);
            const float y0 = std::max(y0i, y0j);
            const float x1 = std::min(x1i, x1j);
            const float y1 = std::min(y1i, y1j);

            const float width = x1 - x0 + coordinates_offset;
            const float height = y1 - y0 + coordinates_offset;
            const float area = std::max(0.0f, width) * std::max(0.0f, height);

            const float b_width = x1j - x0j;
            const float b_height = y1j - y0j;
            const float b_area = (b_width + coordinates_offset) * (b_height + coordinates_offset);

            const float intersection_area = area / (a_area + b_area - area);

            is_dead[tail] =
                (nms_thresh < intersection_area) && (x0i <= x1j) && (y0i <= y1j) && (x0j <= x1i) && (y0j <= y1i);
        }
    }
    *num_out = count;
}

void fill_output_blobs(const float* proposals,
                       const int64_t* roi_indices,
                       float* rois,
                       float* scores,
                       const int64_t num_proposals,
                       const int64_t num_rois,
                       const int64_t post_nms_topn) {
    const float* src_x0 = proposals + 0 * num_proposals;
    const float* src_y0 = proposals + 1 * num_proposals;
    const float* src_x1 = proposals + 2 * num_proposals;
    const float* src_y1 = proposals + 3 * num_proposals;
    const float* src_score = proposals + 4 * num_proposals;

    for (int64_t i = 0; i < num_rois; ++i) {
        int64_t index = roi_indices[i];
        rois[i * 4 + 0] = src_x0[index];
        rois[i * 4 + 1] = src_y0[index];
        rois[i * 4 + 2] = src_x1[index];
        rois[i * 4 + 3] = src_y1[index];
        scores[i] = src_score[index];
    }

    if (num_rois < post_nms_topn) {
        for (int64_t i = 4 * num_rois; i < 4 * post_nms_topn; i++) {
            rois[i] = 0.0f;
        }
        for (int64_t i = num_rois; i < post_nms_topn; i++) {
            scores[i] = 0.0f;
        }
    }
}
}  // namespace

namespace ov {
namespace reference {
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
    float* output_scores) {
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

    const float nms_thresh = attrs.nms_threshold;
    const int64_t post_nms_topn = attrs.post_nms_count;

    // number of all proposals = num_anchors * H * W
    const int64_t num_proposals = anchors_num * bottom_H * bottom_W;

    // number of top-n proposals before NMS
    const int64_t pre_nms_topn = std::min(num_proposals, attrs.pre_nms_count);

    // number of final RoIs
    int64_t num_rois = 0;

    std::vector<ProposalBox> proposals(num_proposals);
    std::vector<float> unpacked_boxes(5 * pre_nms_topn);
    std::vector<int64_t> is_dead(pre_nms_topn);

    refine_anchors(deltas,
                   scores,
                   anchors,
                   reinterpret_cast<float*>(proposals.data()),
                   anchors_num,
                   bottom_H,
                   bottom_W,
                   img_H,
                   img_W,
                   min_box_H,
                   min_box_W,
                   static_cast<const float>(std::log(1000. / 16.)),
                   1.0f);
    std::partial_sort(proposals.begin(),
                      proposals.begin() + pre_nms_topn,
                      proposals.end(),
                      [](const ProposalBox& struct1, const ProposalBox& struct2) {
                          return (struct1.score > struct2.score);
                      });

    unpack_boxes(reinterpret_cast<float*>(proposals.data()), unpacked_boxes.data(), pre_nms_topn);

    std::vector<int64_t> roi_indices(post_nms_topn);

    nms_cpu(pre_nms_topn,
            is_dead.data(),
            unpacked_boxes.data(),
            roi_indices.data(),
            &num_rois,
            0,
            nms_thresh,
            post_nms_topn,
            0.0f);
    fill_output_blobs(unpacked_boxes.data(),
                      roi_indices.data(),
                      output_rois,
                      output_scores,
                      pre_nms_topn,
                      num_rois,
                      post_nms_topn);
}

void experimental_detectron_proposals_single_image_postprocessing(void* prois,
                                                                  void* pscores,
                                                                  const element::Type output_type,
                                                                  const std::vector<float>& output_rois,
                                                                  const std::vector<float>& output_scores,
                                                                  const Shape& output_rois_shape,
                                                                  const Shape& output_scores_shape) {
    size_t rois_num = output_rois_shape[0];

    switch (output_type) {
    case element::Type_t::bf16: {
        bfloat16* rois_ptr = reinterpret_cast<bfloat16*>(prois);
        bfloat16* scores_ptr = reinterpret_cast<bfloat16*>(pscores);
        for (size_t i = 0; i < rois_num; ++i) {
            rois_ptr[4 * i + 0] = bfloat16(output_rois[4 * i + 0]);
            rois_ptr[4 * i + 1] = bfloat16(output_rois[4 * i + 1]);
            rois_ptr[4 * i + 2] = bfloat16(output_rois[4 * i + 2]);
            rois_ptr[4 * i + 3] = bfloat16(output_rois[4 * i + 3]);
            scores_ptr[i] = bfloat16(output_scores[i]);
        }
    } break;
    case element::Type_t::f16: {
        float16* rois_ptr = reinterpret_cast<float16*>(prois);
        float16* scores_ptr = reinterpret_cast<float16*>(pscores);
        for (size_t i = 0; i < rois_num; ++i) {
            rois_ptr[4 * i + 0] = float16(output_rois[4 * i + 0]);
            rois_ptr[4 * i + 1] = float16(output_rois[4 * i + 1]);
            rois_ptr[4 * i + 2] = float16(output_rois[4 * i + 2]);
            rois_ptr[4 * i + 3] = float16(output_rois[4 * i + 3]);
            scores_ptr[i] = float16(output_scores[i]);
        }
    } break;
    case element::Type_t::f32: {
        float* rois_ptr = reinterpret_cast<float*>(prois);
        float* scores_ptr = reinterpret_cast<float*>(pscores);
        memcpy(rois_ptr, output_rois.data(), shape_size(output_rois_shape) * sizeof(float));
        memcpy(scores_ptr, output_scores.data(), shape_size(output_scores_shape) * sizeof(float));
    } break;
    default:;
        OPENVINO_THROW("Unsupported input data type: "
                       "ExperimentalDetectronGenerateProposalsSingleImage operation"
                       " supports only fp32, fp16, or bf16 data.");
    }
}
}  // namespace reference
}  // namespace ov
