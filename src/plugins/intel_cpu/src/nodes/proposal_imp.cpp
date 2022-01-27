// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "proposal_imp.hpp"

#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#if defined(HAVE_AVX2)
#include <immintrin.h>
#endif
#include "ie_parallel.hpp"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

void enumerate_proposals_cpu(const float* bottom4d, const float* d_anchor4d, const float* anchors,
                             float* proposals, const int num_anchors, const int bottom_H,
                             const int bottom_W, const float img_H, const float img_W,
                             const float min_box_H, const float min_box_W, const int feat_stride,
                             const float box_coordinate_scale, const float box_size_scale,
                             float coordinates_offset, bool initial_clip, bool swap_xy, bool clip_before_nms) {
    const int bottom_area = bottom_H * bottom_W;

    const float* p_anchors_wm = anchors + 0 * num_anchors;
    const float* p_anchors_hm = anchors + 1 * num_anchors;
    const float* p_anchors_wp = anchors + 2 * num_anchors;
    const float* p_anchors_hp = anchors + 3 * num_anchors;

    parallel_for2d(bottom_H, bottom_W, [&](size_t h, size_t w) {
        const float x = static_cast<float>((swap_xy ? h : w) * feat_stride);
        const float y = static_cast<float>((swap_xy ? w : h) * feat_stride);

        const float* p_box   = d_anchor4d + h * bottom_W + w;
        const float* p_score = bottom4d   + h * bottom_W + w;

        float* p_proposal = proposals + (h * bottom_W + w) * num_anchors * 5;

        for (int anchor = 0; anchor < num_anchors; ++anchor) {
            const float dx = p_box[(anchor * 4 + 0) * bottom_area] / box_coordinate_scale;
            const float dy = p_box[(anchor * 4 + 1) * bottom_area] / box_coordinate_scale;

            const float d_log_w = p_box[(anchor * 4 + 2) * bottom_area] / box_size_scale;
            const float d_log_h = p_box[(anchor * 4 + 3) * bottom_area] / box_size_scale;

            const float score = p_score[anchor * bottom_area];

            float x0 = x + p_anchors_wm[anchor];
            float y0 = y + p_anchors_hm[anchor];
            float x1 = x + p_anchors_wp[anchor];
            float y1 = y + p_anchors_hp[anchor];

            if (initial_clip) {
                // adjust new corner locations to be within the image region
                x0 = std::max<float>(0.0f, std::min<float>(x0, img_W));
                y0 = std::max<float>(0.0f, std::min<float>(y0, img_H));
                x1 = std::max<float>(0.0f, std::min<float>(x1, img_W));
                y1 = std::max<float>(0.0f, std::min<float>(y1, img_H));
            }

            // width & height of box
            const float ww = x1 - x0 + coordinates_offset;
            const float hh = y1 - y0 + coordinates_offset;
            // center location of box
            const float ctr_x = x0 + 0.5f * ww;
            const float ctr_y = y0 + 0.5f * hh;

            // new center location according to gradient (dx, dy)
            const float pred_ctr_x = dx * ww + ctr_x;
            const float pred_ctr_y = dy * hh + ctr_y;
            // new width & height according to gradient d(log w), d(log h)
            const float pred_w = std::exp(d_log_w) * ww;
            const float pred_h = std::exp(d_log_h) * hh;

            // update upper-left corner location
            x0 = pred_ctr_x - 0.5f * pred_w;
            y0 = pred_ctr_y - 0.5f * pred_h;
            // update lower-right corner location
            x1 = pred_ctr_x + 0.5f * pred_w;
            y1 = pred_ctr_y + 0.5f * pred_h;

            // adjust new corner locations to be within the image region,
            if (clip_before_nms) {
                x0 = std::max<float>(0.0f, std::min<float>(x0, img_W - coordinates_offset));
                y0 = std::max<float>(0.0f, std::min<float>(y0, img_H - coordinates_offset));
                x1 = std::max<float>(0.0f, std::min<float>(x1, img_W - coordinates_offset));
                y1 = std::max<float>(0.0f, std::min<float>(y1, img_H - coordinates_offset));
            }

            // recompute new width & height
            const float box_w = x1 - x0 + coordinates_offset;
            const float box_h = y1 - y0 + coordinates_offset;

            p_proposal[5*anchor + 0] = x0;
            p_proposal[5*anchor + 1] = y0;
            p_proposal[5*anchor + 2] = x1;
            p_proposal[5*anchor + 3] = y1;
            p_proposal[5*anchor + 4] = (min_box_W <= box_w) * (min_box_H <= box_h) * score;
        }
    });
}

void unpack_boxes(const float* p_proposals, float* unpacked_boxes, int pre_nms_topn, bool store_prob) {
    if (store_prob) {
        parallel_for(pre_nms_topn, [&](size_t i) {
            unpacked_boxes[0 * pre_nms_topn + i] = p_proposals[5 * i + 0];
            unpacked_boxes[1 * pre_nms_topn + i] = p_proposals[5 * i + 1];
            unpacked_boxes[2 * pre_nms_topn + i] = p_proposals[5 * i + 2];
            unpacked_boxes[3 * pre_nms_topn + i] = p_proposals[5 * i + 3];
            unpacked_boxes[4 * pre_nms_topn + i] = p_proposals[5 * i + 4];
        });
    } else {
        parallel_for(pre_nms_topn, [&](size_t i) {
            unpacked_boxes[0 * pre_nms_topn + i] = p_proposals[5 * i + 0];
            unpacked_boxes[1 * pre_nms_topn + i] = p_proposals[5 * i + 1];
            unpacked_boxes[2 * pre_nms_topn + i] = p_proposals[5 * i + 2];
            unpacked_boxes[3 * pre_nms_topn + i] = p_proposals[5 * i + 3];
        });
    }
}

void nms_cpu(const int num_boxes, int is_dead[],
             const float* boxes, int index_out[], std::size_t* const num_out,
             const int base_index, const float nms_thresh, const int max_num_out,
             float coordinates_offset) {
    const int num_proposals = num_boxes;
    std::size_t count = 0;

    const float* x0 = boxes + 0 * num_proposals;
    const float* y0 = boxes + 1 * num_proposals;
    const float* x1 = boxes + 2 * num_proposals;
    const float* y1 = boxes + 3 * num_proposals;

    for (int box = 0; box < num_boxes; ++box) {
        if (is_dead[box])
            continue;

        index_out[count++] = base_index + box;
        if (count == max_num_out)
            break;

        int tail = box + 1;

        for (; tail < num_boxes; ++tail) {
            float res = 0.0f;

            const float x0i = x0[box];
            const float y0i = y0[box];
            const float x1i = x1[box];
            const float y1i = y1[box];

            const float x0j = x0[tail];
            const float y0j = y0[tail];
            const float x1j = x1[tail];
            const float y1j = y1[tail];

            if (x0i <= x1j && y0i <= y1j && x0j <= x1i && y0j <= y1i) {
                // overlapped region (= box)
                const float x0 = std::max<float>(x0i, x0j);
                const float y0 = std::max<float>(y0i, y0j);
                const float x1 = std::min<float>(x1i, x1j);
                const float y1 = std::min<float>(y1i, y1j);

                // intersection area
                const float width  = std::max<float>(0.0f,  x1 - x0 + coordinates_offset);
                const float height = std::max<float>(0.0f,  y1 - y0 + coordinates_offset);
                const float area   = width * height;

                // area of A, B
                const float A_area = (x1i - x0i + coordinates_offset) * (y1i - y0i + coordinates_offset);
                const float B_area = (x1j - x0j + coordinates_offset) * (y1j - y0j + coordinates_offset);

                // IoU
                res = area / (A_area + B_area - area);
            }

            if (nms_thresh < res)
                is_dead[tail] = 1;
        }
    }

    *num_out = count;
}

void retrieve_rois_cpu(const int num_rois, const int item_index,
                              const int num_proposals,
                              const float* proposals, const int roi_indices[],
                              float* rois, int post_nms_topn_,
                              bool normalize, float img_h, float img_w, bool clip_after_nms, float* probs) {
    const float *src_x0 = proposals + 0 * num_proposals;
    const float *src_y0 = proposals + 1 * num_proposals;
    const float *src_x1 = proposals + 2 * num_proposals;
    const float *src_y1 = proposals + 3 * num_proposals;
    const float *src_probs = proposals + 4 * num_proposals;

    parallel_for(num_rois, [&](size_t roi) {
        int index = roi_indices[roi];

        float x0 = src_x0[index];
        float y0 = src_y0[index];
        float x1 = src_x1[index];
        float y1 = src_y1[index];

        if (clip_after_nms) {
            x0 = std::max<float>(0.0f, std::min<float>(x0, img_w));
            y0 = std::max<float>(0.0f, std::min<float>(y0, img_h));
            x1 = std::max<float>(0.0f, std::min<float>(x1, img_w));
            y1 = std::max<float>(0.0f, std::min<float>(y1, img_h));
        }

        if (normalize) {
            x0 /= img_w;
            y0 /= img_h;
            x1 /= img_w;
            y1 /= img_h;
        }

        rois[roi * 5 + 0] = static_cast<float>(item_index);
        rois[roi * 5 + 1] = x0;
        rois[roi * 5 + 2] = y0;
        rois[roi * 5 + 3] = x1;
        rois[roi * 5 + 4] = y1;

        if (probs)
            probs[roi] = src_probs[index];
    });

    if (num_rois < post_nms_topn_) {
        for (int i = 5 * num_rois; i < 5 * post_nms_topn_; i++) {
            rois[i] = 0.f;
        }

        // marker at end of boxes list
        rois[num_rois * 5 + 0] = -1;
    }
}

} // namespace node
} // namespace intel_cpu
} // namespace ov
