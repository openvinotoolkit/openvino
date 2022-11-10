// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_refine_anchors_kernel.hpp"

namespace ov {
namespace intel_cpu {

template <x64::cpu_isa_t isa>
void jit_refine_anchors_kernel_fp32<isa>::generate() {
    this->preamble();

    /*
     * @see
        for (int anchor = 0; anchor < anchors_num; ++anchor) {
            int a_idx = anchor_idx(h, w, anchor, 0);
            float x0 = anchors[a_idx + 0];
            float y0 = anchors[a_idx + 1];
            float x1 = anchors[a_idx + 2];
            float y1 = anchors[a_idx + 3];

            const float dx = deltas[delta_idx(anchor, 0, h, w)];
            const float dy = deltas[delta_idx(anchor, 1, h, w)];
            const float d_log_w = deltas[delta_idx(anchor, 2, h, w)];
            const float d_log_h = deltas[delta_idx(anchor, 3, h, w)];

            const float score = scores[score_idx(anchor, 0, h, w)];

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

            int p_idx = proposal_idx(h, w, anchor, 0);
            proposals[p_idx + 0] = x0;
            proposals[p_idx + 1] = y0;
            proposals[p_idx + 2] = x1;
            proposals[p_idx + 3] = y1;
            proposals[p_idx + 4] = score;
            proposals[p_idx + 5] = (min_box_W <= box_w) * (min_box_H <= box_h) * 1.0;
        }
     */



//    const float score = scores[score_idx(anchor, 0, h, w)];

    /*
    // width & height of box
    const float ww = x1 - x0 + coordinates_offset;
    const float hh = y1 - y0 + coordinates_offset;
     */
    // const float ww = x1 - x0 + coordinates_offset;
    uni_vaddss(reg_ww, reg_x0, reg_coordinates_offset);
    uni_vsubps(reg_ww, reg_ww, reg_x1);
    // const float hh = y1 - y0 + coordinates_offset;
    uni_vaddss(reg_hh, reg_y0, reg_coordinates_offset);
    uni_vsubps(reg_hh, reg_hh, reg_y1);

    /* center location of box
        const float ctr_x = x0 + 0.5f * ww;
        const float ctr_y = y0 + 0.5f * hh;
     */
    // const float ctr_x = x0 + 0.5f * ww;
    uni_vmulss(reg_ctr_x, reg_ww, reg_scale_0_5);
    uni_vaddps(reg_ctr_x, reg_ctr_x, reg_x0);
    // const float ctr_y = y0 + 0.5f * hh;
    uni_vmulss(reg_ctr_y, reg_hh, reg_scale_0_5);
    uni_vaddps(reg_ctr_y, reg_ctr_y, reg_y0);

    /* new center location according to deltas (dx, dy)
        const float pred_ctr_x = dx * ww + ctr_x;
        const float pred_ctr_y = dy * hh + ctr_y;
     */
    // const float pred_ctr_x = dx * ww + ctr_x;
    uni_vmulps(reg_pred_ctr_x, reg_dx, reg_ww);
    uni_vaddps(reg_pred_ctr_x, reg_pred_ctr_x, reg_ctr_x);
    // const float pred_ctr_y = dy * hh + ctr_y;
    uni_vmulps(reg_pred_ctr_y, reg_dy, reg_hh);
    uni_vaddps(reg_pred_ctr_y, reg_pred_ctr_y, reg_ctr_y);

    /* new width & height according to deltas d(log w), d(log h)
        const float pred_w = std::exp(std::min(d_log_w, max_delta_log_wh)) * ww;
        const float pred_h = std::exp(std::min(d_log_h, max_delta_log_wh)) * hh;
     */
    // const float pred_w = std::exp(std::min(d_log_w, max_delta_log_wh)) * ww;
    uni_vminss(reg_pred_w, reg_d_log_w, reg_max_delta_log_wh);
    uni_expf(reg_pred_w);
    uni_vmulps(reg_pred_w, reg_pred_w, reg_ww);
    // const float pred_h = std::exp(std::min(d_log_h, max_delta_log_wh)) * hh;
    uni_vminss(reg_pred_h, reg_d_log_h, reg_max_delta_log_wh);
    uni_expf(reg_pred_h);
    uni_vmulps(reg_pred_h, reg_pred_h, reg_hh);

    /* update upper-left corner location
        x0 = pred_ctr_x - 0.5f * pred_w;
        y0 = pred_ctr_y - 0.5f * pred_h;
     */
    // x0 = pred_ctr_x - 0.5f * pred_w;
    uni_vmulss(reg_x0, reg_pred_w, reg_scale_0_5);
    uni_vaddps(reg_x0, reg_pred_ctr_x, reg_x0);
    // y0 = pred_ctr_y - 0.5f * pred_h;
    uni_vmulss(reg_y0, reg_pred_h, reg_scale_0_5);
    uni_vaddps(reg_y0, reg_pred_ctr_y, reg_y0);

    /* update lower-right corner location
        x1 = pred_ctr_x + 0.5f * pred_w - coordinates_offset;
        y1 = pred_ctr_y + 0.5f * pred_h - coordinates_offset;
     */
    // x1 = pred_ctr_x + 0.5f * pred_w - coordinates_offset;
    uni_vmulss(reg_x1, reg_pred_w, reg_scale_0_5);
    uni_vsubss(reg_x1, reg_x1, reg_coordinates_offset);
    uni_vaddps(reg_x1, reg_pred_ctr_x, reg_x1);
    // y1 = pred_ctr_y + 0.5f * pred_h - coordinates_offset;
    uni_vmulss(reg_y1, reg_pred_h, reg_scale_0_5);
    uni_vsubss(reg_y1, reg_y1, reg_coordinates_offset);
    uni_vaddps(reg_y1, reg_pred_ctr_y, reg_y1);

    sub(reg_img_W, reg_coordinates_offset);
    sub(reg_img_H, reg_coordinates_offset);
    /* adjust new corner locations to be within the image region,
        x0 = std::max<float>(0.0f, std::min<float>(x0, img_W - coordinates_offset));
        y0 = std::max<float>(0.0f, std::min<float>(y0, img_H - coordinates_offset));
     */
    // x0 = std::max<float>(0.0f, std::min<float>(x0, img_W - coordinates_offset));
    uni_vminss(reg_x0, reg_x0, reg_img_W);
    uni_vmaxss(reg_x0, reg_0_0, reg_x0);
    // y0 = std::max<float>(0.0f, std::min<float>(y0, img_H - coordinates_offset));
    uni_vminss(reg_y0, reg_y0, reg_img_H);
    uni_vmaxss(reg_y0, reg_0_0, reg_y0);

    /* adjust new corner locations to be within the image region,
        x1 = std::max<float>(0.0f, std::min<float>(x1, img_W - coordinates_offset));
        y1 = std::max<float>(0.0f, std::min<float>(y1, img_H - coordinates_offset));
     */
    // x1 = std::max<float>(0.0f, std::min<float>(x1, img_W - coordinates_offset));
    uni_vminss(reg_x1, reg_x1, reg_img_W);
    uni_vmaxss(reg_x1, reg_0_0, reg_x1);
    // y1 = std::max<float>(0.0f, std::min<float>(y1, img_H - coordinates_offset));
    uni_vminss(reg_y1, reg_y1, reg_img_H);
    uni_vmaxss(reg_y1, reg_0_0, reg_y1);

    /* recompute new width & height
        const float box_w = x1 - x0 + coordinates_offset;
        const float box_h = y1 - y0 + coordinates_offset;
     */
    // const float box_w = x1 - x0 + coordinates_offset;
    uni_vaddss(reg_box_w, reg_x0, reg_coordinates_offset);
    uni_vaddps(reg_box_w, reg_x1, reg_box_w);
    // const float box_h = y1 - y0 + coordinates_offset;
    uni_vaddss(reg_box_h, reg_y0, reg_scale_0_5);
    uni_vaddps(reg_box_h, reg_y1, reg_box_h);

    this->postamble();

    exp_injector->prepare_table();
}

template struct jit_refine_anchors_kernel_fp32<x64::avx512_core>;
template struct jit_refine_anchors_kernel_fp32<x64::avx2>;
template struct jit_refine_anchors_kernel_fp32<x64::sse41>;

}
}
