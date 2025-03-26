// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "proposal_imp.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#if defined(HAVE_AVX2)
#    include <immintrin.h>
#endif
#include "openvino/core/parallel.hpp"

namespace ov::Extensions::Cpu::XARCH {

static void enumerate_proposals_cpu(const float* bottom4d,
                                    const float* d_anchor4d,
                                    const float* anchors,
                                    float* proposals,
                                    const int num_anchors,
                                    const int bottom_H,
                                    const int bottom_W,
                                    const float img_H,
                                    const float img_W,
                                    const float min_box_H,
                                    const float min_box_W,
                                    const int feat_stride,
                                    const float box_coordinate_scale,
                                    const float box_size_scale,
                                    float coordinates_offset,
                                    bool initial_clip,
                                    bool swap_xy,
                                    bool clip_before_nms) {
    const int bottom_area = bottom_H * bottom_W;

    const float* p_anchors_wm = anchors + 0 * num_anchors;
    const float* p_anchors_hm = anchors + 1 * num_anchors;
    const float* p_anchors_wp = anchors + 2 * num_anchors;
    const float* p_anchors_hp = anchors + 3 * num_anchors;

    parallel_for2d(bottom_H, bottom_W, [&](size_t h, size_t w) {
        const auto x = static_cast<float>((swap_xy ? h : w) * feat_stride);
        const auto y = static_cast<float>((swap_xy ? w : h) * feat_stride);

        const float* p_box = d_anchor4d + h * bottom_W + w;
        const float* p_score = bottom4d + h * bottom_W + w;

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

            p_proposal[5 * anchor + 0] = x0;
            p_proposal[5 * anchor + 1] = y0;
            p_proposal[5 * anchor + 2] = x1;
            p_proposal[5 * anchor + 3] = y1;
            p_proposal[5 * anchor + 4] = (min_box_W <= box_w) * (min_box_H <= box_h) * score;
        }
    });
}

static void unpack_boxes(const float* p_proposals, float* unpacked_boxes, int pre_nms_topn, bool store_prob) {
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

static void nms_cpu(const int num_boxes,
                    int is_dead[],
                    const float* boxes,
                    int index_out[],
                    int* const num_out,
                    const int base_index,
                    const float nms_thresh,
                    const int max_num_out,
                    float coordinates_offset) {
    const int num_proposals = num_boxes;
    int count = 0;

    const float* x0 = boxes + 0 * num_proposals;
    const float* y0 = boxes + 1 * num_proposals;
    const float* x1 = boxes + 2 * num_proposals;
    const float* y1 = boxes + 3 * num_proposals;

    std::memset(is_dead, 0, num_boxes * sizeof(int));

#if defined(HAVE_AVX2)
    __m256 vc_fone = _mm256_set1_ps(coordinates_offset);
    __m256i vc_ione = _mm256_set1_epi32(1);
    __m256 vc_zero = _mm256_set1_ps(0.0f);

    __m256 vc_nms_thresh = _mm256_set1_ps(nms_thresh);
#endif

    for (int box = 0; box < num_boxes; ++box) {
        if (is_dead[box]) {
            continue;
        }

        index_out[count++] = base_index + box;
        if (count == max_num_out) {
            break;
        }

        int tail = box + 1;

#if defined(HAVE_AVX2)
        __m256 vx0i = _mm256_set1_ps(x0[box]);
        __m256 vy0i = _mm256_set1_ps(y0[box]);
        __m256 vx1i = _mm256_set1_ps(x1[box]);
        __m256 vy1i = _mm256_set1_ps(y1[box]);

        __m256 vA_width = _mm256_sub_ps(vx1i, vx0i);
        __m256 vA_height = _mm256_sub_ps(vy1i, vy0i);
        __m256 vA_area = _mm256_mul_ps(_mm256_add_ps(vA_width, vc_fone), _mm256_add_ps(vA_height, vc_fone));

        for (; tail <= num_boxes - 8; tail += 8) {
            __m256i* pdst = reinterpret_cast<__m256i*>(is_dead + tail);
            __m256i vdst = _mm256_loadu_si256(pdst);

            __m256 vx0j = _mm256_loadu_ps(x0 + tail);
            __m256 vy0j = _mm256_loadu_ps(y0 + tail);
            __m256 vx1j = _mm256_loadu_ps(x1 + tail);
            __m256 vy1j = _mm256_loadu_ps(y1 + tail);

            __m256 vx0 = _mm256_max_ps(vx0i, vx0j);
            __m256 vy0 = _mm256_max_ps(vy0i, vy0j);
            __m256 vx1 = _mm256_min_ps(vx1i, vx1j);
            __m256 vy1 = _mm256_min_ps(vy1i, vy1j);

            __m256 vwidth = _mm256_add_ps(_mm256_sub_ps(vx1, vx0), vc_fone);
            __m256 vheight = _mm256_add_ps(_mm256_sub_ps(vy1, vy0), vc_fone);
            __m256 varea = _mm256_mul_ps(_mm256_max_ps(vc_zero, vwidth), _mm256_max_ps(vc_zero, vheight));

            __m256 vB_width = _mm256_sub_ps(vx1j, vx0j);
            __m256 vB_height = _mm256_sub_ps(vy1j, vy0j);
            __m256 vB_area = _mm256_mul_ps(_mm256_add_ps(vB_width, vc_fone), _mm256_add_ps(vB_height, vc_fone));

            __m256 vdivisor = _mm256_sub_ps(_mm256_add_ps(vA_area, vB_area), varea);
            __m256 vintersection_area = _mm256_div_ps(varea, vdivisor);

            __m256 vcmp_0 = _mm256_cmp_ps(vx0i, vx1j, _CMP_LE_OS);
            __m256 vcmp_1 = _mm256_cmp_ps(vy0i, vy1j, _CMP_LE_OS);
            __m256 vcmp_2 = _mm256_cmp_ps(vx0j, vx1i, _CMP_LE_OS);
            __m256 vcmp_3 = _mm256_cmp_ps(vy0j, vy1i, _CMP_LE_OS);
            __m256 vcmp_4 = _mm256_cmp_ps(vc_nms_thresh, vintersection_area, _CMP_LT_OS);

            vcmp_0 = _mm256_and_ps(vcmp_0, vcmp_1);
            vcmp_2 = _mm256_and_ps(vcmp_2, vcmp_3);
            vcmp_4 = _mm256_and_ps(vcmp_4, vcmp_0);
            vcmp_4 = _mm256_and_ps(vcmp_4, vcmp_2);

            _mm256_storeu_si256(pdst, _mm256_blendv_epi8(vdst, vc_ione, _mm256_castps_si256(vcmp_4)));
        }
#endif

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
                const float width = std::max<float>(0.0f, x1 - x0 + coordinates_offset);
                const float height = std::max<float>(0.0f, y1 - y0 + coordinates_offset);
                const float area = width * height;

                // area of A, B
                const float A_area = (x1i - x0i + coordinates_offset) * (y1i - y0i + coordinates_offset);
                const float B_area = (x1j - x0j + coordinates_offset) * (y1j - y0j + coordinates_offset);

                // IoU
                res = area / (A_area + B_area - area);
            }

            if (nms_thresh < res) {
                is_dead[tail] = 1;
            }
        }
    }

    *num_out = count;
}

static void retrieve_rois_cpu(const int num_rois,
                              const int item_index,
                              const int num_proposals,
                              const float* proposals,
                              const int roi_indices[],
                              float* rois,
                              int post_nms_topn_,
                              bool normalize,
                              float img_h,
                              float img_w,
                              bool clip_after_nms,
                              float* probs) {
    const float* src_x0 = proposals + 0 * num_proposals;
    const float* src_y0 = proposals + 1 * num_proposals;
    const float* src_x1 = proposals + 2 * num_proposals;
    const float* src_y1 = proposals + 3 * num_proposals;
    const float* src_probs = proposals + 4 * num_proposals;

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

        if (probs) {
            probs[roi] = src_probs[index];
        }
    });

    if (num_rois < post_nms_topn_) {
        for (int i = 5 * num_rois; i < 5 * post_nms_topn_; i++) {
            rois[i] = 0.f;
        }

        // marker at end of boxes list
        rois[num_rois * 5 + 0] = -1;
    }
}

void proposal_exec(const float* input0,
                   const float* input1,
                   std::vector<size_t> dims0,
                   std::array<float, 4> img_info,
                   const float* anchors,
                   int* roi_indices,
                   float* output0,
                   float* output1,
                   proposal_conf& conf) {
    // Prepare memory
    const float* p_bottom_item = input0;
    const float* p_d_anchor_item = input1;

    float* p_roi_item = output0;
    float* p_prob_item = output1;
    auto store_prob = p_prob_item != nullptr;

    // bottom shape: (2 x num_anchors) x H x W
    const int bottom_H = dims0[2];
    const int bottom_W = dims0[3];

    // input image height & width
    const float img_H = img_info[conf.swap_xy ? 1 : 0];
    const float img_W = img_info[conf.swap_xy ? 0 : 1];

    // scale factor for height & width
    const float scale_H = img_info[2];
    const float scale_W = img_info[3];

    // minimum box width & height
    const float min_box_H = conf.min_size_ * scale_H;
    const float min_box_W = conf.min_size_ * scale_W;

    // number of all proposals = num_anchors * H * W
    const int num_proposals = conf.anchors_shape_0 * bottom_H * bottom_W;

    // number of top-n proposals before NMS
    const int pre_nms_topn = std::min<int>(num_proposals, conf.pre_nms_topn_);

    // number of final RoIs
    int num_rois = 0;

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
    const int unpacked_boxes_buffer_size = store_prob ? 5 * pre_nms_topn : 4 * pre_nms_topn;
    std::vector<float> unpacked_boxes(unpacked_boxes_buffer_size);
    std::vector<int> is_dead(pre_nms_topn);

    // Execute
    int nn = dims0[0];
    for (int n = 0; n < nn; ++n) {
        enumerate_proposals_cpu(p_bottom_item + num_proposals + n * num_proposals * 2,
                                p_d_anchor_item + n * num_proposals * 4,
                                anchors,
                                reinterpret_cast<float*>(&proposals_[0]),
                                conf.anchors_shape_0,
                                bottom_H,
                                bottom_W,
                                img_H,
                                img_W,
                                min_box_H,
                                min_box_W,
                                conf.feat_stride_,
                                conf.box_coordinate_scale_,
                                conf.box_size_scale_,
                                conf.coordinates_offset,
                                conf.initial_clip,
                                conf.swap_xy,
                                conf.clip_before_nms);
        std::partial_sort(proposals_.begin(),
                          proposals_.begin() + pre_nms_topn,
                          proposals_.end(),
                          [](const ProposalBox& struct1, const ProposalBox& struct2) {
                              return (struct1.score > struct2.score);
                          });

        unpack_boxes(reinterpret_cast<float*>(&proposals_[0]), &unpacked_boxes[0], pre_nms_topn, store_prob);
        nms_cpu(pre_nms_topn,
                &is_dead[0],
                &unpacked_boxes[0],
                roi_indices,
                &num_rois,
                0,
                conf.nms_thresh_,
                conf.post_nms_topn_,
                conf.coordinates_offset);

        float* p_probs = store_prob ? p_prob_item + n * conf.post_nms_topn_ : nullptr;
        retrieve_rois_cpu(num_rois,
                          n,
                          pre_nms_topn,
                          &unpacked_boxes[0],
                          roi_indices,
                          p_roi_item + n * conf.post_nms_topn_ * 5,
                          conf.post_nms_topn_,
                          conf.normalize_,
                          img_H,
                          img_W,
                          conf.clip_after_nms,
                          p_probs);
    }
}

}  // namespace ov::Extensions::Cpu::XARCH
