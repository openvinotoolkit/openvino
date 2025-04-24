// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/generate_proposal.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <utility>

#include "openvino/core/shape.hpp"
#include "openvino/op/generate_proposals.hpp"

struct sProposalBox {
    float x0;
    float y0;
    float x1;
    float y1;
    float score;
    float keep;
};

static void generate_proposal_refine_anchors(const std::vector<float>& deltas,
                                             const std::vector<float>& scores,
                                             const std::vector<float>& anchors,
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
            // base index for anchors: anchors shape is [bottom_H, bottom_W, anchors_num, 4]
            int64_t a_idx = (h * bottom_W + w) * anchors_num * 4;
            // base index for proposals: proposals shape is [bottom_H, bottom_W, anchors_num, 6]
            int64_t p_idx = (h * bottom_W + w) * anchors_num * 6;
            // base index for scores: scores shape is [anchors_num, bottom_H, bottom_W]
            int64_t sc_idx = h * bottom_W + w;
            // base index for deltas: anchors shape is [anchors_num, 4, bottom_H, bottom_W]
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
                proposals[p_idx + 4] = score;
                proposals[p_idx + 5] = (min_box_W <= box_w) * (min_box_H <= box_h) * 1.0f;

                // update index for next anchor iter
                a_idx += 4;  // anchors shape is [bottom_H, bottom_W, anchors_num, 4], so add 4 for next anchor iter
                p_idx += 6;  // proposals shape is [bottom_H, bottom_W, anchors_num, 6], so add 6 for next anchor iter
                sc_idx += bottom_area;  // scores shape is [anchors_num, bottom_H, bottom_W], so add bottom_H * bottom_W
                                        // for next anchor iter
                d_idx += 4 * bottom_area;  // deltas shape is [anchors_num, 4, bottom_H, bottom_W], so add 4 * bottom_H
                                           // * bottom_W for next anchor iter
            }
        }
    }
}

static void generate_proposal_unpack_boxes(const float* p_proposals,
                                           float* unpacked_boxes,
                                           int64_t* is_dead,
                                           int64_t pre_nms_topn) {
    for (int64_t i = 0; i < pre_nms_topn; ++i) {
        unpacked_boxes[0 * pre_nms_topn + i] = p_proposals[6 * i + 0];
        unpacked_boxes[1 * pre_nms_topn + i] = p_proposals[6 * i + 1];
        unpacked_boxes[2 * pre_nms_topn + i] = p_proposals[6 * i + 2];
        unpacked_boxes[3 * pre_nms_topn + i] = p_proposals[6 * i + 3];
        unpacked_boxes[4 * pre_nms_topn + i] = p_proposals[6 * i + 4];
        is_dead[i] = (p_proposals[6 * i + 5] == 1.0) ? 0 : 1;
    }
}

static void generate_proposal_nms_cpu(const int64_t num_boxes,
                                      int64_t is_dead[],
                                      const float* boxes,
                                      int64_t index_out[],
                                      int64_t& num_out,
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

            if ((nms_thresh < intersection_area) && (x0i <= x1j) && (y0i <= y1j) && (x0j <= x1i) && (y0j <= y1i))
                is_dead[tail] = 1;
        }
    }
    num_out = count;
}

static void generate_proposal_fill_output_blobs(const float* proposals,
                                                const int64_t* roi_indices,
                                                std::vector<float>& rois,
                                                std::vector<float>& scores,
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
        rois.push_back(src_x0[index]);
        rois.push_back(src_y0[index]);
        rois.push_back(src_x1[index]);
        rois.push_back(src_y1[index]);
        scores.push_back(src_score[index]);
    }
}

namespace ov {
namespace reference {

static void generate_proposals_single_image(const std::vector<float>& im_info,
                                            const std::vector<float>& anchors,
                                            const std::vector<float>& deltas,
                                            const std::vector<float>& scores,
                                            const op::v9::GenerateProposals::Attributes& attrs,
                                            const Shape& im_info_shape,
                                            const Shape& anchors_shape,
                                            const Shape& deltas_shape,
                                            const Shape& scores_shape,
                                            std::vector<float>& output_rois,
                                            std::vector<float>& output_scores,
                                            int64_t& num_rois) {
    const int64_t anchors_num = static_cast<int64_t>(scores_shape[1]);

    // bottom shape: batch x (num_anchors) x H x W
    const int64_t bottom_H = static_cast<int64_t>(deltas_shape[2]);
    const int64_t bottom_W = static_cast<int64_t>(deltas_shape[3]);

    // input image height & width
    const float img_H = im_info[0];
    const float img_W = im_info[1];

    // scale factor for height & width
    float scale_h = 1.0;
    float scale_w = 1.0;
    if (im_info.size() == 3) {
        scale_h = im_info[2];
        scale_w = im_info[2];
    } else if (im_info.size() == 4) {
        scale_h = im_info[2];
        scale_w = im_info[3];
    }

    // minimum box width & height
    const float min_box_H = attrs.min_size * scale_h;
    const float min_box_W = attrs.min_size * scale_w;

    const float nms_thresh = attrs.nms_threshold;
    const int64_t post_nms_topn = attrs.post_nms_count;

    // number of all proposals = num_anchors * H * W
    const int64_t num_proposals = anchors_num * bottom_H * bottom_W;

    // number of top-n proposals before NMS
    const int64_t pre_nms_topn = std::min(num_proposals, attrs.pre_nms_count);

    // bbox normalized flag
    const float coordinates_offset = attrs.normalized ? 0.f : 1.f;

    std::vector<sProposalBox> proposals(num_proposals);
    std::vector<float> unpacked_boxes(5 * pre_nms_topn);
    std::vector<int64_t> is_dead(pre_nms_topn);

    generate_proposal_refine_anchors(deltas,
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
                                     coordinates_offset);
    std::partial_sort(proposals.begin(),
                      proposals.begin() + pre_nms_topn,
                      proposals.end(),
                      [](const sProposalBox& struct1, const sProposalBox& struct2) {
                          return (struct1.score > struct2.score);
                      });

    generate_proposal_unpack_boxes(reinterpret_cast<float*>(proposals.data()),
                                   unpacked_boxes.data(),
                                   is_dead.data(),
                                   pre_nms_topn);

    std::vector<int64_t> roi_indices(post_nms_topn);

    generate_proposal_nms_cpu(pre_nms_topn,
                              is_dead.data(),
                              unpacked_boxes.data(),
                              roi_indices.data(),
                              num_rois,
                              0,
                              nms_thresh,
                              post_nms_topn,
                              coordinates_offset);
    generate_proposal_fill_output_blobs(unpacked_boxes.data(),
                                        roi_indices.data(),
                                        output_rois,
                                        output_scores,
                                        pre_nms_topn,
                                        num_rois,
                                        post_nms_topn);
}

void generate_proposals(const std::vector<float>& im_info,
                        const std::vector<float>& anchors,
                        const std::vector<float>& deltas,
                        const std::vector<float>& scores,
                        const op::v9::GenerateProposals::Attributes& attrs,
                        const Shape& im_info_shape,
                        const Shape& anchors_shape,
                        const Shape& deltas_shape,
                        const Shape& scores_shape,
                        std::vector<float>& output_rois,
                        std::vector<float>& output_scores,
                        std::vector<int64_t>& num_rois) {
    const auto im_info_size =
        std::accumulate(im_info_shape.begin() + 1, im_info_shape.end(), size_t(1), std::multiplies<size_t>());
    const auto deltas_size =
        std::accumulate(deltas_shape.begin() + 1, deltas_shape.end(), size_t(1), std::multiplies<size_t>());
    const auto scores_size =
        std::accumulate(scores_shape.begin() + 1, scores_shape.end(), size_t(1), std::multiplies<size_t>());
    for (size_t i = 0; i < im_info_shape[0]; i++) {
        std::vector<float> cur_im_info(im_info.begin() + i * im_info_size,
                                       im_info.begin() + i * im_info_size + im_info_size);
        std::vector<float> cur_deltas(deltas.begin() + i * deltas_size, deltas.begin() + i * deltas_size + deltas_size);
        std::vector<float> cur_scores(scores.begin() + i * scores_size, scores.begin() + i * scores_size + scores_size);
        std::vector<float> output_roi;
        std::vector<float> output_score;
        int64_t num_roi;
        generate_proposals_single_image(cur_im_info,
                                        anchors,
                                        cur_deltas,
                                        cur_scores,
                                        attrs,
                                        im_info_shape,
                                        anchors_shape,
                                        deltas_shape,
                                        scores_shape,
                                        output_roi,
                                        output_score,
                                        num_roi);
        output_rois.insert(output_rois.end(), output_roi.begin(), output_roi.end());
        output_scores.insert(output_scores.end(), output_score.begin(), output_score.end());
        num_rois.push_back(num_roi);
    }
}

void generate_proposals_postprocessing(void* prois,
                                       void* pscores,
                                       void* proi_num,
                                       const element::Type& output_type,
                                       const element::Type& roi_num_type,
                                       const std::vector<float>& output_rois,
                                       const std::vector<float>& output_scores,
                                       const std::vector<int64_t>& num_rois,
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
                       "GenerateProposals operation"
                       " supports only fp32, fp16, or bf16 data.");
    }

    for (size_t i = 0; i < num_rois.size(); i++) {
        switch (roi_num_type) {
        case element::Type_t::i32: {
            int32_t* roi_num_ptr = reinterpret_cast<int32_t*>(proi_num);
            roi_num_ptr[i] = static_cast<int32_t>(num_rois[i]);
        } break;
        case element::Type_t::i64: {
            int64_t* roi_num_ptr = reinterpret_cast<int64_t*>(proi_num);
            roi_num_ptr[i] = static_cast<int64_t>(num_rois[i]);
        } break;
        default:;
            OPENVINO_THROW("Unsupported data type on output port 3: "
                           " supports only int32 or int64.");
        }
    }
}
}  // namespace reference
}  // namespace ov
