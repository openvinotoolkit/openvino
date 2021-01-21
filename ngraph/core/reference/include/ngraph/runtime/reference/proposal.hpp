//*****************************************************************************
// Copyright 2021 Intel Corporation
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
#include "ngraph/op/proposal.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace details
            {
                template <typename T>
                struct ProposalBox
                {
                    T x0;
                    T y0;
                    T x1;
                    T y1;
                    T score;
                };

                static std::vector<float> generate_anchors(const op::ProposalAttrs& attrs,
                                                           std::vector<float>& anchors)
                {
                    // Framework specific parameters
                    auto coordinates_offset = attrs.framework == "tensorflow" ? 0.0f : 1.0f;
                    auto round_ratios = attrs.framework == "tensorflow" ? false : true;
                    auto shift_anchors = attrs.framework == "tensorflow" ? true : false;

                    auto base_size = attrs.base_size;
                    auto num_ratios = attrs.ratio.size();
                    auto ratios = attrs.ratio.data();
                    auto num_scales = attrs.scale.size();
                    auto scales = attrs.scale.data();
                    auto anchors_ptr = anchors.data();

                    // base box's width & height & center location
                    const float base_area = static_cast<float>(base_size * base_size);
                    const float half_base_size = base_size * 0.5f;
                    const float center = 0.5f * (base_size - coordinates_offset);

                    // enumerate all transformed boxes
                    for (int ratio = 0; ratio < num_ratios; ++ratio)
                    {
                        // transformed width & height for given ratio factors
                        float ratio_w;
                        float ratio_h;
                        if (round_ratios)
                        {
                            ratio_w = std::roundf(std::sqrt(base_area / ratios[ratio]));
                            ratio_h = std::roundf(ratio_w * ratios[ratio]);
                        }
                        else
                        {
                            ratio_w = std::sqrt(base_area / ratios[ratio]);
                            ratio_h = ratio_w * ratios[ratio];
                        }

                        float* const p_anchors_wm =
                            anchors_ptr + 0 * num_ratios * num_scales + ratio * num_scales;
                        float* const p_anchors_hm =
                            anchors_ptr + 1 * num_ratios * num_scales + ratio * num_scales;
                        float* const p_anchors_wp =
                            anchors_ptr + 2 * num_ratios * num_scales + ratio * num_scales;
                        float* const p_anchors_hp =
                            anchors_ptr + 3 * num_ratios * num_scales + ratio * num_scales;

                        for (int scale = 0; scale < num_scales; ++scale)
                        {
                            // transformed width & height for given scale factors
                            const float scale_w =
                                0.5f * (ratio_w * scales[scale] - coordinates_offset);
                            const float scale_h =
                                0.5f * (ratio_h * scales[scale] - coordinates_offset);

                            // (x1, y1, x2, y2) for transformed box
                            p_anchors_wm[scale] = center - scale_w;
                            p_anchors_hm[scale] = center - scale_h;
                            p_anchors_wp[scale] = center + scale_w;
                            p_anchors_hp[scale] = center + scale_h;

                            if (shift_anchors)
                            {
                                p_anchors_wm[scale] -= half_base_size;
                                p_anchors_hm[scale] -= half_base_size;
                                p_anchors_wp[scale] -= half_base_size;
                                p_anchors_hp[scale] -= half_base_size;
                            }
                        }
                    }
                    return anchors;
                }

                template <typename T>
                static void enumerate_proposals(const T* bottom4d,
                                                const T* d_anchor4d,
                                                const float* anchors,
                                                std::vector<ProposalBox<T>>& proposals,
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
                                                bool clip_before_nms)
                {
                    const int bottom_area = bottom_H * bottom_W;

                    const float* p_anchors_wm = anchors + 0 * num_anchors;
                    const float* p_anchors_hm = anchors + 1 * num_anchors;
                    const float* p_anchors_wp = anchors + 2 * num_anchors;
                    const float* p_anchors_hp = anchors + 3 * num_anchors;

                    for (size_t h = 0; h < bottom_H; ++h)
                    {
                        for (size_t w = 0; w < bottom_W; ++w)
                        {
                            const float x = static_cast<float>((swap_xy ? h : w) * feat_stride);
                            const float y = static_cast<float>((swap_xy ? w : h) * feat_stride);

                            const T* p_box = d_anchor4d + h * bottom_W + w;
                            const T* p_score = bottom4d + h * bottom_W + w;

                            size_t proposal_off = (h * bottom_W + w) * num_anchors;

                            for (int anchor = 0; anchor < num_anchors; ++anchor)
                            {
                                const T dx = p_box[(anchor * 4 + 0) * bottom_area] /
                                             static_cast<T>(box_coordinate_scale);
                                const T dy = p_box[(anchor * 4 + 1) * bottom_area] /
                                             static_cast<T>(box_coordinate_scale);

                                const T d_log_w = p_box[(anchor * 4 + 2) * bottom_area] /
                                                  static_cast<T>(box_size_scale);
                                const T d_log_h = p_box[(anchor * 4 + 3) * bottom_area] /
                                                  static_cast<T>(box_size_scale);

                                const T score = p_score[anchor * bottom_area];

                                float x0 = x + p_anchors_wm[anchor];
                                float y0 = y + p_anchors_hm[anchor];
                                float x1 = x + p_anchors_wp[anchor];
                                float y1 = y + p_anchors_hp[anchor];

                                if (initial_clip)
                                {
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
                                const T pred_ctr_x =
                                    dx * static_cast<T>(ww) + static_cast<T>(ctr_x);
                                const T pred_ctr_y =
                                    dy * static_cast<T>(hh) + static_cast<T>(ctr_y);
                                // new width & height according to gradient d(log w), d(log h)
                                const T pred_w = std::exp(d_log_w) * static_cast<T>(ww);
                                const T pred_h = std::exp(d_log_h) * static_cast<T>(hh);

                                // update upper-left corner location
                                x0 = pred_ctr_x - 0.5f * pred_w;
                                y0 = pred_ctr_y - 0.5f * pred_h;
                                // update lower-right corner location
                                x1 = pred_ctr_x + 0.5f * pred_w;
                                y1 = pred_ctr_y + 0.5f * pred_h;

                                // adjust new corner locations to be within the image region,
                                if (clip_before_nms)
                                {
                                    x0 = std::max<float>(
                                        0.0f, std::min<float>(x0, img_W - coordinates_offset));
                                    y0 = std::max<float>(
                                        0.0f, std::min<float>(y0, img_H - coordinates_offset));
                                    x1 = std::max<float>(
                                        0.0f, std::min<float>(x1, img_W - coordinates_offset));
                                    y1 = std::max<float>(
                                        0.0f, std::min<float>(y1, img_H - coordinates_offset));
                                }

                                // recompute new width & height
                                const float box_w = x1 - x0 + coordinates_offset;
                                const float box_h = y1 - y0 + coordinates_offset;

                                proposals[proposal_off + anchor].x0 = static_cast<T>(x0);
                                proposals[proposal_off + anchor].y0 = static_cast<T>(y0);
                                proposals[proposal_off + anchor].x1 = static_cast<T>(x1);
                                proposals[proposal_off + anchor].y1 = static_cast<T>(y1);
                                proposals[proposal_off + anchor].score =
                                    static_cast<T>((min_box_W <= box_w) * (min_box_H <= box_h)) *
                                    score;
                            }
                        }
                    }
                }

                template <typename T>
                static void nms(const int num_boxes,
                                std::vector<int>& is_dead,
                                // it was
                                // const std::vector<T>& boxes,
                                const std::vector<ProposalBox<T>>& proposals,
                                std::vector<int>& index_out,
                                int* const num_out,
                                const int base_index,
                                const float nms_thresh,
                                const int max_num_out,
                                T coordinates_offset)
                {
                    int count = 0;
                    // was
                    // std::memset(is_dead, 0, num_boxes * sizeof(int));
                    std::fill(is_dead.begin(), is_dead.begin() + num_boxes, 0);
                    for (int box = 0; box < num_boxes; ++box)
                    {
                        if (is_dead[box])
                            continue;

                        index_out[count++] = base_index + box;
                        if (count == max_num_out)
                            break;

                        int tail = box + 1;
                        for (; tail < num_boxes; ++tail)
                        {
                            float res = 0.0f;

                            const T x0i = proposals[box].x0;
                            const T y0i = proposals[box].y0;
                            const T x1i = proposals[box].x1;
                            const T y1i = proposals[box].y1;

                            const T x0j = proposals[tail].x0;
                            const T y0j = proposals[tail].y0;
                            const T x1j = proposals[tail].x1;
                            const T y1j = proposals[tail].y1;

                            if (x0i <= x1j && y0i <= y1j && x0j <= x1i && y0j <= y1i)
                            {
                                // overlapped region (= box)
                                const T x0 = std::max<T>(x0i, x0j);
                                const T y0 = std::max<T>(y0i, y0j);
                                const T x1 = std::min<T>(x1i, x1j);
                                const T y1 = std::min<T>(y1i, y1j);
                                // intersection area
                                const T width = std::max<T>(0.0f, x1 - x0 + coordinates_offset);
                                const T height = std::max<T>(0.0f, y1 - y0 + coordinates_offset);
                                const T area = width * height;
                                // area of A, B
                                const T A_area = (x1i - x0i + coordinates_offset) *
                                                 (y1i - y0i + coordinates_offset);
                                const T B_area = (x1j - x0j + coordinates_offset) *
                                                 (y1j - y0j + coordinates_offset);

                                // IoU
                                res = static_cast<float>(area / (A_area + B_area - area));
                            }
                            if (nms_thresh < res)
                                is_dead[tail] = 1;
                        }
                    }
                    *num_out = count;
                }

                template <typename T>
                static void retrieve_rois(const int num_rois,
                                          const int item_index,
                                          const int num_proposals,
                                          const std::vector<ProposalBox<T>>& proposals,
                                          // was
                                          // const int roi_indices[],
                                          const std::vector<int>& roi_indices,
                                          T* rois,
                                          int post_nms_topn_,
                                          bool normalize,
                                          float img_h,
                                          float img_w,
                                          bool clip_after_nms)
                {
                    for (size_t roi = 0; roi < num_rois; ++roi)
                    {
                        int index = roi_indices[roi];
                        T x0 = proposals[index].x0;
                        T y0 = proposals[index].y0;
                        T x1 = proposals[index].x1;
                        T y1 = proposals[index].y1;

                        if (clip_after_nms)
                        {
                            x0 = std::max<T>(0.0f, std::min<T>(x0, static_cast<T>(img_w)));
                            y0 = std::max<T>(0.0f, std::min<T>(y0, static_cast<T>(img_h)));
                            x1 = std::max<T>(0.0f, std::min<T>(x1, static_cast<T>(img_w)));
                            y1 = std::max<T>(0.0f, std::min<T>(y1, static_cast<T>(img_h)));
                        }

                        if (normalize)
                        {
                            x0 /= static_cast<T>(img_w);
                            y0 /= static_cast<T>(img_h);
                            x1 /= static_cast<T>(img_w);
                            y1 /= static_cast<T>(img_h);
                        }

                        rois[roi * 5 + 0] = static_cast<T>(item_index);
                        rois[roi * 5 + 1] = x0;
                        rois[roi * 5 + 2] = y0;
                        rois[roi * 5 + 3] = x1;
                        rois[roi * 5 + 4] = y1;
                    }

                    if (num_rois < post_nms_topn_)
                    {
                        for (int i = 5 * num_rois; i < 5 * post_nms_topn_; i++)
                        {
                            rois[i] = static_cast<T>(0.f);
                        }
                    }
                }
            } // namespace details

            template <typename T>
            void proposal(const T* class_probs,
                          const T* bbox_deltas,
                          const T* image_shape,
                          T* output,
                          const Shape& class_probs_shape,
                          const Shape& bbox_deltas_shape,
                          const Shape& image_shape_shape,
                          const Shape& output_shape,
                          const op::ProposalAttrs& attrs)
            {
                const T* p_bottom_item = class_probs;
                const T* p_d_anchor_item = bbox_deltas;
                T* p_roi_item = output;
                // bottom shape (batch_size * (2 * num_anchors) * H * W)
                const int bottom_H = class_probs_shape[2];
                const int bottom_W = class_probs_shape[3];
                // input image height and width
                const T img_H = image_shape[0];
                const T img_W = image_shape[1];
                // scale factor for H and W, depends on shape of image_shape
                // can be split into H and W {image_height, image_width, scale_height, scale_width}
                // or be the same for both   {image_height, image_width, scale_height_and_width}
                const T scale_H = image_shape[2];
                const T scale_W = (image_shape_shape.size() < 4 ? scale_H : image_shape[3]);
                const T min_box_H = attrs.min_size * scale_H;
                const T min_box_W = attrs.min_size * scale_W;
                // get number of proposals
                // class_probs shape is {batch_size, anchor_count*2, bottom_H, bottom_W}
                const int anchor_count = class_probs_shape[1] / 2;
                const int num_proposals = anchor_count * bottom_H * bottom_W;
                // final RoI count
                int num_rois = 0;
                std::vector<details::ProposalBox<T>> proposals(num_proposals);
                size_t pre_nms_topn =
                    num_proposals < attrs.pre_nms_topn ? num_proposals : attrs.pre_nms_topn;
                std::vector<int> is_dead(pre_nms_topn);
                std::vector<int> roi_indices(attrs.post_nms_topn);

                std::vector<float> anchors(4 * anchor_count);
                details::generate_anchors(attrs, anchors);

                int batch_num = class_probs_shape[0];
                float coordinates_offset = attrs.framework == "tensorflow" ? 0.0f : 1.0f;
                bool initial_clip = attrs.framework == "tensorflow" ? true : false;
                bool swap_xy = attrs.framework == "tensorflow" ? true : false;

                for (int batch_idx = 0; batch_idx < batch_num; ++batch_idx)
                {
                    details::enumerate_proposals(p_bottom_item + num_proposals +
                                                     batch_idx * num_proposals * 2,
                                                 p_d_anchor_item + batch_idx * num_proposals * 4,
                                                 anchors.data(),
                                                 proposals,
                                                 anchor_count,
                                                 bottom_H,
                                                 bottom_W,
                                                 img_H,
                                                 img_W,
                                                 min_box_H,
                                                 min_box_W,
                                                 attrs.feat_stride,
                                                 attrs.box_coordinate_scale,
                                                 attrs.box_size_scale,
                                                 coordinates_offset,
                                                 initial_clip,
                                                 swap_xy,
                                                 attrs.clip_before_nms);

                    std::partial_sort(proposals.begin(),
                                      proposals.begin() + pre_nms_topn,
                                      proposals.end(),
                                      [](const details::ProposalBox<T>& box1,
                                         const details::ProposalBox<T>& box2) {
                                          return (box1.score > box2.score);
                                      });
                    details::nms(pre_nms_topn,
                                 is_dead,
                                 proposals,
                                 roi_indices,
                                 &num_rois,
                                 0,
                                 attrs.nms_thresh,
                                 attrs.post_nms_topn,
                                 static_cast<T>(coordinates_offset));

                    details::retrieve_rois(num_rois,
                                           batch_idx,
                                           pre_nms_topn,
                                           proposals,
                                           roi_indices,
                                           p_roi_item + batch_idx * attrs.post_nms_topn * 5,
                                           attrs.post_nms_topn,
                                           attrs.normalize,
                                           img_H,
                                           img_W,
                                           attrs.clip_after_nms);
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
