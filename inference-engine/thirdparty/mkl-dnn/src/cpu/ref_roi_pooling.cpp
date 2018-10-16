/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <assert.h>
#include <math.h>
#include <float.h>
#include <algorithm>
#include <mkldnn_types.h>
#include <iostream>

#include "c_types_map.hpp"
#include "type_helpers.hpp"

#include "ref_roi_pooling.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
void ref_roi_pooling_fwd_t<data_type>::execute_forward_generic() {
    int roi_idx = 1;
    int data_idx = 0;

    const memory_desc_wrapper dst_d(conf_.dst_pd());
    memory_desc_wrapper src_data_d = conf_.src_pd(data_idx);
    memory_desc_wrapper src_roi_d = conf_.src_pd(roi_idx);

    if (src_roi_d.dims()[0] < src_data_d.dims()[0]) {
        roi_idx = 0;
        data_idx = 1;

        src_data_d = conf_.src_pd(data_idx);
        src_roi_d = conf_.src_pd(roi_idx);
    }

    auto dst = reinterpret_cast<data_t*>(this->memory(0));
    const data_t* src_data = reinterpret_cast<const data_t*>(this->input_memory(data_idx));
    const data_t* src_roi = reinterpret_cast<const data_t*>(this->input_memory(roi_idx));

    int C = src_data_d.dims()[1];
    int H = src_data_d.dims()[2];
    int W = src_data_d.dims()[3];

    int ROIS = src_roi_d.dims()[0];

    double spatial_scale = conf_.spatialScale();
    int pooled_h = conf_.pooledH();
    int pooled_w = conf_.pooledW();

    for (size_t i = 0; i < dst_d.size() / sizeof(data_t); i++) {
        dst[i] = -FLT_MAX;
    }

    int real_rois = 0;
    for (; real_rois < ROIS; real_rois++) {
        int roi_off;
        if(src_roi_d.ndims() == 4) {
            roi_off = src_roi_d.off(real_rois, 0, 0, 0);
        }
        else {
            roi_off = src_roi_d.off(real_rois, 0);
        }

        const data_t* src_roi_ptr = &src_roi[roi_off];
        int roi_batch_ind = src_roi_ptr[0];
        if (roi_batch_ind == -1) {
            break;
        }
    }
    int n = 0;
    for (; n < real_rois; ++n) {
        int roi_off;
        if(src_roi_d.ndims() == 4) {
            roi_off = src_roi_d.off(n, 0, 0, 0);
        }
        else {
            roi_off = src_roi_d.off(n, 0);
        }

        const data_t* src_roi_ptr = &src_roi[roi_off];
        int roi_batch_ind = src_roi_ptr[0];

        if (conf_.desc()->alg_kind == mkldnn_roi_pooling_max) {
            int roi_start_w = round(src_roi_ptr[1] * spatial_scale);
            int roi_start_h = round(src_roi_ptr[2] * spatial_scale);
            int roi_end_w = round(src_roi_ptr[3] * spatial_scale);
            int roi_end_h = round(src_roi_ptr[4] * spatial_scale);

            int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
            int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);


            for (int c = 0; c < C; ++c) {
                for (int ph = 0; ph < pooled_h; ++ph) {
                    for (int pw = 0; pw < pooled_w; ++pw) {
                        int hstart = (ph * roi_height) / pooled_h;
                        if ((hstart * pooled_h) > (ph * roi_height)) {
                            --hstart;
                        }

                        int wstart = (pw * roi_width) / pooled_w;
                        if ((wstart * pooled_w) > (pw * roi_width)) {
                            --wstart;
                        }

                        int hend = ((ph + 1) * roi_height) / pooled_h;
                        if ((hend * pooled_h) < ((ph + 1) * roi_height)) {
                            ++hend;
                        }

                        int wend = ((pw + 1) * roi_width) / pooled_w;
                        if ((wend * pooled_w) < ((pw + 1) * roi_width)) {
                            ++wend;
                        }

                        hstart = std::min(std::max(hstart + roi_start_h, 0), H);
                        hend = std::min(std::max(hend + roi_start_h, 0), H);
                        wstart = std::min(std::max(wstart + roi_start_w, 0), W);
                        wend = std::min(std::max(wend + roi_start_w, 0), W);

                        bool is_empty = (hend <= hstart) || (wend <= wstart);

                        const int pool_index = dst_d.off(n, c, ph, pw);

                        if (is_empty) {
                            dst[pool_index] = 0;
                        }

                        for (int h = hstart; h < hend; ++h) {
                            for (int w = wstart; w < wend; ++w) {
                                data_t batch_data = src_data[src_data_d.off(roi_batch_ind, c, h, w)];

                                if (batch_data > dst[pool_index]) {
                                    dst[pool_index] = batch_data;
                                }
                            }
                        }
                    }
                }
            }
        } else if (conf_.desc()->alg_kind == mkldnn_roi_pooling_bilinear) {
            float roi_start_w_ = src_roi_ptr[1];
            float roi_start_h_ = src_roi_ptr[2];
            float roi_end_w_   = src_roi_ptr[3];
            float roi_end_h_   = src_roi_ptr[4];

            float height_scale = (roi_end_h_ - roi_start_h_) * (H - 1) / (pooled_h - 1);
            float width_scale  = (roi_end_w_ - roi_start_w_) * (W - 1) / (pooled_w - 1);

            for (int c = 0; c < C; ++c) {
                for (int ph = 0; ph < pooled_h; ++ph) {
                    for (int pw = 0; pw < pooled_w; ++pw) {
                        float in_y = (ph * height_scale + roi_start_h_ * (H - 1));
                        float in_x = (pw * width_scale  + roi_start_w_ * (W - 1));

                        if (in_y < 0 || in_y > H - 1 || in_x < 0 || in_x > W - 1) {
                            dst[dst_d.off(n, c, ph, pw)] = 0;
                        } else {
                            int top_y_index    = static_cast<int>(floorf(in_y));
                            int bottom_y_index = static_cast<int>(ceilf(in_y));
                            int left_x_index   = static_cast<int>(floorf(in_x));
                            int right_x_index  = static_cast<int>(ceilf(in_x));

                            if (right_x_index > W - 1)
                                right_x_index = W - 1;

                            if (bottom_y_index > H - 1)
                                bottom_y_index = H - 1;

                            const float top_left     = src_data[src_data_d.off(roi_batch_ind, c, top_y_index, left_x_index)];
                            const float top_right    = src_data[src_data_d.off(roi_batch_ind, c, top_y_index, right_x_index)];
                            const float bottom_left  = src_data[src_data_d.off(roi_batch_ind, c, bottom_y_index, left_x_index)];
                            const float bottom_right = src_data[src_data_d.off(roi_batch_ind, c, bottom_y_index, right_x_index)];

                            const float top    = top_left + (top_right - top_left) * (in_x - left_x_index);
                            const float bottom = bottom_left + (bottom_right - bottom_left) * (in_x - left_x_index);

                            dst[dst_d.off(n, c, ph, pw)] = top + (bottom - top) * (in_y - top_y_index);
                        }
                    }
                }
            }
        }
    }

    for (; n < ROIS; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int ph = 0; ph < pooled_h; ++ph) {
                for (int pw = 0; pw < pooled_w; ++pw) {
                    dst[dst_d.off(n, c, ph, pw)] = 0;
                }
            }
        }
    }
}

template struct ref_roi_pooling_fwd_t<data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
