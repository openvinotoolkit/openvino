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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "jit_uni_roi_pooling.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "nstl.hpp"
#include "mkldnn_thread.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
void jit_uni_roi_pooling_fwd_t<isa>::execute_forward() const {
    auto src_data = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto src_roi = reinterpret_cast<const data_t*>(this->input_memory(1));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper src_d(pd()->src_pd(0));
    const memory_desc_wrapper src_roi_d(pd()->src_pd(1));
    const memory_desc_wrapper dst_d(pd()->dst_pd());

    const auto &jpp = pd()->jpp_;

    int cb_work = utils::div_up(jpp.nb_c, jpp.nb_c_blocking);
    int MB = jpp.mb;

    int real_rois = 0;
    for (; real_rois < MB; real_rois++) {
        int roi_off;
        if (src_roi_d.ndims() == 4) {
            roi_off = src_roi_d.off(real_rois, 0, 0, 0);
        } else {
            roi_off = src_roi_d.off(real_rois, 0);
        }

        const data_t *src_roi_ptr = &src_roi[roi_off];
        int roi_batch_ind = src_roi_ptr[0];
        if (roi_batch_ind == -1) {
            break;
        }
    }

    const int work_amount = MB * cb_work * jpp.oh * jpp.ow;

    auto ker = [&](const int ithr, const int nthr) {
        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        int n{0}, cbb{0}, oh{0}, ow{0};
        utils::nd_iterator_init(start, n, MB, cbb, cb_work, oh, jpp.oh, ow, jpp.ow);

        for (int iwork = start; iwork < end; iwork++) {
            auto arg = jit_roi_pool_call_s();

            int cb = cbb * jpp.nb_c_blocking;
            int cb_num = jpp.nb_c_blocking;

            arg.c_blocks = nstl::min(cb + cb_num, jpp.nb_c) - cb;

            if (n >= real_rois) {
                arg.dst = &dst[dst_d.blk_off(n, cb, oh, ow)];
                arg.bin_area = 0;

                (*kernel_)(&arg);
            } else {
                int roi_off;
                if(src_roi_d.ndims() == 4) {
                    roi_off = src_roi_d.off((int)n, 0, 0, 0);
                }
                else {
                    roi_off = src_roi_d.off((int)n, 0);
                }
                const data_t* src_roi_ptr = &src_roi[roi_off];

                int roi_batch_ind = src_roi_ptr[0];

                if (jpp.alg == mkldnn_roi_pooling_max) {
                    int roi_start_w = round(src_roi_ptr[1] * jpp.spatial_scale);
                    int roi_start_h = round(src_roi_ptr[2] * jpp.spatial_scale);
                    int roi_end_w = round(src_roi_ptr[3] * jpp.spatial_scale);
                    int roi_end_h = round(src_roi_ptr[4] * jpp.spatial_scale);

                    int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
                    int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);


                    int hstart = (oh * roi_height) / jpp.pooled_h;
                    if ((hstart * jpp.pooled_h) > (oh * roi_height)) {
                        --hstart;
                    }

                    int wstart = (ow * roi_width) / jpp.pooled_w;
                    if ((wstart * jpp.pooled_w) > (ow * roi_width)) {
                        --wstart;
                    }

                    int hend = ((oh + 1) * roi_height) / jpp.pooled_h;
                    if ((hend * jpp.pooled_h) < ((oh + 1) * roi_height)) {
                        ++hend;
                    }

                    int wend = ((ow + 1) * roi_width) / jpp.pooled_w;
                    if ((wend * jpp.pooled_w) < ((ow + 1) * roi_width)) {
                        ++wend;
                    }

                    hstart = std::min(std::max(hstart + roi_start_h, 0), jpp.ih);
                    hend = std::min(std::max(hend + roi_start_h, 0), jpp.ih);
                    wstart = std::min(std::max(wstart + roi_start_w, 0), jpp.iw);
                    wend = std::min(std::max(wend + roi_start_w, 0), jpp.iw);

                    arg.src = &src_data[src_d.blk_off(roi_batch_ind, cb, hstart, wstart)];
                    arg.dst = &dst[dst_d.blk_off(n, cb, oh, ow)];

                    arg.bin_area = (hend - hstart) * (wend - wstart);
                    arg.kh = hend - hstart;
                    arg.kw = wend - wstart;
                } else {
                    float roi_start_w_ = src_roi_ptr[1];
                    float roi_start_h_ = src_roi_ptr[2];
                    float roi_end_w_   = src_roi_ptr[3];
                    float roi_end_h_   = src_roi_ptr[4];

                    float height_scale = ((roi_end_h_ - roi_start_h_) * (jpp.ih - 1)) / (jpp.pooled_h - 1);
                    float width_scale  = ((roi_end_w_ - roi_start_w_) * (jpp.iw - 1)) / (jpp.pooled_w - 1);

                    float in_y = (oh * height_scale + roi_start_h_ * (jpp.ih - 1));
                    float in_x = (ow * width_scale  + roi_start_w_ * (jpp.iw - 1));

                    arg.dst = &dst[dst_d.blk_off(n, cb, oh, ow)];
                    if (in_y < 0 || in_y > jpp.ih - 1 || in_x < 0 || in_x > jpp.iw - 1) {
                        arg.bin_area = 0;
                    } else {
                        int top_y_index    = static_cast<int>(floorf(in_y));
                        int bottom_y_index = static_cast<int>(ceilf(in_y));
                        int left_x_index   = static_cast<int>(floorf(in_x));
                        int right_x_index  = static_cast<int>(ceilf(in_x));

                        arg.xf = in_x - left_x_index;
                        arg.yf = in_y - top_y_index;

                        if (right_x_index > jpp.iw - 1)
                            right_x_index = jpp.iw - 1;

                        if (bottom_y_index > jpp.ih - 1)
                            bottom_y_index = jpp.ih - 1;

                        arg.xoff = (size_t)((right_x_index - left_x_index) * jpp.c_block * sizeof(float));
                        arg.yoff = (size_t)((bottom_y_index - top_y_index) * jpp.iw * jpp.c_block * sizeof(float));

                        arg.src = &src_data[src_d.blk_off(roi_batch_ind, cb, top_y_index, left_x_index)];
                        arg.bin_area = 1;
                    }
                }


                (*kernel_)(&arg);
            }

            utils::nd_iterator_step(n, MB, cbb, cb_work, oh, jpp.oh, ow, jpp.ow);
        }
    };

    parallel(0, (size_t)work_amount, ker);
}

template struct jit_uni_roi_pooling_fwd_t<sse42>;
template struct jit_uni_roi_pooling_fwd_t<avx2>;
template struct jit_uni_roi_pooling_fwd_t<avx512_common>;

}
}
}
