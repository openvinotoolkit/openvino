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

#ifndef _IP_HPP
#define _IP_HPP

#include "mkldnn.h"

#include "common.hpp"
#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"

namespace ip {

struct prb_t {
    dir_t dir;
    int mb;
    int ic, ih, iw;
    int oc;

    mkldnn_data_type_t src_dt, wei_dt, acc_dt, dst_dt;
};

inline size_t src_off_f(const prb_t *p, int mb, int ic, int ih, int iw) {
    return ((mb * p->ic + ic) * p->ih + ih) * p->iw + iw;
}

inline size_t wei_off_f(const prb_t *p, int oc, int ic, int ih, int iw) {
    return ((oc * p->ic + ic) * p->ih + ih) * p->iw + iw;
}

inline size_t bia_off_f(const prb_t *p, int oc) { return oc; }

inline size_t dst_off_f(const prb_t *p, int mb, int oc) {
    return mb * p->oc + oc;
}

void compute_ref_fwd(const prb_t *p, dnn_mem_t &src_m, dnn_mem_t &wei_m,
        dnn_mem_t &bia_m, dnn_mem_t &dst_m);
void compute_ref_bwd_d(const prb_t *p, dnn_mem_t &diff_src_m, dnn_mem_t &wei_m,
        dnn_mem_t &diff_dst_m);
void compute_ref_bwd_w(const prb_t *p, dnn_mem_t &src_m, dnn_mem_t &diff_wei_m,
        dnn_mem_t &diff_bia_m, dnn_mem_t &diff_dst_m);

int doit(prb_t *p, res_t *res);

int bench(int argc, char **argv);

}

#endif
