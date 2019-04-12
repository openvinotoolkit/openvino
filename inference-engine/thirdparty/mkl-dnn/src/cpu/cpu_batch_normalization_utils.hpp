/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#ifndef CPU_BATCH_NORMALIZATION_UTILS_HPP
#define CPU_BATCH_NORMALIZATION_UTILS_HPP

#include "batch_normalization_pd.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {
namespace bnorm_utils {

void cache_balance(size_t working_set_size, int C_blks, int &C_blks_per_iter,
        int &iters);

bool thread_balance(bool do_blocking, bool spatial_thr_allowed, int ithr,
        int nthr, int N, int C_blks, int SP, int &C_ithr, int &C_nthr,
        int &C_blk_s, int &C_blk_e, int &N_ithr, int &N_nthr, int &N_s,
        int &N_e, int &S_ithr, int &S_nthr, int &S_s, int &S_e);

bool is_spatial_thr(const batch_normalization_pd_t *bdesc, int simd_w,
        int data_size);

}
}
}
}

#endif
