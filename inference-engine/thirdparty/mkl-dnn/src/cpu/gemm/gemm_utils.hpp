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

#ifndef GEMM_UTILS_HPP
#define GEMM_UTILS_HPP

namespace mkldnn {
namespace impl {
namespace cpu {

namespace gemm_utils {
void sum_two_matrices(
        int m, int n, float *p_src, int ld_src, float *p_dst, int ld_dst);

void calc_nthr_nocopy_avx512_common(int m,
        int n, int k, int nthrs, int *nthrs_m, int *nthrs_n, int *nthrs_k,
        int *BM, int *BN, int *BK);

void calc_nthr_nocopy_avx(int m, int n, int k,
        int nthrs, int *nthrs_m, int *nthrs_n, int *nthrs_k, int *BM, int *BN,
        int *BK);

void partition_unit_diff(
        int ithr, int nthr, int n, int *t_offset, int *t_block);
};

}
}
}
#endif
