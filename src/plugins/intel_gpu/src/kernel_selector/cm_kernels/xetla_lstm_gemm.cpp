/*******************************************************************************
 * Copyright (c) 2022-2025 Intel Corporation
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

namespace KERNEL_NAME {

#include "include/xetla_lstm.h"

#define dtype_a fp16
#define dtype_b fp16
#define dtype_c float
#define dtype_acc float

//#define INPUT_SIZE 64 // or 256
#define hidden_size 128
#define directions 2

#define mem_layout_x mem_layout::row_major
#define mem_layout_w mem_layout::col_major
#define mem_layout_out mem_layout::row_major

#define mem_space_x mem_space::global
#define mem_space_w mem_space::global
#define mem_space_out mem_space::global

#define arch_tag gpu_arch::Xe

_GENX_MAIN_ void KERNEL_NAME(
        dtype_a *x [[type("svmptr_t")]],
        dtype_b *W [[type("svmptr_t")]], 
        dtype_b *B [[type("svmptr_t")]],
        int *sequence_lengths [[type("svmptr_t")]],
        dtype_acc *out [[type("svmptr_t")]]) {
    sycl::nd_item<3> item;
    using gemm_t = __xetla_kernel_lstm_gemm<dtype_a, dtype_b, dtype_c,
            dtype_acc, INPUT_SIZE, hidden_size, directions, mem_layout_x,
            mem_layout_w, mem_layout_out, mem_space_x, mem_space_w,
            mem_space_out, arch_tag>;
    if constexpr (gemm_t::gemm_t::barrier_count != 0) {
        cm_nbarrier_init(gemm_t::gemm_t::barrier_count);
    }
    if constexpr (gemm_t::gemm_t::slm_size != 0) {
        cm_slm_init(gemm_t::gemm_t::slm_size);
    }
    gemm_t::run(item, x, W, B, sequence_lengths, out);
}
}
