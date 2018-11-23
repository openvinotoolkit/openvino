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

#include "mkldnn_thread.hpp"
#include "simple_sum.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t data_type>
void simple_sum_t<data_type>::execute() {
    auto output = reinterpret_cast<data_t *>(this->memory());
    const int num_arrs = conf_.n_inputs();
    const memory_desc_wrapper o_d(conf_.dst_pd());
    output += o_d.blk_off(0);
    const size_t nelems = o_d.nelems();
    const data_t *input_ptrs[max_num_arrs];

    for (int a = 0; a < num_arrs; ++a) {
        const memory_desc_wrapper i_d(conf_.src_pd(a));

        input_ptrs[a] = reinterpret_cast<const data_t *>(
                this->input_memory(a)) + i_d.blk_off(0);
    }

    const size_t block_size = 16 * 1024 / sizeof(data_type);
    const size_t blocks_number = nelems / block_size;
    const size_t tail = nelems % block_size;

    const auto &scales = conf_.scales_;
    parallel(0, [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        balance211(blocks_number, nthr, ithr, start, end);

        for (size_t nb = start; nb < end; ++nb) {
            size_t start_e = nb * block_size;
            size_t end_e = start_e + block_size;

            PRAGMA_OMP_SIMD()
            for (size_t e = start_e; e < end_e; e++) {
                output[e] = data_t(scales[0] * input_ptrs[0][e]);
            }
            for (int a = 1; a < num_arrs; a++) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] += data_t(scales[a] * input_ptrs[a][e]);
                }
            }
        }

        if (tail != 0 && ithr == nthr - 1) {
            size_t start_e = nelems - tail;
            size_t end_e = nelems;

            PRAGMA_OMP_SIMD()
            for (size_t e = start_e; e < end_e; e++) {
                output[e] = data_t(scales[0] * input_ptrs[0][e]);
            }
            for (int a = 1; a < num_arrs; a++) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] += data_t(scales[a] * input_ptrs[a][e]);
                }
            }
        }
    });
}

template struct simple_sum_t<data_type::f32>;

}
}
}
