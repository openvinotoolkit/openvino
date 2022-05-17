/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "gpu/ocl/gen9_sum.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t gen9_sum_t::execute(const exec_ctx_t &ctx) const {
    auto &output = CTX_OUT_STORAGE(DNNL_ARG_DST);
    const int num_arrs = pd()->n_inputs();
    const memory_desc_wrapper o_d(pd()->dst_md());
    const size_t nelems = o_d.nelems(true);
    compute::kernel_arg_list_t arg_list;

    for (int a = 0; a < max_num_arrs; ++a) {
        if (a < num_arrs) {
            auto &input = CTX_IN_STORAGE(DNNL_ARG_MULTIPLE_SRC + a);
            arg_list.set(a, input);
        } else
            arg_list.set(a, memory_storage_t::empty_storage());
    }
    arg_list.set(16, output);
    arg_list.set(17, CTX_GPU_RES_STORAGE(SCALES_));

    const size_t total_width = utils::div_up(nelems, vector_size);
    const size_t lws = 256;

    compute::nd_range_t nd_range({utils::rnd_up(total_width, lws)}, {lws});

    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
