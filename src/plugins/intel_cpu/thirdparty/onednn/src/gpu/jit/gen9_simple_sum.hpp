/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef GPU_JIT_GEN9_SIMPLE_SUM_HPP
#define GPU_JIT_GEN9_SIMPLE_SUM_HPP

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_sum_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

struct gen9_simple_sum_t : public gpu_primitive_t {
    struct pd_t : public gpu_sum_pd_t {
        using gpu_sum_pd_t::gpu_sum_pd_t;

        DECLARE_SUM_PD_T("ngen:simple:any", gen9_simple_sum_t);

        status_t init(engine_t *engine) {
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);
            if (!compute_engine->mayiuse_ngen_kernels())
                return status::unimplemented;

            const int n = n_inputs();

            constexpr auto data_type = data_type::f32;

            bool ok = gpu_sum_pd_t::init(engine) == status::success;
            if (!ok) return status::unimplemented;

            const memory_desc_wrapper o_d(dst_md());
            ok = ok && o_d.data_type() == data_type && o_d.is_dense();
            if (!ok) return status::unimplemented;

            for (int i = 0; i < n; ++i) {
                const memory_desc_wrapper i_d(src_md(i));
                if (i_d != o_d) return status::unimplemented;
            }

            return status::success;
        }
    };

    gen9_simple_sum_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    virtual status_t init(engine_t *engine);

    virtual status_t execute(const exec_ctx_t &ctx) const {
        status_t status = status::success;
        auto &output = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DST, status);
        CHECK(status);

        const int num_arrs = pd()->n_inputs();
        const memory_desc_wrapper o_d(pd()->dst_md());
        const size_t nelems = o_d.nelems();

        for (int a = 0; a < num_arrs; ++a) {
            auto &input = CTX_IN_STORAGE(DNNL_ARG_MULTIPLE_SRC + a);
            const float scale = pd()->scales()[a];

            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, input);
            arg_list.set(1, output);
            arg_list.set(2, scale);
            arg_list.set(3, a);

            size_t gws[3] = {nelems, 1, 1};
            size_t lws[3] = {1, 1, 1};
            auto nd_range = compute::nd_range_t(gws, lws);
            status = parallel_for(ctx, nd_range, kernel_, arg_list);
            if (status != status::success) return status;
        }
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)gpu_primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_JIT_GEN9_SIMPLE_SUM_HPP
