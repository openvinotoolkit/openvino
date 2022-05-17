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

#ifndef GPU_OCL_GEN9_SUM_HPP
#define GPU_OCL_GEN9_SUM_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/gpu_sum_pd.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen9_sum_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_sum_pd_t {
        using gpu_sum_pd_t::gpu_sum_pd_t;

        DECLARE_SUM_PD_T("ocl:gen9:any", gen9_sum_t);

        status_t init(engine_t *engine) {
            const int n = n_inputs();

            if (n > max_num_arrs) return status::unimplemented;

            bool ok = gpu_sum_pd_t::init(engine) == status::success;

            if (!ok) return status::unimplemented;

            const memory_desc_wrapper o_d(dst_md());

            for (int i = 0; i < n; ++i) {
                const memory_desc_wrapper i_d(src_md(i));
                if (i_d != o_d) return status::unimplemented;
            }

            return status::success;
        }
    };

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        const memory_desc_wrapper data_d(pd()->dst_md());
        const memory_desc_wrapper data_s(pd()->src_md());

        kernel_ctx.set_data_type(data_s.data_type());

        kernel_ctx.define_int("VECT_DT_N", vector_size);
        kernel_ctx.define_int("N_INPUTS", pd()->n_inputs());
        kernel_ctx.define_int("N_ELEMS", data_d.nelems(true));

        def_memory_desc_info(
                kernel_ctx, memory_desc_info_t::create(data_d), "SRC");
        def_memory_desc_info(
                kernel_ctx, memory_desc_info_t::create(data_s), "DST");

        create_kernel(engine, &kernel_, "gen9_sum", kernel_ctx);

        if (!kernel_) return status::runtime_error;
        return status::success;
    }

    status_t init_res_storage(
            engine_t *engine, gpu_resource_t *r) const override {
        const dim_t count = pd()->n_inputs();
        const float *s_data = pd()->scales();

        const size_t size = count * sizeof(float);
        std::unique_ptr<memory_storage_t> scales;
        memory_storage_t *scale = nullptr;
        auto s = engine->create_memory_storage(&scale, size);
        if (s != status::success) return s;
        float *mapped_mem_storage = nullptr;
        s = scale->map_data((void **)&mapped_mem_storage, nullptr, size);
        if (s != status::success) return s;
        utils::array_copy(mapped_mem_storage, s_data, count);
        s = scale->unmap_data((void *)mapped_mem_storage, nullptr);
        if (s != status::success) return s;
        scales.reset(scale);
        r->add_memory_storage(SCALES_, std::move(scales));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    enum { max_num_arrs = 16 };
    const int vector_size = 8;
    enum { SCALES_ = 0 };
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
