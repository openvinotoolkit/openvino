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

#ifndef GPU_OCL_GEMM_GEMM_WITH_POST_OPS_HPP
#define GPU_OCL_GEMM_GEMM_WITH_POST_OPS_HPP

#include "gpu/compute/compute.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gemm/gpu_gemm_utils.hpp"
#include "gpu/gpu_gemm_pd.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gemm_with_post_ops_t : public gpu_gemm_t {
    using gpu_gemm_t::gpu_gemm_t;
    struct pd_t : public gpu_gemm_pd_t {

        DECLARE_COMMON_PD_T("ocl:gemm_with_po:any", gemm_with_post_ops_t);

        pd_t(const gemm_desc_t *adesc, const primitive_attr_t *attr,
                const hint_class *hint_fwd_pd)
            : gpu_gemm_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other) = default;

        status_t init(engine_t *engine);

        void init_scratchpad();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        bool use_scratchpad() const {
            return use_scratchpad_with_post_op_worker;
        }

        std::shared_ptr<primitive_desc_t> gemm_pd_;
        bool use_scratchpad_with_post_op_worker = false;
        bool use_reorder = false;
        compute::dispatch_t dispatch_;
        attr_info_t attr_info_;
        float scale_ = 1;
    };

    status_t init(engine_t *engine) override {
        auto ret_status = pd()->gemm_pd_->create_primitive(gemm_prim_, engine);
        CHECK(ret_status);
        compute::kernel_ctx_t kernel_ctx;
        ret_status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(ret_status);
        ret_status = create_kernel(
                engine, &post_process_kernel_, "gemm_post_ops", kernel_ctx);
        return ret_status;
    }

    status_t execute(const gemm_exec_ctx_t &ctx) const override;

protected:
    primitive_list_t nested_primitives() const override {
        primitive_list_t list = {gemm_prim_.get()};
        return list;
    }
    status_t init_res_storage(
            engine_t *engine, gpu_resource_t *r) const override {
        const auto *attr = pd()->attr();
        std::unique_ptr<memory_storage_t> tmp_mem_storage;
        //TODO: Add zero points support
        for (const auto idx : {A0_, B0_, C0_}) {
            CHECK(gemm_utils::prepare_zero_points(
                    attr, engine, idx, tmp_mem_storage));
            r->add_memory_storage(idx, std::move(tmp_mem_storage));
        }
        CHECK(gemm_utils::prepare_scales(attr, engine, tmp_mem_storage));
        r->add_memory_storage(SCALES_, std::move(tmp_mem_storage));
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> gemm_prim_;
    compute::kernel_t post_process_kernel_;
    enum {
        A0_ = DNNL_ARG_A,
        B0_ = DNNL_ARG_B,
        C0_ = DNNL_ARG_C,
        SCALES_ = DNNL_ARG_ATTR_OUTPUT_SCALES
    };
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
