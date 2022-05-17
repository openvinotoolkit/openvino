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

#ifndef GPU_OCL_REF_PRELU_HPP
#define GPU_OCL_REF_PRELU_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_prelu_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ref_prelu_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_prelu_fwd_pd_t {
        using gpu_prelu_fwd_pd_t::gpu_prelu_fwd_pd_t;

        DECLARE_COMMON_PD_T("prelu_ref:any", ref_prelu_fwd_t);

        status_t init(engine_t *engine) {

            bool ok = is_fwd() && set_default_formats()
                    && attr()->has_default_values();

            if (!ok) return status::unimplemented;

            return init_conf(engine);
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        prelu_conf_t conf;
    };

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        status = create_kernel(engine, &kernel_, "ref_prelu_fwd", kernel_ctx);
        CHECK(status);

        return status::success;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

struct ref_prelu_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_prelu_bwd_pd_t {
        using gpu_prelu_bwd_pd_t::gpu_prelu_bwd_pd_t;

        pd_t(const prelu_desc_t *adesc, const primitive_attr_t *attr,
                const prelu_fwd_pd_t *hint_fwd_pd)
            : gpu_prelu_bwd_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other) = default;

        ~pd_t() = default;

        DECLARE_COMMON_PD_T("prelu_ref:any", ref_prelu_bwd_t);

        status_t init(engine_t *engine) {

            bool ok = !is_fwd() && set_default_formats()
                    && attr()->has_default_values();

            if (!ok) return status::unimplemented;

            status_t status = init_conf(engine);
            if (conf.reduce_diff_weights) {
                CHECK(init_reduction(engine));
                init_scratchpad();
            }

            return status;
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        void init_scratchpad();

        status_t init_reduction(engine_t *engine) {
            reduction_desc_t rdesc;
            dnnl_memory_desc_t red_diff_mem_desc(*src_md(0));
            red_diff_mem_desc.data_type = dnnl_f32;
            dnnl_reduction_desc_init(&rdesc,
                    dnnl_alg_kind_t::dnnl_reduction_sum, &red_diff_mem_desc,
                    diff_weights_md(0), 0, 0);
            primitive_attr_t reduction_attr(*attr());
            if (!reduction_attr.is_initialized()) return status::out_of_memory;
            dnnl_primitive_desc_iterator it(
                    engine, (op_desc_t *)&rdesc, &reduction_attr, nullptr);
            if (!it.is_initialized()) return status::invalid_arguments;
            reduction_pd_ = *(++it);
            if (reduction_pd_)
                return status::success;
            else {
                return status::invalid_arguments;
            }
        }

        prelu_conf_t conf;
        std::shared_ptr<primitive_desc_t> reduction_pd_;
    };

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        status = create_kernel(engine, &kernel_, "ref_prelu_bwd", kernel_ctx);
        CHECK(status);

        if (pd()->conf.reduce_diff_weights) {
            status = pd()->reduction_pd_->create_primitive(
                    reduction_p_, engine);
            CHECK(status);
        }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

protected:
    primitive_list_t nested_primitives() const override {
        if (pd()->conf.reduce_diff_weights) {
            return {reduction_p_.get()};
        } else
            return {};
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
    std::shared_ptr<primitive_t> reduction_p_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
