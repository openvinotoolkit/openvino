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

#ifndef GPU_OCL_XE_HP_BF16_CONVOLUTION_HPP
#define GPU_OCL_XE_HP_BF16_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct xe_hp_bf16_convolution_bwd_weights_t : public gpu_primitive_t {
    struct pd_t : public gpu_convolution_bwd_weights_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : gpu_convolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:xe_hp", xe_hp_bf16_convolution_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);
            if (!compute_engine->is_xe_hp() && !compute_engine->is_xe_hpg())
                return status::unimplemented;

            bool ok = desc()->prop_kind == backward_weights
                    && desc()->alg_kind == alg_kind::convolution_direct
                    && (expect_data_types(bf16, bf16, bf16, bf16, f32)
                            || expect_data_types(bf16, bf16, f32, bf16,
                                    f32) //bf16 wei, f32 bias
                            || expect_data_types(bf16, f32, bf16, bf16,
                                    f32) //f32 wei, bf16 bias
                            || expect_data_types(bf16, f32, f32, bf16,
                                    f32)) //f32 wei, f32 bias
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && attr()->has_default_values();

            if (!ok) return status::unimplemented;

            CHECK(init_conf(engine));

            if (!compute_engine->mayiuse_sub_group({8, conf.sub_group_size}))
                return status::unimplemented;

            ok = set_default_formats_common(
                    conf.src_tag, conf.wei_tag, conf.dst_tag);
            return ok ? status::success : status::unimplemented;
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        conv_conf_t conf;
        offsets_t off;
    };

    status_t init(engine_t *engine) override {
        std::vector<const char *> kernel_names;
        // split barrier is disabled due to worse performance
        if (pd()->conf.use_split_barrier)
            kernel_names.push_back("xe_hp_conv_bwd_wei_bf16_split_bar");
        else
            kernel_names.push_back("xe_hp_conv_bwd_wei_bf16");
        kernel_names.push_back("xe_hp_wei_f32_zero_init");
        if (pd()->conf.weights_data_type == data_type::bf16)
            kernel_names.push_back("xe_hp_wei_convert_f32_to_bf16");

        compute::kernel_ctx_t kernel_ctx;
        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        std::vector<compute::kernel_t> kernels;
        CHECK(create_kernels(engine, &kernels, kernel_names, kernel_ctx));
        conv_kernel_ = kernels[0];
        zero_init_kernel_ = kernels[1];

        if (pd()->conf.weights_data_type == data_type::bf16)
            convert_f32_to_bf16_kernel_ = kernels[2];

        return status::success;
    }

    xe_hp_bf16_convolution_bwd_weights_t(const pd_t *apd)
        : gpu_primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)gpu_primitive_t::pd().get(); }

    compute::kernel_t conv_kernel_;
    compute::kernel_t zero_init_kernel_;
    compute::kernel_t convert_f32_to_bf16_kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
