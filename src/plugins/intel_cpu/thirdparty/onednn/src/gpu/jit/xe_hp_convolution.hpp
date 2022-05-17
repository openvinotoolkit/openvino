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

#ifndef GPU_JIT_XE_HP_CONVOLUTION_HPP
#define GPU_JIT_XE_HP_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/jit/jit_eltwise_injector.hpp"
#include "gpu/primitive_conf.hpp"
#include "oneapi/dnnl/dnnl_config.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

struct xe_hp_convolution_fwd_t : public gpu_primitive_t {
    struct pd_t : public gpu_convolution_fwd_pd_t {
        using gpu_convolution_fwd_pd_t::gpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T("ngen:xe_hp", xe_hp_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;

            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);
            if (!compute_engine->is_xe_hp() && !compute_engine->is_xe_hpg())
                return status::unimplemented;
            if (!compute_engine->mayiuse_ngen_kernels())
                return status::unimplemented;
            if (!compute_engine->mayiuse_large_grf_mode())
                return status::unimplemented;

            const auto attr_skip_mask
                    = primitive_attr_t::skip_mask_t::oscale_runtime
                    | primitive_attr_t::skip_mask_t::post_ops
                    | primitive_attr_t::skip_mask_t::sum_dt;
            bool ok = set_default_alg_kind(alg_kind::convolution_direct)
                    && is_fwd() && data_types_ok()
                    && attr()->has_default_values(
                            attr_skip_mask, desc()->dst_desc.data_type)
                    && post_ops_ok(attr())
                    && IMPLICATION(!attr()->output_scales_.has_default_values(),
                            utils::one_of(src_md_.data_type, s8, u8)
                                    && utils::one_of(
                                            attr()->output_scales_.mask_, 0,
                                            1 << 1));
            if (!ok) return status::unimplemented;

            CHECK(init_conf(engine));

            ok = set_default_formats_common(
                    conf.src_tag, conf.wei_tag, conf.dst_tag);
            if (!ok) return status::unimplemented;

            CHECK(attr_.set_default_formats(dst_md(0)));

            return status::success;
        }

        bool data_types_ok() const {
            using namespace data_type;

            auto src_dt = invariant_src_md()->data_type;
            auto wei_dt = invariant_wei_md()->data_type;
            auto dst_dt = invariant_dst_md()->data_type;
            auto acc_dt = desc_.accum_data_type;

            bool is_int8 = (acc_dt == s32);
            is_int8 &= utils::one_of(src_dt, u8, s8);
            is_int8 &= utils::one_of(wei_dt, u8, s8);
            is_int8 &= utils::one_of(dst_dt, u8, s8, s32, f32);
            if (is_int8) return true;

            // Ignore accumulator type set to f16 and use f32.
            bool is_f16 = (acc_dt == f16 || acc_dt == f32);
            is_f16 &= (src_dt == f16);
            is_f16 &= (wei_dt == f16);
            is_f16 &= utils::one_of(dst_dt, f16, f32);
            if (is_f16) return true;

            bool is_bf16 = (acc_dt == f32);
            is_bf16 &= (src_dt == bf16);
            is_bf16 &= (wei_dt == bf16);
            is_bf16 &= utils::one_of(dst_dt, bf16, f32);
            if (is_bf16) return true;

            // Not supported.
            return false;
        }

        bool post_ops_ok(const primitive_attr_t *attr) const {
            if (!post_ops_with_binary_ok(attr, invariant_dst_md()->data_type))
                return false;

            auto &po = attr->post_ops_;
            int bin_cnt = 0;
            for (int i = 0; i < po.len(); i++) {
                if (po.entry_[i].is_binary()) {
                    // Limited binary postops support:
                    // common and per_oc policy
                    // f32 data type
                    // only one binary operation in postops list
                    auto src1_binary_po_d = memory_desc_wrapper(
                            po.entry_[i].binary.src1_desc);
                    int mask_binary_po = utils::get_dims_mask(dst_md_.dims,
                            src1_binary_po_d.dims(), dst_md_.ndims);
                    if (mask_binary_po != 0 && mask_binary_po != 2)
                        return false;
                    if (src1_binary_po_d.data_type() != data_type::f32)
                        return false;
                    if (++bin_cnt > 1) return false;
                }
                if (po.entry_[i].is_eltwise()) {
                    if (!jit_eltwise_injector_f32_is_supported(
                                po.entry_[i].eltwise.alg))
                        return false;
                }
            }
            return true;
        }

        status_t init_conf(engine_t *engine);

        conv_conf_t conf;
    };

    using gpu_primitive_t::gpu_primitive_t;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

protected:
    status_t init_res_storage(
            engine_t *engine, gpu_resource_t *r) const override {
        return init_output_scales_res_storage(engine, r, OSCALES_);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;

    const int OSCALES_ = 0;
};

struct xe_hp_convolution_bwd_data_t : public gpu_primitive_t {
    struct pd_t : public gpu_convolution_bwd_data_pd_t {
        using gpu_convolution_bwd_data_pd_t::gpu_convolution_bwd_data_pd_t;

        DECLARE_COMMON_PD_T("ngen:xe_hp", xe_hp_convolution_bwd_data_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;

            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            if (!compute_engine->is_xe_hp() && !compute_engine->is_xe_hpg())
                return status::unimplemented;
            if (!compute_engine->mayiuse_ngen_kernels())
                return status::unimplemented;
            if (!compute_engine->mayiuse_large_grf_mode())
                return status::unimplemented;

            bool ok = set_default_alg_kind(alg_kind::convolution_direct)
                    && is_bwd_d() && data_types_ok()
                    && attr()->has_default_values();

            if (!ok) return status::unimplemented;

            CHECK(init_conf(engine));

            // In conf, we express bwd_data as a fwd convolution
            // So we use the appropriate tags here
            ok = set_default_formats_common(
                    conf.dst_tag, conf.wei_tag, conf.src_tag);

            return ok ? status::success : status::unimplemented;
        }

        bool data_types_ok() const {
            using namespace data_type;

            auto src_dt = invariant_src_md()->data_type;
            auto wei_dt = invariant_wei_md()->data_type;
            auto dst_dt = invariant_dst_md()->data_type;
            auto acc_dt = desc_.accum_data_type;

            // Ignore accumulator type set to f16 and use f32.
            bool is_f16 = (acc_dt == f16 || acc_dt == f32);
            is_f16 &= (dst_dt == f16);
            is_f16 &= (wei_dt == f16);
            is_f16 &= utils::one_of(src_dt, f16, f32);
            if (is_f16) return true;

            bool is_bf16 = (acc_dt == f32);
            is_bf16 &= (dst_dt == bf16);
            is_bf16 &= (wei_dt == bf16);
            is_bf16 &= utils::one_of(src_dt, bf16, f32);
            if (is_bf16) return true;

            // Not supported.
            return false;
        }

        status_t init_conf(engine_t *engine);

        conv_conf_t conf;
    };

    using gpu_primitive_t::gpu_primitive_t;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

struct xe_hp_convolution_bwd_weights_t : public gpu_primitive_t {
    struct pd_t : public gpu_convolution_bwd_weights_pd_t {
        using gpu_convolution_bwd_weights_pd_t::
                gpu_convolution_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T("ngen:xe_hp", xe_hp_convolution_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;

            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            if (!compute_engine->is_xe_hp() && !compute_engine->is_xe_hpg())
                return status::unimplemented;
            if (!compute_engine->mayiuse_ngen_kernels())
                return status::unimplemented;
            if (!compute_engine->mayiuse_large_grf_mode())
                return status::unimplemented;

            // XXX: supported configs:
            // diff_src/diff_dst: bf16, acc: f32
            // wei/bia: bf16,f32
            bool ok = set_default_alg_kind(alg_kind::convolution_direct)
                    && desc()->prop_kind == backward_weights
                    && utils::one_of(true,
                            expect_data_types(bf16, f32, f32, bf16, f32),
                            expect_data_types(bf16, bf16, f32, bf16, f32),
                            expect_data_types(bf16, f32, bf16, bf16, f32),
                            expect_data_types(bf16, bf16, bf16, bf16, f32))
                    && attr()->has_default_values();

            if (!ok) return status::unimplemented;

            CHECK(init_conf(engine));
            init_scratchpad();

            ok = set_default_formats_common(
                    conf.src_tag, conf.wei_tag, conf.dst_tag);
            return ok ? status::success : status::unimplemented;
        }

        status_t init_conf(engine_t *engine);
        void init_scratchpad();

        conv_conf_t conf;
    };

    using gpu_primitive_t::gpu_primitive_t;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::vector<compute::kernel_t> kernels_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
