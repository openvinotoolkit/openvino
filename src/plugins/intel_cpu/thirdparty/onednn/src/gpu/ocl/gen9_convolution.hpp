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

#ifndef GPU_OCL_GEN9_CONVOLUTION_HPP
#define GPU_OCL_GEN9_CONVOLUTION_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/gpu_eltwise_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen9_convolution_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : gpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:gen9:blocked", gen9_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            auto src_data_t = this->desc()->src_desc.data_type;
            auto dst_data_t = this->desc()->dst_desc.data_type;

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;

            bool ok = set_default_alg_kind(alg_kind::convolution_direct)
                    && utils::one_of(this->desc()->prop_kind, forward_training,
                            forward_inference)
                    && this->desc()->alg_kind == alg_kind::convolution_direct
                    && utils::one_of(true,
                            expect_data_types(f32, f32, f32, f32, f32),
                            expect_data_types(f32, f32, f32, s8, f32),
                            expect_data_types(f16, f16, f16, s8, f16),
                            expect_data_types(f16, f16, f16, f16, f16))
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && IMPLICATION(src_data_t == f16,
                            true
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp16)
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::
                                                    intel_subgroups_short))
                    && !has_zero_dim_memory()
                    && attr()->has_default_values(attr_skip_mask, dst_data_t)
                    && post_ops_with_binary_ok(attr(), dst_md()->data_type);
            if (!ok) return status::unimplemented;

            CHECK(init_conf(engine));

            if (!compute_engine->mayiuse_sub_group(conf.sub_group_size))
                return status::unimplemented;

            ok = set_default_formats_common(
                    conf.src_tag, conf.wei_tag, conf.dst_tag);
            if (!ok) return status::unimplemented;

            CHECK(attr_.set_default_formats(dst_md(0)));

            return status::success;
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        conv_conf_t conf;
    };

    status_t init(engine_t *engine) override {
        const char *kernel_name = nullptr;

        if (pd()->conf.is_nhwc
                && utils::one_of(pd()->conf.src_data_type, data_type::f32,
                        data_type::f16)) {
            kernel_name = "gen9_conv_nhwc_fwd";

        } else if (pd()->conf.is_depthwise) {
            kernel_name = "gen9_conv_dw_fwd";
        } else if (utils::one_of(pd()->desc()->src_desc.data_type,
                           data_type::f16, data_type::f32)) {
            kernel_name = "gen9_conv_fwd";
        } else {
            assert(!"not expected");
        }

        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        create_kernel(engine, &kernel_, kernel_name, kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

struct gen9_convolution_bwd_data_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : gpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:ncsp:any", gen9_convolution_bwd_data_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            bool ok = set_default_alg_kind(alg_kind::convolution_direct)
                    && this->desc()->prop_kind == backward_data
                    && this->desc()->alg_kind == alg_kind::convolution_direct
                    && utils::one_of(true,
                            expect_data_types(
                                    f32, f32, data_type::undef, f32, f32),
                            expect_data_types(
                                    f16, f16, data_type::undef, f16, f16))
                    && IMPLICATION(this->with_bias()
                                    && this->desc()->diff_dst_desc.data_type
                                            != f16,
                            this->desc()->bias_desc.data_type == f32)
                    && IMPLICATION(this->with_bias()
                                    && this->desc()->diff_dst_desc.data_type
                                            == f16,
                            this->desc()->bias_desc.data_type == f16)
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && !has_zero_dim_memory() && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            CHECK(init_conf(engine));

            if (!compute_engine->mayiuse_sub_group(conf.sub_group_size))
                return status::unimplemented;

            ok = set_default_formats_common(
                    conf.src_tag, conf.wei_tag, conf.dst_tag);
            return ok ? status::success : status::unimplemented;
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        conv_conf_t conf;
    };

    status_t init(engine_t *engine) override {
        const char *kernel_name = nullptr;
        if (pd()->conf.is_depthwise) {
            kernel_name = "gen9_conv_dw_bwd_data";
        } else {
            if (pd()->conf.is_nhwc)
                kernel_name = "gen9_conv_nhwc_bwd_data";
            else
                kernel_name = "gen9_conv_bwd_data";
        }

        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        create_kernel(engine, &kernel_, kernel_name, kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

struct gen9_convolution_bwd_weights_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_convolution_bwd_weights_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : gpu_convolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &rhs) = default;

        DECLARE_COMMON_PD_T("ocl:ncsp:any", gen9_convolution_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            bool ok = set_default_alg_kind(alg_kind::convolution_direct)
                    && this->desc()->prop_kind == backward_weights
                    && this->desc()->alg_kind == alg_kind::convolution_direct
                    && utils::one_of(this->desc()->diff_weights_desc.data_type,
                            f32, bf16)
                    && utils::one_of(
                            this->desc()->src_desc.data_type, f32, bf16)
                    && utils::one_of(
                            this->desc()->diff_dst_desc.data_type, f32, bf16)
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && compute_engine->mayiuse(
                            compute::device_ext_t::khr_int64_base_atomics)
                    && !has_zero_dim_memory() && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            CHECK(init_conf(engine));
            if (!compute_engine->mayiuse_sub_group(conf.sub_group_size))
                return status::unimplemented;

            if (!IMPLICATION(utils::one_of(bf16,
                                     this->desc()->diff_weights_desc.data_type,
                                     this->desc()->src_desc.data_type,
                                     this->desc()->diff_dst_desc.data_type),
                        conf.ver == ver_1stconv))
                return status::unimplemented;

            init_scratchpad();
            return status::success;
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        conv_conf_t conf;
        std::shared_ptr<primitive_desc_t> rpd_wei_;
        std::shared_ptr<primitive_desc_t> rpd_bia_;

    private:
        status_t init_scratchpad();
    };

    status_t init(engine_t *engine) override {
        const char *kernel_name;
        if (pd()->conf.is_nhwc) {
            kernel_name = "gen9_conv_nhwc_bwd_weights";
        } else {
            kernel_name = "gen9_conv_bwd_weights";
        }
        if (pd()->conf.reorder_wei) {
            CHECK(pd()->rpd_wei_->create_primitive(wei_reorder_, engine));
        }
        if (pd()->conf.reorder_bias) {
            CHECK(pd()->rpd_bia_->create_primitive(bia_reorder_, engine));
        }
        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        create_kernel(engine, &kernel_, kernel_name, kernel_ctx);
        if (!kernel_) return status::runtime_error;
        return status::success;
    }

    primitive_list_t nested_primitives() const override {
        primitive_list_t prim_list;
        if (pd()->conf.reorder_wei)
            prim_list.emplace(prim_list.begin(), wei_reorder_.get());
        if (pd()->conf.reorder_bias)
            prim_list.emplace(prim_list.begin(), bia_reorder_.get());

        return prim_list;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
    std::shared_ptr<primitive_t> wei_reorder_;
    std::shared_ptr<primitive_t> bia_reorder_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
