/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "gpu/jit/conv/gen_convolution.hpp"

#include <iostream>
#include <utility>

#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/conv_kernel.hpp"
#include "gpu/jit/conv/kernel_arg_info.hpp"
#include "gpu/jit/conv/utils.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using namespace compute;

static size_t icache_size(ngen::HW arch) {
    switch (arch) {
        case gpu_gen9: return 48 * 1024;
        case gpu_xe_lp: return 48 * 1024;
        case gpu_xe_hp: return 48 * 1024;
        case gpu_xe_hpg: return 96 * 1024;
        default: return 0;
    }
}

template <template <ngen::HW> class KernelT, ngen::HW arch, typename... ArgsT>
std::unique_ptr<jit::jit_generator_base> make_generator(ArgsT &&... args) {

    auto raw_kernel = new KernelT<arch>(std::forward<ArgsT>(args)...);
    if (raw_kernel->getRootStreamLength() > icache_size(arch)) {
        ir_warning() << raw_kernel->kernel_name()
                     << " larger than icache, kernel: "
                     << raw_kernel->getRootStreamLength()
                     << " bytes, icache: " << icache_size(arch) << " bytes\n";
    }
    return std::unique_ptr<jit::jit_generator_base>(raw_kernel);
}

template <template <ngen::HW> class KernelT, typename... ArgsT>
compute::kernel_t make_kernel(
        gpu_primitive_t *primitive, engine_t *engine, ArgsT &&... args) {
    auto compute_engine = utils::downcast<compute_engine_t *>(engine);
    auto device_info = compute_engine->device_info();

    std::unique_ptr<jit::jit_generator_base> jit_kernel;
    switch (device_info->gpu_arch()) {
        case gpu_arch_t::gen9:
            jit_kernel = make_generator<KernelT, gpu_gen9>(
                    std::forward<ArgsT>(args)...);
            break;
        case gpu_arch_t::xe_lp:
            jit_kernel = make_generator<KernelT, gpu_xe_lp>(
                    std::forward<ArgsT>(args)...);
            break;
        case gpu_arch_t::xe_hp:
            jit_kernel = make_generator<KernelT, gpu_xe_hp>(
                    std::forward<ArgsT>(args)...);
            break;
        case gpu_arch_t::xe_hpg:
            jit_kernel = make_generator<KernelT, gpu_xe_hpg>(
                    std::forward<ArgsT>(args)...);
            break;
        default: break;
    }

    if (!jit_kernel) return compute::kernel_t();

    compute::kernel_t kernel;
    status_t status = primitive->create_kernel(engine, &kernel, *jit_kernel);
    if (status != status::success) return compute::kernel_t();
    return kernel;
}

class gen_convolution_t {
public:
    template <typename T>
    static status_t init_pd(T *pd, engine_t *engine) {
        auto *compute_engine = utils::downcast<compute_engine_t *>(engine);

        if (!compute_engine->mayiuse_ngen_kernels())
            return status::unimplemented;
        if (!pd->set_default_alg_kind(alg_kind::convolution_direct))
            return status::unimplemented;
        pd->cfg = std::make_shared<conv_config_t>();
        pd->kernel_arg_info = std::make_shared<kernel_arg_info_t>();
        CHECK(pd->cfg->init(pd, &pd->attr_, engine));

        CHECK(init_kernel_arg_info(pd, *pd->kernel_arg_info));

        return status::success;
    }

    gen_convolution_t() = default;

    template <typename T>
    status_t init(T *primitive, engine_t *engine) {
        try {
            auto &cfg = get_cfg(primitive);

            ir_trace() << "Configuration:" << std::endl;
            ir_trace() << cfg;

            kernel_ = make_kernel<conv_kernel_t>(primitive, engine, cfg,
                    primitive->pd(), *primitive->pd()->kernel_arg_info);
            if (!kernel_) return status::runtime_error;

            if (cfg.zero_out_output) {
                bool with_dpas = utils::one_of(
                        cfg.fma_kind, fma_kind_t::dpas, fma_kind_t::dpasw);

                zero_out_kernel_ = make_kernel<zero_out_kernel_t>(
                        primitive, engine, cfg.simd_size, cfg.regs, with_dpas);
                if (!zero_out_kernel_) return status::runtime_error;
            }
        } catch (...) {
            // If verbose is enabled, print the primitive case and rethrow the
            // exception.
            if (get_verbose())
                printf("dnnl_verbose,error,%s\n",
                        primitive->pd()->info(engine));
            throw;
        }

        return status::success;
    }

    template <typename T>
    status_t execute(const T *primitive, const exec_ctx_t &ctx) const {
        auto &kernel_arg_info = *primitive->pd()->kernel_arg_info;

        std::vector<memory_storage_wrapper_t> storage_list;
        kernel_arg_info.init_memory_storage_list(storage_list, ctx, primitive);

        kernel_arg_list_t arg_list;
        kernel_arg_info.set_args(arg_list, storage_list);

        auto &cfg = get_cfg(primitive);

        if (cfg.zero_out_output) {
            for (int i = 0; i < kernel_arg_info.nargs(); i++) {
                if (kernel_arg_info.is_input(i)) continue;

                int key = kernel_arg_info.key(i);
                if (kernel_arg_info.is_scratchpad(i)) {
                    if (!utils::one_of(key,
                                memory_tracking::names::key_conv_wei_reduction,
                                memory_tracking::names::key_conv_bia_reduction))
                        continue;
                }
                const auto &storage
                        = kernel_arg_info.arg_storage(i, ctx, primitive);
                size_t size = kernel_arg_info.arg_size(i, primitive);

                kernel_arg_list_t arg_list;
                arg_list.set(0, *storage.get());
                arg_list.set(1, uint32_t(size));

                int bytes_per_thr = zero_out_kernel_t<>::bytes_per_thr;
                compute::nd_range_t nd_range(
                        {utils::div_up(size, bytes_per_thr) * cfg.simd_size});
                CHECK(primitive->parallel_for(
                        ctx, nd_range, zero_out_kernel_, arg_list));
            }
        }

        auto nd_range = cfg.nd_range();
        CHECK(primitive->parallel_for(ctx, nd_range, kernel_, arg_list));

        return status::success;
    }

private:
    template <typename T>
    static const conv_config_t &get_cfg(const T *primitive) {
        return *primitive->pd()->cfg;
    }

    template <typename T>
    static status_t init_kernel_arg_info(
            const T *pd, kernel_arg_info_t &kernel_arg_info) {
        auto &cfg = *pd->cfg;
        auto *attr = pd->attr();

        // Initialize main arguments.
        if (cfg.is_fwd) {
            kernel_arg_info.register_user_arg(
                    make_buffer("src"), DNNL_ARG_SRC, /*is_input=*/true);
            kernel_arg_info.register_user_arg(
                    make_buffer("wei"), DNNL_ARG_WEIGHTS, /*is_input=*/true);
            kernel_arg_info.register_user_arg(
                    make_buffer("dst"), DNNL_ARG_DST, /*is_input=*/false);
        } else if (cfg.is_bwd_d) {
            kernel_arg_info.register_user_arg(
                    make_buffer("dst"), DNNL_ARG_DIFF_DST, /*is_input=*/true);
            kernel_arg_info.register_user_arg(
                    make_buffer("wei"), DNNL_ARG_WEIGHTS, /*is_input=*/true);
            kernel_arg_info.register_user_arg(
                    make_buffer("src"), DNNL_ARG_DIFF_SRC, /*is_input=*/false);
        } else if (cfg.is_bwd_w) {
            kernel_arg_info.register_user_arg(
                    make_buffer("src"), DNNL_ARG_SRC, /*is_input=*/true);
            kernel_arg_info.register_user_arg(
                    make_buffer("dst"), DNNL_ARG_DIFF_DST, /*is_input=*/true);
            if (!cfg.do_post_wei_reorder) {
                kernel_arg_info.register_user_arg(make_buffer("wei"),
                        DNNL_ARG_DIFF_WEIGHTS, /*is_input=*/false);
            }
            if (cfg.with_bias && !cfg.do_post_bia_reorder) {
                kernel_arg_info.register_user_arg(make_buffer("bia"),
                        DNNL_ARG_DIFF_BIAS, /*is_input=*/false);
            }
        } else {
            ir_error_not_expected();
        }

        // Initialize post-op arguments.
        if ((cfg.is_fwd || cfg.is_bwd_d) && cfg.with_bias) {
            kernel_arg_info.register_user_arg(
                    make_buffer("bia"), DNNL_ARG_BIAS, /*is_input=*/true);
        }

        bool with_oscales = !attr->output_scales_.has_default_values();
        if (with_oscales) {
            bool is_runtime_oscales = !attr->output_scales_.defined();
            bool is_common_oscales = (attr->output_scales_.mask_ == 0);
            if (is_runtime_oscales) {
                kernel_arg_info.register_user_arg(make_buffer("oscales"),
                        DNNL_ARG_ATTR_OUTPUT_SCALES, /*is_input=*/true);
            } else if (is_common_oscales) {
                auto oscales_buf = var_t::make(type_t::f32(), "oscales");
                auto value = float_imm_t::make(attr->output_scales_.scales_[0]);
                kernel_arg_info.register_internal_arg(oscales_buf, value);
            } else {
                kernel_arg_info.register_resource_arg(make_buffer("oscales"));
            }
        }

        for (int i = 0; i < attr->post_ops_.len(); i++) {
            auto &po = attr->post_ops_.entry_[i];
            if (po.is_binary()) {
                auto buf = make_buffer("binary_rhs_" + std::to_string(i));
                kernel_arg_info.register_user_arg(buf,
                        DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1,
                        /*is_input=*/true);
            }
        }

        return status::success;
    }

    kernel_t kernel_;
    kernel_t zero_out_kernel_;
};

status_t gen_convolution_fwd_t::pd_t::init(engine_t *engine) {
    if (!is_fwd()) return status::unimplemented;
    CHECK(gen_convolution_t::init_pd(this, engine));
    return status::success;
}

status_t gen_convolution_fwd_t::init(engine_t *engine) {
    impl_.reset(new gen_convolution_t());
    return impl_->init(this, engine);
}

status_t gen_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

status_t gen_convolution_fwd_t::init_res_storage(
        engine_t *engine, gpu_resource_t *r) const {
    auto &kernel_arg_info = *pd()->kernel_arg_info;
    for (int i = 0; i < kernel_arg_info.nargs(); i++) {
        if (!kernel_arg_info.is_resource(i)) continue;

        auto &arg_name = kernel_arg_info.arg_name(i);
        int key = kernel_arg_info.key(i);
        if (arg_name == "oscales") {
            CHECK(init_output_scales_res_storage(engine, r, key));
        } else {
            ir_error_not_expected();
        }
    }
    return status::success;
}

status_t gen_convolution_bwd_data_t::pd_t::init(engine_t *engine) {
    if (!is_bwd_d()) return status::unimplemented;
    CHECK(gen_convolution_t::init_pd(this, engine));
    return status::success;
}

status_t gen_convolution_bwd_weights_t::pd_t::init(engine_t *engine) {
    if (!is_bwd_w()) return status::unimplemented;
    CHECK(gen_convolution_t::init_pd(this, engine));

    CHECK(init_scratchpad(*kernel_arg_info));
    return status::success;
}

status_t gen_convolution_bwd_weights_t::pd_t::init_scratchpad(
        kernel_arg_info_t &kernel_arg_info) {
    auto scratchpad = scratchpad_registry().registrar();
    if (cfg->do_post_wei_reorder) {
        size_t tmp_wei_size = memory_desc_wrapper(diff_weights_md())
                                      .nelems(/*with_padding=*/true)
                * types::data_type_size(data_type::f32);
        scratchpad.book(memory_tracking::names::key_conv_wei_reduction,
                tmp_wei_size, 1, ocl::OCL_BUFFER_ALIGNMENT);
        kernel_arg_info.register_scratchpad_arg(make_buffer("wei"),
                memory_tracking::names::key_conv_wei_reduction,
                /*is_input=*/false, tmp_wei_size);
    }
    if (cfg->do_post_bia_reorder) {
        size_t tmp_bia_size = memory_desc_wrapper(diff_weights_md(1))
                                      .nelems(/*with_padding=*/true)
                * types::data_type_size(data_type::f32);
        scratchpad.book(memory_tracking::names::key_conv_bia_reduction,
                tmp_bia_size, 1, ocl::OCL_BUFFER_ALIGNMENT);
        kernel_arg_info.register_scratchpad_arg(make_buffer("bia"),
                memory_tracking::names::key_conv_bia_reduction,
                /*is_input=*/false, tmp_bia_size);
    }
    return status::success;
}

status_t gen_convolution_bwd_data_t::init(engine_t *engine) {
    impl_.reset(new gen_convolution_t());
    return impl_->init(this, engine);
}

status_t gen_convolution_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

status_t gen_convolution_bwd_weights_t::init(engine_t *engine) {
    auto &cfg = *pd()->cfg;
    impl_.reset(new gen_convolution_t());
    bool with_dpas
            = utils::one_of(cfg.fma_kind, fma_kind_t::dpas, fma_kind_t::dpasw);
    if (cfg.do_post_wei_reorder || cfg.do_post_bia_reorder) {
        reorder_kernel_ = make_kernel<reorder_kernel_t>(
                this, engine, cfg.simd_size, cfg.regs, with_dpas);
        if (!reorder_kernel_) return status::runtime_error;
    }
    return impl_->init(this, engine);
}

status_t gen_convolution_bwd_weights_t::execute(const exec_ctx_t &ctx) const {
    CHECK(impl_->execute(this, ctx));
    auto &cfg = *pd()->cfg;
    if (cfg.do_post_wei_reorder) {
        auto scratchpad = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_conv_wei_reduction);
        int elems = memory_desc_wrapper(pd()->diff_weights_md()).nelems(true);
        kernel_arg_list_t arg_list;
        arg_list.set(0, *scratchpad);
        arg_list.set(1, CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS));
        arg_list.set(2, uint32_t(elems));
        int elems_per_thr = reorder_kernel_t<>::elems_per_thr;
        compute::nd_range_t nd_range(
                {utils::div_up(elems, elems_per_thr) * cfg.simd_size, 1, 1});
        CHECK(parallel_for(ctx, nd_range, reorder_kernel_, arg_list));
    }
    if (cfg.do_post_bia_reorder) {
        auto scratchpad = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_conv_bia_reduction);
        int elems = memory_desc_wrapper(pd()->diff_weights_md(1)).nelems(true);
        kernel_arg_list_t arg_list;
        arg_list.set(0, *scratchpad);
        arg_list.set(1, CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS));
        arg_list.set(2, uint32_t(elems));
        int elems_per_thr = reorder_kernel_t<>::elems_per_thr;
        compute::nd_range_t nd_range(
                {utils::div_up(elems, elems_per_thr) * cfg.simd_size, 1, 1});
        CHECK(parallel_for(ctx, nd_range, reorder_kernel_, arg_list));
    }
    return status::success;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
