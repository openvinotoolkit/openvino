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

#ifndef GPU_JIT_GEMM_GEN_GEMM_HPP
#define GPU_JIT_GEMM_GEN_GEMM_HPP

#include <assert.h>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/compute/kernel.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gpu_gemm_pd.hpp"
#include "gpu/jit/gemm/gen_gemm_kernel.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

struct gen_gemm_t : public gpu_gemm_t {

    struct pd_t : public gpu_gemm_pd_t {
        using gpu_gemm_pd_t::gpu_gemm_pd_t;

        DECLARE_COMMON_PD_T("jit:gemm:any", gen_gemm_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            using namespace primitive_kind;
            using smask_t = primitive_attr_t::skip_mask_t;
            using arch_t = compute::gpu_arch_t;

            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            // LIMITATIONS:
            // - runtime dims are not supported
            bool ok = true;

            auto attr_skip_mask = smask_t::oscale | smask_t::post_ops;

            dev_info_ = compute_engine->device_info();
            arch_ = dev_info_->gpu_arch();

            ok = set_default_formats();
            if (!ok) return status::unimplemented;

            const auto d = desc();

            if (d->c_type() == s32) {
                ok = ok && utils::one_of(d->a_type(), u8, s8)
                        && utils::one_of(d->b_type(), u8, s8)
                        && d->acc_type == d->c_type()
                        && (attr()->zero_points_.has_default_values(
                                    DNNL_ARG_DST)
                                || !attr()->zero_points_.defined(DNNL_ARG_DST));

                if (attr()->zero_points_.defined(DNNL_ARG_SRC)) {
                    const int *ao_i32 = nullptr;
                    attr()->zero_points_.get(
                            DNNL_ARG_SRC, nullptr, nullptr, &ao_i32);
                    ab_zp_ |= (*ao_i32 != 0);
                } else if (!attr()->zero_points_.has_default_values(
                                   DNNL_ARG_SRC))
                    return status::unimplemented;

                if (attr()->zero_points_.defined(DNNL_ARG_WEIGHTS)) {
                    const int *bo_i32 = nullptr;
                    attr()->zero_points_.get(
                            DNNL_ARG_WEIGHTS, nullptr, nullptr, &bo_i32);
                    ab_zp_ |= (*bo_i32 != 0);
                } else if (!attr()->zero_points_.has_default_values(
                                   DNNL_ARG_WEIGHTS))
                    return status::unimplemented;

                int cmask_a = 0, cmask_b = 0, cmask_c = 0;
                attr()->zero_points_.get(
                        DNNL_ARG_WEIGHTS, nullptr, &cmask_b, nullptr);
                attr()->zero_points_.get(
                        DNNL_ARG_SRC, nullptr, &cmask_a, nullptr);
                attr()->zero_points_.get(
                        DNNL_ARG_DST, nullptr, &cmask_c, nullptr);
                ok &= (cmask_a == 0) && (cmask_b == 0)
                        && utils::one_of(cmask_c, 0, 1 << 0, 1 << 1);

                attr_skip_mask |= smask_t::zero_points_runtime;
            } else if (d->c_type() == bf16) {
                ok = ok && d->a_type() == bf16 && d->b_type() == bf16
                        && utils::one_of(d->acc_type, bf16, f32);
            } else {
                ok = ok && utils::one_of(d->c_type(), f32, f16)
                        && d->a_type() == d->c_type()
                        && d->b_type() == d->c_type()
                        && d->acc_type == d->c_type();
            }

            ok = ok && !has_blocks() && batch_dims() <= 2
                    && !utils::one_of(DNNL_RUNTIME_DIM_VAL, d->m(), d->n(),
                            d->k(), d->lda(), d->ldb(), d->ldc(), d->batch())
                    && IMPLICATION(with_bias(),
                            (d->bias_type() == d->c_type())
                                    && utils::one_of(
                                            d->bias_type(), f32, bf16, f16)
                                    && utils::one_of(
                                            bias_cmask(), 0, 1 << 0, 1 << 1)
                                    && (d->bias_desc.ndims <= 3))
                    && compute_engine->mayiuse_ngen_kernels()
                    && attr()->has_default_values(attr_skip_mask)
                    && attr()->output_scales_.mask_ == 0
                    && IMPLICATION(with_bias(),
                            utils::one_of(d->c_type(), f32, f16, bf16)
                                    && utils::one_of(
                                            bias_cmask(), 0, 1 << 0, 1 << 1)
                                    && (attr()->zero_points_.has_default_values(
                                            DNNL_ARG_DST)));

            // check if there is sum post op and only at first place
            ok &= IMPLICATION(attr()->post_ops_.find(sum) != -1,
                    attr()->post_ops_.find(sum) == 0
                            && attr()->post_ops_.find(sum, 1) == -1);

            // check if post ops are supported
            ok &= IMPLICATION(attr()->post_ops_.len() > 0,
                    jit_post_op_injector_is_supported(attr()->post_ops_, true));

            if (!ok) return status::unimplemented;

            ok &= utils::one_of(arch_, arch_t::gen9, arch_t::xe_lp,
                    arch_t::xe_hp, arch_t::xe_hpg);

            bool int8_ok = arch_ < arch_t::xe_hp;

            // int8 not enabled on Xe_HP/Xe_HPG for now. bf16 only enabled on Xe_HP+.
            ok &= IMPLICATION(utils::one_of(d->a_type(), s8, u8), int8_ok);
            ok &= IMPLICATION(d->c_type() == bf16, arch_ >= arch_t::xe_hp);

            if (!ok) return status::unimplemented;

            choose_kernel();

            // k-parallel kernels don't support post-ops.
            bool with_eltwise = (attr()->post_ops_.find(eltwise) != -1);
            ok &= IMPLICATION(tag_ == 'K', !with_bias() && !with_eltwise);

            if (!ok) return status::unimplemented;

            return status::success;
        }

        void choose_kernel() {
            using kernel_t = gen_gemm_nocopy_kernel_t;

            const auto &d = desc();
            kernel_t::choose_unrolls(arch_, dev_info_->hw_threads(),
                    eff_transa(), eff_transb(), d->a_type(), d->b_type(),
                    d->c_type(), eff_align_a(), eff_align_b(), align_c(),
                    eff_m(), eff_n(), d->k(), d->batch(), batch_dims(),
                    unroll_m_, unroll_n_, tag_);
        }

        bool set_default_formats() {
            using namespace data_type;
            using namespace format_tag;
            using arch_t = compute::gpu_arch_t;

            auto d = desc();

            auto m = d->m();
            auto n = d->n();
            auto k = d->k();
            auto a_t = d->a_type();
            auto b_t = d->b_type();
            auto c_t = d->c_type();
            auto a_t_sz = types::data_type_size(a_t);
            auto b_t_sz = types::data_type_size(b_t);

            bool is_f16 = utils::everyone_is(f16, a_t, b_t, c_t);
            bool is_xe_hp_plus = arch_ >= arch_t::xe_hp;

            // Rename memory descriptors following column major format.
            auto &a_desc = desc_.b_desc;
            auto &b_desc = desc_.a_desc;

            memory_desc_wrapper a_mdw(&a_desc);
            memory_desc_wrapper b_mdw(&b_desc);

            bool is_a_trans = a_mdw.matches_one_of_tag(ba, acb);
            bool is_b_trans = b_mdw.matches_one_of_tag(ba, acb);

            auto lda = is_a_trans ? m : k;
            auto ldb = is_b_trans ? k : n;

            auto is_aligned = [](dim_t ld, size_t sz, int byte) {
                return ld * sz % byte == 0;
            };

            bool a_4B_aligned = is_aligned(lda, a_t_sz, 4);
            bool b_4B_aligned = is_aligned(ldb, b_t_sz, 4);
            bool ab_4B_aligned = a_4B_aligned && b_4B_aligned;

            bool a_tn_4B_aligned = is_aligned(k, a_t_sz, 4);
            bool b_tn_4B_aligned = is_aligned(k, b_t_sz, 4);
            bool ab_tn_4B_aligned = a_tn_4B_aligned && b_tn_4B_aligned;

            bool use_tn = (m <= 32 || n <= 32) && !ab_4B_aligned
                    && ab_tn_4B_aligned;

            bool a_any = a_mdw.format_any();
            bool b_any = b_mdw.format_any();
            bool batch = d->is_batched();

            auto dotrans = batch ? acb : ba;
            auto notrans = batch ? abc : ab;

            if (is_f16 && is_xe_hp_plus && use_tn) {
                if (a_any && b_any) {
                    CHECK(memory_desc_init_by_tag(a_desc, dotrans));
                    CHECK(memory_desc_init_by_tag(b_desc, notrans));
                } else if (a_any && !is_b_trans) {
                    CHECK(memory_desc_init_by_tag(a_desc, dotrans));
                } else if (b_any && is_a_trans) {
                    CHECK(memory_desc_init_by_tag(b_desc, notrans));
                }
            }

            return gpu_gemm_pd_t::set_default_formats();
        }

        bool with_c_zero_points() const {
            return !attr()->zero_points_.has_default_values(DNNL_ARG_DST);
        }

        float alpha() const { return attr()->output_scales_.scales_[0]; }

        float beta() const {
            using namespace primitive_kind;
            const auto &p = attr()->post_ops_;
            return p.contain(sum, 0) ? p.entry_[0].sum.scale : 0.f;
        }

        bool with_bias() const {
            return desc()->bias_type() != data_type::undef;
        }

        int bias_cmask() const {
            unsigned char to_cmask[4] = {0, 2, 1, 3};
            return with_bias() ? to_cmask[(desc()->bias_mask() >> 1) & 3] : -1;
        }

        bool with_ab_zero_points() const { return ab_zp_; }

        bool swap_ab() const {
            bool check_lda
                    = ((desc()->transa() == dnnl_notrans && desc()->lda() == 1)
                            || (desc()->transa() == dnnl_trans));
            return (desc()->a_type() == data_type::f16 && desc()->m() == 1
                    && desc()->ldc() == 1 && check_lda);
        }

        int batch_dims() const {
            return nstl::max(desc()->c_desc.ndims - 2, 0);
        }

        int align_a() const {
            return int(utils::max_pow2_div(
                    types::data_type_size(desc()->a_type()) * desc()->lda()));
        }
        int align_b() const {
            return int(utils::max_pow2_div(
                    types::data_type_size(desc()->b_type()) * desc()->ldb()));
        }
        int align_c() const {
            return int(utils::max_pow2_div(
                    types::data_type_size(desc()->c_type()) * desc()->ldc()));
        }

        int eff_align_a() const { return !swap_ab() ? align_a() : align_b(); }
        int eff_align_b() const { return !swap_ab() ? align_b() : align_a(); }
        bool eff_transa() const {
            return !swap_ab() ? (desc()->transa() == dnnl_trans)
                              : (desc()->transb() == dnnl_notrans);
        }
        bool eff_transb() const {
            return !swap_ab() ? (desc()->transb() == dnnl_trans) : false;
        }
        dim_t eff_m() const { return !swap_ab() ? desc()->m() : desc()->n(); }
        dim_t eff_n() const { return !swap_ab() ? desc()->n() : desc()->m(); }

        size_t dyn_offset_a = 0;
        size_t dyn_offset_b = 0;
        size_t dyn_offset_c = 0;
        size_t dyn_offset_co = 0;

        bool ab_zp_ = false;
        int unroll_m_ = 0, unroll_n_ = 0;
        char tag_ = '\0';

        const compute::device_info_t *dev_info_;
        compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;
    };

    gen_gemm_t(const pd_t *apd) : gpu_gemm_t(apd) {}

    status_t init(engine_t *engine) override { return init_nocopy(engine); }

    status_t init_nocopy(engine_t *engine) {
        using kernel_t = gen_gemm_nocopy_kernel_t;

        const auto &d = pd()->desc();
        auto c_type = d->c_type();
        auto co_type = pd()->with_bias() ? d->bias_type() : c_type;
        auto acc_type = c_type;

        if (acc_type == data_type::bf16) acc_type = data_type::f32;

        if (get_verbose() >= 2) {
            char tag_s[2] = {pd()->tag_, 0};
            printf("dnnl_verbose,info,gpu,gemm,kernel:%dx%d,%s\n",
                    pd()->unroll_m_, pd()->unroll_n_, tag_s);
        }

        kernel_t kernel;

        auto status = kernel.init(pd()->arch_, pd()->batch_dims(),
                pd()->eff_transa(), pd()->eff_transb(),
                pd()->with_ab_zero_points(), pd()->with_c_zero_points(),
                pd()->with_bias(), pd()->attr()->post_ops_, d->a_type(),
                d->b_type(), c_type, co_type, acc_type, pd()->eff_align_a(),
                pd()->eff_align_b(), pd()->align_c(), pd()->unroll_m_,
                pd()->unroll_n_, pd()->tag_);

        if (status != status::success) return status;

        create_kernel(engine, &nocopy_kernel_, kernel);

        nocopy_info_ = kernel.driver_info();

        return status::success;
    }

    status_t execute(const gemm_exec_ctx_t &ctx) const override;

private:
    status_t launch_nocopy(const gemm_exec_ctx_t &ctx,
            compute::compute_stream_t *s, const memory_storage_t &a,
            const memory_storage_t &b, const memory_storage_t &c,
            const memory_storage_t &co, int64_t offset_a, int64_t offset_b,
            int64_t offset_c, int32_t offset_co, int32_t lda, int32_t ldb,
            int32_t ldc, int32_t m, int32_t n, int32_t k, int32_t k0,
            float alpha, float beta, int16_t ao, int16_t bo, int32_t cmask,
            bool last_k_block, bool swapab, bool disable_hilbert) const;

    compute::kernel_t nocopy_kernel_;
    CommonDriverInfo nocopy_info_;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
