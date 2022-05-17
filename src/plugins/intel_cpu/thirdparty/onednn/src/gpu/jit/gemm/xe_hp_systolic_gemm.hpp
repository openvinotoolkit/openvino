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

#ifndef GPU_JIT_XE_HP_SYSTOLIC_GEMM_HPP
#define GPU_JIT_XE_HP_SYSTOLIC_GEMM_HPP

#include <assert.h>
#include <memory>
#include <tuple>

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "common/memory_storage.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gpu_gemm_pd.hpp"
#include "gpu/jit/gemm/gen_gemm_kernel.hpp"
#include "gpu/jit/gemm/xe_hp_systolic_gemm_kernel.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

struct xe_hp_systolic_gemm_t : public gpu_gemm_t {
    struct pd_t : public gpu_gemm_pd_t {
        using hint_class = void;

        pd_t(const gemm_desc_t *adesc, const primitive_attr_t *attr,
                const hint_class *)
            : gpu_gemm_pd_t(adesc, attr, nullptr) {}

        DECLARE_COMMON_PD_T("jit:xe_hp:gemm:any", xe_hp_systolic_gemm_t);

        status_t init(engine_t *engine);

        bool use_fma();
        bool set_default_formats(data_type_t dt);

        size_t dyn_offset_a = 0;
        size_t dyn_offset_b = 0;
        size_t dyn_offset_c = 0;

        data_type_t impl_acc_type() const {
            using namespace data_type;
            return utils::one_of(desc()->c_type(), f16, bf16, f32) ? f32 : s32;
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

        bool packed_a() const { return packed_a_; }
        bool packed_b() const { return packed_b_; }
        bool packed_c() const { return packed_c_; }

        dim_t lda_packed() const {
            return packed_a() ? desc()->b_desc.padded_dims[with_batch() ? 1 : 0]
                              : 0;
        }
        dim_t ldb_packed() const {
            return packed_b() ? desc()->a_desc.padded_dims[with_batch() ? 2 : 1]
                              : 0;
        }
        dim_t ldc_packed() const {
            return packed_c() ? desc()->c_desc.padded_dims[with_batch() ? 2 : 1]
                              : 0;
        }

        bool with_batch() const { return desc()->is_batched(); }
        bool with_ab_zero_points() const { return a_zp_ || b_zp_; }
        bool with_c_zero_points() const { return c_zp_; }

        int unroll_m() const { return unroll_m_; }
        int unroll_n() const { return unroll_n_; }
        bool use_new_kernels() const { return use_new_kernels_; }
        char kernel_tag() const { return kernel_tag_; }

        const compute::device_info_t *dev_info_ = nullptr;

    private:
        bool any_prepacked_ = false;
        bool packed_a_ = false, packed_b_ = false, packed_c_ = false;
        bool a_zp_ = false, b_zp_ = false, c_zp_ = false;
        bool use_new_kernels_ = false;
        int unroll_m_ = 0;
        int unroll_n_ = 0;
        char kernel_tag_ = '\0';
    };

    status_t init(engine_t *engine) override;
    status_t init_res_storage(
            engine_t *engine, gpu_resource_t *r) const override;

public:
    xe_hp_systolic_gemm_t(const pd_t *apd) : gpu_gemm_t(apd) {}

    virtual status_t execute(const gemm_exec_ctx_t &ctx) const override;

private:
    status_t init_compute_old(engine_t *engine);
    status_t init_compute_new(engine_t *engine);

    bool enable_mn_blocking() const;
    std::tuple<int64_t, int64_t, int64_t> get_blocking() const;

    status_t launch_clear_sum(const gemm_exec_ctx_t &ctx, int64_t r, int64_t c,
            const memory_storage_t &dst, int32_t offset_dst, int32_t ld_dst,
            bool copyb) const;
    status_t launch_copy(const gemm_exec_ctx_t &ctx, int64_t r, int64_t c,
            const memory_storage_t &src, int64_t offset_src, int64_t ld_src,
            const memory_storage_t &dst, int32_t offset_dst, int32_t ld_dst,
            bool copyb) const;
    status_t launch_compute(const gemm_exec_ctx_t &ctx, int32_t m, int32_t n,
            int32_t k, const memory_storage_t &ap, int64_t offset_a,
            int32_t lda, const memory_storage_t &bp, int64_t offset_b,
            int32_t ldb, const memory_storage_t &c, int64_t offset_c,
            int32_t ldc, float alpha, float beta, int16_t ao, int16_t bo,
            const memory_storage_t &co, int32_t offset_co, bool first_k_block,
            bool last_k_block, int32_t batch, int32_t stride_a,
            int32_t stride_b, int32_t stride_c) const;

    static int64_t nice_ld(int64_t ld, int sz, bool get_max = false) {
        const auto align = 32;
        const auto no_align = 64;

        auto new_ld = (ld * sz + align - 1) & ~(align - 1);
        if (get_max || (new_ld & (no_align - 1)) == 0) new_ld += align;

        return new_ld / sz;
    }

    int64_t get_ld_packed(int64_t k, bool get_max = false) const {
        using compute_kernel_t = xehp_systolic_gemm_kernel_t<gpu_xe_hp>;

        auto a_type = pd()->desc()->a_type();
        auto a_sz = types::data_type_size(a_type);

        auto ld = utils::rnd_up(k, compute_kernel_t::unroll_k(a_type));
        if (pd()->with_ab_zero_points()) ld += 32 / a_sz;

        return nice_ld(ld, int(a_sz), get_max);
    }

    int64_t max_ld_packed(int64_t k) const { return get_ld_packed(k, true); }

    static const int A_PACKED_ = 0;
    static const int B_PACKED_ = 1;

    compute::kernel_t kernel_[2][2]; // [first_k_block][last_k_block]
    compute::kernel_t copy_kernel_[2][2]; // [trans][clear_sum]

    CommonDriverInfo compute_info_;

    compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;
    int eu_count_ = 0;

    char co_kind_ = 'N';
    bool walk_n_first_ = false;

    const pd_t *pd() const { return (const pd_t *)gpu_primitive_t::pd().get(); }
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
