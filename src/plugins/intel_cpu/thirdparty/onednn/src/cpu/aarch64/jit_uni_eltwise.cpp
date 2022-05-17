/*******************************************************************************
* Copyright 2017-2021 Intel Corporation
* Copyright 2021 FUJITSU LIMITED
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

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/jit_generator.hpp"

#include "cpu/aarch64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/aarch64/jit_uni_eltwise.hpp"

#define GET_OFF(field) offsetof(jit_args_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;

struct jit_args_t {
    const void *src; // fwd: src;  bwd: src/dst based on alg;
    const void *dst; // fwd: dst;  bwd: diff_src;
    const void *diff_dst; // fwd: nullptr;  bwd: diff_dst;
    size_t work_amount;
};

struct jit_uni_eltwise_kernel : public jit_generator {
    jit_uni_eltwise_kernel(const eltwise_pd_t *pd) : pd_(pd) {}

    void operator()(jit_args_t *p) { jit_generator::operator()(p); }

protected:
    const eltwise_pd_t *pd_;

    data_type_t data_type() const { return pd_->src_md()->data_type; }
    bool is_bf16() const { return data_type() == data_type::bf16; }
    int dtype_size() const { return types::data_type_size(data_type()); }
};

// jit kernels
namespace {

template <cpu_isa_t isa>
struct jit_uni_kernel_t : public jit_uni_eltwise_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_kernel)

    jit_uni_kernel_t(const eltwise_pd_t *pd) : jit_uni_eltwise_kernel(pd) {
        const auto &desc = *pd_->desc();
        // there's no auxiliary vregs on fwd path
        const bool is_fwd = pd_->is_fwd();
        const bool save_state = is_fwd ? false : true;
        eltwise_injector_.reset(new jit_uni_eltwise_injector_f32<isa>(this,
                desc.alg_kind, desc.alpha, desc.beta, 1.f, save_state,
                reg_injector_table, injector_mask, injector_p_tmp0,
                injector_p_all, is_fwd, pd_->use_dst()));
    }

    void generate() override {
        const bool is_fwd = pd_->is_fwd();
        preamble();

        XReg param = param1;
        add_imm(X_TMP_0, param, GET_OFF(src), X_TMP_1);
        ldr(reg_src, ptr(X_TMP_0));
        add_imm(X_TMP_0, param, GET_OFF(dst), X_TMP_1);
        ldr(reg_dst, ptr(X_TMP_0));
        if (!is_fwd) {
            add_imm(X_TMP_0, param, GET_OFF(diff_dst), X_TMP_1);
            ldr(reg_diff_dst, ptr(X_TMP_0));
        }
        add_imm(X_TMP_0, param, GET_OFF(work_amount), X_TMP_1);
        ldr(reg_work_amount, ptr(X_TMP_0));
        eltwise_injector_->load_table_addr();
        ptrue(p_512.b);

        Label reminder_loop_start, reminder_loop_end;
        Label vectorized_loop_start, vectorized_loop_end;

        cmp(reg_work_amount, simd_w());
        b(LT, reminder_loop_start);

        L(vectorized_loop_start);

        // TODO: consider improving.
        // This piece of code is responsible for the preserve_zero function
        // being a natural restriction of this implementation. It works with any
        // dense and blocked layout, but the problem raises when blocking
        // dimension is not divisible by block size. For such case, the code
        // below should save the mask, where zero padding should be preserved
        // and apply it on register before storing into dst memory. Until
        // there's a restriction on certain blocked layouts, when this behavior
        // can be relevantly easy controlled, this will cost much from code
        // perspective and will complicate the compute logic significantly.
        ldr(vmm_src, ptr(reg_src));
        eltwise_injector_->compute_vector(vmm_src.getIdx());
        if (!is_fwd) {
            ldr(ZReg(vmm_diff_dst.getIdx()), ptr(reg_diff_dst));
            fmul(vmm_src.s, vmm_src.s, vmm_diff_dst);
        }
        str(vmm_src, ptr(reg_dst));

        const auto shift = cpu_isa_traits<isa>::vlen;
        add_imm(reg_src, reg_src, shift, X_TMP_0);
        add_imm(reg_dst, reg_dst, shift, X_TMP_0);
        if (!is_fwd) add_imm(reg_diff_dst, reg_diff_dst, shift, X_TMP_0);

        sub_imm(reg_work_amount, reg_work_amount, simd_w(), X_TMP_0);
        cmp(reg_work_amount, simd_w());
        b(GE, vectorized_loop_start);

        L(vectorized_loop_end);

        L(reminder_loop_start);

        cmp(reg_work_amount, 0);
        b(LE, reminder_loop_end);

        ld1(xmm_src[0], ptr(reg_src));
        eltwise_injector_->compute_vector(xmm_src.getIdx());
        if (!is_fwd) {
            ld1(xmm_diff_dst[0], ptr(reg_diff_dst));
            fmul(xmm_src, xmm_src, xmm_diff_dst);
        }
        st1(xmm_src[0], ptr(reg_dst));
        add_imm(reg_src, reg_src, dtype_size(), X_TMP_0);
        add_imm(reg_dst, reg_dst, dtype_size(), X_TMP_0);
        if (!is_fwd) add_imm(reg_diff_dst, reg_diff_dst, dtype_size(), X_TMP_0);

        subs(reg_work_amount, reg_work_amount, 1);
        b(reminder_loop_start);

        L(reminder_loop_end);

        postamble();

        eltwise_injector_->prepare_table();
    }

private:
    using TReg = typename cpu_isa_traits<isa>::TReg;
    using TRegS = typename cpu_isa_traits<isa>::TRegS;

    int simd_w() {
        int simd_w = cpu_isa_traits<isa>::vlen / dtype_size();
        /* Return value is used for CMP (immediate). */
        assert(simd_w < (1 << 12));
        return simd_w;
    }

    XReg reg_src = x11;
    XReg reg_dst = x8;
    XReg reg_injector_table = x9;
    XReg reg_diff_dst = x10;
    XReg reg_work_amount = x6;
    XReg imm_addr64 = x3;
    PReg injector_mask = p1;
    PReg injector_p_tmp0 = p4;
    PReg injector_p_all = p7;

    VReg4S xmm_src {1};
    TReg vmm_src {1};
    VReg4S xmm_diff_dst {2};
    TRegS vmm_diff_dst {2};
    std::unique_ptr<jit_uni_eltwise_injector_f32<isa>> eltwise_injector_;

    PReg p_512 {7}; /* Index is temporal. */
    PReg p_tmp0 {4}; /* Index is temporal. */
};

} // namespace

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_fwd_t<isa, d_type>::pd_t::init(engine_t *engine) {
    using namespace alg_kind;

    const memory_desc_wrapper data_d(src_md());

    bool ok = mayiuse(isa) && is_fwd() && src_md()->data_type == d_type
            && !has_zero_dim_memory()
            && data_d.is_dense(true)
            // refer to a comment in jit_uni_kernel why this is needed
            && IMPLICATION(!data_d.is_dense(), is_zero_preserved())
            && attr()->has_default_values();

    ok &= utils::one_of(desc_.alg_kind, eltwise_relu_use_dst_for_bwd,
            eltwise_relu, eltwise_elu_use_dst_for_bwd, eltwise_elu,
            eltwise_tanh_use_dst_for_bwd, eltwise_tanh, eltwise_square,
            eltwise_abs, eltwise_sqrt_use_dst_for_bwd, eltwise_sqrt,
            eltwise_linear, eltwise_bounded_relu, eltwise_soft_relu,
            eltwise_logistic_use_dst_for_bwd, eltwise_logistic,
            eltwise_exp_use_dst_for_bwd, eltwise_exp, eltwise_gelu_tanh,
            eltwise_swish, eltwise_log, eltwise_clip, eltwise_gelu_erf,
            eltwise_round);

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_fwd_t<isa, d_type>::jit_uni_eltwise_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_fwd_t<isa, d_type>::~jit_uni_eltwise_fwd_t() = default;

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_fwd_t<isa, d_type>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, new jit_uni_kernel_t<isa>(pd())));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_fwd_t<isa, d_type>::execute(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper data_d(pd()->src_md());
    const auto nelems = data_d.nelems(true);
    const int simd_w = 64 / data_d.data_type_size();

    src += data_d.offset0();
    dst += data_d.offset0();

    parallel(0, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};

        balance211(utils::div_up(nelems, simd_w), nthr, ithr, start, end);
        start = nstl::min(nelems, start * simd_w);
        end = nstl::min(nelems, end * simd_w);
        if (start == end) return;

        jit_args_t args;
        args.src = src + start;
        args.dst = dst + start;
        args.diff_dst = nullptr;
        args.work_amount = end - start;
        (*kernel_)(&args);
    });

    return status::success;
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_bwd_t<isa, d_type>::pd_t::init(engine_t *engine) {
    using namespace alg_kind;

    const memory_desc_wrapper data_d(src_md());

    bool ok = mayiuse(isa) && !is_fwd()
            && utils::everyone_is(
                    d_type, src_md()->data_type, diff_src_md()->data_type)
            && !has_zero_dim_memory() && set_default_formats_common()
            && data_d.is_dense(true)
            // refer to a comment in jit_uni_kernel why this is needed
            && IMPLICATION(!data_d.is_dense(), is_zero_preserved())
            && data_d == memory_desc_wrapper(diff_dst_md())
            && attr()->has_default_values();

    ok &= utils::one_of(desc_.alg_kind, eltwise_relu_use_dst_for_bwd,
            eltwise_relu, eltwise_elu_use_dst_for_bwd, eltwise_elu,
            eltwise_tanh_use_dst_for_bwd, eltwise_tanh, eltwise_square,
            eltwise_abs, eltwise_sqrt_use_dst_for_bwd, eltwise_sqrt,
            eltwise_linear, eltwise_bounded_relu, eltwise_soft_relu,
            eltwise_logistic_use_dst_for_bwd, eltwise_logistic,
            eltwise_exp_use_dst_for_bwd, eltwise_exp, eltwise_gelu_tanh,
            eltwise_swish, eltwise_log, eltwise_clip, eltwise_gelu_erf);

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_bwd_t<isa, d_type>::jit_uni_eltwise_bwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_bwd_t<isa, d_type>::~jit_uni_eltwise_bwd_t() = default;

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_bwd_t<isa, d_type>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, new jit_uni_kernel_t<isa>(pd())));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_bwd_t<isa, d_type>::execute(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto src = pd()->use_dst() ? CTX_IN_MEM(const data_t *, DNNL_ARG_DST)
                               : CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DIFF_SRC, status);
    CHECK(status);

    const memory_desc_wrapper data_d(pd()->src_md());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_md());
    const auto nelems = data_d.nelems(true);
    const int simd_w = 64 / data_d.data_type_size();

    src += data_d.offset0();
    diff_dst += diff_data_d.offset0();
    diff_src += diff_data_d.offset0();

    parallel(0, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};

        balance211(utils::div_up(nelems, simd_w), nthr, ithr, start, end);
        start = nstl::min(nelems, start * simd_w);
        end = nstl::min(nelems, end * simd_w);
        if (start == end) return;

        jit_args_t args;
        args.src = src + start;
        args.dst = diff_src + start;
        args.diff_dst = diff_dst + start;
        args.work_amount = end - start;
        (*kernel_)(&args);
    });

    return status::success;
}

template struct jit_uni_eltwise_fwd_t<sve_512, data_type::f32>;
template struct jit_uni_eltwise_bwd_t<sve_512, data_type::f32>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
