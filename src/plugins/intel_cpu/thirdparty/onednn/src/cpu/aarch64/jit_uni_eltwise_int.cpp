/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/jit_generator.hpp"

#include "cpu/aarch64/jit_uni_eltwise_int.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;

struct jit_args_t {
    const void *from;
    const void *for_comparison;
    const void *to;
    size_t work_amount;
};

struct jit_uni_eltwise_int_kernel : public jit_generator {
    jit_uni_eltwise_int_kernel(const eltwise_desc_t &desc) : desc_(desc) {}

    void operator()(jit_args_t *p) { jit_generator::operator()(p); }

protected:
    data_type_t data_type() const { return desc_.data_desc.data_type; }
    int dtype_size() const { return types::data_type_size(data_type()); }

    const eltwise_desc_t &desc() const { return desc_; }

private:
    const eltwise_desc_t &desc_;
};

/* jit kernels */
namespace {
using namespace Xbyak_aarch64;

template <cpu_isa_t isa>
struct jit_uni_subkernel_int_t : public jit_uni_eltwise_int_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_subkernel_int)

    jit_uni_subkernel_int_t(const eltwise_desc_t &desc)
        : jit_uni_eltwise_int_kernel(desc) {
        using namespace data_type;

        // Relu and linear for int types: s32, s8, u8; Only forward direction
        assert(utils::one_of(desc.alg_kind, alg_kind::eltwise_relu,
                alg_kind::eltwise_linear));
        assert(utils::one_of(data_type(), s32, data_type::s8, u8));
        assert(isa == sve_512);
    }

    void generate() override {
        XReg param = abi_param1;

        const size_t vlen = cpu_isa_traits<isa>::vlen;
        const size_t simd_w = vlen / sizeof(float);
        const size_t loop_dec[] = {simd_w, 1};
        const size_t uf[] = {1, 1};
        const size_t shift[] = {dtype_size() * simd_w, (size_t)dtype_size()};
        const bool loop_vectorize[] = {true, false};

        preamble();

#define GET_OFF(field) offsetof(jit_args_t, field)
        add_imm(X_TMP_0, param, GET_OFF(from), X_TMP_1);
        ldr(reg_from, ptr(X_TMP_0));

        add_imm(X_TMP_0, param, GET_OFF(to), X_TMP_1);
        ldr(reg_to, ptr(X_TMP_0));

        add_imm(X_TMP_0, param, GET_OFF(work_amount), X_TMP_1);
        ldr(reg_work_amount, ptr(X_TMP_0));
#undef GET_OFF

        mov_imm(W_TMP_0, float2int(desc().alpha));
        mov_imm(W_TMP_1, float2int(desc().beta));
        dup(ts_alpha, W_TMP_0);
        dup(ts_beta, W_TMP_1);

        eor(t_zero.d, t_zero.d, t_zero.d);

        if (isa == sve_512) {
            ptrue(p_vl1.b, VL1);
            ptrue(p_all_one.b);
        }

        Label loop_label[3];

        for (int id = 0; id < 2; id++) {
            L(loop_label[id]);
            mov_imm(X_TMP_0, uf[id] * loop_dec[id] - 1);
            cmp(reg_work_amount, X_TMP_0);

            b(LE, loop_label[id + 1]);

            compute_step(
                    loop_vectorize[id], uf[id], shift[id], desc().alg_kind);

            add_imm(reg_from, reg_from, uf[id] * shift[id], X_TMP_0);
            add_imm(reg_to, reg_to, uf[id] * shift[id], X_TMP_0);
            sub_imm(reg_work_amount, reg_work_amount, uf[id] * loop_dec[id],
                    X_TMP_0);
            b(loop_label[id]);
        }

        L(loop_label[2]);
        postamble();
    }

private:
    using TReg = typename cpu_isa_traits<isa>::TReg;
    using TRegS = typename cpu_isa_traits<isa>::TRegS;

    const XReg reg_from = x1;
    const XReg reg_to = x8;
    const XReg reg_work_amount = x6;
    const XReg imm_addr64 = x3;

    const TReg t_tmp0 = TReg(31);

    const TReg t_saturation_ubound = TReg(26);
    const TRegS ts_alpha = TRegS(27);
    const TRegS ts_beta = TRegS(28);
    const TReg t_zero = TReg(29);

    const PReg p_vl1 = p0;
    const PReg p_mask = p1;
    const PReg p_mask_int8 = p_vl1; // Mask for store 1 byte in case of SVE_512
    const PReg p_all_one = p3;

    bool is32bit() const { return data_type() == data_type::s32; }

    // Load 32bit data type (s32)
    void load_32bit(
            const bool vectorize, const TReg &vr_from, const XReg &mem_from) {

        if (vectorize) {
            // load full TReg size
            uni_ldr(vr_from, mem_from);
        } else {
            // load exactly one data item
            ldr(W_TMP_0, ptr(mem_from));
            mov(vr_from.s, W_TMP_0);
        }
    }

    // Load 8bit data type (u8/s8)
    void load_8bit(const bool vectorize, const TReg &vr_from,
            const XReg &mem_from, bool is_signed) {

        // data type u8/s8 load as s32
        if (vectorize) {
            // load full TReg size
            ldr(QReg(t_tmp0.getIdx()), ptr(mem_from));
            zip1(t_tmp0.b, t_tmp0.b, t_tmp0.b);
            zip1(t_tmp0.h, t_tmp0.h, t_tmp0.h);

            if (is_signed)
                sxtb(vr_from.s, p_all_one / T_m, t_tmp0.s);
            else
                uxtb(vr_from.s, p_all_one / T_m, t_tmp0.s);
        } else {
            // load exactly one data item
            ldurb(W_TMP_0, ptr(mem_from));
            uni_clear(vr_from);

            if (is_signed)
                sxtb(W_TMP_0, W_TMP_0);
            else
                uxtb(W_TMP_0, W_TMP_0);

            mov(VReg(vr_from.getIdx()).d[0], X_TMP_0);
        }
    }

    // Load vregs with data from mem
    void load(const bool vectorize, const TReg &vr_from, const XReg &mem_from) {

        // Branching on data size
        if (is32bit())
            load_32bit(vectorize, vr_from, mem_from);
        else
            load_8bit(
                    vectorize, vr_from, mem_from, data_type() == data_type::s8);
    }

    // Processing
    void process_linear(const TReg &vr_to, const TReg &vr_from);
    void process_relu(const TReg &vr_to, const TReg &vr_from);

    // Store s32 for any isa
    void store_32bit(
            const bool vectorize, const XReg &mem_to, const TReg &vr_to) {
        if (vectorize) {
            // store full TReg size
            uni_str(vr_to, mem_to);
        } else {
            // store exactly one data item
            st1w(vr_to.s, p_vl1, ptr(mem_to));
        }
    }

    // Store 8 bit int - isa-dependent
    void store_8bit(const bool vectorize, const XReg &mem_to, const TReg &vr_to,
            bool is_signed);

    // Store results from vregs to mem
    void store(const bool vectorize, const XReg &mem_to, const TReg &vr_to) {
        // Branching on data size
        if (is32bit())
            store_32bit(vectorize, mem_to, vr_to);
        else
            store_8bit(vectorize, mem_to, vr_to, data_type() == data_type::s8);
    }

    void compute_step(bool vectorize, const size_t uf, const size_t shift,
            const alg_kind_t alg) {

        auto vreg_from = [&](const size_t i) -> TReg { return TReg(i + 1); };
        auto vreg_to = [&](const size_t i) -> TReg { return TReg(uf + i + 1); };

        // 1. Load (vregs <- mem)
        for (size_t i = 0; i < uf; i++) {
            add_imm(reg_from, reg_from, i * shift, X_TMP_0);
            load(vectorize, vreg_from(i), reg_from);
        }

        // 2. Process (vregs <- vergs)
        switch (alg) {
            case alg_kind::eltwise_linear:
                for (size_t i = 0; i < uf; i++)
                    process_linear(vreg_to(i), vreg_from(i));
                break;
            case alg_kind::eltwise_relu:
                for (size_t i = 0; i < uf; i++)
                    process_relu(vreg_to(i), vreg_from(i));
                break;
            default: assert(!"unsupported alg");
        }

        // 3. Store (mem <- vregs)
        for (size_t i = 0; i < uf; i++) {
            add_imm(reg_to, reg_to, i * shift, X_TMP_0);
            store(vectorize, reg_to, vreg_to(i));
        }
    }
};

template <cpu_isa_t isa>
void jit_uni_subkernel_int_t<isa>::process_linear(
        const TReg &vr_to, const TReg &vr_from) {

    scvtf(vr_to.s, p_all_one / T_m, vr_from.s);
    fmad(vr_to.s, p_all_one / T_m, ts_alpha, ts_beta);

    // Saturate before converting from f32 to s32
    XReg reg_tmp = x10;

    uni_clear(t_zero);
    init_saturate_f32(
            t_zero, t_saturation_ubound, reg_tmp, data_type::f32, data_type());
    saturate_f32(vr_to, t_zero, t_saturation_ubound, data_type(), p_all_one);

    frinti(vr_to.s, p_all_one / T_m, vr_to.s);
    fcvtzs(vr_to.s, p_all_one / T_m, vr_to.s);
}

template <cpu_isa_t isa>
void jit_uni_subkernel_int_t<isa>::process_relu(
        const TReg &vr_to, const TReg &vr_from) {
    assert(!"unsupported isa");
}

template <>
void jit_uni_subkernel_int_t<sve_512>::process_relu(
        const TReg &vr_to, const TReg &vr_from) {

    scvtf(vr_from.s, p_all_one / T_m, vr_from.s);

    fmul(vr_to.s, vr_from.s, ts_alpha);

    fcmgt(p_mask.s, p_all_one / T_z, vr_from.s, t_zero.s);

    sel(vr_to.s, p_mask / T_m, vr_from.s, vr_to.s);

    frinti(vr_to.s, p_all_one / T_m, vr_to.s);
    fcvtzs(vr_to.s, p_all_one / T_m, vr_to.s);
}

template <cpu_isa_t isa>
void jit_uni_subkernel_int_t<isa>::store_8bit(const bool vectorize,
        const XReg &mem_to, const TReg &vr_to, bool is_signed) {
    assert(!"unsupported isa");
}

template <>
void jit_uni_subkernel_int_t<sve_512>::store_8bit(const bool vectorize,
        const XReg &mem_to, const TReg &vr_to, bool is_signed) {
    if (vectorize) {
        // store full TReg size
        mov(t_tmp0.d, vr_to.d);
        if (is_signed) {
            smin(t_tmp0.s, 127);
            smax(t_tmp0.s, -128);
        } else {
            umin(t_tmp0.s, 255);
        }
        st1b(t_tmp0.s, p_all_one, ptr(mem_to));
    } else {
        // store exactly one data item
        // s32 save as s8/u8
        mov(t_tmp0.d, vr_to.d);
        if (is_signed) {
            smin(t_tmp0.s, 127);
            smax(t_tmp0.s, -128);
        } else {
            umin(t_tmp0.s, 255);
        }
        st1b(t_tmp0.s, p_mask_int8, ptr(mem_to));
    }
}

} /* namespace */

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_int_fwd_t<isa, d_type>::pd_t::init(engine_t *engine) {
    bool ok = mayiuse(isa)
            && desc()->data_desc.data_type == d_type
            // only relu and linear so far
            && utils::one_of(desc()->alg_kind, alg_kind::eltwise_relu,
                    alg_kind::eltwise_linear)
            && !has_zero_dim_memory()
            && memory_desc_wrapper(src_md()).is_dense(true)
            && attr()->has_default_values();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_int_fwd_t<isa, d_type>::jit_uni_eltwise_int_fwd_t(
        const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_int_fwd_t<isa, d_type>::init(engine_t *engine) {
    const auto &desc = *pd()->desc();
    CHECK(safe_ptr_assign(kernel_, new jit_uni_subkernel_int_t<isa>(desc)));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_int_fwd_t<isa, d_type>::~jit_uni_eltwise_int_fwd_t() {
    delete kernel_;
}

template <cpu_isa_t isa, impl::data_type_t d_type>
status_t jit_uni_eltwise_int_fwd_t<isa, d_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper data_d(pd()->data_md());

    const size_t nelems = data_d.nelems(true);

    src += data_d.offset0();
    dst += data_d.offset0();

    const int cache_line = 64 / data_d.data_type_size();
    parallel(0, [&](const int ithr, const int nthr) {
        size_t start {0}, end {0};

        balance211(utils::div_up(nelems, cache_line), nthr, ithr, start, end);
        start = nstl::min(nelems, start * cache_line);
        end = nstl::min(nelems, end * cache_line);

        auto arg = jit_args_t();
        arg.from = (const void *)&src[start];
        arg.for_comparison = (const void *)&src[start];
        arg.to = (const void *)&dst[start];
        arg.work_amount = end - start;
        if (arg.work_amount) (*kernel_)(&arg);
    });
    return status::success;
}

using namespace data_type;

template struct jit_uni_eltwise_int_fwd_t<sve_512, s32>;
template struct jit_uni_eltwise_int_fwd_t<sve_512, data_type::s8>;
template struct jit_uni_eltwise_int_fwd_t<sve_512, u8>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
