/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <assert.h>

#include "c_types_map.hpp"
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

#include "jit_uni_eltwise.hpp"
#include "jit_uni_softmax.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace {

typedef float data_t;

using namespace Xbyak;

template <cpu_isa_t isa>
struct jit_softmax_t: public jit_generator {
    struct call_params_t {
        // keep all sizes at 8 bytes -- jit code expects this
        const data_t *src, *dst;
        size_t soff_max;
    };
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_softmax_t)

    // cpu specific part
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const AddressFrame &vmmword = (isa == sse42) ? xword :
                                  (isa == avx2) ? yword : zword;
    const int vlen = cpu_isa_traits<isa>::vlen;

    const softmax_fwd_pd_t *sdesc_;

    void (*ker)(const call_params_t *);
    void operator()(const call_params_t *p) { (*ker)(p); }
    jit_uni_eltwise_injector_f32<isa> *eltwise_injector_;

    Reg64 reg_param = abi_param1;

    Reg64 reg_injector_table = rax;
    Reg64 reg_src = r8;
    Reg64 reg_dst = r9;
    Reg64 reg_soff = r10;
    Reg64 reg_soff_max = r11;
    Reg64 reg_rsoff = r12;
    Reg64 reg_tmp = r13;

    Opmask injector_mask = Opmask(1);
    Opmask ktail_mask = Opmask(2); // axis tail processing

    Vmm vtmp; // assigned at placed where used
    Vmm vtail_mask = Vmm(0);
    Xmm xneg_flt_max = Xmm(12);
    Vmm vneg_flt_max = Vmm(isa == avx512_common ? 28 : 12);
    Xmm xone = Xmm(13);
    Vmm vone = Vmm(isa == avx512_common ? 29 : 13);
    Vmm vsum = Vmm(isa == avx512_common ? 30 : 14);
    Vmm vmax = Vmm(isa == avx512_common ? 31 : 15);

    size_t simd_w_;
    size_t axis_simd_full_;
    size_t axis_simd_tail_;
    size_t unroll_regs_;
    size_t n_loops_;
    size_t loop_tail_;

    void compute_predefined_variables() {
        simd_w_ = vlen / sizeof(data_t);
        axis_simd_full_ = sdesc_->axis_size() / simd_w_;
        axis_simd_tail_ = sdesc_->axis_size() % simd_w_;
        unroll_regs_ = 4;
        n_loops_ = axis_simd_full_ / unroll_regs_;
        loop_tail_ = axis_simd_full_ - n_loops_ * unroll_regs_;
    }

    void load_common_params() {
        mov(reg_tmp, float2int(1.0f));
        movq(xone, reg_tmp);
        uni_vbroadcastss(vone, xone);
        mov(reg_tmp, float2int(-FLT_MAX));
        movq(xneg_flt_max, reg_tmp);
        uni_vbroadcastss(vneg_flt_max, xneg_flt_max);

#       define PARAM_OFF(x) offsetof(call_params_t, x)
        mov(reg_soff_max, ptr[reg_param + PARAM_OFF(soff_max)]);
        mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
        mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
#       undef PARAM_OFF
    }

    void prepare_tail_mask_sse42() {
        if (!axis_simd_tail_) return;

        static const uint32_t mask_f32[8] = {0xffffffff, 0, 0, 0};
        mov(reg_tmp, reinterpret_cast<size_t>(mask_f32));
        movups(vtail_mask, ptr[reg_tmp]);
    }

    void prepare_tail_mask_avx2() {
        if (!axis_simd_tail_) return;

        static const uint32_t mask_f32[16] = {0xffffffff, 0xffffffff,
                0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                0xffffffff, 0, 0, 0, 0, 0, 0, 0, 0};

        mov(reg_tmp, reinterpret_cast<size_t>(
                    &mask_f32[8 - axis_simd_tail_ % simd_w_]));
        vmovups(vtail_mask, ptr[reg_tmp]);
    }

    void prepare_tail_mask_avx512() {
        if (!axis_simd_tail_) return;

        const int mask_f32 = (1 << axis_simd_tail_) - 1;

        Reg32 regw_tmp = reg_tmp.cvt32();
        mov(regw_tmp, mask_f32);
        kmovw(ktail_mask, regw_tmp);
    }

    void uni_vmovups_tail(const Operand &dst, const Operand &src) {
        if (isa == avx512_common)
            uni_vmovups_tail_avx512(dst, src);
        else if (isa == avx2)
            uni_vmovups_tail_avx2(dst, src);
    }

    void uni_vmovups_tail_avx2(const Operand &dst, const Operand &src) {
        if (dst.isMEM())
            vmaskmovps(dst.getAddress(), vtail_mask, Vmm(src.getIdx()));
        else
            vmaskmovps(Vmm(dst.getIdx()), vtail_mask, src.getAddress());
    }

    void uni_vmovups_tail_avx512(const Operand &dst, const Operand &src) {
        if (dst.isMEM())
            vmovups(dst.getAddress() | ktail_mask, Vmm(src.getIdx()));
        else
            vmovups(Vmm(dst.getIdx()) | ktail_mask | T_z, src.getAddress());
    }

    Address src_ptr(size_t offt = 0) {
        return vmmword[reg_src + reg_soff + offt];
    }

    Address dst_ptr(size_t offt = 0) {
        return vmmword[reg_dst + reg_soff + offt];
    }

    enum class op_t : unsigned { max, sum };

    void perform_op(Vmm v, Vmm vtmp, op_t op) {
        if (op == op_t::max) uni_vmaxps(v, v, vtmp);
        else if (op == op_t::sum) uni_vaddps(v, v, vtmp);
    }

    void get_horizontal_op(Vmm &v, Vmm &vtmp, op_t op) {
        if (isa == avx512_common) {
            vshuff32x4(vtmp, v, v, 0x4E); // 256-bit shuffle
            perform_op(v, vtmp, op);
            vshuff32x4(vtmp, v, v, 0xB1); // 128/256-bit shuffle
        } else if (isa == avx2) {
            vperm2f128(vtmp, v, v, 0x1); // 128/256-bit shuffle
        }
        perform_op(v, vtmp, op);
        vshufps(vtmp, v, v, 0x4E); // 64/128-bit shuffle
        perform_op(v, vtmp, op);
        vshufps(vtmp, v, v, 0xB1); // 32/64-bit shuffle
        perform_op(v, vtmp, op);
    }

    template <typename body_t>
    void axis_loop(body_t body) {
        Label main_loop, tail_loop, tail_axis;

        mov(reg_rsoff, reg_soff_max); // reverse soff to dispatch between labels
        xor_(reg_soff, reg_soff);     // soff to get addr of src/dst
        L(main_loop); {
            if (n_loops_) {
                cmp(reg_rsoff, unroll_regs_ * vlen);
                jl(tail_loop, T_NEAR);

                body(unroll_regs_);
                sub(reg_rsoff, unroll_regs_ * vlen);
                add(reg_soff, unroll_regs_ * vlen);
                jmp(main_loop);
            }
        }

        L(tail_loop); {
            if (loop_tail_) {
                body(loop_tail_);
                add(reg_soff, loop_tail_ * vlen);
            }
        }

        L(tail_axis); {
            if (axis_simd_tail_) {
                body(1, true);
            }
        }
    }

    void forward() {
        auto accumulate_vmax = [&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                if (!tail)
                    uni_vmaxps(vmax, vmax, src_ptr(vlen * i));
                else {
                    if (isa == avx512_common)
                        uni_vmaxps(vmax | ktail_mask, vmax, src_ptr(vlen * i));
                    else if (isa == avx2) {
                        vtmp = Vmm(i + 1);
                        uni_vmovups_tail(vtmp, src_ptr(vlen * i));
                        uni_vblendvps(vtmp, vneg_flt_max, vtmp, vtail_mask);
                        uni_vmaxps(vmax, vmax, vtmp);
                    }
                }
            }
        };

        auto accumulate_vsum = [&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg = Vmm(i + 1);
                if (!tail) {
                    uni_vmovups(vreg, src_ptr(vlen * i));
                    uni_vsubps(vreg, vreg, vmax);
                    eltwise_injector_->compute_vector(vreg.getIdx());
                    uni_vaddps(vsum, vsum, vreg);
                    uni_vmovups(dst_ptr(vlen * i), vreg);
                } else {
                    uni_vmovups_tail(vreg, src_ptr(vlen * i));
                    uni_vsubps(vreg, vreg, vmax);
                    eltwise_injector_->compute_vector(vreg.getIdx());
                    if (isa == avx512_common)
                        uni_vaddps(vsum | ktail_mask, vsum, vreg);
                    else if (isa == avx2) {
                        vtmp = Vmm(vreg.getIdx() + 1); // next after vreg
                        uni_vpxor(vtmp, vtmp, vtmp);
                        uni_vblendvps(vtmp, vtmp, vreg, vtail_mask);
                        uni_vaddps(vsum, vsum, vtmp);
                    }
                    uni_vmovups_tail(dst_ptr(vlen * i), vreg);
                }
            }
        };

        auto compute_dst = [&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg = Vmm(i + 1);
                if (!tail) {
                    uni_vmulps(vreg, vsum, dst_ptr(vlen * i));
                    uni_vmovups(dst_ptr(vlen * i), vreg);
                } else {
                    if (isa == avx512_common)
                        uni_vmulps(vreg | ktail_mask, vsum, dst_ptr(vlen * i));
                    else if (isa == avx2) {
                        uni_vmovups_tail(vreg, dst_ptr(vlen * i));
                        uni_vmulps(vreg, vreg, vsum);
                    }
                    uni_vmovups_tail(dst_ptr(vlen * i), vreg);
                }
            }
        };

        uni_vmovups(vmax, vneg_flt_max); // flush to -FLT_MAX before accumulation
        axis_loop(accumulate_vmax);

        get_horizontal_op(vmax, vtmp = vsum, op_t::max);

        uni_vpxor(vsum, vsum, vsum); // flush to zero before accumulation
        axis_loop(accumulate_vsum);

        get_horizontal_op(vsum, vtmp = vmax, op_t::sum);

        uni_vdivps(vsum, vone, vsum);
        axis_loop(compute_dst);
    }

    jit_softmax_t(const softmax_fwd_pd_t *sdesc): sdesc_(sdesc) {
        static_assert(utils::one_of(isa, sse42, avx2, avx512_common),
                "unsupported isa");

        compute_predefined_variables();

        eltwise_injector_ = new jit_uni_eltwise_injector_f32<isa>(
                this, alg_kind::eltwise_exp, 0.0f, 0.0f, true,
                reg_injector_table, injector_mask);

        preamble();

        eltwise_injector_->load_table_addr();

        if (isa == avx512_common)
            prepare_tail_mask_avx512();
        else if (isa == avx2)
            prepare_tail_mask_avx2();
        else if (isa == sse42)
            prepare_tail_mask_sse42();

        load_common_params();
        forward();

        postamble();

        eltwise_injector_->prepare_table();

        ker = reinterpret_cast<decltype(ker)>(const_cast<uint8_t*>(
                    this->getCode()));
    }

    ~jit_softmax_t() {
        delete eltwise_injector_;
    }
};

// keep two sse42 functions separately to have common part human-friendly code
template <>
void jit_softmax_t<sse42>::uni_vmovups_tail(
        const Operand &dst, const Operand &src) = delete;
template <>
void jit_softmax_t<sse42>::uni_vmovups_tail_avx2(
        const Operand &dst, const Operand &src) = delete;
template <>
void jit_softmax_t<sse42>::uni_vmovups_tail_avx512(
        const Operand &dst, const Operand &src) = delete;

template <>
void jit_softmax_t<sse42>::get_horizontal_op(Vmm &v, Vmm &vtmp, op_t op) {
    uni_vmovups(vtmp, v);
    shufps(vtmp, vtmp, 0x4E); // 64/128-bit shuffle
    perform_op(v, vtmp, op);
    uni_vmovups(vtmp, v);
    shufps(vtmp, vtmp, 0xB1); // 32/64-bit shuffle
    perform_op(v, vtmp, op);
}

template <>
void jit_softmax_t<sse42>::forward() {
    auto accumulate_vmax = [&](int unroll, bool tail = false) {
        for (int i = 0; i < unroll; i++) {
            Vmm vreg = Vmm(i + 1);
            if (!tail) {
                // SIGSEGV on unaligned addr if do maxps directly on memory
                uni_vmovups(vreg, src_ptr(vlen * i));
                uni_vmaxps(vmax, vmax, vreg);
            } else {
                vtmp = Vmm(vreg.getIdx() + 1); // next after vreg

                for (size_t j = 0; j < axis_simd_tail_; j++) {
                    uni_vmovups(vreg, vneg_flt_max);
                    uni_vmovss(vtmp, src_ptr(vlen * i + sizeof(data_t) * j));
                    uni_vblendvps(vreg, vreg, vtmp, vtail_mask);
                    uni_vmaxps(vmax, vmax, vreg);
                }
            }
        }
     };

    auto accumulate_vsum = [&](int unroll, bool tail = false) {
        for (int i = 0; i < unroll; i++) {
            Vmm vreg = Vmm(i + 1);
            if (!tail) {
                uni_vmovups(vreg, src_ptr(vlen * i));
                uni_vsubps(vreg, vreg, vmax);
                eltwise_injector_->compute_vector(vreg.getIdx());
                uni_vaddps(vsum, vsum, vreg);
                uni_vmovups(dst_ptr(vlen * i), vreg);
            } else {
                vtmp = Vmm(vreg.getIdx() + 1); // next after vreg

                for (size_t j = 0; j < axis_simd_tail_; j++) {
                    uni_vmovss(vreg, src_ptr(vlen * i + sizeof(data_t) * j));
                    uni_vsubps(vreg, vreg, vmax);
                    eltwise_injector_->compute_vector(vreg.getIdx());
                    uni_vpxor(vtmp, vtmp, vtmp);
                    uni_vblendvps(vtmp, vtmp, vreg, vtail_mask);
                    uni_vaddps(vsum, vsum, vtmp);
                    uni_vmovss(dst_ptr(vlen * i + sizeof(data_t) * j), vreg);
                }
            }
        }
    };

    auto compute_dst = [&](int unroll, bool tail = false) {
        for (int i = 0; i < unroll; i++) {
            Vmm vreg = Vmm(i + 1);
            if (!tail) {
                uni_vmovups(vreg, dst_ptr(vlen * i));
                uni_vmulps(vreg, vreg, vsum);
                uni_vmovups(dst_ptr(vlen * i), vreg);
            } else {
                for (size_t j = 0; j < axis_simd_tail_; j++) {
                    uni_vmovss(vreg, dst_ptr(vlen * i + sizeof(data_t) * j));
                    uni_vmulps(vreg, vreg, vsum);
                    uni_vmovss(dst_ptr(vlen * i + sizeof(data_t) * j), vreg);
                }
            }
        }
    };

    uni_vmovups(vmax, vneg_flt_max); // flush to -FLT_MAX before accumulation
    axis_loop(accumulate_vmax);

    get_horizontal_op(vmax, vtmp = vsum, op_t::max);

    uni_vpxor(vsum, vsum, vsum); // flush accumulator before using
    axis_loop(accumulate_vsum);

    get_horizontal_op(vsum, vtmp = vmax, op_t::sum);

    uni_vdivps(vsum, vone, vsum, vtmp = vmax);
    axis_loop(compute_dst);
}

}

template <cpu_isa_t isa>
jit_uni_softmax_fwd_t<isa>::jit_uni_softmax_fwd_t(const pd_t *apd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(apd, inputs, outputs) {
    softmax_driver_ = new softmax_impl::driver_t<isa>(pd());
}

template <cpu_isa_t isa>
jit_uni_softmax_fwd_t<isa>::~jit_uni_softmax_fwd_t() {
    delete softmax_driver_;
}

template <cpu_isa_t isa>
void jit_uni_softmax_fwd_t<isa>::execute_forward() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));

    const auto ou_stride = pd()->outer_stride();
    const auto outer_size = utils::array_product(pd()->src_pd()->desc()->dims, pd()->desc()->softmax_axis);

    parallel_nd(outer_size, [&](int ou) {
        const data_t *src_ptr = src + ou * ou_stride;
        data_t *dst_ptr = dst + ou * ou_stride;
        softmax_driver_->exec(src_ptr, dst_ptr);
    });
}

namespace softmax_impl {

template <cpu_isa_t isa>
struct driver_t: public c_compatible {

    driver_t(const softmax_fwd_pd_t *sdesc)
        : sdesc_(sdesc), ker_(sdesc_) {}
    ~driver_t() {}

    void exec(const data_t *src, data_t *dst) {
        typename jit_softmax_t<isa>::call_params_t p;
        p.soff_max = sizeof(data_t) * sdesc_->axis_size();
        p.src = src;
        p.dst = dst;
        ker_(&p);
    }

private:
    const softmax_fwd_pd_t *sdesc_;

    jit_softmax_t<isa> ker_;
};

}

/* struct instantiation */
template struct jit_uni_softmax_fwd_t<sse42>;
template struct jit_uni_softmax_fwd_t<avx2>;
template struct jit_uni_softmax_fwd_t<avx512_common>;

}
}
}
