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

#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "simple_q10n.hpp"
#include "gemm_inner_product_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace inner_product_utils {

using namespace alg_kind;
using namespace math;

template <cpu_isa_t isa, data_type_t acc_type, data_type_t dst_type>
jit_pp_kernel_t<isa, acc_type, dst_type>::jit_pp_kernel_t(
        const cpu_inner_product_fwd_pd_t *pd)
    : ker_(nullptr)
    , bf16_emu_(nullptr)
    , OC_(pd->OC())
    , bias_data_type_(data_type::undef)
    , bias_data_type_size_(0)
    , do_scale_(false)
    , scale_idx_mult_(0)
    , rmode_(round_mode::nearest)
    , do_bias_(pd->with_bias())
    , max_OC_loop_unroll_(utils::one_of(isa, avx512_core_bf16, avx512_common) ? 13 : 6)
    , idx_compute_vreg_start_(0)
    , idx_compute_vreg_max_(utils::one_of(isa, avx512_core_bf16, avx512_common) ? 31 : 15)
    , compute_vregs_per_iter_(1) {
    using namespace types;
    using namespace Xbyak;

    if (utils::one_of(isa, avx2, sse42)) {
        idx_compute_vreg_start_ += 2;   //  Vmm(0), Vmm(1) - for masks
    }
    do_scale_ = !pd->attr()->output_scales_.has_default_values();
    if (do_scale_) {
        scale_idx_mult_ = (pd->attr()->output_scales_.mask_ == (1 << 1));
        vreg_scale = Vmm(idx_compute_vreg_start_++);
    }
    rmode_ = pd->attr()->round_mode_;
    post_ops_ = pd->attr()->post_ops_;
    bool only_eltwise = true;
    for (int i = 0; i < post_ops_.len_; i++) {
        auto &post_op = post_ops_.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors_.push_back(new jit_uni_eltwise_injector_f32<isa == avx512_core_bf16 ? avx512_common : isa>(
                    this, post_op.eltwise, true, eltwise_reserved, mask_post_op_reserved));
        } else if (post_op.is_depthwise()) {
            only_eltwise = false;
            depthwise_injectors_.push_back(new jit_uni_depthwise_injector_f32<isa == avx512_core_bf16 ? avx512_common : isa>(
                    this, post_op.depthwise.alg, mask_post_op_reserved));
        } else {
            only_eltwise = false;
        }
    }
    if (post_ops_.len_ > 0 && !only_eltwise) {
        vreg_d_weights = Vmm(idx_compute_vreg_max_--);
        vreg_d_bias = Vmm(idx_compute_vreg_max_--);
    }
    if (dst_type == data_type::u8 || utils::one_of(isa, avx2, sse42))
        vreg_zero = Vmm(idx_compute_vreg_start_++);

    if (do_bias_) {
        bias_data_type_ = pd->desc()->bias_desc.data_type;
        assert(bias_data_type_ != data_type::undef);
        bias_data_type_size_ = data_type_size(bias_data_type_);
        compute_vregs_per_iter_++;
    }

    if (dst_type == data_type::bf16 && isa == avx512_common) {
        idx_compute_vreg_max_ = 27;
        bf16_emu_ = new bf16_emulation_t(this,
                            bf16_emu_reserv_1, bf16_emu_reserv_2,
                            bf16_emu_reserv_3, bf16_emu_reserv_4,
                            bf16_emu_reserv_5, bf16_emu_reserv_5);
    }

    int max_unroll = (idx_compute_vreg_max_ - idx_compute_vreg_start_ + 1) / compute_vregs_per_iter_;
    max_OC_loop_unroll_ = nstl::min(max_OC_loop_unroll_, max_unroll);

    generate();
}

template <data_type_t acc_type, data_type_t dst_type>
ref_pp_kernel_t<acc_type, dst_type>::ref_pp_kernel_t(
        const cpu_inner_product_fwd_pd_t *pd)
        : OC_(pd->OC())
        , bias_data_type_(data_type::undef)
        , do_scale_(false)
        , scale_idx_mult_(0)
        , rmode_(round_mode::nearest)
        , do_bias_(pd->with_bias()) {
    do_scale_ = !pd->attr()->output_scales_.has_default_values();
    if (do_scale_)
        scale_idx_mult_ = (pd->attr()->output_scales_.mask_ == (1 << 1));

    rmode_ = pd->attr()->round_mode_;
    post_ops_ = pd->attr()->post_ops_;

    if (do_bias_) {
        bias_data_type_ = pd->desc()->bias_desc.data_type;
        assert(bias_data_type_ != data_type::undef);
    }
    // use fallback code for older CPUs since they do not have optimized
    // x8s8s32 GEMM anyways. The configuration variables above are used by
    // the fallback code.
    for (int i = 0; i < post_ops_.len_; i++) {
        auto &post_op = post_ops_.entry_[i];
        if (post_op.is_eltwise()) {
            ref_eltwise_impls_.push_back(new ref_eltwise_scalar_fwd_t(
                    post_op.eltwise.alg, post_op.eltwise.alpha, post_op.eltwise.beta));
        } else if (post_op.is_depthwise()) {
            ref_depthwise_impls_.push_back(new ref_depthwise_scalar_fwd_t(
                    post_op.depthwise.alg));
        }
    }
}

template<cpu_isa_t isa, data_type_t acc_type, data_type_t dst_type>
void jit_pp_kernel_t<isa, acc_type, dst_type>::generate()
{
    using namespace Xbyak;
    using namespace utils;
    using namespace round_mode;

    preamble();

    using ker_args_ = ker_args<acc_type, dst_type>;
#define PARAM_OFF(x) offsetof(ker_args_, x)
    mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
    mov(reg_acc, ptr[reg_param + PARAM_OFF(acc)]);
    mov(reg_bias, ptr[reg_param + PARAM_OFF(bias)]);
    if (do_scale_)
        mov(reg_scales, ptr[reg_param + PARAM_OFF(scales)]);
    mov(reg_len, ptr[reg_param + PARAM_OFF(len)]);
    mov(reg_oc_offset, ptr[reg_param + PARAM_OFF(oc_offset)]);
    if (do_scale_ && scale_idx_mult_ == 0)
        uni_vbroadcastss(vreg_scale, dword[reg_scales]);
#undef PARAM_OFF

    if (dst_type == data_type::u8 || utils::one_of(isa, avx2, sse42))
        uni_vpxor(vreg_zero, vreg_zero, vreg_zero);

    if (utils::one_of(isa, avx2, sse42))
        mov(reg_table, l_table);

    auto apply_post_ops = [&](size_t offset, int idx) {
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        for (int i = 0; i < post_ops_.len_; i++) {
            auto& post_op = post_ops_.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors_[eltwise_inj_idx]->compute_vector(vreg_dst(idx).getIdx());
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data + offset));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data + offset));
                lea(reg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float)]);
                lea(reg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float)]);
                depthwise_injectors_[depthwise_inj_idx]->compute_vector_range(vreg_dst(idx).getIdx(), vreg_dst(idx).getIdx() + 1, reg_d_weights, reg_d_bias);
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || dst_type == mkldnn_f32 || i != post_ops_.len_ - 1;

                if (post_op.quantization.crop_low_data->count_ != 1) {
                    mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.crop_low_data->shifts_ + offset));
                    uni_vmovups(vreg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float)]);
                } else {
                    mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.crop_low_data->shifts_));
                    uni_vbroadcastss(vreg_d_weights, ptr[reg_d_weights]);
                }

                if (post_op.quantization.crop_high_data->count_ != 1) {
                    mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.crop_high_data->shifts_ + offset));
                    uni_vmovups(vreg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float)]);
                } else {
                    mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.crop_high_data->shifts_));
                    uni_vbroadcastss(vreg_d_bias, ptr[reg_d_bias]);
                }

                uni_vmaxps(vreg_dst(idx), vreg_dst(idx), vreg_d_weights);
                uni_vminps(vreg_dst(idx), vreg_dst(idx), vreg_d_bias);

                if (post_op.quantization.input_scale_data->count_ != 1) {
                    mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.input_scale_data->scales_ + offset));
                    uni_vmovups(vreg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float)]);
                } else {
                    mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.input_scale_data->scales_));
                    uni_vbroadcastss(vreg_d_weights, ptr[reg_d_weights]);
                }

                if (post_op.quantization.input_shift_data->count_ != 1) {
                    mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.input_shift_data->shifts_ + offset));
                    uni_vmovups(vreg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float)]);
                } else {
                    mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.input_shift_data->shifts_));
                    uni_vbroadcastss(vreg_d_bias, ptr[reg_d_bias]);
                }

                uni_vfmadd213ps(vreg_dst(idx), vreg_d_weights, vreg_d_bias);

                if (do_rounding)
                    uni_vroundps(vreg_dst(idx), vreg_dst(idx), 0);

                if (do_dequantization) {
                    if (post_op.quantization.output_scale_data->count_ != 1) {
                        mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.output_scale_data->scales_ + offset));
                        uni_vmovups(vreg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float)]);
                    } else {
                        mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.output_scale_data->scales_));
                        uni_vbroadcastss(vreg_d_weights, ptr[reg_d_weights]);
                    }

                    if (post_op.quantization.output_shift_data->count_ != 1) {
                        mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.output_shift_data->shifts_ + offset));
                        uni_vmovups(vreg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float)]);
                    } else {
                        mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.output_shift_data->shifts_));
                        uni_vbroadcastss(vreg_d_bias, ptr[reg_d_bias]);
                    }

                    uni_vfmadd213ps(vreg_dst(idx), vreg_d_weights, vreg_d_bias);
                }
            }
        }
    };

    // Load accumulated value, convert to float, apply bias (if any), scaling,
    // and eltwise (if any); then convert to destination type and store
    auto compute = [&](size_t offset, int idx, bool apply_mask) {
        auto acc_addr = ptr[reg_acc + offset * sizeof(acc_data_t)];
        if (dst_type == data_type::bf16 && isa == avx512_common)
            bf16_emu_->init_vcvtneps2bf16();

        if (do_scale_ && scale_idx_mult_ == 1) {
            auto scale_addr = ptr[reg_scales + offset * sizeof(float)];
            auto vreg_scale_msk_ = vreg_scale;
            if (utils::one_of(isa, avx512_core_bf16, avx512_common)) {
                if (apply_mask)
                    vreg_scale_msk_ = vreg_scale_msk_ | kreg_rem_mask;
                uni_vmovups(vreg_scale_msk_, scale_addr);
            } else {
                if (apply_mask)
                    if (isa != sse42) {
                        uni_vblendvps(vreg_scale, vreg_zero, scale_addr, vreg_mask);
                    } else {
                        uni_vmovups(vreg_scale, vreg_zero);
                        uni_vblendvps(vreg_scale, vreg_scale, scale_addr, vreg_mask);
                    }
                else
                    uni_vmovups(vreg_scale, scale_addr);
            }
        }

        auto vreg_dst_ = vreg_dst(idx);
        if (utils::one_of(isa, avx512_core_bf16, avx512_common)) {
            if (apply_mask)
                vreg_dst_ = vreg_dst_ | kreg_rem_mask;

            switch (acc_type) {
                case data_type::s32: uni_vcvtdq2ps(vreg_dst_, acc_addr); break;
                case data_type::f32: uni_vmovups(vreg_dst_, acc_addr); break;
            }
        } else {
            if (apply_mask) {
                if (isa != sse42) {
                    uni_vblendvps(vreg_dst_, vreg_zero, acc_addr, vreg_mask);
                } else {
                    uni_vmovups(vreg_dst_, acc_addr);
                }
                if (acc_type == data_type::s32)
                    uni_vcvtdq2ps(vreg_dst_, vreg_dst_);
            } else {
                if (isa == sse42) {
                    uni_vmovups(vreg_dst_, acc_addr);
                    if (acc_type == data_type::s32) uni_vcvtdq2ps(vreg_dst_, vreg_dst_);
                } else {
                    switch (acc_type) {
                        case data_type::s32: uni_vcvtdq2ps(vreg_dst_, acc_addr); break;
                        case data_type::f32: uni_vmovups(vreg_dst_, acc_addr); break;
                    }
                }
            }
        }

        if (do_bias_) {
            auto bias_addr = ptr[reg_bias + offset * bias_data_type_size_];
            auto vreg_bias_ = vreg_bias(idx);
            if (utils::one_of(isa, avx512_core_bf16, avx512_common) && apply_mask)
                vreg_bias_ = vreg_bias_ | kreg_rem_mask;

            switch (bias_data_type_) {
            case data_type::s8:
                uni_vpmovsxbd(vreg_bias_, bias_addr);
                break;
            case data_type::u8:
                uni_vpmovzxbd(vreg_bias_, bias_addr);
                break;
            case data_type::s32:
            case data_type::f32:
                uni_vmovups(vreg_bias_, bias_addr);
                break;
            case data_type::bf16:
                vpmovzxwd(vreg_bias_, bias_addr);
                vpslld(vreg_bias(idx), vreg_bias(idx), 0x10);
                break;
            default: assert(!"unimplemented");
            }
            if (utils::one_of(bias_data_type_, data_type::u8, data_type::s8, data_type::s32))
                uni_vcvtdq2ps(vreg_bias(idx), vreg_bias(idx));
            uni_vaddps(vreg_dst(idx), vreg_dst(idx), vreg_bias(idx));
        }

        if (do_scale_)
            uni_vmulps(vreg_dst(idx), vreg_dst(idx), vreg_scale);

        apply_post_ops(offset, idx);

        if (dst_type == data_type::u8)
            uni_vmaxps(vreg_dst(idx), vreg_dst(idx), vreg_zero);

        if (utils::one_of(dst_type, data_type::s8, data_type::u8, data_type::s32)) {
            if (utils::one_of(isa, avx512_core_bf16, avx512_common)) {
                auto rmode_control = (rmode_ == nearest ? T_rn_sae : T_rd_sae);
                vcvtps2dq(vreg_dst(idx) | rmode_control, vreg_dst(idx));
            } else {
                if (rmode_ == nearest) {
                    uni_vcvtps2dq(vreg_dst(idx), vreg_dst(idx));
                } else if (rmode_ == down) {
                    uni_vroundps(vreg_dst(idx), vreg_dst(idx), 1);
                    uni_vcvtps2dq(vreg_dst(idx), vreg_dst(idx));
                } else {
                    assert(!"unimplemented");
                }
            }
        } else if (dst_type == data_type::bf16) {
            if (isa == avx512_core_bf16)
                vcvtneps2bf16(ymm_dst(idx), vreg_dst(idx));
            else
                bf16_emu_->r_vcvtneps2bf16(ymm_dst(idx), zmm_dst(idx));
        }

        auto dst_addr = ptr[reg_dst + offset * sizeof(dst_data_t)];
        switch (dst_type) {
        case data_type::s8:
            if (utils::one_of(isa, avx512_core_bf16, avx512_common)) {
                vpmovsdb(dst_addr, vreg_dst_);
            } else {
                uni_vpackssdw(vreg_dst_, vreg_dst_, vreg_dst_);
                if (isa != sse42)
                    vpermq(ymm_dst(idx), ymm_dst(idx), 0x08);
                uni_vpacksswb(vreg_dst_, vreg_dst_, vreg_dst_);
                if (isa != sse42) {
                    if (apply_mask) {
                        vmaskmovps(dst_addr, vreg_store_mask, vreg_dst_);
                    } else {
                        vmovq(dst_addr, xmm_dst(idx));
                    }
                } else {
                    if (apply_mask) {
                        lea(reg_ptr_maskmovdqu_dst, dst_addr);
                        maskmovdqu(vreg_dst_, vreg_store_mask);
                    } else {
                        movd(dst_addr, xmm_dst(idx));
                    }
                }
            }
            break;
        case data_type::u8:
            if (utils::one_of(isa, avx512_core_bf16, avx512_common)) {
                vpmovusdb(dst_addr, vreg_dst_);
            } else {
                uni_vpackusdw(vreg_dst_, vreg_dst_, vreg_dst_);
                if (isa != sse42)
                    vpermq(ymm_dst(idx), ymm_dst(idx), 0x08);
                uni_vpackuswb(vreg_dst_, vreg_dst_, vreg_dst_);
                if (isa != sse42) {
                    if (apply_mask) {
                        vmaskmovps(dst_addr, vreg_store_mask, vreg_dst_);
                    } else {
                        vmovq(dst_addr, xmm_dst(idx));
                    }
                } else {
                    if (apply_mask) {
                        lea(reg_ptr_maskmovdqu_dst, dst_addr);
                        maskmovdqu(vreg_dst_, vreg_store_mask);
                    } else {
                        movd(dst_addr, xmm_dst(idx));
                    }
                }
            }
            break;
        case data_type::f32:
        case data_type::s32:
            if (utils::one_of(isa, avx512_core_bf16, avx512_common)) {
                uni_vmovups(dst_addr, vreg_dst_);
            } else {
                if (apply_mask) {
                    if (isa != sse42) {
                        vmaskmovps(dst_addr, vreg_mask, vreg_dst_);
                    } else {
                        lea(reg_ptr_maskmovdqu_dst, dst_addr);
                        maskmovdqu(vreg_dst_, vreg_mask);
                    }
                } else {
                    uni_vmovups(dst_addr, vreg_dst_);
                }
            }
            break;
        case data_type::bf16:
            vmovdqu16(dst_addr,
                      apply_mask
                              ? ymm_dst(idx) | kreg_rem_mask
                              : ymm_dst(idx));
            break;
        default: assert(!"unimplemented");
        }
    };

    // Advance all pointers by an immediate
    auto advance_ptrs_imm = [&](size_t offset) {
        add(reg_dst, offset * sizeof(dst_data_t));
        add(reg_acc, offset * sizeof(acc_data_t));
        if (do_scale_ && scale_idx_mult_ == 1)
            add(reg_scales, offset * sizeof(float));
        if (do_bias_)
            add(reg_bias, offset * bias_data_type_size_);
    };

    // Advance all pointers by a value stored in a register
    auto advance_ptrs_reg = [&](Reg64 offset) {
        lea(reg_dst, ptr[reg_dst + offset * sizeof(dst_data_t)]);
        lea(reg_acc, ptr[reg_acc + offset * sizeof(acc_data_t)]);
        if (do_scale_ && scale_idx_mult_ == 1)
            lea(reg_scales, ptr[reg_scales + offset * sizeof(float)]);
        if (do_bias_)
            lea(reg_bias, ptr[reg_bias + offset * bias_data_type_size_]);
    };

    // Rewind pointers that point to data that is indixed by output channel
    // (bias or per-oc scaling factors)
    auto rewind_ptrs = [&]() {
        if (do_bias_)
            sub(reg_bias, OC_ * bias_data_type_size_);
        if (do_scale_ && scale_idx_mult_ == 1)
            sub(reg_scales, OC_ * sizeof(float));
    };

    //      <-------------------- OC ------------------------------->
    //
    // ^    +....................+----------------------------------+
    // |    :   not accessed     |          Prologue loop           |
    // |    +--------------------+----------------------------------+
    //      |                                                       |
    // M    |                 Main loop (unrolled)                  |
    // B    |                                                       |
    //      +--------------------------------+----------------------+
    // |    |       Epilogue loop            |      not accessed    :
    // v    +--------------------------------+......................+

    bool do_post_ops = post_ops_.len_ > 0;

    Label prologue_end;
    cmp(reg_oc_offset, 0);
    je(prologue_end, T_NEAR);

    // Prologue loop
    {
        mov(reg_tmp, OC_);
        sub(reg_tmp, reg_oc_offset);
        cmp(reg_tmp, reg_len);
        cmovg(reg_tmp, reg_len);
        sub(reg_len, reg_tmp);

        Label prologue_loop, prologue_loop_tail, prologue_loop_end;
        cmp(reg_tmp, vlen);
        jl(prologue_loop_tail, T_NEAR);
        L(prologue_loop); {
            compute(0, 0, false);
            advance_ptrs_imm(vlen);
            if (do_post_ops)
                add(reg_oc_offset, vlen);
            sub(reg_tmp, vlen);
            cmp(reg_tmp, vlen);
            jge(prologue_loop, T_NEAR);
        }

        L(prologue_loop_tail);
        if (utils::one_of(isa, avx512_core_bf16, avx512_common)) {
            mov(reg_rem_mask, 1);
            shl(reg_rem_mask, cl); // cl == reg_tmp because reg_tmp <= vlen here
            sub(reg_rem_mask, 1);
            jz(prologue_loop_end, T_NEAR);

            kmovq(kreg_rem_mask, reg_rem_mask);
        } else {
            mov(reg_shift_table, vlen);
            sub(reg_shift_table, reg_tmp);
            uni_vmovups(vreg_mask, ptr[reg_table + reg_shift_table * sizeof(float)]);
            if (dst_type == data_type::s8 || dst_type == data_type::u8) {
                mov(reg_shift_table, vlen * sizeof(float));
                sub(reg_shift_table, reg_tmp);
                uni_vmovups(vreg_store_mask, ptr[reg_table + reg_shift_table]);
            }
        }
        compute(0, 0, true);
        advance_ptrs_reg(reg_tmp);

        L(prologue_loop_end);
        rewind_ptrs();
    }
    L(prologue_end);

    // Main loop
    Label main_loop_end;
    {
        cmp(reg_len, OC_);
        jl(main_loop_end, T_NEAR);

        size_t OC_loop, OC_tail;
        if (OC_ < max_OC_loop_unroll_ * vlen) {
            // Fully unroll small loops
            OC_loop = 0;
            OC_tail = OC_;
        } else {
            OC_loop = vlen * default_OC_loop_unroll_;
            OC_tail = OC_ % OC_loop;
        }

        assert(!!OC_loop || !!OC_tail);

        if (OC_tail % vlen) {
            int vlen_tail = OC_tail % vlen;
            if (utils::one_of(isa, avx512_core_bf16, avx512_common)) {
                unsigned tail_mask = (1 << vlen_tail) - 1;
                mov(reg_tmp, tail_mask);
                kmovq(kreg_rem_mask, reg_tmp);
            } else {
                mov(reg_shift_table, vlen - vlen_tail);
                uni_vmovups(vreg_mask, ptr[reg_table + reg_shift_table * sizeof(float)]);
                if (dst_type == data_type::s8 || dst_type == data_type::u8) {
                    mov(reg_shift_table, vlen * sizeof(float));
                    sub(reg_shift_table, vlen_tail);
                    uni_vmovups(vreg_store_mask, ptr[reg_table + reg_shift_table]);
                }
            }
        }

        Label main_loop;
        L(main_loop); {
            if (do_post_ops)
                mov(reg_oc_offset, 0);

            if (OC_loop) {
                mov(reg_tmp, rnd_dn(OC_, OC_loop));
                Label oc_loop;
                L(oc_loop); {
                    for (size_t offset = 0; offset < OC_loop; offset += vlen)
                        compute(offset, offset / vlen, false);
                    advance_ptrs_imm(OC_loop);
                    if (do_post_ops)
                        add(reg_oc_offset, OC_loop);
                    sub(reg_tmp, OC_loop);
                    jnz(oc_loop);
                }
            }

            if (OC_tail) {
                for (size_t offset = 0; offset < OC_tail; offset += vlen) {
                    bool use_mask = (offset + vlen) > OC_tail;
                    compute(offset, offset / vlen, use_mask);
                }
                advance_ptrs_imm(OC_tail);
            }

            rewind_ptrs();
            sub(reg_len, OC_);
            cmp(reg_len, OC_);
            jge(main_loop, T_NEAR);
        }
    }
    L(main_loop_end);

    // Epilogue loop
    Label epilogue_end;
    {
        cmp(reg_len, 0);
        je(epilogue_end, T_NEAR);

        Label epilogue_loop, epilogue_loop_tail;
        if (do_post_ops)
            mov(reg_oc_offset, 0);
        cmp(reg_len, vlen);
        jl(epilogue_loop_tail, T_NEAR);
        L(epilogue_loop); {
            compute(0, 0, false);
            sub(reg_len, vlen);
            advance_ptrs_imm(vlen);
            if (do_post_ops)
                add(reg_oc_offset, vlen);
            cmp(reg_len, vlen);
            jge(epilogue_loop, T_NEAR);
        }

        L(epilogue_loop_tail);
        mov(reg_tmp, reg_len); // reg_tmp is rcx, and we need cl for the shift
        if (utils::one_of(isa, avx512_core_bf16, avx512_common)) {
            mov(reg_rem_mask, 1);
            shl(reg_rem_mask, cl); // reg_tmp == rcx and reg_tail < vlen == 16
            sub(reg_rem_mask, 1);
            jz(epilogue_end, T_NEAR);
            kmovq(kreg_rem_mask, reg_rem_mask);
        } else {
            mov(reg_shift_table, vlen);
            sub(reg_shift_table, reg_tmp);
            uni_vmovups(vreg_mask, ptr[reg_table + reg_shift_table * sizeof(float)]);
            if (dst_type == data_type::s8 || dst_type == data_type::u8) {
                mov(reg_shift_table, vlen * sizeof(float));
                sub(reg_shift_table, reg_tmp);
                uni_vmovups(vreg_store_mask, ptr[reg_table + reg_shift_table]);
            }
        }
        compute(0, 0, true);
    }

    L(epilogue_end);

    postamble();

    for (auto& inj : eltwise_injectors_)
        inj->prepare_table();

    if (utils::one_of(isa, avx2, sse42)) {
        align(64);
        L(l_table);
        for (size_t i = 0; i < vlen; i++) dd(0xFFFFFFFF);
        for (size_t i = 0; i < vlen; i++) dd(0x00000000);
    }

    ker_ = getCode<decltype(ker_)>();
}

template <cpu_isa_t isa, data_type_t acc_type, data_type_t dst_type>
void jit_pp_kernel_t<isa, acc_type, dst_type>::operator()(dst_data_t *dst,
        const acc_data_t *acc, const char *bias, const float *scales,
        size_t start, size_t end) {
    using math::get_bias;

    if (end <= start)
        return;

    // JIT
    ker_args<acc_type, dst_type> args;
    size_t oc_offset = start % OC_;
    args.dst = dst + start;
    args.acc = acc + start;
    args.bias = bias + oc_offset * bias_data_type_size_;
    args.scales = scales + scale_idx_mult_ * oc_offset;
    args.len = end - start;
    args.oc_offset = oc_offset;
    ker_(&args);
};

template <data_type_t acc_type, data_type_t dst_type>
void ref_pp_kernel_t<acc_type, dst_type>::operator()(dst_data_t *dst, const acc_data_t *acc,
        const char *bias, const float *scales, size_t start, size_t end) {
    using math::get_bias;

    if (end <= start)
        return;

    // Fallback
    size_t oc = start % OC_;
    for (size_t i = start; i < end; i++) {
        float d = (float)acc[i];
        if (do_bias_)
            d += get_bias(bias, oc, bias_data_type_);
        if (do_scale_)
            d *= scales[oc * scale_idx_mult_];

        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        for (int j = 0; j < post_ops_.len_; j++) {
            auto &post_op = post_ops_.entry_[j];
            if (post_op.is_eltwise()) {
                d = ref_eltwise_impls_[eltwise_inj_idx]->compute_scalar(d);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                auto depthwise_weights = post_op.depthwise.weights_data;
                auto depthwise_bias = post_op.depthwise.biases_data;
                d = ref_depthwise_impls_[depthwise_inj_idx]->compute_scalar(d, depthwise_weights + oc, depthwise_bias + oc);
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || dst_type == mkldnn_f32 || i != post_ops_.len_ - 1;

                auto quant = post_op.quantization;
                auto pcl = quant.crop_low_data->shifts_;
                auto pch = quant.crop_high_data->shifts_;
                auto pisc = quant.input_scale_data->scales_;
                auto pish = quant.input_shift_data->shifts_;
                auto posc = quant.output_scale_data->scales_;
                auto posh = quant.output_shift_data->shifts_;

                int cl_idx = quant.crop_low_data->count_ == 1 ? 0 : oc;
                int ch_idx = quant.crop_high_data->count_ == 1 ? 0 : oc;
                int isc_idx = quant.input_scale_data->count_ == 1 ? 0 : oc;
                int ish_idx = quant.input_shift_data->count_ == 1 ? 0 : oc;
                int osc_idx = quant.output_scale_data->count_ == 1 ? 0 : oc;
                int osh_idx = quant.output_shift_data->count_ == 1 ? 0 : oc;

                d = nstl::min(pch[ch_idx], nstl::max(pcl[cl_idx], d));
                d = d * pisc[isc_idx] + pish[ish_idx];

                if (do_rounding)
                    d = roundf(d);

                if (do_dequantization)
                    d = d * posc[osc_idx] + posh[osh_idx];
            }
        }
        dst[i] = qz_a1b0<float, dst_data_t>()(d, rmode_);
        oc = (oc == OC_ - 1) ? 0 : oc + 1;
    }
};

using namespace data_type;

template class jit_pp_kernel_t<avx512_core_bf16, f32, f32>;
template class jit_pp_kernel_t<avx512_core_bf16, s32, f32>;
template class jit_pp_kernel_t<avx512_core_bf16, s32, s32>;
template class jit_pp_kernel_t<avx512_core_bf16, s32, s8>;
template class jit_pp_kernel_t<avx512_core_bf16, s32, u8>;
template class jit_pp_kernel_t<avx512_core_bf16, f32, bf16>;

template class jit_pp_kernel_t<avx512_common, f32, f32>;
template class jit_pp_kernel_t<avx512_common, s32, f32>;
template class jit_pp_kernel_t<avx512_common, s32, s32>;
template class jit_pp_kernel_t<avx512_common, s32, s8>;
template class jit_pp_kernel_t<avx512_common, s32, u8>;
template class jit_pp_kernel_t<avx512_common, f32, bf16>;

template class jit_pp_kernel_t<avx2, f32, f32>;
template class jit_pp_kernel_t<avx2, s32, f32>;
template class jit_pp_kernel_t<avx2, s32, s32>;
template class jit_pp_kernel_t<avx2, s32, s8>;
template class jit_pp_kernel_t<avx2, s32, u8>;

template class jit_pp_kernel_t<sse42, f32, f32>;
template class jit_pp_kernel_t<sse42, s32, f32>;
template class jit_pp_kernel_t<sse42, s32, s32>;
template class jit_pp_kernel_t<sse42, s32, s8>;
template class jit_pp_kernel_t<sse42, s32, u8>;

template class ref_pp_kernel_t<f32, f32>;
template class ref_pp_kernel_t<s32, f32>;
template class ref_pp_kernel_t<s32, s32>;
template class ref_pp_kernel_t<s32, s8>;
template class ref_pp_kernel_t<s32, u8>;
}

}
}
}
