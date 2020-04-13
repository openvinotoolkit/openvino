/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#include <common/memory_tracking.hpp>
#include <common/primitive_attr.hpp>
#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "utils.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "math_utils.hpp"

#include "simple_q10n.hpp"

#include "gemm_x8s8s32x_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::math;
using namespace mkldnn::impl::memory_tracking::names;

template <data_type_t src_type, data_type_t dst_type>
void _gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>::
execute_forward() const {
    auto src_base = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto wei_base = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bia_base = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst_base = reinterpret_cast<dst_data_t *>(this->memory());

    const memory_tracking::grantor_t scratchpad = this->scratchpad();

    const jit_gemm_conv_conf_t &jcp = this->pd()->jcp_;

    if (jcp.with_input_zp) {
        auto output_compensation = scratchpad.get<int32_t>(key_conv_padded_compensation);
        for (int i = 0; i < this->pd()->attr()->output_compensations_.count_; i++) {
            output_compensation[i] = (int32_t)this->pd()->attr()->output_compensations_.shifts_[i];
        }
    }

    assert(IMPLICATION(jcp.ow_block != jcp.ow, jcp.oh_block == 1));

    const int nb_oh = div_up(jcp.oh, jcp.oh_block);
    const int nb_ow = div_up(jcp.ow, jcp.ow_block);
    const size_t work_amount = jcp.ngroups * jcp.mb * nb_oh * nb_ow;
    parallel(jcp.nthr, work_amount, [&](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src_base, wei_base, bia_base, dst_base,
                scratchpad);
    });
}

template <data_type_t src_type, data_type_t dst_type>
template <cpu_isa_t isa>
_gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>::jit_pp_ker_t<isa>::jit_pp_ker_t(const pd_t *pd)
    : ker_(nullptr)
    , jcp_(pd->jcp_)
    , OC_(pd->jcp_.oc)
    , OS_(pd->jcp_.os)
    , bias_data_type_(data_type::undef)
    , bias_data_type_size_(0)
    , do_scale_(false)
    , scale_idx_mult_(0)
    , rmode_(round_mode::nearest)
    , do_bias_(pd->with_bias())
    , do_eltwise_(false)
    , do_sum_(false)
    , sum_scale_(0)
    , sum_data_type_(mkldnn_f32)
    , with_weights_zp_(false)
    , default_OC_loop_unroll_(4)
    , max_OC_loop_unroll_(isa == avx512_common ? 12 : 6)
    , idx_compute_vreg_start_(0)
    , idx_compute_vreg_max_(isa == avx512_common ? 31 : 15)
    , compute_vregs_per_iter_(1)
    , post_ops_(pd->attr()->post_ops_)
{
    using namespace types;

    const auto dst_md = memory_desc_wrapper(pd->dst_pd());
    if (pd->ndims() == 5)
        this->dst_os_stride_ = dst_md.blk_off(0, 0, 0, 0, 1);
    else if (pd->ndims() == 4)
        this->dst_os_stride_ = dst_md.blk_off(0, 0, 0, 1);

    if (utils::one_of(isa, avx2, sse42)) {
        idx_compute_vreg_start_ += 2;   //  Vmm(0), Vmm(1) - for masks
    }
    do_scale_ = !pd->attr()->output_scales_.has_default_values();
    if (do_scale_) {
        scale_idx_mult_ = (pd->attr()->output_scales_.mask_ == (1 << 1));
        vreg_scale = Vmm(idx_compute_vreg_start_++);
    }
    rmode_ = pd->attr()->round_mode_;
    if (dst_type == data_type::u8 || utils::one_of(isa, avx2, sse42))
        vreg_zero = Vmm(idx_compute_vreg_start_++);

    bool only_eltwise_or_sum = true;
    for (int idx = 0; idx < post_ops_.len_; ++idx) {
        const auto &e = post_ops_.entry_[idx];
        if (e.is_eltwise(true)) {
            do_eltwise_ = true;
        } else if (e.is_sum()) {
            do_sum_ = true;
            sum_scale_ = e.sum.scale;
            sum_data_type_ = e.sum.data_type;
        } else {
            only_eltwise_or_sum = false;
        }
    }
    if (post_ops_.len_ > 0 && !only_eltwise_or_sum) {
        vreg_d_weights = Vmm(idx_compute_vreg_max_--);
        vreg_d_bias = Vmm(idx_compute_vreg_max_--);
    }

    do_signed_scaling_ = jcp_.signed_input;
    if (do_signed_scaling_)
        vreg_signed_scale = Vmm(idx_compute_vreg_start_++);

    if (do_bias_) {
        bias_data_type_ = pd->desc()->bias_desc.data_type;
        assert(bias_data_type_ != data_type::undef);
        bias_data_type_size_ = data_type_size(bias_data_type_);
        compute_vregs_per_iter_++;
    }
    if (do_sum_) {
        vreg_sum_scale = Vmm(idx_compute_vreg_start_++);
        compute_vregs_per_iter_++;
    }

    with_weights_zp_ = !pd->attr()->weights_zero_points_.has_default_values();
    if (with_weights_zp_)
        vreg_comp = Vmm(idx_compute_vreg_start_++);

    for (int i = 0; i < post_ops_.len_; i++) {
        auto &post_op = post_ops_.entry_[i];
        if (post_op.is_eltwise()) {
            jit_eltwise_injectors_.push_back(new jit_uni_eltwise_injector_f32<isa>(
                    this, post_op.eltwise, true, eltwise_reserved, mask_post_op_reserved));
        } else if (post_op.is_depthwise()) {
            jit_depthwise_injectors_.push_back(new jit_uni_depthwise_injector_f32<isa>(
                    this, post_op.depthwise.alg, mask_post_op_reserved));
        }
    }

    int max_unroll = (idx_compute_vreg_max_ - idx_compute_vreg_start_ + 1) / compute_vregs_per_iter_;
    max_OC_loop_unroll_ = nstl::min(max_OC_loop_unroll_, max_unroll);
    default_OC_loop_unroll_ = nstl::min(default_OC_loop_unroll_, max_unroll);

    generate();
}

template <data_type_t src_type, data_type_t dst_type>
_gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>::ref_pp_ker_t::ref_pp_ker_t(const pd_t *pd)
        : ker_(nullptr)
        , jcp_(pd->jcp_)
        , OC_(pd->jcp_.oc)
        , bias_data_type_(data_type::undef)
        , do_scale_(false)
        , scale_idx_mult_(0)
        , rmode_(round_mode::nearest)
        , do_bias_(pd->with_bias())
        , with_weights_zp_(false)
        , post_ops_(pd->attr()->post_ops_)
{
    using namespace types;

    const auto dst_md = memory_desc_wrapper(pd->dst_pd());
    if (pd->ndims() == 5)
        this->dst_os_stride_ = dst_md.blk_off(0, 0, 0, 0, 1);
    else if (pd->ndims() == 4)
        this->dst_os_stride_ = dst_md.blk_off(0, 0, 0, 1);

    do_scale_ = !pd->attr()->output_scales_.has_default_values();
    if (do_scale_) {
        scale_idx_mult_ = (pd->attr()->output_scales_.mask_ == (1 << 1));
    }
    rmode_ = pd->attr()->round_mode_;

    if (do_bias_) {
        bias_data_type_ = pd->desc()->bias_desc.data_type;
        assert(bias_data_type_ != data_type::undef);
    }

    with_weights_zp_ = !pd->attr()->weights_zero_points_.has_default_values();

    // use fallback code for unsupported cases
    for (int i = 0; i < post_ops_.len_; i++) {
        auto &post_op = post_ops_.entry_[i];
        if (post_op.is_eltwise()) {
            ref_eltwise_injectors_.push_back(new ref_eltwise_scalar_fwd_t(
                    post_op.eltwise.alg, post_op.eltwise.alpha, post_op.eltwise.beta));
        } else if (post_op.is_depthwise()) {
            ref_depthwise_injectors_.push_back(new ref_depthwise_scalar_fwd_t(
                    post_op.depthwise.alg));
        }
    }
}

template <data_type_t src_type, data_type_t dst_type>
template <cpu_isa_t isa>
void _gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>::jit_pp_ker_t<isa>::generate()
{
    using namespace Xbyak;
    using namespace utils;
    using namespace round_mode;

    preamble();

#define PARAM_OFF(x) offsetof(ker_args, x)
    mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
    mov(reg_acc, ptr[reg_param + PARAM_OFF(acc)]);
    mov(reg_bias, ptr[reg_param + PARAM_OFF(bias)]);
    mov(reg_scales, ptr[reg_param + PARAM_OFF(scales)]);
    mov(reg_len, ptr[reg_param + PARAM_OFF(len)]);
    mov(reg_oc_offset, ptr[reg_param + PARAM_OFF(oc_offset)]);
    mov(reg_g_offset, ptr[reg_param + PARAM_OFF(g_offset)]);
    if (do_sum_)
        uni_vbroadcastss(vreg_sum_scale, ptr[reg_param + PARAM_OFF(sum_scale)]);
    if (do_signed_scaling_)
        uni_vbroadcastss(vreg_signed_scale, ptr[reg_param + PARAM_OFF(signed_scale)]);
    if (do_scale_ && scale_idx_mult_ == 0)
        uni_vbroadcastss(vreg_scale, dword[reg_scales]);
    if (with_weights_zp_) {
        mov(reg_weights_zp, ptr[reg_param + PARAM_OFF(weights_zp)]);
        mov(reg_weights_zp_compensation, ptr[reg_param + PARAM_OFF(weights_zp_compensation)]);
    }
#undef PARAM_OFF

    if (do_eltwise_ || dst_type == data_type::u8 || utils::one_of(isa, avx2, sse42))
        uni_vpxor(vreg_zero, vreg_zero, vreg_zero);

    if (utils::one_of(isa, avx2, sse42))
        mov(reg_table, l_table);

    auto apply_post_ops = [&](size_t offset, int idx) {
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        if (with_weights_zp_) {
            push(reg_weights_zp);
            push(reg_weights_zp_compensation);
        }
        for (int i = 0; i < post_ops_.len_; i++) {
            auto& post_op = post_ops_.entry_[i];
            if (post_op.is_sum()) {
                auto dst_addr = ptr[reg_dst + offset * sizeof(dst_data_t)];
                auto vreg_prev_dst_ = vreg_prev_dst(idx);
                switch (sum_data_type_) {
                case data_type::f32:
                case data_type::s32: uni_vmovups(vreg_prev_dst_, dst_addr); break;
                case data_type::s8: uni_vpmovsxbd(vreg_prev_dst_, dst_addr); break;
                case data_type::u8: uni_vpmovzxbd(vreg_prev_dst_, dst_addr); break;
                default: assert(!"unsupported data type");
                }
                if (sum_data_type_ != data_type::f32)
                    uni_vcvtdq2ps(vreg_prev_dst(idx), vreg_prev_dst(idx));

                uni_vfmadd231ps(vreg_dst(idx), vreg_prev_dst(idx), vreg_sum_scale);
            } else if (post_op.is_eltwise()) {
                jit_eltwise_injectors_[eltwise_inj_idx]->compute_vector_range(vreg_dst(idx).getIdx(), vreg_dst(idx).getIdx() + 1);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                add(reg_oc_offset, reg_g_offset);
                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data + offset));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data + offset));
                lea(reg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float)]);
                lea(reg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float)]);
                jit_depthwise_injectors_[depthwise_inj_idx]->compute_vector_range(vreg_dst(idx).getIdx(), vreg_dst(idx).getIdx() + 1, reg_d_weights, reg_d_bias);
                depthwise_inj_idx++;
                sub(reg_oc_offset, reg_g_offset);
            } else if (post_op.is_quantization()) {
                add(reg_oc_offset, reg_g_offset);
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
                sub(reg_oc_offset, reg_g_offset);
            }
        }
        if (with_weights_zp_) {
            pop(reg_weights_zp_compensation);
            pop(reg_weights_zp);
        }
    };

    // Load accumulated value, convert to float, apply weights_zp (if any),
    // bias (if any), scaling, and simple operations (if any);
    // then convert to destination type and store
    auto compute = [&](size_t offset, int idx, bool apply_mask) {
        auto acc_addr = ptr[reg_acc + offset * sizeof(acc_data_t)];

        if (do_scale_ && scale_idx_mult_ > 0) {
            assert(scale_idx_mult_ == 1);
            auto scale_addr = ptr[reg_scales + offset * sizeof(float)];
            auto vreg_scale_ = vreg_scale;
            if (isa == avx512_common) {
                if (apply_mask)
                    vreg_scale_ = vreg_scale_ | kreg_rem_mask_short;
                uni_vmovups(vreg_scale_, scale_addr);
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
        if (isa == avx512_common) {
            if (apply_mask)
                vreg_dst_ = vreg_dst_ | kreg_rem_mask_short;
            uni_vcvtdq2ps(vreg_dst_, acc_addr);
        } else {
            if (apply_mask) {
                if (isa != sse42) {
                    uni_vblendvps(vreg_dst_, vreg_zero, acc_addr, vreg_mask);
                } else {
                    uni_vmovups(vreg_dst_, acc_addr);
                }
                uni_vcvtdq2ps(vreg_dst_, vreg_dst_);
            } else {
                if (isa == sse42) {
                    uni_vmovups(vreg_dst_, acc_addr);
                    uni_vcvtdq2ps(vreg_dst_, vreg_dst_);
                } else {
                    uni_vcvtdq2ps(vreg_dst_, acc_addr);
                }
            }
        }

        if (do_signed_scaling_)
            uni_vmulps(vreg_dst(idx), vreg_dst(idx), vreg_signed_scale);

        if (with_weights_zp_) {
            uni_vmovups(vreg_d_weights, ptr[reg_weights_zp + offset * sizeof(float)]);
            uni_vpbroadcastd(vreg_comp, ptr[reg_weights_zp_compensation]);

            uni_vcvtdq2ps(vreg_comp, vreg_comp);
            uni_vmulps(vreg_comp, vreg_comp, vreg_d_weights);
            uni_vsubps(vreg_dst(idx), vreg_dst(idx), vreg_comp);
        }

        if (do_bias_) {
            auto bias_addr = ptr[reg_bias + offset * bias_data_type_size_];
            auto vreg_bias_ = vreg_bias(idx);
            if (isa == avx512_common && apply_mask)
                vreg_bias_ = vreg_bias_ | kreg_rem_mask_short;

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
            default: assert(!"unimplemented");
            }
            if (bias_data_type_ != data_type::f32)
                uni_vcvtdq2ps(vreg_bias(idx), vreg_bias(idx));
            uni_vaddps(vreg_dst(idx), vreg_dst(idx), vreg_bias(idx));
        }

        if (do_scale_)
            uni_vmulps(vreg_dst(idx), vreg_dst(idx), vreg_scale);

        apply_post_ops(offset, idx);

        if (dst_type != data_type::f32) {
            if (isa == avx512_common) {
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
        }

        if (dst_type == data_type::u8)
            uni_vpmaxsd(vreg_dst(idx), vreg_dst(idx), vreg_zero);

        auto dst_addr = ptr[reg_dst + offset * sizeof(dst_data_t)];
        switch (dst_type) {
        case data_type::s8:
            if (isa == avx512_common) {
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
            if (isa == avx512_common) {
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
            if (isa == avx512_common) {
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
        default: assert(!"unimplemented");
        }
    };

    // Advance all pointers by an immediate
    auto advance_ptrs_imm = [&](size_t offset) {
        add(reg_dst, offset * sizeof(dst_data_t));
        add(reg_acc, offset * sizeof(acc_data_t));
        if (scale_idx_mult_) {
            assert(scale_idx_mult_ == 1);
            add(reg_scales, offset * sizeof(float));
        }
        if (do_bias_)
            add(reg_bias, offset * bias_data_type_size_);
        if (with_weights_zp_)
            add(reg_weights_zp, offset * sizeof(float));
    };

    // Advance all pointers by a value stored in a register
    auto advance_ptrs_reg = [&](Reg64 offset) {
        lea(reg_dst, ptr[reg_dst + offset * sizeof(dst_data_t)]);
        lea(reg_acc, ptr[reg_acc + offset * sizeof(acc_data_t)]);
        if (scale_idx_mult_) {
            assert(scale_idx_mult_ == 1);
            lea(reg_scales, ptr[reg_scales + offset * sizeof(float)]);
        }
        if (do_bias_)
            lea(reg_bias, ptr[reg_bias + offset * bias_data_type_size_]);
        if (with_weights_zp_)
            lea(reg_weights_zp, ptr[reg_weights_zp + offset * sizeof(float)]);
    };

    // Rewind pointers that point to data that is indexed by output channel
    // (bias or per-oc scaling factors)
    auto rewind_ptrs = [&]() {
        if (with_weights_zp_)
            sub(reg_weights_zp, OC_ * sizeof(float));
        if (do_bias_)
            sub(reg_bias, OC_ * bias_data_type_size_);
        if (scale_idx_mult_) {
            assert(scale_idx_mult_ == 1);
            sub(reg_scales, OC_ * sizeof(float));
        }
        add(reg_dst, (this->dst_os_stride_ - OC_) * sizeof(dst_data_t));
    };

    //                    <--------- OC --------------->
    //
    // ^  ................+..............+-------------+.......................
    // |  .               : not accessed |Prologue loop|                      .
    // |  .               +--------------+-------------+                      .
    //    .               |                            |                      .
    // O  .               |  Main loop (unrolled)      |                      .
    // S  .               |                            |                      .
    //    .               +--------------+-------------+                      .
    // |  .               | Epilogue loop|not accessed :                      .
    // v  ................+--------------+.............+.......................

    bool do_post_ops = post_ops_.len_ != 0;

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
        if (isa == avx512_common) {
            mov(reg_rem_mask_short, 1);
            // cl == reg_tmp because reg_tmp <= vlen here
            shl(reg_rem_mask_short, cl);
            sub(reg_rem_mask_short, 1);
            jz(prologue_loop_end, T_NEAR);

            kmovq(kreg_rem_mask_short, reg_rem_mask_short);
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
        if (with_weights_zp_)
            add(reg_weights_zp_compensation, sizeof(int32_t));
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
        }
        else {
            OC_loop = vlen * default_OC_loop_unroll_;
            OC_tail = OC_ % OC_loop;
        }

        assert(!!OC_loop || !!OC_tail);

        if (OC_tail % vlen) {
            int vlen_tail = OC_tail % vlen;
            if (isa == avx512_common) {
                unsigned tail_mask = (1 << vlen_tail) - 1;
                mov(reg_tmp, tail_mask);
                kmovq(kreg_rem_mask_short, reg_tmp);
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
            if (with_weights_zp_)
                add(reg_weights_zp_compensation, sizeof(int32_t));
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
        if (isa == avx512_common) {
            mov(reg_rem_mask_short, 1);
            shl(reg_rem_mask_short, cl); // reg_tmp == rcx and reg_tail < vlen
            sub(reg_rem_mask_short, 1);
            jz(epilogue_end, T_NEAR);
            kmovq(kreg_rem_mask_short, reg_rem_mask_short);
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

    for (auto& inj : jit_eltwise_injectors_)
        inj->prepare_table();

    if (utils::one_of(isa, avx2, sse42)) {
        align(64);
        L(l_table);
        for (size_t i = 0; i < vlen; i++) dd(0xFFFFFFFF);
        for (size_t i = 0; i < vlen; i++) dd(0x00000000);
    }

    ker_ = getCode<decltype(ker_)>();
}

template <data_type_t src_type, data_type_t dst_type>
template <cpu_isa_t isa>
void _gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>::jit_pp_ker_t<isa>::operator ()
    (dst_data_t *dst, acc_data_t *acc, const char *bias, const float *scales, float signed_scale,
     int g, size_t start, size_t end, float* weights_zp, int32_t* weights_zp_compensation)
{
    using math::get_bias;

    if (end <= start)
        return;

    // JIT
    ker_args args;
    size_t oc_offset = start % OC_;
    size_t os_offset = start / OC_;
    args.acc = acc + start;
    args.dst = dst + os_offset * this->dst_os_stride_ + oc_offset;
    args.bias = bias + (g * jcp_.oc + oc_offset) * bias_data_type_size_;
    args.scales = scales + scale_idx_mult_ * (g * jcp_.oc + oc_offset);
    args.sum_scale = sum_scale_;
    args.signed_scale = signed_scale;
    args.len = end - start;
    args.oc_offset = oc_offset;
    args.weights_zp = weights_zp + (g * jcp_.oc + oc_offset);
    args.weights_zp_compensation = weights_zp_compensation + os_offset;
    args.g_offset = g * jcp_.oc;
    ker_(&args);
};

template <data_type_t src_type, data_type_t dst_type>
void _gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>::ref_pp_ker_t::operator ()
        (dst_data_t *dst, acc_data_t *acc, const char *bias, const float *scales, float signed_scale,
         int g, size_t start, size_t end, float* weights_zp, int32_t* weights_zp_compensation)
{
    using math::get_bias;

    if (end <= start)
        return;

    const size_t first_oc = start % OC_;
    const size_t last_oc = (end - 1) % OC_;
    const size_t first_os = start / OC_;
    const size_t last_os = (end - 1) / OC_;

    // Fallback
    if (post_ops_.len_ == 0) {
        for (size_t os = first_os; os <= last_os; os++) {
            const size_t start_oc = (os == first_os) ? first_oc : 0;
            const size_t end_oc = (os == last_os) ? last_oc : OC_ - 1;
            for (size_t oc = start_oc; oc <= end_oc; oc++) {
                const size_t acc_off = os * jcp_.oc + oc;
                const size_t dst_off = os * this->dst_os_stride_ + oc;

                float d = (float)(acc[acc_off]);
                if (jcp_.signed_input)
                    d *= signed_scale;

                if (with_weights_zp_)
                    d -= weights_zp[g * jcp_.oc + oc] * (float)weights_zp_compensation[os];

                if (do_bias_)
                    d += get_bias(bias, g * jcp_.oc + oc,
                        bias_data_type_);

                d *= scales[(g * jcp_.oc + oc) * scale_idx_mult_];
                dst[dst_off] = qz_a1b0<float, dst_data_t>()(d, rmode_);
            }
        }
    } else {
        float* acc_fp = reinterpret_cast<float*>(acc);

        auto load = [&](int idx, size_t oc, size_t os, size_t acc_off, size_t dst_off) {
            float d;
            if (idx == 0) {
                d = (float) (acc[acc_off]);

                if (jcp_.signed_input)
                    d *= signed_scale;

                if (with_weights_zp_)
                    d -= weights_zp[g * jcp_.oc + oc] * (float)weights_zp_compensation[os];

                if (do_bias_)
                    d += get_bias(bias, g * jcp_.oc + oc,
                                  bias_data_type_);

                d *= scales[(g * jcp_.oc + oc) * scale_idx_mult_];
            } else {
                d = acc_fp[acc_off];
            }

            return d;
        };

        auto store = [&](int idx, float d, size_t acc_off, size_t dst_off) {
            if (idx == post_ops_.len_ - 1)
                dst[dst_off] = qz_a1b0<float, dst_data_t>()(d, rmode_);
            else
                acc_fp[acc_off] = d;
        };

        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        for (int i = 0; i < post_ops_.len_; i++) {
            auto &post_op = post_ops_.entry_[i];
            if (post_op.is_eltwise()) {
                for (size_t os = first_os; os <= last_os; os++) {
                    const size_t start_oc = (os == first_os) ? first_oc : 0;
                    const size_t end_oc = (os == last_os) ? last_oc : OC_ - 1;
                    for (size_t oc = start_oc; oc <= end_oc; oc++) {
                        const size_t acc_off = os * jcp_.oc + oc;
                        const size_t dst_off = os * this->dst_os_stride_ + oc;

                        float d = load(i, oc, os, acc_off, dst_off);

                        d = ref_eltwise_injectors_[eltwise_inj_idx]->compute_scalar(d);

                        store(i, d, acc_off, dst_off);
                    }
                }
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                for (size_t os = first_os; os <= last_os; os++) {
                    const size_t start_oc = (os == first_os) ? first_oc : 0;
                    const size_t end_oc = (os == last_os) ? last_oc : OC_ - 1;
                    for (size_t oc = start_oc; oc <= end_oc; oc++) {
                        const size_t acc_off = os * jcp_.oc + oc;
                        const size_t dst_off = os * this->dst_os_stride_ + oc;

                        auto depthwise_weights = post_op.depthwise.weights_data;
                        auto depthwise_bias = post_op.depthwise.biases_data;

                        float d = load(i, oc, os, acc_off, dst_off);

                        d = ref_depthwise_injectors_[depthwise_inj_idx]->compute_scalar(d, depthwise_weights + g * jcp_.oc + oc,
                                                                                        depthwise_bias + g * jcp_.oc + oc);

                        store(i, d, acc_off, dst_off);
                    }
                }
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                for (size_t os = first_os; os <= last_os; os++) {
                    const size_t start_oc = (os == first_os) ? first_oc : 0;
                    const size_t end_oc = (os == last_os) ? last_oc : OC_ - 1;
                    for (size_t oc = start_oc; oc <= end_oc; oc++) {
                        const size_t acc_off = os * jcp_.oc + oc;
                        const size_t dst_off = os * this->dst_os_stride_ + oc;

                        auto quant = post_op.quantization;
                        auto pcl = quant.crop_low_data->shifts_;
                        auto pch = quant.crop_high_data->shifts_;
                        auto pisc = quant.input_scale_data->scales_;
                        auto pish = quant.input_shift_data->shifts_;
                        auto posc = quant.output_scale_data->scales_;
                        auto posh = quant.output_shift_data->shifts_;

                        float d = load(i, oc, os, acc_off, dst_off);

                        int cl_idx = quant.crop_low_data->count_ == 1 ? 0 : g * jcp_.oc + oc;
                        int ch_idx = quant.crop_high_data->count_ == 1 ? 0 : g * jcp_.oc + oc;
                        int isc_idx = quant.input_scale_data->count_ == 1 ? 0 : g * jcp_.oc + oc;
                        int ish_idx = quant.input_shift_data->count_ == 1 ? 0 : g * jcp_.oc + oc;
                        int osc_idx = quant.output_scale_data->count_ == 1 ? 0 : g * jcp_.oc + oc;
                        int osh_idx = quant.output_shift_data->count_ == 1 ? 0 : g * jcp_.oc + oc;

                        d = nstl::min(pch[ch_idx], nstl::max(pcl[cl_idx], d));
                        d = d * pisc[isc_idx] + pish[ish_idx];
                        d = roundf(d);
                        d = d * posc[osc_idx] + posh[osh_idx];

                        store(i, d, acc_off, dst_off);
                    }
                }
            } else if (post_op.is_sum()) {
                for (size_t os = first_os; os <= last_os; os++) {
                    const size_t start_oc = (os == first_os) ? first_oc : 0;
                    const size_t end_oc = (os == last_os) ? last_oc : OC_ - 1;
                    for (size_t oc = start_oc; oc <= end_oc; oc++) {
                        const size_t acc_off = os * jcp_.oc + oc;
                        const size_t dst_off = os * this->dst_os_stride_ + oc;

                        float d = load(i, oc, os, acc_off, dst_off);

                        d += post_op.sum.scale * get_sum((char*)dst, dst_off, post_op.sum.data_type);

                        store(i, d, acc_off, dst_off);
                    }
                }
            }
        }
    }
};

template <data_type_t src_type, data_type_t dst_type>
void _gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>::
execute_forward_thr(const int ithr, const int nthr, const src_data_t *src_base,
        const wei_data_t *wei_base, const char *bia_base, dst_data_t *dst_base,
        const memory_tracking::grantor_t &scratchpad) const {
    const jit_gemm_conv_conf_t &jcp = this->pd()->jcp_;

    const auto src_md = memory_desc_wrapper(pd()->src_pd());
    const size_t src_mb_stride = src_md.blk_off(1);
    const size_t src_g_stride = src_md.blk_off(0, 1) * jcp.ic;

    const auto wei_md = memory_desc_wrapper(pd()->weights_pd(0));
    const size_t wei_g_stride = pd()->with_groups() ? wei_md.blk_off(1) : 0;

    const auto dst_md = memory_desc_wrapper(pd()->dst_pd());
    const size_t dst_mb_stride = dst_md.blk_off(1);
    const size_t dst_g_stride = dst_md.blk_off(0, 1) * jcp.oc;

    const float *scales = pd()->attr()->output_scales_.scales_;

    const uint8_t *input_zp_base = nullptr;
    if (jcp.with_input_zp) {
        input_zp_base = pd()->attr()->input_zero_points_.shifts_;
    }

    float *weights_zp = nullptr;
    int32_t *weights_zp_compensation = nullptr;
    if (jcp.with_weights_zp) {
        weights_zp = pd()->attr()->weights_zero_points_.shifts_;
        weights_zp_compensation = scratchpad.get<int32_t>(key_weights_zp_compensation) + ithr * jcp.oh * jcp.ow;
    }

    int32_t *output_compensation = nullptr;
    if (jcp.with_input_zp) {
        output_compensation = scratchpad.get<int32_t>(key_conv_padded_compensation);
    }

    uint8_t *__restrict col = scratchpad.get<uint8_t>(key_conv_gemm_col)
            + (ptrdiff_t)ithr * jcp.im2col_sz;
    src_data_t *__restrict imtr = scratchpad.get<src_data_t>(key_conv_gemm_imtr)
            + (ptrdiff_t)ithr * jcp.is * jcp.ic;
    acc_data_t *__restrict acc = scratchpad.get<acc_data_t>(key_conv_int_dat_in_acc_dt)
            + (ptrdiff_t)ithr * jcp.oh_block * jcp.ow_block * jcp.oc;

    const ptrdiff_t offset = (ptrdiff_t)jcp.ngroups * jcp.ks * jcp.ic * jcp.oc;
    const int32_t *_wei_comp = (jcp.with_input_zp) ? output_compensation
                                                   : (const int32_t *)(wei_base + offset);

    int g{ 0 }, n{ 0 }, ohb{ 0 }, owb{ 0 };
    size_t start = 0, end = 0;

    const bool is_problem_3d = pd()->ndims() == 5;
    assert(IMPLICATION(is_problem_3d,
                       jcp.oh_block == jcp.oh && jcp.ow_block == jcp.ow
                       && jcp.ic_block == jcp.ic));

    const int nb_oh = div_up(jcp.oh, jcp.oh_block);
    const int nb_ow = div_up(jcp.ow, jcp.ow_block);
    const size_t work_amount = (size_t)jcp.ngroups * jcp.mb * nb_oh * nb_ow;
    balance211(work_amount, nthr, ithr, start, end);
    nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, ohb,
                nb_oh, owb, nb_ow);
    uint8_t shift = jcp.signed_input ? 128 : 0;
    parallel_nd(jcp.im2col_sz, [&](ptrdiff_t i) { col[i] = shift; });

    for (size_t iwork = start; iwork < end; ++iwork) {
        int oh = ohb * jcp.oh_block;
        int ow = owb * jcp.ow_block;
        const src_data_t *__restrict src = src_base + n * src_mb_stride
            + g * src_g_stride;
        const wei_data_t *__restrict wei = wei_base + g * wei_g_stride;
        dst_data_t *__restrict dst =
                dst_base + n * dst_mb_stride + g * dst_g_stride;
        const int32_t *__restrict wei_comp = _wei_comp + g * jcp.oc;
        const int h_step = nstl::min(jcp.oh_block, jcp.oh - oh);
        const int w_step = nstl::min(jcp.ow_block, jcp.ow - ow);

        const uint8_t *__restrict input_zp = nullptr;
        if (jcp.with_input_zp)
            input_zp = input_zp_base + g * jcp.ic;

        if (jcp.im2col_sz && is_problem_3d)
            jit_gemm_convolution_utils::transpose_u8<src_data_t>(
                    jcp, src, imtr, input_zp);

        for (int od = 0; od < jcp.od; od++) {
            if (jcp.im2col_sz || jcp.with_weights_zp) {
                if (!is_problem_3d)
                    jit_gemm_convolution_utils::im2col_u8<src_data_t>(
                            jcp, src, imtr, col, oh, h_step, ow, w_step, input_zp, weights_zp_compensation);
                else
                    jit_gemm_convolution_utils::im2col_u8_3d<src_data_t>(jcp, imtr, col, od, input_zp,
                                                                         weights_zp_compensation);
            }

            const int M = jcp.oc;
            const int K = jcp.ks * jcp.ic;
            const int N = h_step * w_step;
            const int LDA = M * jcp.ngroups;
            const int LDB = jcp.im2col_sz ? N : K * jcp.ngroups;
            const char *BT = jcp.im2col_sz ? "T" : "N";
            const int8_t off_a = 0;
            const uint8_t off_b = 0;
            const int32_t off_c = 0;
            const float onef = 1.0f, zerof = 0.0f;
            gemm_s8x8s32("N", BT, (jcp.signed_input || jcp.with_input_zp) ? "C" : "F",
                &M, &N, &K, &onef, wei, &LDA, &off_a,
                jcp.im2col_sz ? col : (uint8_t *)src + od * jcp.ngroups * jcp.ic * N, &LDB, &off_b,
                &zerof, acc, &M, (jcp.signed_input || jcp.with_input_zp) ? wei_comp : &off_c);


            parallel(0, (size_t) N * jcp.oc, [&](int ithr, int nthr) {
                size_t start, end;
                balance211((size_t) N * jcp.oc, nthr, ithr, start, end);
                (*pp_ker_)(dst + (od * jcp.oh * jcp.ow + oh * jcp.ow + ow) * pp_ker_->dst_os_stride_,
                           acc, bia_base, scales, jcp.signed_input ? 1.f / jcp.wei_adj_scale : 1.f,
                           g, start, end, weights_zp, weights_zp_compensation);
            });
        }

        nd_iterator_step(n, jcp.mb, g, jcp.ngroups, ohb, nb_oh,
                    owb, nb_ow);
    }
}

template <data_type_t dst_type>
void _gemm_u8s8s32x_convolution_bwd_data_t<dst_type>::
execute_backward_data() const {
    auto diff_dst_base = reinterpret_cast<const diff_dst_data_t *>
            (this->input_memory(0));
    auto wei_base = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bia_base = reinterpret_cast<const char *>(this->input_memory(2));
    auto diff_src_base = reinterpret_cast<diff_src_data_t *>(this->memory());

    auto scratchpad = this->scratchpad();

    const jit_gemm_conv_conf_t &jcp = this->pd()->jcp_;

    const size_t work_amount = jcp.ngroups * jcp.mb;
    parallel(jcp.nthr, work_amount, [&](const int ithr, const int nthr) {
        execute_backward_data_thr(ithr, nthr, diff_dst_base, wei_base,
                bia_base, diff_src_base, scratchpad);
    });
}

template <data_type_t dst_type>
void _gemm_u8s8s32x_convolution_bwd_data_t<dst_type>::
execute_backward_data_thr(const int ithr, const int nthr,
        const diff_dst_data_t *diff_dst_base, const wei_data_t *wei_base,
        const char *bia_base, diff_src_data_t *diff_src_base,
        const memory_tracking::grantor_t &scratchpad) const
{
    const jit_gemm_conv_conf_t &jcp = this->pd()->jcp_;

    const auto diff_dst_md = memory_desc_wrapper(pd()->diff_dst_pd());
    const size_t diff_dst_mb_stride = diff_dst_md.blk_off(1);
    const size_t diff_dst_g_stride = diff_dst_md.blk_off(0, 1) * jcp.oc;

    const auto wei_md = memory_desc_wrapper(pd()->weights_pd(0));
    const size_t wei_g_stride = pd()->with_groups() ? wei_md.blk_off(1) : 0;

    const auto diff_src_md = memory_desc_wrapper(pd()->diff_src_pd());
    const size_t diff_src_mb_stride = diff_src_md.blk_off(1);
    const size_t diff_src_g_stride = diff_src_md.blk_off(0, 1) * jcp.ic;
    const size_t diff_src_os_stride = diff_src_md.blk_off(0, 0, 0, 1);

    /* scale_idx_mult = 1 for per_oc scales and 0, otherwise */
    const int scale_idx_mult = pd()->attr()->output_scales_.mask_ == (1 << 1);
    const float *scales = pd()->attr()->output_scales_.scales_;
    const auto rmode = pd()->attr()->round_mode_;
    const size_t work_amount = jcp.ngroups * jcp.mb;

    auto col = scratchpad.get<acc_data_t>(key_conv_gemm_col)
        + (ptrdiff_t)ithr * jcp.im2col_sz;
    auto acc = scratchpad.get<acc_data_t>(key_conv_int_dat_in_acc_dt)
        + (ptrdiff_t)ithr * jcp.is * jcp.ic;

    int n{0}, g{0};
    size_t start = 0, end = 0;

    balance211(work_amount, nthr, ithr, start, end);
    nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups);

    for (size_t iwork = start; iwork < end; ++iwork) {
        const diff_dst_data_t *diff_dst = diff_dst_base
            + n * diff_dst_mb_stride + g * diff_dst_g_stride;
        const wei_data_t *wei = wei_base + g * wei_g_stride;
        diff_src_data_t *diff_src = diff_src_base + n * diff_src_mb_stride
            + g * diff_src_g_stride;

        const int M = jcp.ks * jcp.ic;
        const int N = jcp.os;
        const int K = jcp.oc;
        const int8_t off_a = 0;
        const diff_dst_data_t off_b = 0;
        const int32_t off_c = 0;
        const float onef = 1.0, zerof = 0.0;
        const int LD = K * jcp.ngroups;

        gemm_s8x8s32("T", "N", "F", &M, &N, &K, &onef,
                wei, &LD, &off_a, diff_dst, &LD, &off_b,
                &zerof, jcp.im2col_sz ? col : acc, &M, &off_c);

        if (jcp.im2col_sz)
            jit_gemm_convolution_utils::col2im_s32(jcp, col, acc);

        parallel_nd(jcp.is, jcp.ic, [&](int is, int ic) {
            float d = (float)acc[is * jcp.ic + ic];
            if (jcp.with_bias)
                d += get_bias(bia_base, g * jcp.ic + ic,
                        pd()->desc()->bias_desc.data_type);
            d *= scales[(g * jcp.ic + ic) * scale_idx_mult];
            const size_t diff_src_off = is * diff_src_os_stride + ic;
            diff_src[diff_src_off] =
                qz_a1b0<float, diff_src_data_t>()(d, rmode);
        });
        nd_iterator_step(n, jcp.mb, g, jcp.ngroups);
    }
}

using namespace data_type;

template class _gemm_x8s8s32x_convolution_fwd_t<u8, f32>::jit_pp_ker_t<avx512_common>;
template class _gemm_x8s8s32x_convolution_fwd_t<u8, s32>::jit_pp_ker_t<avx512_common>;
template class _gemm_x8s8s32x_convolution_fwd_t<u8, s8>::jit_pp_ker_t<avx512_common>;
template class _gemm_x8s8s32x_convolution_fwd_t<u8, u8>::jit_pp_ker_t<avx512_common>;

template class _gemm_x8s8s32x_convolution_fwd_t<s8, f32>::jit_pp_ker_t<avx512_common>;
template class _gemm_x8s8s32x_convolution_fwd_t<s8, s32>::jit_pp_ker_t<avx512_common>;
template class _gemm_x8s8s32x_convolution_fwd_t<s8, s8>::jit_pp_ker_t<avx512_common>;
template class _gemm_x8s8s32x_convolution_fwd_t<s8, u8>::jit_pp_ker_t<avx512_common>;

template class _gemm_x8s8s32x_convolution_fwd_t<u8, f32>::jit_pp_ker_t<avx2>;
template class _gemm_x8s8s32x_convolution_fwd_t<u8, s32>::jit_pp_ker_t<avx2>;
template class _gemm_x8s8s32x_convolution_fwd_t<u8, s8>::jit_pp_ker_t<avx2>;
template class _gemm_x8s8s32x_convolution_fwd_t<u8, u8>::jit_pp_ker_t<avx2>;

template class _gemm_x8s8s32x_convolution_fwd_t<s8, f32>::jit_pp_ker_t<avx2>;
template class _gemm_x8s8s32x_convolution_fwd_t<s8, s32>::jit_pp_ker_t<avx2>;
template class _gemm_x8s8s32x_convolution_fwd_t<s8, s8>::jit_pp_ker_t<avx2>;
template class _gemm_x8s8s32x_convolution_fwd_t<s8, u8>::jit_pp_ker_t<avx2>;

template class _gemm_x8s8s32x_convolution_fwd_t<u8, f32>::jit_pp_ker_t<sse42>;
template class _gemm_x8s8s32x_convolution_fwd_t<u8, s32>::jit_pp_ker_t<sse42>;
template class _gemm_x8s8s32x_convolution_fwd_t<u8, s8>::jit_pp_ker_t<sse42>;
template class _gemm_x8s8s32x_convolution_fwd_t<u8, u8>::jit_pp_ker_t<sse42>;

template class _gemm_x8s8s32x_convolution_fwd_t<s8, f32>::jit_pp_ker_t<sse42>;
template class _gemm_x8s8s32x_convolution_fwd_t<s8, s32>::jit_pp_ker_t<sse42>;
template class _gemm_x8s8s32x_convolution_fwd_t<s8, s8>::jit_pp_ker_t<sse42>;
template class _gemm_x8s8s32x_convolution_fwd_t<s8, u8>::jit_pp_ker_t<sse42>;

template struct _gemm_x8s8s32x_convolution_fwd_t<u8, f32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<u8, s32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<u8, s8>;
template struct _gemm_x8s8s32x_convolution_fwd_t<u8, u8>;

template struct _gemm_x8s8s32x_convolution_fwd_t<s8, f32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<s8, s32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<s8, s8>;
template struct _gemm_x8s8s32x_convolution_fwd_t<s8, u8>;

template struct _gemm_u8s8s32x_convolution_bwd_data_t<f32>;
template struct _gemm_u8s8s32x_convolution_bwd_data_t<s32>;
template struct _gemm_u8s8s32x_convolution_bwd_data_t<s8>;
template struct _gemm_u8s8s32x_convolution_bwd_data_t<u8>;
}
}
}
