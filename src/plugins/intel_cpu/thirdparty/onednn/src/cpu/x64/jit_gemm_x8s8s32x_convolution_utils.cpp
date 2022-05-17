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

#include <cstdlib>
#include <functional>

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_gemm_x8s8s32x_conv_zp_src_pad_comp.hpp"
#include "cpu/x64/jit_gemm_x8s8s32x_convolution_utils.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace gemm_x8s8s32x_convolution_utils {
using namespace dnnl::impl::cpu::gemm_x8s8s32x_convolution_utils;

template <cpu_isa_t isa>
struct jit_pp_ker_t : pp_ker_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            gemm_x8s8s32x_convolution_utils::jit_pp_ker_t);

    jit_pp_ker_t(const convolution_pd_t *pd, const conv_gemm_conf_t &jcp)
            : pp_ker_t(pd, jcp)
            , do_eltwise_(false)
            , do_sum_(false)
            , sum_scale_(0)
            , sum_data_type_(dnnl_f32)
            , default_OC_loop_unroll_(4)
            , max_OC_loop_unroll_(isa == avx512_common ? 12 : 6)
            , idx_compute_vreg_start_(0)
            , idx_compute_vreg_max_(isa == avx512_common ? 31 : 15)
            , compute_vregs_per_iter_(1)
    {
        if (utils::one_of(isa, avx2, sse41)) {
            idx_compute_vreg_start_ += 2;   //  Vmm(0), Vmm(1) - for masks
        }
        if (do_scale_) {
            vreg_scale = Vmm(idx_compute_vreg_start_++);
        }
        dst_data_type_size_ = types::data_type_size(dst_data_type_);
        if (dst_data_type_ == data_type::u8 || utils::one_of(isa, avx2, sse41)) {
            vreg_zero = Vmm(idx_compute_vreg_start_++);
        }
        bool only_eltwise_or_sum = true;
        for (int idx = 0; idx < post_ops_.len(); ++idx) {
            const auto &e = post_ops_.entry_[idx];
            if (e.is_eltwise(true)) {
                do_eltwise_ = true;
            } else if (e.is_sum()) {
                do_sum_ = true;
                sum_scale_ = e.sum.scale;
                sum_data_type_ = e.sum.dt;
            } else {
                only_eltwise_or_sum = false;
            }
        }
        if (post_ops_.len() > 0 && !only_eltwise_or_sum) {
            vreg_d_weights = Vmm(idx_compute_vreg_max_--);
            vreg_d_bias = Vmm(idx_compute_vreg_max_--);
        }

        do_signed_scaling_ = jcp_.signed_input;
        if (do_signed_scaling_)
            vreg_signed_scale = Vmm(idx_compute_vreg_start_++);

        if (do_bias_) {
            bias_data_type_size_ = types::data_type_size(bias_data_type_);
            compute_vregs_per_iter_++;
        }
        if (do_sum_) {
            vreg_sum_scale = Vmm(idx_compute_vreg_start_++);
            compute_vregs_per_iter_++;
        }

        for (int i = 0; i < post_ops_.len(); i++) {
            auto &post_op = post_ops_.entry_[i];
            if (post_op.is_eltwise()) {
                jit_eltwise_injectors_.push_back(new jit_uni_eltwise_injector_f32<isa>(
                        this, post_op.eltwise, true, eltwise_reserved, mask_post_op_reserved));
            } else if (post_op.is_depthwise()) {
                jit_depthwise_injectors_.push_back(new jit_uni_depthwise_injector_f32<isa>(
                        this, post_op, mask_post_op_reserved));
            }
        }

        int max_unroll = (idx_compute_vreg_max_ - idx_compute_vreg_start_ + 1) / compute_vregs_per_iter_;
        max_OC_loop_unroll_ = nstl::min(max_OC_loop_unroll_, max_unroll);
        default_OC_loop_unroll_ = nstl::min(default_OC_loop_unroll_, max_unroll);
    }
    ~jit_pp_ker_t() {
        for (auto inj : jit_eltwise_injectors_)
            delete inj;
        jit_eltwise_injectors_.clear();
        for (auto inj : jit_depthwise_injectors_)
            delete inj;
        jit_depthwise_injectors_.clear();
    }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    void operator()(void *void_dst, acc_data_t *acc, const char *bias, const float *scales, float sum_scale, float signed_scale,
                    int g, size_t start, size_t end,
                    const zero_point_call_params_t &zp,
                    const void * post_ops_binary_rhs_arg_vec,
                    const void * /* dst_orig */, const exec_ctx_t &ctx,
                    const memory_desc_t &dst_md,
        const single_gemm_conv_chunk_desc_t &chunk_desc) const override {

        if (end <= start) return;

        char *dst = (char *)void_dst;

        ker_args_t args;
        size_t oc_offset = start % OC_;
        size_t os_offset = start / OC_;
        args.acc = acc + start;
        args.dst = dst
                   + (os_offset * dst_os_stride_ + oc_offset)
                     * dst_data_type_size_;
        args.bias = bias + (g * jcp_.oc + oc_offset) * bias_data_type_size_;
        args.scales = scales + scale_idx_mult_ * (g * jcp_.oc + oc_offset);
        args.sum_scale = sum_scale_;
        args.signed_scale = signed_scale;
        args.len = end - start;
        args.oc_offset = oc_offset;
        args.g_offset = g * jcp_.oc;
        args.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec;
        jit_generator::operator()(&args);
    }

private:
    void generate() override;

    struct ker_args_t {
        char *dst;
        const acc_data_t *acc;
        const char *bias;
        const float *scales;
        float sum_scale;
        float signed_scale;
        size_t len;
        size_t oc_offset;
        size_t g_offset;
        const void *post_ops_binary_rhs_arg_vec;
    };

    nstl::vector<jit_uni_eltwise_injector_f32<isa> *> jit_eltwise_injectors_;
    nstl::vector<jit_uni_depthwise_injector_f32<isa> *> jit_depthwise_injectors_;

    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    static const size_t vlen = cpu_isa_traits<isa>::vlen / sizeof(float);

    Xbyak::Reg64 reg_param = abi_param1;
    Xbyak::Reg64 reg_dst = rdx;
    Xbyak::Reg64 reg_acc = rax;
    Xbyak::Reg64 reg_bias = rbx;
    Xbyak::Reg64 reg_scales = rsi;
    Xbyak::Reg64 reg_g_offset = rbp;

    Xbyak::Reg64 reg_len = r8;
    Xbyak::Reg64 reg_tmp = rcx; // intentional for shifting purposes
    Xbyak::Reg64 reg_oc_offset = r9;
    Xbyak::Reg64 reg_rem_mask_short = r10;
    Xbyak::Opmask kreg_rem_mask_short = k1;

    Vmm vreg_zero, vreg_scale, vreg_sum_scale, vreg_signed_scale, vreg_comp;

    //  sse41/avx2
    Xbyak::Reg64 reg_ptr_maskmovdqu_dst = rdi; // sse41: store destination - must be rdi
    Xbyak::Label l_table;
    Xbyak::Reg64 reg_table = r12;
    Xbyak::Reg64 reg_shift_table = r13;
    Vmm vreg_mask = Vmm(0); //  sse41: mask for blendvps must be in xmm0
    Vmm vreg_store_mask = Vmm(1);

    //  post_ops
    Xbyak::Opmask mask_post_op_reserved = k2;
    Xbyak::Reg64 eltwise_reserved = rax;
    Xbyak::Reg64 reg_d_weights = r14;
    Xbyak::Reg64 reg_d_bias = r15;
    Vmm vreg_d_weights, vreg_d_bias;

    size_t dst_data_type_size_ = 0;
    size_t bias_data_type_size_ = 0;

    bool do_eltwise_;
    bool do_sum_;
    float sum_scale_;
    data_type_t sum_data_type_;
    bool do_signed_scaling_;

    int default_OC_loop_unroll_;
    int max_OC_loop_unroll_;
    int idx_compute_vreg_start_;
    int idx_compute_vreg_max_;
    int compute_vregs_per_iter_;

    int idx_vreg_dst(int iter) {
        int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_ + 0;
        assert(idx <= idx_compute_vreg_max_);
        return idx;
    }
    int idx_vreg_bias(int iter) {
        int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_ + 1;
        assert(idx <= idx_compute_vreg_max_);
        return idx;
    }
    int idx_vreg_prev_dst(int iter) {
        int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_ + 2;
        assert(idx <= idx_compute_vreg_max_);
        return idx;
    }

    Vmm vreg_dst(int idx) { return Vmm(idx_vreg_dst(idx)); };
    Xbyak::Ymm ymm_dst(int idx) { return Xbyak::Ymm(idx_vreg_dst(idx)); };
    Xbyak::Xmm xmm_dst(int idx) { return Xbyak::Xmm(idx_vreg_dst(idx)); };
    Vmm vreg_bias(int idx) { return Vmm(idx_vreg_bias(idx)); };
    Vmm vreg_prev_dst(int idx) { return Vmm(idx_vreg_prev_dst(idx)); };
};

template <cpu_isa_t isa>
void jit_pp_ker_t<isa>::generate() {
    using namespace Xbyak;
    using namespace utils;

    preamble();

    const auto &p = post_ops_;
    std::size_t post_ops_pointers_count = 0;
    for (int i = 0; i < p.len(); i++) {
        if (p.entry_[i].is_depthwise() || p.entry_[i].is_quantization()) {
            post_ops_pointers_count++;
        }
    }

#define PARAM_OFF(x) offsetof(ker_args_t, x)
    if (post_ops_pointers_count != 0) {
        sub(rsp, post_ops_pointers_count * sizeof(float *));

        auto aux_reg0 = reg_dst;
        auto aux_reg1 = reg_acc;

        mov(aux_reg0, ptr[reg_param + PARAM_OFF(post_ops_binary_rhs_arg_vec)]);
        for (size_t i = 0; i < post_ops_pointers_count; i++) {
            mov(aux_reg1, ptr[aux_reg0 + i * sizeof(float *)]);
            mov(ptr[rsp + i * sizeof(float *)], aux_reg1);
        }
    }

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
#undef PARAM_OFF

    if (do_eltwise_ || dst_data_type_ == data_type::u8 || utils::one_of(isa, avx2, sse41))
        uni_vpxor(vreg_zero, vreg_zero, vreg_zero);

    if (utils::one_of(isa, avx2, sse41))
        mov(reg_table, l_table);

    auto apply_post_ops = [&](size_t offset, int idx) {
        std::size_t post_ops_data_offset = 0;
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        for (int i = 0; i < post_ops_.len(); i++) {
            auto& post_op = post_ops_.entry_[i];
            if (post_op.is_sum()) {
                auto dst_addr = ptr[reg_dst + offset * dst_data_type_size_];
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

                const Xbyak::RegExp depthwise_arg_base = rsp + post_ops_data_offset;
                mov(reg_d_weights, ptr[depthwise_arg_base]);
                lea(reg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float) + offset]);

                jit_depthwise_injectors_[depthwise_inj_idx]->compute_vector_range(vreg_dst(idx).getIdx(), vreg_dst(idx).getIdx() + 1, reg_d_weights, reg_d_weights);

                sub(reg_oc_offset, reg_g_offset);

                post_ops_data_offset += jit_depthwise_injectors_[depthwise_inj_idx]->memoryStep();
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                add(reg_oc_offset, reg_g_offset);
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || dst_data_type_ == dnnl_f32 || i != post_ops_.len() - 1;

                const Xbyak::RegExp quantization_arg_base = rsp + post_ops_data_offset;
                size_t crop_low_off = post_op.quantization.offset[post_op.quantization.crop_low] * sizeof(float);
                if (post_op.quantization.per_channel[post_op.quantization.crop_low]) {
                    mov(reg_d_weights, ptr[quantization_arg_base]);
                    uni_vmovups(vreg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float) + offset * sizeof(float) + crop_low_off]);
                } else {
                    mov(reg_d_weights, ptr[quantization_arg_base]);
                    uni_vbroadcastss(vreg_d_weights, ptr[reg_d_weights + crop_low_off]);
                }

                size_t crop_high_off = post_op.quantization.offset[post_op.quantization.crop_high] * sizeof(float);
                if (post_op.quantization.per_channel[post_op.quantization.crop_high]) {
                    mov(reg_d_bias, ptr[quantization_arg_base]);
                    uni_vmovups(vreg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float) + offset * sizeof(float) + crop_high_off]);
                } else {
                    mov(reg_d_bias, ptr[quantization_arg_base]);
                    uni_vbroadcastss(vreg_d_bias, ptr[reg_d_bias + crop_high_off]);
                }

                uni_vmaxps(vreg_dst(idx), vreg_dst(idx), vreg_d_weights);
                uni_vminps(vreg_dst(idx), vreg_dst(idx), vreg_d_bias);

                size_t inp_scale_off = post_op.quantization.offset[post_op.quantization.inp_scale] * sizeof(float);
                if (post_op.quantization.per_channel[post_op.quantization.inp_scale]) {
                    mov(reg_d_weights, ptr[quantization_arg_base]);
                    uni_vmovups(vreg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float) + offset * sizeof(float) + inp_scale_off]);
                } else {
                    mov(reg_d_weights, ptr[quantization_arg_base ]);
                    uni_vbroadcastss(vreg_d_weights, ptr[reg_d_weights + inp_scale_off]);
                }

                size_t inp_shift_off = post_op.quantization.offset[post_op.quantization.inp_shift] * sizeof(float);
                if (post_op.quantization.per_channel[post_op.quantization.inp_shift]) {
                    mov(reg_d_bias, ptr[quantization_arg_base]);
                    uni_vmovups(vreg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float) + offset * sizeof(float) + inp_shift_off]);
                } else {
                    mov(reg_d_bias, ptr[quantization_arg_base]);
                    uni_vbroadcastss(vreg_d_bias, ptr[reg_d_bias + inp_shift_off]);
                }

                uni_vfmadd213ps(vreg_dst(idx), vreg_d_weights, vreg_d_bias);

                if (do_rounding)
                    uni_vroundps(vreg_dst(idx), vreg_dst(idx), 0);

                if (do_dequantization) {
                    size_t output_scale_off = post_op.quantization.offset[post_op.quantization.output_scale] * sizeof(float);
                    if (post_op.quantization.per_channel[post_op.quantization.output_scale]) {
                        mov(reg_d_weights, ptr[quantization_arg_base ]);
                        uni_vmovups(vreg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float) + offset * sizeof(float) + output_scale_off]);
                    } else {
                        mov(reg_d_weights, ptr[quantization_arg_base]);
                        uni_vbroadcastss(vreg_d_weights, ptr[reg_d_weights + output_scale_off]);
                    }

                    size_t output_shift_off = post_op.quantization.offset[post_op.quantization.output_shift] * sizeof(float);
                    if (post_op.quantization.per_channel[post_op.quantization.output_shift]) {
                        mov(reg_d_bias, ptr[quantization_arg_base]);
                        uni_vmovups(vreg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float) + offset * sizeof(float) + output_shift_off]);
                    } else {
                        mov(reg_d_bias, ptr[quantization_arg_base]);
                        uni_vbroadcastss(vreg_d_bias, ptr[reg_d_bias + output_shift_off]);
                    }

                    uni_vfmadd213ps(vreg_dst(idx), vreg_d_weights, vreg_d_bias);
                }
                sub(reg_oc_offset, reg_g_offset);

                post_ops_data_offset += sizeof(float*);
            }
        }
    };

    // Load accumulated value, convert to float,
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
                    if (isa != sse41) {
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
                if (isa != sse41) {
                    uni_vblendvps(vreg_dst_, vreg_zero, acc_addr, vreg_mask);
                } else {
                    uni_vmovups(vreg_dst_, acc_addr);
                }
                uni_vcvtdq2ps(vreg_dst_, vreg_dst_);
            } else {
                if (isa == sse41) {
                    uni_vmovups(vreg_dst_, acc_addr);
                    uni_vcvtdq2ps(vreg_dst_, vreg_dst_);
                } else {
                    uni_vcvtdq2ps(vreg_dst_, acc_addr);
                }
            }
        }

        if (do_signed_scaling_)
            uni_vmulps(vreg_dst(idx), vreg_dst(idx), vreg_signed_scale);

        if (do_bias_) {
            auto bias_addr = ptr[reg_bias + offset * bias_data_type_size_];
            auto vreg_bias_ = vreg_bias(idx);
            if (isa == avx512_common && apply_mask)
                vreg_bias_ = vreg_bias_ | kreg_rem_mask_short;

            switch (bias_data_type_) {
                case data_type::s8: uni_vpmovsxbd(vreg_bias_, bias_addr); break;
                case data_type::u8: uni_vpmovzxbd(vreg_bias_, bias_addr); break;
                case data_type::s32:
                case data_type::f32: uni_vmovups(vreg_bias_, bias_addr); break;
                default: assert(!"unimplemented");
            }
            if (bias_data_type_ != data_type::f32)
                uni_vcvtdq2ps(vreg_bias(idx), vreg_bias(idx));
            uni_vaddps(vreg_dst(idx), vreg_dst(idx), vreg_bias(idx));
        }

        if (do_scale_)
            uni_vmulps(vreg_dst(idx), vreg_dst(idx), vreg_scale);

        apply_post_ops(offset, idx);

        if (dst_data_type_ != data_type::f32) {
            if (isa == avx512_common) {
                auto rmode_control = T_rn_sae;
                vcvtps2dq(vreg_dst(idx) | rmode_control, vreg_dst(idx));
            } else {
                uni_vcvtps2dq(vreg_dst(idx), vreg_dst(idx));
            }
        }

        if (dst_data_type_ == data_type::u8)
            uni_vpmaxsd(vreg_dst(idx), vreg_dst(idx), vreg_zero);

        auto dst_addr = ptr[reg_dst + offset * dst_data_type_size_];
        switch (dst_data_type_) {
            case data_type::s8:
                if (isa == avx512_common) {
                    vpmovsdb(dst_addr, vreg_dst_);
                } else {
                    uni_vpackssdw(vreg_dst_, vreg_dst_, vreg_dst_);
                    if (isa != sse41)
                        vpermq(ymm_dst(idx), ymm_dst(idx), 0x08);
                    uni_vpacksswb(vreg_dst_, vreg_dst_, vreg_dst_);
                    if (apply_mask) {
                        lea(reg_ptr_maskmovdqu_dst, dst_addr);
                        maskmovdqu(vreg_dst_, vreg_store_mask);
                    } else {
                        if (isa != sse41) {
                            vmovq(dst_addr, xmm_dst(idx));
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
                    if (isa != sse41)
                        vpermq(ymm_dst(idx), ymm_dst(idx), 0x08);
                    uni_vpackuswb(vreg_dst_, vreg_dst_, vreg_dst_);
                    if (apply_mask) {
                        lea(reg_ptr_maskmovdqu_dst, dst_addr);
                        maskmovdqu(vreg_dst_, vreg_store_mask);
                    } else {
                        if (isa != sse41) {
                            vmovq(dst_addr, xmm_dst(idx));
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
                        if (isa != sse41) {
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
        add(reg_dst, offset * dst_data_type_size_);
        add(reg_acc, offset * sizeof(acc_data_t));
        if (scale_idx_mult_) {
            assert(scale_idx_mult_ == 1);
            add(reg_scales, offset * sizeof(float));
        }
        if (do_bias_)
            add(reg_bias, offset * bias_data_type_size_);
    };

    // Advance all pointers by a value stored in a register
    auto advance_ptrs_reg = [&](Reg64 offset) {
        lea(reg_dst, ptr[reg_dst + offset * dst_data_type_size_]);
        lea(reg_acc, ptr[reg_acc + offset * sizeof(acc_data_t)]);
        if (scale_idx_mult_) {
            assert(scale_idx_mult_ == 1);
            lea(reg_scales, ptr[reg_scales + offset * sizeof(float)]);
        }
        if (do_bias_)
            lea(reg_bias, ptr[reg_bias + offset * bias_data_type_size_]);
    };

    // Rewind pointers that point to data that is indexed by output channel
    // (bias or per-oc scaling factors)
    auto rewind_ptrs = [&]() {
        if (do_bias_)
            sub(reg_bias, OC_ * bias_data_type_size_);
        if (scale_idx_mult_) {
            assert(scale_idx_mult_ == 1);
            sub(reg_scales, OC_ * sizeof(float));
        }
        add(reg_dst, (dst_os_stride_ - OC_) * dst_data_type_size_);
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

    bool do_post_ops = post_ops_.len() != 0;

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
        L(prologue_loop);
        {
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
            if (dst_data_type_ == data_type::s8 || dst_data_type_ == data_type::u8) {
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
            if (isa == avx512_common) {
                unsigned tail_mask = (1 << vlen_tail) - 1;
                mov(reg_tmp, tail_mask);
                kmovq(kreg_rem_mask_short, reg_tmp);
            } else {
                mov(reg_shift_table, vlen - vlen_tail);
                uni_vmovups(vreg_mask, ptr[reg_table + reg_shift_table * sizeof(float)]);
                if (dst_data_type_ == data_type::s8 || dst_data_type_ == data_type::u8) {
                    mov(reg_shift_table, vlen * sizeof(float));
                    sub(reg_shift_table, vlen_tail);
                    uni_vmovups(vreg_store_mask, ptr[reg_table + reg_shift_table]);
                }
            }
        }

        Label main_loop;
        L(main_loop);
        {
            if (do_post_ops)
                mov(reg_oc_offset, 0);

            if (OC_loop) {
                mov(reg_tmp, rnd_dn(OC_, OC_loop));
                Label oc_loop;
                L(oc_loop);
                {
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
        L(epilogue_loop);
        {
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
            if (dst_data_type_ == data_type::s8 || dst_data_type_ == data_type::u8) {
                mov(reg_shift_table, vlen * sizeof(float));
                sub(reg_shift_table, reg_tmp);
                uni_vmovups(vreg_store_mask, ptr[reg_table + reg_shift_table]);
            }
        }
        compute(0, 0, true);
    }

    L(epilogue_end);

    if (post_ops_pointers_count != 0) {
        add(rsp, post_ops_pointers_count * sizeof(float *));
    }

    postamble();

    for (auto& inj : jit_eltwise_injectors_)
        inj->prepare_table();

    if (utils::one_of(isa, avx2, sse41)) {
        align(64);
        L(l_table);
        for (size_t i = 0; i < vlen; i++) dd(0xFFFFFFFF);
        for (size_t i = 0; i < vlen; i++) dd(0x00000000);
    }
}

pp_ker_t *jit_pp_ker_create(
        const convolution_pd_t *pd, const conv_gemm_conf_t &jcp) {
    if (mayiuse(avx512_common)) {
        return new jit_pp_ker_t<avx512_common>(pd, jcp);
    } else if (mayiuse(avx2)) {
        return new jit_pp_ker_t<avx2>(pd, jcp);
    } else if (mayiuse(sse41)) {
        return new jit_pp_ker_t<sse41>(pd, jcp);
    }
    return nullptr;
}

} // namespace gemm_x8s8s32x_convolution_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
