/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
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
#include <cpu/cpu_primitive.hpp>

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/x64/injectors/injector_utils.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_avx512_core_x8s8s32x_1x1_conv_kernel.hpp"
#include "cpu/x64/jit_uni_1x1_conv_utils.hpp"

#define GET_OFF(field) offsetof(jit_1x1_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::prop_kind;
using namespace Xbyak;

template <typename Vmm>
_jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::
        _jit_avx512_core_x8s8s32x_1x1_conv_kernel(
                const jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr,
                const memory_desc_t &dst_md)
    : jcp(ajcp), attr_(attr), postops_injector_(nullptr) {
    if (jcp.with_eltwise || jcp.with_binary || jcp.with_sum || jcp.with_depthwise || jcp.with_quantization) {
        using namespace binary_injector;
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = false;
        static constexpr unsigned helper_vmm_idx = 31;
        const size_t oc_block_tail = jcp.oc_block % isa_simd_width_;
        const size_t tail_size = oc_block_tail
                ? oc_block_tail
                : jcp.oc_without_padding % isa_simd_width_;
        static constexpr bool use_exact_tail_scalar_bcast = true;

        const rhs_arg_static_params_t rhs_arg_static_params {helper_vmm_idx,
                r14, r15, preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec),
                memory_desc_wrapper(dst_md), tail_size, postops_mask,
                use_exact_tail_scalar_bcast};
        const static_params_t static_params {
                this->param1, rhs_arg_static_params};
        quantization_injector::static_params_t quantization_static_params
                {zmm_d_weights.getIdx(), zmm_d_bias.getIdx(), reg_d_weights, reg_d_bias};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<avx512_core>>(
                this, jcp.post_ops, static_params, quantization_static_params);
    }
}

template <typename Vmm>
void _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::bcast_loop(
        int load_loop_blk) {
    mov(aux1_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_bcast_data, reg_bcast_data);

    mov(aux_reg_output_data, reg_output_data);
    mov(bcast_loop_iter, EVEX_compress_addr(rsp, bcast_loop_work_off));

    Label bcast_loop;
    Label bcast_loop_tail;

    cmp(bcast_loop_iter, jcp.ur);
    jl(bcast_loop_tail, T_NEAR);

    L(bcast_loop);
    {
        assert(jcp.bcast_block % jcp.ur == 0);
        int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        for (int i = 0; i < num_substeps; i++) {
            reduce_loop(load_loop_blk, jcp.ur, i, false);
            if (i < num_substeps - 1) {
                add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_substep);
                add(aux_reg_output_data, jcp.bcast_loop_output_substep);
            } else {
                add(aux1_reg_bcast_data,
                        jcp.bcast_loop_bcast_step
                                - (num_substeps - 1)
                                        * jcp.bcast_loop_bcast_substep);
                int output_offset = jcp.bcast_loop_output_step
                        - (num_substeps - 1) * jcp.bcast_loop_output_substep;

                add(aux_reg_output_data, output_offset);
            }
        }
        sub(bcast_loop_iter, jcp.bcast_block);
        cmp(bcast_loop_iter, jcp.bcast_block);
        jge(bcast_loop, T_NEAR);
    }

    L(bcast_loop_tail);
    if (jcp.ur_tail) {
        Label bcast_loop_tail_out;
        cmp(bcast_loop_iter, 0);
        jz(bcast_loop_tail_out, T_NEAR);
        reduce_loop(load_loop_blk, jcp.ur_tail, 0, true);
        L(bcast_loop_tail_out);
    }
}

template <typename Vmm>
void _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::cvt2ps(data_type_t type_in,
        const Vmm vmm_in, const Xbyak::Operand &op, bool mask_flag) {
    const Vmm vmm = mask_flag ? vmm_in | k_load_dim_mask | T_z : vmm_in;
    switch (type_in) {
        case data_type::f32:
        case data_type::s32: vmovups(vmm, op); break;
        case data_type::s8: vpmovsxbd(vmm, op); break;
        case data_type::u8: vpmovzxbd(vmm, op); break;
        default: assert(!"unsupported data type");
    }
    if (type_in != data_type::f32) vcvtdq2ps(vmm_in, vmm_in);
}

template <typename F>
static void iterate(const int load_loop_blk, const int ur,
        const bool last_oc_block_flag, const bool force_masking, const F &f) {
    for (int i_load = 0; i_load < load_loop_blk; i_load++) {
        const bool mask_flag = force_masking
                || (last_oc_block_flag && i_load + 1 == load_loop_blk);
        for (int i_ur = 0; i_ur < ur; i_ur++)
            f(mask_flag, i_load, i_ur);
    }
}
template <typename F>
static void iterate(const int load_loop_blk, const int ur,
        const bool last_oc_block_flag, const F &f) {
    iterate(load_loop_blk, ur, last_oc_block_flag, false, f);
}
template <typename F>
static void iterate(const int load_loop_blk, const int ur, const F &f) {
    iterate(load_loop_blk, ur, false, false, f);
}

template <typename Vmm>
Address _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::output_ptr(
        const int i_load, const int i_ur) {
    const size_t ur_stride = jcp.with_dw_conv
            ? jcp.nb_load_blocking * jcp.oc_block * i_ur
            : jcp.oc_without_padding * jcp.ngroups * i_ur;

    return EVEX_compress_addr(aux_reg_output_data,
            jcp.typesize_out * (ur_stride + i_load * jcp.load_block));
};

template <typename Vmm>
int _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::vreg_accum_idx(
        const int load_loop_blk, int i_load, int i_ur) const {
    return (i_ur * load_loop_blk + i_load);
};

template <typename Vmm>
Vmm _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::vreg_accum(
        const int load_loop_blk, int i_load, int i_ur) const {
    return Vmm(vreg_accum_idx(load_loop_blk, i_load, i_ur));
};

template <typename Vmm>
void _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::apply_sum(
        const int load_loop_blk, const int ur, const bool mask_flag_in,
        const float *p_sum_scale, const int32_t *p_sum_zp) {
    if (jcp.with_sum) {
        const float sum_scale = *p_sum_scale;
        const int32_t sum_zp = *p_sum_zp;
        const auto sum_injector_lam
                = [this, sum_scale, sum_zp, load_loop_blk](const bool mask_flag,
                          const int i_load, const int i_ur) {
                      const auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                      cvt2ps(jcp.sum_dt, vmm_prev_dst, output_ptr(i_load, i_ur),
                              mask_flag);
                      if (sum_zp != 0) vsubps(vmm_prev_dst, vmm_tmp);
                      if (sum_scale == 1.f)
                          vaddps(r, vmm_prev_dst);
                      else
                          vfmadd231ps(
                                  r, vmm_prev_dst, zword_b[reg_ptr_sum_scale]);
                  };
        const auto sum_injector = [=]() {
            iterate(load_loop_blk, ur, mask_flag_in, sum_injector_lam);
        };
        if (sum_zp != 0) vcvtdq2ps(vmm_tmp, ptr_b[rsp + reg_ptr_sum_zp_off]);
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);
    }
}

template <typename Vmm>
void _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::apply_postops(
        const int load_loop_blk, const int ur, const bool mask_flag_in,
        const float *p_sum_scale, const int32_t *p_sum_zp) {
    if (jcp.with_eltwise || jcp.with_binary || jcp.with_sum || jcp.with_depthwise || jcp.with_quantization) {
        std::map<size_t, int> vmm_idx_off;
        iterate(load_loop_blk, ur,
                [&](const bool, const int i_load, const int i_ur) {
                    vmm_idx_off.insert({vreg_accum_idx(load_loop_blk, i_load, i_ur), i_load * jcp.load_block * sizeof(float)});
                });
        depthwise_injector::dynamic_params_t ddp {zmm_d_weights.getIdx(), zmm_d_bias.getIdx(), reg_d_weights, reg_d_bias,
                                                  reg_oc_off, vmm_idx_off,
                                                  this->rsp, base_post_ops_data_offset};
        quantization_injector::dynamic_params_t qdp {reg_oc_off, vmm_idx_off, jcp.dst_dt,
                                                     this->rsp, base_post_ops_data_offset};

        apply_sum(load_loop_blk, ur, mask_flag_in, p_sum_scale, p_sum_zp);

        injector_utils::vmm_index_set_t vmm_idxs;
        if (jcp.with_binary) {
            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params,
                    rhs_arg_params_tail;
            const auto mask_tail = jcp.oc_without_padding % jcp.load_block;
            const bool oc_blk_is_smaller_than_vmm
                    = jcp.oc_block < isa_simd_width_;
            iterate(load_loop_blk, ur, mask_tail, oc_blk_is_smaller_than_vmm,
                    [&](const bool mask_flag, const int i_load,
                            const int i_ur) {
                        const int ur_stride = jcp.with_dw_conv
                                ? jcp.nb_load_blocking * jcp.oc_block * i_ur
                                : jcp.oc_without_padding * jcp.ngroups * i_ur;
                        const int aux_output_l_off
                                = (ur_stride + i_load * jcp.load_block);
                        const auto vmm_idx
                                = vreg_accum_idx(load_loop_blk, i_load, i_ur);
                        vmm_idxs.emplace(vmm_idx);

                        rhs_arg_params_tail.vmm_idx_to_oc_elem_off_addr.emplace(
                                vmm_idx, ptr[param1 + GET_OFF(oc_l_off)]);
                        rhs_arg_params_tail.vmm_idx_to_oc_elem_off_val.emplace(
                                vmm_idx, i_load * jcp.load_block);
                        rhs_arg_params_tail.vmm_idx_to_oc_off_oprnd.emplace(
                                vmm_idx, oc_off_oprnd);
                        rhs_arg_params_tail.vmm_idx_to_out_off_oprnd.emplace(
                                vmm_idx, out_off_oprnd);
                        rhs_arg_params_tail.vmm_idx_to_out_elem_off_val.emplace(
                                vmm_idx, aux_output_l_off);
                        if (mask_flag)
                            rhs_arg_params_tail.vmm_tail_idx_.emplace(vmm_idx);
                    });
            rhs_arg_params = rhs_arg_params_tail;
            rhs_arg_params.vmm_tail_idx_.clear();

            const injector_utils::register_preserve_guard_t register_guard(
                    this, {out_off_oprnd});
            const size_t reg_guard_stack_occupied
                    = register_guard.stack_space_occupied();

            mov(abi_param1,
                    EVEX_compress_addr(rsp,
                            reg_abi_param1_backup + reg_guard_stack_occupied));
            mov(oc_off_oprnd,
                    EVEX_compress_addr(rsp,
                            reg_binary_post_op_acc_off
                                    + reg_guard_stack_occupied));
            mov(out_off_oprnd, aux_reg_output_data);
            sub(out_off_oprnd, ptr[param1 + GET_OFF(dst_orig)]);
            shr(out_off_oprnd, std::log2(types::data_type_size(jcp.dst_dt)));

            Label postops_done;
            if (mask_tail || oc_blk_is_smaller_than_vmm) {
                Label postops_no_tail;
                if (mask_tail) {
                    test(reg_reduce_pos_flag, FLAG_OC_LAST);
                    jz(postops_no_tail, T_NEAR);
                    cmp(reg_load_loop_work, 0);
                    jg(postops_no_tail, T_NEAR);
                }
                postops_injector_->compute_vector_range(
                        vmm_idxs, rhs_arg_params_tail);
                jmp(postops_done, T_NEAR);
                L(postops_no_tail);
            }
            postops_injector_->compute_vector_range(vmm_idxs, rhs_arg_params, ddp, qdp);
            L(postops_done);

        } else {
            iterate(load_loop_blk, ur,
                    [&](const bool, const int i_load, const int i_ur) {
                        vmm_idxs.emplace(
                                vreg_accum_idx(load_loop_blk, i_load, i_ur));
                    });
            postops_injector_->compute_vector_range(vmm_idxs, binary_injector::rhs_arg_dynamic_params_t(), ddp, qdp);
        }
    }
}

template <typename Vmm>
void _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::reduce_loop(
        int load_loop_blk, int ur, int substep, bool wraparound) {
    auto vreg_load
            = [=](int i_load) { return Vmm(ur * load_loop_blk + i_load); };

    auto vmm_bias_alpha = [=]() { return Vmm(ur * load_loop_blk); };

    auto xmm_bias_alpha = [=]() { return Xmm(ur * load_loop_blk); };
    auto bias_ptr = [=](int i_load) {
        return EVEX_compress_addr(
                reg_bias_data, jcp.typesize_bia * jcp.oc_block * i_load);
    };

    auto comp_ptr = [=](int i_load) {
        return EVEX_compress_addr(
                reg_comp_data, sizeof(int32_t) * jcp.oc_block * i_load);
    };

    auto scale_ptr = [=](int i_load) {
        return EVEX_compress_addr(reg_ptr_scales,
                jcp.is_oc_scale * (sizeof(float) * jcp.oc_block * i_load));
    };

    auto bcast_ptr = [=](int i_reduce, int i_ur, bool bcast) {
        assert(i_ur < jcp.ur);
        assert(i_reduce <= jcp.reduce_loop_unroll);
        assert(jcp.reduce_loop_unroll == jcp.reduce_block);

        int offt = (jcp.ic_without_padding * i_ur * jcp.ngroups + i_reduce);

        return EVEX_compress_addr(
                aux_reg_bcast_data, jcp.typesize_in * offt, bcast);
    };

    auto load_ptr = [=](int i_reduce, int i_load) {
        int u0 = i_reduce % jcp.reduce_loop_unroll;
        int u1 = i_reduce / jcp.reduce_loop_unroll;

        int offt = (i_load * jcp.reduce_dim + u0) * jcp.load_block;

        return EVEX_compress_addr(aux_reg_load_data,
                u1 * jcp.reduce_loop_load_step + jcp.typesize_in * offt);
    };

    auto init = [=]() {
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                vpxord(r, r, r);
            }
        if (jcp.signed_input) {
            mov(reg_scratch, -128);
            vpbroadcastb(vmm_shift, reg_scratch.cvt8());
        }
    };

    auto store = [=](const bool mask_flag_in) {
        const auto &p = attr_.post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);
        const float *p_sum_scale = nullptr;
        const int32_t *p_sum_zp = nullptr;
        if (sum_idx != -1) {
            p_sum_scale = &p.entry_[sum_idx].sum.scale;
            p_sum_zp = &p.entry_[sum_idx].sum.zero_point;
        }
        const auto p_sum_scale_val = p_sum_scale ? *p_sum_scale : 1.f;
        const auto p_sum_zp_val = p_sum_zp ? *p_sum_zp : 0;
        const bool is_scale_or_zp_sum
                = p_sum_zp_val != 0 || p_sum_scale_val != 1.f;
        mov(EVEX_compress_addr(rsp, reg_bcast_data_off), reg_bcast_data);
        mov(reg_ptr_scales, EVEX_compress_addr(rsp, reg_ptr_sum_scale_off));
        if (is_scale_or_zp_sum) {
            mov(EVEX_compress_addr(rsp, reg_load_data_off), reg_load_data);
            if (p_sum_zp_val != 0) {
                mov(reg_load_data, p_sum_zp_val);
                mov(ptr[rsp + reg_ptr_sum_zp_off], reg_load_data);
            }
            if (p_sum_scale_val != 1.f)
                mov(reg_ptr_sum_scale, reinterpret_cast<size_t>(p_sum_scale));
        }
        if (jcp.signed_input && jcp.ver != ver_vnni) {
            mov(reg_scratch, float2int(jcp.wei_adj_scale));
            vmovq(xmm_bias_alpha(), reg_scratch);
            vbroadcastss(vmm_bias_alpha(), xmm_bias_alpha());
        }
        if (jcp.src_zero_point) {
            mov(reg_zp_compensation,
                    EVEX_compress_addr(rsp, reg_zp_compensation_off));
            mov(reg_src_zero_point,
                    EVEX_compress_addr(rsp, reg_src_zero_point_off));
        }
        for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
            const bool mask_flag = mask_flag_in && i_load == load_loop_blk - 1;
            auto vmm_bias = vmm_tmp;
            auto vmm_comp = vmm_bcast;
            if (jcp.with_bias) {
                if (jcp.signed_input || jcp.with_input_zp)
                    mov(reg_bias_data,
                            EVEX_compress_addr(rsp, reg_bias_data_off));
                cvt2ps(jcp.bia_dt, vmm_bias, bias_ptr(i_load), mask_flag);
                if (jcp.signed_input && jcp.ver != ver_vnni)
                    vmulps(vmm_bias, vmm_bias, vmm_bias_alpha());
            }
            if (jcp.signed_input || jcp.with_input_zp) {
                mov(reg_comp_data, EVEX_compress_addr(rsp, reg_comp_data_off));
                cvt2ps(data_type::s32, vmm_comp, comp_ptr(i_load), mask_flag);
            }
            if (jcp.src_zero_point) {
                // zero_point: conv(src_x8, wei_s8) - src_shift_s32 * compensation_s32
                const int zp_offset = sizeof(int32_t) * i_load * jcp.load_block;
                vmovups(vmm_zp,
                        EVEX_compress_addr(reg_zp_compensation, zp_offset));
                vpmulld(vmm_zp, vmm_zp,
                        EVEX_compress_addr(
                                reg_src_zero_point, 0, jcp.zp_src_is_common));
                // upscale to f32
                const Vmm vmm_
                        = mask_flag ? vmm_zp | k_load_dim_mask | T_z : vmm_zp;
                vcvtdq2ps(vmm_, vmm_);
            }
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                vcvtdq2ps(r, r);
                if (jcp.signed_input || jcp.with_input_zp) vaddps(r, r, vmm_comp);
                if (jcp.src_zero_point) vaddps(r, r, vmm_zp);
                if (jcp.with_bias) vaddps(r, r, vmm_bias);

                const Vmm mask_vmm = mask_flag ? r | k_load_dim_mask | T_z : r;
                vmulps(mask_vmm, r, scale_ptr(i_load));
            }
        }

        apply_postops(load_loop_blk, ur, mask_flag_in, p_sum_scale, p_sum_zp);

        if (jcp.dst_zero_point) {
            mov(reg_dst_zero_point,
                    EVEX_compress_addr(rsp, reg_dst_zero_point_off));
            vcvtdq2ps(vmm_zp, EVEX_compress_addr(reg_dst_zero_point, 0, true));

            /* Add dst zero_point to accumulator */
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    const auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                    vaddps(r, r, vmm_zp);
                }
            }
        }

        // Properly saturate the accumulators for integer datatypes
        if (one_of(jcp.dst_dt, u8, s8, s32)) {
            init_saturate_f32(vmm_zero, vmm_saturation,
                    reg_ptr_saturation_ubound, f32, jcp.dst_dt);
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                    saturate_f32(r, vmm_zero, vmm_saturation, jcp.dst_dt);
                    vcvtps2dq(r, r);
                }
            }
        }

        // store to the destination
        for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
            const bool mask_flag = mask_flag_in && i_load == load_loop_blk - 1;
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                const Vmm r_vmm = mask_flag ? r | k_load_dim_mask : r;

                switch (jcp.dst_dt) {
                    case data_type::f32:
                    case data_type::s32:
                        vmovups(output_ptr(i_load, i_ur), r_vmm);
                        break;
                    case data_type::s8:
                        vpmovsdb(output_ptr(i_load, i_ur), r_vmm);
                        break;
                    case data_type::u8:
                        vpmovusdb(output_ptr(i_load, i_ur), r_vmm);
                        break;
                    default: assert(!"unknown dst_dt");
                }
            }
        }
        mov(reg_bcast_data, EVEX_compress_addr(rsp, reg_bcast_data_off));
        if (is_scale_or_zp_sum)
            mov(reg_load_data, EVEX_compress_addr(rsp, reg_load_data_off));
    };

    auto compute = [=](Vmm vreg_acc, Vmm vreg_wei, Vmm vreg_src) {
        if (jcp.ver == ver_vnni) {
            vpdpbusd(vreg_acc, vreg_src, vreg_wei);
        } else {
            vpmaddubsw(vmm_tmp, vreg_src, vreg_wei);
            vpmaddwd(vmm_tmp, vmm_tmp, vmm_one);
            vpaddd(vreg_acc, vreg_acc, vmm_tmp);
        }
    };

    auto fma_block = [=](bool last_block) {
        int reduce_step = 4;
        int ic_tail_size = jcp.ic_without_padding % reduce_step;
        int loop_unroll = last_block && jcp.ic != jcp.ic_without_padding
                ? rnd_up(jcp.ic_without_padding % jcp.ic_block, reduce_step)
                : jcp.reduce_loop_unroll;
        for (int i_reduce = 0; i_reduce < loop_unroll;
                i_reduce += reduce_step) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load)
                vmovups(vreg_load(i_load), load_ptr(i_reduce, i_load));
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                if (last_block && ic_tail_size != 0
                        && i_reduce == loop_unroll - reduce_step) {
                    Xmm xmm_bcast = Xmm(vmm_bcast.getIdx());
                    load_bytes(xmm_bcast, aux_reg_bcast_data,
                            jcp.ic_without_padding * i_ur + i_reduce,
                            ic_tail_size);
                    vpbroadcastd(vmm_bcast, xmm_bcast);
                } else {
                    vpbroadcastd(vmm_bcast, bcast_ptr(i_reduce, i_ur, false));
                }
                if (jcp.signed_input) vpsubb(vmm_bcast, vmm_bcast, vmm_shift);
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    compute(vreg_accum(load_loop_blk, i_load, i_ur),
                            vreg_load(i_load), vmm_bcast);
                }
            }
        }
    };

    Label reduce_loop;
    Label reduce_loop_tail;

    push(reg_oc_off);

    mov(aux_reg_load_data, reg_load_data);

    mov(aux_reg_bcast_data, aux1_reg_bcast_data);
    init();

    mov(reduce_loop_iter, reg_reduce_loop_work);
    sub(reduce_loop_iter, jcp.reduce_loop_unroll);
    jle(reduce_loop_tail, T_NEAR);

    L(reduce_loop);
    {
        fma_block(false);
        add(aux_reg_bcast_data, jcp.reduce_loop_bcast_step);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        jg(reduce_loop, T_NEAR);
    }

    L(reduce_loop_tail);
    if (jcp.ic != jcp.ic_without_padding) {
        fma_block(true);
    } else {
        fma_block(false);
    }

    pop(reg_oc_off);

    if (jcp.oc_without_padding != jcp.oc) {
        Label end_store, common_store;
        mov(EVEX_compress_addr(rsp, reg_bcast_data_off), reg_bcast_data);

        /*Check if it is the last load_loop_blk*/
        sub(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
        cmp(reg_load_loop_work, 0);
        jg(common_store, T_NEAR);

        /*Check if it is the last ocb*/
        test(reg_reduce_pos_flag, FLAG_OC_LAST);
        jz(common_store, T_NEAR);

        store(true);
        jmp(end_store, T_NEAR);

        L(common_store);
        store(false);

        L(end_store);

        add(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
    } else {
        store(false);
    }
}

template <typename Vmm>
void _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::generate() {
    preamble();

    if (postops_injector_)
        postops_injector_->push_post_ops_data_on_stack(this->param1, GET_OFF(post_ops_binary_rhs_arg_vec), reg_load_data, reg_output_data);

    const int simd_w = jcp.ic_block;
    xor_(reg_scratch, reg_scratch);
    Reg16 _t = reg_scratch.cvt16();
    mov(_t, 0x1);
    vpbroadcastw(vmm_one, _t);

    sub(rsp, stack_space_needed);
    base_post_ops_data_offset += stack_space_needed;

    if (jcp.with_binary) {
        const auto zeroed_reg = r15;
        xor_(zeroed_reg, zeroed_reg);
        mov(EVEX_compress_addr(rsp, reg_binary_post_op_acc_off), zeroed_reg);
        mov(EVEX_compress_addr(rsp, reg_abi_param1_backup), abi_param1);
    }

    if (jcp.with_bias) mov(reg_bias_data, ptr[param1 + GET_OFF(bias_data)]);
    if (jcp.signed_input || jcp.with_input_zp) {
        mov(EVEX_compress_addr(rsp, reg_bias_data_off), reg_bias_data);
        mov(reg_comp_data, ptr[param1 + GET_OFF(compensation)]);
        mov(EVEX_compress_addr(rsp, reg_comp_data_off), reg_comp_data);
    }
    if (jcp.src_zero_point) {
        mov(reg_zp_compensation, ptr[param1 + GET_OFF(zp_compensation)]);
        mov(EVEX_compress_addr(rsp, reg_zp_compensation_off),
                reg_zp_compensation);
        mov(reg_src_zero_point, ptr[param1 + GET_OFF(src_zero_point)]);
        mov(EVEX_compress_addr(rsp, reg_src_zero_point_off),
                reg_src_zero_point);
    }
    if (jcp.dst_zero_point) {
        mov(reg_dst_zero_point, ptr[param1 + GET_OFF(dst_zero_point)]);
        mov(EVEX_compress_addr(rsp, reg_dst_zero_point_off),
                reg_dst_zero_point);
    }
    mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);
    mov(EVEX_compress_addr(rsp, reg_ptr_sum_scale_off), reg_ptr_scales);
    mov(reg_bcast_data, ptr[param1 + GET_OFF(bcast_data)]);
    mov(reg_load_data, ptr[param1 + GET_OFF(load_data)]);
    mov(reg_output_data, ptr[param1 + GET_OFF(output_data)]);

    mov(reg_load_loop_work, ptr[param1 + GET_OFF(load_dim)]);
    mov(reg_bcast_loop_work, ptr[param1 + GET_OFF(bcast_dim)]);
    mov(EVEX_compress_addr(rsp, bcast_loop_work_off), reg_bcast_loop_work);
    mov(reg_reduce_loop_work, ptr[param1 + GET_OFF(reduce_dim)]);
    mov(reg_reduce_pos_flag, ptr[param1 + GET_OFF(first_last_flag)]);
    mov(reg_oc_off, ptr[param1 + GET_OFF(oc_off)]);

    const int load_dim_tail
            = (one_of(jcp.prop_kind, forward_training, forward_inference)
                              ? jcp.oc_without_padding
                              : jcp.load_dim)
            % jcp.load_block;
    if (load_dim_tail) {
        Reg32 reg_tail_32 = reg_load_dim_tail_mask.cvt32();
        mov(reg_tail_32, (1 << load_dim_tail) - 1);
        kmovw(k_load_dim_tail_mask, reg_tail_32);
        kmovw(postops_mask, reg_tail_32);
    } else if (jcp.with_binary)
        if (jcp.oc_block != isa_simd_width_) {
            const int mask = (1 << jcp.oc_block) - 1;
            const Reg32 reg_tail_32 = reg_load_dim_tail_mask.cvt32();
            mov(reg_tail_32, mask);
            kmovw(postops_mask, reg_tail_32);
        }

    auto load_loop_body = [=](int load_loop_blk) {
        if (load_dim_tail) {
            kxnorw(k_load_dim_mask, k_load_dim_mask, k_load_dim_mask);
            Label no_update_mask;
            test(reg_reduce_pos_flag, FLAG_OC_LAST);
            jz(no_update_mask, T_NEAR);
            cmp(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
            jg(no_update_mask, T_NEAR);
            kmovw(k_load_dim_mask, k_load_dim_tail_mask);
            L(no_update_mask);
        }
        bcast_loop(load_loop_blk);
        add(reg_load_data, load_loop_blk * jcp.load_loop_load_step);
        if (jcp.with_bias) {
            if (jcp.signed_input || jcp.with_input_zp)
                mov(reg_bias_data, EVEX_compress_addr(rsp, reg_bias_data_off));
            add(reg_bias_data,
                    load_loop_blk * jcp.load_block * jcp.typesize_bia);
            if (jcp.signed_input || jcp.with_input_zp)
                mov(EVEX_compress_addr(rsp, reg_bias_data_off), reg_bias_data);
        }
        if (jcp.with_binary) {
            mov(reg_scratch,
                    EVEX_compress_addr(rsp, reg_binary_post_op_acc_off));
            add(reg_scratch, jcp.load_block * load_loop_blk);
            mov(EVEX_compress_addr(rsp, reg_binary_post_op_acc_off),
                    reg_scratch);
        }
        if (jcp.signed_input || jcp.with_input_zp) {
            mov(reg_comp_data, EVEX_compress_addr(rsp, reg_comp_data_off));
            add(reg_comp_data,
                    load_loop_blk * jcp.load_block * sizeof(int32_t));
            mov(EVEX_compress_addr(rsp, reg_comp_data_off), reg_comp_data);
        }
        if (jcp.src_zero_point) {
            mov(reg_zp_compensation,
                    EVEX_compress_addr(rsp, reg_zp_compensation_off));
            add(reg_zp_compensation,
                    load_loop_blk * jcp.load_block * sizeof(int32_t));
            mov(EVEX_compress_addr(rsp, reg_zp_compensation_off),
                    reg_zp_compensation);
        }
        mov(EVEX_compress_addr(rsp, reg_bcast_data_off), reg_bcast_data);
        mov(reg_ptr_scales, EVEX_compress_addr(rsp, reg_ptr_sum_scale_off));
        add(reg_ptr_scales,
                jcp.is_oc_scale * load_loop_blk * jcp.load_block
                        * sizeof(float));
        mov(EVEX_compress_addr(rsp, reg_ptr_sum_scale_off), reg_ptr_scales);
        mov(reg_bcast_data, EVEX_compress_addr(rsp, reg_bcast_data_off));
        add(reg_output_data, load_loop_blk * jcp.load_block * jcp.typesize_out);
        sub(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
        add(reg_oc_off, load_loop_blk * jcp.oc_block * sizeof(float));
    };

    Label load_loop_blk[7];

    static const int ur_cases_fma_expl_bcast[] = {2, 5, 6, 9, 14, 32};
    const int size_ur_cases_fma = sizeof(ur_cases_fma_expl_bcast);
    const int *ur_cases_fma = ur_cases_fma_expl_bcast;
    const int *ur_cases = ur_cases_fma;
    const int num_ur_cases = (size_ur_cases_fma) / sizeof(*ur_cases);

    for (int ur_idx = num_ur_cases - 1; ur_idx > 0; ur_idx--) {
        int label_idx = num_ur_cases - ur_idx - 1;
        if (jcp.ur <= ur_cases[ur_idx]) {
            cmp(reg_load_loop_work, simd_w * (label_idx + 1));
            jle(load_loop_blk[label_idx], T_NEAR);
        }
    }

    for (int ur_idx = 0; ur_idx < num_ur_cases; ur_idx++) {
        if (jcp.ur <= ur_cases[ur_idx]) {
            int label_idx = num_ur_cases - ur_idx - 1;
            L(load_loop_blk[label_idx]);
            {
                if (label_idx == 0) {
                    cmp(reg_load_loop_work, 0);
                    je(load_loop_blk[num_ur_cases], T_NEAR);
                }

                for (int _i = 1; _i <= label_idx + 1; _i++) {
                    prefetcht0(ptr[reg_load_data + _i * jcp.ic * jcp.oc_block]);
                    prefetcht1(ptr[reg_output_data + _i * jcp.oc_block]);
                }

                load_loop_body(label_idx + 1);
                if (label_idx - 1 > 0) {
                    cmp(reg_load_loop_work, 2 * label_idx * simd_w);
                    je(load_loop_blk[label_idx - 1], T_NEAR);
                }
                cmp(reg_load_loop_work, (label_idx + 1) * simd_w);
                jge(load_loop_blk[label_idx]);
            }
            for (int idx = label_idx - 1; idx > 0; --idx) {
                cmp(reg_load_loop_work, simd_w * (idx + 1));
                je(load_loop_blk[idx], T_NEAR);
            }
            if (ur_idx < num_ur_cases - 2) {
                cmp(reg_load_loop_work, simd_w);
                jle(load_loop_blk[0], T_NEAR);
            }
        }
    }
    L(load_loop_blk[num_ur_cases]);

    base_post_ops_data_offset -= stack_space_needed;
    add(rsp, stack_space_needed);

    if (postops_injector_)
        postops_injector_->reset_stack_pointer();

    postamble();

    if (jcp.with_eltwise) postops_injector_->prepare_table();
}

status_t jit_avx512_core_x8s8s32x_1x1_conv_kernel::init_conf(
        jit_1x1_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_t *&src_md, memory_desc_t &weights_md,
        memory_desc_t &dst_md, memory_desc_t &bias_md,
        const primitive_attr_t &attr, int nthreads, bool reduce_src) {

    if (!mayiuse(avx512_core)) return status::unimplemented;

    const memory_desc_wrapper src_d(src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    if (!one_of(src_d.data_type(), data_type::u8, data_type::s8)
            || weights_d.data_type() != data_type::s8
            || !one_of(dst_d.data_type(), data_type::f32, data_type::s32,
                    data_type::s8, data_type::u8))
        return status::unimplemented;

    jcp.nthr = nthreads;

    jcp.ver = ver_avx512_core;
    if (mayiuse(avx512_core_vnni)) jcp.ver = ver_vnni;

    int ndims = src_d.ndims();
    jcp.ndims = ndims;

    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;

    const bool is_1d = ndims == 3;
    const bool is_3d = ndims == 5;

    jcp.id = is_3d ? src_d.dims()[2] : 1;
    jcp.ih = is_1d ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = is_3d ? dst_d.dims()[2] : 1;
    jcp.oh = is_1d ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];

    jcp.kd = is_3d ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = is_1d ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = is_3d ? cd.padding[0][0] : 0;
    jcp.t_pad = is_1d ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = is_3d ? cd.strides[0] : 1;
    jcp.stride_h = is_1d ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;
    jcp.signed_input = (src_d.data_type() == data_type::s8);

    jcp.with_input_zp = !attr.input_zero_points_.has_default_values();
    jcp.with_weights_zp = !attr.weights_zero_points_.has_default_values();

    if (jcp.with_input_zp) {
        if (attr.input_zero_points_.count_ != 1 && attr.input_zero_points_.count_ != jcp.ic * jcp.ngroups)
            return status::unimplemented;

        if (attr.output_compensations_.count_ != jcp.oc * jcp.ngroups)
            return status::unimplemented;
    }

    if (jcp.with_weights_zp)
        return status::unimplemented;

    dim_t output_spatial = jcp.od * jcp.oh * jcp.ow;
    dim_t input_spatial = jcp.id * jcp.ih * jcp.iw;

    // FIXME: jcp.os and jcp.is fields have data type of int
    if (output_spatial > INT_MAX || input_spatial > INT_MAX)
        return status::unimplemented;

    jcp.os = output_spatial;
    jcp.is = input_spatial;

    const auto &post_ops = attr.post_ops_;
    const int dw_conv_ind = post_ops.find(primitive_kind::convolution);
    jcp.with_dw_conv = dw_conv_ind != -1;
    // Using dw_conv_ind as upper-bound below, as post-ops after it will be
    // handled in depthwise convolution.
    const int eltwise_ind
            = post_ops.find(primitive_kind::eltwise, 0, dw_conv_ind);
    jcp.with_eltwise = eltwise_ind != -1;

    const int binary_ind
            = post_ops.find(primitive_kind::binary, 0, dw_conv_ind);
    jcp.with_binary = binary_ind != -1;

    const int sum_ind = post_ops.find(primitive_kind::sum, 0, dw_conv_ind);
    jcp.with_sum = sum_ind != -1;
    if (jcp.with_sum)
        jcp.sum_dt = post_ops.entry_[sum_ind].sum.dt;

    jcp.with_depthwise = post_ops.find(primitive_kind::depthwise, 0, dw_conv_ind) != -1;
    jcp.with_quantization = post_ops.find(primitive_kind::quantization, 0, dw_conv_ind) != -1;

    if (dw_conv_ind >= 0) {
        // dw_conv and post_ops after it are handled externally, so skip them
        jcp.post_ops.entry_.assign(post_ops.entry_.cbegin(),
                post_ops.entry_.cbegin() + dw_conv_ind);
    } else {
        jcp.post_ops = post_ops;
    }

    const auto zp = attr.zero_points_;
    jcp.dst_zero_point = !zp.has_default_values(DNNL_ARG_DST);
    jcp.src_zero_point = !zp.has_default_values(DNNL_ARG_SRC);
    jcp.zp_src_is_common
            = zp.common(DNNL_ARG_SRC); // otherwise, it's per-channel
    assert(IMPLICATION(jcp.src_zero_point, jcp.zp_src_is_common));

    if ((jcp.dst_zero_point || jcp.src_zero_point) && jcp.with_dw_conv)
        return status::unimplemented;

    format_tag_t dat_tag = utils::pick(
            ndims - 3, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
    jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag);

    bool args_ok = jcp.src_tag == dat_tag && jcp.dst_tag == dat_tag;
    if (!args_ok) return status::unimplemented;

    if (jcp.ngroups == 1) {
        jcp.oc = rnd_up(jcp.oc, 16);
        jcp.ic = rnd_up(jcp.ic, 16);
    }

    using namespace injector;
    static constexpr bool sum_at_pos_0_only = false;
    static constexpr bool sum_requires_scale_one = false;
    static constexpr bool sum_requires_zp_zero = false;
    const bool post_ops_ok_ = post_ops_ok({avx512_core, {eltwise, binary, sum, depthwise, quantization},
            jcp.post_ops, &dst_d, sum_at_pos_0_only, sum_requires_scale_one,
            sum_requires_zp_zero});
    if (!post_ops_ok_) return status::unimplemented;

    const int simd_w = (jcp.ic % 16 == 0 && jcp.oc % 16 == 0)
            ? 16
            : (jcp.ic % 8 == 0 && jcp.oc % 8 == 0) ? 8 : 4;

    auto set_or_check_wei_format = [&]() -> bool {
        using namespace format_tag;
        using namespace memory_extra_flags;
        const format_tag_t wei_tags[3][2][3]
                = {{{OIw4i16o4i, OIhw4i16o4i, OIdhw4i16o4i},
                           {gOIw4i16o4i, gOIhw4i16o4i, gOIdhw4i16o4i}},
                        {{OIw2i8o4i, OIhw2i8o4i, OIdhw2i8o4i},
                                {gOIw2i8o4i, gOIhw2i8o4i, gOIdhw2i8o4i}},
                        {{OIw4o4i, OIhw4o4i, OIdhw4o4i},
                                {gOIw4o4i, gOIhw4o4i, gOIdhw4o4i}}};

        const int simd_idx = simd_w == 16 ? 0 : simd_w == 8 ? 1 : 2;
        const auto wei_tag = wei_tags[simd_idx][with_groups][ndims - 3];
        memory_desc_t want_wei_md = weights_md;
        memory_desc_init_by_tag(want_wei_md, wei_tag);
        if (jcp.signed_input) {
            want_wei_md.extra.flags = 0 | compensation_conv_s8s8;
            want_wei_md.extra.compensation_mask
                    = (1 << 0) + (with_groups ? (1 << 1) : 0);
            want_wei_md.extra.scale_adjust = 1.f;
        }
        if (jcp.src_zero_point) {
            want_wei_md.extra.flags |= compensation_conv_asymmetric_src;
            want_wei_md.extra.asymm_compensation_mask
                    = (1 << 0) + (with_groups ? (1 << 1) : 0);
        }

        if (weights_md.format_kind == format_kind::any) {
            weights_md = want_wei_md;
            return true;
        }

        return weights_md == want_wei_md;
    };

    if (!set_or_check_wei_format()) return status::unimplemented;

    args_ok = true && jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0
            && jcp.f_pad == 0 && jcp.t_pad == 0 && jcp.l_pad == 0
            && jcp.stride_d == 1 && jcp.stride_h == 1
            && jcp.stride_w == 1 // TODO: support some strides
            && jcp.od == jcp.id && jcp.oh == jcp.ih
            && jcp.ow == jcp.iw // enforce rpad = 0
            && jcp.kd == 1 && jcp.kh == 1 && jcp.kw == 1;
    if (!args_ok) return status::unimplemented;

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;
    jcp.sum_dt = post_ops.get_sum_dt(jcp.dst_dt);

    jcp.ic_block = jcp.oc_block = simd_w;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_bia
            = jcp.with_bias ? types::data_type_size(bias_d.data_type()) : 0;

    const int SMALL_SPATIAL = 7 * 7;
    const int BIG_REDUCE_DIM = 1024;

    int load_blocking = 0;
    int load_blocking_max = 0;
    int bcast_blocking = 0;
    int bcast_blocking_max = 0;
    int reduce_blocking = 0;
    int reduce_blocking_max = 0;
    jcp.load_grp_count = 1;
    jcp.use_vmovntps = false;

    const int L2_size
            = platform::get_per_core_cache_size(2) / sizeof(jcp.typesize_in);
    const int L2_capacity = (L2_size * 3) / 4;

    int size_treshold = 28;
    int max_regs = 0;
    int min_regs = 6;
    if (jcp.ver == ver_vnni)
        max_regs = ((jcp.oh > size_treshold && jcp.ow > size_treshold)
                           && (jcp.oc < 128 || jcp.ic < 128))
                ? min_regs
                : 9;
    else
        max_regs = 8;
    jcp.expl_bcast = true;

    if (jcp.mb == 1 && jcp.ic > 128
            && (jcp.oh <= size_treshold && jcp.ow <= size_treshold)) {
        if (jcp.os <= SMALL_SPATIAL && jcp.oc * jcp.ic < L2_size)
            max_regs = min_regs; // mobilenet_v2 performance improvement
        jcp.ur = nstl::min(max_regs, jcp.os);
    } else {
        const int spatial = jcp.od * jcp.oh;
        jcp.ur = 1;
        for (int ur_w = max_regs; ur_w >= min_regs; ur_w--) {
            if ((spatial >= size_treshold && spatial % ur_w == 0)
                    || (spatial < size_treshold && jcp.os % ur_w == 0)) {
                jcp.ur = ur_w;
                break;
            }
        }
        if (jcp.ur == 1) {
            jcp.ur = nstl::min(max_regs, jcp.os);
            int os_tail = jcp.os % max_regs;
            for (int i = max_regs; i >= min_regs; i--) {
                int i_tail = jcp.os % i;
                if (i_tail > os_tail || i_tail == 0) {
                    jcp.ur = i;
                    os_tail = i_tail;
                    if (i_tail == 0) break;
                }
            }
        }
    }
    if (jcp.with_dw_conv) jcp.ur = nstl::min(jcp.ow, jcp.ur);

    jcp.reduce_dim = jcp.ic;
    jcp.reduce_block = jcp.ic_block;

    jcp.load_dim = jcp.oc;
    jcp.load_block = jcp.oc_block;

    jcp.bcast_dim = jcp.is;

    jcp.bcast_block = jcp.ur;

    jcp.reduce_loop_unroll = jcp.reduce_block;
    jcp.reduce_loop_bcast_step = jcp.reduce_loop_unroll * jcp.typesize_in;

    jcp.reduce_loop_load_step
            = jcp.reduce_loop_unroll * jcp.load_block * jcp.typesize_in;

    jcp.bcast_loop_output_step
            = jcp.ur * jcp.ngroups * jcp.oc_without_padding * jcp.typesize_out;
    jcp.bcast_loop_output_substep = -1; // unused
    jcp.bcast_loop_bcast_step
            = jcp.ur * jcp.ngroups * jcp.ic_without_padding * jcp.typesize_in;
    jcp.bcast_loop_bcast_substep = -1; // unused

    jcp.load_loop_load_step = jcp.reduce_dim * jcp.load_block * jcp.typesize_in;

    jcp.load_loop_iter_step = jcp.load_block;

    jcp.loop_order = reduce_src ? loop_blr : loop_lbr;

    int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    reduce_blocking = nb_reduce;
    if (jcp.bcast_dim <= SMALL_SPATIAL && jcp.reduce_dim >= BIG_REDUCE_DIM)
        reduce_blocking = 64;
    else if (jcp.bcast_dim > SMALL_SPATIAL && jcp.reduce_dim >= BIG_REDUCE_DIM)
        reduce_blocking = 16;
    reduce_blocking = best_divider(nb_reduce, 1, reduce_blocking, true);
    reduce_blocking *= jcp.reduce_block;

    bool cmp_reduce = reduce_blocking <= jcp.reduce_dim;
    if (cmp_reduce) jcp.loop_order = reduce_src ? loop_rbl : loop_rlb;
    load_blocking = jcp.load_dim;

    jcp.load_grp_count = div_up(jcp.nthr, jcp.mb * jcp.ngroups * nb_bcast);
    jcp.load_grp_count = best_divider(
            jcp.nthr, jcp.load_grp_count, 2 * jcp.load_grp_count, false);

    if (jcp.bcast_dim <= SMALL_SPATIAL
            && jcp.load_dim * jcp.reduce_dim >= L2_size) {
        jcp.load_grp_count = nstl::max(jcp.load_grp_count, 4);
    } else if (jcp.bcast_dim <= SMALL_SPATIAL && jcp.mb <= jcp.nthr
            && jcp.load_dim > 512 && jcp.load_dim / jcp.reduce_dim >= 4) {
        jcp.load_grp_count = nstl::max(jcp.load_grp_count, 2); //
        load_blocking = jcp.load_block;
    }

    bcast_blocking = div_up(jcp.mb * jcp.ngroups * nb_bcast,
                             div_up(jcp.nthr, jcp.load_grp_count))
            * jcp.bcast_block;
    bcast_blocking = nstl::min(jcp.bcast_dim, bcast_blocking);
    bcast_blocking = rnd_up(bcast_blocking, jcp.bcast_block);

    int space_for_bcast = (L2_capacity - /* kernel_size - */
            2 * jcp.load_block * reduce_blocking - jcp.ur * reduce_blocking
            - 3 * 1024);
    if (jcp.reduce_dim * jcp.bcast_dim > L2_capacity) space_for_bcast /= 2;

    int bcast_in_cache
            = nstl::max(jcp.bcast_block, space_for_bcast / reduce_blocking);
    bcast_blocking = nstl::min(
            bcast_blocking, rnd_dn(bcast_in_cache, jcp.bcast_block));

    load_blocking_max = load_blocking;
    bcast_blocking_max = bcast_blocking * 3 / 2;
    reduce_blocking_max = reduce_blocking;

    assert(load_blocking);
    assert(load_blocking_max);
    assert(bcast_blocking);
    assert(bcast_blocking_max);
    assert(reduce_blocking);
    assert(reduce_blocking_max);
    assert(load_blocking % jcp.load_block == 0);
    assert(reduce_blocking % jcp.reduce_block == 0);
    assert(load_blocking_max % jcp.load_block == 0);
    assert(reduce_blocking_max % jcp.reduce_block == 0);

    assert(jcp.reduce_loop_unroll % 4 == 0);
    assert(jcp.reduce_dim % jcp.reduce_loop_unroll == 0);

    assert(jcp.bcast_block % jcp.ur == 0);
    assert(jcp.reduce_dim % jcp.reduce_block == 0);

    jcp.ur_tail = (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim) % jcp.ur;

    jcp.nb_bcast_blocking = bcast_blocking / jcp.bcast_block;
    jcp.nb_bcast_blocking_max = bcast_blocking_max / jcp.bcast_block;
    jcp.nb_load_blocking = load_blocking / jcp.load_block;
    jcp.nb_load_blocking_max = load_blocking_max / jcp.load_block;
    jcp.nb_reduce_blocking = reduce_blocking / jcp.reduce_block;
    jcp.nb_reduce_blocking_max = reduce_blocking_max / jcp.reduce_block;

    jcp.nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.load_dim, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    // miniumum size of load dim chunk for work distribution within threads
    jcp.nb_load_chunk = 1;
    // peformance improvements for googlenet_v3, mb=1;
    // TODO: generalize this condition and rewrite it in appropriate manner
    int ncores_per_socket = (int)cpu().getNumCores(
            Xbyak::util::IntelCpuTopologyLevel::CoreLevel);
    if (jcp.mb == 1 && jcp.nb_load % 4 == 0 && jcp.ic / jcp.oc >= 4
            && jcp.ic * jcp.oc <= L2_size && jcp.nthr <= ncores_per_socket) {
        jcp.nb_load_chunk = 4;
        jcp.load_grp_count = nstl::max(jcp.nb_load / 4, jcp.load_grp_count);
    }

    /* adjust the thread decomposition
     * to improve the perf for small size problem
     * the threshold 8192 is empirical
     * simply set the thread to max of nb_load and nb_bcast now
     * TODO: add get_thr_eff func to compute optimal thread
     * TODO: Threshold can be increase when init stride > 1 */
    auto bcast_size
            = (dim_t)jcp.mb * jcp.ngroups * jcp.bcast_dim * jcp.reduce_dim;
    if (jcp.typesize_in * bcast_size < 8192 && jcp.ngroups < jcp.nthr
            && jcp.nb_bcast * jcp.nb_load < jcp.nthr) {
        int nthr = nstl::max(jcp.nb_load, jcp.nb_bcast);
        jcp.nthr = nstl::min(jcp.nthr, nthr);
    }

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    // only common and per-oc-channel scales are supported
    const bool oscales_ok = one_of(oscales.mask_, 0, 1 << 1);
    if (!oscales_ok) return status::unimplemented;

    jcp.wei_adj_scale
            = (weights_d.extra().flags & memory_extra_flags::scale_adjust)
            ? weights_d.extra().scale_adjust
            : 1.f;

    return status::success;
}

void jit_avx512_core_x8s8s32x_1x1_conv_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace dnnl::impl::memory_tracking::names;

    if (jcp.signed_input && jcp.ver != ver_vnni) {
        dim_t count = nstl::max<dim_t>(
                attr.output_scales_.count_, (dim_t)jcp.ic_block);
        scratchpad.book<float>(key_conv_adjusted_scales, count);
    }
}

template struct _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Xbyak::Zmm>;
template struct _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Xbyak::Ymm>;
template struct _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Xbyak::Xmm>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
