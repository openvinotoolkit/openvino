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

#include <assert.h>

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
#include "cpu/x64/jit_uni_1x1_conv_utils.hpp"
#include "cpu/x64/jit_uni_x8s8s32x_1x1_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_1x1_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::data_type;
using namespace Xbyak;
using namespace injector_utils;

template <cpu_isa_t isa, typename Vmm>
_jit_uni_x8s8s32x_1x1_conv_kernel<isa, Vmm>::_jit_uni_x8s8s32x_1x1_conv_kernel(
        const jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr,
        const memory_desc_t &dst_md)
    : jit_generator(nullptr, MAX_CODE_SIZE, true, isa), jcp(ajcp), attr_(attr) {
    if (jcp.with_eltwise || jcp.with_binary || jcp.with_sum || jcp.with_depthwise || jcp.with_quantization) {
        using namespace binary_injector;
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = true;
        rhs_arg_static_params_t rhs_arg_static_params {15, r13, r14,
                preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec),
                memory_desc_wrapper(dst_md)};
        static_params_t static_params {this->param1, rhs_arg_static_params};
        quantization_injector::static_params_t quantization_static_params
                {vmm_d_weights.getIdx(), vmm_d_bias.getIdx(), reg_d_weights, reg_d_bias};

        postops_injector_
                = utils::make_unique<injector::jit_uni_postops_injector_t<isa>>(
                        this, jcp.post_ops, static_params, quantization_static_params);
    }
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_1x1_conv_kernel<isa, Vmm>::cvt2ps(data_type_t type_in,
        const Vmm &vmm_in, const Reg64 &reg, int offset, int load_size) {
    load_data(type_in, vmm_in, reg, offset, load_size);
    if (type_in != data_type::f32) uni_vcvtdq2ps(vmm_in, vmm_in);
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_1x1_conv_kernel<isa, Vmm>::bcast_loop(
        int load_loop_blk) {
    mov(aux1_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_bcast_data, reg_bcast_data);

    mov(aux_reg_output_data, reg_output_data);
    mov(reg_bcast_loop_iter, ptr[rsp + bcast_loop_work_off]);

    Label bcast_loop;
    Label bcast_loop_tail;

    cmp(reg_bcast_loop_iter, jcp.ur);
    jl(bcast_loop_tail, T_NEAR);

    L(bcast_loop);
    {
        assert(jcp.bcast_block == jcp.ur);
        reduce_loop(load_loop_blk, jcp.ur, 0, false);
        add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_step);
        add(aux_reg_output_data, jcp.bcast_loop_output_step);

        sub(reg_bcast_loop_iter, jcp.bcast_block);
        cmp(reg_bcast_loop_iter, jcp.bcast_block);
        jge(bcast_loop, T_NEAR);
    }

    L(bcast_loop_tail);
    if (jcp.ur_tail) {
        Label bcast_loop_tail_out;
        cmp(reg_bcast_loop_iter, 0);
        jz(bcast_loop_tail_out, T_NEAR);
        reduce_loop(load_loop_blk, jcp.ur_tail, 0, true);
        L(bcast_loop_tail_out);
    }
}

template <cpu_isa_t isa, typename Vmm>
int _jit_uni_x8s8s32x_1x1_conv_kernel<isa, Vmm>::output_ptr(
        const int i_load, const int i_ur) {
    const size_t ur_stride = jcp.with_dw_conv
            ? jcp.nb_load_blocking * jcp.oc_block * i_ur
            : jcp.oc_without_padding * i_ur;

    return jcp.typesize_out * (ur_stride + i_load * jcp.load_block);
};

template <cpu_isa_t isa, typename Vmm>
int _jit_uni_x8s8s32x_1x1_conv_kernel<isa, Vmm>::vreg_accum_idx(
        const int load_loop_blk, const int i_load, const int i_ur) {
    const int vmm_idx = i_ur * load_loop_blk + i_load;
    assert(vmm_idx < ker_max_reg_idx);
    return (15 - vmm_idx);
};

template <cpu_isa_t isa, typename Vmm>
Vmm _jit_uni_x8s8s32x_1x1_conv_kernel<isa, Vmm>::vreg_accum(
        const int load_loop_blk, const int i_load, const int i_ur) {
    return Vmm(vreg_accum_idx(load_loop_blk, i_load, i_ur));
};

template <typename F>
void iterate(const int ur, const int load_loop_blk, const F &f) {
    for (int i_ur = 0; i_ur < ur; ++i_ur)
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            f(i_ur, i_load);
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_1x1_conv_kernel<isa, Vmm>::apply_sum(const int ur,
        const int load_loop_blk, const bool mask_flag_in,
        const float *p_sum_scale, const int32_t *p_sum_zp) {

    if (jcp.with_sum) {
        assert(!utils::any_null(p_sum_scale, p_sum_zp)
                && "p_sum_scale or p_sum_zp = nullptr");
        const float sum_scale = *p_sum_scale;
        const int32_t sum_zp = *p_sum_zp;
        const auto sum_injector_lam = [this, mask_flag_in, load_loop_blk,
                                              sum_scale, sum_zp](const int i_ur,
                                              const int i_load) {
            const bool mask_flag = mask_flag_in && i_load == load_loop_blk - 1;
            const auto ymm_prev_dst = vmm_zero;

            const auto r = vreg_accum(load_loop_blk, i_load, i_ur);
            cvt2ps(jcp.sum_dt, ymm_prev_dst, aux_reg_output_data,
                    output_ptr(i_load, i_ur),
                    mask_flag ? get_tail_size() : simd_w);

            if (sum_zp != 0) {
                uni_vbroadcastss(vmm_tmp, ptr[reg_ptr_sum_zp]);
                uni_vcvtdq2ps(vmm_tmp, vmm_tmp);
                uni_vsubps(vmm_prev_dst, vmm_prev_dst, vmm_tmp);
            }
            if (sum_scale == 1.f)
                uni_vaddps(r, r, ymm_prev_dst);
            else {
                uni_vbroadcastss(vmm_tmp, ptr[reg_ptr_sum_scale]);
                uni_vfmadd231ps(r, ymm_prev_dst, vmm_tmp);
            }
        };
        const auto sum_injector
                = [=]() { iterate(ur, load_loop_blk, sum_injector_lam); };
        if (sum_zp != 0)
            mov(reg_ptr_sum_zp, reinterpret_cast<size_t>(p_sum_zp));
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);
    }
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_1x1_conv_kernel<isa, Vmm>::apply_postops(const int ur,
        const int load_loop_blk, const bool mask_flag_in,
        const float *p_sum_scale, const int32_t *p_sum_zp) {

    if (jcp.with_eltwise || jcp.with_binary || jcp.with_sum || jcp.with_depthwise || jcp.with_quantization) {
        std::map<size_t, int> vmm_idx_off;
        iterate(ur, load_loop_blk, [&](const int i_ur, const int i_load) {
            vmm_idx_off.insert({vreg_accum_idx(load_loop_blk, i_load, i_ur), i_load * jcp.load_block * sizeof(float)});
        });
        depthwise_injector::dynamic_params_t ddp {vmm_d_weights.getIdx(), vmm_d_bias.getIdx(), reg_d_weights, reg_d_bias,
                                                  reg_oc_off, vmm_idx_off,
                                                  this->rsp, base_post_ops_data_offset};
        quantization_injector::dynamic_params_t qdp {reg_oc_off, vmm_idx_off, jcp.dst_dt,
                                                     this->rsp, base_post_ops_data_offset};

        if (jcp.with_sum && *p_sum_zp != 0)
            mov(ptr[rsp + reg_bcast_loop_iter_off], reg_ptr_sum_zp);
        apply_sum(ur, load_loop_blk, mask_flag_in, p_sum_scale, p_sum_zp);

        binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
        vmm_index_set_t vmm_idxs;
        if (jcp.with_binary) {
            iterate(ur, load_loop_blk, [&](const int i_ur, const int i_load) {
                const int ur_stride = jcp.with_dw_conv
                        ? jcp.nb_load_blocking * jcp.oc_block * i_ur
                        : jcp.oc_without_padding * jcp.ngroups * i_ur;
                const int aux_output_offset
                        = (ur_stride + i_load * jcp.load_block);
                const auto vmm_idx
                        = vreg_accum_idx(load_loop_blk, i_load, i_ur);
                vmm_idxs.emplace(vmm_idx);
                rhs_arg_params.vmm_idx_to_oc_elem_off_addr.emplace(
                        vmm_idx, ptr[param1 + GET_OFF(oc_l_off)]);
                rhs_arg_params.vmm_idx_to_oc_elem_off_val.emplace(
                        vmm_idx, i_load * jcp.load_block);
                rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
                        vmm_idx, oc_off_oprnd);
                rhs_arg_params.vmm_idx_to_out_off_oprnd.emplace(
                        vmm_idx, out_off_oprnd);
                rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                        vmm_idx, aux_output_offset);
            });

            const injector_utils::register_preserve_guard_t register_guard(
                    this, {oc_off_oprnd, out_off_oprnd});
            mov(oc_off_oprnd,
                    ptr[rsp + reg_binary_post_op_acc_off
                            + register_guard.stack_space_occupied()]);
            mov(out_off_oprnd, aux_reg_output_data);
            sub(out_off_oprnd, ptr[param1 + GET_OFF(dst_orig)]);
            shr(out_off_oprnd, std::log2(types::data_type_size(jcp.dst_dt)));

            postops_injector_->compute_vector_range(vmm_idxs, rhs_arg_params, ddp, qdp);
        } else {
            iterate(ur, load_loop_blk, [&](const int i_ur, const int i_load) {
                vmm_idxs.emplace(vreg_accum_idx(load_loop_blk, i_load, i_ur));
            });
            postops_injector_->compute_vector_range(vmm_idxs, rhs_arg_params, ddp, qdp);
        }
        if (jcp.with_sum && *p_sum_zp != 0)
            mov(reg_ptr_sum_zp, ptr[rsp + reg_bcast_loop_iter_off]);
    }
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_1x1_conv_kernel<isa, Vmm>::reduce_loop(
        int load_loop_blk, int ur, int substep, bool wraparound) {

    // use 0x10001 to represent 2 words of 0x1
    // and avoid using uni_vpbroadcastb that is missing in jit generator
    const auto xmm_one = Xmm(vmm_one.getIdx());
    mov(reg_init_bcast, 0x10001);
    uni_vmovq(xmm_one, reg_init_bcast);
    uni_vpbroadcastd(vmm_one, xmm_one);

    auto vreg_load = [&](int i_load) {
        const int vmm_idx = ur * load_loop_blk + i_load;
        assert(vmm_idx < ker_max_reg_idx);
        /* remap the register indices to
         * avoid passing xmm0 to eltwise injector */
        return Vmm(15 - vmm_idx);
    };

    auto vmm_bias_alpha = [&]() {
        const int vmm_idx = ur * load_loop_blk;
        assert(vmm_idx < ker_max_reg_idx);
        return Vmm(15 - vmm_idx);
    };

    auto xmm_bias_alpha = [&]() { return Xmm(vmm_bias_alpha().getIdx()); };

    auto bcast_ptr = [&](int i_reduce, int i_ur) {
        assert(i_ur < jcp.ur);
        assert(i_reduce <= jcp.reduce_loop_unroll);
        assert(jcp.reduce_loop_unroll == jcp.reduce_block);

        int offt = (jcp.ic_without_padding * i_ur + i_reduce);

        return ptr[aux_reg_bcast_data + jcp.typesize_in * offt];
    };

    auto load_ptr = [&](int i_reduce, int i_load) {
        int u0 = i_reduce % jcp.reduce_loop_unroll;
        int u1 = i_reduce / jcp.reduce_loop_unroll;

        int offt = (i_load * jcp.reduce_dim + u0) * jcp.load_block;

        return ptr[aux_reg_load_data + u1 * jcp.reduce_loop_load_step
                + jcp.typesize_in * offt];
    };

    auto init = [&]() {
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                uni_vpxor(r, r, r);
            }
        if (jcp.signed_input) {
            // Used 0x80808080 to represents 2 words of 128
            // to avoid using uni_vpbroadcastb that is missing in jit generator
            auto xmm_shift = Xbyak::Xmm(vmm_shift.getIdx());
            auto _t32 = reg_init_bcast.cvt32();
            mov(_t32, 0x80808080);
            uni_vpinsrd(xmm_shift, xmm_shift, _t32, 0);
            uni_vpbroadcastd(vmm_shift, xmm_shift);
        }
    };

    auto store = [&](const bool mask_flag_in) {
        const auto &p = attr_.post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);
        const float *p_sum_scale
                = (sum_idx != -1) ? &p.entry_[sum_idx].sum.scale : nullptr;
        const int32_t *p_sum_zp
                = (sum_idx != -1) ? &p.entry_[sum_idx].sum.zero_point : nullptr;
        mov(ptr[rsp + reg_bcast_data_off], reg_bcast_data);
        mov(reg_ptr_scales, ptr[rsp + reg_ptr_sum_scale_off]);
        if (p_sum_scale && *p_sum_scale != 1.f) {
            mov(ptr[rsp + reg_load_data_off], reg_load_data);
            mov(reg_ptr_sum_scale, reinterpret_cast<size_t>(p_sum_scale));
        }
        if (jcp.signed_input && jcp.ver != ver_vnni) {
            mov(reg_store_bcast, float2int(jcp.wei_adj_scale));
            uni_vmovq(xmm_bias_alpha(), reg_store_bcast);
            uni_vbroadcastss(vmm_bias_alpha(), xmm_bias_alpha());
        }
        if (jcp.src_zero_point) {
            mov(reg_zp_compensation, ptr[rsp + reg_zp_compensation_off]);
            mov(reg_src_zero_point, ptr[rsp + reg_src_zero_point_off]);
        }
        for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
            if (jcp.src_zero_point) {
                uni_vpbroadcastd(vmm_zp, ptr[reg_src_zero_point]);
            }
            const bool mask_flag = mask_flag_in && i_load == load_loop_blk - 1;
            const int load_size = mask_flag ? get_tail_size() : simd_w;
            const auto ptr_scales_offset
                    = jcp.is_oc_scale * (sizeof(float) * jcp.oc_block * i_load);
            if (jcp.with_bias) {
                if (jcp.signed_input || jcp.with_input_zp)
                    mov(reg_bias_data, ptr[rsp + reg_bias_data_off]);
                cvt2ps(jcp.bia_dt, vmm_bias, reg_bias_data,
                        jcp.typesize_bia * jcp.oc_block * i_load, load_size);
                if (jcp.signed_input && jcp.ver != ver_vnni)
                    uni_vmulps(vmm_bias, vmm_bias, vmm_bias_alpha());
            }
            if (jcp.signed_input || jcp.with_input_zp) {
                mov(reg_comp_data, ptr[rsp + reg_comp_data_off]);
                cvt2ps(data_type::s32, vmm_comp, reg_comp_data,
                        sizeof(int32_t) * jcp.oc_block * i_load, load_size);
            }
            if (jcp.src_zero_point) {
                const int zp_offset = sizeof(int32_t) * i_load * jcp.oc_block;
                load_data(data_type::s32, vmm_zp_comp, reg_zp_compensation,
                        zp_offset, load_size);
                uni_vpmulld(vmm_zp_comp, vmm_zp_comp, vmm_zp);

                // upscale to f32
                uni_vcvtdq2ps(vmm_zp_comp, vmm_zp_comp);
            }

            if (mask_flag) {
                uni_vpxor(vmm_scale, vmm_scale, vmm_scale);
                cvt2ps(data_type::f32, vmm_scale, reg_ptr_scales,
                        ptr_scales_offset, get_tail_size());
            } else {
                uni_vmovups(vmm_scale, ptr[reg_ptr_scales + ptr_scales_offset]);
            }

            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                const auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                uni_vcvtdq2ps(r, r);
                if (jcp.signed_input || jcp.with_input_zp) uni_vaddps(r, r, vmm_comp);
                if (jcp.src_zero_point) uni_vaddps(r, r, vmm_zp_comp);
                if (jcp.with_bias) uni_vaddps(r, r, vmm_bias);

                uni_vmulps(r, r, vmm_scale);
            }
        }

        apply_postops(ur, load_loop_blk, mask_flag_in, p_sum_scale, p_sum_zp);

        if (jcp.dst_zero_point) {
            mov(reg_dst_zero_point, ptr[rsp + reg_dst_zero_point_off]);
            uni_vpbroadcastd(vmm_zp, ptr[reg_dst_zero_point]);
            uni_vcvtdq2ps(vmm_zp, vmm_zp);

            /* Add dst zero_point to accumulator */
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    const auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                    uni_vaddps(r, r, vmm_zp);
                }
            }
        }

        // Properly saturate the accumulators for integer datatypes
        if (utils::one_of(jcp.dst_dt, u8, s8, s32)) {
            init_saturate_f32(vmm_zero, vmm_saturation, aux_reg_saturation, f32,
                    jcp.dst_dt);

            for (int i_ur = 0; i_ur < ur; ++i_ur)
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                    saturate_f32(r, vmm_zero, vmm_saturation, jcp.dst_dt);
                    uni_vcvtps2dq(r, r);
                }
        }

        /* write out register to output_addr */
        for (int i_ur = 0; i_ur < ur; ++i_ur) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                const bool mask_flag
                        = mask_flag_in && i_load == load_loop_blk - 1;
                auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                store_data(jcp.dst_dt, r, aux_reg_output_data,
                        output_ptr(i_load, i_ur),
                        mask_flag ? get_tail_size() : simd_w);
            }
        }
        mov(reg_bcast_data, ptr[rsp + reg_bcast_data_off]);
        if (p_sum_scale && *p_sum_scale != 1.f)
            mov(reg_load_data, ptr[rsp + reg_load_data_off]);
    };

    auto compute = [&](Vmm vreg_acc, Vmm vreg_wei, Vmm vreg_src) {
        if (jcp.ver == ver_vnni) {
            vpdpbusd(vreg_acc, vreg_src, vreg_wei, VexEncoding);
        } else {
            uni_vpmaddubsw(vmm_tmp, vreg_src, vreg_wei);
            uni_vpmaddwd(vmm_tmp, vmm_tmp, vmm_one);
            uni_vpaddd(vreg_acc, vreg_acc, vmm_tmp);
        }
    };

    auto fma_block = [&](bool last_block) {
        int reduce_step = 4;
        int ic_tail_size = jcp.ic_without_padding % reduce_step;
        int loop_unroll = last_block && jcp.ic != jcp.ic_without_padding
                ? rnd_up(jcp.ic_without_padding % jcp.ic_block, reduce_step)
                : jcp.reduce_loop_unroll;
        for (int i_reduce = 0; i_reduce < loop_unroll;
                i_reduce += reduce_step) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load)
                uni_vmovups(vreg_load(i_load), load_ptr(i_reduce, i_load));
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                if (last_block && ic_tail_size != 0
                        && i_reduce == loop_unroll - reduce_step) {
                    load_bytes(vmm_bcast, aux_reg_bcast_data,
                            jcp.ic_without_padding * i_ur + i_reduce,
                            ic_tail_size);
                    uni_vpbroadcastd(vmm_bcast, Xmm(vmm_bcast.getIdx()));
                } else {
                    uni_vpbroadcastd(vmm_bcast, bcast_ptr(i_reduce, i_ur));
                }
                if (jcp.signed_input)
                    uni_vpsubb(vmm_bcast, vmm_bcast, vmm_shift);
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

    mov(reg_reduce_loop_iter, reg_reduce_loop_work);
    sub(reg_reduce_loop_iter, jcp.reduce_loop_unroll);
    jle(reduce_loop_tail, T_NEAR);

    L(reduce_loop);
    {
        fma_block(false);
        add(aux_reg_bcast_data, jcp.reduce_loop_bcast_step);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reg_reduce_loop_iter, jcp.reduce_loop_unroll);
        jg(reduce_loop, T_NEAR);
    }

    L(reduce_loop_tail);
    fma_block(jcp.ic != jcp.ic_without_padding);

    pop(reg_oc_off);

    if (jcp.oc_without_padding != jcp.oc) {
        Label end_store, common_store;
        mov(ptr[rsp + reg_bcast_data_off], reg_bcast_data);

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

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_1x1_conv_kernel<isa, Vmm>::generate() {
    preamble();

    if (postops_injector_)
        postops_injector_->push_post_ops_data_on_stack(this->param1, GET_OFF(post_ops_binary_rhs_arg_vec), reg_load_data, reg_output_data);

    sub(rsp, stack_space_needed);
    base_post_ops_data_offset += stack_space_needed;

    if (jcp.with_binary) {
        // zero initialize binary post_ops offset accumulator (store on stack)
        const auto binary_post_op_acc_off_reg = r15;
        xor_(binary_post_op_acc_off_reg, binary_post_op_acc_off_reg);
        mov(ptr[rsp + reg_binary_post_op_acc_off], binary_post_op_acc_off_reg);
    }

    if (jcp.with_bias) mov(reg_bias_data, ptr[param1 + GET_OFF(bias_data)]);
    if (jcp.signed_input || jcp.with_input_zp) {
        mov(ptr[rsp + reg_bias_data_off], reg_bias_data);
        mov(reg_comp_data, ptr[param1 + GET_OFF(compensation)]);
        mov(ptr[rsp + reg_comp_data_off], reg_comp_data);
    }
    if (jcp.src_zero_point) {
        mov(reg_zp_compensation, ptr[param1 + GET_OFF(zp_compensation)]);
        mov(ptr[rsp + reg_zp_compensation_off], reg_zp_compensation);
        mov(reg_src_zero_point, ptr[param1 + GET_OFF(src_zero_point)]);
        mov(ptr[rsp + reg_src_zero_point_off], reg_src_zero_point);
    }
    if (jcp.dst_zero_point) {
        mov(reg_dst_zero_point, ptr[param1 + GET_OFF(dst_zero_point)]);
        mov(ptr[rsp + reg_dst_zero_point_off], reg_dst_zero_point);
    }
    mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);
    mov(ptr[rsp + reg_ptr_sum_scale_off], reg_ptr_scales);
    mov(reg_bcast_data, ptr[param1 + GET_OFF(bcast_data)]);
    mov(reg_load_data, ptr[param1 + GET_OFF(load_data)]);
    mov(reg_output_data, ptr[param1 + GET_OFF(output_data)]);

    mov(reg_load_loop_work, ptr[param1 + GET_OFF(load_dim)]);
    mov(reg_bcast_loop_work, ptr[param1 + GET_OFF(bcast_dim)]);
    mov(ptr[rsp + bcast_loop_work_off], reg_bcast_loop_work);
    mov(reg_reduce_loop_work, ptr[param1 + GET_OFF(reduce_dim)]);
    mov(reg_reduce_pos_flag, ptr[param1 + GET_OFF(first_last_flag)]);
    mov(reg_oc_off, ptr[param1 + GET_OFF(oc_off)]);

    auto load_loop_body = [&](int load_loop_blk) {
        bcast_loop(load_loop_blk);
        add(reg_load_data, load_loop_blk * jcp.load_loop_load_step);
        if (jcp.with_bias) {
            if (jcp.signed_input || jcp.with_input_zp)
                mov(reg_bias_data, ptr[rsp + reg_bias_data_off]);
            add(reg_bias_data,
                    load_loop_blk * jcp.load_block * jcp.typesize_bia);
            if (jcp.signed_input || jcp.with_input_zp)
                mov(ptr[rsp + reg_bias_data_off], reg_bias_data);
        }
        if (jcp.with_binary) {
            mov(aux_reg_load_data,
                    EVEX_compress_addr(rsp, reg_binary_post_op_acc_off));
            add(aux_reg_load_data, jcp.load_block * load_loop_blk);
            mov(EVEX_compress_addr(rsp, reg_binary_post_op_acc_off),
                    aux_reg_load_data);
        }
        if (jcp.signed_input || jcp.with_input_zp) {
            mov(reg_comp_data, ptr[rsp + reg_comp_data_off]);
            add(reg_comp_data,
                    load_loop_blk * jcp.load_block * sizeof(int32_t));
            mov(ptr[rsp + reg_comp_data_off], reg_comp_data);
        }
        if (jcp.src_zero_point) {
            mov(reg_zp_compensation, ptr[rsp + reg_zp_compensation_off]);
            add(reg_zp_compensation,
                    load_loop_blk * jcp.load_block * sizeof(int32_t));
            mov(ptr[rsp + reg_zp_compensation_off], reg_zp_compensation);
        }
        mov(ptr[rsp + reg_bcast_data_off], reg_bcast_data);
        mov(reg_ptr_scales, ptr[rsp + reg_ptr_sum_scale_off]);
        add(reg_ptr_scales,
                jcp.is_oc_scale * load_loop_blk * jcp.load_block
                        * sizeof(float));
        mov(ptr[rsp + reg_ptr_sum_scale_off], reg_ptr_scales);
        mov(reg_bcast_data, ptr[rsp + reg_bcast_data_off]);
        add(reg_output_data, load_loop_blk * jcp.load_block * jcp.typesize_out);
        sub(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
        add(reg_oc_off, load_loop_blk * jcp.oc_block * sizeof(float));
    };

    static const int ur_cases[] = {2, 3, 5, 12};
    constexpr int num_ur_cases = sizeof(ur_cases) / sizeof(*ur_cases);
    Label load_loop_blk[num_ur_cases + 1];

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

template <cpu_isa_t isa>
status_t jit_uni_x8s8s32x_1x1_conv_kernel<isa>::init_conf(
        jit_1x1_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const memory_desc_wrapper &bias_d,
        primitive_attr_t &attr, int nthreads, bool reduce_src) {
    if (!mayiuse(isa)) return status::unimplemented;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    if (!one_of(src_d.data_type(), data_type::u8, data_type::s8)
            || weights_d.data_type() != data_type::s8
            || !one_of(dst_d.data_type(), data_type::f32, data_type::s32,
                    data_type::s8, data_type::u8))
        return status::unimplemented;

    const int ndims = src_d.ndims();
    jcp.nthr = nthreads;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
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

    jcp.os = jcp.od * jcp.oh * jcp.ow;
    jcp.is = jcp.id * jcp.ih * jcp.iw;

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

    jcp.with_depthwise = post_ops.find(primitive_kind::depthwise, 0, dw_conv_ind) != -1;
    jcp.with_quantization = post_ops.find(primitive_kind::quantization, 0, dw_conv_ind) != -1;

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

    bool args_ok = true && jcp.ngroups == 1 && jcp.src_tag == dat_tag
            && jcp.dst_tag == dat_tag;
    if (!args_ok) return status::unimplemented;

    jcp.ver = mayiuse(avx2_vnni) ? ver_vnni : ver_unused;

    jcp.oc = rnd_up(jcp.oc, simd_w);
    jcp.ic = rnd_up(jcp.ic, simd_w);

    if (dw_conv_ind >= 0) {
        // dw_conv and post_ops after it are handled externally, so skip them
        jcp.post_ops.entry_.assign(post_ops.entry_.cbegin(),
                post_ops.entry_.cbegin() + dw_conv_ind);
    } else {
        jcp.post_ops = post_ops;
    }

    for (auto &post_op : jcp.post_ops.entry_)
        if (post_op.is_binary() && post_op.binary.src1_desc.dims[1] != 1) {
            post_op.binary.src1_desc.dims[1] = jcp.oc;
        }

    using namespace injector;
    const bool post_ops_ok_ = post_ops_ok({isa, {eltwise, binary, sum, depthwise, quantization},
            jcp.post_ops, &dst_d, false, false, false});
    if (!post_ops_ok_) return status::unimplemented;

    args_ok = true && jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0
            && jcp.f_pad == 0 && jcp.t_pad == 0 && jcp.l_pad == 0
            && jcp.stride_w == 1 && jcp.stride_h == 1 && jcp.stride_d == 1
            && jcp.ow == jcp.iw && jcp.oh == jcp.ih && jcp.od == jcp.id
            && jcp.kd == 1 && jcp.kh == 1 && jcp.kw == 1;
    if (!args_ok) return status::unimplemented;

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;
    if (jcp.with_sum)
        jcp.sum_dt = post_ops.get_sum_dt(jcp.dst_dt);

    jcp.ic_block = jcp.oc_block = simd_w;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_bia
            = jcp.with_bias ? types::data_type_size(bias_d.data_type()) : 0;

    const int SMALL_SPATIAL = 7 * 7;
    const int BIG_REDUCE_DIM = 512;

    int load_blocking = 0;
    int load_blocking_max = 0;
    int bcast_blocking = 0;
    int bcast_blocking_max = 0;
    int reduce_blocking = 0;
    int reduce_blocking_max = 0;
    jcp.load_grp_count = 1;

    const int L2_size
            = platform::get_per_core_cache_size(2) / sizeof(jcp.typesize_in);
    const int L2_capacity = (L2_size * 3) / 4;

    int size_threshold = 28;

    int min_regs = 3;
    int max_regs = 5;

    if (jcp.mb == 1 && jcp.ic > 128
            && (jcp.oh <= size_threshold && jcp.ow <= size_threshold)) {
        if (jcp.os <= SMALL_SPATIAL && jcp.oc * jcp.ic < L2_size)
            max_regs = min_regs = 3;
        jcp.ur = nstl::min(max_regs, jcp.os);
    } else {
        const int spatial = jcp.od * jcp.oh;
        jcp.ur = 1;
        for (int ur_w = max_regs; ur_w >= min_regs; ur_w--) {
            if ((spatial >= size_threshold && spatial % ur_w == 0)
                    || (spatial < size_threshold && jcp.os % ur_w == 0)) {
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
            = jcp.ur * jcp.oc_without_padding * jcp.typesize_out;
    jcp.bcast_loop_bcast_step
            = jcp.ur * jcp.ic_without_padding * jcp.typesize_in;

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
            && jcp.load_dim > 256 && jcp.load_dim / jcp.reduce_dim >= 4) {
        jcp.load_grp_count = nstl::max(jcp.load_grp_count, 2);
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

    const bool params_ok = true && load_blocking > 0 && load_blocking_max > 0
            && bcast_blocking > 0 && bcast_blocking_max > 0
            && reduce_blocking > 0 && reduce_blocking_max > 0
            && load_blocking % jcp.load_block == 0
            && reduce_blocking % jcp.reduce_block == 0
            && load_blocking_max % jcp.load_block == 0
            && reduce_blocking_max % jcp.reduce_block == 0
            && jcp.reduce_loop_unroll % 4 == 0
            && jcp.reduce_dim % jcp.reduce_loop_unroll == 0
            && jcp.bcast_block % jcp.ur == 0
            && jcp.reduce_dim % jcp.reduce_block == 0;

    assert(params_ok && "parameter values are inconsistent");
    if (!params_ok) return status::unimplemented;

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

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_1x1_conv_kernel<isa>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace dnnl::impl::memory_tracking::names;

    if (jcp.signed_input && jcp.ver != ver_vnni) {
        dim_t count = nstl::max<dim_t>(attr.output_scales_.count_, 8);
        scratchpad.book<float>(key_conv_adjusted_scales, count);
    }
}

template struct _jit_uni_x8s8s32x_1x1_conv_kernel<avx2, Ymm>;
template struct _jit_uni_x8s8s32x_1x1_conv_kernel<sse41, Xmm>;
template struct jit_uni_x8s8s32x_1x1_conv_kernel<avx2>;
template struct jit_uni_x8s8s32x_1x1_conv_kernel<sse41>;
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
