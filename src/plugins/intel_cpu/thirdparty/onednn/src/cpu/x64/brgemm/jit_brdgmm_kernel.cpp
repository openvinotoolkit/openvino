/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/brgemm/brgemm_types.hpp"
#include "cpu/x64/brgemm/jit_brdgmm_kernel.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"

#define GET_OFF(field) offsetof(brgemm_kernel_params_t, field)
#define GET_OFF_BATCH_ELEMENT(field) offsetof(brgemm_batch_element_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace Xbyak;

jit_brdgmm_kernel_base_t::jit_brdgmm_kernel_base_t(const brgemm_t &abrd)
    : jit_generator(nullptr, MAX_CODE_SIZE, true, avx512_common), brg(abrd) {

    if (brg.with_eltwise || brg.with_binary || brg.with_sum) {

        static constexpr bool preserve_gpr = false;
        static constexpr bool preserve_vmm = false;
        static constexpr bool use_exact_tail_scalar_bcast = false;
        const auto dst_md_wrapper = memory_desc_wrapper(brg.dst_md);

        static const bcast_set_t enabled_bcast_strategy
                = {broadcasting_strategy_t::scalar,
                        broadcasting_strategy_t::per_oc};
        const binary_injector::rhs_arg_static_params_t rhs_sp {
                static_cast<size_t>(vmm_b().getIdx()), reg_binary_rhs, reg_tmp,
                preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec), dst_md_wrapper,
                static_cast<size_t>(n_vlen_tail()), k_mask,
                use_exact_tail_scalar_bcast};
        const binary_injector::static_params_t bsp {
                this->param1, enabled_bcast_strategy, rhs_sp};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<avx512_core>>(
                this, brg.attr->post_ops_, bsp);

        using namespace dnnl::impl::cpu::binary_injector_utils;
        std::tie(with_binary_per_oc_bcast_)
                = bcast_strategies_present_tup(brg.attr->post_ops_.entry_,
                        dst_md_wrapper, broadcasting_strategy_t::per_oc);
        handle_binary_po_offset_ = with_binary_per_oc_bcast_;
    }
}

jit_brdgmm_kernel_base_t::Vmm jit_brdgmm_kernel_base_t::vmm_mask(
        const Vmm vmm_in, bool mask_flag, bool store) {
    return mask_flag ? (store ? vmm_in | k_mask : vmm_in | k_mask | T_z)
                     : vmm_in;
}

jit_brdgmm_kernel_base_t::Wmm jit_brdgmm_kernel_base_t::wmm_mask(
        const Wmm wmm_in, bool mask_flag, bool store) {
    return mask_flag ? (store ? wmm_in | k_mask : wmm_in | k_mask | T_z)
                     : wmm_in;
}

void jit_brdgmm_kernel_base_t::read_params() {
    Label label_done;

    mov(reg_BS, ptr[param1 + GET_OFF(BS)]);
    mov(reg_aux_C, ptr[param1 + GET_OFF(ptr_C)]);
    mov(reg_aux_D, ptr[param1 + GET_OFF(ptr_D)]);

    if (brg.type == brgemm_offs) {
        mov(reg_A, ptr[param1 + GET_OFF(ptr_A)]);
        mov(reg_B, ptr[param1 + GET_OFF(ptr_B)]);
    } else if (brg.type == brgemm_strd) {
        mov(reg_aux1_A, ptr[param1 + GET_OFF(ptr_A)]);
        mov(reg_aux1_B, ptr[param1 + GET_OFF(ptr_B)]);
        if (brg.brgattr.max_bs > 1) {
            mov(ptr[rsp + reg_A_offs_], reg_aux1_A);
            mov(ptr[rsp + reg_B_offs_], reg_aux1_B);
        }
    }

    if (one_of(brg.type, brgemm_addr, brgemm_offs) || has_vpad()) {
        mov(reg_aux_batch_addr, ptr[param1 + GET_OFF(batch)]);
        if (brg.brgattr.max_bs > 1)
            mov(ptr[rsp + reg_batch0_addr_offs_], reg_aux_batch_addr);
    }

    if (brg.with_bias) {
        mov(reg_tmp, ptr[param1 + GET_OFF(ptr_bias)]);
        mov(ptr[rsp + reg_bias_offs_], reg_tmp);
    }

    if (brg.with_scales) {
        mov(reg_tmp, ptr[param1 + GET_OFF(ptr_scales)]);
        mov(ptr[rsp + reg_scales_offs_], reg_tmp);
    }

    if (brg.with_binary) mov(ptr[rsp + abi_param1_offs_], param1);

    if (with_binary_per_oc_bcast_) {
        mov(reg_tmp, ptr[param1 + GET_OFF(oc_logical_off)]);
        mov(ptr[rsp + reg_binary_postops_oc_l_offs_], reg_tmp);
    }
}

void jit_brdgmm_kernel_base_t::load_accumulators(int m_blocks, int n_blocks) {
    for_(int m = 0; m < m_blocks; ++m)
    for (int n = 0; n < n_blocks; ++n) {
        auto vmm = accm(m_blocks, n_blocks, m, n);
        vxorps(vmm, vmm, vmm);
    }
}

void jit_brdgmm_kernel_base_t::restore_A_B_matrices() {
    if (brg.brgattr.max_bs > 1
            && (one_of(brg.type, brgemm_addr, brgemm_offs) || has_vpad()))
        mov(reg_aux_batch_addr, ptr[rsp + reg_batch0_addr_offs_]);

    if (brg.type == brgemm_strd && brg.brgattr.max_bs > 1) {
        mov(reg_aux1_A, ptr[rsp + reg_A_offs_]);
        mov(reg_aux1_B, ptr[rsp + reg_B_offs_]);
    }
}

void jit_brdgmm_kernel_base_t::set_A_B_matrices() {

    if (brg.type == brgemm_addr) {
        mov(reg_aux_A, ptr[reg_aux_batch_addr + GET_OFF_BATCH_ELEMENT(ptr.A)]);
        mov(reg_aux_B, ptr[reg_aux_batch_addr + GET_OFF_BATCH_ELEMENT(ptr.B)]);
    } else if (brg.type == brgemm_offs) {
        mov(reg_aux_A, reg_A);
        mov(reg_aux_B, reg_B);
        add(reg_aux_A,
                ptr[reg_aux_batch_addr + GET_OFF_BATCH_ELEMENT(offset.A)]);
        add(reg_aux_B,
                ptr[reg_aux_batch_addr + GET_OFF_BATCH_ELEMENT(offset.B)]);
    } else if (brg.type == brgemm_strd) {
        mov(reg_aux_A, reg_aux1_A);
        mov(reg_aux_B, reg_aux1_B);
        if (brg.brgattr.max_bs > 1) {
            add(reg_aux1_A, brg.stride_a);
            add(reg_aux1_B, brg.stride_b);
        }
    }

    add(reg_aux_A, reg_a_offset);
    lea(reg_aux_B, ptr[reg_aux_B + reg_aux_N * brg.typesize_B]);
}

void jit_brdgmm_kernel_base_t::advance_A_B_matrices() {
    if (brg.brgattr.max_bs > 1
            && (one_of(brg.type, brgemm_addr, brgemm_offs) || has_vpad()))
        add(reg_aux_batch_addr, sizeof(brgemm_batch_element_t));
}

void jit_brdgmm_kernel_base_t::cvt2ps(data_type_t type_in, const Vmm vmm_in,
        const Xbyak::Operand &op, bool mask_flag, bool store) {
    const Vmm vmm = vmm_mask(vmm_in, mask_flag, store);
    switch (type_in) {
        case data_type::f32:
        case data_type::s32: vmovups(vmm, op); break;
        case data_type::bf16:
            vpmovzxwd(vmm, op);
            vpslld(vmm, vmm, 16);
            break;
        case data_type::s8: vpmovsxbd(vmm, op); break;
        case data_type::u8: vpmovzxbd(vmm, op); break;
        default: assert(!"unsupported data type");
    }
    if (!one_of(type_in, data_type::f32, data_type::bf16))
        vcvtdq2ps(vmm_in, vmm_in);
}

void jit_brdgmm_kernel_base_t::apply_post_ops(
        int m_blocks, int n_blocks, bool has_n_tail) {

    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;

    if (brg.with_binary) {
        mov(reg_binary_params, ptr[rsp + abi_param1_offs_]);

        if (handle_binary_po_offset_) {
            mov(reg_binary_po_stack_frame, rsp);

            for_(int m_i = 0; m_i < m_blocks; m_i++)
            for (int n_i = 0; n_i < n_blocks; n_i++) {
                const auto vmm_idx
                        = accm(m_blocks, n_blocks, m_i, n_i).getIdx();
                rhs_arg_params.vmm_idx_to_oc_elem_off_addr.emplace(vmm_idx,
                        ptr[reg_binary_po_stack_frame
                                + reg_binary_postops_oc_l_offs_]);
                if (with_binary_per_oc_bcast_)
                    rhs_arg_params.vmm_idx_to_oc_elem_off_val.emplace(
                            vmm_idx, oc_logical_offset(n_i));
                if (n_i + 1 == n_blocks && has_n_tail)
                    rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
            }
        }
    }

    const auto sum_injector = [&] {
        const float *p_sum_scale = &brg.sum_scale;
        const int32_t *p_sum_zp = &brg.sum_zp;
        const bool p_sum_scale_reg_set = *p_sum_scale != 1.f;
        const bool p_sum_zp_reg_set = *p_sum_zp != 0;

        const injector_utils::conditional_register_preserve_guard_t
                register_guard_sum_scale(
                        (handle_binary_po_offset_) && p_sum_scale_reg_set, this,
                        {reg_ptr_sum_scale});
        const injector_utils::conditional_register_preserve_guard_t
                register_guard_sum_zp(p_sum_zp_reg_set, this, {reg_ptr_sum_zp});

        if (p_sum_scale_reg_set)
            mov(reg_ptr_sum_scale, reinterpret_cast<size_t>(p_sum_scale));

        auto vmm_sum_zp = vmm_tmp(0);
        if (p_sum_zp_reg_set) {
            mov(reg_ptr_sum_zp, reinterpret_cast<size_t>(p_sum_zp));
            vcvtdq2ps(vmm_sum_zp, ptr_b[reg_ptr_sum_zp]);
        }

        for (int m_i = 0; m_i < m_blocks; m_i++) {
            for (int n_i = 0; n_i < n_blocks; n_i++) {
                const auto vmm = accm(m_blocks, n_blocks, m_i, n_i);
                const auto addr = ptr[reg_aux_D + D_offset(m_i, n_i)];
                const auto vmm_prev_dst = vmm_tmp(1);
                const bool mask_flag = has_n_tail && (n_i + 1 == n_blocks);
                cvt2ps(brg.sum_dt, vmm_prev_dst, addr, mask_flag, false);
                if (p_sum_zp_reg_set) vsubps(vmm_prev_dst, vmm_sum_zp);
                if (!p_sum_scale_reg_set)
                    vaddps(vmm, vmm_prev_dst);
                else
                    vfmadd231ps(vmm, vmm_prev_dst, zword_b[reg_ptr_sum_scale]);
            }
        }
    };

    if (brg.with_sum) {
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);
    }

    postops_injector_->compute_vector_range(
            32 - m_blocks * n_blocks, 32, rhs_arg_params);
}

void jit_brdgmm_kernel_base_t::store_accumulators_apply_post_ops(
        int m_blocks, int n_blocks, bool has_n_tail) {

    if (brg.with_bias) {
        mov(reg_aux_bias, ptr[rsp + reg_bias_offs_]);
        lea(reg_aux_bias, ptr[reg_aux_bias + reg_aux_N * brg.typesize_bias]);
    }

    const bool dq2ps_required = brg.is_int8;
    for (int n = 0; n < n_blocks; n++) {
        auto vmm_bias = vmm_tmp(0);
        if (brg.with_bias) {
            const bool mask_flag = has_n_tail && n + 1 == n_blocks;
            auto ptr_bias = ptr[reg_aux_bias + bias_offset(n)];
            cvt2ps(brg.dt_bias, vmm_bias, ptr_bias, mask_flag, false);
        }
        for (int m = 0; m < m_blocks; m++) {
            auto vmm = accm(m_blocks, n_blocks, m, n);
            if (dq2ps_required) vcvtdq2ps(vmm, vmm);
            if (brg.with_bias) { vaddps(vmm, vmm, vmm_bias); }
        }
    }

    if (brg.with_scales) {
        mov(reg_aux_scales, ptr[rsp + reg_scales_offs_]);
        if (brg.is_oc_scale) {
            lea(reg_aux_scales,
                    ptr[reg_aux_scales + reg_aux_N * sizeof(float)]);
        }
        for (int m = 0; m < m_blocks; m++) {
            for (int n = 0; n < n_blocks; n++) {
                const bool mask_flag = has_n_tail && n + 1 == n_blocks;
                const Vmm vmm = vmm_mask(
                        accm(m_blocks, n_blocks, m, n), mask_flag, false);
                if (brg.is_oc_scale) {
                    vmulps(vmm, vmm, ptr[reg_aux_scales + scales_offset(n)]);
                } else {
                    vmulps(vmm, vmm, zword_b[reg_aux_scales]);
                }
            }
        }
    }

    if (postops_injector_) apply_post_ops(m_blocks, n_blocks, has_n_tail);

    const bool dt_requires_saturation
            = one_of(brg.dt_d, data_type::u8, data_type::s8, data_type::s32);
    auto vmm_lbound = vmm_tmp(0);
    auto vmm_ubound = vmm_tmp(1);
    if (dt_requires_saturation) {
        init_saturate_f32(
                vmm_lbound, vmm_ubound, reg_tmp, data_type::f32, brg.dt_d);
    }

    for (int m = 0; m < m_blocks; m++) {
        if (dt_requires_saturation) {
            for (int n = 0; n < n_blocks; n++) {
                auto vmm = accm(m_blocks, n_blocks, m, n);
                saturate_f32(vmm, vmm_lbound, vmm_ubound, brg.dt_d);
                vcvtps2dq(vmm, vmm);
            }
        }
        for (int n = 0; n < n_blocks; n++) {
            auto addr = ptr[reg_aux_D + D_offset(m, n)];
            auto vmm = accm(m_blocks, n_blocks, m, n);
            auto wmm = Wmm(vmm.getIdx());
            const bool mask_flag = n + 1 == n_blocks && has_n_tail;
            const Vmm r_vmm = vmm_mask(vmm, mask_flag, true);
            const Wmm r_wmm = wmm_mask(wmm, mask_flag, true);
            switch (brg.dt_d) {
                case data_type::f32:
                case data_type::s32: vmovups(addr, r_vmm); break;
                case data_type::bf16:
                    vcvtneps2bf16(wmm, vmm);
                    vmovdqu16(addr, r_wmm);
                    break;
                case data_type::s8: vpmovsdb(addr, r_vmm); break;
                case data_type::u8: vpmovusdb(addr, r_vmm); break;
                default: assert(!"unknown dst_dt");
            }
        }
    }
}

void jit_brdgmm_kernel_base_t::store_accumulators_without_post_ops(
        int m_blocks, int n_blocks, bool has_n_tail) {

    const bool dt_requires_saturation
            = brg.is_int8 && brg.dt_c != data_type::s32;
    auto vmm_lbound = vmm_tmp(0);
    auto vmm_ubound = vmm_tmp(1);
    if (dt_requires_saturation) {
        init_saturate_f32(
                vmm_lbound, vmm_ubound, reg_tmp, data_type::f32, brg.dt_d);
    }

    for (int m = 0; m < m_blocks; m++) {
        for (int n = 0; n < n_blocks; n++) {
            const bool mask_flag = has_n_tail && n + 1 == n_blocks;
            auto vmm_acc = accm(m_blocks, n_blocks, m, n);
            auto vmm_acc_masked = vmm_mask(vmm_acc, mask_flag, true);
            if (dt_requires_saturation) {
                saturate_f32(vmm_acc, vmm_lbound, vmm_ubound, brg.dt_d);
                vcvtps2dq(vmm_acc, vmm_acc);
            }
            vmovups(ptr[reg_aux_C + C_offset(m, n)], vmm_acc_masked);
        }
    }
}

void jit_brdgmm_kernel_base_t::store_accumulators(
        int m_blocks, int n_blocks, bool has_n_tail) {

    if (is_fast_vnni_int8()) {
        for (int m_i = 0; m_i < m_blocks; ++m_i) {
            for (int n_i = 0; n_i < n_blocks; ++n_i) {
                auto vmm_out = accm(m_blocks, n_blocks, m_i, n_i);
                vpermd(vmm_out, vmm_permute(), vmm_out);
            }
        }
    }

    const bool are_post_ops_applicable
            = one_of(true, brg.with_eltwise, brg.with_binary, brg.with_scales,
                    brg.with_bias, brg.with_sum, brg.dt_d != brg.dt_c);

    Label label_done;
    if (are_post_ops_applicable) {
        store_accumulators_apply_post_ops(m_blocks, n_blocks, has_n_tail);
    } else {
        store_accumulators_without_post_ops(m_blocks, n_blocks, has_n_tail);
    }
}

void jit_brdgmm_kernel_base_t::brdgmm_microkernel(int m_blocks, int n_blocks,
        bool has_top_padding, bool has_bottom_padding, bool has_tail) {

    const bool has_padding = has_top_padding || has_bottom_padding;
    const int max_bvmms
            = accm(m_blocks, n_blocks, 0, 0).getIdx() - vmm_b(0).getIdx();

    auto load_a = [&](Vmm vmma, int m_i, int n_i) {
        const bool mask_flag = has_tail && (n_i + 1 == n_blocks);
        const auto addr = ptr[reg_aux_A + A_offset(m_i, n_i)];
        vmma = vmm_mask(vmma, mask_flag, false);
        if (brg.is_f32) {
            vmovups(vmma, addr);
        } else if (brg.is_bf16) {
            vpmovzxwd(vmma, addr);
        } else if (brg.is_int8) {
            if (is_fast_vnni_int8()) {
                assert(!mask_flag);
                vbroadcasti32x4(vmma, addr);
            } else {
                vpmovzxbd(vmma, addr);
            }
        }
    };

    auto load_b = [&](Vmm vmmb, int n_i) {
        const auto addr = ptr[reg_aux_B + B_offset(n_i)];
        if (brg.is_f32) {
            vmovups(vmmb, addr);
        } else if (brg.is_int8) {
            // wei is sign extend(s8), where as src is zero extended(u8).
            if (is_fast_vnni_int8()) {
                vbroadcasti32x4(vmmb, addr);
                vmovdqu8(vmmb | kblend_mask | T_z, vmmb);
            } else {
                vpmovsxbd(vmmb, addr);
            }
        } else if (brg.is_bf16) {
            vpmovzxwd(vmmb, addr);
        }
    };

    auto dot_product = [&](Vmm vmma, Vmm vmmb, int m_i, int n_i) {
        auto vmm_acc = accm(m_blocks, n_blocks, m_i, n_i);
        if (brg.is_f32) {
            if (is_fma_embd()) {
                const bool mask_flag = has_tail && (n_i + 1 == n_blocks);
                const auto addr = ptr[reg_aux_A + A_offset(m_i, n_i)];
                vmm_acc = vmm_mask(vmm_acc, mask_flag, false);
                vfmadd231ps(vmm_acc, vmmb, addr);
            } else {
                vfmadd231ps(vmm_acc, vmma, vmmb);
            }
        } else if (brg.is_bf16) {
            vdpbf16ps(vmm_acc, vmma, vmmb);
        } else if (brg.is_int8) {
            vpdpbusd(vmm_acc, vmma, vmmb);
        }
    };

    if (!has_padding) {
        // preload vmm_b if possible.
        for (int nb_i = 0; nb_i < n_blocks; nb_i += max_bvmms) {
            const int n_e = nstl::min(nb_i + max_bvmms, n_blocks) - nb_i;
            for (int i = 0; i < n_e; ++i) {
                const int n_i = nb_i + i;
                load_b(vmm_b(i), n_i);
            }
            for (int m_i = 0; m_i < m_blocks; ++m_i) {
                for (int i = 0; i < n_e; ++i) {
                    const int n_i = nb_i + i;
                    if (!is_fma_embd()) load_a(vmm_a(), m_i, n_i);
                    dot_product(vmm_a(), vmm_b(i), m_i, n_i);
                }
            }
        }
    } else {

        const int n_e = max_bvmms >= n_blocks ? n_blocks : (max_bvmms - 1);
        for (int i = 0; i < n_e; ++i) {
            load_b(vmm_b(i), i);
        }

        Label done;
        Label jmp_table_base;
        std::vector<Label> jmp_table_labels(m_blocks);
        if (has_top_padding) {
            // jmp table
            mov(reg_table_base, jmp_table_base);
            lea(reg_table_base,
                    ptr[reg_table_base + reg_aux_A_vpad_top * sizeof(void *)]);
            jmp(ptr[reg_table_base]);
            align(8);
            L(jmp_table_base);
            for (int m_i = 0; m_i < m_blocks; ++m_i) {
                putL(jmp_table_labels[m_i]);
            }
        }

        for (int m_i = 0; m_i < m_blocks; ++m_i) {
            L(jmp_table_labels[m_i]);
            if (has_bottom_padding) {
                cmp(reg_aux_A_vpad_bottom, m_blocks - m_i);
                jge(done, T_NEAR);
            }

            for (int n_i = 0; n_i < n_blocks; ++n_i) {
                if (!is_fma_embd()) load_a(vmm_a(), m_i, n_i);
                if (n_i < n_e) {
                    dot_product(vmm_a(), vmm_b(n_i), m_i, n_i);
                } else {
                    // preloaded vmm_b not available
                    const int b_idx = max_bvmms - 1;
                    load_b(vmm_b(b_idx), n_i);
                    dot_product(vmm_a(), vmm_b(b_idx), m_i, n_i);
                }
            }
        }
        L(done);
    }
}

void jit_brdgmm_kernel_base_t::batch_loop(
        const int m_blocks, const int n_blocks, bool has_n_tail) {

    auto get_padding_info = [&]() {
        const bool do_check_effective_padding = check_effective_padding();
        if (has_vpad()) {
            Label no_top_padding;

            if (brg.brgattr.max_bottom_vpad > 0) {
                if (do_check_effective_padding) {
                    Label done_adjust_bottom_padding;
                    mov(reg_aux_A_vpad_bottom, reg_aux_M);
                    add(reg_aux_A_vpad_bottom, m_blocks - M());
                    add(reg_aux_A_vpad_bottom,
                            ptr[reg_aux_batch_addr
                                    + GET_OFF_BATCH_ELEMENT(vvpad.bottom)]);
                    jge(done_adjust_bottom_padding, T_NEAR);
                    xor_(reg_aux_A_vpad_bottom, reg_aux_A_vpad_bottom);
                    L(done_adjust_bottom_padding);
                } else {
                    mov(reg_aux_A_vpad_bottom,
                            ptr[reg_aux_batch_addr
                                    + GET_OFF_BATCH_ELEMENT(vvpad.bottom)]);
                }
                mov(reg_total_padding, reg_aux_A_vpad_bottom);
            }
            if (brg.brgattr.max_top_vpad > 0) {
                mov(reg_aux_A_vpad_top,
                        ptr[reg_aux_batch_addr
                                + GET_OFF_BATCH_ELEMENT(vvpad.top)]);
                if (do_check_effective_padding) {
                    Label done_adjust_top_padding;
                    sub(reg_aux_A_vpad_top, reg_aux_M);
                    jge(done_adjust_top_padding, T_NEAR);
                    xor_(reg_aux_A_vpad_top, reg_aux_A_vpad_top);
                    L(done_adjust_top_padding);
                }
                if (brg.brgattr.max_bottom_vpad > 0) {
                    add(reg_total_padding, reg_aux_A_vpad_top);
                } else {
                    mov(reg_total_padding, reg_aux_A_vpad_top);
                }
            }
        }
    };

    auto call_brdgmm_microkernel = [&]() {
        const int tpad = brg.brgattr.max_top_vpad;
        const int bpad = brg.brgattr.max_bottom_vpad;
        const bool vpad_exists = has_vpad();
        Label microkernel_with_padding, done_microkernel;

        if (vpad_exists) {
            cmp(reg_total_padding, 0);
            jg(microkernel_with_padding, T_NEAR);
        }
        brdgmm_microkernel(m_blocks, n_blocks, false, false, has_n_tail);
        if (vpad_exists) {
            jmp(done_microkernel, T_NEAR);
            L(microkernel_with_padding);
            if ((tpad + bpad) >= m_blocks) {
                cmp(reg_total_padding, m_blocks);
                jge(done_microkernel, T_NEAR);
            }
            brdgmm_microkernel(m_blocks, n_blocks, tpad, bpad, has_n_tail);
        }
        L(done_microkernel);
    };

    Label bs_loop_label, done_bs_loop;
    load_accumulators(m_blocks, n_blocks);
    cmp(reg_BS, 0);
    jle(done_bs_loop, T_NEAR);
    mov(reg_BS_loop, reg_BS);
    restore_A_B_matrices();

    L(bs_loop_label);
    {
        set_A_B_matrices();
        get_padding_info();
        advance_A_B_matrices();
        call_brdgmm_microkernel();
        dec(reg_BS_loop);
        jg(bs_loop_label, T_NEAR);
    }

    L(done_bs_loop);

    store_accumulators(m_blocks, n_blocks, has_n_tail);
}

void jit_brdgmm_kernel_base_t::compute_loop() {

    const bool has_m_blocking_tail = m_blocking_tail() > 0;
    const int loop_m = (nb_m_blocking() - has_m_blocking_tail);
    const bool do_loop_m = loop_m > 1;

    const bool has_n_blocking_tail = n_blocking_tail() > 0;
    const int loop_n = nb_n_blocking() - has_n_blocking_tail;
    const bool do_loop_n = loop_n > 1;

    auto n_loop = [&](int m_blocks) {
        Label n_loop_label;
        const int n_blocks = n_blocking();
        const int n_loop_step = oc_logical_offset(n_blocks);
        const int n_loop_work = (nb_n_blocking() - has_n_blocking_tail)
                * n_blocks * n_vlen_blk();
        const bool vlen_tail_in_loop
                = n_vlen_tail() != 0 && !has_n_blocking_tail;

        xor_(reg_aux_N, reg_aux_N);

        L(n_loop_label);
        {
            if (do_loop_n) {
                if (vlen_tail_in_loop) {
                    Label done_k_mask;
                    cmp(reg_aux_N, n_loop_work - n_loop_step);
                    jl(done_k_mask, T_NEAR);
                    kmovd(k_mask, k_tail_mask);
                    L(done_k_mask);
                }
            }

            batch_loop(m_blocks, n_blocks, vlen_tail_in_loop);

            if (do_loop_n || has_n_blocking_tail) {
                add(reg_aux_N, n_loop_step);
                add(reg_a_offset, n_loop_step * brg.typesize_A);
                add(reg_aux_C, n_loop_step * brg.typesize_C);
                add(reg_aux_D, n_loop_step * brg.typesize_D);
                if (with_binary_per_oc_bcast_) {
                    add(qword[rsp + reg_binary_postops_oc_l_offs_],
                            n_loop_step);
                }
            }

            if (do_loop_n) {
                cmp(reg_aux_N, n_loop_work);
                jl(n_loop_label, T_NEAR);
            }
        }

        if (has_n_blocking_tail) {
            batch_loop(m_blocks, n_blocking_tail(), n_vlen_tail() != 0);
        }
    };

    auto m_loop = [&]() {
        Label m_loop_label;
        const int m_blocks = m_blocking();
        const bool reset_mask
                = n_vlen_tail() != 0 && do_loop_n && !has_n_blocking_tail;

        xor_(reg_aux_M, reg_aux_M);
        xor_(reg_a_offset, reg_a_offset);

        L(m_loop_label);
        {
            if (reset_mask) kxnorq(k_mask, k_mask, k_mask);
            n_loop(m_blocks);

            if (do_loop_m || has_m_blocking_tail) {
                add(reg_aux_M, m_blocks);
                const int n_loop_offset = (do_loop_n || has_n_blocking_tail)
                        * loop_n * n_blocking();
                add(reg_a_offset, A_offset(m_blocks, -n_loop_offset));
                add(reg_aux_C, C_offset(m_blocks, -n_loop_offset));
                add(reg_aux_D, D_offset(m_blocks, -n_loop_offset));
                if (with_binary_per_oc_bcast_) {
                    add(qword[rsp + reg_binary_postops_oc_l_offs_],
                            oc_logical_offset(-n_loop_offset));
                }
            }

            if (do_loop_m) {
                cmp(reg_aux_M, loop_m * m_blocking());
                jl(m_loop_label, T_NEAR);
            }
        }

        if (m_blocking_tail() > 0) {
            if (reset_mask) { kxnorq(k_mask, k_mask, k_mask); }
            n_loop(m_blocking_tail());
        }
    };

    assert(m_vlen_tail() == 0);
    m_loop();
}

void jit_brdgmm_kernel_base_t::generate() {

    preamble();
    sub(rsp, stack_space_needed_);

    Label permute_index_table;
    if (is_fast_vnni_int8()) {
        mov(reg_tmp, 0x8888444422221111);
        kmovq(kblend_mask, reg_tmp);
        // load permute indices from data section
        mov(reg_tmp, permute_index_table);
        vmovdqu32(vmm_permute(), ptr[reg_tmp]);
    }

    if (n_vlen_tail() != 0) {
        const auto tail_mask = size_t((1 << n_vlen_tail()) - 1);
        const bool has_n_blocking_tail = n_blocking_tail() > 0;
        mov(reg_tmp, tail_mask);
        if (has_n_blocking_tail || nb_n_blocking() <= 1) {
            // The mask can be set only once.
            kmovq(k_mask, reg_tmp);
        } else {
            // Need to adjust mask, and set only when needed.
            // So store it temporarily in k_tail_mask.
            kmovq(k_tail_mask, reg_tmp);
        }
    } else if (brg.with_binary) {
        // the post-ops injector seems to use mask unconditionally
        // set a default mask.
        kxnorq(k_mask, k_mask, k_mask);
    }

    read_params();
    compute_loop();

    add(rsp, stack_space_needed_);
    postamble();

    if (brg.with_eltwise) postops_injector_->prepare_table();

    if (is_fast_vnni_int8()) {
        align(64);
        L(permute_index_table);
        const uint32_t _idx[]
                = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
        for (size_t i = 0; i < sizeof(_idx) / sizeof(_idx[0]); ++i)
            dd(_idx[i]);
    }
}

brdgmm_kernel_t::brdgmm_kernel_t(const brgemm_t abrd) : brgemm_kernel_t(abrd) {
    brgemm_kernel_ = new jit_brdgmm_kernel_base_t(abrd);
}

status_t brdgmm_kernel_t::create_kernel() {
    return brgemm_kernel_->create_kernel();
}

void brdgmm_kernel_t::operator()(brgemm_kernel_params_t *params) const {
    (*brgemm_kernel_)(params);
}

brdgmm_kernel_t::~brdgmm_kernel_t() {
    delete brgemm_kernel_;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
