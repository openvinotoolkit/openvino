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

#ifndef CPU_X64_JIT_BRDGMM_KERNEL_HPP
#define CPU_X64_JIT_BRDGMM_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/brgemm/brgemm_types.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
struct jit_brdgmm_kernel_base_t : public jit_generator {
    jit_brdgmm_kernel_base_t(const brgemm_t &abrd);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brdgmm_kernel_base_t)

    brgemm_t brg;

    static bool is_fast_vnni_int8(const brgemm_t &brg) {
        return brg.is_dgmm && brg.is_int8 && brg.ldb_tail /*n_vlen_tail*/ == 0;
    }

private:
    // using alias for Zmm and Ymm. useful for future avx2 support.
    using Vmm = Xbyak::Zmm;
    using Wmm =
            typename utils::conditional<std::is_same<Vmm, Xbyak::Zmm>::value,
                    Xbyak::Ymm, Xbyak::Xmm>::type;
    std::unique_ptr<injector::jit_uni_postops_injector_t<avx512_core, Vmm>>
            postops_injector_;

    using reg64_t = const Xbyak::Reg64;
    // Register decomposition
    const reg64_t param1 = abi_param1;
    const reg64_t reg_A = abi_not_param1;
    const reg64_t reg_B = r8;
    const reg64_t reg_aux_batch_addr = r15;
    const reg64_t reg_BS = rsi;

    // loop variables
    const reg64_t reg_BS_loop = r12;
    const reg64_t reg_aux_M = r13;
    const reg64_t reg_aux_D = rbx;
    const reg64_t reg_aux_C = rdx;
    const reg64_t reg_aux_A = r10;
    const reg64_t reg_aux_B = abi_param1;
    const reg64_t reg_aux1_A = reg_A; // brgemm_strd
    const reg64_t reg_aux1_B = reg_B; // brgemm_strd
    const reg64_t reg_a_offset = r9;
    const reg64_t reg_aux_N = r11;

    const reg64_t reg_aux_A_vpad_top = r14;
    const reg64_t reg_aux_A_vpad_bottom = rbp;

    const reg64_t reg_table_base = rax;
    const reg64_t reg_tmp = reg_table_base;
    const reg64_t reg_total_padding = reg_table_base;
    const reg64_t reg_aux_bias = reg_table_base;
    const reg64_t reg_aux_scales = reg_table_base;
    const reg64_t reg_binary_po_stack_frame = reg_BS_loop;
    const reg64_t reg_binary_params = abi_param1; // default for binary ops
    const reg64_t reg_binary_rhs = reg_aux_A;
    const reg64_t reg_ptr_sum_scale = reg_aux_A_vpad_top;
    const reg64_t reg_ptr_sum_zp = reg_aux_A_vpad_bottom;

    Xbyak::Opmask k_mask = Xbyak::Opmask(2);
    Xbyak::Opmask k_tail_mask = Xbyak::Opmask(3);
    Xbyak::Opmask kblend_mask = Xbyak::Opmask(4);

    constexpr static int max_vmms_ = 32;
    constexpr static int reg_batch0_addr_offs_ = 0;
    constexpr static int reg_bias_offs_ = 8;
    constexpr static int reg_scales_offs_ = 16;
    constexpr static int reg_A_offs_ = 24; // brgemm_strd
    constexpr static int reg_B_offs_ = 32; // brgemm_strd
    constexpr static int abi_param1_offs_ = 40;
    constexpr static int reg_binary_postops_oc_l_offs_ = 48;
    constexpr static int reg_data_C_ptr_offs_ = 56;
    constexpr static int stack_space_needed_ = 64;

    bool handle_binary_po_offset_ = false;
    bool with_binary_per_oc_bcast_ = false;

    inline int M() { return brg.bcast_dim; };
    inline int N() { return brg.load_dim; };
    inline int m_vlen_blk() { return brg.bd_block; };
    inline int nb_m_vlen_blk() { return brg.bdb; };
    inline int m_vlen_tail() { return brg.bdb_tail; };
    inline int m_blocking() { return brg.bd_block2; };
    inline int nb_m_blocking() { return brg.bdb2; };
    inline int m_blocking_tail() { return brg.bdb2_tail; };

    inline int n_vlen_blk() { return brg.ld_block; };
    inline int nb_n_vlen_blk() { return brg.ldb; };
    inline int n_vlen_tail() { return brg.ldb_tail; };
    inline int n_blocking() { return brg.ld_block2; };
    inline int nb_n_blocking() { return brg.ldb2; };
    inline int n_blocking_tail() { return brg.ldb2_tail; };

    bool is_fma_embd() { return brg.is_f32; }
    bool is_fast_vnni_int8() { return is_fast_vnni_int8(brg); }
    Vmm vmm_permute() { return Vmm(0); } // used in fast_vnni_int8
    Vmm vmm_a() { return Vmm(is_fast_vnni_int8()); }
    Vmm vmm_b(int bi = 0) {
        return Vmm(is_fast_vnni_int8() + !is_fma_embd() + bi);
    }
    Vmm accm(int m_blocks, int n_blocks, int m, int n) {
        assert(m_blocks <= m_blocking() && m < m_blocks);
        assert(n_blocks <= n_blocking() && n < n_blocks);
        const int accm_start = max_vmms_ - m_blocks * n_blocks;
        const int accm_rel_idx = n * m_blocks + m;
        const int idx = accm_start + accm_rel_idx;
        assert(idx < max_vmms_ && idx > vmm_b(0).getIdx());
        return Vmm(idx);
    }
    Vmm vmm_tmp(int i) {
        const int idx = max_vmms_ - m_blocking() * n_blocking() - 1 - i;
        assert(idx > (is_fast_vnni_int8() - 1));
        return Vmm(idx);
    }
    Vmm vmm_mask(const Vmm vmm_in, bool mask_flag, bool store);
    Wmm wmm_mask(const Wmm wmm_in, bool mask_flag, bool store);

    void read_params();
    void load_accumulators(int m_blocks, int n_blocks);
    void restore_A_B_matrices();
    void set_A_B_matrices();
    void advance_A_B_matrices();
    void brdgmm_microkernel(int m_blocks, int n_blocks, bool has_top_padding,
            bool has_bottom_padding, bool has_tail = false);
    void compute_loop();
    void batch_loop(const int m_blocks, const int n_blocks, bool has_n_tail);
    void cvt2ps(data_type_t type_in, const Vmm vmm_in, const Xbyak::Operand &op,
            bool mask_flag, bool store);
    void apply_post_ops(int m_blocks, int n_blocks, bool has_n_tail);
    void store_accumulators(int m_blocks, int n_blocks, bool has_n_tail);
    void store_accumulators_without_post_ops(
            int m_blocks, int n_blocks, bool has_n_tail);
    void store_accumulators_apply_post_ops(
            int m_blocks, int n_blocks, bool has_n_tail);

    bool has_vpad() {
        return brg.brgattr.max_top_vpad > 0 || brg.brgattr.max_bottom_vpad > 0;
    }
    bool check_effective_padding() { return has_vpad() && M() > m_blocking(); }

    int oc_logical_offset(int n) { return n * n_vlen_blk(); }
    int A_offset(int m, int n) {
        return brg.typesize_A * (m * brg.LDA + n * n_vlen_blk());
    }
    int B_offset(int n) { return brg.typesize_B * n * n_vlen_blk(); }
    int C_offset(int m, int n) {
        return brg.typesize_C * (m * brg.LDC + n * n_vlen_blk());
    }
    int D_offset(int m, int n) {
        return brg.typesize_D * (m * brg.LDD + n * n_vlen_blk());
    }
    int bias_offset(int n) { return brg.typesize_bias * n * n_vlen_blk(); }
    int scales_offset(int n) {
        return sizeof(float) * brg.is_oc_scale * n * n_vlen_blk();
    }

    void generate() override;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif