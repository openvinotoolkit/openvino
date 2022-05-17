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

#ifndef CPU_X64_JIT_BRGEMM_POST_OPS_HPP
#define CPU_X64_JIT_BRGEMM_POST_OPS_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/cpu_engine.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_brgemm_primitive_conf.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct brgemm_kernel_diff_bias_t {
    void *ptr_diff_dst;
    void *ptr_diff_bias_acc;
    void *ptr_diff_bias;
    int flags;
};

#define GET_OFF(field) offsetof(brgemm_kernel_diff_bias_t, field)

struct jit_brgemm_kernel_diff_bias_t : public jit_generator {
    jit_brgemm_kernel_diff_bias_t(
            const jit_brgemm_primitive_conf_t &ajbgp, const brgemm_t &abrg)
        : brg_(abrg)
        , ddst_dt_(ajbgp.dst_dt)
        , bia_dt_(ajbgp.bia_dt)
        , acc_dt_(ajbgp.acc_dt)
        , ddst_typesize_(types::data_type_size(ddst_dt_))
        , bia_typesize_(types::data_type_size(bia_dt_))
        , acc_typesize_(types::data_type_size(acc_dt_)) {}

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_diff_bias_t)

private:
    brgemm_t brg_;
    data_type_t ddst_dt_;
    data_type_t bia_dt_;
    data_type_t acc_dt_;

    int ddst_typesize_;
    int bia_typesize_;
    int acc_typesize_;

    using reg64_t = const Xbyak::Reg64;
    // Register decomposition
    const reg64_t param1 = abi_param1;
    const reg64_t reg_ddst = r15;
    const reg64_t reg_bias = r14;
    const reg64_t reg_bias_acc = r13;
    const reg64_t aux_reg_ddst = r12;
    const reg64_t reg_k_iter = r11;
    const reg64_t reg_flag = r10;

    Xbyak::Opmask k_full_mask = Xbyak::Opmask(2);
    Xbyak::Opmask k_tail_mask = Xbyak::Opmask(3);
    Xbyak::Zmm vreg_unit = Xbyak::Zmm(31);

    const int n_max_regs_ = 4;

    const Xbyak::Zmm zmm_mask(const Xbyak::Zmm zmm_in, bool mask_flag,
            bool store, Xbyak::Opmask ktail_mask) {
        return mask_flag
                ? (store ? zmm_in | ktail_mask : zmm_in | ktail_mask | T_z)
                : zmm_in;
    }

    void loop_by_N(int n_loop, int nb_tail) {

        mov(aux_reg_ddst, reg_ddst);
        int mult = ddst_dt_ == data_type::bf16 ? 2 : 1;
        int n_iters = n_loop;
        if (nb_tail > 0) n_iters--;
        Xbyak::Label k_loop, init_zero, init_done;
        auto get_bias_reg = [=](int n) { return Xbyak::Zmm(n); };
        auto get_bias_reg_lower = [=](int n) { return Xbyak::Ymm(n); };
        auto get_ddst_reg = [=](int n) { return Xbyak::Zmm(n + n_max_regs_); };
        int n_ = 0;

        test(reg_flag, FLAG_REDUCE_FIRST);
        jnz(init_zero, T_NEAR); // FLAG_REDUCE_FIRST is set

        for (; n_ < n_iters; n_++) {
            auto vbias = get_bias_reg(n_);
            auto addr = ptr[reg_bias_acc + acc_typesize_ * n_ * brg_.ld_block];
            vmovups(vbias, addr);
        }
        if (nb_tail > 0) {
            auto vbias = zmm_mask(get_bias_reg(n_), true, false, k_tail_mask);
            auto addr = ptr[reg_bias_acc + acc_typesize_ * n_ * brg_.ld_block];
            vmovups(vbias, addr);
        }
        jmp(init_done, T_NEAR);
        L(init_zero);

        for (int n_ = 0; n_ < n_loop; n_++) {
            vxorpd(get_bias_reg(n_), get_bias_reg(n_), get_bias_reg(n_));
        }
        L(init_done);

        mov(reg_k_iter, utils::div_up(brg_.reduce_dim, mult));
        L(k_loop);
        {
            int n_ = 0;
            for (; n_ < n_iters; n_++) {
                auto vddst = get_ddst_reg(n_);
                auto vbias = get_bias_reg(n_);
                auto addr = ptr[aux_reg_ddst
                        + ddst_typesize_ * mult * n_ * brg_.ld_block];
                vmovups(vddst, addr);
                if (ddst_dt_ == data_type::bf16)
                    vdpbf16ps(vbias, vreg_unit, vddst);
                else
                    vaddps(vbias, vbias, vddst);
            }

            if (nb_tail > 0) {
                auto vddst = get_ddst_reg(n_);
                auto vddst_load = zmm_mask(vddst, true, false, k_tail_mask);
                auto vbias = get_bias_reg(n_);

                auto addr = ptr[aux_reg_ddst
                        + ddst_typesize_ * mult * n_ * brg_.ld_block];
                vmovups(vddst_load, addr);
                if (ddst_dt_ == data_type::bf16)
                    vdpbf16ps(vbias, vreg_unit, vddst);
                else
                    vaddps(vbias, vbias, vddst);
            }

            add(aux_reg_ddst, ddst_typesize_ * mult * brg_.LDB);

            sub(reg_k_iter, 1);
            jnz(k_loop, T_NEAR);
        }

        Xbyak::Label store_final, store_done;
        test(reg_flag, FLAG_REDUCE_LAST);
        jnz(store_final, T_NEAR); // FLAG_REDUCE_LAST is set

        n_ = 0;
        for (; n_ < n_iters; n_++) {
            auto vbias = get_bias_reg(n_);
            auto addr = ptr[reg_bias_acc + acc_typesize_ * n_ * brg_.ld_block];
            vmovups(addr, vbias);
        }
        if (nb_tail > 0) {
            auto addr = ptr[reg_bias_acc + acc_typesize_ * n_ * brg_.ld_block];
            auto vbias = zmm_mask(get_bias_reg(n_), true, true, k_tail_mask);
            vmovups(addr, vbias);
        }
        jmp(store_done, T_NEAR);

        L(store_final);
        n_ = 0;
        for (; n_ < n_iters; n_++) {
            auto vbias = get_bias_reg(n_);
            auto addr = ptr[reg_bias + bia_typesize_ * n_ * brg_.ld_block];
            if (bia_dt_ == data_type::bf16) {
                auto vbias_lower = get_bias_reg_lower(n_);
                vcvtneps2bf16(vbias_lower, vbias);
                vmovups(addr, vbias_lower);
            } else
                vmovups(addr, vbias);
        }
        if (nb_tail > 0) {
            auto addr = ptr[reg_bias + bia_typesize_ * n_ * brg_.ld_block];
            if (bia_dt_ == data_type::bf16) {
                auto vbias = get_bias_reg(n_);
                auto vbias_lower = get_bias_reg_lower(n_);
                vcvtneps2bf16(vbias_lower, vbias);
                auto vbias_store = zmm_mask(vbias, true, true, k_tail_mask);
                vmovdqu16(addr, vbias_store);
            } else {
                auto vbias
                        = zmm_mask(get_bias_reg(n_), true, true, k_tail_mask);
                vmovups(addr, vbias);
            }
        }
        L(store_done);
    }

    void generate() override {
        preamble();

        int nb = utils::div_up(brg_.load_dim, brg_.ld_block);
        int nb_tail = brg_.load_dim % brg_.ld_block;

        int n_loop = nb / n_max_regs_;
        int n_loop_tail = nb % n_max_regs_;
        if (n_loop_tail == 0 && nb_tail > 0) {
            n_loop--;
            n_loop_tail = n_max_regs_;
        }

        const auto full_mask = size_t {0xffffffffffffffff};
        const auto tail_mask = size_t((1 << nb_tail) - 1);
        reg64_t reg_mask = rax;

        mov(reg_mask, full_mask);
        kmovq(k_full_mask, reg_mask);
        mov(reg_mask, tail_mask);
        kmovq(k_tail_mask, reg_mask);

        if (ddst_dt_ == data_type::bf16) {
            auto reg_unit_val = reg_mask.cvt16();
            mov(reg_unit_val, 0x3f80); // bf16 value of 1.
            vpbroadcastw(vreg_unit, reg_unit_val);
        }

        mov(reg_ddst, ptr[param1 + GET_OFF(ptr_diff_dst)]);
        mov(reg_bias_acc, ptr[param1 + GET_OFF(ptr_diff_bias_acc)]);
        mov(reg_bias, ptr[param1 + GET_OFF(ptr_diff_bias)]);
        mov(reg_flag, ptr[param1 + GET_OFF(flags)]);

        int mult = ddst_dt_ == data_type::bf16 ? 2 : 1;
        for (int nb_ = 0; nb_ < n_loop; nb_++) {
            loop_by_N(n_max_regs_, 0);

            add(reg_ddst, ddst_typesize_ * mult * n_max_regs_ * brg_.ld_block);
            add(reg_bias, bia_typesize_ * n_max_regs_ * brg_.ld_block);
            add(reg_bias_acc, acc_typesize_ * n_max_regs_ * brg_.ld_block);
        }

        if (n_loop_tail > 0) loop_by_N(n_loop_tail, nb_tail);
        postamble();
    }
};

#undef GET_OFF

#define GET_OFF(field) offsetof(brgemm_kernel_post_ops_t, field)

struct brgemm_kernel_post_ops_t {
    void *ptr_in;
    void *ptr_out;
    void *ptr_bias;
    void *ptr_scales;
    const void *ptr_binary_post_ops_rhs;
    size_t oc_l_offset;
};

struct jit_brgemm_kernel_post_ops : public jit_generator {

    jit_brgemm_kernel_post_ops(const jit_brgemm_conv_conf_t &ajcp,
            const brgemm_t &abrg, const primitive_attr_t &aattr)
        : brg(abrg)
        , jcp(ajcp)
        , attr(aattr)
        , postops_injector_(nullptr)
        , with_binary_per_oc_bcast_(brg.with_binary
                  && binary_injector::any_binary_postop_rhs_per_oc_broadcast(
                          brg.attr->post_ops_,
                          memory_desc_wrapper(brg.dst_md))) {

        if ((jcp.with_sum && brg.beta != 0)
                || ((jcp.with_binary || jcp.with_eltwise) && brg.alpha != 0)) {
            static constexpr bool preserve_gpr = true;
            static constexpr bool preserve_vmm = true;
            static constexpr bool use_exact_tail_scalar_bcast = false;

            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    static_cast<size_t>(Xbyak::Zmm(28).getIdx()), this->rax,
                    this->r11, preserve_gpr, preserve_vmm,
                    GET_OFF(ptr_binary_post_ops_rhs),
                    memory_desc_wrapper(brg.dst_md),
                    static_cast<size_t>(brg.load_dim % brg.ld_block),
                    k_tail_mask, use_exact_tail_scalar_bcast};
            const binary_injector::static_params_t bsp {this->param1, rhs_sp};

            static constexpr bool save_state = true;
            const auto &reserved_eltwise_gpr = rax;
            const auto reserved_eltwise_maskr = Xbyak::Opmask(1);

            const eltwise_injector::static_params_t esp {
                    save_state, reserved_eltwise_gpr, reserved_eltwise_maskr};

            postops_injector_ = utils::make_unique<
                    injector::jit_uni_postops_injector_t<avx512_common>>(
                    this, attr.post_ops_, bsp, esp);
        }

        const auto &oscales = attr.output_scales_;
        is_oc_scale_ = oscales.mask_ == 1 << 1;

        LDD_ = brg.LDD;
        inp_dt_ = brg.dt_c;
        out_dt_ = brg.dt_d;
        bia_dt_ = jcp.bia_dt;
        inp_typesize_ = types::data_type_size(inp_dt_);
        out_typesize_ = types::data_type_size(out_dt_);
        bia_typesize_ = (jcp.with_bias) ? types::data_type_size(bia_dt_) : 0;
    }

    ~jit_brgemm_kernel_post_ops() = default;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_post_ops)

    brgemm_t brg;
    jit_brgemm_conv_conf_t jcp;
    const primitive_attr_t &attr;

private:
    int LDD_;

    data_type_t inp_dt_;
    data_type_t out_dt_;
    data_type_t bia_dt_;

    std::unique_ptr<injector::jit_uni_postops_injector_t<avx512_common>>
            postops_injector_;
    const bool with_binary_per_oc_bcast_;

    int inp_typesize_;
    int out_typesize_;
    int bia_typesize_;

    int is_oc_scale_;

    using reg64_t = const Xbyak::Reg64;

    // Register decomposition
    const reg64_t param1 = abi_param1;
    const reg64_t reg_in = r15;
    const reg64_t reg_out = r14;
    const reg64_t aux_reg_in = r13;
    const reg64_t aux_reg_out = r12;

    const reg64_t reg_bias = r11;
    const reg64_t aux_reg_bias = r10;

    const reg64_t reg_scales = r9;
    const reg64_t aux_reg_scales = r8;

    const reg64_t reg_ptr_sum_scale = rdx;
    const reg64_t reg_ptr_sum_zp = rsi;

    const reg64_t reg_oc_l_offset_ = abi_not_param1;
    const reg64_t aux_reg_oc_l_offset_ = rbx;

    Xbyak::Opmask k_full_mask = Xbyak::Opmask(2);
    Xbyak::Opmask k_tail_mask = Xbyak::Opmask(3);

    const int n_block2_ = 4;

    const Xbyak::Zmm zmm_mask(const Xbyak::Zmm zmm_in, bool mask_flag,
            bool store, Xbyak::Opmask ktail_mask) {
        return mask_flag
                ? (store ? zmm_in | ktail_mask : zmm_in | ktail_mask | T_z)
                : zmm_in;
    }

    const Xbyak::Ymm ymm_mask(const Xbyak::Ymm ymm_in, bool mask_flag,
            bool store, Xbyak::Opmask ktail_mask) {
        return mask_flag
                ? (store ? ymm_in | ktail_mask : ymm_in | ktail_mask | T_z)
                : ymm_in;
    }

    void cvt2ps(data_type_t type_in, const Xbyak::Zmm zmm_in,
            const Xbyak::Operand &op, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask) {
        const Xbyak::Zmm zmm = zmm_mask(zmm_in, mask_flag, store, ktail_mask);
        switch (type_in) {
            case data_type::f32:
            case data_type::s32: vmovups(zmm, op); break;
            case data_type::s8: vpmovsxbd(zmm, op); break;
            case data_type::u8: vpmovzxbd(zmm, op); break;
            case data_type::bf16:
                vpmovzxwd(zmm, op);
                vpslld(zmm, zmm, 16);
                break;
            default: assert(!"unsupported data type");
        }
        if (!utils::one_of(type_in, data_type::f32, data_type::bf16))
            vcvtdq2ps(zmm_in, zmm_in);
    }

    Xbyak::Zmm vector(int m, int n, int n_block) {
        return Xbyak::Zmm(m * n_block + n);
    };

    void inject_attr_postops(int m_block, int n_block, int tail = 0) {
        const auto &p = attr.post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);
        const auto k_mask = tail == 0 ? k_full_mask : k_tail_mask;
        const auto sum_dt = p.get_sum_dt(out_dt_);

        const auto sum_injector = [&] {
            const float *p_sum_scale = &p.entry_[sum_idx].sum.scale;
            const int32_t *p_sum_zp = &p.entry_[sum_idx].sum.zero_point;
            if (*p_sum_scale != 1.f)
                mov(reg_ptr_sum_scale, (size_t)p_sum_scale);
            auto zmm_sum_zp = Xbyak::Zmm(30);
            if (*p_sum_zp != 0) {
                mov(reg_ptr_sum_zp, (size_t)p_sum_zp);
                vcvtdq2ps(zmm_sum_zp, ptr_b[reg_ptr_sum_zp]);
            }

            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                const auto zmm = vector(m, n, n_block);
                const auto addr = ptr[aux_reg_out
                        + out_typesize_ * (m * LDD_ + n * brg.ld_block)];

                const auto zmm_prev_dst = Xbyak::Zmm(31);
                cvt2ps(sum_dt, zmm_prev_dst, addr, true, false, k_mask);
                if (*p_sum_zp != 0) vsubps(zmm_prev_dst, zmm_sum_zp);
                if (*p_sum_scale == 1.f)
                    vaddps(zmm, zmm_prev_dst);
                else
                    vfmadd231ps(zmm, zmm_prev_dst, zword_b[reg_ptr_sum_scale]);
            }
        };

        if (jcp.with_sum && brg.beta != 0) {
            postops_injector_->set_lambda_injector(
                    primitive_kind::sum, sum_injector);
        }

        binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;

        if (with_binary_per_oc_bcast_) {
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                const auto zmm_idx = vector(m, n, n_block).getIdx();

                rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
                        zmm_idx, aux_reg_oc_l_offset_);
                rhs_arg_params.vmm_idx_to_oc_elem_off_val.emplace(
                        zmm_idx, n * brg.ld_block);
                if (tail) rhs_arg_params.vmm_tail_idx_.emplace(zmm_idx);
            }
        }

        postops_injector_->compute_vector_range(
                0, m_block * n_block, rhs_arg_params);
    }

    void apply_post_ops(int m_block, int n_block, int tail = 0) {
        const auto vector
                = [=](int m, int n) { return Xbyak::Zmm(m * n_block + n); };
        auto k_mask = (tail == 0) ? k_full_mask : k_tail_mask;
        const auto &p = attr.post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);

        // brg.alpha == 0 means no read from input, no bias, no eltwise - just
        // initialize registers by zero at the beginning of kernel
        // brg.beta == 0 means no sum - just registers write to output
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            if (brg.alpha == 0) {
                if (sum_idx != -1 && brg.beta != 0) {
                    // if sum then have to init zmm each time
                    vpxord(vector(m, n), vector(m, n), vector(m, n));
                }
            } else {
                auto inp_addr = ptr[aux_reg_in
                        + inp_typesize_ * (m * brg.LDC + n * brg.ld_block)];
                cvt2ps(inp_dt_, vector(m, n), inp_addr, true, false, k_mask);
            }
        }

        if (brg.alpha != 0 && jcp.with_bias) {
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                auto zmm_bias = Xbyak::Zmm(31);
                auto bias_addr = ptr[aux_reg_bias
                        + bia_typesize_ * (n * brg.ld_block)];

                cvt2ps(bia_dt_, zmm_bias, bias_addr, true, false, k_mask);
                vaddps(vector(m, n), zmm_bias);
            }
        }

        if (brg.alpha != 0) {
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                const Xbyak::Zmm zmm
                        = zmm_mask(vector(m, n), true, false, k_mask);
                vmulps(zmm, zmm,
                        ptr[aux_reg_scales
                                + is_oc_scale_ * sizeof(float)
                                        * (n * brg.ld_block)]);
            }
        }

        if (postops_injector_) inject_attr_postops(m_block, n_block, tail);

        const bool dt_requires_saturation = utils::one_of(
                brg.dt_d, data_type::u8, data_type::s8, data_type::s32);

        const reg64_t reg_tmp_gpr = rax;
        auto zmm_lbound = Xbyak::Zmm(31);
        auto zmm_ubound = Xbyak::Zmm(30);
        if (dt_requires_saturation) {
            init_saturate_f32(zmm_lbound, zmm_ubound, reg_tmp_gpr,
                    data_type::f32, brg.dt_d);
        }

        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            auto zmm = vector(m, n);
            auto addr = ptr[aux_reg_out
                    + out_typesize_ * (m * LDD_ + n * brg.ld_block)];

            if (out_dt_ == data_type::bf16) {
                Xbyak::Ymm ymm = Xbyak::Ymm(zmm.getIdx());
                if (brg.alpha != 0 || (sum_idx != -1 && brg.beta != 0))
                    vcvtneps2bf16(ymm, zmm);
                const Xbyak::Ymm r_ymm = ymm_mask(ymm, true, true, k_mask);
                vmovdqu16(addr, r_ymm);
            } else {
                if (brg.alpha != 0 || (sum_idx != -1 && brg.beta != 0)) {
                    saturate_f32(zmm, zmm_lbound, zmm_ubound, brg.dt_d);
                    if (out_dt_ != data_type::f32) vcvtps2dq(zmm, zmm);
                }

                const Xbyak::Zmm r_zmm = zmm_mask(zmm, true, true, k_mask);
                switch (out_dt_) {
                    case data_type::f32:
                    case data_type::s32: vmovups(addr, r_zmm); break;
                    case data_type::s8: vpmovsdb(addr, r_zmm); break;
                    case data_type::u8: vpmovusdb(addr, r_zmm); break;
                    default: assert(!"unknown dst_dt");
                }
            }
        }
    }

    void loop_by_N(int m_block, int nb2, int nb2_tail, int nb_tail) {

        if (brg.alpha) {
            mov(aux_reg_in, reg_in);
            if (jcp.with_bias) mov(aux_reg_bias, reg_bias);
            if (with_binary_per_oc_bcast_)
                mov(aux_reg_oc_l_offset_, reg_oc_l_offset_);
            mov(aux_reg_scales, reg_scales);
        }
        mov(aux_reg_out, reg_out);

        for (int n_loop_ = 0; n_loop_ < nb2; n_loop_++) {
            apply_post_ops(m_block, n_block2_);

            const auto oc_l_offset = n_block2_ * brg.ld_block;

            add(aux_reg_out, out_typesize_ * oc_l_offset);
            if (brg.alpha != 0) {
                add(aux_reg_in, inp_typesize_ * oc_l_offset);

                if (jcp.with_bias)
                    add(aux_reg_bias, bia_typesize_ * oc_l_offset);
                if (with_binary_per_oc_bcast_)
                    add(aux_reg_oc_l_offset_, oc_l_offset);

                add(aux_reg_scales, is_oc_scale_ * sizeof(float) * oc_l_offset);
            }
        }
        if (nb2_tail > 0) {
            apply_post_ops(m_block, nb2_tail);
            const auto oc_l_offset = nb2_tail * brg.ld_block;

            add(aux_reg_out, out_typesize_ * oc_l_offset);
            if (brg.alpha != 0) {
                add(aux_reg_in, inp_typesize_ * oc_l_offset);
                if (jcp.with_bias)
                    add(aux_reg_bias, bia_typesize_ * oc_l_offset);
                if (with_binary_per_oc_bcast_)
                    add(aux_reg_oc_l_offset_, oc_l_offset);

                add(aux_reg_scales, is_oc_scale_ * sizeof(float) * oc_l_offset);
            }
        }
        if (nb_tail > 0) {
            apply_post_ops(m_block, 1, nb_tail);

            if (brg.alpha != 0) {
                add(aux_reg_in, inp_typesize_ * (nb_tail));
                if (jcp.with_bias) add(aux_reg_bias, bia_typesize_ * (nb_tail));
                if (with_binary_per_oc_bcast_)
                    add(aux_reg_oc_l_offset_, nb_tail);
                add(aux_reg_scales, is_oc_scale_ * bia_typesize_ * (nb_tail));
            }
            add(aux_reg_out, out_typesize_ * (nb_tail));
        }
    }

    void generate() override {
        preamble();

        int nb = brg.load_dim / brg.ld_block;
        int nb_tail = brg.load_dim % brg.ld_block;

        int nb2 = nb / n_block2_;
        int nb2_tail = nb % n_block2_;
        int n_block = (nb2 == 0) ? nstl::max(1, nb2_tail) : n_block2_;

        int m_max_regs = 28 / n_block;
        int m_block = nstl::min(brg.bcast_dim, m_max_regs);

        int mb = brg.bcast_dim / m_block;
        int mb_tail = brg.bcast_dim % m_block;

        const auto full_mask = size_t {0xffffffffffffffff};
        const auto tail_mask = size_t((1 << nb_tail) - 1);

        reg64_t reg_mask = rax;

        mov(reg_mask, full_mask);
        kmovq(k_full_mask, reg_mask);
        mov(reg_mask, tail_mask);
        kmovq(k_tail_mask, reg_mask);

        if (brg.alpha != 0) {
            mov(reg_in, ptr[param1 + GET_OFF(ptr_in)]);
            mov(reg_scales, ptr[param1 + GET_OFF(ptr_scales)]);

            if (jcp.with_bias) mov(reg_bias, ptr[param1 + GET_OFF(ptr_bias)]);
            if (with_binary_per_oc_bcast_)
                mov(reg_oc_l_offset_, ptr[param1 + GET_OFF(oc_l_offset)]);
        }
        mov(reg_out, ptr[param1 + GET_OFF(ptr_out)]);

        // brg.alpha == 0 means no read from input, no bias, no eltwise - just
        // initialize registers by zero
        // brg.beta == 0 means no sum - just registers write to output
        if (brg.alpha == 0) {
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                auto zmm = Xbyak::Zmm(m * n_block + n);
                vpxord(zmm, zmm, zmm);
            }
        }

        for (int mb_ = 0; mb_ < mb; mb_++) {
            loop_by_N(m_block, nb2, nb2_tail, nb_tail);

            if (brg.alpha != 0)
                add(reg_in, inp_typesize_ * (m_block * brg.LDC));
            add(reg_out, out_typesize_ * (m_block * LDD_));
        }
        if (mb_tail > 0) loop_by_N(mb_tail, nb2, nb2_tail, nb_tail);

        postamble();

        if (brg.alpha != 0 && jcp.with_eltwise)
            postops_injector_->prepare_table();
    }
};

#undef GET_OFF

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
