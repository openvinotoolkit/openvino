// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/visibility.hpp"

#if defined(OPENVINO_ARCH_ARM64)

#include "nodes/kernels/aarch64/jit_uni_topk_kernel.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_label.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#include "nodes/kernels/aarch64/sve_utils.hpp"

#include "openvino/core/type/float16.hpp"
#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu::node {

#define GET_OFF(field) offsetof(jit_topk_call_args, field)

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
struct jit_uni_topk_kernel_aarch64 : public jit_uni_topk_kernel, public dnnl::impl::cpu::aarch64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_topk_kernel_aarch64)

    explicit jit_uni_topk_kernel_aarch64(jit_topk_config_params jcp)
        : jit_uni_topk_kernel(jcp),
          jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = ov::intel_cpu::jit_kernel_cast<decltype(ker_)>(jit_ker());
    }

    void generate() override {
        preamble();
        using namespace Xbyak_aarch64;

        const bool is_f32 = jcp_.precision == ov::element::f32;
        const bool is_bf16 = jcp_.precision == ov::element::bf16;
        const bool is_f16 = jcp_.precision == ov::element::f16 || (jcp_.data_size == 2 && !is_bf16);
        const bool is_i32 = jcp_.precision == ov::element::i32;
        const bool is_fp = is_f32 || is_f16;
        const bool top1 = jcp_.top_k == 1;
        const bool planar_layout = jcp_.layout != TopKLayoutType::topk_blocked;
        const bool blocked_innermost =
            jcp_.layout == TopKLayoutType::topk_blocked && jcp_.topk_innermost;
        const bool can_use_top1 = top1 && (is_fp || is_i32) && planar_layout;
        const bool can_use_top1_blocked = top1 && (is_fp || is_i32) && blocked_innermost;

        using namespace dnnl::impl::cpu::aarch64;
        Label fallback;
        Label done;

        if (can_use_top1) {
            XReg reg_params = abi_param1;
            XReg reg_topk_rt = x7;
            XReg reg_src = x8;
            XReg reg_dst = x9;
            XReg reg_dst_idx = x10;
            XReg reg_work_amount = x11;
            XReg reg_axis_dim = x12;
            XReg reg_sort_stride = x13;
            XReg reg_stride_bytes = x14;
            XReg reg_ptr = x15;

            WReg w_lane = w16;
            WReg w_i = w17;
            WReg w_end = w18;
            WReg w_idx = w19;
            WReg w_best_idx = w20;
            WReg w_val_scalar = w4;
            WReg w_best_val = w5;

            VReg4S v_best_val(0);
            VReg4S v_best_idx(1);
            VReg4S v_val(2);
            VReg4S v_idx(3);
            VReg4S v_mask(4);
            VReg4S v_eq(5);
            VReg4S v_idx_lt(6);
            VReg4S v_idx_step(7);
            VReg4S v_tmp(8);

            auto init_idx_step = [&]() {
                movi(v_idx_step, 0);
                mov(W_TMP_0, 1);
                ins(VReg4S(v_idx_step.getIdx())[1], W_TMP_0);
                mov(W_TMP_0, 2);
                ins(VReg4S(v_idx_step.getIdx())[2], W_TMP_0);
                mov(W_TMP_0, 3);
                ins(VReg4S(v_idx_step.getIdx())[3], W_TMP_0);
            };

            auto emit_cmp_select = [&](const VReg4S& dst_val,
                                       const VReg4S& dst_idx,
                                       const VReg4S& src_val,
                                       const VReg4S& src_idx) {
                if (is_fp) {
                    if (jcp_.mode_max) {
                        fcmgt(v_mask, src_val, dst_val);
                        fcmeq(v_eq, src_val, dst_val);
                        cmgt(v_idx_lt, dst_idx, src_idx);
                        and_(VReg16B(v_eq.getIdx()), VReg16B(v_eq.getIdx()), VReg16B(v_idx_lt.getIdx()));
                        orr(VReg16B(v_mask.getIdx()), VReg16B(v_mask.getIdx()), VReg16B(v_eq.getIdx()));
                    } else {
                        fcmgt(v_mask, dst_val, src_val);
                        fcmeq(v_eq, src_val, dst_val);
                        cmgt(v_idx_lt, dst_idx, src_idx);
                        and_(VReg16B(v_eq.getIdx()), VReg16B(v_eq.getIdx()), VReg16B(v_idx_lt.getIdx()));
                        orr(VReg16B(v_mask.getIdx()), VReg16B(v_mask.getIdx()), VReg16B(v_eq.getIdx()));
                        fcmeq(v_tmp, dst_val, dst_val);
                        fcmeq(v_idx_lt, src_val, src_val);
                        and_(VReg16B(v_tmp.getIdx()), VReg16B(v_tmp.getIdx()), VReg16B(v_idx_lt.getIdx()));
                        mvn(VReg16B(v_tmp.getIdx()), VReg16B(v_tmp.getIdx()));
                        orr(VReg16B(v_mask.getIdx()), VReg16B(v_mask.getIdx()), VReg16B(v_tmp.getIdx()));
                    }
                } else {
                    if (jcp_.mode_max) {
                        cmgt(v_mask, src_val, dst_val);
                    } else {
                        cmgt(v_mask, dst_val, src_val);
                    }
                    cmeq(v_eq, src_val, dst_val);
                    cmgt(v_idx_lt, dst_idx, src_idx);
                    and_(VReg16B(v_eq.getIdx()), VReg16B(v_eq.getIdx()), VReg16B(v_idx_lt.getIdx()));
                    orr(VReg16B(v_mask.getIdx()), VReg16B(v_mask.getIdx()), VReg16B(v_eq.getIdx()));
                }

                bit(VReg16B(dst_val.getIdx()), VReg16B(src_val.getIdx()), VReg16B(v_mask.getIdx()));
                bit(VReg16B(dst_idx.getIdx()), VReg16B(src_idx.getIdx()), VReg16B(v_mask.getIdx()));
            };

            auto emit_reduce = [&]() {
                ext(VReg16B(v_val.getIdx()), VReg16B(v_best_val.getIdx()), VReg16B(v_best_val.getIdx()), 8);
                ext(VReg16B(v_idx.getIdx()), VReg16B(v_best_idx.getIdx()), VReg16B(v_best_idx.getIdx()), 8);
                emit_cmp_select(v_best_val, v_best_idx, v_val, v_idx);

                ext(VReg16B(v_val.getIdx()), VReg16B(v_best_val.getIdx()), VReg16B(v_best_val.getIdx()), 4);
                ext(VReg16B(v_idx.getIdx()), VReg16B(v_best_idx.getIdx()), VReg16B(v_best_idx.getIdx()), 4);
                emit_cmp_select(v_best_val, v_best_idx, v_val, v_idx);
            };

            auto emit_scalar_tail_f32 = [&](const WReg& w_axis_lim, const XReg& reg_stride) {
                Label tail_loop;
                Label tail_end;
                Label tail_update;
                Label tail_next;

                mov(w_idx, w_i);
                L(tail_loop);
                cmp(w_idx, w_axis_lim);
                bge(tail_end);

                ldr(SReg(v_val.getIdx()), ptr(reg_ptr));
                fcmp(SReg(v_val.getIdx()), SReg(v_best_val.getIdx()));
                if (jcp_.mode_max) {
                    b(VS, tail_next);
                    bgt(tail_update);
                } else {
                    b(VS, tail_update);
                    blt(tail_update);
                }
                b(tail_next);

                L(tail_update);
                fmov(SReg(v_best_val.getIdx()), SReg(v_val.getIdx()));
                mov(w_best_idx, w_idx);

                L(tail_next);
                add(reg_ptr, reg_ptr, reg_stride);
                add(w_idx, w_idx, 1);
                b(tail_loop);

                L(tail_end);
            };

            auto emit_scalar_tail_i32 = [&](const WReg& w_axis_lim, const XReg& reg_stride) {
                Label tail_loop;
                Label tail_end;
                Label tail_update;
                Label tail_next;

                mov(w_idx, w_i);
                L(tail_loop);
                cmp(w_idx, w_axis_lim);
                bge(tail_end);

                ldr(w_val_scalar, ptr(reg_ptr));
                cmp(w_val_scalar, w_best_val);
                if (jcp_.mode_max) {
                    bgt(tail_update);
                } else {
                    blt(tail_update);
                }
                b(tail_next);

                L(tail_update);
                mov(w_best_val, w_val_scalar);
                mov(w_best_idx, w_idx);

                L(tail_next);
                add(reg_ptr, reg_ptr, reg_stride);
                add(w_idx, w_idx, 1);
                b(tail_loop);

                L(tail_end);
            };

            auto emit_scalar_tail_f16 = [&](const WReg& w_axis_lim, const XReg& reg_stride) {
                Label tail_loop;
                Label tail_end;
                Label tail_update;
                Label tail_next;

                HReg h_val(0);
                SReg s_val(v_val.getIdx());
                SReg s_best(v_best_val.getIdx());

                mov(w_idx, w_i);
                L(tail_loop);
                cmp(w_idx, w_axis_lim);
                bge(tail_end);

                ldr(h_val, ptr(reg_ptr));
                fcvt(s_val, h_val);
                fcmp(s_val, s_best);
                if (jcp_.mode_max) {
                    b(VS, tail_next);
                    bgt(tail_update);
                } else {
                    b(VS, tail_update);
                    blt(tail_update);
                }
                b(tail_next);

                L(tail_update);
                fmov(s_best, s_val);
                mov(w_best_idx, w_idx);

                L(tail_next);
                add(reg_ptr, reg_ptr, reg_stride);
                add(w_idx, w_idx, 1);
                b(tail_loop);

                L(tail_end);
            };

            ldr(reg_topk_rt, ptr(reg_params, static_cast<int32_t>(GET_OFF(top_k))));
            cmp(reg_topk_rt, 1);
            bne(fallback);

            ldr(reg_src, ptr(reg_params, static_cast<int32_t>(GET_OFF(src))));
            ldr(reg_dst, ptr(reg_params, static_cast<int32_t>(GET_OFF(dst))));
            ldr(reg_dst_idx, ptr(reg_params, static_cast<int32_t>(GET_OFF(index))));
            ldr(reg_work_amount, ptr(reg_params, static_cast<int32_t>(GET_OFF(work_amount))));
            ldr(reg_axis_dim, ptr(reg_params, static_cast<int32_t>(GET_OFF(axis_dim))));
            ldr(reg_sort_stride, ptr(reg_params, static_cast<int32_t>(GET_OFF(sort_stride))));

            cbz(reg_axis_dim, done);
            cbz(reg_work_amount, done);

            mov(X_TMP_0, static_cast<uint64_t>(jcp_.data_size));
            mul(reg_stride_bytes, reg_sort_stride, X_TMP_0);

            const int data_size = jcp_.data_size;
            const int vec_bytes = 4 * data_size;
            const int lane_shift = data_size == 1 ? 0 : (data_size == 2 ? 1 : 2);

            init_idx_step();
            if constexpr (isa == dnnl::impl::cpu::aarch64::asimd) {
                // asimd uses the common FP/I32 top1 path below
            }

            if constexpr (isa != dnnl::impl::cpu::aarch64::asimd) {
                if (!jcp_.topk_innermost && (is_fp || is_i32)) {
                    const bool use_sve2 = sve_utils::with_cpu_sve2();

                    XReg x_lane = XReg(w_lane.getIdx());
                    XReg x_vlen = X_TMP_0;
                    XReg x_ptr = reg_ptr;
                    XReg x_ptr_dst = x24;
                    XReg x_ptr_dst_idx = x25;
                    XReg x_limit = x26;
                    XReg x_tmp = X_TMP_4;
                    XReg x_tail = X_TMP_1;

                    PReg p_g = p1;
                    PReg p_cmp = p2;
                    PReg p_eq = p3;
                    PReg p_sel = p5;
                    PReg p_g_h = p6;

                    ZRegS z_best(0);
                    ZRegS z_val(1);
                    ZRegS z_best_idx(2);
                    ZRegS z_new_idx(3);
                    ZRegH z_tmp_h(4);

                    WReg w_axis_dim = WReg(reg_axis_dim.getIdx());
                    WReg w_vlen = WReg(x_vlen.getIdx());
                    WReg w_tail = WReg(x_tail.getIdx());

                    cntw(x_vlen);
                    if (is_f16) {
                        whilelt(p_g_h.h, wzr, w_vlen);
                    }
                    subs(x_limit, reg_work_amount, x_vlen);
                    mov(x_tmp, -1);
                    csel(x_limit, x_limit, x_tmp, GE);
                    mov(x_lane, 0);

                    Label sve_loop;
                    Label sve_done;
                    Label axis_loop;
                    Label axis_end;

                    L(sve_loop);
                    cmp(x_lane, reg_work_amount);
                    bge(sve_done);

                    Label full_lane;
                    Label pred_done;
                    cmp(x_lane, x_limit);
                    ble(full_lane);
                    sub(x_tail, reg_work_amount, x_lane);
                    cmp(x_tail, x_vlen);
                    csel(x_tail, x_vlen, x_tail, GT);
                    whilelt(p_g.s, wzr, w_tail);
                    if (is_f16) {
                        whilelt(p_g_h.h, wzr, w_tail);
                    }
                    b(pred_done);
                    L(full_lane);
                    ptrue(p_g.s);
                    if (is_f16) {
                        whilelt(p_g_h.h, wzr, w_vlen);
                    }
                    mov(w_tail, w_vlen);
                    L(pred_done);

                    // base pointers
                    add(x_ptr, reg_src, x_lane, LSL, lane_shift);
                    add(x_ptr_dst, reg_dst, x_lane, LSL, lane_shift);
                    add(x_ptr_dst_idx, reg_dst_idx, x_lane, LSL, 2);

                    if (is_f16) {
                        ld1h(z_tmp_h, p_g_h / T_z, ptr(x_ptr));
                        zip1(z_tmp_h, z_tmp_h, z_tmp_h);
                        fcvt(z_best, p_g / T_z, z_tmp_h);
                    } else {
                        ld1w(z_best, p_g / T_z, ptr(x_ptr));
                    }
                    dup(z_best_idx, 0);
                    add(x_ptr, x_ptr, reg_stride_bytes);

                    auto emit_axis_iter = [&]() {
                        if (is_f16) {
                            ld1h(z_tmp_h, p_g_h / T_z, ptr(x_ptr));
                            zip1(z_tmp_h, z_tmp_h, z_tmp_h);
                            fcvt(z_val, p_g / T_z, z_tmp_h);
                        } else {
                            ld1w(z_val, p_g / T_z, ptr(x_ptr));
                        }
                        dup(z_new_idx, w_i);
                        if (is_fp) {
                            if (jcp_.mode_max) {
                                fcmgt(p_cmp.s, p_g / T_z, z_val, z_best);
                            } else {
                                fcmgt(p_cmp.s, p_g / T_z, z_best, z_val);
                                fcmne(p_eq.s, p_g / T_z, z_best, z_best);
                                fcmne(p_sel.s, p_g / T_z, z_val, z_val);
                                orr(p_eq.b, p_g, p_eq.b, p_sel.b);
                                orr(p_cmp.b, p_g, p_cmp.b, p_eq.b);
                            }
                        } else {
                            if (jcp_.mode_max) {
                                cmpgt(p_cmp.s, p_g / T_z, z_val, z_best);
                            } else {
                                cmpgt(p_cmp.s, p_g / T_z, z_best, z_val);
                            }
                        }
                        sel(z_best, p_cmp, z_val, z_best);
                        sel(z_best_idx, p_cmp, z_new_idx, z_best_idx);

                        add(x_ptr, x_ptr, reg_stride_bytes);
                        add(w_i, w_i, 1);
                    };

                    mov(w_i, 1);
                    mov(w_end, use_sve2 ? 8 : 4);
                    Label axis_tail;
                    Label axis_done;
                    cmp(w_axis_dim, w_end);
                    ble(axis_tail);
                    sub(w_end, w_axis_dim, w_end);
                    L(axis_loop);
                    cmp(w_i, w_end);
                    bgt(axis_tail);
                    if (use_sve2) {
                        for (int unroll = 0; unroll < 8; ++unroll) {
                            emit_axis_iter();
                        }
                    } else {
                        for (int unroll = 0; unroll < 4; ++unroll) {
                            emit_axis_iter();
                        }
                    }
                    b(axis_loop);
                    L(axis_tail);
                    cmp(w_i, w_axis_dim);
                    bge(axis_done);
                    if (is_f16) {
                        ld1h(z_tmp_h, p_g_h / T_z, ptr(x_ptr));
                        zip1(z_tmp_h, z_tmp_h, z_tmp_h);
                        fcvt(z_val, p_g / T_z, z_tmp_h);
                    } else {
                        ld1w(z_val, p_g / T_z, ptr(x_ptr));
                    }
                    dup(z_new_idx, w_i);
                    if (is_fp) {
                        if (jcp_.mode_max) {
                            fcmgt(p_cmp.s, p_g / T_z, z_val, z_best);
                        } else {
                            fcmgt(p_cmp.s, p_g / T_z, z_best, z_val);
                            fcmne(p_eq.s, p_g / T_z, z_best, z_best);
                            fcmne(p_sel.s, p_g / T_z, z_val, z_val);
                            orr(p_eq.b, p_g, p_eq.b, p_sel.b);
                            orr(p_cmp.b, p_g, p_cmp.b, p_eq.b);
                        }
                    } else {
                        if (jcp_.mode_max) {
                            cmpgt(p_cmp.s, p_g / T_z, z_val, z_best);
                        } else {
                            cmpgt(p_cmp.s, p_g / T_z, z_best, z_val);
                        }
                    }
                    sel(z_best, p_cmp, z_val, z_best);
                    sel(z_best_idx, p_cmp, z_new_idx, z_best_idx);
                    add(x_ptr, x_ptr, reg_stride_bytes);
                    add(w_i, w_i, 1);
                    cmp(w_i, w_axis_dim);
                    blt(axis_tail);
                    L(axis_done);

                    if (is_f16) {
                        WReg w_tail_h = W_TMP_1;
                        lsl(w_tail_h, w_tail, 1);
                        whilelt(p_sel.h, wzr, w_tail_h);
                        fcvt(z_tmp_h, p_sel / T_z, z_best);
                        uzp1(z_tmp_h, z_tmp_h, z_tmp_h);
                        st1h(z_tmp_h, p_g_h, ptr(x_ptr_dst));
                    } else {
                        st1w(z_best, p_g, ptr(x_ptr_dst));
                    }
                    st1w(z_best_idx, p_g, ptr(x_ptr_dst_idx));

                    add(x_lane, x_lane, x_vlen);
                    b(sve_loop);

                    L(sve_done);
                    b(done);
                }
            }

            if constexpr (isa != dnnl::impl::cpu::aarch64::asimd) {
                if (jcp_.topk_innermost && (is_fp || is_i32)) {
                    const bool use_sve2 = sve_utils::with_cpu_sve2();

                    XReg x_lane = XReg(w_lane.getIdx());
                    XReg x_vlen = X_TMP_0;
                    XReg x_ptr = reg_ptr;
                    XReg x_ptr_dst = X_TMP_4;
                    XReg x_step = x26;

                    PReg p_g = p1;
                    PReg p_cmp = p2;
                    PReg p_eq = p3;
                    PReg p_sel = p5;
                    PReg p_all = p0;
                    PReg p_g_h = p6;

                    ZRegS z_best(0);
                    ZRegS z_val(1);
                    ZRegS z_best_idx(2);
                    ZRegS z_new_idx(3);
                    ZRegH z_tmp_h(4);
                    ZRegS z_idx_step(5);
                    SReg s_red(6);
                    SReg s_tmp(7);

                    WReg w_axis_dim = WReg(reg_axis_dim.getIdx());
                    WReg w_vlen = WReg(x_vlen.getIdx());
                    WReg w_tail = W_TMP_0;

                    Label lane_loop;
                    Label lane_end;
                    Label axis_loop;
                    Label axis_end;

                    cntw(x_vlen);
                    udiv(w_end, w_axis_dim, w_vlen);
                    msub(w_tail, w_end, w_vlen, w_axis_dim);
                    mul(w_end, w_end, w_vlen);
                    mov(x_lane, 0);
                    index(z_idx_step, 0, 1);
                    mov(x_step, x_vlen);
                    lsl(x_step, x_step, lane_shift);

                    L(lane_loop);
                    cmp(x_lane, reg_work_amount);
                    bge(lane_end);

                    add(x_ptr, reg_src, x_lane, LSL, lane_shift);

                    Label scalar_init;
                    Label scalar_loop;
                    Label scalar_update;
                    Label scalar_next;
                    Label store_best;

                    cbz(w_end, scalar_init);

                    mov(w_i, 0);
                    ptrue(p_g.s);
                    if (is_f16) {
                        whilelt(p_g_h.h, wzr, w_vlen);
                    }
                    if (is_f16) {
                        ld1h(z_tmp_h, p_g_h / T_z, ptr(x_ptr));
                        zip1(z_tmp_h, z_tmp_h, z_tmp_h);
                        fcvt(z_best, p_g / T_z, z_tmp_h);
                    } else {
                        ld1w(z_best, p_g / T_z, ptr(x_ptr));
                    }
                    dup(z_best_idx, w_i);
                    add(z_best_idx, z_best_idx, z_idx_step);
                    add(x_ptr, x_ptr, x_step);
                    add(w_i, w_i, w_vlen);

                    auto emit_axis_iter = [&]() {
                        if (is_f16) {
                            ld1h(z_tmp_h, p_g_h / T_z, ptr(x_ptr));
                            zip1(z_tmp_h, z_tmp_h, z_tmp_h);
                            fcvt(z_val, p_g / T_z, z_tmp_h);
                        } else {
                            ld1w(z_val, p_g / T_z, ptr(x_ptr));
                        }
                        dup(z_new_idx, w_i);
                        add(z_new_idx, z_new_idx, z_idx_step);
                        if (is_fp) {
                            if (jcp_.mode_max) {
                                fcmgt(p_cmp.s, p_g / T_z, z_val, z_best);
                            } else {
                                fcmgt(p_cmp.s, p_g / T_z, z_best, z_val);
                                fcmne(p_eq.s, p_g / T_z, z_best, z_best);
                                fcmne(p_sel.s, p_g / T_z, z_val, z_val);
                                orr(p_eq.b, p_g, p_eq.b, p_sel.b);
                                orr(p_cmp.b, p_g, p_cmp.b, p_eq.b);
                            }
                        } else {
                            if (jcp_.mode_max) {
                                cmpgt(p_cmp.s, p_g / T_z, z_val, z_best);
                            } else {
                                cmpgt(p_cmp.s, p_g / T_z, z_best, z_val);
                            }
                        }
                        sel(z_best, p_cmp, z_val, z_best);
                        sel(z_best_idx, p_cmp, z_new_idx, z_best_idx);

                        add(x_ptr, x_ptr, x_step);
                        add(w_i, w_i, w_vlen);

                        cmp(w_i, w_end);
                        bge(axis_end);
                    };

                    L(axis_loop);
                    cmp(w_i, w_end);
                    bge(axis_end);
                    if (use_sve2) {
                        for (int unroll = 0; unroll < 8; ++unroll) {
                            emit_axis_iter();
                        }
                    } else {
                        for (int unroll = 0; unroll < 4; ++unroll) {
                            emit_axis_iter();
                        }
                    }
                    b(axis_loop);
                    L(axis_end);

                    ptrue(p_all.s);
                    if (is_fp) {
                        XReg x_stack_tmp = X_TMP_1;
                        XReg x_stack_vals = X_TMP_2;
                        XReg x_stack_idx = X_TMP_3;
                        XReg x_red_ptr = X_TMP_4;
                        WReg w_idx_tmp = W_TMP_1;
                        Label reduce_loop;
                        Label reduce_done;
                        Label reduce_update;
                        Label reduce_skip;

                        mov(x_stack_tmp, x_vlen);
                        lsl(x_stack_tmp, x_stack_tmp, 2);  // bytes per vector (vlen * sizeof(float))
                        sub(sp, sp, x_stack_tmp, LSL, 1);  // space for vals + idx
                        mov(x_stack_vals, sp);
                        add(x_stack_idx, x_stack_vals, x_stack_tmp);

                        st1w(z_best, p_all, ptr(x_stack_vals));
                        st1w(z_best_idx, p_all, ptr(x_stack_idx));

                        mov(w_i, 0);
                        ldr(s_red, ptr(x_stack_vals));
                        ldr(w_best_idx, ptr(x_stack_idx));
                        add(w_i, w_i, 1);

                        L(reduce_loop);
                        cmp(w_i, w_vlen);
                        bge(reduce_done);
                        add(x_red_ptr, x_stack_vals, w_i, UXTW, 2);
                        ldr(s_tmp, ptr(x_red_ptr));
                        add(x_red_ptr, x_stack_idx, w_i, UXTW, 2);
                        ldr(w_idx_tmp, ptr(x_red_ptr));
                        fcmp(s_tmp, s_red);
                        if (jcp_.mode_max) {
                            b(VS, reduce_skip);
                            bgt(reduce_update);
                        } else {
                            b(VS, reduce_update);
                            blt(reduce_update);
                        }
                        b(reduce_skip);

                        L(reduce_update);
                        fmov(s_red, s_tmp);
                        mov(w_best_idx, w_idx_tmp);

                        L(reduce_skip);
                        add(w_i, w_i, 1);
                        b(reduce_loop);

                        L(reduce_done);

                        add(sp, sp, x_stack_tmp, LSL, 1);
                    } else {
                        if (jcp_.mode_max) {
                            smaxv(s_red, p_all, z_best);
                        } else {
                            sminv(s_red, p_all, z_best);
                        }
                        fmov(w_best_val, s_red);
                        dup(z_val, w_best_val);
                        cmpeq(p_eq.s, p_all / T_z, z_best, z_val);
                        pfirst(p_eq.b, p_eq.b);
                        lastb(w_best_idx, p_eq, z_best_idx);
                    }

                    mov(w_i, w_end);

                    cbz(w_tail, store_best);
                    b(scalar_loop);

                    L(scalar_init);
                    mov(w_i, 0);
                    if (is_f16) {
                        ldr(HReg(v_val.getIdx()), ptr(x_ptr));
                        fcvt(s_red, HReg(v_val.getIdx()));
                    } else if (is_f32) {
                        ldr(s_red, ptr(x_ptr));
                    } else {
                        ldr(w_best_val, ptr(x_ptr));
                    }
                    mov(w_best_idx, w_i);
                    add(x_ptr, x_ptr, (1 << lane_shift));
                    add(w_i, w_i, 1);

                    L(scalar_loop);
                    cmp(w_i, w_axis_dim);
                    bge(store_best);
                    if (is_fp) {
                        if (is_f16) {
                            ldr(HReg(v_val.getIdx()), ptr(x_ptr));
                            fcvt(s_tmp, HReg(v_val.getIdx()));
                        } else {
                            ldr(s_tmp, ptr(x_ptr));
                        }
                        fcmp(s_tmp, s_red);
                        if (jcp_.mode_max) {
                            b(VS, scalar_next);
                            bgt(scalar_update);
                        } else {
                            b(VS, scalar_update);
                            blt(scalar_update);
                        }
                        b(scalar_next);
                    } else {
                        ldr(w_val_scalar, ptr(x_ptr));
                        cmp(w_val_scalar, w_best_val);
                        if (jcp_.mode_max) {
                            bgt(scalar_update);
                        } else {
                            blt(scalar_update);
                        }
                        b(scalar_next);
                    }
                    L(scalar_update);
                    if (is_fp) {
                        fmov(s_red, s_tmp);
                    } else {
                        mov(w_best_val, w_val_scalar);
                    }
                    mov(w_best_idx, w_i);
                    L(scalar_next);
                    add(x_ptr, x_ptr, (1 << lane_shift));
                    add(w_i, w_i, 1);
                    b(scalar_loop);

                    L(store_best);

                    add(x_ptr_dst, reg_dst, x_lane, LSL, lane_shift);
                    add(x_ptr, reg_dst_idx, x_lane, LSL, 2);
                    if (is_f32) {
                        str(s_red, ptr(x_ptr_dst));
                    } else if (is_f16) {
                        fcvt(HReg(v_best_val.getIdx()), s_red);
                        str(HReg(v_best_val.getIdx()), ptr(x_ptr_dst));
                    } else {
                        str(w_best_val, ptr(x_ptr_dst));
                    }
                    str(w_best_idx, ptr(x_ptr));

                    add(x_lane, x_lane, 1);
                    b(lane_loop);
                    L(lane_end);

                    b(done);
                }
            }

            if (jcp_.topk_innermost) {
                mov(reg_stride_bytes, static_cast<uint64_t>(jcp_.data_size));
                mov(w_lane, 0);
                Label lane_loop;
                Label lane_end;
                L(lane_loop);
                WReg w_axis_dim = WReg(reg_axis_dim.getIdx());
                WReg w_work_amount = WReg(reg_work_amount.getIdx());
                cmp(w_lane, w_work_amount);
                bge(lane_end);

                // reg_ptr = src + lane * data_size
                mov(reg_ptr, reg_src);
                add(reg_ptr, reg_ptr, w_lane, UXTW, lane_shift);

                Label scalar_top1;
                Label after_store;
                cmp(w_axis_dim, 4);
                blt(scalar_top1);

                if (is_f16) {
                    ld1(VReg4H(v_best_val.getIdx()), post_ptr(reg_ptr, vec_bytes));
                    fcvtl(v_best_val, VReg4H(v_best_val.getIdx()));
                } else {
                    ld1(v_best_val, post_ptr(reg_ptr, vec_bytes));
                }
                orr(VReg16B(v_best_idx.getIdx()), VReg16B(v_idx_step.getIdx()), VReg16B(v_idx_step.getIdx()));
                mov(w_i, 4);
                sub(w_end, w_axis_dim, 4);

                Label vec_loop;
                Label vec_end;
                L(vec_loop);
                cmp(w_i, w_end);
                bgt(vec_end);

                if (is_f16) {
                    ld1(VReg4H(v_val.getIdx()), post_ptr(reg_ptr, vec_bytes));
                    fcvtl(v_val, VReg4H(v_val.getIdx()));
                } else {
                    ld1(v_val, post_ptr(reg_ptr, vec_bytes));
                }
                dup(v_idx, w_i);
                add(v_idx, v_idx, v_idx_step);
                emit_cmp_select(v_best_val, v_best_idx, v_val, v_idx);
                add(w_i, w_i, 4);
                cmp(w_i, w_end);
                bgt(vec_end);

                if (is_f16) {
                    ld1(VReg4H(v_val.getIdx()), post_ptr(reg_ptr, vec_bytes));
                    fcvtl(v_val, VReg4H(v_val.getIdx()));
                } else {
                    ld1(v_val, post_ptr(reg_ptr, vec_bytes));
                }
                dup(v_idx, w_i);
                add(v_idx, v_idx, v_idx_step);
                emit_cmp_select(v_best_val, v_best_idx, v_val, v_idx);
                add(w_i, w_i, 4);
                b(vec_loop);
                L(vec_end);

                if (is_i32) {
                    SReg s_red(v_val.getIdx());
                    if (jcp_.mode_max) {
                        smaxv(s_red, v_best_val);
                    } else {
                        sminv(s_red, v_best_val);
                    }
                    fmov(w_best_val, s_red);

                    dup(v_val, w_best_val);
                    cmeq(v_mask, v_best_val, v_val);
                    mov(w_val_scalar, 0x7fffffff);
                    dup(v_idx, w_val_scalar);
                    bit(VReg16B(v_idx.getIdx()), VReg16B(v_best_idx.getIdx()), VReg16B(v_mask.getIdx()));
                    uminv(s_red, v_idx);
                    fmov(w_best_idx, s_red);
                } else {
                    emit_reduce();
                    umov(w_best_idx, VReg4S(v_best_idx.getIdx())[0]);
                }
                if (is_f32) {
                    umov(w_best_val, VReg4S(v_best_val.getIdx())[0]);
                    fmov(SReg(v_best_val.getIdx()), w_best_val);
                    emit_scalar_tail_f32(w_axis_dim, reg_stride_bytes);
                    str(SReg(v_best_val.getIdx()), ptr(reg_dst));
                } else if (is_f16) {
                    umov(w_best_val, VReg4S(v_best_val.getIdx())[0]);
                    fmov(SReg(v_best_val.getIdx()), w_best_val);
                    emit_scalar_tail_f16(w_axis_dim, reg_stride_bytes);
                    fcvt(HReg(v_best_val.getIdx()), SReg(v_best_val.getIdx()));
                    str(HReg(v_best_val.getIdx()), ptr(reg_dst));
                } else {
                    emit_scalar_tail_i32(w_axis_dim, reg_stride_bytes);
                    str(w_best_val, ptr(reg_dst));
                }
                str(w_best_idx, ptr(reg_dst_idx));
                b(after_store);

                L(scalar_top1);
                mov(w_best_idx, 0);
                mov(w_i, 1);
                if (is_f32) {
                    ldr(SReg(v_best_val.getIdx()), ptr(reg_ptr));
                    add(reg_ptr, reg_ptr, reg_stride_bytes);
                    emit_scalar_tail_f32(w_axis_dim, reg_stride_bytes);
                    str(SReg(v_best_val.getIdx()), ptr(reg_dst));
                } else if (is_f16) {
                    ldr(HReg(v_best_val.getIdx()), ptr(reg_ptr));
                    fcvt(SReg(v_best_val.getIdx()), HReg(v_best_val.getIdx()));
                    add(reg_ptr, reg_ptr, reg_stride_bytes);
                    emit_scalar_tail_f16(w_axis_dim, reg_stride_bytes);
                    fcvt(HReg(v_best_val.getIdx()), SReg(v_best_val.getIdx()));
                    str(HReg(v_best_val.getIdx()), ptr(reg_dst));
                } else if (is_i32) {
                    ldr(w_best_val, ptr(reg_ptr));
                    add(reg_ptr, reg_ptr, reg_stride_bytes);
                    emit_scalar_tail_i32(w_axis_dim, reg_stride_bytes);
                    str(w_best_val, ptr(reg_dst));
                }
                str(w_best_idx, ptr(reg_dst_idx));

                L(after_store);

                add(reg_dst, reg_dst, data_size);
                add(reg_dst_idx, reg_dst_idx, 4);
                add(w_lane, w_lane, 1);
                b(lane_loop);
                L(lane_end);
            } else {
                // Vectorized across lanes.
                mov(w_lane, 0);
                Label vec_lane_loop;
                Label vec_lane_end;
                Label tail_lane_loop;
                L(vec_lane_loop);
                WReg w_axis_dim = WReg(reg_axis_dim.getIdx());
                WReg w_work_amount = WReg(reg_work_amount.getIdx());
                cmp(w_lane, w_work_amount);
                bge(tail_lane_loop);
                subs(w_end, w_work_amount, w_lane);
                cmp(w_end, 4);
                blt(tail_lane_loop);

                add(reg_ptr, reg_src, w_lane, UXTW, lane_shift);
                XReg reg_ptr_dst = x26;
                XReg reg_ptr_dst_idx = x27;
                add(reg_ptr_dst, reg_dst, w_lane, UXTW, lane_shift);
                add(reg_ptr_dst_idx, reg_dst_idx, w_lane, UXTW, 2);

                if (is_f16) {
                    ld1(VReg4H(v_best_val.getIdx()), ptr(reg_ptr));
                    fcvtl(v_best_val, VReg4H(v_best_val.getIdx()));
                } else {
                    ld1(v_best_val, ptr(reg_ptr));
                }
                mov(W_TMP_0, 0);
                dup(v_best_idx, W_TMP_0);
                add(reg_ptr, reg_ptr, reg_stride_bytes);

                Label axis_loop;
                Label axis_end;

                auto emit_axis_iter = [&]() {
                    if (is_f16) {
                        ld1(VReg4H(v_val.getIdx()), ptr(reg_ptr));
                        fcvtl(v_val, VReg4H(v_val.getIdx()));
                    } else {
                        ld1(v_val, ptr(reg_ptr));
                    }
                    emit_cmp_select(v_best_val, v_best_idx, v_val, v_idx);
                    add(reg_ptr, reg_ptr, reg_stride_bytes);
                    add(w_i, w_i, 1);
                    add(v_idx, v_idx, v_idx_step);
                    cmp(w_i, w_axis_dim);
                    bge(axis_end);
                };

                mov(w_i, 1);
                dup(v_idx, w_i);
                movi(v_idx_step, 1);
                L(axis_loop);
                cmp(w_i, w_axis_dim);
                bge(axis_end);
                for (int unroll = 0; unroll < 2; ++unroll) {
                    emit_axis_iter();
                }
                b(axis_loop);
                L(axis_end);

                if (is_f16) {
                    fcvtn(VReg4H(v_val.getIdx()), v_best_val);
                    st1(VReg4H(v_val.getIdx()), ptr(reg_ptr_dst));
                } else {
                    st1(v_best_val, ptr(reg_ptr_dst));
                }
                st1(v_best_idx, ptr(reg_ptr_dst_idx));

                add(w_lane, w_lane, 4);
                b(vec_lane_loop);

                L(tail_lane_loop);
                cmp(w_lane, w_work_amount);
                bge(vec_lane_end);

                // Scalar tail lanes.
                Label tail_lane_body;
                Label tail_lane_done;
                L(tail_lane_body);
                cmp(w_lane, w_work_amount);
                bge(tail_lane_done);

                add(reg_ptr, reg_src, w_lane, UXTW, lane_shift);
                add(reg_ptr_dst, reg_dst, w_lane, UXTW, lane_shift);
                add(reg_ptr_dst_idx, reg_dst_idx, w_lane, UXTW, 2);

                mov(w_i, 0);
                if (is_f32) {
                    ldr(SReg(v_best_val.getIdx()), ptr(reg_ptr));
                } else if (is_f16) {
                    ldr(HReg(v_best_val.getIdx()), ptr(reg_ptr));
                    fcvt(SReg(v_best_val.getIdx()), HReg(v_best_val.getIdx()));
                } else {
                    ldr(w_best_val, ptr(reg_ptr));
                }
                mov(w_best_idx, 0);
                add(reg_ptr, reg_ptr, reg_stride_bytes);
                mov(w_i, 1);
                if (is_f32) {
                    emit_scalar_tail_f32(w_axis_dim, reg_stride_bytes);
                    str(SReg(v_best_val.getIdx()), ptr(reg_ptr_dst));
                } else if (is_f16) {
                    emit_scalar_tail_f16(w_axis_dim, reg_stride_bytes);
                    fcvt(HReg(v_best_val.getIdx()), SReg(v_best_val.getIdx()));
                    str(HReg(v_best_val.getIdx()), ptr(reg_ptr_dst));
                } else {
                    emit_scalar_tail_i32(w_axis_dim, reg_stride_bytes);
                    str(w_best_val, ptr(reg_ptr_dst));
                }
                str(w_best_idx, ptr(reg_ptr_dst_idx));

                add(w_lane, w_lane, 1);
                b(tail_lane_body);
                L(tail_lane_done);

                L(vec_lane_end);
            }

            b(done);
        }

        if (can_use_top1_blocked) {
            XReg reg_params = abi_param1;
            XReg reg_topk_rt = x7;
            XReg reg_src = x8;
            XReg reg_dst = x9;
            XReg reg_dst_idx = x10;
            XReg reg_work_amount = x11;
            XReg reg_axis_dim = x12;
            XReg reg_sort_stride = x13;
            XReg reg_lane_ptr = x14;
            XReg reg_block_ptr = x15;
            XReg reg_blk_stride_bytes = x6;
            XReg reg_ptr = x7;
            XReg reg_dst_lane = x16;
            XReg reg_dst_idx_lane = x17;

            WReg w_lane = w19;
            WReg w_blocks = w20;
            WReg w_tail = w21;
            WReg w_block = w22;
            WReg w_block_base = w0;
            WReg w_offset = w1;
            WReg w_idx = w2;
            WReg w_best_idx = w3;
            WReg w_val_scalar = W_TMP_1;
            WReg w_best_val = W_TMP_2;

            VReg4S v_best_val(0);
            VReg4S v_vec_val(2);
            VReg4S v_vec_idx(3);
            VReg4S v_mask(4);
            VReg4S v_eq(5);
            VReg4S v_idx_lt(6);
            VReg4S v_idx_step(7);
            VReg4S v_tmp_val(8);
            VReg4S v_tmp_idx(9);
            VReg4S v_blk_best_val(10);
            VReg4S v_blk_best_idx(11);

            auto init_idx_step = [&]() {
                movi(v_idx_step, 0);
                mov(W_TMP_0, 1);
                ins(VReg4S(v_idx_step.getIdx())[1], W_TMP_0);
                mov(W_TMP_0, 2);
                ins(VReg4S(v_idx_step.getIdx())[2], W_TMP_0);
                mov(W_TMP_0, 3);
                ins(VReg4S(v_idx_step.getIdx())[3], W_TMP_0);
            };

            auto emit_cmp_select = [&](const VReg4S& dst_val,
                                       const VReg4S& dst_idx,
                                       const VReg4S& src_val,
                                       const VReg4S& src_idx) {
                if (is_fp) {
                    if (jcp_.mode_max) {
                        fcmgt(v_mask, src_val, dst_val);
                    } else {
                        fcmgt(v_mask, dst_val, src_val);
                        fcmeq(v_eq, dst_val, dst_val);
                        fcmeq(v_idx_lt, src_val, src_val);
                        and_(VReg16B(v_eq.getIdx()), VReg16B(v_eq.getIdx()), VReg16B(v_idx_lt.getIdx()));
                        mvn(VReg16B(v_eq.getIdx()), VReg16B(v_eq.getIdx()));
                        orr(VReg16B(v_mask.getIdx()), VReg16B(v_mask.getIdx()), VReg16B(v_eq.getIdx()));
                    }
                } else {
                    if (jcp_.mode_max) {
                        cmgt(v_mask, src_val, dst_val);
                    } else {
                        cmgt(v_mask, dst_val, src_val);
                    }
                }

                bit(VReg16B(dst_val.getIdx()), VReg16B(src_val.getIdx()), VReg16B(v_mask.getIdx()));
                bit(VReg16B(dst_idx.getIdx()), VReg16B(src_idx.getIdx()), VReg16B(v_mask.getIdx()));
            };

            auto emit_reduce = [&]() {
                ext(VReg16B(v_tmp_val.getIdx()), VReg16B(v_vec_val.getIdx()), VReg16B(v_vec_val.getIdx()), 8);
                ext(VReg16B(v_tmp_idx.getIdx()), VReg16B(v_vec_idx.getIdx()), VReg16B(v_vec_idx.getIdx()), 8);
                emit_cmp_select(v_vec_val, v_vec_idx, v_tmp_val, v_tmp_idx);

                ext(VReg16B(v_tmp_val.getIdx()), VReg16B(v_vec_val.getIdx()), VReg16B(v_vec_val.getIdx()), 4);
                ext(VReg16B(v_tmp_idx.getIdx()), VReg16B(v_vec_idx.getIdx()), VReg16B(v_vec_idx.getIdx()), 4);
                emit_cmp_select(v_vec_val, v_vec_idx, v_tmp_val, v_tmp_idx);
            };

            auto emit_scalar_update = [&]() {
                Label update;
                Label next;

                if (is_fp) {
                    fcmp(SReg(v_tmp_val.getIdx()), SReg(v_best_val.getIdx()));
                    if (jcp_.mode_max) {
                        b(VS, next);
                        bgt(update);
                    } else {
                        b(VS, update);
                        blt(update);
                    }
                    b(next);
                } else {
                    cmp(w_val_scalar, w_best_val);
                    if (jcp_.mode_max) {
                        bgt(update);
                    } else {
                        blt(update);
                    }
                    b(next);
                }

                L(update);
                if (is_fp) {
                    fmov(SReg(v_best_val.getIdx()), SReg(v_tmp_val.getIdx()));
                } else {
                    mov(w_best_val, w_val_scalar);
                }
                mov(w_best_idx, w_idx);

                L(next);
            };

            const int lane_shift = jcp_.data_size == 1 ? 0 : (jcp_.data_size == 2 ? 1 : 2);
            const int blk_shift = jcp_.blk_size == 16 ? 4 : 3;
            const int blk_mask = jcp_.blk_size - 1;
            const int lane_mul_shift = blk_shift + lane_shift;
            const int idx_lane_shift = blk_shift + 2;

            auto add_lane_ptr = [&](const XReg& dst, const XReg& base, const WReg& lane, int shift) {
                if (shift <= 4) {
                    add(dst, base, lane, UXTW, shift);
                } else {
                    XReg x_lane = XReg(lane.getIdx());
                    XReg x_tmp = X_TMP_0;
                    mov(x_tmp, x_lane);
                    lsl(x_tmp, x_tmp, shift);
                    add(dst, base, x_tmp);
                }
            };

            auto emit_offset_vec_scalar = [&]() {
                add(reg_ptr, reg_block_ptr, w_offset, UXTW, lane_shift);
                if (is_f16) {
                    ld1(VReg4H(v_vec_val.getIdx()), ptr(reg_ptr));
                    fcvtl(v_vec_val, VReg4H(v_vec_val.getIdx()));
                } else {
                    ld1(v_vec_val, ptr(reg_ptr));
                }

                add(w_idx, w_block_base, w_offset);
                dup(v_vec_idx, w_idx);
                add(v_vec_idx, v_vec_idx, v_idx_step);

                if (is_fp) {
                    emit_reduce();
                    umov(w_idx, VReg4S(v_vec_idx.getIdx())[0]);
                    umov(w_val_scalar, VReg4S(v_vec_val.getIdx())[0]);
                    fmov(SReg(v_tmp_val.getIdx()), w_val_scalar);
                    emit_scalar_update();
                } else {
                    SReg s_red(v_tmp_val.getIdx());
                    if (jcp_.mode_max) {
                        smaxv(s_red, v_vec_val);
                    } else {
                        sminv(s_red, v_vec_val);
                    }
                    fmov(w_val_scalar, s_red);
                    dup(v_tmp_val, VReg4S(v_tmp_val.getIdx())[0]);
                    cmeq(v_mask, v_vec_val, v_tmp_val);
                    mov(w_idx, 0x7fffffff);
                    dup(v_tmp_idx, w_idx);
                    bit(VReg16B(v_tmp_idx.getIdx()), VReg16B(v_vec_idx.getIdx()), VReg16B(v_mask.getIdx()));
                    uminv(s_red, v_tmp_idx);
                    fmov(w_idx, s_red);
                    emit_scalar_update();
                }
            };

            auto emit_offset_vec_init = [&]() {
                add(reg_ptr, reg_block_ptr, w_offset, UXTW, lane_shift);
                if (is_f16) {
                    ld1(VReg4H(v_blk_best_val.getIdx()), ptr(reg_ptr));
                    fcvtl(v_blk_best_val, VReg4H(v_blk_best_val.getIdx()));
                } else {
                    ld1(v_blk_best_val, ptr(reg_ptr));
                }

                add(w_idx, w_block_base, w_offset);
                dup(v_blk_best_idx, w_idx);
                add(v_blk_best_idx, v_blk_best_idx, v_idx_step);
            };

            auto emit_offset_vec_update = [&]() {
                add(reg_ptr, reg_block_ptr, w_offset, UXTW, lane_shift);
                if (is_f16) {
                    ld1(VReg4H(v_vec_val.getIdx()), ptr(reg_ptr));
                    fcvtl(v_vec_val, VReg4H(v_vec_val.getIdx()));
                } else {
                    ld1(v_vec_val, ptr(reg_ptr));
                }

                add(w_idx, w_block_base, w_offset);
                dup(v_vec_idx, w_idx);
                add(v_vec_idx, v_vec_idx, v_idx_step);

                emit_cmp_select(v_blk_best_val, v_blk_best_idx, v_vec_val, v_vec_idx);
            };

            auto emit_block_reduce_update = [&]() {
                orr(VReg16B(v_vec_val.getIdx()), VReg16B(v_blk_best_val.getIdx()), VReg16B(v_blk_best_val.getIdx()));
                orr(VReg16B(v_vec_idx.getIdx()), VReg16B(v_blk_best_idx.getIdx()), VReg16B(v_blk_best_idx.getIdx()));

                if (is_fp) {
                    emit_reduce();
                    umov(w_idx, VReg4S(v_vec_idx.getIdx())[0]);
                    umov(w_val_scalar, VReg4S(v_vec_val.getIdx())[0]);
                    fmov(SReg(v_tmp_val.getIdx()), w_val_scalar);
                    emit_scalar_update();
                } else {
                    SReg s_red(v_tmp_val.getIdx());
                    if (jcp_.mode_max) {
                        smaxv(s_red, v_vec_val);
                    } else {
                        sminv(s_red, v_vec_val);
                    }
                    fmov(w_val_scalar, s_red);
                    dup(v_tmp_val, VReg4S(v_tmp_val.getIdx())[0]);
                    cmeq(v_mask, v_vec_val, v_tmp_val);
                    mov(w_idx, 0x7fffffff);
                    dup(v_tmp_idx, w_idx);
                    bit(VReg16B(v_tmp_idx.getIdx()), VReg16B(v_vec_idx.getIdx()), VReg16B(v_mask.getIdx()));
                    uminv(s_red, v_tmp_idx);
                    fmov(w_idx, s_red);
                    emit_scalar_update();
                }
            };

            ldr(reg_topk_rt, ptr(reg_params, static_cast<int32_t>(GET_OFF(top_k))));
            cmp(reg_topk_rt, 1);
            bne(fallback);

            ldr(reg_src, ptr(reg_params, static_cast<int32_t>(GET_OFF(src))));
            ldr(reg_dst, ptr(reg_params, static_cast<int32_t>(GET_OFF(dst))));
            ldr(reg_dst_idx, ptr(reg_params, static_cast<int32_t>(GET_OFF(index))));
            ldr(reg_work_amount, ptr(reg_params, static_cast<int32_t>(GET_OFF(work_amount))));
            ldr(reg_axis_dim, ptr(reg_params, static_cast<int32_t>(GET_OFF(axis_dim))));
            ldr(reg_sort_stride, ptr(reg_params, static_cast<int32_t>(GET_OFF(sort_stride))));

            cbz(reg_axis_dim, done);
            cbz(reg_work_amount, done);

            mov(reg_blk_stride_bytes, reg_sort_stride);
            lsl(reg_blk_stride_bytes, reg_blk_stride_bytes, blk_shift);
            if (lane_shift) {
                lsl(reg_blk_stride_bytes, reg_blk_stride_bytes, lane_shift);
            }

            if (blk_shift) {
                lsr(w_blocks, WReg(reg_axis_dim.getIdx()), blk_shift);
                and_(w_tail, WReg(reg_axis_dim.getIdx()), blk_mask);
            } else {
                mov(w_blocks, WReg(reg_axis_dim.getIdx()));
                mov(w_tail, 0);
            }

            init_idx_step();

            if constexpr (isa != dnnl::impl::cpu::aarch64::asimd) {
                const bool use_sve2 = sve_utils::with_cpu_sve2();

                XReg x_vlen = X_TMP_3;
                WReg w_vlen = WReg(x_vlen.getIdx());
                XReg x_ptr_vec = X_TMP_4;

                PReg p_blk = p1;
                PReg p_eq = p2;
                PReg p_cmp = p3;
                PReg p_sel = p5;
                PReg p_blk_h = p6;

                ZRegS z_blk_val(16);
                ZRegS z_blk_idx(17);
                ZRegS z_blk_idx_max(18);
                ZRegS z_blk_sel_idx(19);
                ZRegS z_blk_scalar(20);
                ZRegS z_idx_base(21);
                ZRegH z_blk_h(22);
                ZRegS z_blk_best_val(23);
                ZRegS z_blk_best_idx(24);
                SReg s_red(6);

                Label sve_lane_loop;
                Label sve_lane_end;
                Label sve_block_loop;
                Label sve_block_end;
                Label sve_offset_loop;
                Label sve_offset_end;
                Label sve_tail_block;
                Label sve_tail_end;

                cntw(x_vlen);
                index(z_idx_base, 0, 1);
                mov(w_val_scalar, 0x7fffffff);
                dup(z_blk_idx_max, w_val_scalar);

                WReg w_blk_full = w4;
                WReg w_blk_tail = w5;
                WReg w_blk_chunks = w7;
                mov(w_blk_full, jcp_.blk_size);
                udiv(w_blk_chunks, w_blk_full, w_vlen);
                msub(w_blk_tail, w_blk_chunks, w_vlen, w_blk_full);
                sub(w_blk_full, w_blk_full, w_blk_tail);

                auto emit_sve_cmp_select = [&]() {
                    if (is_fp) {
                        if (jcp_.mode_max) {
                            fcmgt(p_cmp.s, p_blk / T_z, z_blk_val, z_blk_best_val);
                        } else {
                            fcmgt(p_cmp.s, p_blk / T_z, z_blk_best_val, z_blk_val);
                            fcmne(p_eq.s, p_blk / T_z, z_blk_best_val, z_blk_best_val);
                            fcmne(p_sel.s, p_blk / T_z, z_blk_val, z_blk_val);
                            orr(p_eq.b, p_blk, p_eq.b, p_sel.b);
                            orr(p_cmp.b, p_blk, p_cmp.b, p_eq.b);
                        }
                    } else {
                        if (jcp_.mode_max) {
                            cmpgt(p_cmp.s, p_blk / T_z, z_blk_val, z_blk_best_val);
                        } else {
                            cmpgt(p_cmp.s, p_blk / T_z, z_blk_best_val, z_blk_val);
                        }
                    }

                    sel(z_blk_best_val, p_cmp, z_blk_val, z_blk_best_val);
                    sel(z_blk_best_idx, p_cmp, z_blk_idx, z_blk_best_idx);
                };

                auto emit_sve_chunk_full_init = [&](const WReg& w_off) {
                    ptrue(p_blk.s);
                    if (is_f16) {
                        ptrue(p_blk_h.h);
                    }
                    add(x_ptr_vec, reg_block_ptr, w_off, UXTW, lane_shift);
                    if (is_f16) {
                        ld1h(z_blk_h, p_blk_h / T_z, ptr(x_ptr_vec));
                        zip1(z_blk_h, z_blk_h, z_blk_h);
                        fcvt(z_blk_best_val, p_blk / T_z, z_blk_h);
                    } else {
                        ld1w(z_blk_best_val, p_blk / T_z, ptr(x_ptr_vec));
                    }

                    add(w_idx, w_block_base, w_off);
                    mov(z_blk_best_idx, p_blk, z_idx_base);
                    dup(z_blk_scalar, w_idx);
                    add(z_blk_best_idx, z_blk_best_idx, z_blk_scalar);
                };

                auto emit_sve_chunk_full_update = [&](const WReg& w_off) {
                    ptrue(p_blk.s);
                    if (is_f16) {
                        ptrue(p_blk_h.h);
                    }
                    add(x_ptr_vec, reg_block_ptr, w_off, UXTW, lane_shift);
                    if (is_f16) {
                        ld1h(z_blk_h, p_blk_h / T_z, ptr(x_ptr_vec));
                        zip1(z_blk_h, z_blk_h, z_blk_h);
                        fcvt(z_blk_val, p_blk / T_z, z_blk_h);
                    } else {
                        ld1w(z_blk_val, p_blk / T_z, ptr(x_ptr_vec));
                    }

                    add(w_idx, w_block_base, w_off);
                    mov(z_blk_idx, p_blk, z_idx_base);
                    dup(z_blk_scalar, w_idx);
                    add(z_blk_idx, z_blk_idx, z_blk_scalar);

                    emit_sve_cmp_select();
                };

                auto emit_sve_block_reduce_update = [&]() {
                    ptrue(p_blk.s);
                    mov(w_val_scalar, 0x7fffffff);
                    dup(z_blk_idx_max, w_val_scalar);

                    if (is_fp) {
                        if (jcp_.mode_max) {
                            fmaxv(SReg(v_tmp_val.getIdx()), p_blk, z_blk_best_val);
                        } else {
                            fminv(SReg(v_tmp_val.getIdx()), p_blk, z_blk_best_val);
                        }
                        fmov(w_val_scalar, SReg(v_tmp_val.getIdx()));
                        dup(z_blk_scalar, w_val_scalar);
                        fcmeq(p_eq.s, p_blk, z_blk_best_val, z_blk_scalar);
                    } else {
                        if (jcp_.mode_max) {
                            smaxv(s_red, p_blk, z_blk_best_val);
                        } else {
                            sminv(s_red, p_blk, z_blk_best_val);
                        }
                        fmov(w_val_scalar, s_red);
                        dup(z_blk_scalar, w_val_scalar);
                        cmpeq(p_eq.s, p_blk, z_blk_best_val, z_blk_scalar);
                    }

                    sel(z_blk_sel_idx, p_eq, z_blk_best_idx, z_blk_idx_max);
                    uminv(s_red, p_blk, z_blk_sel_idx);
                    fmov(w_idx, s_red);

                    if (is_fp) {
                        emit_scalar_update();
                    } else {
                        emit_scalar_update();
                    }
                };

                auto emit_sve_chunk_scalar = [&](const WReg& w_off, bool is_tail, bool full) {
                    if (full) {
                        ptrue(p_blk.s);
                        if (is_f16) {
                            ptrue(p_blk_h.h);
                        }
                    } else {
                        if (is_tail) {
                            mov(w_idx, w_tail);
                        } else {
                            mov(w_idx, jcp_.blk_size);
                        }
                        sub(w_idx, w_idx, w_off);
                        whilelt(p_blk.s, wzr, w_idx);
                        if (is_f16) {
                            whilelt(p_blk_h.h, wzr, w_idx);
                        }
                    }

                    add(x_ptr_vec, reg_block_ptr, w_off, UXTW, lane_shift);
                    if (is_f16) {
                        ld1h(z_blk_h, p_blk_h / T_z, ptr(x_ptr_vec));
                        zip1(z_blk_h, z_blk_h, z_blk_h);
                        fcvt(z_blk_val, p_blk / T_z, z_blk_h);
                    } else {
                        ld1w(z_blk_val, p_blk / T_z, ptr(x_ptr_vec));
                    }

                    add(w_idx, w_block_base, w_off);
                    mov(z_blk_idx, p_blk, z_idx_base);
                    dup(z_blk_scalar, w_idx);
                    add(z_blk_idx, z_blk_idx, z_blk_scalar);

                    if (is_fp) {
                        if (jcp_.mode_max) {
                            fmaxv(SReg(v_tmp_val.getIdx()), p_blk, z_blk_val);
                        } else {
                            fminv(SReg(v_tmp_val.getIdx()), p_blk, z_blk_val);
                        }
                        fmov(w_val_scalar, SReg(v_tmp_val.getIdx()));
                        dup(z_blk_scalar, w_val_scalar);
                        fcmeq(p_eq.s, p_blk, z_blk_val, z_blk_scalar);
                    } else {
                        if (jcp_.mode_max) {
                            smaxv(s_red, p_blk, z_blk_val);
                        } else {
                            sminv(s_red, p_blk, z_blk_val);
                        }
                        fmov(w_val_scalar, s_red);
                        dup(z_blk_scalar, w_val_scalar);
                        cmpeq(p_eq.s, p_blk, z_blk_val, z_blk_scalar);
                    }

                    sel(z_blk_sel_idx, p_eq, z_blk_idx, z_blk_idx_max);
                    uminv(s_red, p_blk, z_blk_sel_idx);
                    fmov(w_idx, s_red);

                    if (is_fp) {
                        emit_scalar_update();
                    } else {
                        emit_scalar_update();
                    }
                };

                mov(w_lane, 0);
                L(sve_lane_loop);
                cmp(w_lane, WReg(reg_work_amount.getIdx()));
                bge(sve_lane_end);

                add_lane_ptr(reg_lane_ptr, reg_src, w_lane, lane_mul_shift);
                add_lane_ptr(reg_dst_lane, reg_dst, w_lane, lane_mul_shift);
                add_lane_ptr(reg_dst_idx_lane, reg_dst_idx, w_lane, idx_lane_shift);

                if (is_f32) {
                    ldr(SReg(v_best_val.getIdx()), ptr(reg_lane_ptr));
                } else if (is_f16) {
                    ldr(HReg(v_best_val.getIdx()), ptr(reg_lane_ptr));
                    fcvt(SReg(v_best_val.getIdx()), HReg(v_best_val.getIdx()));
                } else {
                    ldr(w_best_val, ptr(reg_lane_ptr));
                }
                mov(w_best_idx, 0);

                mov(w_block, 0);
                mov(w_block_base, 0);
                mov(reg_block_ptr, reg_lane_ptr);

                L(sve_block_loop);
                cmp(w_block, w_blocks);
                bge(sve_block_end);

                Label sve_block_full;
                Label sve_block_done;
                Label sve_block_fallback_done;

                cbz(w_blk_full, sve_block_done);

                L(sve_block_full);
                mov(w_offset, 0);
                emit_sve_chunk_full_init(w_offset);
                add(w_offset, w_offset, w_vlen);

                L(sve_offset_loop);
                cmp(w_offset, w_blk_full);
                bge(sve_offset_end);
                if (use_sve2) {
                    for (int unroll = 0; unroll < 4; ++unroll) {
                        emit_sve_chunk_full_update(w_offset);
                        add(w_offset, w_offset, w_vlen);
                        cmp(w_offset, w_blk_full);
                        bge(sve_offset_end);
                    }
                } else {
                    for (int unroll = 0; unroll < 2; ++unroll) {
                        emit_sve_chunk_full_update(w_offset);
                        add(w_offset, w_offset, w_vlen);
                        cmp(w_offset, w_blk_full);
                        bge(sve_offset_end);
                    }
                }
                b(sve_offset_loop);
                L(sve_offset_end);

                emit_sve_block_reduce_update();

                cbz(w_blk_tail, sve_block_done);
                mov(w_offset, w_blk_full);
                emit_sve_chunk_scalar(w_offset, false, false);

                L(sve_block_done);
                cbnz(w_blk_full, sve_block_fallback_done);
                mov(w_offset, 0);
                emit_sve_chunk_scalar(w_offset, false, false);
                L(sve_block_fallback_done);

                add(reg_block_ptr, reg_block_ptr, reg_blk_stride_bytes);
                add(w_block_base, w_block_base, jcp_.blk_size);
                add(w_block, w_block, 1);
                b(sve_block_loop);
                L(sve_block_end);

                cbz(w_tail, sve_tail_end);
                if (use_sve2) {
                    Label sve2_tail_loop;
                    Label sve2_tail_done;
                    Label sve2_tail_tail;
                    Label sve2_tail_partial;

                    mov(w_offset, 0);
                    L(sve2_tail_loop);
                    cmp(w_offset, w_tail);
                    bge(sve2_tail_done);

                    mov(W_TMP_0, w_offset);
                    add(W_TMP_0, W_TMP_0, w_vlen);
                    add(W_TMP_0, W_TMP_0, w_vlen);
                    add(W_TMP_0, W_TMP_0, w_vlen);
                    add(W_TMP_0, W_TMP_0, w_vlen);
                    cmp(W_TMP_0, w_tail);
                    bgt(sve2_tail_tail);

                    emit_sve_chunk_scalar(w_offset, true, true);
                    add(w_offset, w_offset, w_vlen);
                    emit_sve_chunk_scalar(w_offset, true, true);
                    add(w_offset, w_offset, w_vlen);
                    emit_sve_chunk_scalar(w_offset, true, true);
                    add(w_offset, w_offset, w_vlen);
                    emit_sve_chunk_scalar(w_offset, true, true);
                    add(w_offset, w_offset, w_vlen);
                    b(sve2_tail_loop);

                    L(sve2_tail_tail);
                    cmp(w_offset, w_tail);
                    bge(sve2_tail_done);
                    add(W_TMP_0, w_offset, w_vlen);
                    cmp(W_TMP_0, w_tail);
                    bgt(sve2_tail_partial);
                    emit_sve_chunk_scalar(w_offset, true, true);
                    mov(w_offset, W_TMP_0);
                    b(sve2_tail_tail);
                    L(sve2_tail_partial);
                    emit_sve_chunk_scalar(w_offset, true, false);
                    add(w_offset, w_offset, w_vlen);
                    b(sve2_tail_tail);
                    L(sve2_tail_done);
                    b(sve_tail_end);
                } else {
                    mov(w_offset, 0);
                    L(sve_tail_block);
                    cmp(w_offset, w_tail);
                    bge(sve_tail_end);

                    Label sve_tail_one;
                    add(w_val_scalar, w_offset, w_vlen);
                    add(w_idx, w_val_scalar, w_vlen);
                    cmp(w_idx, w_tail);
                    bgt(sve_tail_one);

                    emit_sve_chunk_scalar(w_offset, true, true);
                    mov(w_offset, w_val_scalar);
                    emit_sve_chunk_scalar(w_offset, true, true);
                    add(w_offset, w_offset, w_vlen);
                    b(sve_tail_block);
                    L(sve_tail_one);
                    Label sve_tail_partial;
                    Label sve_tail_done;
                    add(W_TMP_0, w_offset, w_vlen);
                    cmp(W_TMP_0, w_tail);
                    bgt(sve_tail_partial);
                    emit_sve_chunk_scalar(w_offset, true, true);
                    b(sve_tail_done);
                    L(sve_tail_partial);
                    emit_sve_chunk_scalar(w_offset, true, false);
                    L(sve_tail_done);

                    add(w_offset, w_offset, w_vlen);
                    b(sve_tail_block);
                }
                L(sve_tail_end);

                if (is_f32) {
                    str(SReg(v_best_val.getIdx()), ptr(reg_dst_lane));
                } else if (is_f16) {
                    fcvt(HReg(v_best_val.getIdx()), SReg(v_best_val.getIdx()));
                    str(HReg(v_best_val.getIdx()), ptr(reg_dst_lane));
                } else {
                    str(w_best_val, ptr(reg_dst_lane));
                }
                str(w_best_idx, ptr(reg_dst_idx_lane));

                add(w_lane, w_lane, 1);
                b(sve_lane_loop);
                L(sve_lane_end);

                b(done);
            }

            Label lane_loop;
            Label lane_end;
            Label block_loop;
            Label block_end;
            Label offset_loop;
            Label offset_end;
            Label tail_loop;
            Label tail_end;

            mov(w_lane, 0);
            L(lane_loop);
            cmp(w_lane, WReg(reg_work_amount.getIdx()));
            bge(lane_end);

            add_lane_ptr(reg_lane_ptr, reg_src, w_lane, lane_mul_shift);
            add_lane_ptr(reg_dst_lane, reg_dst, w_lane, lane_mul_shift);
            add_lane_ptr(reg_dst_idx_lane, reg_dst_idx, w_lane, idx_lane_shift);

            if (is_f32) {
                ldr(SReg(v_best_val.getIdx()), ptr(reg_lane_ptr));
            } else if (is_f16) {
                ldr(HReg(v_best_val.getIdx()), ptr(reg_lane_ptr));
                fcvt(SReg(v_best_val.getIdx()), HReg(v_best_val.getIdx()));
            } else {
                ldr(w_best_val, ptr(reg_lane_ptr));
            }
            mov(w_best_idx, 0);

            mov(w_block, 0);
            mov(w_block_base, 0);
            mov(reg_block_ptr, reg_lane_ptr);

            L(block_loop);
            cmp(w_block, w_blocks);
            bge(block_end);

            mov(w_offset, 0);
            emit_offset_vec_init();
            add(w_offset, w_offset, 4);
            L(offset_loop);
            cmp(w_offset, jcp_.blk_size);
            bge(offset_end);
            emit_offset_vec_update();
            add(w_offset, w_offset, 4);
            b(offset_loop);
            L(offset_end);

            emit_block_reduce_update();

            add(reg_block_ptr, reg_block_ptr, reg_blk_stride_bytes);
            add(w_block_base, w_block_base, jcp_.blk_size);
            add(w_block, w_block, 1);
            b(block_loop);
            L(block_end);

            cbz(w_tail, tail_end);
            mov(w_offset, 0);
            and_(W_TMP_0, w_tail, ~3);
            cbz(W_TMP_0, tail_loop);
            Label tail_vec_loop;
            Label tail_scalar_start;
            L(tail_vec_loop);
            cmp(w_offset, W_TMP_0);
            bge(tail_scalar_start);
            emit_offset_vec_scalar();
            add(w_offset, w_offset, 4);
            b(tail_vec_loop);
            L(tail_scalar_start);

            L(tail_loop);
            cmp(w_offset, w_tail);
            bge(tail_end);
            add(reg_ptr, reg_block_ptr, w_offset, UXTW, lane_shift);
            if (is_f32) {
                ldr(SReg(v_tmp_val.getIdx()), ptr(reg_ptr));
                fmov(w_val_scalar, SReg(v_tmp_val.getIdx()));
            } else if (is_f16) {
                ldr(HReg(v_tmp_val.getIdx()), ptr(reg_ptr));
                fcvt(SReg(v_tmp_val.getIdx()), HReg(v_tmp_val.getIdx()));
                fmov(w_val_scalar, SReg(v_tmp_val.getIdx()));
            } else if (is_i32) {
                ldr(w_val_scalar, ptr(reg_ptr));
            }

            add(w_idx, w_block_base, w_offset);
            emit_scalar_update();

            add(w_offset, w_offset, 1);
            b(tail_loop);
            L(tail_end);

            if (is_f32) {
                str(SReg(v_best_val.getIdx()), ptr(reg_dst_lane));
            } else if (is_f16) {
                fcvt(HReg(v_best_val.getIdx()), SReg(v_best_val.getIdx()));
                str(HReg(v_best_val.getIdx()), ptr(reg_dst_lane));
            } else {
                str(w_best_val, ptr(reg_dst_lane));
            }
            str(w_best_idx, ptr(reg_dst_idx_lane));

            add(w_lane, w_lane, 1);
            b(lane_loop);
            L(lane_end);

            b(done);
        }

        if (jcp_.algorithm == TopKAlgorithm::topk_bitonic_sort) {
            using namespace Xbyak_aarch64;
            const bool blocked_innermost =
                jcp_.layout == TopKLayoutType::topk_blocked && jcp_.topk_innermost;

            XReg reg_params = abi_param1;
            XReg reg_src = x8;
            XReg reg_dst = x9;
            XReg reg_dst_idx = x10;
            XReg reg_prc = x11;
            XReg reg_prc_idx = x12;
            XReg reg_work_amount = x13;
            XReg reg_axis_dim = x14;
            XReg reg_top_k = x15;
            XReg reg_sort_stride = x16;
            XReg reg_bitonic_idx = x17;
            XReg reg_bitonic_k_idx = x18;
            XReg reg_stride_bytes = x19;
            XReg reg_prc_stride = x20;
            XReg reg_blk_stride = x21;
            XReg reg_aux = x22;
            XReg reg_aux_idx = x4;

            WReg w_i = w5;
            WReg w_j = w6;
            WReg w_cnt = w7;
            WReg w_work_amount = WReg(reg_work_amount.getIdx());
            WReg w_axis_dim = WReg(reg_axis_dim.getIdx());
            WReg w_top_k = WReg(reg_top_k.getIdx());

            ldr(reg_src, ptr(reg_params, static_cast<int32_t>(GET_OFF(src))));
            ldr(reg_prc, ptr(reg_params, static_cast<int32_t>(GET_OFF(process))));
            ldr(reg_prc_idx, ptr(reg_params, static_cast<int32_t>(GET_OFF(process_index))));
            ldr(reg_dst, ptr(reg_params, static_cast<int32_t>(GET_OFF(dst))));
            ldr(reg_dst_idx, ptr(reg_params, static_cast<int32_t>(GET_OFF(index))));
            ldr(reg_work_amount, ptr(reg_params, static_cast<int32_t>(GET_OFF(work_amount))));
            ldr(reg_axis_dim, ptr(reg_params, static_cast<int32_t>(GET_OFF(axis_dim))));
            ldr(reg_top_k, ptr(reg_params, static_cast<int32_t>(GET_OFF(top_k))));
            ldr(reg_sort_stride, ptr(reg_params, static_cast<int32_t>(GET_OFF(sort_stride))));
            ldr(reg_bitonic_idx, ptr(reg_params, static_cast<int32_t>(GET_OFF(bitonic_idx_buf))));
            ldr(reg_bitonic_k_idx, ptr(reg_params, static_cast<int32_t>(GET_OFF(bitonic_k_idx_buf))));

            cbz(reg_axis_dim, done);
            cbz(reg_top_k, done);
            cbz(reg_work_amount, done);

            mov(X_TMP_0, static_cast<uint64_t>(jcp_.data_size));
            mul(reg_stride_bytes, reg_sort_stride, X_TMP_0);
            mov(X_TMP_0, 4);
            mul(reg_prc_stride, reg_sort_stride, X_TMP_0);

            if (blocked_innermost) {
                mov(X_TMP_0, static_cast<uint64_t>(jcp_.blk_size));
                mul(reg_blk_stride, reg_sort_stride, X_TMP_0);
            }

            auto emit_cmp_swap_asimd = [&](const VReg4S& v_val_l,
                                           const VReg4S& v_idx_l,
                                           const VReg4S& v_val_r,
                                           const VReg4S& v_idx_r,
                                           const VReg4S& v_mask,
                                           const VReg4S& v_tmp,
                                           const VReg4S& v_eq,
                                           bool cmp_val) {
                if (cmp_val) {
                    if (is_fp) {
                        if (jcp_.mode_max) {
                            fcmgt(v_mask, v_val_r, v_val_l);
                        } else {
                            fcmgt(v_mask, v_val_l, v_val_r);
                        }
                        fcmeq(v_eq, v_val_l, v_val_r);
                    } else {
                        if (jcp_.mode_max) {
                            cmgt(v_mask, v_val_r, v_val_l);
                        } else {
                            cmgt(v_mask, v_val_l, v_val_r);
                        }
                        cmeq(v_eq, v_val_l, v_val_r);
                    }
                    cmgt(v_tmp, v_idx_l, v_idx_r);
                    and_(VReg16B(v_eq.getIdx()), VReg16B(v_eq.getIdx()), VReg16B(v_tmp.getIdx()));
                    orr(VReg16B(v_mask.getIdx()), VReg16B(v_mask.getIdx()), VReg16B(v_eq.getIdx()));
                } else {
                    cmgt(v_mask, v_idx_l, v_idx_r);
                }

                orr(VReg16B(v_tmp.getIdx()), VReg16B(v_val_l.getIdx()), VReg16B(v_val_l.getIdx()));
                bit(VReg16B(v_val_l.getIdx()), VReg16B(v_val_r.getIdx()), VReg16B(v_mask.getIdx()));
                bit(VReg16B(v_val_r.getIdx()), VReg16B(v_tmp.getIdx()), VReg16B(v_mask.getIdx()));

                orr(VReg16B(v_tmp.getIdx()), VReg16B(v_idx_l.getIdx()), VReg16B(v_idx_l.getIdx()));
                bit(VReg16B(v_idx_l.getIdx()), VReg16B(v_idx_r.getIdx()), VReg16B(v_mask.getIdx()));
                bit(VReg16B(v_idx_r.getIdx()), VReg16B(v_tmp.getIdx()), VReg16B(v_mask.getIdx()));
            };

            if constexpr (isa != dnnl::impl::cpu::aarch64::asimd) {
                const bool use_sve2 = sve_utils::with_cpu_sve2();

                XReg x_lane = x3;
                XReg x_vlen = x4;
                XReg x_src_ptr = x24;
                XReg x_dst_ptr = x27;
                XReg x_dst_idx_ptr = x28;
                XReg x_prc_ptr = x29;
                XReg x_prc_idx_ptr = x30;

                PReg p_g = p0;
                PReg p_cmp = p1;
                PReg p_eq = p2;
                PReg p_idx = p3;
                PReg p_g_h = p6;

                ZRegS z_val_l(0);
                ZRegS z_val_r(1);
                ZRegS z_idx_l(2);
                ZRegS z_idx_r(3);
                ZRegS z_tmp(4);
                ZRegH z_tmp_h(5);
                WReg w_elt_num = WReg(X_TMP_0.getIdx());
                WReg w_elt_num_h = W_TMP_1;
                ptrue(p_g.s);
                if (is_f16) {
                    ptrue(p_g_h.h);
                }

                auto emit_cmp_swap_sve = [&](bool cmp_val) {
                    if (cmp_val) {
                        if (is_fp) {
                            if (jcp_.mode_max) {
                                fcmgt(p_cmp.s, p_g / T_z, z_val_r, z_val_l);
                            } else {
                                fcmgt(p_cmp.s, p_g / T_z, z_val_l, z_val_r);
                            }
                            fcmeq(p_eq.s, p_g / T_z, z_val_l, z_val_r);
                        } else {
                            if (jcp_.mode_max) {
                                cmpgt(p_cmp.s, p_g / T_z, z_val_r, z_val_l);
                            } else {
                                cmpgt(p_cmp.s, p_g / T_z, z_val_l, z_val_r);
                            }
                            cmpeq(p_eq.s, p_g / T_z, z_val_l, z_val_r);
                        }
                        cmpgt(p_idx.s, p_g / T_z, z_idx_l, z_idx_r);
                        and_(p_idx.b, p_g, p_eq.b, p_idx.b);
                        orr(p_cmp.b, p_g, p_cmp.b, p_idx.b);
                    } else {
                        cmpgt(p_cmp.s, p_g / T_z, z_idx_l, z_idx_r);
                    }

                    mov(z_tmp, p_g, z_val_l);
                    sel(z_val_l, p_cmp, z_val_r, z_val_l);
                    sel(z_val_r, p_cmp, z_tmp, z_val_r);

                    mov(z_tmp, p_g, z_idx_l);
                    sel(z_idx_l, p_cmp, z_idx_r, z_idx_l);
                    sel(z_idx_r, p_cmp, z_tmp, z_idx_r);
                };

                auto emit_bitonic_swap = [&](bool cmp_val) {
                    ldr(w_i, ptr(reg_aux));
                    ldr(w_j, ptr(reg_aux, 4));

                    add(x_prc_ptr, reg_prc, x_lane, LSL, 2);
                    add(x_prc_ptr, x_prc_ptr, XReg(w_i.getIdx()), UXTW, 2);
                    add(x_prc_idx_ptr, reg_prc_idx, x_lane, LSL, 2);
                    add(x_prc_idx_ptr, x_prc_idx_ptr, XReg(w_i.getIdx()), UXTW, 2);
                    ld1w(z_val_l, p_g / T_z, ptr(x_prc_ptr));
                    ld1w(z_idx_l, p_g / T_z, ptr(x_prc_idx_ptr));

                    add(x_prc_ptr, reg_prc, x_lane, LSL, 2);
                    add(x_prc_ptr, x_prc_ptr, XReg(w_j.getIdx()), UXTW, 2);
                    add(x_prc_idx_ptr, reg_prc_idx, x_lane, LSL, 2);
                    add(x_prc_idx_ptr, x_prc_idx_ptr, XReg(w_j.getIdx()), UXTW, 2);
                    ld1w(z_val_r, p_g / T_z, ptr(x_prc_ptr));
                    ld1w(z_idx_r, p_g / T_z, ptr(x_prc_idx_ptr));

                    emit_cmp_swap_sve(cmp_val);

                    add(x_prc_ptr, reg_prc, x_lane, LSL, 2);
                    add(x_prc_ptr, x_prc_ptr, XReg(w_i.getIdx()), UXTW, 2);
                    add(x_prc_idx_ptr, reg_prc_idx, x_lane, LSL, 2);
                    add(x_prc_idx_ptr, x_prc_idx_ptr, XReg(w_i.getIdx()), UXTW, 2);
                    st1w(z_val_l, p_g, ptr(x_prc_ptr));
                    st1w(z_idx_l, p_g, ptr(x_prc_idx_ptr));

                    add(x_prc_ptr, reg_prc, x_lane, LSL, 2);
                    add(x_prc_ptr, x_prc_ptr, XReg(w_j.getIdx()), UXTW, 2);
                    add(x_prc_idx_ptr, reg_prc_idx, x_lane, LSL, 2);
                    add(x_prc_idx_ptr, x_prc_idx_ptr, XReg(w_j.getIdx()), UXTW, 2);
                    st1w(z_val_r, p_g, ptr(x_prc_ptr));
                    st1w(z_idx_r, p_g, ptr(x_prc_idx_ptr));
                };

                auto emit_bitonic_sort = [&](bool cmp_val) {
                    if (cmp_val) {
                        mov(w_cnt, jcp_.bitonic_idx_cnt);
                        mov(reg_aux, reg_bitonic_idx);
                    } else {
                        mov(w_cnt, jcp_.bitonic_k_idx_cnt);
                        mov(reg_aux, reg_bitonic_k_idx);
                    }

                    Label sort_loop;
                    Label sort_done;
                    L(sort_loop);
                    cmp(w_cnt, 0);
                    beq(sort_done);
                    if (use_sve2) {
                        Label sort_tail;
                        cmp(w_cnt, 8);
                        blt(sort_tail);
                        for (int unroll = 0; unroll < 4; ++unroll) {
                            emit_bitonic_swap(cmp_val);
                            add(reg_aux, reg_aux, 8);
                            sub(w_cnt, w_cnt, 2);
                        }
                        b(sort_loop);
                        L(sort_tail);
                    }
                    emit_bitonic_swap(cmp_val);
                    add(reg_aux, reg_aux, 8);
                    sub(w_cnt, w_cnt, 2);
                    b(sort_loop);
                    L(sort_done);
                };

                auto emit_bitonic_vector = [&](const XReg& x_src_base,
                                               const XReg& x_dst_base,
                                               const XReg& x_dst_idx_base) {
                    mov(x_src_ptr, x_src_base);
                    add(x_prc_ptr, reg_prc, x_lane, LSL, 2);
                    add(x_prc_idx_ptr, reg_prc_idx, x_lane, LSL, 2);
                    mov(w_i, 0);
                    Label load_loop;
                    Label load_end;
                    L(load_loop);
                    cmp(w_i, w_axis_dim);
                    bge(load_end);

                    if (is_f16) {
                        ld1h(z_tmp_h, p_g_h / T_z, ptr(x_src_ptr));
                        zip1(z_tmp_h, z_tmp_h, z_tmp_h);
                        fcvt(z_val_l, p_g / T_z, z_tmp_h);
                    } else {
                        ld1w(z_val_l, p_g / T_z, ptr(x_src_ptr));
                    }

                    st1w(z_val_l, p_g, ptr(x_prc_ptr));
                    dup(z_idx_l, w_i);
                    st1w(z_idx_l, p_g, ptr(x_prc_idx_ptr));

                    add(x_src_ptr, x_src_ptr, reg_stride_bytes);
                    add(x_prc_ptr, x_prc_ptr, reg_prc_stride);
                    add(x_prc_idx_ptr, x_prc_idx_ptr, reg_prc_stride);
                    add(w_i, w_i, 1);
                    b(load_loop);
                    L(load_end);

                    emit_bitonic_sort(true);
                    if (jcp_.sort_index) {
                        emit_bitonic_sort(false);
                    }

                    add(x_prc_ptr, reg_prc, x_lane, LSL, 2);
                    add(x_prc_idx_ptr, reg_prc_idx, x_lane, LSL, 2);
                    mov(x_dst_ptr, x_dst_base);
                    mov(x_dst_idx_ptr, x_dst_idx_base);
                    mov(w_i, 0);
                    Label store_loop;
                    Label store_end;
                    L(store_loop);
                    cmp(w_i, w_top_k);
                    bge(store_end);

                    ld1w(z_val_l, p_g / T_z, ptr(x_prc_ptr));
                    ld1w(z_idx_l, p_g / T_z, ptr(x_prc_idx_ptr));

                    if (is_f16) {
                        whilelt(p_idx.h, wzr, w_elt_num_h);
                        fcvt(z_tmp_h, p_idx / T_z, z_val_l);
                        uzp1(z_tmp_h, z_tmp_h, z_tmp_h);
                        st1h(z_tmp_h, p_g_h, ptr(x_dst_ptr));
                    } else {
                        st1w(z_val_l, p_g, ptr(x_dst_ptr));
                    }
                    st1w(z_idx_l, p_g, ptr(x_dst_idx_ptr));

                    add(x_dst_ptr, x_dst_ptr, reg_stride_bytes);
                    add(x_dst_idx_ptr, x_dst_idx_ptr, reg_prc_stride);
                    add(x_prc_ptr, x_prc_ptr, reg_prc_stride);
                    add(x_prc_idx_ptr, x_prc_idx_ptr, reg_prc_stride);
                    add(w_i, w_i, 1);
                    b(store_loop);
                    L(store_end);
                };

                auto emit_bitonic_blk_on_channel = [&](const XReg& x_src_base,
                                                       const XReg& x_dst_base,
                                                       const XReg& x_dst_idx_base,
                                                       const WReg& w_elt_num) {
                    const int blk_shift = jcp_.blk_size == 16 ? 4 : 3;
                    const int blk_mask = jcp_.blk_size - 1;
                    const int lane_shift = jcp_.data_size == 1 ? 0 : (jcp_.data_size == 2 ? 1 : 2);

                    XReg x_block = X_TMP_0;
                    XReg x_offset = X_TMP_2;
                    XReg x_idx = X_TMP_3;
                    WReg w_val_scalar = w1;
                    WReg w_idx_scalar = w2;

                    add(x_prc_ptr, reg_prc, x_lane, LSL, 2);
                    add(x_prc_idx_ptr, reg_prc_idx, x_lane, LSL, 2);

                    mov(w_i, 0);
                    Label load_i_loop;
                    Label load_i_end;
                    L(load_i_loop);
                    cmp(w_i, w_axis_dim);
                    bge(load_i_end);

                    mov(w_j, 0);
                    Label load_j_loop;
                    Label load_j_end;
                    L(load_j_loop);
                    cmp(w_j, w_elt_num);
                    bge(load_j_end);

                    lsr(x_block, XReg(w_i.getIdx()), blk_shift);
                    and_(x_offset, XReg(w_i.getIdx()), blk_mask);
                    mul(x_idx, x_block, reg_blk_stride);
                    add(x_idx, x_idx, x_offset);
                    add(x_idx, x_idx, XReg(w_j.getIdx()), LSL, blk_shift);
                    add(x_src_ptr, x_src_base, x_idx, LSL, lane_shift);

                    if (is_f32) {
                        ldr(SReg(z_val_l.getIdx()), ptr(x_src_ptr));
                    } else if (is_f16) {
                        ldr(HReg(z_tmp_h.getIdx()), ptr(x_src_ptr));
                        fcvt(SReg(z_val_l.getIdx()), HReg(z_tmp_h.getIdx()));
                    } else if (is_i32) {
                        ldr(w_val_scalar, ptr(x_src_ptr));
                    }

                    mul(x_prc_ptr, XReg(w_i.getIdx()), reg_prc_stride);
                    add(x_prc_ptr, reg_prc, x_prc_ptr);
                    add(x_prc_ptr, x_prc_ptr, x_lane, LSL, 2);
                    add(x_prc_ptr, x_prc_ptr, XReg(w_j.getIdx()), LSL, 2);
                    if (is_f32 || is_f16) {
                        str(SReg(z_val_l.getIdx()), ptr(x_prc_ptr));
                    } else {
                        str(w_val_scalar, ptr(x_prc_ptr));
                    }

                    mul(x_prc_idx_ptr, XReg(w_i.getIdx()), reg_prc_stride);
                    add(x_prc_idx_ptr, reg_prc_idx, x_prc_idx_ptr);
                    add(x_prc_idx_ptr, x_prc_idx_ptr, x_lane, LSL, 2);
                    add(x_prc_idx_ptr, x_prc_idx_ptr, XReg(w_j.getIdx()), LSL, 2);
                    str(w_i, ptr(x_prc_idx_ptr));

                    add(w_j, w_j, 1);
                    b(load_j_loop);
                    L(load_j_end);

                    add(w_i, w_i, 1);
                    b(load_i_loop);
                    L(load_i_end);

                    emit_bitonic_sort(true);
                    if (jcp_.sort_index) {
                        emit_bitonic_sort(false);
                    }

                    mov(w_i, 0);
                    Label store_i_loop;
                    Label store_i_end;
                    L(store_i_loop);
                    cmp(w_i, w_top_k);
                    bge(store_i_end);

                    mov(w_j, 0);
                    Label store_j_loop;
                    Label store_j_end;
                    L(store_j_loop);
                    cmp(w_j, w_elt_num);
                    bge(store_j_end);

                    mul(x_prc_ptr, XReg(w_i.getIdx()), reg_prc_stride);
                    add(x_prc_ptr, reg_prc, x_prc_ptr);
                    add(x_prc_ptr, x_prc_ptr, x_lane, LSL, 2);
                    add(x_prc_ptr, x_prc_ptr, XReg(w_j.getIdx()), LSL, 2);
                    if (is_f32 || is_f16) {
                        ldr(SReg(z_val_l.getIdx()), ptr(x_prc_ptr));
                    } else {
                        ldr(w_val_scalar, ptr(x_prc_ptr));
                    }

                    mul(x_prc_idx_ptr, XReg(w_i.getIdx()), reg_prc_stride);
                    add(x_prc_idx_ptr, reg_prc_idx, x_prc_idx_ptr);
                    add(x_prc_idx_ptr, x_prc_idx_ptr, x_lane, LSL, 2);
                    add(x_prc_idx_ptr, x_prc_idx_ptr, XReg(w_j.getIdx()), LSL, 2);
                    ldr(w_idx_scalar, ptr(x_prc_idx_ptr));

                    lsr(x_block, XReg(w_i.getIdx()), blk_shift);
                    and_(x_offset, XReg(w_i.getIdx()), blk_mask);
                    mul(x_idx, x_block, reg_blk_stride);
                    add(x_idx, x_idx, x_offset);
                    add(x_idx, x_idx, XReg(w_j.getIdx()), LSL, blk_shift);
                    add(x_dst_ptr, x_dst_base, x_idx, LSL, lane_shift);
                    add(x_dst_idx_ptr, x_dst_idx_base, x_idx, LSL, 2);

                    if (is_f32) {
                        str(SReg(z_val_l.getIdx()), ptr(x_dst_ptr));
                    } else if (is_f16) {
                        fcvt(HReg(z_tmp_h.getIdx()), SReg(z_val_l.getIdx()));
                        str(HReg(z_tmp_h.getIdx()), ptr(x_dst_ptr));
                    } else {
                        str(w_val_scalar, ptr(x_dst_ptr));
                    }
                    str(w_idx_scalar, ptr(x_dst_idx_ptr));

                    add(w_j, w_j, 1);
                    b(store_j_loop);
                    L(store_j_end);

                    add(w_i, w_i, 1);
                    b(store_i_loop);
                    L(store_i_end);
                };

                cntw(x_vlen);
                mov(x_lane, 0);

                Label lane_loop;
                Label lane_end;
                L(lane_loop);
                cmp(x_lane, reg_work_amount);
                bge(lane_end);

                sub(X_TMP_0, reg_work_amount, x_lane);
                cmp(X_TMP_0, x_vlen);
                csel(w_elt_num, WReg(x_vlen.getIdx()), WReg(X_TMP_0.getIdx()), GE);
                lsl(w_elt_num_h, w_elt_num, 1);
                whilelt(p_g.s, wzr, w_elt_num);
                if (is_f16) {
                    whilelt(p_g_h.h, wzr, w_elt_num);
                }

                if (blocked_innermost) {
                    const int blk_shift = jcp_.blk_size == 16 ? 4 : 3;
                    const int lane_shift = jcp_.data_size == 1 ? 0 : (jcp_.data_size == 2 ? 1 : 2);
                    const int lane_shift_blk = blk_shift + lane_shift;
                    const int idx_shift_blk = blk_shift + 2;
                    mov(X_TMP_0, x_lane);
                    lsl(X_TMP_0, X_TMP_0, lane_shift_blk);
                    add(x_src_ptr, reg_src, X_TMP_0);
                    mov(X_TMP_0, x_lane);
                    lsl(X_TMP_0, X_TMP_0, lane_shift_blk);
                    add(x_dst_ptr, reg_dst, X_TMP_0);
                    mov(X_TMP_0, x_lane);
                    lsl(X_TMP_0, X_TMP_0, idx_shift_blk);
                    add(x_dst_idx_ptr, reg_dst_idx, X_TMP_0);

                    emit_bitonic_blk_on_channel(x_src_ptr, x_dst_ptr, x_dst_idx_ptr, w_elt_num);
                } else {
                    const int lane_shift = jcp_.data_size == 1 ? 0 : (jcp_.data_size == 2 ? 1 : 2);
                    add(x_src_ptr, reg_src, x_lane, LSL, lane_shift);
                    add(x_dst_ptr, reg_dst, x_lane, LSL, lane_shift);
                    add(x_dst_idx_ptr, reg_dst_idx, x_lane, LSL, 2);

                    emit_bitonic_vector(x_src_ptr, x_dst_ptr, x_dst_idx_ptr);
                }

                add(x_lane, x_lane, XReg(w_elt_num.getIdx()));
                b(lane_loop);
                L(lane_end);

                b(done);
            } else {
                const int vec_step = 4;
                const int lane_shift = jcp_.data_size == 1 ? 0 : (jcp_.data_size == 2 ? 1 : 2);
                const int blk_shift = jcp_.blk_size == 16 ? 4 : 3;
                const int blk_mask = jcp_.blk_size - 1;

                VReg4S v_val_l(0);
                VReg4S v_val_r(1);
                VReg4S v_idx_l(2);
                VReg4S v_idx_r(3);
                VReg4S v_mask(4);
                VReg4S v_tmp(5);
                VReg4H v_tmp_h(6);
                VReg4S v_eq(7);

                auto emit_bitonic_swap = [&](bool cmp_val) {
                    ldr(w_i, ptr(reg_aux));
                    ldr(w_j, ptr(reg_aux, 4));

                    add(reg_aux_idx, reg_prc, XReg(w_i.getIdx()), UXTW, 2);
                    ld1(v_val_l, ptr(reg_aux_idx));
                    add(reg_aux_idx, reg_prc_idx, XReg(w_i.getIdx()), UXTW, 2);
                    ld1(v_idx_l, ptr(reg_aux_idx));

                    add(reg_aux_idx, reg_prc, XReg(w_j.getIdx()), UXTW, 2);
                    ld1(v_val_r, ptr(reg_aux_idx));
                    add(reg_aux_idx, reg_prc_idx, XReg(w_j.getIdx()), UXTW, 2);
                    ld1(v_idx_r, ptr(reg_aux_idx));

                    emit_cmp_swap_asimd(v_val_l, v_idx_l, v_val_r, v_idx_r, v_mask, v_tmp, v_eq, cmp_val);

                    add(reg_aux_idx, reg_prc, XReg(w_i.getIdx()), UXTW, 2);
                    st1(v_val_l, ptr(reg_aux_idx));
                    add(reg_aux_idx, reg_prc_idx, XReg(w_i.getIdx()), UXTW, 2);
                    st1(v_idx_l, ptr(reg_aux_idx));

                    add(reg_aux_idx, reg_prc, XReg(w_j.getIdx()), UXTW, 2);
                    st1(v_val_r, ptr(reg_aux_idx));
                    add(reg_aux_idx, reg_prc_idx, XReg(w_j.getIdx()), UXTW, 2);
                    st1(v_idx_r, ptr(reg_aux_idx));
                };

                auto emit_bitonic_sort = [&](bool cmp_val) {
                    if (cmp_val) {
                        mov(w_cnt, jcp_.bitonic_idx_cnt);
                        mov(reg_aux, reg_bitonic_idx);
                    } else {
                        mov(w_cnt, jcp_.bitonic_k_idx_cnt);
                        mov(reg_aux, reg_bitonic_k_idx);
                    }

                    Label sort_loop;
                    Label sort_done;
                    L(sort_loop);
                    cmp(w_cnt, 0);
                    beq(sort_done);
                    emit_bitonic_swap(cmp_val);
                    add(reg_aux, reg_aux, 8);
                    sub(w_cnt, w_cnt, 2);
                    b(sort_loop);
                    L(sort_done);
                };

                auto emit_load_vec = [&](const XReg& x_ptr) {
                    if (is_f16) {
                        ld1(v_tmp_h, ptr(x_ptr));
                        fcvtl(v_val_l, v_tmp_h);
                    } else {
                        ld1(v_val_l, ptr(x_ptr));
                    }
                };

                auto emit_store_vec = [&](const XReg& x_ptr) {
                    if (is_f16) {
                        fcvtn(v_tmp_h, v_val_l);
                        st1(v_tmp_h, ptr(x_ptr));
                    } else {
                        st1(v_val_l, ptr(x_ptr));
                    }
                };

                auto emit_bitonic_vector = [&](const XReg& x_src_base,
                                               const XReg& x_dst_base,
                                               const XReg& x_dst_idx_base) {
                    XReg x_src_ptr = x3;
                    XReg x_dst_ptr = x4;
                    XReg x_dst_idx_ptr = x5;
                    XReg x_prc_ptr = x6;
                    XReg x_prc_idx_ptr = x7;

                    mov(x_src_ptr, x_src_base);
                    mov(x_prc_ptr, reg_prc);
                    mov(x_prc_idx_ptr, reg_prc_idx);

                    mov(w_i, 0);
                    Label load_loop;
                    Label load_end;
                    L(load_loop);
                    cmp(w_i, w_axis_dim);
                    bge(load_end);

                    emit_load_vec(x_src_ptr);
                    st1(v_val_l, ptr(x_prc_ptr));
                    dup(v_idx_l, w_i);
                    st1(v_idx_l, ptr(x_prc_idx_ptr));

                    add(x_src_ptr, x_src_ptr, reg_stride_bytes);
                    add(x_prc_ptr, x_prc_ptr, reg_prc_stride);
                    add(x_prc_idx_ptr, x_prc_idx_ptr, reg_prc_stride);
                    add(w_i, w_i, 1);
                    b(load_loop);
                    L(load_end);

                    emit_bitonic_sort(true);
                    if (jcp_.sort_index) {
                        emit_bitonic_sort(false);
                    }

                    mov(x_dst_ptr, x_dst_base);
                    mov(x_dst_idx_ptr, x_dst_idx_base);
                    mov(x_prc_ptr, reg_prc);
                    mov(x_prc_idx_ptr, reg_prc_idx);

                    mov(w_i, 0);
                    Label store_loop;
                    Label store_end;
                    L(store_loop);
                    cmp(w_i, w_top_k);
                    bge(store_end);

                    ld1(v_val_l, ptr(x_prc_ptr));
                    ld1(v_idx_l, ptr(x_prc_idx_ptr));

                    emit_store_vec(x_dst_ptr);
                    st1(v_idx_l, ptr(x_dst_idx_ptr));

                    add(x_dst_ptr, x_dst_ptr, reg_stride_bytes);
                    add(x_dst_idx_ptr, x_dst_idx_ptr, reg_prc_stride);
                    add(x_prc_ptr, x_prc_ptr, reg_prc_stride);
                    add(x_prc_idx_ptr, x_prc_idx_ptr, reg_prc_stride);
                    add(w_i, w_i, 1);
                    b(store_loop);
                    L(store_end);
                };

                auto emit_bitonic_blk_on_channel = [&](const XReg& x_src_base,
                                                       const XReg& x_dst_base,
                                                       const XReg& x_dst_idx_base,
                                                       const WReg& w_elt_num) {
                    XReg x_block = x3;
                    XReg x_offset = x4;
                    XReg x_idx = x5;
                    XReg x_ptr = x6;
                    XReg x_prc_ptr = x7;
                    XReg x_prc_idx_ptr = x27;
                    WReg w_val_scalar = w1;
                    WReg w_idx_scalar = w2;

                    mov(w_i, 0);
                    Label load_i_loop;
                    Label load_i_end;
                    L(load_i_loop);
                    cmp(w_i, w_axis_dim);
                    bge(load_i_end);

                    mov(w_j, 0);
                    Label load_j_loop;
                    Label load_j_end;
                    L(load_j_loop);
                    cmp(w_j, w_elt_num);
                    bge(load_j_end);

                    lsr(x_block, XReg(w_i.getIdx()), blk_shift);
                    and_(x_offset, XReg(w_i.getIdx()), blk_mask);
                    mul(x_idx, x_block, reg_blk_stride);
                    add(x_idx, x_idx, x_offset);
                    add(x_idx, x_idx, XReg(w_j.getIdx()), LSL, blk_shift);
                    add(x_ptr, x_src_base, x_idx, LSL, lane_shift);

                    if (is_f32) {
                        ldr(SReg(v_val_l.getIdx()), ptr(x_ptr));
                    } else if (is_f16) {
                        ldr(HReg(v_tmp_h.getIdx()), ptr(x_ptr));
                        fcvt(SReg(v_val_l.getIdx()), HReg(v_tmp_h.getIdx()));
                    } else if (is_i32) {
                        ldr(w_val_scalar, ptr(x_ptr));
                    }

                    mul(x_prc_ptr, XReg(w_i.getIdx()), reg_prc_stride);
                    add(x_prc_ptr, reg_prc, x_prc_ptr);
                    add(x_prc_ptr, x_prc_ptr, XReg(w_j.getIdx()), LSL, 2);
                    if (is_f32 || is_f16) {
                        str(SReg(v_val_l.getIdx()), ptr(x_prc_ptr));
                    } else {
                        str(w_val_scalar, ptr(x_prc_ptr));
                    }

                    mul(x_prc_idx_ptr, XReg(w_i.getIdx()), reg_prc_stride);
                    add(x_prc_idx_ptr, reg_prc_idx, x_prc_idx_ptr);
                    add(x_prc_idx_ptr, x_prc_idx_ptr, XReg(w_j.getIdx()), LSL, 2);
                    str(w_i, ptr(x_prc_idx_ptr));

                    add(w_j, w_j, 1);
                    b(load_j_loop);
                    L(load_j_end);

                    add(w_i, w_i, 1);
                    b(load_i_loop);
                    L(load_i_end);

                    emit_bitonic_sort(true);
                    if (jcp_.sort_index) {
                        emit_bitonic_sort(false);
                    }

                    mov(w_i, 0);
                    Label store_i_loop;
                    Label store_i_end;
                    L(store_i_loop);
                    cmp(w_i, w_top_k);
                    bge(store_i_end);

                    mov(w_j, 0);
                    Label store_j_loop;
                    Label store_j_end;
                    L(store_j_loop);
                    cmp(w_j, w_elt_num);
                    bge(store_j_end);

                    mul(x_prc_ptr, XReg(w_i.getIdx()), reg_prc_stride);
                    add(x_prc_ptr, reg_prc, x_prc_ptr);
                    add(x_prc_ptr, x_prc_ptr, XReg(w_j.getIdx()), LSL, 2);
                    if (is_f32 || is_f16) {
                        ldr(SReg(v_val_l.getIdx()), ptr(x_prc_ptr));
                    } else {
                        ldr(w_val_scalar, ptr(x_prc_ptr));
                    }

                    mul(x_prc_idx_ptr, XReg(w_i.getIdx()), reg_prc_stride);
                    add(x_prc_idx_ptr, reg_prc_idx, x_prc_idx_ptr);
                    add(x_prc_idx_ptr, x_prc_idx_ptr, XReg(w_j.getIdx()), LSL, 2);
                    ldr(w_idx_scalar, ptr(x_prc_idx_ptr));

                    lsr(x_block, XReg(w_i.getIdx()), blk_shift);
                    and_(x_offset, XReg(w_i.getIdx()), blk_mask);
                    mul(x_idx, x_block, reg_blk_stride);
                    add(x_idx, x_idx, x_offset);
                    add(x_idx, x_idx, XReg(w_j.getIdx()), LSL, blk_shift);
                    add(x_ptr, x_dst_base, x_idx, LSL, lane_shift);

                    if (is_f32) {
                        str(SReg(v_val_l.getIdx()), ptr(x_ptr));
                    } else if (is_f16) {
                        fcvt(HReg(v_tmp_h.getIdx()), SReg(v_val_l.getIdx()));
                        str(HReg(v_tmp_h.getIdx()), ptr(x_ptr));
                    } else {
                        str(w_val_scalar, ptr(x_ptr));
                    }
                    add(x_ptr, x_dst_idx_base, x_idx, LSL, 2);
                    str(w_idx_scalar, ptr(x_ptr));

                    add(w_j, w_j, 1);
                    b(store_j_loop);
                    L(store_j_end);

                    add(w_i, w_i, 1);
                    b(store_i_loop);
                    L(store_i_end);
                };

                WReg w_vec_step = w28;
                mov(w_vec_step, vec_step);

                Label lane_loop;
                Label lane_tail;
                Label lane_end;

                L(lane_loop);
                cmp(w_work_amount, vec_step);
                blt(lane_tail);

                if (blocked_innermost) {
                    emit_bitonic_blk_on_channel(reg_src, reg_dst, reg_dst_idx, w_vec_step);
                    add(reg_src, reg_src, vec_step * jcp_.blk_size * jcp_.data_size);
                    add(reg_dst, reg_dst, vec_step * jcp_.blk_size * jcp_.data_size);
                    add(reg_dst_idx, reg_dst_idx, vec_step * jcp_.blk_size * sizeof(int32_t));
                } else {
                    emit_bitonic_vector(reg_src, reg_dst, reg_dst_idx);
                    add(reg_src, reg_src, vec_step * jcp_.data_size);
                    add(reg_dst, reg_dst, vec_step * jcp_.data_size);
                    add(reg_dst_idx, reg_dst_idx, vec_step * sizeof(int32_t));
                }

                add(reg_prc, reg_prc, vec_step * 4);
                add(reg_prc_idx, reg_prc_idx, vec_step * sizeof(int32_t));

                sub(w_work_amount, w_work_amount, vec_step);
                b(lane_loop);

                L(lane_tail);
                cbz(w_work_amount, lane_end);

                str(reg_src, ptr(reg_params, static_cast<int32_t>(GET_OFF(src))));
                str(reg_prc, ptr(reg_params, static_cast<int32_t>(GET_OFF(process))));
                str(reg_prc_idx, ptr(reg_params, static_cast<int32_t>(GET_OFF(process_index))));
                str(reg_dst, ptr(reg_params, static_cast<int32_t>(GET_OFF(dst))));
                str(reg_dst_idx, ptr(reg_params, static_cast<int32_t>(GET_OFF(index))));
                str(reg_work_amount, ptr(reg_params, static_cast<int32_t>(GET_OFF(work_amount))));
                b(fallback);

                L(lane_end);
                b(done);
            }
        }

        L(fallback);
        {
            using namespace Xbyak_aarch64;
            const bool blocked_innermost =
                jcp_.layout == TopKLayoutType::topk_blocked && jcp_.topk_innermost;

            XReg reg_params = abi_param1;
            XReg reg_src = x8;
            XReg reg_dst = x9;
            XReg reg_dst_idx = x10;
            XReg reg_prc = x11;
            XReg reg_prc_idx = x12;
            XReg reg_work_amount = x13;
            XReg reg_axis_dim = x14;
            XReg reg_top_k = x15;
            XReg reg_sort_stride = x16;
            XReg reg_k_eff = x17;
            XReg reg_lane = x18;
            XReg reg_vals_base = x19;
            XReg reg_idx_base = x20;
            XReg reg_tmp = x6;
            XReg reg_blk_stride = x7;
            XReg reg_scratch_stride = x5;
            XReg reg_prc_stride = x21;

            WReg w_lane = WReg(reg_lane.getIdx());
            WReg w_i = w22;
            WReg w_j = w27;
            WReg w_k_eff = WReg(reg_k_eff.getIdx());
            WReg w_axis_dim = WReg(reg_axis_dim.getIdx());
            WReg w_top_k = WReg(reg_top_k.getIdx());
            WReg w_pos = w23;
            WReg w_left = w24;
            WReg w_right = w25;
            WReg w_child = w26;
            WReg w_val0 = w0;
            WReg w_val1 = w1;
            WReg w_idx0 = w2;
            WReg w_idx1 = w3;
            SReg s_val0(0);
            SReg s_val1(1);
            HReg h_val0(0);

            ldr(reg_src, ptr(reg_params, static_cast<int32_t>(GET_OFF(src))));
            ldr(reg_prc, ptr(reg_params, static_cast<int32_t>(GET_OFF(process))));
            ldr(reg_prc_idx, ptr(reg_params, static_cast<int32_t>(GET_OFF(process_index))));
            ldr(reg_dst, ptr(reg_params, static_cast<int32_t>(GET_OFF(dst))));
            ldr(reg_dst_idx, ptr(reg_params, static_cast<int32_t>(GET_OFF(index))));
            ldr(reg_work_amount, ptr(reg_params, static_cast<int32_t>(GET_OFF(work_amount))));
            ldr(reg_axis_dim, ptr(reg_params, static_cast<int32_t>(GET_OFF(axis_dim))));
            ldr(reg_top_k, ptr(reg_params, static_cast<int32_t>(GET_OFF(top_k))));
            ldr(reg_sort_stride, ptr(reg_params, static_cast<int32_t>(GET_OFF(sort_stride))));

            cbz(reg_axis_dim, done);
            cbz(reg_top_k, done);
            cbz(reg_work_amount, done);

            cmp(reg_axis_dim, reg_top_k);
            csel(reg_k_eff, reg_axis_dim, reg_top_k, LT);

            add(reg_scratch_stride, reg_top_k, 1);
            lsl(reg_scratch_stride, reg_scratch_stride, 2);

            if (blocked_innermost) {
                mov(reg_tmp, static_cast<uint64_t>(jcp_.blk_size));
                mul(reg_blk_stride, reg_sort_stride, reg_tmp);
            }

            const int lane_shift = jcp_.data_size == 1 ? 0 : (jcp_.data_size == 2 ? 1 : 2);
            const int idx_shift = 2;
            const int blk_shift = jcp_.blk_size == 16 ? 4 : 3;
            const int blk_mask = jcp_.blk_size - 1;

            auto emit_addr = [&](const XReg& reg_base,
                                 const WReg& w_axis,
                                 const WReg& w_ln,
                                 const XReg& reg_out,
                                 int elem_shift) {
                if (blocked_innermost) {
                    XReg x_axis = XReg(w_axis.getIdx());
                    XReg x_block = X_TMP_0;
                    XReg x_offset = X_TMP_1;
                    XReg x_lane_mul = X_TMP_2;
                    XReg x_idx = X_TMP_3;
                    lsr(x_block, x_axis, blk_shift);
                    and_(x_offset, x_axis, blk_mask);
                    lsl(x_lane_mul, XReg(w_ln.getIdx()), blk_shift);
                    mul(x_idx, x_block, reg_blk_stride);
                    add(x_idx, x_idx, x_offset);
                    add(x_idx, x_idx, x_lane_mul);
                    add(reg_out, reg_base, x_idx, LSL, elem_shift);
                } else {
                    XReg x_axis = XReg(w_axis.getIdx());
                    XReg x_lane = XReg(w_ln.getIdx());
                    XReg x_idx = X_TMP_1;
                    mul(x_idx, x_axis, reg_sort_stride);
                    add(x_idx, x_idx, x_lane);
                    add(reg_out, reg_base, x_idx, LSL, elem_shift);
                }
            };

            auto emit_scratch_addr = [&](const XReg& reg_base, const WReg& w_idx, const XReg& reg_out) {
                add(reg_out, reg_base, XReg(w_idx.getIdx()), LSL, 2);
            };

            auto emit_load_src = [&](const WReg& w_axis, const WReg& w_ln) {
                emit_addr(reg_src, w_axis, w_ln, reg_tmp, lane_shift);
                if (is_f32) {
                    ldr(s_val0, ptr(reg_tmp));
                } else if (is_f16) {
                    ldr(h_val0, ptr(reg_tmp));
                    fcvt(s_val0, h_val0);
                } else if (is_i32) {
                    ldr(w_val0, ptr(reg_tmp));
                }
            };

            auto emit_store_dst = [&](const WReg& w_axis, const WReg& w_ln) {
                emit_addr(reg_dst, w_axis, w_ln, reg_tmp, lane_shift);
                if (is_f32) {
                    str(s_val0, ptr(reg_tmp));
                } else if (is_f16) {
                    fcvt(h_val0, s_val0);
                    str(h_val0, ptr(reg_tmp));
                } else {
                    str(w_val0, ptr(reg_tmp));
                }
            };

            auto emit_store_idx = [&](const WReg& w_axis, const WReg& w_ln) {
                emit_addr(reg_dst_idx, w_axis, w_ln, reg_tmp, idx_shift);
                str(w_idx0, ptr(reg_tmp));
            };

            auto emit_load_scratch_val0 = [&](const XReg& reg_base, const WReg& w_idx) {
                emit_scratch_addr(reg_base, w_idx, reg_tmp);
                if (is_f32 || is_f16) {
                    ldr(s_val0, ptr(reg_tmp));
                } else {
                    ldr(w_val0, ptr(reg_tmp));
                }
            };

            auto emit_load_scratch_val1 = [&](const XReg& reg_base, const WReg& w_idx) {
                emit_scratch_addr(reg_base, w_idx, reg_tmp);
                if (is_f32 || is_f16) {
                    ldr(s_val1, ptr(reg_tmp));
                } else {
                    ldr(w_val1, ptr(reg_tmp));
                }
            };

            auto emit_store_scratch_val0 = [&](const XReg& reg_base, const WReg& w_idx) {
                emit_scratch_addr(reg_base, w_idx, reg_tmp);
                if (is_f32 || is_f16) {
                    str(s_val0, ptr(reg_tmp));
                } else {
                    str(w_val0, ptr(reg_tmp));
                }
            };

            auto emit_store_scratch_val1 = [&](const XReg& reg_base, const WReg& w_idx) {
                emit_scratch_addr(reg_base, w_idx, reg_tmp);
                if (is_f32 || is_f16) {
                    str(s_val1, ptr(reg_tmp));
                } else {
                    str(w_val1, ptr(reg_tmp));
                }
            };

            auto emit_load_scratch_idx = [&](const XReg& reg_base, const WReg& w_idx, const WReg& w_out) {
                emit_scratch_addr(reg_base, w_idx, reg_tmp);
                ldr(w_out, ptr(reg_tmp));
            };

            auto emit_store_scratch_idx = [&](const XReg& reg_base, const WReg& w_idx, const WReg& w_in) {
                emit_scratch_addr(reg_base, w_idx, reg_tmp);
                str(w_in, ptr(reg_tmp));
            };

            auto emit_better = [&](Label& l_true, Label& l_false) {
                if (is_f32 || is_f16) {
                    fcmp(s_val0, s_val1);
                    b(VS, l_false);
                    if (jcp_.mode_max) {
                        b(GT, l_true);
                    } else {
                        b(LT, l_true);
                    }
                    Label eq_done;
                    b(NE, eq_done);
                    cmp(w_idx0, w_idx1);
                    b(LT, l_true);
                    L(eq_done);
                    b(l_false);
                } else {
                    cmp(w_val0, w_val1);
                    if (jcp_.mode_max) {
                        b(GT, l_true);
                        b(l_false);
                    } else {
                        b(LT, l_true);
                        b(l_false);
                    }
                }
            };

            auto emit_heap_better = [&](Label& l_true, Label& l_false) {
                if (is_f32 || is_f16) {
                    fcmp(s_val0, s_val1);
                    b(VS, l_false);
                    if (jcp_.mode_max) {
                        b(GT, l_true);
                    } else {
                        b(LT, l_true);
                    }
                    Label eq_done;
                    b(NE, eq_done);
                    cmp(w_idx0, w_idx1);
                    b(LT, l_true);
                    L(eq_done);
                    b(l_false);
                } else {
                    cmp(w_val0, w_val1);
                    if (jcp_.mode_max) {
                        b(GT, l_true);
                        b(l_false);
                    } else {
                        b(LT, l_true);
                        b(l_false);
                    }
                }
            };

            auto emit_load_node0 = [&](const WReg& w_idx) {
                emit_load_scratch_val0(reg_vals_base, w_idx);
                emit_load_scratch_idx(reg_idx_base, w_idx, w_idx0);
            };

            auto emit_load_node1 = [&](const WReg& w_idx) {
                emit_load_scratch_val1(reg_vals_base, w_idx);
                emit_load_scratch_idx(reg_idx_base, w_idx, w_idx1);
            };

            auto emit_store_node0 = [&](const WReg& w_idx) {
                emit_store_scratch_val0(reg_vals_base, w_idx);
                emit_store_scratch_idx(reg_idx_base, w_idx, w_idx0);
            };

            auto emit_store_node1 = [&](const WReg& w_idx) {
                emit_store_scratch_val1(reg_vals_base, w_idx);
                emit_store_scratch_idx(reg_idx_base, w_idx, w_idx1);
            };

            auto emit_swap_nodes = [&](const WReg& w_idx_a, const WReg& w_idx_b) {
                emit_load_node0(w_idx_a);
                emit_load_node1(w_idx_b);
                emit_store_node1(w_idx_a);
                emit_store_node0(w_idx_b);
            };

            auto emit_heapify = [&](const WReg& w_root, const WReg& w_valid, bool cmp_val) {
                Label heap_loop;
                Label heap_done;
                Label child_done;

                mov(w_j, w_root);
                L(heap_loop);
                add(w_left, w_j, w_j);
                add(w_left, w_left, 1);
                cmp(w_left, w_valid);
                bgt(heap_done);

                add(w_right, w_left, 1);
                mov(w_child, w_left);
                cmp(w_right, w_valid);
                bgt(child_done);

                if (cmp_val) {
                    emit_load_node0(w_left);
                    emit_load_node1(w_right);
                    Label left_better;
                    Label left_not;
                    emit_heap_better(left_better, left_not);
                    L(left_better);
                    mov(w_child, w_right);
                    L(left_not);
                } else {
                    emit_load_scratch_idx(reg_idx_base, w_left, w_idx0);
                    emit_load_scratch_idx(reg_idx_base, w_right, w_idx1);
                    cmp(w_idx1, w_idx0);
                    ble(child_done);
                    mov(w_child, w_right);
                }
                L(child_done);

                emit_load_node0(w_j);
                emit_load_node1(w_child);
                if (cmp_val) {
                    Label parent_better;
                    Label parent_not;
                    emit_heap_better(parent_better, parent_not);
                    L(parent_better);
                    emit_store_node1(w_j);
                    emit_store_node0(w_child);
                    mov(w_j, w_child);
                    b(heap_loop);
                    L(parent_not);
                    b(heap_done);
                } else {
                    cmp(w_idx0, w_idx1);
                    bge(heap_done);
                    emit_store_node1(w_j);
                    emit_store_node0(w_child);
                    mov(w_j, w_child);
                    b(heap_loop);
                }
                L(heap_done);
            };

            if (jcp_.algorithm == TopKAlgorithm::topk_heap_sort) {
                auto emit_heap_build = [&](bool cmp_val) {
                    Label build_done;
                    Label build_loop;
                    mov(w_i, w_k_eff);
                    cmp(w_i, 1);
                    ble(build_done);
                    lsr(w_i, w_i, 1);
                    sub(w_i, w_i, 1);
                    mov(w_pos, w_k_eff);
                    sub(w_pos, w_pos, 1);
                    L(build_loop);
                    emit_heapify(w_i, w_pos, cmp_val);
                    cmp(w_i, 0);
                    beq(build_done);
                    sub(w_i, w_i, 1);
                    b(build_loop);
                    L(build_done);
                };

                auto emit_heap_extract = [&](bool cmp_val) {
                    Label extract_done;
                    Label extract_loop;
                    mov(w_i, w_k_eff);
                    cmp(w_i, 1);
                    ble(extract_done);
                    sub(w_i, w_i, 1);
                    L(extract_loop);
                    cmp(w_i, 0);
                    beq(extract_done);
                    mov(w_j, 0);
                    emit_swap_nodes(w_j, w_i);
                    sub(w_pos, w_i, 1);
                    emit_heapify(w_j, w_pos, cmp_val);
                    sub(w_i, w_i, 1);
                    b(extract_loop);
                    L(extract_done);
                };

                Label lane_loop;
                Label lane_end;
                Label init_loop;
                Label init_done;
                Label scan_loop;
                Label scan_done;
                Label store_loop;
                Label store_done;

                mov(reg_lane, 0);
                L(lane_loop);
                cmp(reg_lane, reg_work_amount);
                bge(lane_end);

                mul(reg_tmp, reg_lane, reg_scratch_stride);
                add(reg_vals_base, reg_prc, reg_tmp);
                add(reg_idx_base, reg_prc_idx, reg_tmp);

                // init
                mov(w_i, 0);
                L(init_loop);
                cmp(w_i, w_k_eff);
                bge(init_done);
                emit_load_src(w_i, w_lane);
                emit_store_scratch_val0(reg_vals_base, w_i);
                emit_store_scratch_idx(reg_idx_base, w_i, w_i);
                add(w_i, w_i, 1);
                b(init_loop);
                L(init_done);

                emit_heap_build(true);

                // scan rest
                mov(w_i, w_k_eff);
                L(scan_loop);
                cmp(w_i, w_axis_dim);
                bge(scan_done);
                emit_load_src(w_i, w_lane);
                mov(w_idx0, w_i);
                mov(w_j, 0);
                emit_load_scratch_val1(reg_vals_base, w_j);
                emit_load_scratch_idx(reg_idx_base, w_j, w_idx1);
                Label insert_true;
                Label insert_false;
                emit_heap_better(insert_true, insert_false);
                L(insert_false);
                add(w_i, w_i, 1);
                b(scan_loop);
                L(insert_true);
                emit_store_scratch_val0(reg_vals_base, w_j);
                emit_store_scratch_idx(reg_idx_base, w_j, w_idx0);
                mov(w_pos, w_k_eff);
                sub(w_pos, w_pos, 1);
                emit_heapify(w_j, w_pos, true);
                add(w_i, w_i, 1);
                b(scan_loop);
                L(scan_done);

                if (jcp_.sort_index) {
                    emit_heap_build(false);
                    emit_heap_extract(false);
                } else {
                    emit_heap_extract(true);
                }

                // store outputs
                mov(w_i, 0);
                L(store_loop);
                cmp(w_i, w_top_k);
                bge(store_done);
                cmp(w_i, w_k_eff);
                sub(w_pos, w_k_eff, 1);
                csel(w_pos, w_i, w_pos, LT);
                emit_load_scratch_val0(reg_vals_base, w_pos);
                emit_load_scratch_idx(reg_idx_base, w_pos, w_idx0);
                emit_store_dst(w_i, w_lane);
                emit_store_idx(w_i, w_lane);
                add(w_i, w_i, 1);
                b(store_loop);
                L(store_done);

                add(reg_lane, reg_lane, 1);
                b(lane_loop);
                L(lane_end);
            } else if (jcp_.algorithm == TopKAlgorithm::topk_bitonic_sort) {
                XReg reg_aux = X_TMP_0;
                WReg w_cnt = w_child;

                auto emit_prc_addr = [&](const XReg& reg_base, const WReg& w_axis, const XReg& reg_out) {
                    mul(reg_out, XReg(w_axis.getIdx()), reg_prc_stride);
                    add(reg_out, reg_out, reg_base);
                };

                auto emit_prc_addr_by_offset = [&](const XReg& reg_base, const WReg& w_offset, const XReg& reg_out) {
                    add(reg_out, reg_base, XReg(w_offset.getIdx()), LSL, 2);
                };

                auto emit_load_prc_val0 = [&](const WReg& w_axis) {
                    emit_prc_addr(reg_vals_base, w_axis, reg_tmp);
                    if (is_f32 || is_f16) {
                        ldr(s_val0, ptr(reg_tmp));
                    } else {
                        ldr(w_val0, ptr(reg_tmp));
                    }
                };

                auto emit_load_prc_val0_by_offset = [&](const WReg& w_offset) {
                    emit_prc_addr_by_offset(reg_vals_base, w_offset, reg_tmp);
                    if (is_f32 || is_f16) {
                        ldr(s_val0, ptr(reg_tmp));
                    } else {
                        ldr(w_val0, ptr(reg_tmp));
                    }
                };

                auto emit_load_prc_val1_by_offset = [&](const WReg& w_offset) {
                    emit_prc_addr_by_offset(reg_vals_base, w_offset, reg_tmp);
                    if (is_f32 || is_f16) {
                        ldr(s_val1, ptr(reg_tmp));
                    } else {
                        ldr(w_val1, ptr(reg_tmp));
                    }
                };

                auto emit_store_prc_val0_by_offset = [&](const WReg& w_offset) {
                    emit_prc_addr_by_offset(reg_vals_base, w_offset, reg_tmp);
                    if (is_f32 || is_f16) {
                        str(s_val0, ptr(reg_tmp));
                    } else {
                        str(w_val0, ptr(reg_tmp));
                    }
                };

                auto emit_store_prc_val1_by_offset = [&](const WReg& w_offset) {
                    emit_prc_addr_by_offset(reg_vals_base, w_offset, reg_tmp);
                    if (is_f32 || is_f16) {
                        str(s_val1, ptr(reg_tmp));
                    } else {
                        str(w_val1, ptr(reg_tmp));
                    }
                };

                auto emit_load_prc_idx0 = [&](const WReg& w_axis) {
                    emit_prc_addr(reg_idx_base, w_axis, reg_tmp);
                    ldr(w_idx0, ptr(reg_tmp));
                };

                auto emit_load_prc_idx0_by_offset = [&](const WReg& w_offset) {
                    emit_prc_addr_by_offset(reg_idx_base, w_offset, reg_tmp);
                    ldr(w_idx0, ptr(reg_tmp));
                };

                auto emit_load_prc_idx1_by_offset = [&](const WReg& w_offset) {
                    emit_prc_addr_by_offset(reg_idx_base, w_offset, reg_tmp);
                    ldr(w_idx1, ptr(reg_tmp));
                };

                auto emit_store_prc_idx0_by_offset = [&](const WReg& w_offset) {
                    emit_prc_addr_by_offset(reg_idx_base, w_offset, reg_tmp);
                    str(w_idx0, ptr(reg_tmp));
                };

                auto emit_store_prc_idx1_by_offset = [&](const WReg& w_offset) {
                    emit_prc_addr_by_offset(reg_idx_base, w_offset, reg_tmp);
                    str(w_idx1, ptr(reg_tmp));
                };

                Label lane_loop;
                Label lane_end;
                Label load_loop;
                Label load_done;
                Label sort_loop;
                Label sort_done;
                Label sort_k_loop;
                Label sort_k_done;
                Label store_loop;
                Label store_done;

                mov(reg_tmp, 4);
                mul(reg_prc_stride, reg_sort_stride, reg_tmp);

                mov(reg_lane, 0);
                L(lane_loop);
                cmp(reg_lane, reg_work_amount);
                bge(lane_end);

                add(reg_vals_base, reg_prc, reg_lane, LSL, 2);
                add(reg_idx_base, reg_prc_idx, reg_lane, LSL, 2);

                // load
                mov(w_i, 0);
                L(load_loop);
                cmp(w_i, w_axis_dim);
                bge(load_done);
                emit_load_src(w_i, w_lane);
                emit_prc_addr(reg_vals_base, w_i, reg_tmp);
                if (is_f32 || is_f16) {
                    str(s_val0, ptr(reg_tmp));
                } else {
                    str(w_val0, ptr(reg_tmp));
                }
                emit_prc_addr(reg_idx_base, w_i, reg_tmp);
                str(w_i, ptr(reg_tmp));
                add(w_i, w_i, 1);
                b(load_loop);
                L(load_done);

                // bitonic sort by value
                ldr(reg_aux, ptr(reg_params, static_cast<int32_t>(GET_OFF(bitonic_idx_buf))));
                mov(w_cnt, jcp_.bitonic_idx_cnt);
                L(sort_loop);
                cmp(w_cnt, 0);
                beq(sort_done);
                ldr(w_i, ptr(reg_aux));
                ldr(w_j, ptr(reg_aux, 4));
                emit_load_prc_val0_by_offset(w_i);
                emit_load_prc_idx0_by_offset(w_i);
                emit_load_prc_val1_by_offset(w_j);
                emit_load_prc_idx1_by_offset(w_j);
                Label no_swap;
                Label do_swap;
                emit_better(no_swap, do_swap);
                L(do_swap);
                emit_store_prc_val1_by_offset(w_i);
                emit_store_prc_idx1_by_offset(w_i);
                emit_store_prc_val0_by_offset(w_j);
                emit_store_prc_idx0_by_offset(w_j);
                L(no_swap);
                add(reg_aux, reg_aux, 8);
                sub(w_cnt, w_cnt, 2);
                b(sort_loop);
                L(sort_done);

                if (jcp_.sort_index) {
                    Label sort_k_no_swap;
                    ldr(reg_aux, ptr(reg_params, static_cast<int32_t>(GET_OFF(bitonic_k_idx_buf))));
                    mov(w_cnt, jcp_.bitonic_k_idx_cnt);
                    L(sort_k_loop);
                    cmp(w_cnt, 0);
                    beq(sort_k_done);
                    ldr(w_i, ptr(reg_aux));
                    ldr(w_j, ptr(reg_aux, 4));
                    emit_load_prc_idx0_by_offset(w_i);
                    emit_load_prc_idx1_by_offset(w_j);
                    cmp(w_idx0, w_idx1);
                    b(LE, sort_k_no_swap);
                    emit_load_prc_val0_by_offset(w_i);
                    emit_load_prc_val1_by_offset(w_j);
                    emit_store_prc_val1_by_offset(w_i);
                    emit_store_prc_idx1_by_offset(w_i);
                    emit_store_prc_val0_by_offset(w_j);
                    emit_store_prc_idx0_by_offset(w_j);
                    L(sort_k_no_swap);
                    add(reg_aux, reg_aux, 8);
                    sub(w_cnt, w_cnt, 2);
                    b(sort_k_loop);
                    L(sort_k_done);
                }

                // store outputs
                mov(w_i, 0);
                L(store_loop);
                cmp(w_i, w_top_k);
                bge(store_done);
                emit_load_prc_val0(w_i);
                emit_load_prc_idx0(w_i);
                emit_store_dst(w_i, w_lane);
                emit_store_idx(w_i, w_lane);
                add(w_i, w_i, 1);
                b(store_loop);
                L(store_done);

                add(reg_lane, reg_lane, 1);
                b(lane_loop);
                L(lane_end);
            } else {
                if (jcp_.bubble_inplace && jcp_.top_k <= 8) {
                    const int topk = jcp_.top_k;
                    const int reg_base = 8;
                    std::vector<SReg> s_vals;
                    std::vector<SReg> s_idxs;
                    s_vals.reserve(topk);
                    s_idxs.reserve(topk);
                    for (int i = 0; i < topk; ++i) {
                        s_vals.emplace_back(SReg(reg_base + i));
                        s_idxs.emplace_back(SReg(reg_base + topk + i));
                    }
                    const int tmp_base = reg_base + 2 * topk;
                    SReg s_tmp_val(tmp_base);
                    SReg s_tmp_idx(tmp_base + 1);

                    auto emit_cmp_better = [&](const SReg& s_val_l, const SReg& s_val_r, Label& l_true, Label& l_false) {
                        if (is_f32 || is_f16) {
                            fcmp(s_val_l, s_val_r);
                            if (jcp_.mode_max) {
                                b(VS, l_false);
                                b(LT, l_true);
                                b(l_false);
                            } else {
                                b(VS, l_true);
                                b(GT, l_true);
                                b(l_false);
                            }
                        } else {
                            fmov(w_val0, s_val_l);
                            fmov(w_val1, s_val_r);
                            cmp(w_val0, w_val1);
                            if (jcp_.mode_max) {
                                b(LT, l_true);
                                b(l_false);
                            } else {
                                b(GT, l_true);
                                b(l_false);
                            }
                        }
                    };

                    auto emit_swap_if_better = [&](const SReg& s_val_l,
                                                   const SReg& s_idx_l,
                                                   const SReg& s_val_r,
                                                   const SReg& s_idx_r) {
                        Label swap_true;
                        Label swap_false;
                        emit_cmp_better(s_val_l, s_val_r, swap_true, swap_false);
                        L(swap_true);
                        fmov(s_tmp_val, s_val_l);
                        fmov(s_val_l, s_val_r);
                        fmov(s_val_r, s_tmp_val);
                        fmov(s_tmp_idx, s_idx_l);
                        fmov(s_idx_l, s_idx_r);
                        fmov(s_idx_r, s_tmp_idx);
                        L(swap_false);
                    };

                    auto emit_swap_if_idx = [&](const SReg& s_val_l,
                                                const SReg& s_idx_l,
                                                const SReg& s_val_r,
                                                const SReg& s_idx_r) {
                        Label swap_true;
                        Label swap_false;
                        fmov(w_idx0, s_idx_l);
                        fmov(w_idx1, s_idx_r);
                        cmp(w_idx0, w_idx1);
                        b(LE, swap_false);
                        L(swap_true);
                        fmov(s_tmp_val, s_val_l);
                        fmov(s_val_l, s_val_r);
                        fmov(s_val_r, s_tmp_val);
                        fmov(s_tmp_idx, s_idx_l);
                        fmov(s_idx_l, s_idx_r);
                        fmov(s_idx_r, s_tmp_idx);
                        L(swap_false);
                    };

                    Label lane_loop;
                    Label lane_end;
                    Label scan_loop;
                    Label scan_done;

                    mov(reg_lane, 0);
                    L(lane_loop);
                    cmp(reg_lane, reg_work_amount);
                    bge(lane_end);

                    // init values/indices
                    for (int i = 0; i < topk; ++i) {
                        mov(w_i, i);
                        emit_load_src(w_i, w_lane);
                        if (is_f32 || is_f16) {
                            fmov(s_vals[i], s_val0);
                        } else {
                            fmov(s_vals[i], w_val0);
                        }
                        mov(w_idx0, i);
                        fmov(s_idxs[i], w_idx0);
                    }

                    // initial insertion sort
                    for (int i = 1; i < topk; ++i) {
                        for (int j = i; j > 0; --j) {
                            emit_swap_if_better(s_vals[j - 1], s_idxs[j - 1], s_vals[j], s_idxs[j]);
                        }
                    }

                    // scan rest
                    mov(w_i, topk);
                    L(scan_loop);
                    cmp(w_i, w_axis_dim);
                    bge(scan_done);
                    emit_load_src(w_i, w_lane);
                    if (is_f32 || is_f16) {
                        fmov(s_tmp_val, s_val0);
                    } else {
                        fmov(s_tmp_val, w_val0);
                    }
                    fmov(s_tmp_idx, w_i);
                    Label insert_true;
                    Label insert_false;
                    emit_cmp_better(s_vals[topk - 1], s_tmp_val, insert_true, insert_false);
                    L(insert_false);
                    add(w_i, w_i, 1);
                    b(scan_loop);
                    L(insert_true);
                    fmov(s_vals[topk - 1], s_tmp_val);
                    fmov(s_idxs[topk - 1], s_tmp_idx);
                    for (int j = topk - 1; j > 0; --j) {
                        emit_swap_if_better(s_vals[j - 1], s_idxs[j - 1], s_vals[j], s_idxs[j]);
                    }
                    add(w_i, w_i, 1);
                    b(scan_loop);
                    L(scan_done);

                    if (jcp_.sort_index) {
                        for (int i = 1; i < topk; ++i) {
                            for (int j = i; j > 0; --j) {
                                emit_swap_if_idx(s_vals[j - 1], s_idxs[j - 1], s_vals[j], s_idxs[j]);
                            }
                        }
                    }

                    // store outputs
                    for (int i = 0; i < topk; ++i) {
                        mov(w_i, i);
                        if (is_f32 || is_f16) {
                            fmov(s_val0, s_vals[i]);
                        } else {
                            fmov(w_val0, s_vals[i]);
                        }
                        fmov(w_idx0, s_idxs[i]);
                        emit_store_dst(w_i, w_lane);
                        emit_store_idx(w_i, w_lane);
                    }

                    add(reg_lane, reg_lane, 1);
                    b(lane_loop);
                    L(lane_end);
                } else {
                    Label lane_loop;
                    Label lane_end;
                    Label init_loop;
                    Label init_done;
                    Label sort_i_loop;
                    Label sort_j_loop;
                    Label sort_j_done;
                    Label sort_done;
                    Label scan_loop;
                    Label scan_done;
                    Label scan_sort_j_loop;
                    Label scan_sort_j_done;
                    Label sort_idx_i_loop;
                    Label sort_idx_j_loop;
                    Label sort_idx_j_done;
                    Label sort_idx_done;
                    Label store_loop;
                    Label store_done;

                    mov(reg_lane, 0);
                    L(lane_loop);
                    cmp(reg_lane, reg_work_amount);
                    bge(lane_end);

                    mul(reg_tmp, reg_lane, reg_scratch_stride);
                    add(reg_vals_base, reg_prc, reg_tmp);
                    add(reg_idx_base, reg_prc_idx, reg_tmp);

                    // init
                    mov(w_i, 0);
                    L(init_loop);
                    cmp(w_i, w_k_eff);
                    bge(init_done);
                    emit_load_src(w_i, w_lane);
                    emit_store_scratch_val0(reg_vals_base, w_i);
                    emit_store_scratch_idx(reg_idx_base, w_i, w_i);
                    add(w_i, w_i, 1);
                    b(init_loop);
                    L(init_done);

                    // initial insertion sort
                    mov(w_i, 1);
                    L(sort_i_loop);
                    cmp(w_i, w_k_eff);
                    bge(sort_done);
                    mov(w_j, w_i);
                    L(sort_j_loop);
                    cmp(w_j, 0);
                    beq(sort_j_done);
                    sub(w_pos, w_j, 1);
                    emit_load_scratch_val0(reg_vals_base, w_j);
                    emit_load_scratch_val1(reg_vals_base, w_pos);
                    emit_load_scratch_idx(reg_idx_base, w_j, w_idx0);
                    emit_load_scratch_idx(reg_idx_base, w_pos, w_idx1);
                    Label better_true;
                    Label better_false;
                    emit_better(better_true, better_false);
                    L(better_true);
                    emit_store_scratch_val1(reg_vals_base, w_j);
                    emit_store_scratch_val0(reg_vals_base, w_pos);
                    emit_store_scratch_idx(reg_idx_base, w_j, w_idx1);
                    emit_store_scratch_idx(reg_idx_base, w_pos, w_idx0);
                    sub(w_j, w_j, 1);
                    b(sort_j_loop);
                    L(better_false);
                    L(sort_j_done);
                    add(w_i, w_i, 1);
                    b(sort_i_loop);
                    L(sort_done);

                    // scan rest
                    mov(w_i, w_k_eff);
                    L(scan_loop);
                    cmp(w_i, w_axis_dim);
                    bge(scan_done);
                    emit_load_src(w_i, w_lane);
                    sub(w_pos, w_k_eff, 1);
                    emit_load_scratch_val1(reg_vals_base, w_pos);
                    emit_load_scratch_idx(reg_idx_base, w_pos, w_idx1);
                    mov(w_idx0, w_i);
                    Label insert_true;
                    Label insert_false;
                    emit_better(insert_true, insert_false);
                    L(insert_false);
                    add(w_i, w_i, 1);
                    b(scan_loop);
                    L(insert_true);
                    emit_store_scratch_val0(reg_vals_base, w_k_eff);
                    emit_store_scratch_idx(reg_idx_base, w_k_eff, w_i);
                    mov(w_j, w_k_eff);
                    L(scan_sort_j_loop);
                    cmp(w_j, 0);
                    beq(scan_sort_j_done);
                    sub(w_pos, w_j, 1);
                    emit_load_scratch_val0(reg_vals_base, w_j);
                    emit_load_scratch_val1(reg_vals_base, w_pos);
                    emit_load_scratch_idx(reg_idx_base, w_j, w_idx0);
                    emit_load_scratch_idx(reg_idx_base, w_pos, w_idx1);
                    Label swap_true;
                    Label swap_false;
                    emit_better(swap_true, swap_false);
                    L(swap_true);
                    emit_store_scratch_val1(reg_vals_base, w_j);
                    emit_store_scratch_val0(reg_vals_base, w_pos);
                    emit_store_scratch_idx(reg_idx_base, w_j, w_idx1);
                    emit_store_scratch_idx(reg_idx_base, w_pos, w_idx0);
                    sub(w_j, w_j, 1);
                    b(scan_sort_j_loop);
                    L(swap_false);
                    L(scan_sort_j_done);
                    add(w_i, w_i, 1);
                    b(scan_loop);
                    L(scan_done);

                    if (jcp_.sort_index) {
                        mov(w_i, 1);
                        L(sort_idx_i_loop);
                        cmp(w_i, w_k_eff);
                        bge(sort_idx_done);
                        mov(w_j, w_i);
                        L(sort_idx_j_loop);
                        cmp(w_j, 0);
                        beq(sort_idx_j_done);
                        sub(w_pos, w_j, 1);
                        emit_load_scratch_idx(reg_idx_base, w_j, w_idx0);
                        emit_load_scratch_idx(reg_idx_base, w_pos, w_idx1);
                        cmp(w_idx1, w_idx0);
                        b(LE, sort_idx_j_done);
                        emit_store_scratch_idx(reg_idx_base, w_j, w_idx1);
                        emit_store_scratch_idx(reg_idx_base, w_pos, w_idx0);
                        emit_load_scratch_val0(reg_vals_base, w_j);
                        emit_load_scratch_val1(reg_vals_base, w_pos);
                        emit_store_scratch_val1(reg_vals_base, w_j);
                        emit_store_scratch_val0(reg_vals_base, w_pos);
                        sub(w_j, w_j, 1);
                        b(sort_idx_j_loop);
                        L(sort_idx_j_done);
                        add(w_i, w_i, 1);
                        b(sort_idx_i_loop);
                    }
                    L(sort_idx_done);

                    // store outputs
                    mov(w_i, 0);
                    L(store_loop);
                    cmp(w_i, w_top_k);
                    bge(store_done);
                    cmp(w_i, w_k_eff);
                    sub(w_pos, w_k_eff, 1);
                    csel(w_pos, w_i, w_pos, LT);
                    emit_load_scratch_val0(reg_vals_base, w_pos);
                    emit_load_scratch_idx(reg_idx_base, w_pos, w_idx0);
                    emit_store_dst(w_i, w_lane);
                    emit_store_idx(w_i, w_lane);
                    add(w_i, w_i, 1);
                    b(store_loop);
                    L(store_done);

                    add(reg_lane, reg_lane, 1);
                    b(lane_loop);
                    L(lane_end);
                }
            }
        }
        L(done);
        postamble();
    }
};

std::shared_ptr<jit_uni_topk_kernel> create_topk_kernel_aarch64(const jit_topk_config_params& jcp) {
    const bool has_sve = sve_utils::with_cpu_sve();
    if (has_sve) {
        if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_512)) {
            return std::make_shared<jit_uni_topk_kernel_aarch64<dnnl::impl::cpu::aarch64::sve_512>>(jcp);
        }
        if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_384)) {
            return std::make_shared<jit_uni_topk_kernel_aarch64<dnnl::impl::cpu::aarch64::sve_384>>(jcp);
        }
        if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_256)) {
            return std::make_shared<jit_uni_topk_kernel_aarch64<dnnl::impl::cpu::aarch64::sve_256>>(jcp);
        }
        if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_128)) {
            return std::make_shared<jit_uni_topk_kernel_aarch64<dnnl::impl::cpu::aarch64::sve_128>>(jcp);
        }
    }
    if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::asimd)) {
        return std::make_shared<jit_uni_topk_kernel_aarch64<dnnl::impl::cpu::aarch64::asimd>>(jcp);
    }
    return nullptr;
}

}  // namespace ov::intel_cpu::node

#endif  // defined(OPENVINO_ARCH_ARM64)
