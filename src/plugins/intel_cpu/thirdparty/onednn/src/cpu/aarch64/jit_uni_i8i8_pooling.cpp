/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
* Copyright 2020-2021 FUJITSU LIMITED
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
#include "cpu/aarch64/jit_uni_i8i8_pooling.hpp"
#include <math.h>

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

static inline dim_t get_offset(
        const memory_desc_wrapper &mdw, int n, int c, int d, int h, int w) {
    switch (mdw.ndims()) {
        case 3: return mdw.blk_off(n, c, w);
        case 4: return mdw.blk_off(n, c, h, w);
        case 5: return mdw.blk_off(n, c, d, h, w);
        default: assert(!"Invalid tensor dimension in pooling");
    }
    return 0;
}

using namespace Xbyak_aarch64;

using namespace dnnl::impl::utils;
using namespace dnnl::impl::types;
using namespace alg_kind;

#define GET_OFF(field) offsetof(call_params_t, field)

struct call_params_t {
    const char *src_i8;
    const char *dst_i8;
    size_t kd_range;
    size_t kh_range;
    size_t kw_range;
    float idivider;
    const char *src_safe_access;
    const char *dst_safe_access;
};

template <cpu_isa_t isa>
struct jit_uni_i8i8_pooling_fwd_ker_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_i8i8_pooling_fwd_ker_t)

    using TReg = typename cpu_isa_traits<isa>::TReg;

    VReg xreg(int idx) const { return VReg(idx); }
    ZReg yreg(int idx) const { return ZReg(xreg(idx).getIdx()); }
    TReg vreg(int idx) const { return TReg(xreg(idx).getIdx()); }
    XReg reg_param = x0;
    XReg reg_ptr_src_i8 = x4;
    XReg reg_ptr_dst_i8 = x5;
    XReg reg_ptr_maskmovdqu_dst = x3;

    XReg reg_kd_index = x0;
    XReg reg_kh_index = x11;
    XReg reg_kw_index = x10;
    XReg reg_kd = x14;
    XReg reg_kh = x13;
    XReg reg_kw = x12;
    XReg c_iter = x15; // shared with reg_mask; only used after mask init

    XReg aux_reg_src_d
            = x2; // shared with reg_tmp; loaded before each accum loop, unused during store
    XReg aux_reg_src_h = x7;
    XReg aux_reg_src_w = x1;

    XReg reg_tmp = x2; // only used during mask init and store
    XReg reg_src_safe_access = x9;
    XReg reg_dst_safe_access = x1;

    XReg reg_mask = x15; // only used during mask init

    PReg k_cmp_mask = p7;
    PReg mask(int idx) { return PReg(6 - idx); } /* 6, 5, 4, 3 */

    PReg p_all_zero = p0;
    PReg p_512 = p2;
    PReg p_tmp0 = p1;

    VReg xmm_tmp = xreg(0); // temp to init vreg_tmp
    TReg vreg_tmp = vreg(0); // max pooling : holds minimum values for data_type
    TReg vreg_zeros = vreg(1);

    ZReg z_tmp0 = z24;

    int post_op_tail_opmask_idx_ = -1;
    jit_pool_conf_t jpp;

    enum : int { max_vidx_base = 2 };
    //"avg" pool uses more registers for unrolling.
    enum : int { avg_vidx_base = 2 };

    TReg max_base_vr(int idx) const { return vreg(max_vidx_base + idx); }
    TReg avg_base_vr(int idx) const { return vreg(avg_vidx_base + idx); }

    size_t sizeof_src_dt() const { return data_type_size(jpp.src_dt); }
    size_t sizeof_dst_dt() const { return data_type_size(jpp.dst_dt); }

    /* max pooling */
    TReg vreg_src(int idx) const {
        return max_base_vr(idx);
    } // [0    .. ur_c-1]
    TReg vreg_dst(int idx) const {
        return max_base_vr(jpp.ur_c + idx);
    } // [ur_c .. 2*ur_c-1]

    /* avg pooling */
    // s32 used for processing of s8/u8 data
    // thus we need to take into account ratio of sizes s32/i8 = 4
    static constexpr data_type_t avg_proc_dt = data_type::s32;
    enum : int {
        s32_to_i8_ratio = sizeof(typename prec_traits<avg_proc_dt>::type)
                / sizeof(typename prec_traits<data_type::u8>::type),
        max_num_ll = s32_to_i8_ratio,
        mmx_msk_base_reg = 3
    };

    TReg vreg_src_s32(int jj, int ll) {
        return avg_base_vr(3 * max_num_ll * jj + ll + 0 * max_num_ll);
    } // ll: 0..4 [0..3]

    TReg vreg_dst_s32(int jj, int ll) {
        return avg_base_vr(3 * max_num_ll * jj + ll + 1 * max_num_ll);
    } // ll: 0..4 [4..7]

    TReg vreg_dst_f32(int jj, int ll) {
        return avg_base_vr(3 * max_num_ll * jj + ll + 2 * max_num_ll);
    } // ll: 0..4 [8..11]

    static bool post_ops_ok(jit_pool_conf_t &jpp, const primitive_attr_t &attr,
            const memory_desc_wrapper &dst_d);

    void init_tmp_reg();
    void init_mask();

    void load_vreg_mask_q(int ll) {};

    void load_src_max_op(
            int jj, int ll, size_t offset, bool masked, uint64_t msk);
    void load_src_avg_op(
            int jj, int ll, size_t offset, bool masked, uint64_t msk);
    void load_src(int jj, int ll, int c_tail);

    void store_dst_max_op(
            int jj, int ll, size_t offset, bool masked, uint64_t msk);
    void store_dst_avg_op(
            int jj, int ll, size_t offset, bool masked, uint64_t msk);
    void store_dst(int jj, int ll, int c_tail);

    void compute_avg_step(int ur_c, int c_tail);
    void compute_max_op(const int jj);
    void compute_max_step(int ur_c, int c_tail);
    void compute_step(int ur_c, int c_tail);

    void compute_c_block();
    void generate() override;

    static status_t init_conf(jit_pool_conf_t &jpp, const pooling_pd_t *ppd);

    jit_uni_i8i8_pooling_fwd_ker_t(
            const jit_pool_conf_t &jpp_, const memory_desc_t *dst_md)
        : jit_generator(nullptr, MAX_CODE_SIZE, true), jpp(jpp_) {}
};

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_512>::load_src_max_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    if (masked) {
        if (jpp.src_dt == s32) {
            add_imm(X_DEFAULT_ADDR, aux_reg_src_w, offset, X_TMP_0);
            zip1(p_tmp0.b, mask(0).b, p_all_zero.b);
            zip1(p_tmp0.h, p_tmp0.h, p_all_zero.h);
            ld1w(z_tmp0.s, p_tmp0 / T_z, ptr(X_DEFAULT_ADDR));
            mov(vreg_src(jj).s, p_tmp0 / T_m, z_tmp0.s);
        } else {
            add_imm(X_DEFAULT_ADDR, aux_reg_src_w, offset, X_TMP_0);
            ld1b(z_tmp0.b, mask(0) / T_z, ptr(X_DEFAULT_ADDR));
            mov(vreg_src(jj).b, mask(0) / T_m, z_tmp0.b);
        }
    } else {
        add_imm(X_DEFAULT_ADDR, aux_reg_src_w, offset, X_TMP_0);
        ldr(vreg_src(jj), ptr(X_DEFAULT_ADDR));
    }
};

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_512>::load_src_avg_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    const TReg &vr_src = vreg_src_s32(jj, ll);

    switch (jpp.src_dt) {
        case s32:
            add_imm(X_DEFAULT_ADDR, aux_reg_src_w, offset * data_type_size(s32),
                    X_TMP_0);
            if (masked) {
                zip1(p_tmp0.b, mask(ll).b, p_all_zero.b);
                zip1(p_tmp0.h, p_tmp0.h, p_all_zero.h);
                ld1w(z_tmp0.s, p_tmp0 / T_z, ptr(X_DEFAULT_ADDR));
                mov(vr_src.s, p_tmp0 / T_m, z_tmp0.s);
            } else {
                ldr(vr_src, ptr(X_DEFAULT_ADDR));
            }
            break;
        case data_type::s8:
            add_imm(X_DEFAULT_ADDR, aux_reg_src_w, offset, X_TMP_0);
            if (masked) {
                zip1(p_tmp0.b, mask(ll).b, p_all_zero.b);
                zip1(p_tmp0.h, p_tmp0.h, p_all_zero.h);
                // use p_tmp, uzp1 can be eliminate.
                ld1b(z_tmp0.s, p_tmp0 / T_z, ptr(X_DEFAULT_ADDR));
                sxtb(vr_src.s, p_tmp0 / T_m, z_tmp0.s);
            } else {
                ld1b(z_tmp0.s, p_512 / T_z, ptr(X_DEFAULT_ADDR));
                sxtb(vr_src.s, p_512 / T_m, z_tmp0.s);
            }
            break;
        case u8:
            add_imm(X_DEFAULT_ADDR, aux_reg_src_w, offset, X_TMP_0);
            if (masked) {
                zip1(p_tmp0.b, mask(ll).b, p_all_zero.b);
                zip1(p_tmp0.h, p_tmp0.h, p_all_zero.h);
                // use p_tmp, uzp1 can be eliminate.
                ld1b(z_tmp0.s, p_tmp0 / T_z, ptr(X_DEFAULT_ADDR));
                uxtb(vr_src.s, p_tmp0 / T_m, z_tmp0.s);
            } else {
                ldr(QReg(z_tmp0.getIdx()), ptr(X_DEFAULT_ADDR));
                zip1(z_tmp0.b, z_tmp0.b, z_tmp0.b);
                zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);
                uxtb(vr_src.s, p_512 / T_m, z_tmp0.s);
            }
            break;
        default: assert(!"unsupported src data type");
    }
};

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::load_src(int jj, int ll, int c_tail) {
    using namespace data_type;

    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;

    switch (jpp.alg) {
        case pooling_max: {
            auto offset = jj * c_block * sizeof_src_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            load_src_max_op(jj, ll, offset, masked, jpp.tail[0]);
            break;
        }
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            auto offset = (ll * (c_block / max_num_ll) + jj * c_block)
                    * sizeof_src_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            load_src_avg_op(jj, ll, offset, masked, jpp.tail[ll]);
            break;
        }
        default: assert(!"unsupported algorithm");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_512>::store_dst_max_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    if (masked) {
        switch (jpp.src_dt) {
            case s32:
                add_imm(X_DEFAULT_ADDR, reg_ptr_dst_i8, offset, X_TMP_0);
                zip1(p_tmp0.b, mask(0).b, p_all_zero.b);
                zip1(p_tmp0.h, p_tmp0.h, p_all_zero.h);
                st1w(vreg_dst(jj).s, p_tmp0, ptr(X_DEFAULT_ADDR));
                break;
            case data_type::s8:
            case u8:
                add_imm(X_DEFAULT_ADDR, reg_ptr_dst_i8, offset, X_TMP_0);
                st1b(vreg_dst(jj).b, mask(0), ptr(X_DEFAULT_ADDR));
                break;
            default: assert(!"unsupported src data type");
        }
    } else {
        add_imm(X_DEFAULT_ADDR, reg_ptr_dst_i8, offset, X_TMP_0);
        str(vreg_dst(jj), ptr(X_DEFAULT_ADDR));
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_512>::store_dst_avg_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    // Don't generate useless code
    if (masked && !msk) return;

    const TReg &vr_dst = vreg_dst_s32(jj, ll);
    switch (jpp.dst_dt) {
        case s32:
            add_imm(X_DEFAULT_ADDR, reg_ptr_dst_i8, offset, X_TMP_0);
            if (masked) {
                zip1(p_tmp0.b, mask(ll).b, p_all_zero.b);
                zip1(p_tmp0.h, p_tmp0.h, p_all_zero.h);
                st1w(vr_dst.s, p_tmp0, ptr(X_DEFAULT_ADDR));
            } else {
                str(vr_dst, ptr(X_DEFAULT_ADDR));
            }
            break;
        case data_type::s8:
            add_imm(X_DEFAULT_ADDR, reg_ptr_dst_i8, offset, X_TMP_0);
            if (masked) {
                mov(z_tmp0.d, vr_dst.d);
                smin(z_tmp0.s, 127);
                smax(z_tmp0.s, -128);
                zip1(p_tmp0.b, mask(ll).b, p_all_zero.b);
                zip1(p_tmp0.h, p_tmp0.h, p_all_zero.h);
                st1b(z_tmp0.s, p_tmp0, ptr(X_DEFAULT_ADDR));
            } else {
                mov(z_tmp0.d, vr_dst.d);
                smin(z_tmp0.s, 127);
                smax(z_tmp0.s, -128);
                st1b(z_tmp0.s, p_512, ptr(X_DEFAULT_ADDR));
            }
            break;
        case u8:
            add_imm(X_DEFAULT_ADDR, reg_ptr_dst_i8, offset, X_TMP_0);
            if (masked) {
                mov(z_tmp0.d, vr_dst.d);
                umin(z_tmp0.s, 255);
                zip1(p_tmp0.b, mask(ll).b, p_all_zero.b);
                zip1(p_tmp0.h, p_tmp0.h, p_all_zero.h);
                st1b(z_tmp0.s, p_tmp0, ptr(X_DEFAULT_ADDR));
            } else {
                mov(z_tmp0.d, vr_dst.d);
                umin(z_tmp0.s, 255);
                st1b(z_tmp0.s, p_512, ptr(X_DEFAULT_ADDR));
            }
            break;
        default: assert(!"unsupported dst data_type");
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::store_dst(
        int jj, int ll, int c_tail) {
    using namespace data_type;

    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;

    switch (jpp.alg) {
        case pooling_max: {
            auto offset = jj * c_block * sizeof_dst_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            store_dst_max_op(jj, ll, offset, masked, jpp.tail[ll]);
            break;
        }
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            auto offset = (ll * (c_block / max_num_ll) + jj * c_block)
                    * sizeof_dst_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            store_dst_avg_op(jj, ll, offset, masked, jpp.tail[ll]);
            break;
        }
        default: assert(!"unsupported pooling algorithm");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_512>::compute_max_op(const int jj) {
    using namespace data_type;

    // Compare
    switch (jpp.src_dt) {
        case s32:
            cmplt(k_cmp_mask.s, p_512 / T_z, vreg_dst(jj).s, vreg_src(jj).s);
            break;
        case data_type::s8:
            cmplt(k_cmp_mask.b, p_512 / T_z, vreg_dst(jj).b, vreg_src(jj).b);
            break;
        case u8:
            cmpls(k_cmp_mask.b, p_512 / T_z, vreg_dst(jj).b, vreg_src(jj).b);
            break;
        default: assert(!"unsupported src data type");
    }

    // move max values into vreg_dst
    if (jpp.src_dt == s32) {
        sel(vreg_dst(jj).s, k_cmp_mask / T_m, vreg_src(jj).s, vreg_dst(jj).s);
    } else {
        sel(vreg_dst(jj).b, k_cmp_mask / T_m, vreg_src(jj).b, vreg_dst(jj).b);
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_max_step(
        int ur_c, int c_tail) {
    Label l_kd, l_kh, l_kw;

    int ih = jpp.ih;
    int iw = jpp.iw;
    int c = jpp.c;

    for (int jj = 0; jj < ur_c; jj++) {
        mov(vreg_dst(jj).d, vreg_tmp.d);
    }

    mov(aux_reg_src_d, reg_ptr_src_i8);
    eor(reg_kd_index, reg_kd_index, reg_kd_index);
    L(l_kd);
    {
        mov(aux_reg_src_h, aux_reg_src_d);
        eor(reg_kh_index, reg_kh_index, reg_kh_index);
        L(l_kh);
        {
            mov(aux_reg_src_w, aux_reg_src_h);
            eor(reg_kw_index, reg_kw_index, reg_kw_index);
            L(l_kw);
            {
                for (int jj = 0; jj < ur_c; jj++) {
                    load_src(jj, 0, c_tail);
                    compute_max_op(jj);
                }
                add(aux_reg_src_w, aux_reg_src_w, c * sizeof_src_dt());
                adds(reg_kw_index, reg_kw_index, 1);
                cmp(reg_kw_index, reg_kw);
                b(LT, l_kw);
            }
            add_imm(aux_reg_src_h, aux_reg_src_h, iw * c * sizeof_src_dt(),
                    X_TMP_0);
            adds(reg_kh_index, reg_kh_index, 1);
            cmp(reg_kh_index, reg_kh);
            b(LT, l_kh);
        }
        add_imm(aux_reg_src_d, aux_reg_src_d, ih * iw * c * sizeof_src_dt(),
                X_TMP_0);
        adds(reg_kd_index, reg_kd_index, 1);
        cmp(reg_kd_index, reg_kd);
        b(LT, l_kd);
    }

    for (int jj = 0; jj < ur_c; jj++)
        store_dst(jj, 0, c_tail);
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_avg_step(
        int ur_c, int c_tail) {
    using namespace data_type;

    Label l_kd, l_kh, l_kw;

    int ih = jpp.ih;
    int iw = jpp.iw;
    int c = jpp.c;

    const int num_ll = data_type_size(avg_proc_dt) / data_type_size(jpp.src_dt);

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ll = 0; ll < num_ll; ll++) {
            bool masked = jj == ur_c - 1 && c_tail;
            size_t msk = jpp.tail[ll];
            if (!(masked && !msk)) {
                // Clearing of src reg is not needed as they are written before read
                eor(vreg_dst_s32(jj, ll).d, vreg_dst_s32(jj, ll).d,
                        vreg_dst_s32(jj, ll).d);
            }
        }
    }

    mov(aux_reg_src_d, reg_ptr_src_i8);
    eor(reg_kd_index, reg_kd_index, reg_kd_index);
    L(l_kd);
    {
        mov(aux_reg_src_h, aux_reg_src_d);
        eor(reg_kh_index, reg_kh_index, reg_kh_index);
        L(l_kh);
        {
            mov(aux_reg_src_w, aux_reg_src_h);
            eor(reg_kw_index, reg_kw_index, reg_kw_index);
            L(l_kw);
            {
                for (int jj = 0; jj < ur_c; jj++) {
                    for (int ll = 0; ll < num_ll; ll++) {
                        bool masked = jj == ur_c - 1 && c_tail;
                        size_t msk = jpp.tail[ll];
                        if (!(masked && !msk)) {
                            load_src(jj, ll, c_tail);
                            add(vreg_dst_s32(jj, ll).s, vreg_dst_s32(jj, ll).s,
                                    vreg_src_s32(jj, ll).s);
                        }
                    }
                }
                add(aux_reg_src_w, aux_reg_src_w, c * sizeof_src_dt());
                adds(reg_kw_index, reg_kw_index, 1);
                cmp(reg_kw_index, reg_kw);
                b(LT, l_kw);
            }
            add_imm(aux_reg_src_h, aux_reg_src_h, iw * c * sizeof_src_dt(),
                    X_TMP_0);
            adds(reg_kh_index, reg_kh_index, 1);
            cmp(reg_kh_index, reg_kh);
            b(LT, l_kh);
        }
        add_imm(aux_reg_src_d, aux_reg_src_d, ih * iw * c * sizeof_src_dt(),
                X_TMP_0);
        adds(reg_kd_index, reg_kd_index, 1);
        cmp(reg_kd_index, reg_kd);
        b(LT, l_kd);
    }

    static constexpr int vlen_size_elem
            = cpu_isa_traits<isa>::vlen / sizeof(float);
    const auto reg_tmp_postops = XReg(15);

    if (jpp.with_binary) {
        mov_imm(X_TMP_0,
                static_cast<int64_t>(
                        static_cast<int8_t>(ur_c * num_ll * vlen_size_elem)));
        mul(reg_tmp_postops, c_iter, X_TMP_0);
    }

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ll = 0; ll < num_ll; ll++) {
            const bool masked = jj == ur_c - 1 && c_tail;
            const size_t msk = jpp.tail[ll];
            if (!(masked && !msk)) {
                const auto &reg_dst_f32 = vreg_dst_f32(jj, ll);
                const auto &reg_dst_s32 = vreg_dst_s32(jj, ll);
                scvtf(reg_dst_f32.s, p_512 / T_m, reg_dst_s32.s);
                fmad(reg_dst_f32.s, p_512 / T_m, vreg_tmp.s, vreg_zeros.s);

                frinti(reg_dst_s32.s, p_512 / T_m, reg_dst_f32.s);
                fcvtzs(reg_dst_s32.s, p_512 / T_m, reg_dst_s32.s);

                store_dst(jj, ll, c_tail);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_step(int ur_c, int c_tail) {
    switch (jpp.alg) {
        case pooling_max: compute_max_step(ur_c, c_tail); break;
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: compute_avg_step(ur_c, c_tail); break;
        default: assert(!"unsupported pooling algorithm");
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_c_block() {
    Label l_main_loop;

    int nb_c = jpp.nb_c;
    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;
    int ur_c_tail = jpp.ur_c_tail;
    int c_steps = nb_c / ur_c;
    int c_tail = jpp.c_tail;

    eor(c_iter, c_iter, c_iter);
    if (c_steps > 0) {
        L(l_main_loop);
        {
            compute_step(ur_c, 0);
            add(reg_ptr_src_i8, reg_ptr_src_i8,
                    ur_c * c_block * sizeof_src_dt());
            add(reg_ptr_dst_i8, reg_ptr_dst_i8,
                    ur_c * c_block * sizeof_dst_dt());
            adds(c_iter, c_iter, 1);
            mov_imm(X_TMP_0, c_steps);
            cmp(c_iter, X_TMP_0);
            b(LT, l_main_loop);
        }
    }

    if (ur_c_tail != 0) { compute_step(ur_c_tail, c_tail); }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_512>::init_mask() {
    using namespace data_type;

    sub(X_TRANSLATOR_STACK, X_TRANSLATOR_STACK, 8 * max_num_ll);

    for (int ll = 0; ll < max_num_ll; ll++) {
        mov_imm(reg_mask, jpp.tail[ll]);
        str(reg_mask, ptr(X_TRANSLATOR_STACK, 8 * ll));
    }
    for (int ll = 0; ll < max_num_ll; ll++) {
        ldr(PReg(mask(ll)), ptr(X_TRANSLATOR_STACK));
        add(X_TRANSLATOR_STACK, X_TRANSLATOR_STACK, 8);
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::init_tmp_reg() {
    using namespace data_type;

    switch (jpp.alg) {
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding:
            add_imm(X_DEFAULT_ADDR, reg_param,
                    offsetof(call_params_t, idivider), X_TMP_0);
            ldr(reg_tmp, ptr(X_DEFAULT_ADDR));
            bic(xmm_tmp.b16, xmm_tmp.b16, xmm_tmp.b16);
            mov(xmm_tmp.d[0], reg_tmp);

            dup(vreg_tmp.s, ZRegS(xmm_tmp.getIdx())[0]);
            break;
        case pooling_max:
            switch (jpp.src_dt) {
                case s32:
                    mov_imm(reg_tmp, nstl::numeric_limits<int32_t>::lowest());
                    break;
                case data_type::s8:
                    mov_imm(reg_tmp, nstl::numeric_limits<int8_t>::lowest());
                    break;
                case u8:
                    mov(reg_tmp, nstl::numeric_limits<uint8_t>::lowest());
                    break;
                default: assert(!"unsupported src data_type");
            }

            bic(xmm_tmp.b16, xmm_tmp.b16, xmm_tmp.b16);
            mov(xmm_tmp.d[0], reg_tmp);
            if (jpp.src_dt == s32) {
                dup(vreg_tmp.s, ZRegS(xmm_tmp.getIdx())[0]);
            } else if (mayiuse(sve_512)) {
                dup(ZRegB(vreg_tmp.getIdx()), ZRegB(xmm_tmp.getIdx())[0]);
            } else {
                assert(!"unreachable");
            }
            break;
        default: assert(!"unsupported pooling algorithm");
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::generate() {
    preamble();

    ptrue(p_512.b);
    pfalse(p_all_zero.b);

    add_imm(X_DEFAULT_ADDR, reg_param, offsetof(call_params_t, src_i8),
            X_TMP_0);
    ldr(reg_ptr_src_i8, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, reg_param, offsetof(call_params_t, dst_i8),
            X_TMP_0);
    ldr(reg_ptr_dst_i8, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, reg_param, offsetof(call_params_t, kd_range),
            X_TMP_0);
    ldr(reg_kd, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, reg_param, offsetof(call_params_t, kh_range),
            X_TMP_0);
    ldr(reg_kh, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, reg_param, offsetof(call_params_t, kw_range),
            X_TMP_0);
    ldr(reg_kw, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, reg_param, offsetof(call_params_t, src_safe_access),
            X_TMP_0);
    ldr(reg_src_safe_access, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, reg_param, offsetof(call_params_t, dst_safe_access),
            X_TMP_0);
    ldr(reg_dst_safe_access, ptr(X_DEFAULT_ADDR));

    eor(VReg16B(vreg_zeros.getIdx()), VReg16B(vreg_zeros.getIdx()),
            VReg16B(vreg_zeros.getIdx()));

    init_mask();
    init_tmp_reg();

    compute_c_block();

    postamble();
}

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_ker_t<isa>::init_conf(
        jit_pool_conf_t &jpp, const pooling_pd_t *ppd) {
    if (!mayiuse(isa)) return status::unimplemented;

    const auto &pd = *ppd->desc();
    const memory_desc_wrapper src_d(ppd->src_md());
    const memory_desc_wrapper dst_d(ppd->dst_md());
    const int ndims = src_d.ndims();
    const bool is_1d = ndims == 3;
    const bool is_3d = ndims == 5;

    jpp.mb = src_d.dims()[0];
    jpp.c = src_d.dims()[1];

    jpp.id = is_3d ? src_d.dims()[ndims - 3] : 1;
    jpp.ih = is_1d ? 1 : src_d.dims()[ndims - 2];
    jpp.iw = src_d.dims()[ndims - 1];

    jpp.od = is_3d ? dst_d.dims()[ndims - 3] : 1;
    jpp.oh = is_1d ? 1 : dst_d.dims()[ndims - 2];
    jpp.ow = dst_d.dims()[ndims - 1];

    jpp.stride_d = is_3d ? pd.strides[ndims - 5] : 1;
    jpp.stride_h = is_1d ? 1 : pd.strides[ndims - 4];
    jpp.stride_w = pd.strides[ndims - 3];

    jpp.kd = is_3d ? pd.kernel[ndims - 5] : 1;
    jpp.kh = is_1d ? 1 : pd.kernel[ndims - 4];
    jpp.kw = pd.kernel[ndims - 3];

    jpp.f_pad = is_3d ? pd.padding[0][ndims - 5] : 0;
    jpp.t_pad = is_1d ? 0 : pd.padding[0][ndims - 4];
    jpp.l_pad = pd.padding[0][ndims - 3];

    int back_pad = calculate_end_padding(
            jpp.f_pad, jpp.od, jpp.id, jpp.stride_d, jpp.kd);
    int bottom_pad = calculate_end_padding(
            jpp.t_pad, jpp.oh, jpp.ih, jpp.stride_h, jpp.kh);
    int right_pad = calculate_end_padding(
            jpp.l_pad, jpp.ow, jpp.iw, jpp.stride_w, jpp.kw);

    if (jpp.f_pad >= jpp.kd || jpp.t_pad >= jpp.kh || jpp.l_pad >= jpp.kw
            || back_pad >= jpp.kd || bottom_pad >= jpp.kh
            || right_pad >= jpp.kw)
        return status::unimplemented;

    jpp.alg = pd.alg_kind;

    jpp.src_dt = pd.src_desc.data_type;
    jpp.dst_dt = pd.dst_desc.data_type;

    // data_type items per one vreg on the <isa>
    //     isa == sve_512 : 64 bytes -> 64 for s8/u8, 16 for s32
    int simd_w = cpu_isa_traits<isa>::vlen / data_type_size(jpp.src_dt);

    /* Verify that vlen-sized memory access happens within the tensor's
     * size, otherwise load/store will always spill outside the memory
     * boundary.*/
    bool safe_load_n_store = IMPLICATION(utils::one_of(isa, sve_512),
            jpp.mb * jpp.c * nstl::min(jpp.id, jpp.od)
                            * nstl::min(jpp.ih, jpp.oh)
                            * nstl::min(jpp.iw, jpp.ow)
                    >= simd_w);
    if (!safe_load_n_store) return status::unimplemented;

    jpp.c_block = simd_w;
    jpp.c_tail = jpp.c % jpp.c_block;
    jpp.nb_c = jpp.c / jpp.c_block;
    jpp.ur_c = 1;
    jpp.ur_c_tail = jpp.c_tail != 0;

    size_t tail_mask = (1ULL << jpp.c_tail) - 1;

    /* If channel_size is bigger than vlen, we can safely assume there is no
     * underflow of memory boundary, so always perform c_tail and save
     * a couple of compute cycles*/
    jpp.safe_c_tail = jpp.c_tail > 0 && jpp.c >= simd_w;

    switch (jpp.alg) {
        case pooling_max:
            jpp.tail[0] = tail_mask;
            jpp.tail[1] = 0;
            jpp.tail[2] = 0;
            jpp.tail[3] = 0;
            break;
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            // avg_proc_dt (s32) defines granularity (because u8/s8 processed as s32)
            // sve_512 : 16
            const size_t msk_gran
                    = cpu_isa_traits<isa>::vlen / data_type_size(avg_proc_dt);
            const size_t msk_msk = (1ULL << msk_gran) - 1;
            size_t m = tail_mask;
            for (size_t ll = 0; ll < max_num_ll; ll++) {
                jpp.tail[ll] = m & msk_msk;
                m = m >> msk_gran;
            }
            break;
        }
        default: return status::unimplemented;
    }

    if (!post_ops_ok(jpp, *ppd->attr(), dst_d)) return status::unimplemented;

    return status::success;
}

template <cpu_isa_t isa>
bool jit_uni_i8i8_pooling_fwd_ker_t<isa>::post_ops_ok(jit_pool_conf_t &jpp,
        const primitive_attr_t &attr, const memory_desc_wrapper &dst_d) {
    const auto &post_ops = attr.post_ops_;
    const auto &entries = post_ops.entry_;
    jpp.with_postops = false;
    jpp.with_eltwise = false;
    jpp.with_binary = false;

    return entries.empty() ? true : false;
}

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_t<isa>::pd_t::jit_conf() {
    return jit_uni_i8i8_pooling_fwd_ker_t<isa>::init_conf(jpp_, this);
}

template <cpu_isa_t isa>
jit_uni_i8i8_pooling_fwd_t<isa>::jit_uni_i8i8_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd), ker_(nullptr) {}

template <cpu_isa_t isa>
jit_uni_i8i8_pooling_fwd_t<isa>::~jit_uni_i8i8_pooling_fwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(ker_,
            new jit_uni_i8i8_pooling_fwd_ker_t<isa>(
                    pd()->jpp_, pd()->invariant_dst_md())));
    return ker_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_t<isa>::execute_forward(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto src_i8 = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto dst_i8 = CTX_OUT_CLEAN_MEM(char *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const auto &jpp = pd()->jpp_;
    /* Calculate when the memory-access will happen outisde of the memory
     * boundary, if so, compute a safe memory access. */
    const auto src_safe_access = reinterpret_cast<char *>(
            reinterpret_cast<ptrdiff_t>(src_i8 + src_d.size() - 1)
            - (cpu_isa_traits<isa>::vlen - 1));

    const auto dst_safe_access = reinterpret_cast<char *>(
            reinterpret_cast<ptrdiff_t>(dst_i8 + dst_d.size() - 1)
            - (cpu_isa_traits<isa>::vlen - 1));

    parallel_nd(
            jpp.mb, jpp.od, jpp.oh, jpp.ow, [&](int n, int od, int oh, int ow) {
                const int id = nstl::max(od * jpp.stride_d - jpp.f_pad, 0);
                const int ih = nstl::max(oh * jpp.stride_h - jpp.t_pad, 0);
                const int iw = nstl::max(ow * jpp.stride_w - jpp.l_pad, 0);

                const int kd_start
                        = nstl::max(0, jpp.f_pad - od * jpp.stride_d);
                const int kd_end = nstl::min(
                        jpp.kd, jpp.id + jpp.f_pad - od * jpp.stride_d);
                const int kh_start
                        = nstl::max(0, jpp.t_pad - oh * jpp.stride_h);
                const int kh_end = nstl::min(
                        jpp.kh, jpp.ih + jpp.t_pad - oh * jpp.stride_h);
                const int kw_start
                        = nstl::max(0, jpp.l_pad - ow * jpp.stride_w);
                const int kw_end = nstl::min(
                        jpp.kw, jpp.iw + jpp.l_pad - ow * jpp.stride_w);

                auto p = call_params_t();
                p.src_i8 = &src_i8[get_offset(src_d, n, 0, id, ih, iw)
                        * src_d.data_type_size()];
                p.dst_i8 = &dst_i8[get_offset(dst_d, n, 0, od, oh, ow)
                        * dst_d.data_type_size()];
                p.kd_range = (size_t)(kd_end - kd_start);
                p.kh_range = (size_t)(kh_end - kh_start);
                p.kw_range = (size_t)(kw_end - kw_start);
                p.idivider = 1.0f
                        / ((jpp.alg == pooling_avg_exclude_padding)
                                        ? p.kd_range * p.kh_range * p.kw_range
                                        : jpp.kd * jpp.kh * jpp.kw);
                p.src_safe_access = src_safe_access;
                p.dst_safe_access = dst_safe_access;
                (*ker_)(&p);
            });
    return status::success;
}

// Explicit instantiation only for supported <isa> values.
//
template struct jit_uni_i8i8_pooling_fwd_ker_t<sve_512>;
template struct jit_uni_i8i8_pooling_fwd_t<sve_512>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
