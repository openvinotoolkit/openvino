/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
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

#include <assert.h>
#include <numeric>

#include "dnnl_debug.h"

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/jit_uni_reorder.hpp"
#include "cpu/cpu_primitive.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

#include "cpu/aarch64/jit_generator.hpp"

// #define TR_DEBUG
#if defined(TR_DEBUG)
#define DEBUg(...) \
    do { \
        __VA_ARGS__ \
    } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

using namespace Xbyak_aarch64;
using namespace dnnl::impl::types;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace tr {

/** Minimal reasonable/desirable kernel size.
 * The constant might be used to determine how a problem should be split
 * between kernel and threading driver. */
const size_t ker_prb_size_min = 64;

/* kernel */
struct jit_uni_reorder_kernel_f32_t : public kernel_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_reorder_kernel_f32)

    void operator()(const call_param_t *c) const override {
        jit_generator::operator()(c);
    }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    enum {
        len_unroll_max = 256,
        ndims_jit_loop_max = 3,
    };

    struct simple_impl_desc_t {
        int ndims_full_unroll;
        int len_last_dim_unroll;
        int len_unroll;
    };

    static bool simple_impl_desc_init(
            const prb_t &prb, simple_impl_desc_t *desc) {
        const int ndims = prb.ndims;

        int ndims_full_unroll = 0;
        int len_last_dim_unroll = 1;
        int len_unroll = 1;

        for (int d = 0; d < ndims; ++d) {
            auto &node = prb.nodes[d];
            if (len_unroll * node.n <= len_unroll_max) {
                ndims_full_unroll++;
                len_unroll *= node.n;
            } else {
                len_last_dim_unroll = len_unroll_max / len_unroll;
                while (node.n % len_last_dim_unroll)
                    --len_last_dim_unroll;
                len_unroll *= len_last_dim_unroll;
                break;
            }
        }

        if (prb.ndims - ndims_full_unroll > ndims_jit_loop_max) return false;

        if (desc) {
            desc->ndims_full_unroll = ndims_full_unroll;
            desc->len_last_dim_unroll = len_last_dim_unroll;
            desc->len_unroll = len_unroll;
        }

        return true;
    }

    static bool applicable(const prb_t &p) {
        using namespace data_type;

        bool ok = true && p.ndims > 0
                && utils::one_of(p.itype, f32, s32, data_type::s8, u8)
                && utils::one_of(p.otype, f32, s32, data_type::s8, u8)
                && utils::everyone_is(0, p.ioff, p.ooff) /* do we need this? */
                && utils::one_of(p.beta, 0.f, 1.f) /* anything else? */
                && simple_impl_desc_init(p, nullptr);
        if (!ok) return false;

        const ptrdiff_t max_stride = (1LL << 31) - 1;
        for (int d = 0; d < p.ndims; ++d) {
            const ptrdiff_t cms = max_stride / p.nodes[d].n;
            bool strides_ok = true
                    && p.nodes[d].is < cms / (int)data_type_size(p.itype)
                    && p.nodes[d].os < cms / (int)data_type_size(p.otype);
            if (!strides_ok) return false;
        }

        return true;
    }

    int n(int d) {
        assert(d < prb_.ndims);
        return (int)prb_.nodes[d].n;
    }
    int is(int d) {
        assert(d < prb_.ndims);
        return (int)prb_.nodes[d].is;
    }
    int os(int d) {
        assert(d < prb_.ndims);
        return (int)prb_.nodes[d].os;
    }
    int ss(int d) {
        assert(d < prb_.ndims);
        return (int)prb_.nodes[d].ss;
    }

    void step(int off, int prev_i_off, int prev_o_off, int prev_s_off,
            int &i_off, int &o_off, int &s_off, int step_size = 1) {
        i_off = prev_i_off;
        o_off = prev_o_off;
        s_off = prev_s_off;

        if (off == 0) return;

        int start_dim = 0, dims_prod = 1;
        for (; start_dim < prb_.ndims && dims_prod != step_size; ++start_dim)
            dims_prod *= n(start_dim);
        assert(start_dim < prb_.ndims);
        off /= step_size;

        for (int d = start_dim; d < prb_.ndims; ++d) {
            i_off += is(d);
            o_off += os(d);
            s_off += ss(d);

            if (off % n(d)) break;

            i_off += -n(d) * is(d);
            o_off += -n(d) * os(d);
            s_off += -n(d) * ss(d);
            off /= n(d);

            if (off == 0) break; /* FIXME: is it really required? */
        }
    }

    void step(int off, int prev_i_off, int prev_o_off, int &i_off, int &o_off,
            int step_size = 1) {
        int dummy = 0;
        step(off, prev_i_off, prev_o_off, dummy, i_off, o_off, dummy,
                step_size);
    }

    void tr8x8_sve256(int i_off, int o_off) {
        using namespace data_type;

        const auto cvt2ps
                = [=](const int startIdx, const int regNum, data_type_t idt) {
                      switch (idt) {
                          case f32:
                              /* do nothing */
                              break;
                          case s32: cvt_z_s32_f32(startIdx, regNum); break;
                          case data_type::s8:
                              cvt_z_s8_s32(startIdx, regNum);
                              cvt_z_s32_f32(startIdx, regNum);
                              break;
                          case u8:
                              cvt_z_u8_s32(startIdx, regNum);
                              cvt_z_s32_f32(startIdx, regNum);
                              break;
                          default: assert(!"unreachable");
                      }
                  };

        const auto cvt2odt = [=](const int startIdx, const int regNum,
                                     data_type_t odt, data_type_t idt) {
            switch (odt) {
                case s32:
                    if (idt == f32)
                        cvt_z_f32_s32(startIdx, regNum);
                    else if (idt == data_type::s8)
                        cvt_z_s8_s32(startIdx, regNum);
                    else if (idt == u8)
                        cvt_z_u8_s32(startIdx, regNum);
                    break;
                case data_type::s8:
                    if (idt == f32) cvt_z_f32_s32(startIdx, regNum);
                    if (utils::one_of(idt, f32, s32))
                        cvt_z_s32_s8(startIdx, regNum);
                    if (idt == u8) cvt_z_u8_s8(startIdx, regNum);
                    break;
                case u8:
                    if (idt == f32) cvt_z_f32_s32(startIdx, regNum);
                    if (utils::one_of(idt, f32, s32))
                        cvt_z_s32_u8(startIdx, regNum);
                    if (idt == data_type::s8) cvt_z_s8_u8(startIdx, regNum);
                    break;
                default: assert(!"unreachable");
            }
        };

        const int unroll = 8;

        const bool interim_f32 = (prb_.itype != f32)
                || utils::one_of(f32, prb_.itype, prb_.otype);

        const bool need_saturation
                = (utils::one_of(prb_.otype, u8, data_type::s8, s32)
                        && interim_f32);
        const uint64_t sveLen = get_sve_length();

        add_imm(X_TMP_0, XReg(x_ptr_in_off), i_off * itype_sz, X_DEFAULT_ADDR);
        add_imm(X_TMP_1, X_TMP_0, is(0) * itype_sz, X_DEFAULT_ADDR);
        add_imm(X_TMP_2, X_TMP_1, is(0) * itype_sz, X_DEFAULT_ADDR);
        add_imm(X_TMP_3, X_TMP_2, is(0) * itype_sz, X_DEFAULT_ADDR);

        if (unroll * itype_sz == 32)
            for (uint32_t i = 0; i < 4; i++)
                ld1w(ZRegS {i}, p_lsb_256 / T_z, ptr(x_tmp_vec[i]));
        else if (unroll * itype_sz == 16)
            for (uint32_t i = 0; i < 4; i++)
                ldr(QReg {i}, ptr(x_tmp_vec[i]));
        else if (unroll * itype_sz == 8)
            for (uint32_t i = 0; i < 4; i++)
                ldr(DReg {i}, ptr(x_tmp_vec[i]));

        add_imm(X_TMP_0, X_TMP_3, is(0) * itype_sz, X_DEFAULT_ADDR);
        add_imm(X_TMP_1, X_TMP_0, is(0) * itype_sz, X_DEFAULT_ADDR);
        add_imm(X_TMP_2, X_TMP_1, is(0) * itype_sz, X_DEFAULT_ADDR);
        add_imm(X_TMP_3, X_TMP_2, is(0) * itype_sz, X_DEFAULT_ADDR);

        if (unroll * itype_sz == 32)
            for (uint32_t i = 0; i < 4; i++)
                ld1w(ZRegS {4 + i}, p_lsb_256 / T_z, ptr(x_tmp_vec[i]));
        else if (unroll * itype_sz == 16)
            for (uint32_t i = 0; i < 4; i++)
                ldr(QReg {4 + i}, ptr(x_tmp_vec[i]));
        else if (unroll * itype_sz == 8)
            for (uint32_t i = 0; i < 4; i++)
                ldr(DReg {4 + i}, ptr(x_tmp_vec[i]));

        if (interim_f32) cvt2ps(0, unroll, prb_.itype);

#if 0
        /* Deubg code */
        index(z0.s, 0, 1);
        mov(z0.s, P_NOT_256/T_m, 0);
        mov(z_tmp_vec[0].s, 16);
        for(uint32_t i=1; i<8; i++) {
          add(ZRegS{i}, ZRegS{i-1}, z_tmp_vec[0].s);
          mov(ZRegS{i}, P_NOT_256/T_m, 0);
        }
#endif

        ptrue(p_tmp0.s, VL4);
        /* 1st turn */
        for (uint32_t i = 0; i < unroll / 2; i++) {
            trn1(z_tmp_vec[i].s, ZRegS {2 * i}, ZRegS {2 * i + 1});
            trn2(z_tmp_vec[unroll / 2 + i].s, ZRegS {2 * i}, ZRegS {2 * i + 1});
        }

        /* 2nd turn */
        trn1(z4.d, z_tmp_vec[0].d, z_tmp_vec[1].d);
        trn1(z5.d, z_tmp_vec[4].d, z_tmp_vec[5].d);
        trn2(z6.d, z_tmp_vec[0].d, z_tmp_vec[1].d);
        trn2(z7.d, z_tmp_vec[4].d, z_tmp_vec[5].d);
        trn1(z_tmp_vec[0].d, z_tmp_vec[2].d, z_tmp_vec[3].d);
        trn1(z_tmp_vec[1].d, z_tmp_vec[6].d, z_tmp_vec[7].d);
        trn2(z_tmp_vec[2].d, z_tmp_vec[2].d, z_tmp_vec[3].d);
        trn2(z_tmp_vec[3].d, z_tmp_vec[6].d, z_tmp_vec[7].d);

        /* 3rd turn */
        for (uint32_t i = 0; i < unroll / 2; i++) {
            mov(ZRegD {i}, ZRegD {unroll / 2 + i});
            mov(z_tmp_vec[unroll / 2 + i].d, z_tmp_vec[i].d);
        }

        /* 4th turn */
        for (uint32_t i = 0; i < unroll / 2; i++) {
            ZRegB z {unroll / 2 + i};
            ZRegB z_tmp = z_tmp_vec[unroll / 2 + i].b;
            /* Move bit 128-255 to 0-127. */
            ext(z, z, 16);
            /* Move bit 0-127 to 128-255. */
            ext(z_tmp, z_tmp, sveLen - 16);
        }

        /* 5th turn */
        for (uint32_t i = 0; i < unroll / 2; i++) {
            ZRegS z0 {i};
            ZRegS z1 {unroll / 2 + i};
            sel(z0, p_tmp0.s, z0, z_tmp_vec[unroll / 2 + i].s);
            sel(z1, p_tmp0, z1, z_tmp_vec[i].s);
        }

        if (need_saturation) {
            init_saturate_f32(ymm_zero, ymm_saturation_ubound, reg_tmp,
                    interim_f32 ? f32 : prb_.itype, prb_.otype);
            for (int i = 0; i < unroll; i++)
                saturate_f32(ZRegS(i), ymm_zero, ymm_saturation_ubound,
                        prb_.otype, p_all);
        }

        if (prb_.otype != f32)
            cvt2odt(0, unroll, prb_.otype, interim_f32 ? f32 : prb_.itype);

        add_imm(X_TMP_0, XReg(x_ptr_out_off), o_off * otype_sz, X_DEFAULT_ADDR);
        add_imm(X_TMP_1, X_TMP_0, os(1) * otype_sz, X_DEFAULT_ADDR);
        add_imm(X_TMP_2, X_TMP_1, os(1) * otype_sz, X_DEFAULT_ADDR);
        add_imm(X_TMP_3, X_TMP_2, os(1) * otype_sz, X_DEFAULT_ADDR);

        if (unroll * otype_sz == 32)
            for (uint32_t i = 0; i < 4; i++)
                st1w(ZRegS {i}, p_lsb_256 / T_z, ptr(x_tmp_vec[i]));
        else if (unroll * otype_sz == 16)
            for (uint32_t i = 0; i < 4; i++)
                str(QReg {i}, ptr(x_tmp_vec[i]));
        else if (unroll * otype_sz == 8)
            for (uint32_t i = 0; i < 4; i++)
                str(DReg {i}, ptr(x_tmp_vec[i]));

        add_imm(X_TMP_0, X_TMP_3, os(1) * otype_sz, X_DEFAULT_ADDR);
        add_imm(X_TMP_1, X_TMP_0, os(1) * otype_sz, X_DEFAULT_ADDR);
        add_imm(X_TMP_2, X_TMP_1, os(1) * otype_sz, X_DEFAULT_ADDR);
        add_imm(X_TMP_3, X_TMP_2, os(1) * otype_sz, X_DEFAULT_ADDR);

        if (unroll * otype_sz == 32)
            for (uint32_t i = 0; i < 4; i++)
                st1w(ZRegS {4 + i}, p_lsb_256 / T_z, ptr(x_tmp_vec[i]));
        else if (unroll * otype_sz == 16)
            for (uint32_t i = 0; i < 4; i++)
                str(QReg {4 + i}, ptr(x_tmp_vec[i]));
        else if (unroll * otype_sz == 8)
            for (uint32_t i = 0; i < 4; i++)
                str(DReg {4 + i}, ptr(x_tmp_vec[i]));
    }

    bool can_do_tr8x8() {
        using namespace data_type;

        return get_sve_length() >= Xbyak_aarch64::util::SVE_256
                && prb_.ndims >= 2
                && ((utils::one_of(prb_.itype, u8, data_type::s8, s32, f32)
                        && utils::one_of(
                                prb_.otype, u8, data_type::s8, s32, f32)))
                && utils::everyone_is(8, n(0), n(1))
                && utils::everyone_is(1, os(0), is(1))
                && prb_.scale_type == scale_type_t::NONE && prb_.beta == 0.f;
    }

    bool process_unroll_tr8x8(int len) {
        if (!can_do_tr8x8()) return false;

        const int step_size = n(0) * n(1);
        int i_off = 0, o_off = 0;
        for (int off = 0; off < len; off += step_size) {
            step(off, i_off, o_off, i_off, o_off, step_size);
            tr8x8_sve256(i_off, o_off);
        }

        return true;
    }

    template <cpu_isa_t isa>
    bool process_direct_copy(int len) {
        using namespace data_type;

        const int simd_w = cpu_isa_traits<isa>::vlen == 16
                ? cpu_isa_traits<isa>::vlen / itype_sz /* use 128-bit VReg */
                : cpu_isa_traits<isa>::vlen / itype_sz
                        / 2; /* use lower half of 512-bit ZReg */

        bool can_do = true && mayiuse(isa)
                && utils::everyone_is(1, os(0), is(0))
                && (false || prb_.itype == prb_.otype
                        || (prb_.itype == s32 && prb_.otype == f32)
                        || (prb_.itype == f32 && prb_.otype == s32))
                && len % simd_w == 0 && n(0) % len == 0
                && prb_.scale_type == scale_type_t::NONE && prb_.beta == 0.f;
        if (!can_do) return false;

        for (int off = 0; off < len;) {
            const int unroll
                    = nstl::min(16 - (prb_.otype == s32), (len - off) / simd_w);

            int ur = 0;
            int tmp_ur = 0;
            while (ur < unroll) {
                int count = 0;
                const int vlen = cpu_isa_traits<isa>::vlen;

                do {
                    add_imm(x_tmp_vec[count++], x_ptr_in_off,
                            (off + ur * simd_w) * itype_sz, X_DEFAULT_ADDR);
                    ur++;
                } while (ur < unroll && count < x_tmp_vec_size);

                for (int i = 0; i < count; i++) {
                    /*                    if (vlen == 64)
                        ldr(ZReg(tmp_ur + i), ptr(x_tmp_vec[i]));
                        else */
                    if (vlen == 64 || vlen == 32)
                        ld1w(ZRegS(tmp_ur + i), p_lsb_256 / T_z,
                                ptr(x_tmp_vec[i]));
                    else if (vlen == 16)
                        ldr(QReg(tmp_ur + i), ptr(x_tmp_vec[i]));
                    else
                        assert(!"unreachable");
                }
                tmp_ur += count;
            }

            if (prb_.itype != prb_.otype) {
                const int vlen = cpu_isa_traits<isa>::vlen;
                for (int ur = 0; ur < unroll; ++ur) {
                    if (prb_.itype == s32 && prb_.otype == f32) {
                        if (vlen == 64 || vlen == 32) {
                            ZRegS r(ur);
                            /* MSB side 256 bits are ignored. */
                            scvtf(r, p_all / T_m, r);
                        } else if (vlen == 16) {
                            VReg4S r(ur);
                            scvtf(r, r);
                        } else
                            assert(!"unreachable");
                    } else if (prb_.itype == f32 && prb_.otype == s32) {
                        /* Out of order can be expected. */
                        if (vlen == 64 || vlen == 32) {
                            ZRegS r(ur);
                            frinti(r, p_all / T_m, r);
                            fcvtzs(r, p_all / T_m, r);
                        } else if (vlen == 16) {
                            VReg4S r(ur);
                            frinti(r, r);
                            fcvtzs(r, r);
                        } else
                            assert(!"unreachable");
                    } else
                        assert(!"unreachable");
                }
            }

            ur = 0;
            tmp_ur = 0;
            while (ur < unroll) {
                int count = 0;
                const int vlen = cpu_isa_traits<isa>::vlen;

                do {
                    add_imm(x_tmp_vec[count++], x_ptr_out_off,
                            (off + ur * simd_w) * otype_sz, X_DEFAULT_ADDR);
                    ur++;
                } while (ur < unroll && count < x_tmp_vec_size);

                for (int i = 0; i < count; i++) {
                    if (vlen == 64 || vlen == 32)
                        st1w(ZRegS(tmp_ur + i), p_lsb_256 / T_z,
                                ptr(x_tmp_vec[i]));
                    else if (vlen == 16)
                        str(QReg(tmp_ur + i), ptr(x_tmp_vec[i]));
                    else
                        assert(!"unreachable");
                }
                tmp_ur += count;
            }

            off += unroll * simd_w;
        }

        return true;
    }

    void process_unroll_generic_step(int reg_unroll, const int *i_off,
            const int *o_off, const int *s_off) {
        using namespace data_type;

        auto cvt2ps
                = [=](const int startIdx, const int regNum, data_type_t idt) {
                      switch (idt) {
                          case f32:
                              /* do nothing */
                              break;
                          case s32: cvt_v_s32_f32(startIdx, regNum); break;
                          case data_type::s8:
                              cvt_v_s8_s32(startIdx, regNum);
                              cvt_v_s32_f32(startIdx, regNum);
                              break;
                          case u8:
                              cvt_v_u8_s32(startIdx, regNum);
                              cvt_v_s32_f32(startIdx, regNum);
                              break;
                          default: assert(!"unreachable");
                      }
                  };

        auto cvt2odt = [=](const int startIdx, const int regNum,
                               data_type_t odt, data_type_t idt) {
            switch (odt) {
                case s32:
                    if (idt == f32)
                        cvt_v_f32_s32(startIdx, regNum);
                    else if (idt == data_type::s8)
                        cvt_v_s8_s32(startIdx, regNum);
                    else if (idt == u8)
                        cvt_v_u8_s32(startIdx, regNum);
                    break;
                case data_type::s8:
                    if (idt == f32) cvt_v_f32_s32(startIdx, regNum);
                    if (idt == f32 || idt == s32)
                        cvt_v_s32_s8(startIdx, regNum);
                    if (idt == u8) { cvt_v_u8_s8(startIdx, regNum); }
                    break;
                case u8:
                    if (idt == f32) cvt_v_f32_s32(startIdx, regNum);
                    if (idt == f32 || idt == s32)
                        cvt_v_s32_u8(startIdx, regNum);
                    if (idt == data_type::s8) cvt_v_s8_u8(startIdx, regNum);
                    break;
                default: assert(!"unreachable");
            }
        };

        /* check whether loading 4 values at once is possible */
        bool can_load_xmm = reg_unroll % 4 == 0;
        for (int ur = 1; ur < reg_unroll; ++ur)
            if (i_off[ur] != i_off[ur - 1] + 1) can_load_xmm = false;
        const int load_step = can_load_xmm ? 4 : 1;

        /* check whether storing 4 values at once is possible */
        bool can_store_xmm = reg_unroll % 4 == 0;
        for (int ur = 1; ur < reg_unroll; ++ur)
            if (o_off[ur] != o_off[ur - 1] + 1) can_store_xmm = false;
        const int ur_step = can_store_xmm ? 4 : 1;

        const bool interim_f32 = false
                || utils::one_of(f32, prb_.itype, prb_.otype)
                || prb_.scale_type != scale_type_t::NONE || prb_.beta != 0.f;

        const bool need_saturation
                = (utils::one_of(prb_.otype, u8, data_type::s8, s32)
                        && interim_f32);

        if (!can_load_xmm && can_store_xmm) {
            assert(ur_step == 4);
            /* load with stride */
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {

                /* x_tmp_vec = X_TMP_0 - X_TMP_4
                 Do not use X_TMP_? as the last arg. */
                for (int r = 0; r < ur_step; ++r) {
                    add_imm(x_tmp_vec[r], x_ptr_in_off,
                            i_off[ur + r] * itype_sz, X_DEFAULT_ADDR);
                }

                for (int r = 0; r < ur_step; ++r) {
                    if (itype_sz == 4)
                        ld1(VReg4S(ur)[r], ptr(x_tmp_vec[r]));
                    else if (itype_sz == 2)
                        ld1(VReg8H(ur)[r], ptr(x_tmp_vec[r]));
                    else
                        ld1(VReg16B(ur)[r], ptr(x_tmp_vec[r]));
                }
            }
        } else {
            int ur = 0;
            int tmp_ur = 0;
            while (ur < reg_unroll) {
                int count = 0;

                do {
                    add_imm(x_tmp_vec[count++], x_ptr_in_off,
                            i_off[ur] * itype_sz, X_DEFAULT_ADDR);
                    ur += load_step;
                } while (ur < reg_unroll && count < x_tmp_vec_size);

                for (int i = 0; i < count; i++) {

                    switch (load_step * itype_sz) {
                        case 16: ldr(QReg(tmp_ur), ptr(x_tmp_vec[i])); break;
                        case 8: ldr(DReg(tmp_ur), ptr(x_tmp_vec[i])); break;
                        case 4: ldr(SReg(tmp_ur), ptr(x_tmp_vec[i])); break;
                        case 2: ldr(HReg(tmp_ur), ptr(x_tmp_vec[i])); break;
                        case 1: ldr(BReg(tmp_ur), ptr(x_tmp_vec[i])); break;
                        default: assert(!"unreachable");
                    }
                    tmp_ur += load_step;
                }
            }
        }

        /* xmm[:] <-- (f32)xmm[:] */
        if (interim_f32) {
            const int cvt_step = nstl::max(load_step, ur_step);
            for (int ur = 0; ur < reg_unroll; ur += cvt_step)
                cvt2ps(ur, 1, prb_.itype);
        }

        if (can_load_xmm && !can_store_xmm) {
            const bool fast_return = true // transposition on the fly
                    && prb_.scale_type != scale_type_t::MANY
                    && prb_.beta == 0.f;
            if (fast_return) {
                if (prb_.scale_type == scale_type_t::COMMON)
                    for (int ur = 0; ur < reg_unroll; ur += load_step)
                        fmul(VReg4S(ur), VReg4S(ur), xmm_scale);
                if (prb_.otype != f32) {
                    init_saturate_f32(xmm_zero, xmm_saturation_ubound, reg_tmp,
                            interim_f32 ? f32 : prb_.itype, prb_.otype);
                    for (int ur = 0; ur < reg_unroll; ur += load_step)
                        if (need_saturation)
                            saturate_f32(VReg4S(ur), xmm_zero,
                                    xmm_saturation_ubound, prb_.otype, p_all);

                    for (int ur = 0; ur < reg_unroll; ur += load_step)
                        cvt2odt(ur, 1, prb_.otype,
                                interim_f32 ? f32 : prb_.itype);
                }
                /* load_step is 1 or 4. */
                for (int ur = 0; ur < reg_unroll; ur += load_step) {
                    for (int r = 0; r < load_step; ++r) {
                        add_imm(x_tmp_vec[r], x_ptr_out_off,
                                o_off[ur + r] * otype_sz, X_DEFAULT_ADDR);
                    }

                    for (int r = 0; r < load_step; ++r) {
                        if (otype_sz == 4)
                            st1(VReg4S(ur)[r], ptr(x_tmp_vec[r]));
                        else if (otype_sz == 2)
                            st1(VReg8H(ur)[r], ptr(x_tmp_vec[r]));
                        else
                            st1(VReg16B(ur)[r], ptr(x_tmp_vec[r]));
                    }
                }
                return;
            }

            /* scatter elements of xmm into 4 xmms */
            if (itype_sz == 4 || interim_f32) {
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    for (int r = 1; r < load_step; ++r) {
                        VReg4S v(ur);
                        VReg4S v_r(ur + r);
                        dup(VReg16B(ur + r), VReg16B(ur)[0]);
                        ins(VReg4S(ur + r)[0], VReg4S(ur)[r]);
                    }
            } else {
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    for (int r = 1; r < load_step; ++r)
                        ext(VReg16B(ur + r), VReg16B(ur), VReg16B(ur),
                                itype_sz * r);
            }
        }

        /* scale and beta processing */
        if (can_store_xmm) {
            /* xmm <-- scale * xmm[:] */
            if (prb_.scale_type == scale_type_t::COMMON) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step)
                    fmul(VReg4S(ur), VReg4S(ur), xmm_scale);
            } else if (prb_.scale_type == scale_type_t::MANY) {
                enum class scale_load_type_t { bcast, load, gather };

                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    scale_load_type_t scale_load_type
                            = scale_load_type_t::bcast; // the best case

                    for (int r = ur + 1; r < ur + ur_step; ++r)
                        if (s_off[r] != s_off[r - 1] + 0)
                            scale_load_type = scale_load_type_t::load;

                    if (scale_load_type == scale_load_type_t::bcast) {
                        VReg4S v(xmm_scale.getIdx());
                        VReg4S v_dst(ur);
                        add_imm(X_TMP_0, x_ptr_scale_off, s_off[ur] * stype_sz,
                                X_DEFAULT_ADDR);
                        ldr(W_TMP_0, ptr(X_TMP_0));
                        dup(v, W_TMP_0);
                        fmul(v_dst, v_dst, v);
                        continue;
                    }

                    // bcast doesn't work, the next try -- load
                    for (int r = ur + 1; r < ur + ur_step; ++r)
                        if (s_off[r] != s_off[r - 1] + 1)
                            scale_load_type = scale_load_type_t::gather;

                    if (scale_load_type == scale_load_type_t::load) {
                        uint32_t idx = xmm_scale.getIdx();
                        VReg4S v_dst(ur);
                        add_imm(X_TMP_0, x_ptr_scale_off, s_off[ur] * stype_sz,
                                X_DEFAULT_ADDR);

                        ldr(QReg {idx}, ptr(X_TMP_0));
                        fmul(v_dst, v_dst, VReg4S {idx});
                        continue;
                    }

                    // load doesn't work as well
                    // so gather the scale factors one by one
                    /*ur_step is 1 or 4. */
                    for (int r = ur; r < ur + ur_step; ++r) {
                        /* x_tmp_vec = X_TMP_0 - X_TMP_4
                         Do not use X_TMP_? as the last arg. */
                        add_imm(x_tmp_vec[r - ur], x_ptr_scale_off,
                                s_off[r] * stype_sz, X_DEFAULT_ADDR);
                    }
                    for (int r = ur; r < ur + ur_step; ++r) {
                        VReg4S v(xmm_scale.getIdx());
                        ld1(v[r - ur], ptr(x_tmp_vec[r - ur]));
                    }
                    fmul(VReg4S(ur), VReg4S(ur), xmm_scale);
                }
            }

            /* dst <-- beta * dst + xmm[:] */
            assert(prb_.beta == 0.f || prb_.beta == 1.f);
            if (prb_.beta == 1.f) {
                int ur = 0;
                int tmp_ur = 0;

                while (ur < reg_unroll) {
                    int count = 0;

                    do {
                        add_imm(x_tmp_vec[count++], x_ptr_out_off,
                                o_off[ur] * otype_sz, X_DEFAULT_ADDR);
                        ur += ur_step;
                    } while (ur < reg_unroll && count < x_tmp_vec_size);

                    assert(count <= z_tmp_vec_size);
                    /* Firstly, data is loaded. */
                    for (int i = 0; i < count; i++) {

                        if (prb_.otype == f32 || prb_.otype == s32) {
                            ldr(QReg(tmp_vec_idx[i]), ptr(x_tmp_vec[i])); // bug
                        } else if (prb_.otype == data_type::s8
                                || prb_.otype == u8) {
                            ldr(SReg(tmp_vec_idx[i]), ptr(x_tmp_vec[i])); // bug
                        } else
                            assert(!"unreachable");
                    }

                    /* Secondly, it is added. */
                    if (prb_.otype == f32) {
                        for (int i = 0; i < count; i++) {
                            VReg4S v(tmp_ur);
                            fadd(v, v, VReg4S(tmp_vec_idx[i]));
                            tmp_ur += ur_step;
                        }
                    } else {
                        for (int i = 0; i < count; i++) {
                            /* cvt2ps() generate successive instructions
                               which have save destination operand,
                               but out of order can be expected. */
                            cvt2ps(tmp_vec_idx[i], 1, prb_.otype);
                        }
                        for (int i = 0; i < count; i++) {
                            VReg4S v(tmp_ur);
                            fadd(v, v, VReg4S(tmp_vec_idx[i]));
                            tmp_ur += ur_step;
                        }
                    }
                }
            }
        } else {
            /* xmm[0] <-- scale * xmm[0] */
            if (prb_.scale_type == scale_type_t::COMMON) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    VReg4S tmp(ur);
                    fmul(tmp, tmp, VReg4S(xmm_scale.getIdx()));
                }
            } else if (prb_.scale_type == scale_type_t::MANY) {
                int ur = 0;
                int tmp_ur = 0;
                while (ur < reg_unroll) {
                    int count = 0;

                    do {
                        add_imm(x_tmp_vec[count++], x_ptr_scale_off,
                                s_off[ur] * stype_sz, X_DEFAULT_ADDR);
                        ur += ur_step;
                    } while (ur < reg_unroll && count < x_tmp_vec_size);

                    for (int i = 0; i < count; i++)
                        ldr(SReg(tmp_vec_idx[i]), ptr(x_tmp_vec[i]));
                    for (int i = 0; i < count; i++) {
                        VReg4S tmp(tmp_ur + ur_step * i);
                        fmul(tmp, tmp, VReg4S(tmp_vec_idx[i]));
                    }

                    tmp_ur += ur_step * count;
                }
            }

            /* dst <-- beta * dst + xmm[0] */
            assert(prb_.beta == 0.f || prb_.beta == 1.f);
            if (prb_.beta == 1.f) {
                int ur = 0;
                int tmp_ur = 0;
                while (ur < reg_unroll) {
                    int count = 0;

                    do {
                        add_imm(x_tmp_vec[count++], x_ptr_out_off,
                                o_off[ur] * otype_sz, X_DEFAULT_ADDR);
                        ur += ur_step;
                    } while (ur < reg_unroll && count < (x_tmp_vec_size / 2));

                    assert(static_cast<size_t>(count) <= z_tmp_vec.size());

                    if (prb_.otype == f32) {
                        /* addss: dest[31:0] <- src1[31:0] + src2[31:0]
                         dset[MAXVL-1:32] (Unmodified) */
                        for (int i = 0; i < count; i++) {
                            ld1(VReg4S(z_tmp_vec[i].getIdx())[0],
                                    ptr(x_tmp_vec[i]));
                        }
                        for (int i = 0; i < count; i++) {
                            SReg s {tmp_vec_idx[i]};
                            fadd(s, s, SReg(tmp_ur + ur_step * i));
                        }
                        for (int i = 0; i < count; i++) {
                            mov(VReg4S(tmp_ur)[0], VReg4S(tmp_vec_idx[i])[0]);
                            tmp_ur += ur_step;
                        }
                    } else {
                        for (int i = 0; i < count; i++) {
                            if (prb_.otype == s32) {
                                ldr(SReg(tmp_vec_idx[i]), ptr(x_tmp_vec[i]));
                            } else if (utils::one_of(
                                               prb_.otype, data_type::s8, u8)) {
                                ldr(BReg(tmp_vec_idx[i]), ptr(x_tmp_vec[i]));
                            } else {
                                assert(!"unsupported o_type");
                            }
                            cvt2ps(tmp_vec_idx[i], 1, prb_.otype);
                        }
                        for (int i = 0; i < count; i++) {
                            VReg4S v(tmp_ur);
                            fadd(v, v, VReg4S(tmp_vec_idx[i]));
                            tmp_ur += ur_step;
                        }
                    }
                }
            }
        }

        if (need_saturation) {
            init_saturate_f32(
                    xmm_zero, xmm_saturation_ubound, reg_tmp, f32, prb_.otype);
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                saturate_f32(VReg4S(ur), xmm_zero, xmm_saturation_ubound,
                        prb_.otype, p_all);
            }
        }

        for (int ur = 0; ur < reg_unroll; ur += ur_step) {
            if (prb_.otype != f32)
                cvt2odt(ur, 1, prb_.otype, interim_f32 ? f32 : prb_.itype);
        }

        int ur = 0;
        int tmp_ur = 0;
        while (ur < reg_unroll) {
            int count = 0;

            do {
                add_imm(x_tmp_vec[count++], x_ptr_out_off, o_off[ur] * otype_sz,
                        X_DEFAULT_ADDR);
                ur += ur_step;
            } while (ur < reg_unroll && count < x_tmp_vec_size);

            for (int i = 0; i < count; i++) {

                switch (ur_step * otype_sz) {
                    case 16: str(QReg(tmp_ur), ptr(x_tmp_vec[i])); break;
                    case 8: str(DReg(tmp_ur), ptr(x_tmp_vec[i])); break;
                    case 4: str(SReg(tmp_ur), ptr(x_tmp_vec[i])); break;
                    case 2: str(HReg(tmp_ur), ptr(x_tmp_vec[i])); break;
                    case 1: str(BReg(tmp_ur), ptr(x_tmp_vec[i])); break;
                    default: assert(!"unreachable");
                }
                tmp_ur += ur_step;
            }
        }
    }

    void process_unroll_generic(int len) {
        const int blk = 8;

        int i_off[2 * blk] = {0};
        int o_off[2 * blk] = {0};
        int s_off[2 * blk] = {0};

        int curr = 0; // will switch between 0 and 1

        for (int off = 0; off < len; off += blk) {
            const int reg_unroll = nstl::min(off + blk, len) - off;

            /* compute offsets */
            for (int ur = off != 0 ? 0 : 1; ur < reg_unroll; ++ur) {
                const int ur_c = curr * blk + ur;
                const int ur_p = (ur_c - 1 + 2 * blk) % (2 * blk); // prev ur
                step(off + ur, i_off[ur_p], o_off[ur_p], s_off[ur_p],
                        i_off[ur_c], o_off[ur_c], s_off[ur_c]);
            }

            process_unroll_generic_step(reg_unroll, i_off + curr * blk,
                    o_off + curr * blk, s_off + curr * blk);

            curr = 1 - curr;
        }
    }

    void loop_begin(Label &l, XReg reg_cnt, int len) {
        mov(reg_cnt, len);
        L(l);
    }

    void loop_end(Label &l, XReg reg_cnt, int len, int i_step, int o_step,
            int s_step) {
        add_imm(reg_off_in, reg_off_in, i_step * itype_sz, X_TMP_0);
        add_imm(reg_off_out, reg_off_out, o_step * otype_sz, X_TMP_0);
        add_imm(x_ptr_in_off, x_ptr_in_off, i_step * itype_sz, X_TMP_0);
        add_imm(x_ptr_out_off, x_ptr_out_off, o_step * otype_sz, X_TMP_0);

        if (prb_.scale_type == scale_type_t::MANY) {
            add_imm(reg_off_scale, reg_off_scale, s_step * stype_sz, X_TMP_0);
            add_imm(x_ptr_scale_off, x_ptr_scale_off, s_step * stype_sz,
                    X_TMP_0);
        }
        subs(reg_cnt, reg_cnt, 1);
        b(NE, l);

        sub_imm(reg_off_in, reg_off_in, len * i_step * itype_sz, X_TMP_0);
        sub_imm(reg_off_out, reg_off_out, len * o_step * otype_sz, X_TMP_0);
        sub_imm(x_ptr_in_off, x_ptr_in_off, len * i_step * itype_sz, X_TMP_0);
        sub_imm(x_ptr_out_off, x_ptr_out_off, len * o_step * otype_sz, X_TMP_0);

        if (prb_.scale_type == scale_type_t::MANY) {
            sub_imm(reg_off_scale, reg_off_scale, len * s_step * stype_sz,
                    X_TMP_0);
            sub_imm(x_ptr_scale_off, x_ptr_scale_off, len * s_step * stype_sz,
                    X_TMP_0);
        }
    }

    bool simple_impl() {
        simple_impl_desc_t d;
        if (!simple_impl_desc_init(prb_, &d)) return false;

        const int nfu = d.ndims_full_unroll;
        const int ldu = d.len_last_dim_unroll;
        const int n_jit_loops = prb_.ndims - d.ndims_full_unroll;
        assert(n_jit_loops <= ndims_jit_loop_max);

        eor(reg_off_in, reg_off_in, reg_off_in);
        eor(reg_off_out, reg_off_out, reg_off_out);
        mov(x_ptr_in_off, XReg(reg_ptr_in.getIdx()));
        mov(x_ptr_out_off, XReg(reg_ptr_out.getIdx()));
        if (prb_.scale_type == scale_type_t::MANY) {
            eor(reg_off_scale, reg_off_scale, reg_off_scale);
            mov(x_ptr_scale_off, XReg(reg_ptr_scale.getIdx()));
        }

        Label l_loop[3];
        XReg reg_cnt[3] = {x15, x14, x13};

        if (n_jit_loops > 2) loop_begin(l_loop[2], reg_cnt[2], n(nfu + 2));

        if (n_jit_loops > 1) loop_begin(l_loop[1], reg_cnt[1], n(nfu + 1));

        if (n_jit_loops > 0)
            loop_begin(l_loop[0], reg_cnt[0], n(nfu + 0) / ldu);

        bool optimized = false;
        optimized = optimized || process_direct_copy<sve_512>(d.len_unroll);
        optimized = optimized || process_direct_copy<asimd>(d.len_unroll);
        optimized = optimized || process_unroll_tr8x8(d.len_unroll);
        if (!optimized) process_unroll_generic(d.len_unroll);

        if (n_jit_loops > 0)
            loop_end(l_loop[0], reg_cnt[0], n(nfu + 0) / ldu, is(nfu + 0) * ldu,
                    os(nfu + 0) * ldu, ss(nfu + 0) * ldu);

        if (n_jit_loops > 1)
            loop_end(l_loop[1], reg_cnt[1], n(nfu + 1), is(nfu + 1),
                    os(nfu + 1), ss(nfu + 1));

        if (n_jit_loops > 2)
            loop_end(l_loop[2], reg_cnt[2], n(nfu + 2), is(nfu + 2),
                    os(nfu + 2), ss(nfu + 2));

        return true;
    }

    void impl() {
        if (simple_impl()) return;
        assert(!"no implementation available");
    }

#define UNROLL_INST(inst, reg, ...) \
    for (size_t i = startIdx; i < startIdx + regNum; i++) { \
        reg tmp(i); \
        inst(__VA_ARGS__); \
    }
#define UNROLL_INST2(inst, ...) \
    for (size_t i = startIdx; i < startIdx + regNum; i++) \
        inst(__VA_ARGS__);

    void cvt_z_s32_f32(const size_t startIdx, const size_t regNum) {
        UNROLL_INST(scvtf, ZRegS, tmp, p_all / T_m, tmp);
    }

    void cvt_v_s32_f32(const size_t startIdx, const size_t regNum) {
        UNROLL_INST(scvtf, VReg4S, tmp, tmp);
    }

    void cvt_z_f32_s32(const size_t startIdx, const size_t regNum) {
        UNROLL_INST(frinti, ZRegS, tmp, p_all / T_m, tmp);
        UNROLL_INST(fcvtzs, ZRegS, tmp, p_all / T_m, tmp);
    }

    void cvt_v_f32_s32(const size_t startIdx, const size_t regNum) {
        UNROLL_INST(frinti, VReg4S, tmp, tmp);
        UNROLL_INST(fcvtzs, VReg4S, tmp, tmp);
    }

    void cvt_z_s8_s32(const size_t startIdx, const size_t regNum) {
        cvt_z_b_s(startIdx, regNum);
        UNROLL_INST(sxtb, ZRegS, tmp, p_all / T_m, tmp);
    }

    void cvt_v_s8_s32(const size_t startIdx, const size_t regNum) {
        UNROLL_INST(sxtl, VReg, tmp.h8, tmp.b8);
        UNROLL_INST(sxtl, VReg, tmp.s4, tmp.h4);
    }

    void cvt_z_s8_f32(const size_t startIdx, const size_t regNum) {
        cvt_z_b_s(startIdx, regNum);
        cvt_z_s32_f32(startIdx, regNum);
    }

    void cvt_v_s8_f32(const size_t startIdx, const size_t regNum) {
        cvt_v_b_s(startIdx, regNum);
        cvt_v_s32_f32(startIdx, regNum);
    }

    void cvt_z_b_s(const size_t startIdx, const size_t regNum) {
        assert(z_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < z_tmp7.getIdx());

        dup(z_tmp7.b, 0);
        UNROLL_INST(zip1, ZRegB, tmp, tmp, z_tmp7.b);
        UNROLL_INST(zip1, ZRegH, tmp, tmp, z_tmp7.h);
    }

    void cvt_v_b_s(const size_t startIdx, const size_t regNum) {
        assert(v_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < v_tmp7.getIdx());

        mov_imm(W_TMP_0, 0);
        dup(v_tmp7.b16, W_TMP_0);
        UNROLL_INST(zip1, VReg16B, tmp, tmp, v_tmp7.b16);
        UNROLL_INST(zip1, VReg8H, tmp, tmp, v_tmp7.h8);
    }

    void cvt_z_u8_s32(const size_t startIdx, const size_t regNum) {
        cvt_z_b_s(startIdx, regNum);
        UNROLL_INST(uxtb, ZRegS, tmp, p_all / T_m, tmp);
    }

    void cvt_v_u8_s32(const size_t startIdx, const size_t regNum) {
        UNROLL_INST(uxtl, VReg, tmp.h8, tmp.b8);
        UNROLL_INST(uxtl, VReg, tmp.s4, tmp.h4);
    }

    void cvt_z_s32_s8(const size_t startIdx, const size_t regNum) {
        assert(z_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < z_tmp7.getIdx());

        dup(z_tmp7.s, 0);
        UNROLL_INST2(smin, ZRegS(i), 127);
        UNROLL_INST2(smax, ZRegS(i), -128);
        UNROLL_INST(uzp1, ZRegH, tmp, tmp, z_tmp7.h);
        UNROLL_INST(uzp1, ZRegB, tmp, tmp, z_tmp7.b);
    }

    void cvt_v_s32_s8(const size_t startIdx, const size_t regNum) {
        assert(v_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < v_tmp7.getIdx());

        mov_imm(W_TMP_0, 127);
        dup(v_tmp7.s4, W_TMP_0);
        mov_imm(W_TMP_0, -128);
        UNROLL_INST2(smin, VReg4S(i), VReg4S(i), v_tmp7.s4);
        dup(v_tmp7.s4, W_TMP_0);
        UNROLL_INST2(smax, VReg4S(i), VReg4S(i), v_tmp7.s4);
        mov_imm(W_TMP_0, 0);
        dup(v_tmp7.s4, W_TMP_0);
        UNROLL_INST(uzp1, VReg8H, tmp, tmp, v_tmp7.h8);
        UNROLL_INST(uzp1, VReg16B, tmp, tmp, v_tmp7.b16);
    }

    void cvt_z_u8_s8(const size_t startIdx, const size_t regNum) {
        UNROLL_INST2(umin, ZRegB(i), 127);
    }

    void cvt_v_u8_s8(const size_t startIdx, const size_t regNum) {
        assert(v_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < v_tmp7.getIdx());

        mov_imm(W_TMP_0, 127);
        dup(v_tmp7.b16, W_TMP_0);
        UNROLL_INST(umin, VReg16B, tmp, tmp, v_tmp7.b16);
    }

    void cvt_z_u32_u8(const size_t startIdx, const size_t regNum) {
        UNROLL_INST2(umin, ZRegS(i), 255);
        UNROLL_INST(uzp1, ZRegH, tmp, tmp, tmp);
        UNROLL_INST(uzp1, ZRegB, tmp, tmp, tmp);
    }

    void cvt_v_u32_u8(const size_t startIdx, const size_t regNum) {
        assert(v_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < v_tmp7.getIdx());

        mov_imm(W_TMP_0, 255);
        dup(v_tmp7.s4, W_TMP_0);
        UNROLL_INST(umin, VReg4S, tmp, tmp, v_tmp7.s4);
        UNROLL_INST(uzp1, VReg8H, tmp, tmp, tmp);
        UNROLL_INST(uzp1, VReg16B, tmp, tmp, tmp);
    }

    void cvt_z_s32_u8(const size_t startIdx, const size_t regNum) {
        assert(z_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < z_tmp7.getIdx());

        dupm(z_tmp7.s, 255);
        UNROLL_INST2(smax, ZRegS(i), 0);
        UNROLL_INST2(smin, ZRegS(i), p_all / T_m, z_tmp7.s);
        UNROLL_INST(uzp1, ZRegH, tmp, tmp, tmp);
        UNROLL_INST(uzp1, ZRegB, tmp, tmp, tmp);
        UNROLL_INST2(mov, ZRegB(i), P_NOT_128 / T_m, 0);
    }

    void cvt_v_s32_u8(const size_t startIdx, const size_t regNum) {
        assert(v_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < v_tmp7.getIdx());

        mov_imm(W_TMP_0, 0);
        dup(v_tmp7.s4, W_TMP_0);
        mov_imm(W_TMP_0, 255);
        UNROLL_INST(smax, VReg4S, tmp, tmp, v_tmp7.s4);
        dup(v_tmp7.s4, W_TMP_0);
        UNROLL_INST(smin, VReg4S, tmp, tmp, v_tmp7.s4);
        UNROLL_INST(uzp1, VReg8H, tmp, tmp, tmp);
        UNROLL_INST(uzp1, VReg16B, tmp, tmp, tmp);
    }

    void cvt_z_s8_u8(const size_t startIdx, const size_t regNum) {
        UNROLL_INST2(smax, ZRegB(i), 0);
    }

    void cvt_v_s8_u8(const size_t startIdx, const size_t regNum) {
        assert(v_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < v_tmp7.getIdx());

        mov_imm(W_TMP_0, 0);
        dup(v_tmp7.b16, W_TMP_0);
        UNROLL_INST(smax, VReg16B, tmp, tmp, v_tmp7.b16);
    }
#undef UNROLL_INST
#undef UNROLL_INST

    jit_uni_reorder_kernel_f32_t(const desc_t &desc) : kernel_t(desc) {
        itype_sz = data_type_size(prb_.itype);
        otype_sz = data_type_size(prb_.otype);
        stype_sz = sizeof(float);
    }

    void generate() override {
        using namespace Xbyak_aarch64::util;
        uint64_t sveLen = get_sve_length();

        preamble();
#define PARAM(x) offsetof(call_param_t, x)
        if (prb_.scale_type == scale_type_t::COMMON) {
            add_imm(X_DEFAULT_ADDR, abi_param1, PARAM(scale), X_TMP_1);
            ldr(X_TMP_0, ptr(X_DEFAULT_ADDR));
            ldr(W_TMP_1, ptr(X_TMP_0));
            dup(xmm_scale, W_TMP_1);
        } else if (prb_.scale_type == scale_type_t::MANY) {
            add_imm(X_DEFAULT_ADDR, abi_param1, PARAM(scale), X_TMP_0);
            ldr(reg_ptr_scale, ptr(X_DEFAULT_ADDR));
        }
        add_imm(X_TMP_0, abi_param1, PARAM(in), X_TMP_2);
        add_imm(X_TMP_1, abi_param1, PARAM(out), X_TMP_2);
        ldr(reg_ptr_in, ptr(X_TMP_0));
        ldr(reg_ptr_out, ptr(X_TMP_1));
#undef PARAM

        mov(x_ptr_in_off, XReg(reg_ptr_in.getIdx()));
        mov(x_ptr_out_off, XReg(reg_ptr_out.getIdx()));
        mov(x_ptr_scale_off, XReg(reg_ptr_scale.getIdx()));

        if (sveLen) { /* SVE is available. */
            ptrue(p_lsb_256.b, VL32);
            ptrue(p_all.b);
        }

        if (can_do_tr8x8()) {
            dup(ymm_zero, 0);

            if (prb_.itype == data_type::u8 && prb_.otype == data_type::s8) {
                mov_imm(reg_tmp, 0x7f7f7f7f7f7f7f7f);
                mov(VReg4S(ymm_8x127b.getIdx())[0], WReg(reg_tmp.getIdx()));
            }
        } else if (mayiuse(sve_512)) {
            movi(xmm_zero, 0);

            if (prb_.itype == data_type::u8 && prb_.otype == data_type::s8) {
                mov(WReg(reg_tmp.getIdx()), 0x7f7f7f7f);
                mov(xmm_4x127b[0], WReg(reg_tmp.getIdx()));
            }
        }

        impl();
        postamble();
    }

private:
    int itype_sz;
    int otype_sz;
    int stype_sz;

    XReg reg_ptr_in = x6;
    XReg reg_ptr_out = x2;
    XReg reg_ptr_scale = abi_not_param1;

    XReg reg_off_in = x8;
    XReg reg_off_out = x9;
    XReg reg_off_scale = x10;

    XReg reg_tmp = x0;

    VReg4S xmm_scale = v15.s;
    VReg4S xmm_zero = v14.s;
    VReg4S xmm_4x127b = v13.s; // TODO: unite with ymm_zero
    ZRegS ymm_zero = z14.s;
    ZRegS ymm_8x127b = z13.s;
    VReg4S xmm_tmp = v12.s;
    VReg4S xmm_saturation_ubound = v12.s;
    ZRegS ymm_saturation_ubound = z12.s;

    /* Note: x22 - x28 are already used as temporal registgers
       in jit_generator.hpp.
       x_ptr_(in|out|scale)_off keeps (base + offset) address. */
    XReg x_ptr_in_off = x16;
    XReg x_ptr_out_off = x18;
    XReg x_ptr_scale_off = x20;

    /* Caution: Chose predicate registers not used by x64's implementation. */
    PReg p_lsb_256 = p7;
    PReg p_all = p6;
    PReg p_tmp0 = p5;

    const std::vector<uint32_t> tmp_vec_idx = {20, 21, 22, 23, 24, 25, 26, 27};
    ZReg z_tmp0 = z20;
    ZReg z_tmp1 = z21;
    ZReg z_tmp2 = z22;
    ZReg z_tmp3 = z23;
    ZReg z_tmp4 = z24;
    ZReg z_tmp5 = z25;
    ZReg z_tmp6 = z26;
    ZReg z_tmp7 = z27;
    VReg v_tmp7 = v27;

    const std::vector<ZReg> z_tmp_vec
            = {z_tmp0, z_tmp1, z_tmp2, z_tmp3, z_tmp4, z_tmp5, z_tmp6, z_tmp7};
    constexpr static int z_tmp_vec_size = 8;
};

status_t kernel_t::desc_init(
        kernel_t::desc_t &desc, const prb_t &prb, int ndims_ker_max) {
    desc.prb = prb;
    desc.prb.ioff = desc.prb.ooff = 0;

    if (ndims_ker_max > prb.ndims) return status::invalid_arguments;

    auto ndims_ker_max_f = [&]() {
        size_t cur_size = 1;
        for (int d = 0; d < prb.ndims; cur_size *= prb.nodes[d++].n)
            if (cur_size >= ker_prb_size_min) return d;
        return prb.ndims;
    };

    if (ndims_ker_max <= 0) ndims_ker_max = ndims_ker_max_f();

    /* traverse through kernel implementations */
    /* TODO: find a better way to do that... */
    desc.id = 0;
    for (int ndims_ker = ndims_ker_max; ndims_ker > 0; --ndims_ker) {
        desc.prb.ndims = ndims_ker;
        if (jit_uni_reorder_kernel_f32_t::applicable(desc.prb))
            return status::success;
    }

    return status::unimplemented;
}

kernel_t *kernel_t::create(const kernel_t::desc_t &desc) {
    switch (desc.id) {
        case 0: return new jit_uni_reorder_kernel_f32_t(desc);
        default: assert(!"unknown kernel id"); return nullptr;
    }

    return nullptr;
}
} // namespace tr

static void prb_block_for_cache(tr::prb_t &prb) {
    /* If strides for 0th and 1st nodes are cache friendly
     * then one can altogether do away with blocking ! */
    const bool cache_blocking_needed = false
            || (prb.nodes[0].is % 64 == 0 && prb.nodes[0].n > 16)
            || (prb.ndims > 1 && prb.nodes[1].is % 64 == 0
                    && prb.nodes[1].n > 16);
    if (!cache_blocking_needed) return;

    int unit_input_stride_idx = -1;
    for (auto idx = 0; idx < prb.ndims; ++idx) {
        if (prb.nodes[idx].is == 1) unit_input_stride_idx = idx;
    }

    /* Re-prioritize the sequential read over sequential write:
     *                             /-> [n0:is0:1][16n1:1:osk]...
     * [n0:is0:1]...[nk:1:osk] -->     or
     *                             \-> [16n1:1:osk][n0:is0:1]... */
    if (unit_input_stride_idx != -1) {
        const auto output_stride = prb.nodes[unit_input_stride_idx].os;
        const auto num_elems = prb.nodes[unit_input_stride_idx].n;

        const bool split_needed = (num_elems > 16) && (num_elems % 16 == 0);
        const int move_location = (output_stride % 4 != 0) ? 0 : 1;
        if (split_needed) prb_node_split(prb, unit_input_stride_idx, 16);

        /* Because of cache-unfriendly nature of unit-output stride node, let
         * us move unit-input stride node on or near front! */
        prb_node_move(prb, unit_input_stride_idx, move_location);
    }

    /* Potentially, split the node with os=1 in two and pull in the node with
     * is=1 between them for better cache reuse:
     * [n0:is0:1][n1:1:os1] --> [16n0:is0:1][n1:1:os1][n0/16:is0*16:16] */
    if (prb.ndims >= 2 && prb.nodes[0].os == 1 && prb.nodes[1].is == 1) {
        const auto input_stride = prb.nodes[0].is;
        const auto num_elems = prb.nodes[0].n;

        const bool split_needed = true && (num_elems > 16)
                && (num_elems % 16 == 0) && (input_stride >= 256)
                && (input_stride % 64 == 0);
        if (split_needed) {
            prb_node_split(prb, 0, 16);
            prb_node_move(prb, 1, 2);
        }
    }
}

/** finds the maximum number of dimension the kernel should process and
 * optionally splits one of the dimension to achieve better balance between
 * parallel driver and the kernel. */
static void prb_thread_kernel_balance(
        tr::prb_t &prb, int &ndims_ker_max, int nthr) {
    size_t sz_total = 1;
    for (int d = 0; d < prb.ndims; ++d)
        sz_total *= prb.nodes[d].n;

    /* sz_drv_min is the minimal size for the parallel
     * driver required for good parallelization */
    const size_t sz_drv_min
            = nstl::min<size_t>(16 * nthr, utils::div_up(sz_total, 1024));

    /* kdims -- # of dimensions processed by a kernel
     * sz_ker_cur -- product of the dimension processed by a kernel
     * sz_drv_cur -- product of the dimension processed by a driver */

    int kdims = prb.ndims;
    size_t sz_drv_cur = 1;
    for (; kdims > 1 && sz_drv_cur < sz_drv_min; --kdims)
        sz_drv_cur *= prb.nodes[kdims - 1].n;

    size_t sz_ker_cur = 1;
    for (int d = 0; d < kdims; ++d)
        sz_ker_cur *= prb.nodes[d].n;

    /* Initially kdims is chosen so that sz_drv_cur >= sz_drv_min.
     *
     * It might happen that for chosen kdims the sz_ker_cur is too small
     * (less than tr::ker_prb_size_min). In that case try to split the
     * innermost driver dimension into two, to increase sz_ker_cur. */
    bool want_borrow_ker_from_drv = true && kdims < prb.ndims
            && sz_ker_cur < tr::ker_prb_size_min && sz_drv_cur > sz_drv_min;
    if (want_borrow_ker_from_drv) {
        /* sz_want_borrow is the minimal sz, so that:
         *  o) sz_ker_cur * sz_want_borrow >= tr::ker_prb_size_min
         *  o) current innermost driver dimension is divisible by
         *     sz_want_borrow (so that we can evenly split that
         *     dimension into two)
         *
         *  In the worst case the minimal sz_want_borrow is equal
         *  to the innermost driver dimension itself. In that case
         *  we will sacrifice it in favor of kernel (is it fine?). */
        size_t sz_want_borrow = utils::div_up(tr::ker_prb_size_min, sz_ker_cur);
        for (; prb.nodes[kdims].n % sz_want_borrow; ++sz_want_borrow)
            ;
        if (sz_want_borrow != prb.nodes[kdims].n)
            prb_node_split(prb, kdims, sz_want_borrow);
        kdims += 1;
    }

    /* On the other hand it might happen that for chosen kdims
     * the sz_drv_cur is too small (less than sz_drv_min). In that case
     * try to split the outermost kernel dimension into two, to increase
     * sz_drv_cur. */
    bool want_borrow_drv_from_ker = true && sz_ker_cur > tr::ker_prb_size_min
            && sz_drv_cur < sz_drv_min;
    if (want_borrow_drv_from_ker) {
        size_t sz_want_borrow = utils::div_up(sz_drv_min, sz_drv_cur);
        for (; prb.nodes[kdims - 1].n % sz_want_borrow; ++sz_want_borrow)
            ;
        if (sz_want_borrow != prb.nodes[kdims - 1].n)
            prb_node_split(
                    prb, kdims - 1, prb.nodes[kdims - 1].n / sz_want_borrow);
    }

    ndims_ker_max = kdims;

    if (want_borrow_ker_from_drv || want_borrow_drv_from_ker) {
        DEBUG({
            printf("split: ");
            prb_dump(prb);
            printf("ndims_ker_max = %d\n", ndims_ker_max);
        });
    }
}

status_t jit_uni_reorder_t::pd_t::create(reorder_pd_t **reorder_pd,
        engine_t *engine, const primitive_attr_t *attr, engine_t *src_engine,
        const memory_desc_t *src_md, engine_t *dst_engine,
        const memory_desc_t *dst_md) {
    auto prb = tr::prb_t();

    status_t prb_init_status = prb_init(prb, *src_md, *dst_md, attr);
    if (prb_init_status != status::success) return prb_init_status;

    DEBUG({
        printf("init : ");
        prb_dump(prb);
    });
    // Sort the prb array in increasing sizes of the output stride
    prb_normalize(prb);
    DEBUG({
        printf("norm : ");
        prb_dump(prb);
    });
    /* Combine the variables, which appear together on both
             * sides of the reorder */
    prb_simplify(prb);
    DEBUG({
        printf("smpl : ");
        prb_dump(prb);
    });

    prb_block_for_cache(prb);
    DEBUG({
        printf("cache: ");
        prb_dump(prb);
    });

    int ndims_ker_max;
    int nthr = dnnl_get_max_threads();
    prb_thread_kernel_balance(prb, ndims_ker_max, nthr);

    tr::kernel_t::desc_t ker_desc;
    status_t ker_init_status
            = tr::kernel_t::desc_init(ker_desc, prb, ndims_ker_max);
    if (ker_init_status != status::success) return ker_init_status;

    const int ndims_driver = prb.ndims - ker_desc.prb.ndims;
    if (ndims_driver > jit_uni_reorder_t::ndims_driver_max)
        return status::unimplemented;

    DEBUG({
        printf("ker  : ");
        prb_dump(ker_desc.prb);
    });

    auto _pd = new pd_t(
            attr, src_engine->kind(), src_md, dst_engine->kind(), dst_md);
    if (_pd == nullptr) return status::out_of_memory;
    if (_pd->init(engine, src_engine, dst_engine) != status::success) {
        delete _pd;
        return status::unimplemented;
    }
    _pd->prb_ = prb;
    _pd->ker_desc_ = ker_desc;
    _pd->init_scratchpad_md();
    _pd->nthr_ = nthr;
    return safe_ptr_assign(*reorder_pd, _pd);
}

void jit_uni_reorder_t::omp_driver_0d(
        int off, const char *in, char *out, const float *scale) const {
    tr::call_param_t c {in, out, scale};
    (*kernel_)(&c);
}

void jit_uni_reorder_t::omp_driver_1d(int ithr, int nthr, int off,
        const char *in, char *out, const float *scale) const {
    const tr::node_t *ns = pd()->prb_.nodes + off;
    for_nd(ithr, nthr, (ptrdiff_t)ns[0].n, [&](ptrdiff_t d0) {
        auto c = tr::call_param_t();
        c.in = in + d0 * ns[0].is * data_type_size(pd()->prb_.itype);
        c.out = out + d0 * ns[0].os * data_type_size(pd()->prb_.otype);
        c.scale = scale + d0 * ns[0].ss;
        (*kernel_)(&c);
    });
}

void jit_uni_reorder_t::omp_driver_2d(int ithr, int nthr, int off,
        const char *in, char *out, const float *scale) const {
    const tr::node_t *ns = pd()->prb_.nodes + off;
    for_nd(ithr, nthr, (ptrdiff_t)ns[1].n, (ptrdiff_t)ns[0].n,
            [&](ptrdiff_t d1, ptrdiff_t d0) {
                auto c = tr::call_param_t();
                c.in = in
                        + (d0 * ns[0].is + d1 * ns[1].is)
                                * data_type_size(pd()->prb_.itype);
                c.out = out
                        + (d0 * ns[0].os + d1 * ns[1].os)
                                * data_type_size(pd()->prb_.otype);
                c.scale = scale + d0 * ns[0].ss + d1 * ns[1].ss;
                (*kernel_)(&c);
            });
}

void jit_uni_reorder_t::omp_driver_3d(int ithr, int nthr, int off,
        const char *in, char *out, const float *scale) const {
    const tr::node_t *ns = pd()->prb_.nodes + off;
    for_nd(ithr, nthr, (ptrdiff_t)ns[2].n, (ptrdiff_t)ns[1].n,
            (ptrdiff_t)ns[0].n, [&](ptrdiff_t d2, ptrdiff_t d1, ptrdiff_t d0) {
                auto c = tr::call_param_t();
                c.in = in
                        + (d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is)
                                * data_type_size(pd()->prb_.itype);
                c.out = out
                        + (d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os)
                                * data_type_size(pd()->prb_.otype);
                c.scale = scale + d0 * ns[0].ss + d1 * ns[1].ss + d2 * ns[2].ss;
                (*kernel_)(&c);
            });
}

void jit_uni_reorder_t::omp_driver_4d(int ithr, int nthr, int off,
        const char *in, char *out, const float *scale) const {
    const tr::node_t *ns = pd()->prb_.nodes + off;
    for_nd(ithr, nthr, (ptrdiff_t)ns[3].n, (ptrdiff_t)ns[2].n,
            (ptrdiff_t)ns[1].n, (ptrdiff_t)ns[0].n,
            [&](ptrdiff_t d3, ptrdiff_t d2, ptrdiff_t d1, ptrdiff_t d0) {
                auto c = tr::call_param_t();
                c.in = in
                        + (d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is
                                  + d3 * ns[3].is)
                                * data_type_size(pd()->prb_.itype);
                c.out = out
                        + (d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os
                                  + d3 * ns[3].os)
                                * data_type_size(pd()->prb_.otype);
                c.scale = scale + d0 * ns[0].ss + d1 * ns[1].ss + d2 * ns[2].ss
                        + d3 * ns[3].ss;
                (*kernel_)(&c);
            });
}

void jit_uni_reorder_t::omp_driver(
        const char *in, char *out, const float *scale) const {
    in += pd()->prb_.ioff * data_type_size(pd()->prb_.itype);
    out += pd()->prb_.ooff * data_type_size(pd()->prb_.otype);

    DEBUG({
        printf("prb : ");
        tr::prb_dump(pd()->prb_);
    });
    DEBUG({
        printf("ker : ");
        tr::prb_dump(pd()->ker_desc_.prb);
    });

    int ndims = pd()->prb_.ndims;
    int ndims_ker = pd()->ker_desc_.prb.ndims;
    assert(ndims - ndims_ker <= ndims_driver_max);

    if (ndims - ndims_ker == 0) {
        omp_driver_0d(ndims_ker, in, out, scale);
    } else {
        parallel(pd()->nthr_, [&](const int ithr, const int nthr) {
            switch (ndims - ndims_ker) {
                case 1:
                    omp_driver_1d(ithr, nthr, ndims_ker, in, out, scale);
                    break;
                case 2:
                    omp_driver_2d(ithr, nthr, ndims_ker, in, out, scale);
                    break;
                case 3:
                    omp_driver_3d(ithr, nthr, ndims_ker, in, out, scale);
                    break;
                case 4:
                    omp_driver_4d(ithr, nthr, ndims_ker, in, out, scale);
                    break;
                default: assert(!"unimplemented");
            }
        });
    }
}

status_t jit_uni_reorder_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, tr::kernel_t::create(pd()->ker_desc_)));
    return kernel_->create_kernel();
}

status_t jit_uni_reorder_t::execute(const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto in = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto out = CTX_OUT_CLEAN_MEM(char *, DNNL_ARG_TO, status);
    CHECK(status);
    DEFINE_SCALES_BUFFER(scales);

    omp_driver(in, out, scales);

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
