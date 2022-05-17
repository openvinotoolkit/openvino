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
#include <numeric>

#include "oneapi/dnnl/dnnl_debug.h"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"
#include "cpu/x64/jit_uni_reorder.hpp"

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

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

#ifdef _WIN32
/* seems like s_addr is a reserved macro on Windows */
#undef s_addr
constexpr static bool is_windows = true;
#else
constexpr static bool is_windows = false;
#endif

using namespace Xbyak;
using namespace dnnl::impl::types;

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace tr {

static bool prb_has_small_strides(const prb_t &prb) {
    constexpr ptrdiff_t max_stride = (1LL << 31) - 1;
    for (int d = 0; d < prb.ndims; ++d) {
        const ptrdiff_t cms = max_stride / prb.nodes[d].n;
        const bool small_strides = true
                && prb.nodes[d].is < cms / (int)data_type_size(prb.itype)
                && prb.nodes[d].os < cms / (int)data_type_size(prb.otype);
        if (!small_strides) return false;
    }
    return true;
}

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
        int ndims_full_unroll = 0;
        int len_last_dim_unroll = 0;
        int tail_len_unroll = 0;
        int len_unroll = 0;
    };

    static bool simple_impl_desc_init(
            const prb_t &prb, simple_impl_desc_t *desc) {
        const int ndims = prb.ndims;

        int ndims_full_unroll = 0;
        int len_last_dim_unroll = 1;
        int tail_len_unroll = 0;
        int len_unroll = 1;

        // It is responsible for finding as many values
        // as kernel can unroll. If tail is present then
        // kernel will unroll only last node (possible improvement).
        // If there is no tail kernel can unroll a few nodes without any loops etc.
        // ndims_full_unroll - how many nodes will be unrolled
        // len_last_dim_unroll - what piece of last unrolled node will be unrolled
        if (prb.is_tail_present) {
            ndims_full_unroll = 1;
            len_unroll = prb.nodes[0].n;
            tail_len_unroll = prb.nodes[0].is_zero_pad_needed
                    ? 0
                    : static_cast<int>(prb.nodes[0].tail_size);
        } else {
            for (int d = 0; d < ndims; ++d) {
                const auto &node = prb.nodes[d];
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
        }

        if (prb.ndims - ndims_full_unroll > ndims_jit_loop_max) return false;

        if (desc) {
            desc->ndims_full_unroll = ndims_full_unroll;
            desc->len_last_dim_unroll = len_last_dim_unroll;
            desc->tail_len_unroll = tail_len_unroll;
            desc->len_unroll = len_unroll;
        }

        return true;
    }

    static bool applicable(const prb_t &p) {
        using namespace data_type;

        bool ok = true && p.ndims > 0
                && utils::one_of(p.itype, f32, bf16, s32, s8, u8)
                && utils::one_of(p.otype, f32, bf16, s32, s8, u8)
                && IMPLICATION(p.itype == bf16,
                        utils::one_of(p.otype, s8, u8, f32, bf16))
                && IMPLICATION(p.otype == bf16,
                        utils::one_of(p.itype, s8, u8, f32, bf16))
                && utils::everyone_is(0, p.ioff, p.ooff) /* do we need this? */
                && utils::one_of(p.beta, 0.f, 1.f) /* anything else? */
                && simple_impl_desc_init(p, nullptr) && mayiuse(sse41)
                && IMPLICATION((p.itype == bf16 || p.otype == bf16),
                        mayiuse(avx512_core))
                && prb_has_small_strides(p);

        return ok;
    }

    Address i_addr(int i_off) {
        return ptr[reg_ptr_in_ + reg_off_in_ + i_off * itype_sz_];
    }

    Address o_addr(int o_off, bool with_type_multiplier = true) {
        if (with_type_multiplier)
            return ptr[reg_ptr_out_ + reg_off_out_ + o_off * otype_sz_];
        else
            return ptr[reg_ptr_out_ + reg_off_out_ + o_off];
    }

    Address s_addr(int s_off) {
        return ptr[reg_ptr_scale_ + reg_off_scale_ + s_off * stype_sz_];
    }

    Address c_addr(int c_off) {
        return ptr[reg_ptr_comp_ + reg_off_comp_ + c_off * sizeof(int32_t)];
    }

    void step(int off, int prev_i_off, int prev_o_off, int prev_s_off,
            int prev_c_off, int &i_off, int &o_off, int &s_off, int &c_off,
            int step_size = 1) {
        i_off = prev_i_off;
        o_off = prev_o_off;
        s_off = prev_s_off;
        c_off = prev_c_off;

        if (off == 0) return;

        int start_dim = 0, dims_prod = 1;
        for (; start_dim < prb_.ndims && dims_prod != step_size; ++start_dim)
            dims_prod *= prb_.n(start_dim);
        assert(start_dim < prb_.ndims);
        off /= step_size;

        for (int dim_id = start_dim; dim_id < prb_.ndims; ++dim_id) {
            i_off += prb_.is(dim_id);
            o_off += prb_.os(dim_id);
            s_off += prb_.ss(dim_id);
            c_off += prb_.cs(dim_id);

            if (off % prb_.n(dim_id)) break;

            i_off += -prb_.n(dim_id) * prb_.is(dim_id);
            o_off += -prb_.n(dim_id) * prb_.os(dim_id);
            s_off += -prb_.n(dim_id) * prb_.ss(dim_id);
            c_off += -prb_.n(dim_id) * prb_.cs(dim_id);

            off /= prb_.n(dim_id);

            if (off == 0) break; /* FIXME: is it really required? */
        }
    }

    void step(int off, int prev_i_off, int prev_o_off, int &i_off, int &o_off,
            int step_size = 1) {
        int dummy = 0;
        step(off, prev_i_off, prev_o_off, dummy, dummy, i_off, o_off, dummy,
                dummy, step_size);
    }

    void tr8x8_avx2(int i_off, int o_off) {
        using namespace data_type;

        const auto cvt2ps
                = [=](const Ymm &dst, const Operand &src, data_type_t idt) {
                      switch (idt) {
                          case f32:
                              if (src.isMEM() || src.getIdx() != dst.getIdx())
                                  vmovups(dst, src);
                              break;
                          case bf16:
                              vpmovzxwd(dst, src);
                              vpslld(dst, dst, 0x10);
                              break;
                          case s32: vcvtdq2ps(dst, src); break;
                          case s8:
                              vpmovsxbd(dst, src);
                              vcvtdq2ps(dst, dst);
                              break;
                          case u8:
                              vpmovzxbd(dst, src);
                              vcvtdq2ps(dst, dst);
                              break;
                          default: assert(!"unreachable");
                      }
                  };

        const auto cvt2odt = [=](const Ymm &ymm, data_type_t odt,
                                     data_type_t idt) {
            const Xmm xmm = Xmm(ymm.getIdx());
            switch (odt) {
                case bf16:
                    if (utils::one_of(idt, f32, s8, u8)) {
                        if (idt != f32) cvt2ps(ymm, ymm, idt);
                        if (mayiuse(avx512_core_bf16)) {
                            vcvtneps2bf16(Xmm(ymm.getIdx()), ymm);
                        } else {
                            bf16_emu_->vcvtneps2bf16(
                                    Ymm(ymm.getIdx()), Zmm(ymm.getIdx()));
                        }
                    }
                    break;
                case s32:
                    if (idt == f32)
                        vcvtps2dq(ymm, ymm);
                    else if (idt == s8)
                        vpmovsxbd(ymm, ymm);
                    else if (idt == u8)
                        vpmovzxbd(ymm, ymm);
                    break;
                case s8:
                    if (idt == bf16) cvt2ps(ymm, ymm, idt);
                    if (utils::one_of(idt, f32, bf16)) vcvtps2dq(ymm, ymm);
                    if (utils::one_of(idt, bf16, f32, s32)) {
                        if (mayiuse(avx512_core)) {
                            vpmovsdb(xmm, ymm);
                        } else {
                            vpackssdw(ymm, ymm, ymm_zero_);
                            vpermq(ymm, ymm, 0x58);
                            vpacksswb(ymm, ymm, ymm_zero_);
                        }
                    }
                    if (idt == u8) vpminub(ymm, ymm, ymm_8x127b_);
                    break;
                case u8:
                    if (idt == bf16) cvt2ps(ymm, ymm, idt);
                    if (utils::one_of(idt, f32, bf16)) vcvtps2dq(ymm, ymm);
                    if (utils::one_of(idt, bf16, f32, s32)) {
                        if (mayiuse(avx512_core)) {
                            vpmaxsd(ymm, ymm, ymm_zero_);
                            vpmovusdb(xmm, ymm);
                        } else {
                            vpackssdw(ymm, ymm, ymm_zero_);
                            vpermq(ymm, ymm, 0x58);
                            vpackuswb(ymm, ymm, ymm_zero_);
                        }
                    }
                    if (idt == s8) vpmaxsb(ymm, ymm, ymm_zero_);
                    break;
                default: assert(!"unreachable");
            }
        };

        auto load = [=](const Ymm &ymm, const Address &addr, int size) {
            const Xmm xmm = Xmm(ymm.getIdx());
            switch (size) {
                case 32: vmovups(ymm, addr); break;
                case 16: vmovups(xmm, addr); break;
                case 8: vmovsd(xmm, addr); break;
                default: assert(!"unreachable");
            }
        };

        auto store = [=](const Address &addr, const Ymm &ymm, int size) {
            const Xmm xmm = Xmm(ymm.getIdx());
            switch (size) {
                case 32: vmovups(addr, ymm); break;
                case 16: vmovups(addr, xmm); break;
                case 8: vmovsd(addr, xmm); break;
                default: assert(!"unreachable");
            }
        };

        const int unroll = 8;

        const bool interim_f32 = (prb_.itype != f32)
                || utils::one_of(f32, prb_.itype, prb_.otype);

        const bool need_saturation
                = (utils::one_of(prb_.otype, u8, s8, s32) && interim_f32);

        for (int i = 0; i < unroll; i++) {
            const int node_0_input_stride = prb_.is(0);
            load(Ymm(i), i_addr(i_off + i * node_0_input_stride),
                    unroll * itype_sz_);

            if (interim_f32) cvt2ps(Ymm(i), Ymm(i), prb_.itype);
        }

        for (int i = 0; i < unroll / 2; i++) {
            vunpcklps(Ymm(unroll + i), Ymm(2 * i), Ymm(2 * i + 1));
            vunpckhps(Ymm(i), Ymm(2 * i), Ymm(2 * i + 1));
        }

        const unsigned int lfloat = 0x44;
        const unsigned int ufloat = 0xee;
        for (int i = 0; i < unroll / 2; i++) {
            const int j = i % 2 == 0 ? unroll + i : i - 1;
            vshufps(Ymm(unroll / 2 + 2 * i), Ymm(j), Ymm(j + 1), lfloat);
            vshufps(Ymm(unroll / 2 + 2 * i + 1), Ymm(j), Ymm(j + 1), ufloat);
        }

        const unsigned int lquad = 0x20;
        for (int i = 0; i < unroll / 2; i++)
            vperm2f128(Ymm(i), Ymm(unroll / 2 + i), Ymm(unroll + i), lquad);

        const unsigned int uquad = 0x31;
        for (int i = unroll / 2; i < unroll; i++)
            vperm2f128(Ymm(i), Ymm(i), Ymm(unroll / 2 + i), uquad);

        if (need_saturation) {
            init_saturate_f32(ymm_zero_, ymm_saturation_ubound_, reg_tmp_,
                    interim_f32 ? f32 : prb_.itype, prb_.otype);
            for (int i = 0; i < unroll; i++)
                saturate_f32(
                        Ymm(i), ymm_zero_, ymm_saturation_ubound_, prb_.otype);
        }

        for (int i = 0; i < unroll; i++) {
            const int node_1_output_stride = prb_.os(1);
            if (prb_.otype != f32)
                cvt2odt(Ymm(i), prb_.otype, interim_f32 ? f32 : prb_.itype);
            store(o_addr(o_off + i * node_1_output_stride), Ymm(i),
                    unroll * otype_sz_);
        }
    }

    bool can_do_tr8x8() {
        using namespace data_type;

        static constexpr int desirable_node_size = 8;
        static constexpr int desirable_stride = 1;

        // This processing is relied on swaping two innermost dimension.
        // Therefore, input stride in second node and output stride in first node
        // have to be equal to 1.

        return mayiuse(avx2) && prb_.ndims >= 2
                && ((utils::one_of(prb_.itype, u8, s8, s32, f32, bf16)
                        && utils::one_of(prb_.otype, u8, s8, s32, f32, bf16)))
                && utils::everyone_is(desirable_node_size, prb_.n(0), prb_.n(1))
                && utils::everyone_is(desirable_stride, prb_.os(0), prb_.is(1))
                && !prb_.is_tail_present
                && prb_.scale_type == scale_type_t::NONE && prb_.beta == 0.f;
    }

    bool process_unroll_tr8x8(const int ndims, const int len) {
        if (!can_do_tr8x8()) return false;

        const int step_size = prb_.n(0) * prb_.n(1);
        int i_off = 0, o_off = 0;
        for (int off = 0; off < len; off += step_size) {
            step(off, i_off, o_off, i_off, o_off, step_size);
            tr8x8_avx2(i_off, o_off);
        }

        return true;
    }

    template <cpu_isa_t isa>
    bool process_direct_copy(const int ndims, const int len) {
        using namespace data_type;

        static constexpr int desirable_stride = 1;
        using Vmm = typename cpu_isa_traits<isa>::Vmm;
        const int simd_w = cpu_isa_traits<isa>::vlen / itype_sz_;

        // TODO: support tail_processing for direct copy

        const bool do_src_zp = prb_.req_src_zp;
        const bool do_dst_zp = prb_.req_dst_zp;
        const bool zp_applicable = IMPLICATION(
                (do_src_zp || do_dst_zp), utils::one_of(prb_.itype, s32, f32));
        const bool can_do = true && mayiuse(isa)
                && compensation_needed_ == false
                && utils::everyone_is(desirable_stride, prb_.os(0), prb_.is(0))
                && (false || (prb_.itype == prb_.otype ? zp_applicable : false)
                        || (prb_.itype == s32 && prb_.otype == f32)
                        || (prb_.itype == f32 && prb_.otype == s32))
                && len % simd_w == 0 && prb_.n(0) % len == 0
                && !prb_.is_tail_present
                && prb_.scale_type == scale_type_t::NONE && prb_.beta == 0.f;
        if (!can_do) return false;

#define PARAM(x) ptr[abi_param1 + offsetof(call_param_t, x)]
        static constexpr int vmm_zp_last_idx = 15;
        const auto vmm_src_zp
                = Vmm(do_dst_zp ? vmm_zp_last_idx - 1 : vmm_zp_last_idx);
        if (do_src_zp) {
            uni_vpbroadcastd(vmm_src_zp, PARAM(src_zp));
            uni_vcvtdq2ps(vmm_src_zp, vmm_src_zp);
        }
        const auto vmm_dst_zp = Vmm(vmm_zp_last_idx);
        if (do_dst_zp) {
            uni_vpbroadcastd(vmm_dst_zp, PARAM(dst_zp));
            uni_vcvtdq2ps(vmm_dst_zp, vmm_dst_zp);
        }
#undef PARAM

        const auto apply_zp_ps = [&](const Vmm vmm) {
            if (do_src_zp) uni_vsubps(vmm, vmm, vmm_src_zp);
            if (do_dst_zp) uni_vaddps(vmm, vmm, vmm_dst_zp);
        };

        for (int off = 0; off < len;) {
            // TODO: we need extra reg for proper saturation if otype == s32
            int unroll
                    = nstl::min(16 - (prb_.otype == s32), (len - off) / simd_w);
            unroll = (do_src_zp || do_dst_zp)
                    ? nstl::min(unroll, 16 - do_src_zp - do_dst_zp)
                    : unroll;

            for (int ur = 0; ur < unroll; ++ur) {
                const auto vmm = Vmm(ur);
                uni_vmovups(vmm, i_addr(off + ur * simd_w));
            }

            if (prb_.itype != prb_.otype) {
                for (int ur = 0; ur < unroll; ++ur) {
                    const auto vmm = Vmm(ur);
                    if (prb_.itype == s32 && prb_.otype == f32) {
                        uni_vcvtdq2ps(vmm, vmm);
                        apply_zp_ps(vmm);
                    } else if (prb_.itype == f32 && prb_.otype == s32) {
                        apply_zp_ps(vmm);
                        uni_vcvtps2dq(vmm, vmm);
                    } else
                        assert(!"unreachable");
                }
            } else if (do_src_zp || do_dst_zp) {
                for (int ur = 0; ur < unroll; ++ur) {
                    const auto vmm = Vmm(ur);
                    if (prb_.otype == f32) {
                        apply_zp_ps(vmm);
                    } else if (prb_.otype == s32) {
                        uni_vcvtdq2ps(vmm, vmm);
                        apply_zp_ps(vmm);
                        uni_vcvtps2dq(vmm, vmm);
                    }
                }
            }

            for (int ur = 0; ur < unroll; ++ur) {
                const auto vmm = Vmm(ur);
                uni_vmovups(o_addr(off + ur * simd_w), vmm);
            }

            off += unroll * simd_w;
        }

        return true;
    }

    void process_unroll_generic_step(int reg_unroll, const int *i_off,
            const int *o_off, const int *s_off, const int *c_off,
            const int *zero_padding, const bool tail_processing) {
        using namespace data_type;

        const auto cvt2ps
                = [=](const Xmm &dst, const Operand &src, data_type_t idt) {
                      Xmm dst_pure = Xmm(dst.getIdx());
                      switch (idt) {
                          case f32:
                              if (src.isMEM() || src.getIdx() != dst.getIdx())
                                  uni_vmovups(dst, src);
                              break;
                          case bf16:
                              if (mayiuse(avx)) {
                                  vpmovzxwd(dst, src);
                                  vpslld(dst, dst, 0x10);
                                  break;
                              } else
                                  assert("unreachable!");
                          case s32: uni_vcvtdq2ps(dst, src); break;
                          case s8:
                              uni_vpmovsxbd(dst, src);
                              uni_vcvtdq2ps(dst_pure, dst);
                              break;
                          case u8:
                              uni_vpmovzxbd(dst, src);
                              uni_vcvtdq2ps(dst_pure, dst);
                              break;
                          default: assert(!"unreachable");
                      }
                  };

        const auto cvt2odt = [=](const Xmm &xmm, data_type_t odt,
                                     data_type_t idt) {
            switch (odt) {
                case bf16:
                    if (!mayiuse(avx)) assert(!"unreachable");
                    if (utils::one_of(idt, f32, s8, u8)) {
                        if (idt != f32) cvt2ps(xmm, xmm, idt);
                        if (mayiuse(avx512_core_bf16)) {
                            vcvtneps2bf16(xmm, xmm);
                        } else {
                            bf16_emu_->vcvtneps2bf16(
                                    Ymm(xmm.getIdx()), Zmm(xmm.getIdx()));
                        }
                    }
                    break;
                case s32:
                    if (idt == f32)
                        uni_vcvtps2dq(xmm, xmm);
                    else if (idt == s8)
                        uni_vpmovsxbd(xmm, xmm);
                    else if (idt == u8)
                        uni_vpmovzxbd(xmm, xmm);
                    break;
                case s8:
                    if (idt == bf16) cvt2ps(xmm, xmm, idt);
                    if (utils::one_of(idt, f32, bf16)) uni_vcvtps2dq(xmm, xmm);
                    if (utils::one_of(idt, bf16, f32, s32)) {
                        if (mayiuse(avx512_core)) {
                            vpmovsdb(xmm, xmm);
                        } else {
                            uni_vpackssdw(xmm, xmm, xmm_zero_);
                            uni_vpacksswb(xmm, xmm, xmm_zero_);
                        }
                    }
                    if (idt == u8) uni_vpminub(xmm, xmm, xmm_4x127b_);
                    break;
                case u8:
                    if (idt == bf16) cvt2ps(xmm, xmm, idt);
                    if (utils::one_of(idt, f32, bf16)) uni_vcvtps2dq(xmm, xmm);
                    if (utils::one_of(idt, bf16, f32, s32)) {
                        if (mayiuse(avx512_core)) {
                            vpmaxsd(xmm, xmm, xmm_zero_);
                            vpmovusdb(xmm, xmm);
                        } else {
                            uni_vpackssdw(xmm, xmm, xmm_zero_);
                            uni_vpackuswb(xmm, xmm, xmm_zero_);
                        }
                    }
                    if (idt == s8) uni_vpmaxsb(xmm, xmm, xmm_zero_);
                    break;
                default: assert(!"unreachable");
            }
        };

        auto load = [=](const Xmm &xmm, const Address &addr, int size) {
            switch (size) {
                case 16: uni_vmovups(xmm, addr); break;
                case 8: uni_vmovsd(xmm, addr); break;
                case 4: uni_vmovss(xmm, addr); break;
                case 2: uni_vpinsrw(xmm, xmm, addr, 0x0); break;
                case 1: uni_vpinsrb(xmm, xmm, addr, 0x0); break;
                default: assert(!"unreachable");
            }
        };

        auto load_bytes
                = [=](const Xmm &xmm, const Address &addr, int size, int imm) {
                      switch (size) {
                          case 4: uni_vpinsrd(xmm, xmm, addr, imm); break;
                          case 2: uni_vpinsrw(xmm, xmm, addr, imm); break;
                          case 1: uni_vpinsrb(xmm, xmm, addr, imm); break;
                          default: assert(!"unreachable");
                      }
                  };

        auto store = [=](const Address &addr, const Xmm &xmm, int size) {
            switch (size) {
                case 16: uni_vmovups(addr, xmm); break;
                case 8: uni_vmovsd(addr, xmm); break;
                case 4: uni_vmovss(addr, xmm); break;
                case 2: uni_vpextrw(addr, xmm, 0x0); break;
                case 1: uni_vpextrb(addr, xmm, 0x0); break;
                default: assert(!"unreachable");
            }
        };

        /* check whether loading 4 values at once is possible */
        static constexpr int xmm_vlen = 4;
        bool can_load_xmm = reg_unroll % xmm_vlen == 0;
        for (int ur = 1; ur < reg_unroll; ++ur)
            if (i_off[ur] != i_off[ur - 1] + 1) {
                can_load_xmm = false;
                break;
            }
        const int load_step = can_load_xmm ? xmm_vlen : 1;

        /* check whether storing 4 values at once is possible */
        bool can_store_xmm = reg_unroll % xmm_vlen == 0;
        for (int ur = 1; ur < reg_unroll; ++ur)
            if (o_off[ur] != o_off[ur - 1] + 1) {
                can_store_xmm = false;
                break;
            }
        const int ur_step = can_store_xmm ? 4 : 1;
        const int load_tail_step
                = !can_load_xmm && can_store_xmm ? ur_step : load_step;

        const bool interim_f32 = interim_f32_needed();

        const bool need_saturation
                = (utils::one_of(prb_.otype, u8, s8, s32) && interim_f32);

        std::vector<int> store_masks;
        if (tail_processing) {
            for (int ur = 0; ur < reg_unroll; ur += load_tail_step) {
                uni_vpxor(Xmm(ur), Xmm(ur), Xmm(ur));
                store_masks.push_back(0);
                for (int r = 0; r < load_tail_step; ++r) {
                    if (zero_padding[ur + r] == 0) {
                        store_masks.back() += 1 << r;
                        load_bytes(
                                Xmm(ur), i_addr(i_off[ur + r]), itype_sz_, r);
                    }
                }
            }
        } else {
            if (!can_load_xmm && can_store_xmm) {
                assert(ur_step == xmm_vlen);
                /* load with stride */
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    for (int r = 0; r < ur_step; ++r) {
                        load_bytes(
                                Xmm(ur), i_addr(i_off[ur + r]), itype_sz_, r);
                    }
                }
            } else {
                for (int ur = 0; ur < reg_unroll; ur += load_step) {
                    load(Xmm(ur), i_addr(i_off[ur]), load_step * itype_sz_);
                }
            }
        }

        /* xmm[:] <-- (f32)xmm[:] */
        if (interim_f32) {
            const int cvt_step = nstl::max(load_step, ur_step);
            for (int ur = 0; ur < reg_unroll; ur += cvt_step)
                cvt2ps(Xmm(ur), Xmm(ur), prb_.itype);
        }

        if (can_load_xmm && !can_store_xmm) {
            const bool fast_return = true // transposition on the fly
                    && prb_.scale_type != scale_type_t::MANY
                    && prb_.beta == 0.f;
            if (fast_return) {
                if (prb_.scale_type == scale_type_t::COMMON)
                    for (int ur = 0; ur < reg_unroll; ur += load_step)
                        uni_vmulps(Xmm(ur), Xmm(ur), xmm_scale_);
                if (prb_.otype != f32) {
                    init_saturate_f32(xmm_zero_, xmm_saturation_ubound_,
                            reg_tmp_, interim_f32 ? f32 : prb_.itype,
                            prb_.otype);
                    for (int ur = 0; ur < reg_unroll; ur += load_step) {
                        if (need_saturation)
                            saturate_f32(Xmm(ur), xmm_zero_,
                                    xmm_saturation_ubound_, prb_.otype);
                        cvt2odt(Xmm(ur), prb_.otype,
                                interim_f32 ? f32 : prb_.itype);
                    }
                }
                for (int ur = 0; ur < reg_unroll; ur += load_step) {
                    for (int r = 0; r < load_step; ++r) {
                        if (otype_sz_ == 4)
                            uni_vpextrd(o_addr(o_off[ur + r]), Xmm(ur), r);
                        else if (otype_sz_ == 2)
                            uni_vpextrw(o_addr(o_off[ur + r]), Xmm(ur), r);
                        else
                            uni_vpextrb(o_addr(o_off[ur + r]), Xmm(ur), r);
                    }
                }
                return;
            }

            /* scatter elements of xmm into 4 xmms */
            if (itype_sz_ == 4 || interim_f32) {
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    for (int r = 1; r < load_step; ++r) {
                        uni_vshufps(Xmm(ur + r), Xmm(ur), Xmm(ur), r);
                    }
            } else {
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    for (int r = 1; r < load_step; ++r) {
                        if (mayiuse(avx))
                            vpalignr(Xmm(ur + r), Xmm(ur), Xmm(ur),
                                    itype_sz_ * r);
                        else {
                            movups(Xmm(ur + r), Xmm(ur));
                            palignr(Xmm(ur + r), Xmm(ur), itype_sz_ * r);
                        }
                    }
            }
        }

        /* src zero point application */
#define PARAM(x) ptr[abi_param1 + offsetof(call_param_t, x)]
        if (prb_.req_src_zp) {
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                const auto xmm = Xmm(ur);
                if (interim_f32)
                    uni_vsubps(xmm, xmm, xmm_src_zp_);
                else
                    uni_vpsubd(xmm, xmm, xmm_src_zp_);
            }
        }

        /* scale and beta processing */
        if (can_store_xmm) {
            /* xmm <-- scale * xmm[:] */
            if (prb_.scale_type == scale_type_t::COMMON) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step)
                    uni_vmulps(Xmm(ur), Xmm(ur), xmm_scale_);
            } else if (prb_.scale_type == scale_type_t::MANY) {
                enum class scale_load_type_t { bcast, load, gather };

                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    scale_load_type_t scale_load_type
                            = scale_load_type_t::bcast; // the best case

                    for (int r = ur + 1; r < ur + ur_step; ++r)
                        if (s_off[r] != s_off[r - 1] + 0)
                            scale_load_type = scale_load_type_t::load;

                    if (scale_load_type == scale_load_type_t::bcast
                            && !tail_processing) {
                        uni_vbroadcastss(xmm_scale_, s_addr(s_off[ur]));
                        uni_vmulps(Xmm(ur), Xmm(ur), xmm_scale_);
                        continue;
                    }

                    // bcast doesn't work, the next try -- load
                    for (int r = ur + 1; r < ur + ur_step; ++r)
                        if (s_off[r] != s_off[r - 1] + 1)
                            scale_load_type = scale_load_type_t::gather;

                    if (scale_load_type == scale_load_type_t::load
                            && !tail_processing) {
                        uni_vmovups(xmm_scale_, s_addr(s_off[ur]));
                        uni_vmulps(Xmm(ur), Xmm(ur), xmm_scale_);
                        continue;
                    }

                    // load doesn't work as well
                    // so gather the scale factors one by one
                    for (int r = ur; r < ur + ur_step; ++r) {
                        if (zero_padding[r] == 0 || !tail_processing)
                            uni_vpinsrd(xmm_scale_, xmm_scale_,
                                    s_addr(s_off[r]), r - ur);
                    }
                    uni_vmulps(Xmm(ur), Xmm(ur), xmm_scale_);
                }
            }

            /* dst <-- beta * dst + xmm[:] */
            assert(prb_.beta == 0.f || prb_.beta == 1.f);
            if (prb_.beta == 1.f) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    if (prb_.otype == f32) {
                        /* non VEX instructions do not support unaligned
                         * memory for instructions other than movups. */
                        if (mayiuse(avx)) {
                            vaddps(Xmm(ur), o_addr(o_off[ur]));
                        } else {
                            /* register xmm(1) is unused */
                            movups(Xmm(1), o_addr(o_off[ur]));
                            addps(Xmm(ur), Xmm(1));
                        }
                    } else {
                        cvt2ps(Xmm(1), o_addr(o_off[ur]), prb_.otype);
                        uni_vaddps(Xmm(ur), Xmm(ur), Xmm(1));
                    }
                }
            }
        } else {
            /* xmm[0] <-- scale * xmm[0] */
            if (prb_.scale_type == scale_type_t::COMMON) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step)
                    uni_vmulss(Xmm(ur), Xmm(ur), xmm_scale_);
            } else if (prb_.scale_type == scale_type_t::MANY) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    if (zero_padding[ur] == 0 || !tail_processing)
                        uni_vmulss(Xmm(ur), Xmm(ur), s_addr(s_off[ur]));
                }
            }

            /* dst <-- beta * dst + xmm[0] */
            assert(prb_.beta == 0.f || prb_.beta == 1.f);
            if (prb_.beta == 1.f) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    if (prb_.otype == f32) {
                        uni_vaddss(Xmm(ur), Xmm(ur), o_addr(o_off[ur]));
                    } else {
                        if (prb_.otype == s32) {
                            uni_vmovss(xmm_tmp_, o_addr(o_off[ur]));
                        } else if (utils::one_of(prb_.otype, s8, u8)) {
                            uni_vpinsrb(
                                    xmm_tmp_, xmm_tmp_, o_addr(o_off[ur]), 0x0);
                        } else if (prb_.otype == bf16) {
                            uni_vpinsrw(
                                    xmm_tmp_, xmm_tmp_, o_addr(o_off[ur]), 0x0);
                        } else {
                            assert(!"unsupported o_type");
                        }
                        cvt2ps(xmm_tmp_, xmm_tmp_, prb_.otype);
                        uni_vaddps(Xmm(ur), Xmm(ur), xmm_tmp_);
                    }
                }
            }
        }

        /* dst zero point application */
        if (prb_.req_dst_zp) {
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                const auto xmm = Xmm(ur);
                if (interim_f32)
                    uni_vaddps(xmm, xmm, xmm_dst_zp_);
                else
                    uni_vpaddd(xmm, xmm, xmm_dst_zp_);
            }
        }
#undef PARAM

        /* adjust scale application */
        if (prb_.scale_adjust != 1.f) {
            uni_vmovd(xmm_tmp_, reg_scale_adjust_);
            uni_vpshufd(xmm_tmp_, xmm_tmp_, 0x0);
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                uni_vmulps(Xmm(ur), Xmm(ur), xmm_tmp_);
            }
        }

        if (need_saturation) {
            init_saturate_f32(xmm_zero_, xmm_saturation_ubound_, reg_tmp_, f32,
                    prb_.otype, compensation_needed_);
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                saturate_f32(Xmm(ur), xmm_zero_, xmm_saturation_ubound_,
                        prb_.otype, compensation_needed_);
            }

            // reset back xmm_zero_ if needed.
            if (compensation_needed_ && (prb_.req_src_zp || prb_.req_dst_zp))
                uni_vxorps(xmm_zero_, xmm_zero_, xmm_zero_);
        }

        if (compensation_needed_) {
            const int xmm_begin = 9;
            const int xmm_end = 11;
            int xmm_id = xmm_begin;
            const auto get_temp_xmm = [&] {
                const Xbyak::Xmm temp {xmm_id++};

                if (xmm_id > xmm_end) { xmm_id = xmm_begin; }

                return temp;
            };
            const bool mayiuse_avx2 = mayiuse(avx2);
            const auto uni_vpaddd_wrapper
                    = [&](const Xmm &xmm, const Address &addr) {
                          if (mayiuse_avx2)
                              vpaddd(xmm, xmm, addr);
                          else {
                              //isas < avx2 demand paddd instruction addr to be aligned
                              uni_vmovups(xmm_tmp_, addr);
                              paddd(xmm, xmm_tmp_);
                          }
                      };
            if (can_store_xmm) {
                enum class comp_load_type_t { bcast, load, gather };

                for (int ur = 0; ur < reg_unroll; ur += ur_step) {

                    bool all_ip_padding_one = true;
                    bool all_ip_padding_zero = true;
                    for (int r = ur; r < ur + ur_step; r++) {
                        if (zero_padding[r] != 1)
                            all_ip_padding_one = false;
                        else
                            all_ip_padding_zero = false;
                    }
                    if (all_ip_padding_one) continue;

                    comp_load_type_t comp_load_type = comp_load_type_t::bcast;

                    for (int r = ur + 1; r < ur + ur_step; ++r)
                        if (c_off[r] != c_off[r - 1] + 0) {
                            comp_load_type = comp_load_type_t::load;
                            break;
                        }

                    if (comp_load_type == comp_load_type_t::bcast
                            && all_ip_padding_zero) {
                        const auto reduction_xmm = get_temp_xmm();
                        const auto xmm_reorder_result = Xmm(ur);
                        uni_vcvtps2dq(reduction_xmm, xmm_reorder_result);
                        uni_vphaddd(
                                reduction_xmm, reduction_xmm, reduction_xmm);
                        uni_vphaddd(
                                reduction_xmm, reduction_xmm, reduction_xmm);
                        const auto comp_addr = c_addr(c_off[ur]);
                        const auto xmm_tmp_ = get_temp_xmm();
                        uni_vmovss(xmm_tmp_, comp_addr);
                        uni_vpaddd(xmm_tmp_, xmm_tmp_, reduction_xmm);
                        uni_vmovss(comp_addr, xmm_tmp_);
                        continue;
                    }

                    if (comp_load_type == comp_load_type_t::load)
                        for (int r = ur + 1; r < ur + ur_step; ++r)
                            if (c_off[r] != c_off[r - 1] + 1) {
                                comp_load_type = comp_load_type_t::gather;
                                break;
                            }

                    if (comp_load_type == comp_load_type_t::load
                            && all_ip_padding_zero) {
                        const auto xmm_reorder_result_dq = get_temp_xmm();
                        const auto xmm_reorder_result = Xmm(ur);
                        const auto comp_addr = c_addr(c_off[ur]);
                        uni_vcvtps2dq(
                                xmm_reorder_result_dq, xmm_reorder_result);
                        uni_vpaddd_wrapper(xmm_reorder_result_dq, comp_addr);
                        uni_vmovups(comp_addr, xmm_reorder_result_dq);
                        continue;
                    }

                    const auto xmm_reorder_result_dq = get_temp_xmm();
                    const auto xmm_reorder_result = Xmm(ur);
                    uni_vcvtps2dq(xmm_reorder_result_dq, xmm_reorder_result);

                    for (int r = ur; r < ur + ur_step; ++r) {
                        if (zero_padding[r] == 0 || !tail_processing) {
                            uni_vshufps(xmm_tmp_, xmm_reorder_result_dq,
                                    xmm_reorder_result_dq, r);
                            const Reg32 reg_tmp_32 = reg_tmp_.cvt32();
                            uni_vmovd(reg_tmp_32, xmm_tmp_);
                            const auto comp_addr = c_addr(c_off[r]);
                            add(comp_addr, reg_tmp_32);
                        }
                    }
                }
            } else {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    if (zero_padding[ur] == 0 || !tail_processing) {
                        const auto xmm_reorder_result_dq = get_temp_xmm();
                        const auto xmm_reorder_result = Xmm(ur);
                        const auto comp_addr = c_addr(c_off[ur]);
                        uni_vcvtps2dq(
                                xmm_reorder_result_dq, xmm_reorder_result);
                        uni_vpaddd_wrapper(xmm_reorder_result_dq, comp_addr);
                        uni_vmovss(comp_addr, xmm_reorder_result_dq);
                    }
                }
            }
        }

        for (int ur = 0; ur < reg_unroll; ur += ur_step) {
            if (prb_.req_src_zp || prb_.req_dst_zp) {
                const bool use_store_masks = !store_masks.empty();
                if (use_store_masks) {
                    const auto mask = ~store_masks[ur / ur_step];
                    uni_vblendps(Xmm(ur), Xmm(ur), xmm_zero_, mask);
                }
            }
            if (prb_.otype != f32)
                cvt2odt(Xmm(ur), prb_.otype, interim_f32 ? f32 : prb_.itype);

            store(o_addr(o_off[ur]), Xmm(ur), ur_step * otype_sz_);
        }
    }

    bool interim_f32_needed() {
        using namespace data_type;

        return utils::one_of(f32, prb_.itype, prb_.otype)
                || prb_.scale_type != scale_type_t::NONE || prb_.beta != 0.f
                || ((prb_.req_src_zp || prb_.req_dst_zp)
                                ? !(prb_.itype == s32 && prb_.otype == s32)
                                : false)
                || (prb_.itype != f32 && compensation_needed_)
                || prb_.scale_adjust != 1.f;
    }

    void process_unroll_generic(
            const int ndims, int len, const bool tail_processing) {
        assert(IMPLICATION(prb_.nodes[0].tail_size > 0,
                len == static_cast<int>(prb_.nodes[0].n)
                        || len == static_cast<int>(prb_.nodes[0].tail_size)));

        const int blk = 8;

        int i_off[2 * blk] = {0};
        int o_off[2 * blk] = {0};
        int s_off[2 * blk] = {0};
        int c_off[2 * blk] = {0};

        int curr = 0; // will switch between 0 and 1

#define PARAM(x) ptr[abi_param1 + offsetof(call_param_t, x)]
        const bool interim_f32 = interim_f32_needed();

        if (prb_.req_src_zp) {
            uni_vbroadcastss(xmm_src_zp_, PARAM(src_zp));
            if (interim_f32) uni_vcvtdq2ps(xmm_src_zp_, xmm_src_zp_);
        }
        if (prb_.req_dst_zp) {
            uni_vbroadcastss(xmm_dst_zp_, PARAM(dst_zp));
            if (interim_f32) uni_vcvtdq2ps(xmm_dst_zp_, xmm_dst_zp_);
        }
#undef PARAM

        for (int off = 0; off < len; off += blk) {
            const int reg_unroll = nstl::min(off + blk, len) - off;
            int zero_padding[blk] = {0};
            const auto curr_blk = curr * blk;

            /* compute offsets and tail*/
            for (int ur = off != 0 ? 0 : 1; ur < reg_unroll; ++ur) {
                const int ur_c = curr_blk + ur;
                const int ur_p = (ur_c - 1 + 2 * blk) % (2 * blk); // prev ur
                const bool is_tail
                        = off + ur >= static_cast<int>(prb_.nodes[0].tail_size);
                step(off + ur, i_off[ur_p], o_off[ur_p], s_off[ur_p],
                        c_off[ur_p], i_off[ur_c], o_off[ur_c], s_off[ur_c],
                        c_off[ur_c]);
                if (tail_processing && is_tail) zero_padding[ur] = 1;
            }

            process_unroll_generic_step(reg_unroll, i_off + curr_blk,
                    o_off + curr_blk, s_off + curr_blk, c_off + curr_blk,
                    zero_padding, tail_processing);

            curr = 1 - curr;
        }
    }

    void compute_ker(
            const int ndims, const int len_unroll, const bool tail_processing) {
        bool optimized = false;
        optimized = optimized || process_direct_copy<avx>(ndims, len_unroll)
                || process_direct_copy<sse41>(ndims, len_unroll)
                || process_unroll_tr8x8(ndims, len_unroll);
        if (!optimized)
            process_unroll_generic(ndims, len_unroll, tail_processing);
    }

    void loop_begin(Label &l, Reg64 reg_cnt, int len) {
        mov(reg_cnt, len);
        L(l);
    }

    void check_if_this_is_last_chunk(const Reg64 &reg_curr_chunk, int node_id) {
        // Chunks are backwards numered i.e:
        // [0] -> [node_size]
        // [1] -> [node_size - 1]
        // ...
        // [node_size - 1] -> [1]

        // It is done like this, because it is easier to decrement counter
        // and check if it is equal to zero than increment and check
        // if it is equal to node_size.
        static constexpr int64_t last_chunk = 1;
        cmp(reg_curr_chunk, last_chunk);
    }

    void zero_dst_memory(const int bytes_to_zeroing) {
        static constexpr int num_of_bytes_in_xmm = 128 / 8;

        const int xmms_to_zeroing
                = std::div(bytes_to_zeroing, num_of_bytes_in_xmm).quot;
        const int tail_to_zeroing
                = std::div(bytes_to_zeroing, num_of_bytes_in_xmm).rem;

        uni_vpxor(xmm_tmp_, xmm_tmp_, xmm_tmp_);

        if (xmms_to_zeroing > 0) {
            Label loop;

            mov(reg_tmp_, xmms_to_zeroing);
            L(loop);
            uni_vmovups(o_addr(0), xmm_tmp_);
            add(reg_off_out_, num_of_bytes_in_xmm);
            dec(reg_tmp_);
            jnz(loop);
        }

        for (int i = 0; i < tail_to_zeroing; i++)
            uni_vpextrb(o_addr(i, false), xmm_tmp_, 0);

        // Restore dst offset to initial value
        if (xmms_to_zeroing > 0)
            sub(reg_off_out_, num_of_bytes_in_xmm * xmms_to_zeroing);
    }

    void finalize_tail_loop(int i_step, int o_step, int s_step, int c_step,
            const int curr_node_id) {
#define CURR_DATA_CHUNK(node_id) \
    ptr[abi_param1 + offsetof(call_param_t, curr_data_chunks) \
            + sizeof(int64_t) * (node_id)]

        static constexpr int empty_chunk_info = -1;

        mov(reg_tmp_, empty_chunk_info);
        mov(CURR_DATA_CHUNK(curr_node_id), reg_tmp_);

        const int padded_area = prb_.nodes[curr_node_id].n
                - prb_.nodes[curr_node_id].tail_size;

        if (prb_.nodes[curr_node_id].is_zero_pad_needed) {
            int num_of_zero_padded_values = padded_area;
            for (int i = curr_node_id - 1; i >= 0; i--) {
                num_of_zero_padded_values *= prb_.nodes[i].n;
            }

            const int bytes_to_zeroing = num_of_zero_padded_values * otype_sz_;
            zero_dst_memory(bytes_to_zeroing);
        }

        // This function is called by loop_end. At the end
        // of loop_end is section that is responsible for
        // restoring offset values. Restoring is based on
        // len value which is equal to prb.nodes[x].n.
        // If fill_zero_padded_area is called then it means
        // offsets were shifted prb.nodes[x].tail_size times.
        // Therefore, this function has to shift offsets by
        // zero pad area.
        add(reg_off_in_, padded_area * i_step * itype_sz_);
        add(reg_off_out_, padded_area * o_step * otype_sz_);
        if (prb_.scale_type == scale_type_t::MANY)
            add(reg_off_scale_, padded_area * s_step * stype_sz_);
        if (compensation_needed_)
            add(reg_off_comp_, padded_area * c_step * sizeof(int32_t));

#undef CURR_DATA_CHUNK
    }

    void loop_end(Label &l, const Reg64 &reg_cnt, int len, int i_step,
            int o_step, int s_step, int c_step, const int curr_node_id) {
        add(reg_off_in_, i_step * itype_sz_);
        add(reg_off_out_, o_step * otype_sz_);
        if (prb_.scale_type == scale_type_t::MANY)
            add(reg_off_scale_, s_step * stype_sz_);
        if (compensation_needed_) add(reg_off_comp_, c_step * sizeof(int32_t));

        dec(reg_cnt);
        jnz(l);

        if (prb_.tail(curr_node_id) != 0) {
            Label if_end;

            // On the stack should be an information if node
            // was processed with tail or not.
            pop(reg_tmp_);

            cmp(reg_tmp_, with_tail_info_);
            jne(if_end, T_NEAR);
            finalize_tail_loop(i_step, o_step, s_step, c_step, curr_node_id);
            L(if_end);
        }

        // Restore offset to initial values. It means before
        // loop execution.
        sub(reg_off_in_, len * i_step * itype_sz_);
        sub(reg_off_out_, len * o_step * otype_sz_);
        if (prb_.scale_type == scale_type_t::MANY)
            sub(reg_off_scale_, len * s_step * stype_sz_);
        if (compensation_needed_)
            sub(reg_off_comp_, len * c_step * sizeof(int32_t));
    }

    void compute_blk_ker(const simple_impl_desc_t &desc) {
#define CURR_DATA_CHUNK(node_id) \
    ptr[abi_param1 + offsetof(call_param_t, curr_data_chunks) \
            + sizeof(int64_t) * (node_id)]

        static constexpr bool with_tail_processing = true;
        Label no_last_chunk, end_label;
        int omp_ndims = prb_.full_ndims - prb_.ndims;

        if (prb_.nodes[0].tail_size > 0) {
            if (!prb_.nodes[0].is_parent_empty()) {
                const int parent_node_id = prb_.nodes[0].parent_node_id;
                mov(reg_tmp_, CURR_DATA_CHUNK(parent_node_id));
                check_if_this_is_last_chunk(reg_tmp_, parent_node_id);
                jne(no_last_chunk, T_NEAR);
            }

            const int len_unroll = desc.tail_len_unroll > 0
                    ? desc.tail_len_unroll
                    : desc.len_unroll;
            compute_ker(omp_ndims, len_unroll, with_tail_processing);
            jmp(end_label, T_NEAR);
        }

        L(no_last_chunk);
        compute_ker(omp_ndims, desc.len_unroll, !with_tail_processing);
        L(end_label);

#undef CURR_DATA_CHUNK
    }

    void create_loops(const simple_impl_desc_t &desc,
            const std::array<const Reg64, 3> &reg_cnt, int jit_loop) {
#define CURR_DATA_CHUNK(node_id) \
    ptr[abi_param1 + offsetof(call_param_t, curr_data_chunks) \
            + sizeof(int64_t) * (node_id)]

        assert(jit_loop <= ndims_jit_loop_max);

        if (jit_loop > 0) {
            const int nfu = desc.ndims_full_unroll;
            const int unroll_factor
                    = jit_loop == 1 ? desc.len_last_dim_unroll : 1;
            const int curr_node_id = nfu + (jit_loop - 1);
            const int parent_node_id = prb_.nodes[curr_node_id].parent_node_id;
            const int tail_size = prb_.tail(curr_node_id) / unroll_factor;
            const int node_size = prb_.n(curr_node_id) / unroll_factor;
            const Reg64 &reg_loop_cnt = reg_cnt[jit_loop - 1];
            const bool curr_node_has_tail = prb_.tail(curr_node_id) != 0;
            Label loop, if_no_tail, if_end;

            if (curr_node_has_tail) {
                if (prb_.nodes[curr_node_id].is_parent_empty()) {
                    mov(reg_loop_cnt, tail_size);
                    // Put info that node is being processed with tail.
                    mov(reg_tmp_, with_tail_info_);
                    push(reg_tmp_);
                } else {
                    mov(reg_tmp_, CURR_DATA_CHUNK(parent_node_id));
                    check_if_this_is_last_chunk(reg_tmp_, parent_node_id);
                    jne(if_no_tail, T_NEAR);
                    mov(reg_loop_cnt, tail_size);
                    // Put info that node is being processed with tail.
                    mov(reg_tmp_, with_tail_info_);
                    push(reg_tmp_);
                    jmp(if_end, T_NEAR);

                    L(if_no_tail);
                    mov(reg_loop_cnt, node_size);
                    // Put info that node is being processed without tail.
                    mov(reg_tmp_, without_tail_info_);
                    push(reg_tmp_);
                    L(if_end);
                }
            }

            if (prb_.is_tail_in_one_of_child_nodes(curr_node_id)) {
                if (!curr_node_has_tail) {
                    mov(reg_loop_cnt, node_size);
                    mov(CURR_DATA_CHUNK(curr_node_id), reg_loop_cnt);
                }
                L(loop);
                if (!prb_.nodes[curr_node_id].is_parent_empty()) {
                    Label if_no_tail_in_child_node;
                    mov(reg_tmp_, CURR_DATA_CHUNK(parent_node_id));
                    check_if_this_is_last_chunk(reg_tmp_, parent_node_id);
                    jne(if_no_tail_in_child_node, T_NEAR);
                    mov(CURR_DATA_CHUNK(curr_node_id), reg_loop_cnt);
                    L(if_no_tail_in_child_node);
                } else {
                    mov(CURR_DATA_CHUNK(curr_node_id), reg_loop_cnt);
                }
            } else if (curr_node_has_tail) {
                L(loop);
            } else {
                loop_begin(loop, reg_loop_cnt, node_size);
            }

            create_loops(desc, reg_cnt, jit_loop - 1);

            loop_end(loop, reg_loop_cnt, node_size,
                    prb_.is(curr_node_id) * unroll_factor,
                    prb_.os(curr_node_id) * unroll_factor,
                    prb_.ss(curr_node_id) * unroll_factor,
                    prb_.cs(curr_node_id) * unroll_factor, curr_node_id);
        } else {
            compute_blk_ker(desc);
        }

#undef CURR_DATA_CHUNK
    }

    bool simple_impl() {
        simple_impl_desc_t d;
        if (!simple_impl_desc_init(prb_, &d)) return false;

        xor_(reg_off_in_, reg_off_in_);
        xor_(reg_off_out_, reg_off_out_);
        if (prb_.scale_type == scale_type_t::MANY)
            xor_(reg_off_scale_, reg_off_scale_);
        if (compensation_needed_) xor_(reg_off_comp_, reg_off_comp_);

        std::array<const Reg64, 3> reg_cnt({{r15, r14, r13}});

        const int n_jit_loops = prb_.ndims - d.ndims_full_unroll;
        create_loops(d, reg_cnt, n_jit_loops);

        return true;
    }

    void impl() {
        if (simple_impl()) return;
        assert(!"no implementation available");
    }

    jit_uni_reorder_kernel_f32_t(const desc_t &desc)
        : kernel_t(desc), bf16_emu_(nullptr) {
        itype_sz_ = data_type_size(prb_.itype);
        otype_sz_ = data_type_size(prb_.otype);
        stype_sz_ = sizeof(float);
        if (prb_.otype == data_type::bf16 && !mayiuse(avx512_core_bf16)) {
            bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
                    bf16_emu_reserv_1_, bf16_emu_reserv_2_, bf16_emu_reserv_3_,
                    bf16_emu_scratch_, bf16_emu_reserv_4_);
        }
    }

    void generate() override {
        Label end_of_kernel;

        preamble();

        if (bf16_emu_) bf16_emu_->init_vcvtneps2bf16();

#define PARAM(x) ptr[abi_param1 + offsetof(call_param_t, x)]
        if (prb_.scale_type == scale_type_t::COMMON) {
            auto reg_ptr_scale__tmp = reg_ptr_in_;
            mov(reg_ptr_scale__tmp, PARAM(scale));
            uni_vbroadcastss(xmm_scale_, ptr[reg_ptr_scale__tmp]);
        } else if (prb_.scale_type == scale_type_t::MANY) {
            mov(reg_ptr_scale_, PARAM(scale));
        }
        if (compensation_needed_)
            mov(reg_ptr_comp_, PARAM(compensation_scratch));
        if (prb_.scale_adjust == 0.5f) { mov(reg_scale_adjust_, 0x3f000000); }
        mov(reg_ptr_in_, PARAM(in));
        mov(reg_ptr_out_, PARAM(out));

        bool is_tail_in_drv_dims = false;
        for (int i = prb_.ndims; i < prb_.full_ndims; i++)
            if (prb_.nodes[i].tail_size > 0) {
                is_tail_in_drv_dims = true;
                break;
            }

        if (is_tail_in_drv_dims) {
            Label reorder_kernel;

            mov(reg_tmp_, PARAM(skip_kernel_execution));
            cmp(reg_tmp_, static_cast<int64_t>(true));
            je(end_of_kernel, T_NEAR);

            mov(reg_tmp_, PARAM(zeroing_data));
            cmp(reg_tmp_, static_cast<int64_t>(false));
            je(reorder_kernel, T_NEAR);
            // If zeroing data is set then all dst memory
            // will be zeroed and nothing more will be done.
            int bytes_to_zeroing = otype_sz_;
            for (int i = 0; i < prb_.ndims; i++) {
                bytes_to_zeroing *= prb_.nodes[i].n;
            }
            xor_(reg_off_out_, reg_off_out_);
            zero_dst_memory(bytes_to_zeroing);
            jmp(end_of_kernel, T_NEAR);
            L(reorder_kernel);
        }
#undef PARAM

        mov(reg_last_loop_cnt_, 1);
        if (can_do_tr8x8()) {
            vxorps(ymm_zero_, ymm_zero_, ymm_zero_);

            if (prb_.itype == data_type::u8 && prb_.otype == data_type::s8) {
                mov(reg_tmp_, 0x7f7f7f7f7f7f7f7f);
                movq(Xmm(ymm_8x127b_.getIdx()), reg_tmp_);
            }
        } else {
            uni_vxorps(xmm_zero_, xmm_zero_, xmm_zero_);

            if (prb_.itype == data_type::u8 && prb_.otype == data_type::s8) {
                mov(reg_tmp_.cvt32(), 0x7f7f7f7f);
                movd(xmm_4x127b_, reg_tmp_.cvt32());
            }
        }

        impl();

        L(end_of_kernel);
        postamble();
    }

    ~jit_uni_reorder_kernel_f32_t() override = default;

private:
    static constexpr int64_t with_tail_info_ = static_cast<int64_t>(true);
    static constexpr int64_t without_tail_info_ = static_cast<int64_t>(false);

    int itype_sz_;
    int otype_sz_;
    int stype_sz_;

    const Reg64 &reg_ptr_in_ = rsi;
    const Reg64 &reg_ptr_out_ = rdx;
    const Reg64 &reg_ptr_scale_ = abi_not_param1;
    const Reg64 &reg_ptr_comp_ = rbx;
    const Reg32 &reg_scale_adjust_ = ebp;

    const Reg64 &reg_off_in_ = r8;
    const Reg64 &reg_off_out_ = r9;
    const Reg64 &reg_off_scale_ = r10;
    const Reg64 &reg_off_comp_ = r11;

    const Reg64 &reg_last_loop_cnt_ = r15;

    const Reg64 &reg_tmp_ = rax;

    const Xmm &xmm_scale_ = xmm15;
    const Xmm &xmm_zero_ = xmm14;
    const Xmm &xmm_4x127b_ = xmm13; // TODO: unite with ymm_zero_
    const Ymm &ymm_zero_ = ymm14;
    const Ymm &ymm_8x127b_ = ymm13;
    const Xmm &xmm_tmp_ = xmm12;
    const Xmm &xmm_src_zp_ = xmm9;
    const Xmm &xmm_dst_zp_ = xmm11;
    const Xmm &xmm_saturation_ubound_ = xmm12;
    const Ymm &ymm_saturation_ubound_ = ymm12;

    /* bf16 support on SKX */
    std::unique_ptr<bf16_emulation_t> bf16_emu_;
    const Zmm bf16_emu_reserv_1_ = Zmm(16);
    const Zmm bf16_emu_reserv_2_ = Zmm(17);
    const Reg64 &bf16_emu_scratch_ = reg_tmp_;
    const Zmm bf16_emu_reserv_3_ = Zmm(18);
    const Zmm bf16_emu_reserv_4_ = Zmm(19);
};

// Seperate class for no unroll/threading burden
struct jit_single_blk_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_single_blk_kernel)
    static bool applicable(const prb_t &p) {
        using namespace data_type;

        bool ok = p.ndims >= 2 && mayiuse(avx2)
                && p.scale_type == scale_type_t::NONE
                && utils::one_of(p.itype, f32) && utils::one_of(p.otype, f32)
                && utils::everyone_is(0, p.ioff, p.ooff) && p.beta == 0.f
                && prb_has_small_strides(p);
        if (!ok) return false;

        int64_t n0 = p.nodes[0].n;
        auto i0 = p.nodes[0].is;
        auto o0 = p.nodes[0].os;
        int64_t n1 = p.nodes[1].n;
        auto i1 = p.nodes[1].is;
        auto o1 = p.nodes[1].os;

        /*
         * for a transpose of plain to 8c case, nodes would be like:
         *     n    is   os
         *     m    1    8
         *     8    m    1
         * or
         *     8    m    1
         *     m    1    8
         */
        ok = (utils::one_of(n0, 8, 16) || utils::one_of(n1, 8, 16))
                && ((i0 == 1 && o1 == 1 && n0 == i1 && o0 == n1)
                        || (o0 == 1 && i1 == 1 && n0 == o1 && i0 == n1));
        if (!ok) return false;

        // Do not handle transpose of dimensions other than last 2
        for (int i = 2; i < p.ndims; ++i) {
            if (p.nodes[i].is != p.nodes[i].os) {
                ok = false;
                break;
            }
        }

        return ok;
    }

    jit_single_blk_kernel_t(const tr::prb_t &prb)
        : prb_(prb)
        , itype_sz_(data_type_size(prb_.itype))
        , otype_sz_(data_type_size(prb_.otype))
        , block_sz(prb.nodes[0].n) {}

    void generate() override {
        auto input_stride
                = prb_.nodes[0].is != 1 ? prb_.nodes[0].is : prb_.nodes[1].is;
        auto output_stride
                = prb_.nodes[0].os != 1 ? prb_.nodes[0].os : prb_.nodes[1].os;

        Label tail_processing;

        const auto load_zp = [&](const Ymm ymm_zp, const Reg64 reg_zp) {
            const Xmm xmm_zp = Xmm(ymm_zp.getIdx());
            uni_vmovq(xmm_zp, reg_zp);
            uni_vpbroadcastd(ymm_zp, xmm_zp);
            uni_vcvtdq2ps(ymm_zp, ymm_zp);
        };

        preamble();

        if (prb_.req_src_zp) load_zp(ymm_src_zp, reg_src_zp);

        if (prb_.req_dst_zp) load_zp(ymm_dst_zp, reg_dst_zp);

        cmp(reg_ptr_tail, true);
        je(tail_processing, T_NEAR);

        if (block_sz == 8) {
            gen_ker8x8(0, 0, input_stride, output_stride, 8, 8);
            block_sz = 8;
        } else if (block_sz == 16) {
            gen_ker16x16_in_8x8(input_stride, output_stride);
            block_sz = 16;
        } else {
            assert(!"unimplemented");
        }

        postamble();

        L(tail_processing);

        if (block_sz == 8) {
            auto i_tail = input_stride % 8 != 0 ? input_stride % 8 : 8;
            auto o_tail = output_stride % 8 != 0 ? output_stride % 8 : 8;
            if (i_tail != o_tail) {
                auto t_mask = i_tail == 8 ? o_tail : i_tail;
                gen_setmask(t_mask);
                gen_ker8x8(0, 0, input_stride, output_stride, i_tail, o_tail);
            }
        } else if (block_sz == 16) {
            auto i_tail = input_stride % 16 != 0 ? input_stride % 16 : 16;
            auto o_tail = output_stride % 16 != 0 ? output_stride % 16 : 16;
            if (i_tail != o_tail) {
                auto t_mask = i_tail == 16 ? o_tail : i_tail;
                t_mask %= 8;
                if (t_mask != 0) gen_setmask(t_mask);
                gen_ker16x16_in_8x8(
                        input_stride, output_stride, i_tail, o_tail);
            }
        } else {
            assert(!"unimplemented");
        }

        postamble();
    }

    void gen_loadu(const Ymm &ymm, const Address &addr, int size) {
        Xmm xmm(ymm.getIdx());
        switch (size) {
            case 32: vmovups(ymm, addr); break;
            case 16: vmovups(xmm, addr); break;
            default: assert(!"unreachable");
        }
    }

    void gen_storeu(const Address &addr, const Ymm &ymm, int size) {
        Xmm xmm(ymm.getIdx());
        switch (size) {
            case 32: vmovups(addr, ymm); break;
            case 16: vmovups(addr, xmm); break;
            default: assert(!"unreachable");
        }
    }

    void gen_maskloadu(
            const Ymm &ymm, const Address &addr, const Ymm mask, int size) {
        Xmm xmm(ymm.getIdx());
        Xmm mask128(mask.getIdx());
        switch (size) {
            case 32: vmaskmovps(ymm, mask, addr); break;
            case 16: vmaskmovps(xmm, mask128, addr); break;
            default: assert(!"unreachable");
        }
    }

    void gen_maskstoreu(
            const Address &addr, const Ymm &ymm, const Ymm mask, int size) {
        Xmm xmm(ymm.getIdx());
        Xmm mask128(mask.getIdx());
        switch (size) {
            case 32: vmaskmovps(addr, mask, ymm); break;
            case 16: vmaskmovps(addr, mask128, xmm); break;
            default: assert(!"unreachable");
        }
    }

    // Register allocation xmm0~11
    void gen_transpose_8x8() {
        constexpr int lane = 8;
        for (int i = 0; i < lane / 2; i++) {
            vunpcklps(Ymm(lane + i), Ymm(2 * i), Ymm(2 * i + 1));
            vunpckhps(Ymm(i), Ymm(2 * i), Ymm(2 * i + 1));
        }

        const unsigned int lfloat = 0x44;
        const unsigned int ufloat = 0xee;
        for (int i = 0; i < lane / 2; i++) {
            int j = i % 2 == 0 ? lane + i : i - 1;
            vshufps(Ymm(lane / 2 + 2 * i), Ymm(j), Ymm(j + 1), lfloat);
            vshufps(Ymm(lane / 2 + 2 * i + 1), Ymm(j), Ymm(j + 1), ufloat);
        }

        const unsigned int lquad = 0x20;
        for (int i = 0; i < lane / 2; i++)
            vperm2f128(Ymm(i), Ymm(lane / 2 + i), Ymm(lane + i), lquad);

        const unsigned int uquad = 0x31;
        for (int i = lane / 2; i < lane; i++)
            vperm2f128(Ymm(i), Ymm(i), Ymm(lane / 2 + i), uquad);
    }

    // keep order nchw -> nChw()C
    // or nChw()C -> nchw
    void gen_setmask(int mask) {
        // all 0, all 1
        vxorps(ymm_tmp, ymm_tmp, ymm_tmp);
        vpcmpeqd(ymm_mask, ymm_mask, ymm_mask);
        // shift by mask to have tail nelems in ymm_mask
        const uint8_t in_mask = 0xFF << mask;
        vpblendd(ymm_mask, ymm_mask, ymm_tmp, in_mask);
    }

    // TODO: Mark parameter with type information
    // XXX: !
    // offset in byte offset
    // stride in element number
    //
    // Gen specific 8x8 transform respect to certain tail condition
    void gen_tr8x8(int i_off, int o_off, int input_stride, int output_stride,
            int in_tail, int out_tail) {
        constexpr int lane = 8;

        if (in_tail == 0 || out_tail == 0) return;

        for (int i = 0; i < out_tail; ++i) {
            if (in_tail != lane) {
                gen_maskloadu(Ymm(i),
                        ptr[reg_ptr_in_ + i_off + i * input_stride * itype_sz_],
                        ymm_mask, lane * itype_sz_);
            } else {
                gen_loadu(Ymm(i),
                        ptr[reg_ptr_in_ + i_off + i * input_stride * itype_sz_],
                        lane * itype_sz_);
            }
            if (prb_.req_src_zp) { vsubps(Ymm(i), Ymm(i), ymm_src_zp); }
        }

        gen_transpose_8x8();

        for (int i = 0; i < in_tail; ++i) {
            if (prb_.req_dst_zp) { vaddps(Ymm(i), Ymm(i), ymm_dst_zp); }
            if (out_tail == lane) {
                gen_storeu(ptr[reg_ptr_out_ + o_off
                                   + i * output_stride * otype_sz_],
                        Ymm(i), lane * otype_sz_);
            } else {
                gen_maskstoreu(ptr[reg_ptr_out_ + o_off
                                       + i * output_stride * otype_sz_],
                        Ymm(i), ymm_mask, lane * otype_sz_);
            }
        }
    }

    // tail: 0 ~ 8
    // support: either in_tail or out_tail is not 8, but not both
    void gen_ker8x8(int i_off, int o_off, int input_stride, int output_stride,
            int in_tail, int out_tail) {
        gen_tr8x8(i_off, o_off, input_stride, output_stride, in_tail, out_tail);
    }

    void gen_ker16x16_in_8x8(int input_stride, int output_stride) {
        const auto lane = 16;
        const auto sub_lane = lane / 2;
        gen_tr8x8(0, 0, input_stride, output_stride, sub_lane, sub_lane);
        gen_tr8x8(input_stride * sub_lane * itype_sz_, sub_lane * otype_sz_,
                input_stride, output_stride, sub_lane, sub_lane);
        gen_tr8x8(sub_lane * itype_sz_, output_stride * sub_lane * otype_sz_,
                input_stride, output_stride, sub_lane, sub_lane);
        gen_tr8x8((input_stride * sub_lane + sub_lane) * itype_sz_,
                (output_stride * sub_lane + sub_lane) * otype_sz_, input_stride,
                output_stride, sub_lane, sub_lane);
    }

    // tail can be 1 ~ 16, using avx2 for now
    void gen_ker16x16_in_8x8(
            int input_stride, int output_stride, int in_tail, int out_tail) {
        constexpr auto lane = 16;
        constexpr auto sub_lane = lane / 2;
        auto tail = in_tail != lane ? in_tail : out_tail;

        const auto l_tail = tail < sub_lane ? tail : sub_lane;
        const auto u_tail = tail < sub_lane ? 0 : tail - sub_lane;

        if (tail == in_tail) {
            gen_tr8x8(0, 0, input_stride, output_stride, l_tail, sub_lane);
            gen_tr8x8(input_stride * sub_lane * itype_sz_, sub_lane * otype_sz_,
                    input_stride, output_stride, l_tail, sub_lane);
            gen_tr8x8(sub_lane * itype_sz_,
                    output_stride * sub_lane * otype_sz_, input_stride,
                    output_stride, u_tail, sub_lane);
            gen_tr8x8(itype_sz_ * (input_stride * sub_lane + sub_lane),
                    otype_sz_ * (output_stride * sub_lane + sub_lane),
                    input_stride, output_stride, u_tail, sub_lane);
        } else {
            gen_tr8x8(0, 0, input_stride, output_stride, sub_lane, l_tail);
            gen_tr8x8(input_stride * sub_lane * itype_sz_, sub_lane * otype_sz_,
                    input_stride, output_stride, sub_lane, u_tail);
            gen_tr8x8(sub_lane * itype_sz_,
                    output_stride * sub_lane * itype_sz_, input_stride,
                    output_stride, sub_lane, l_tail);
            gen_tr8x8(itype_sz_ * (input_stride * sub_lane + sub_lane),
                    otype_sz_ * (output_stride * sub_lane + sub_lane),
                    input_stride, output_stride, sub_lane, u_tail);
        }
    }

private:
    // 6 ~ 12
    constexpr static int xmm_save_for_windows = is_windows ? 7 : 0;
    constexpr static int xmm_save_start_from = 6;
    constexpr static int xmm_width = 16;

    void preamble() {
        if (is_windows) {
            // retrieve 5th function call argument from call stack
            static constexpr int param5 = 0x8;
            mov(reg_dst_zp, ptr[rsp + param5]);
            sub(rsp, xmm_save_for_windows * xmm_width);
            for (int i = 0; i < xmm_save_for_windows; ++i) {
                uni_vmovdqu(ptr[rsp + i * xmm_width],
                        Xbyak::Xmm(xmm_save_start_from + i));
            }
        }
    }

    void postamble() {
        if (is_windows) {
            for (int i = 0; i < xmm_save_for_windows; ++i)
                uni_vmovdqu(Xbyak::Xmm(xmm_save_start_from + i),
                        ptr[rsp + i * xmm_width]);
            add(rsp, xmm_save_for_windows * xmm_width);
        }
        uni_vzeroupper();
        ret();
    }

    const prb_t &prb_;

    int itype_sz_;
    int otype_sz_;
    int block_sz;

    Reg64 reg_ptr_in_ = abi_param1;
    Reg64 reg_ptr_out_ = abi_param2;
    // Windows bool is 1-byte in register
    Reg8 reg_ptr_tail = is_windows ? r8b : dl;
    Reg64 reg_src_zp = abi_param4;
    Reg64 reg_dst_zp = is_windows ? r10 : r8;

    Ymm ymm_mask = ymm12;
    Ymm ymm_tmp = ymm0;
    Ymm ymm_src_zp = ymm14;
    Ymm ymm_dst_zp = ymm15;
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
    static constexpr int num_elems_thr = 16;
    const bool cache_blocking_needed
            = ((prb.nodes[0].is % 64 == 0 && prb.nodes[0].n > num_elems_thr)
                      || (prb.ndims > 1 && prb.nodes[1].is % num_elems_thr == 0
                              && prb.nodes[1].n > num_elems_thr))
            && !prb.is_tail_present;
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

        const bool split_needed = (num_elems > num_elems_thr) && (num_elems % num_elems_thr == 0);
        const int move_location = (output_stride % 4 != 0) ? 0 : 1;
        if (split_needed) prb_node_split(prb, unit_input_stride_idx, num_elems_thr);

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
    size_t size_total = 1;
    for (int d = 0; d < prb.ndims; ++d)
        size_total *= prb.nodes[d].n;

    /* The general expression for size_drv_thr can be written as
     * size_drv_min = C0 + FC * (nthr > 1 ? 1 : 0) + VC * (nthr - 1)
     * where FC and VC are fixed and variable costs respectively.
     * Though for now, the below heuristic seems to be good enough */
    const size_t size_drv_thr = (nthr > 1) ? 16 * nthr : 1;

    /* size_drv_min is the minimal size for the parallel
     * driver required for good parallelization */
    const size_t size_drv_min
            = nstl::min<size_t>(size_drv_thr, utils::div_up(size_total, 1024));

    /* kdims -- # of dimensions processed by a kernel
     * size_ker_cur -- product of the dimension processed by a kernel
     * size_drv_cur -- product of the dimension processed by a driver */

    int kdims = prb.ndims;
    size_t size_drv_cur = 1;
    for (; kdims > 1 && size_drv_cur < size_drv_min; --kdims)
        size_drv_cur *= prb.nodes[kdims - 1].n;

    size_t size_ker_cur = 1;
    for (int d = 0; d < kdims; ++d)
        size_ker_cur *= prb.nodes[d].n;

    /* Initially kdims is chosen so that size_drv_cur >= size_drv_min.
     *
     * It might happen that for chosen kdims the size_ker_cur is too small
     * (less than tr::ker_prb_size_min). In that case try to split the
     * innermost driver dimension into two, to increase size_ker_cur. */
    const bool want_borrow_ker_from_drv = kdims < prb.ndims
            && size_ker_cur < tr::ker_prb_size_min
            && size_drv_cur > size_drv_min;
    if (want_borrow_ker_from_drv) {
        /* size_want_borrow is the minimal size, so that:
         *  o) size_ker_cur * size_want_borrow >= tr::ker_prb_size_min
         *  o) current innermost driver dimension is divisible by
         *     size_want_borrow (so that we can evenly split that
         *     dimension into two)
         *
         *  In the worst case the minimal size_want_borrow is equal
         *  to the innermost driver dimension itself. In that case
         *  we will sacrifice it in favor of kernel (is it fine?). */
        size_t size_want_borrow
                = utils::div_up(tr::ker_prb_size_min, size_ker_cur);
        for (; prb.nodes[kdims].n % size_want_borrow; ++size_want_borrow)
            ;

        if (size_want_borrow != prb.nodes[kdims].n)
            prb_node_split(prb, kdims, size_want_borrow);
        kdims += 1;
    }

    /* On the other hand it might happen that for chosen kdims
     * the size_drv_cur is too small (less than size_drv_min). In that case
     * try to split the outermost kernel dimension into two, to increase
     * size_drv_cur. */
    const bool want_borrow_drv_from_ker = size_ker_cur > tr::ker_prb_size_min
            && size_drv_cur < size_drv_min;
    if (want_borrow_drv_from_ker) {
        size_t size_want_borrow = utils::div_up(size_drv_min, size_drv_cur);
        for (; prb.nodes[kdims - 1].n % size_want_borrow; ++size_want_borrow)
            ;

        if (size_want_borrow != prb.nodes[kdims - 1].n)
            prb_node_split(
                    prb, kdims - 1, prb.nodes[kdims - 1].n / size_want_borrow);
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

status_t jit_uni_reorder_t::pd_t::init(
        engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
    CHECK(cpu_reorder_pd_t::init(engine, src_engine, dst_engine));

    const bool compensation_needed
            = prb_.req_s8s8_comp || prb_.req_asymmetric_comp;
    if (compensation_needed) init_scratchpad();

    return status::success;
}

void jit_uni_reorder_t::pd_t::init_scratchpad() {
    const memory_desc_wrapper id(src_md());
    const auto G = with_groups_ ? id.dims()[0] : 1;
    const auto N = id.dims()[with_groups_ ? 1 : 0];
    static constexpr int cache_line_size = 16;
    const auto wspace_per_thr_size
            = utils::rnd_up(G * N, cache_line_size) * sizeof(int32_t);

    auto scratchpad = scratchpad_registry().registrar();
    const auto compensation_reduce_size = wspace_per_thr_size * nthr_;

    //every thread gets its own scratchpad space for each N
    scratchpad.template book<int32_t>(memory_tracking::names::key_reorder_space,
            compensation_reduce_size);
}

static bool is_with_groups(const memory_desc_t &dst_md) {

    auto dst_d = memory_desc_wrapper(dst_md);
    const int grp_bit = 1 << 1;
    bool with_groups
            = (dst_d.extra().flags & memory_extra_flags::compensation_conv_s8s8)
            && (dst_d.extra().compensation_mask & grp_bit);
    with_groups = with_groups
            || ((dst_d.extra().flags
                        & memory_extra_flags::compensation_conv_asymmetric_src)
                    && (dst_d.extra().asymm_compensation_mask & grp_bit));
    return with_groups;
}

status_t jit_uni_reorder_t::pd_t::create(reorder_pd_t **reorder_pd,
        engine_t *engine, const primitive_attr_t *attr, engine_t *src_engine,
        const memory_desc_t *src_md, engine_t *dst_engine,
        const memory_desc_t *dst_md) {
    auto prb = tr::prb_t();

    const bool with_groups = is_with_groups(*dst_md);

    status_t prb_init_status
            = prb_init(prb, *src_md, *dst_md, attr, with_groups);
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

    if (prb.is_tail_present) prb_node_dependency(prb);

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

    _pd->nthr_ = nthr;
    _pd->prb_ = prb;
    _pd->with_groups_ = with_groups;
    if (_pd->init(engine, src_engine, dst_engine) != status::success) {
        delete _pd;
        return status::unimplemented;
    }
    _pd->ker_desc_ = ker_desc;
    _pd->init_scratchpad_md();

    return safe_ptr_assign(*reorder_pd, _pd);
}

void jit_uni_reorder_t::omp_driver_0d(int off, const char *in, char *out,
        const float *scale, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;

    auto c = tr::call_param_t();
    c.in = in;
    c.out = out;
    c.scale = scale;

    if (prb.is_tail_present) {
        static constexpr int omp_ndims = 0;
        fill_curr_data_chunks(prb, off, nullptr, omp_ndims, c);
    }

    c.src_zp = src_zp;
    c.dst_zp = dst_zp;
    c.compensation_scratch = compensation_scratch;
    (*kernel_)(&c);
}

void jit_uni_reorder_t::omp_driver_1d(int ithr, int nthr, int off,
        const char *in, char *out, const float *scale, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
    for_nd(ithr, nthr, (ptrdiff_t)ns[0].n, [&](ptrdiff_t d0) {
        auto c = tr::call_param_t();
        c.in = in + d0 * ns[0].is * data_type_size(pd()->prb_.itype);
        c.out = out + d0 * ns[0].os * data_type_size(pd()->prb_.otype);
        c.scale = scale + d0 * ns[0].ss;

        if (prb.is_tail_present) {
            static constexpr int omp_ndims = 1;
            const ptrdiff_t omp_data_chunks[omp_ndims] = {d0};
            fill_curr_data_chunks(prb, off, omp_data_chunks, omp_ndims, c);
        }

        c.compensation_scratch = compensation_scratch + d0 * ns[0].cs;
        c.src_zp = src_zp;
        c.dst_zp = dst_zp;
        (*kernel_)(&c);
    });
}

void jit_uni_reorder_t::omp_driver_2d(int ithr, int nthr, int off,
        const char *in, char *out, const float *scale, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
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

                if (prb.is_tail_present) {
                    static constexpr int omp_ndims = 2;
                    const ptrdiff_t omp_data_chunks[omp_ndims] = {d0, d1};
                    fill_curr_data_chunks(
                            prb, off, omp_data_chunks, omp_ndims, c);
                }

                c.compensation_scratch
                        = compensation_scratch + d0 * ns[0].cs + d1 * ns[1].cs;
                c.src_zp = src_zp;
                c.dst_zp = dst_zp;
                (*kernel_)(&c);
            });
}

void jit_uni_reorder_t::omp_driver_3d(int ithr, int nthr, int off,
        const char *in, char *out, const float *scale, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
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

                if (prb.is_tail_present) {
                    static constexpr int omp_ndims = 3;
                    const ptrdiff_t omp_data_chunks[omp_ndims] = {d0, d1, d2};
                    fill_curr_data_chunks(
                            prb, off, omp_data_chunks, omp_ndims, c);
                }

                c.compensation_scratch = compensation_scratch + d0 * ns[0].cs
                        + d1 * ns[1].cs + d2 * ns[2].cs;
                c.src_zp = src_zp;
                c.dst_zp = dst_zp;
                (*kernel_)(&c);
            });
}

void jit_uni_reorder_t::omp_driver_4d(int ithr, int nthr, int off,
        const char *in, char *out, const float *scale, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
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

                if (prb.is_tail_present) {
                    static constexpr int omp_ndims = 4;
                    const ptrdiff_t omp_data_chunks[omp_ndims]
                            = {d0, d1, d2, d3};
                    fill_curr_data_chunks(
                            prb, off, omp_data_chunks, omp_ndims, c);
                }

                c.compensation_scratch = compensation_scratch + d0 * ns[0].cs
                        + d1 * ns[1].cs + d2 * ns[2].cs + d3 * ns[3].cs;
                c.src_zp = src_zp;
                c.dst_zp = dst_zp;
                (*kernel_)(&c);
            });
}

void jit_uni_reorder_t::omp_driver(const char *in, char *out,
        const float *scale, int src_zp, int dst_zp,
        const memory_tracking::grantor_t &scratchpad) const {
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
    const bool req_s8s8_comp = pd()->prb_.req_s8s8_comp;
    const bool req_asymmetric_comp = pd()->prb_.req_asymmetric_comp;
    const bool req_compensation = req_s8s8_comp || req_asymmetric_comp;
    assert(ndims - ndims_ker <= ndims_driver_max);

    int32_t *compensation_reduce_scratch = scratchpad.template get<int32_t>(
            memory_tracking::names::key_reorder_space);

    const memory_desc_wrapper id(pd()->src_md());
    const auto G = pd()->with_groups_ ? id.dims()[0] : 1;
    const auto N = id.dims()[pd()->with_groups_ ? 1 : 0];
    static constexpr int cache_line_size = 16;
    const auto wspace_per_thr_size = utils::rnd_up(G * N, cache_line_size);
    const auto wspace_per_thr_bytes = wspace_per_thr_size * sizeof(int32_t);

    if (ndims - ndims_ker == 0) {
        if (req_compensation)
            std::memset(compensation_reduce_scratch, 0, wspace_per_thr_bytes);

        omp_driver_0d(ndims_ker, in, out, scale, src_zp, dst_zp,
                compensation_reduce_scratch);
    } else {
        parallel(pd()->nthr_, [&](const int ithr, const int nthr) {
            int32_t *compensation_scratch = nullptr;
            if (req_compensation) {
                compensation_scratch = &compensation_reduce_scratch[ithr
                        * wspace_per_thr_size];
                std::memset(compensation_scratch, 0, wspace_per_thr_bytes);
            }

            switch (ndims - ndims_ker) {
                case 1:
                    omp_driver_1d(ithr, nthr, ndims_ker, in, out, scale, src_zp,
                            dst_zp, compensation_scratch);
                    break;
                case 2:
                    omp_driver_2d(ithr, nthr, ndims_ker, in, out, scale, src_zp,
                            dst_zp, compensation_scratch);
                    break;
                case 3:
                    omp_driver_3d(ithr, nthr, ndims_ker, in, out, scale, src_zp,
                            dst_zp, compensation_scratch);
                    break;
                case 4:
                    omp_driver_4d(ithr, nthr, ndims_ker, in, out, scale, src_zp,
                            dst_zp, compensation_scratch);
                    break;
                default: assert(!"unimplemented");
            }
        });
    }

    //reduction of intermediate compensation results to the final output
    if (req_compensation) {
        const int nthr = ndims - ndims_ker == 0 ? 1 : pd()->nthr_;
        reduce_compensation(
                out, compensation_reduce_scratch, nthr, wspace_per_thr_size);
    }
}

void jit_uni_reorder_t::reduce_compensation(char *out,
        const int32_t *compensation_reduce_scratch, const int nthr,
        const dim_t wspace_per_thr_size) const {

    const memory_desc_wrapper id(pd()->dst_md());
    const auto G = pd()->with_groups_ ? id.dims()[0] : 1;
    const auto N = id.dims()[pd()->with_groups_ ? 1 : 0];

    const memory_desc_wrapper od(pd()->dst_md());
    const size_t offset = od.size() - od.additional_buffer_size();

    static constexpr auto comp_dt_size = sizeof(int32_t);
    static constexpr int32_t comp_s8s8_shift = 128;

    // zero out the compensation memory in case of padding
    const auto G_padded = pd()->with_groups_ ? id.padded_dims()[0] : 1;
    const auto N_padded = id.padded_dims()[pd()->with_groups_ ? 1 : 0];
    const auto GN_padded_elems = G_padded * N_padded;
    const auto GN = G * N;
    const bool req_s8s8_comp = pd()->prb_.req_s8s8_comp;
    const bool req_asymmetric_comp = pd()->prb_.req_asymmetric_comp;
    const size_t zp_offset = offset
            + (pd()->prb_.req_s8s8_comp ? GN_padded_elems * comp_dt_size : 0);

    if (GN_padded_elems != GN) {
        if (req_s8s8_comp)
            std::memset(out + offset, 0, GN_padded_elems * comp_dt_size);
        if (req_asymmetric_comp)
            std::memset(out + zp_offset, 0, GN_padded_elems * comp_dt_size);
    }

    parallel_nd(G, N, [&](int g, int n) {
        int32_t acc = 0;
        const auto g_n_off = g * N + n;
        for (int ithr = 0; ithr < nthr; ithr++) {
            acc -= compensation_reduce_scratch[ithr * wspace_per_thr_size
                    + g_n_off];
        }
        if (req_s8s8_comp) {
            int32_t *out_comp = reinterpret_cast<int32_t *>(&out[offset]);
            out_comp[g_n_off] = comp_s8s8_shift * acc;
        }
        if (req_asymmetric_comp) {
            int32_t *out_asym_comp
                    = reinterpret_cast<int32_t *>(&out[zp_offset]);
            out_asym_comp[g_n_off] = acc;
        }
    });
}

void jit_uni_reorder_t::fill_curr_data_chunks(const tr::prb_t &prb,
        const int off, const ptrdiff_t *omp_data_chunks, const int omp_ndims,
        tr::call_param_t &c) const {
    // Chunks are backwards numered i.e:
    // [0] -> [node_size]
    // [1] -> [node_size - 1]
    // ...
    // [node_size - 1] -> [1]

    // It is done like this, because it is easier to decrement counter
    // and check if it is equal to zero than increment and check
    // if it is equal to node_size in jit kernel.

    static constexpr int64_t empty_chunk_info = -1;
    static constexpr int64_t last_chunk = 1;

    for (int curr_node_id = prb.ndims - 1; curr_node_id >= 0; curr_node_id--) {
        const int parent_node_id = prb.nodes[curr_node_id].parent_node_id;
        const bool is_drv_processing_this_node
                = curr_node_id >= off && curr_node_id <= off + omp_ndims - 1;
        const bool is_tail_processing
                = prb.is_tail_in_one_of_child_nodes(curr_node_id)
                || prb.nodes[curr_node_id].tail_size > 0;

        if (is_drv_processing_this_node && is_tail_processing) {
            const int inner_idx = curr_node_id - off;
            assert(inner_idx < omp_ndims);
            const int64_t node_size = prb.nodes[curr_node_id].tail_size > 0
                    ? prb.nodes[curr_node_id].tail_size
                    : prb.nodes[curr_node_id].n;
            const int64_t data_chunk = node_size - omp_data_chunks[inner_idx];

            if (!prb.nodes[curr_node_id].is_parent_empty()) {
                const bool is_parent_chunk_last
                        = c.curr_data_chunks[parent_node_id] == last_chunk;
                c.curr_data_chunks[curr_node_id]
                        = is_parent_chunk_last ? data_chunk : empty_chunk_info;
                c.zeroing_data = static_cast<int64_t>(
                        is_parent_chunk_last && data_chunk <= 0);
            } else {
                c.curr_data_chunks[curr_node_id] = data_chunk;
                c.zeroing_data = static_cast<int64_t>(data_chunk <= 0);
            }
            c.skip_kernel_execution = static_cast<int64_t>(c.zeroing_data
                    && !prb.nodes[curr_node_id].is_zero_pad_needed);
            if (c.zeroing_data || c.skip_kernel_execution) break;
        } else
            c.curr_data_chunks[curr_node_id] = empty_chunk_info;
    }
}

status_t jit_uni_reorder_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, tr::kernel_t::create(pd()->ker_desc_)));
    return kernel_->create_kernel();
}

status_t jit_uni_reorder_t::execute(const exec_ctx_t &ctx) const {
    auto in = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto out = CTX_OUT_MEM(char *, DNNL_ARG_TO);
    DEFINE_SCALES_BUFFER(scales);
    DEFINE_ZERO_POINT_VALUE(src_zp, DNNL_ARG_FROM);
    DEFINE_ZERO_POINT_VALUE(dst_zp, DNNL_ARG_TO);
    const auto &scratchpad = ctx.get_scratchpad_grantor();

    omp_driver(in, out, scales, src_zp, dst_zp, scratchpad);

    return status::success;
}

status_t jit_blk_reorder_t::pd_t::create(reorder_pd_t **reorder_pd,
        engine_t *engine, const primitive_attr_t *attr, engine_t *src_engine,
        const memory_desc_t *src_md, engine_t *dst_engine,
        const memory_desc_t *dst_md) {
    auto prb = tr::prb_t();

    status_t prb_init_status = prb_init(prb, *src_md, *dst_md, attr);
    if (prb_init_status != status::success) return prb_init_status;
    // only uni_reorder supports tail processing now
    // TODO: Add tail processing support in blk_reorder
    if (prb.is_tail_present) return status::unimplemented;

    // NB! Fall back to ref, if input and output both batch-strided
    bool batch_strided_input = false;
    bool batch_strided_output = false;
    if (prb.ndims > 1) {
        int batch_idx = prb.nodes[0].is > prb.nodes[1].is ? 0 : 1;
        int channel_idx = batch_idx == 0 ? 1 : 0;
        batch_strided_input =
                (ptrdiff_t) prb.nodes[channel_idx].n * prb.nodes[channel_idx].is < prb.nodes[batch_idx].is;
        batch_idx = prb.nodes[0].os > prb.nodes[1].os ? 0 : 1;
        channel_idx = batch_idx == 0 ? 1 : 0;
        batch_strided_output =
                (ptrdiff_t) prb.nodes[channel_idx].n * prb.nodes[channel_idx].is < prb.nodes[batch_idx].is;
    }

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
    prb_tile_normalize(prb);
    DEBUG({
        printf("tile : ");
        prb_dump(prb);
    });
    // NB! Fall back to ref, if input and output both batch-strided
    if (batch_strided_input && batch_strided_output)
        return status::unimplemented;

    if (!tr::jit_single_blk_kernel_t::applicable(prb)) {
        return status::unimplemented;
    }

    auto _pd = new pd_t(
            attr, src_engine->kind(), src_md, dst_engine->kind(), dst_md);
    if (_pd == nullptr) return status::out_of_memory;
    _pd->prb_ = prb;
    if (_pd->init(engine, src_engine, dst_engine) != status::success) {
        delete _pd;
        return status::unimplemented;
    }
    _pd->init_scratchpad_md();

    return safe_ptr_assign(*reorder_pd, _pd);
}

void jit_blk_reorder_t::pd_t::prb_tile_normalize(tr::prb_t &p) {
    if (!utils::one_of(p.nodes[0].n, 8ul, 16ul)
            && utils::one_of(p.nodes[1].n, 8ul, 16ul)) {
        nstl::swap(p.nodes[0], p.nodes[1]);
    }
}

jit_blk_reorder_t::jit_blk_reorder_t(const pd_t *apd) : primitive_t(apd) {}
jit_blk_reorder_t::~jit_blk_reorder_t() = default;

status_t jit_blk_reorder_t::init(engine_t *engine) {
    kernel_ = utils::make_unique<tr::jit_single_blk_kernel_t>(pd()->prb_);
    return kernel_->create_kernel();
}

status_t jit_blk_reorder_t::execute(const exec_ctx_t &ctx) const {
    const auto in = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto out = CTX_OUT_MEM(char *, DNNL_ARG_TO);
    DEFINE_ZERO_POINT_VALUE(src_zp, DNNL_ARG_FROM);
    DEFINE_ZERO_POINT_VALUE(dst_zp, DNNL_ARG_TO);

    // kernel handle 2-dimension tiles, a tail is possible
    auto &prb = this->pd()->prb_;
    ptrdiff_t BH = 1;
    for (int i = 2; i < prb.ndims; ++i) {
        BH *= prb.nodes[i].n;
    }

    auto block_sz = prb.n(0);
    auto n1 = prb.n(1);
    auto i1 = prb.is(1);
    auto o1 = prb.os(1);
    auto FL = (n1 + block_sz - 1) / block_sz;
    auto bh_stride = BH == 1 ? 0 : prb.is(2);

    auto itype_sz_ = data_type_size(pd()->prb_.itype);
    auto otype_sz_ = data_type_size(pd()->prb_.otype);

    parallel_nd(BH, FL, [&](dim_t bh, dim_t fl) {
        auto fl_b = fl * block_sz;
        auto bh_b = bh_stride * bh;
        auto *i = in + (bh_b + fl_b * i1) * itype_sz_;
        auto *o = out + (bh_b + fl_b * o1) * otype_sz_;
        (*kernel_)(i, o, n1 - fl_b < block_sz, src_zp, dst_zp);
    });

    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
