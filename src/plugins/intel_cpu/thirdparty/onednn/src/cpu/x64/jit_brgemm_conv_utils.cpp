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

#include "dnnl_types.h"

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_brgemm_conv_utils.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace prop_kind;
using namespace data_type;

namespace brgemm_convolution_utils {

inline status_t init_tag(format_tag_t &tag, memory_desc_t &md,
        const memory_desc_wrapper &mdw, const format_tag_t tag_value,
        bool any_eligible) {

    if (mdw.format_kind() == format_kind::any) {
        if (any_eligible) {
            CHECK(memory_desc_init_by_tag(md, tag_value));
            tag = tag_value;
        } else {
            tag = format_tag::undef;
        }
    } else {
        tag = mdw.matches_one_of_tag(tag_value);
    }

    if (tag != tag_value) return status::unimplemented;

    return status::success;
}

bool is_amx(cpu_isa_t isa) {
    return one_of(isa, avx512_core_bf16_amx_int8, avx512_core_bf16_amx_bf16);
}

bool post_ops_ok(jit_brgemm_conv_conf_t &jcp, primitive_attr_t &attr,
        const memory_desc_wrapper &dst_d) {
    using namespace injector;

    const auto &post_ops = attr.post_ops_;

    return injector::post_ops_ok(post_ops_ok_args_t(get_max_cpu_isa(),
            {sum, eltwise, binary}, post_ops, &dst_d,
            false /*sum_at_pos_0_only*/, false /*sum_requires_scale_one*/,
            false /*sum_requires_zp_zero*/,
            {broadcasting_strategy_t::per_oc,
                    broadcasting_strategy_t::scalar}));
}

status_t pick_tags(jit_brgemm_conv_conf_t &jcp, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md) {
    format_tag_t src_tag, dst_tag, wei_tag;
    dst_tag = pick(jcp.ndims - 3, nwc, nhwc, ndhwc);

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

    const bool is_1d = jcp.ndims == 3;
    const bool is_2d = jcp.ndims == 4;
    const bool is_3d = jcp.ndims == 5;

    if (jcp.wei_plain) {
        jcp.LDB = jcp.oc;
        if (is_3d) {
            if (jcp.wei_dt == f32)
                wei_tag = with_groups ? gdhwio : dhwio;
            else if (jcp.wei_dt == s8)
                wei_tag = with_groups ? gdhwIo4i : dhwIo4i;
            else if (jcp.wei_dt == bf16) {
                wei_tag = with_groups ? gdhwIo2i : dhwIo2i;
            } else
                return status::unimplemented;
        } else if (is_1d) {
            if (jcp.wei_dt == f32)
                wei_tag = with_groups ? gwio : wio;
            else if (jcp.wei_dt == s8)
                wei_tag = with_groups ? gwIo4i : wIo4i;
            else if (jcp.wei_dt == bf16) {
                wei_tag = with_groups ? gwIo2i : wIo2i;
            } else
                return status::unimplemented;
        } else {
            assert(is_2d);
            UNUSED(is_2d);
            if (jcp.wei_dt == f32)
                wei_tag = with_groups ? ghwio : hwio;
            else if (jcp.wei_dt == s8)
                wei_tag = with_groups ? ghwIo4i : hwIo4i;
            else if (jcp.wei_dt == bf16) {
                wei_tag = with_groups ? ghwIo2i : hwIo2i;
            } else
                return status::unimplemented;
        }
    } else {
        jcp.LDB = jcp.oc_block;
        if (jcp.oc_block == 64) {
            if (is_3d) {
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOdhwi64o : Odhwi64o;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOdhwI16i64o4i : OdhwI16i64o4i;
                    else
                        wei_tag = with_groups ? gOdhwI64o4i : OdhwI64o4i;
                } else if (jcp.wei_dt == bf16) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOdhwI16i64o2i : OdhwI16i64o2i;
                    else
                        wei_tag = with_groups ? gOdhwI64o2i : OdhwI64o2i;
                } else
                    return status::unimplemented;
            } else if (is_1d) {
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOwi64o : Owi64o;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOwI16i64o4i : OwI16i64o4i;
                    else
                        wei_tag = with_groups ? gOwI64o4i : OwI64o4i;
                } else if (jcp.wei_dt == bf16) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOwI16i64o2i : OwI16i64o2i;
                    else
                        wei_tag = with_groups ? gOwI64o2i : OwI64o2i;
                } else
                    return status::unimplemented;
            } else {
                assert(is_2d);
                UNUSED(is_2d);
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOhwi64o : Ohwi64o;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOhwI16i64o4i : OhwI16i64o4i;
                    else
                        wei_tag = with_groups ? gOhwI64o4i : OhwI64o4i;
                } else if (jcp.wei_dt == bf16) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOhwI16i64o2i : OhwI16i64o2i;
                    else
                        wei_tag = with_groups ? gOhwI64o2i : OhwI64o2i;
                } else
                    return status::unimplemented;
            }
        } else if (jcp.oc_block == 48) {
            if (is_3d) {
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOdhwi48o : Odhwi48o;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOdhwI16i48o4i : OdhwI16i48o4i;
                    else
                        wei_tag = with_groups ? gOdhwI48o4i : OdhwI48o4i;
                } else if (jcp.wei_dt == bf16) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOdhwI16i48o2i : OdhwI16i48o2i;
                    else
                        wei_tag = with_groups ? gOdhwI48o2i : OdhwI48o2i;
                } else
                    return status::unimplemented;
            } else if (is_1d) {
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOwi48o : Owi48o;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOwI16i48o4i : OwI16i48o4i;
                    else
                        wei_tag = with_groups ? gOwI48o4i : OwI48o4i;
                } else if (jcp.wei_dt == bf16) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOwI16i48o2i : OwI16i48o2i;
                    else
                        wei_tag = with_groups ? gOwI48o2i : OwI48o2i;
                } else
                    return status::unimplemented;
            } else {
                assert(is_2d);
                UNUSED(is_2d);
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOhwi48o : Ohwi48o;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOhwI16i48o4i : OhwI16i48o4i;
                    else
                        wei_tag = with_groups ? gOhwI48o4i : OhwI48o4i;
                } else if (jcp.wei_dt == bf16) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOhwI16i48o2i : OhwI16i48o2i;
                    else
                        wei_tag = with_groups ? gOhwI48o2i : OhwI48o2i;
                } else
                    return status::unimplemented;
            }
        } else if (jcp.oc_block == 32) {
            if (is_3d) {
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOdhwi32o : Odhwi32o;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOdhwI16i32o4i : OdhwI16i32o4i;
                    else
                        wei_tag = with_groups ? gOdhwI32o4i : OdhwI32o4i;
                } else if (jcp.wei_dt == bf16) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOdhwI16i32o2i : OdhwI16i32o2i;
                    else
                        wei_tag = with_groups ? gOdhwI32o2i : OdhwI32o2i;
                } else
                    return status::unimplemented;
            } else if (is_1d) {
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOwi32o : Owi32o;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOwI16i32o4i : OwI16i32o4i;
                    else
                        wei_tag = with_groups ? gOwI32o4i : OwI32o4i;
                } else if (jcp.wei_dt == bf16) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOwI16i32o2i : OwI16i32o2i;
                    else
                        wei_tag = with_groups ? gOwI32o2i : OwI32o2i;
                } else
                    return status::unimplemented;
            } else {
                assert(is_2d);
                UNUSED(is_2d);
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOhwi32o : Ohwi32o;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOhwI16i32o4i : OhwI16i32o4i;
                    else
                        wei_tag = with_groups ? gOhwI32o4i : OhwI32o4i;
                } else if (jcp.wei_dt == bf16) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOhwI16i32o2i : OhwI16i32o2i;
                    else
                        wei_tag = with_groups ? gOhwI32o2i : OhwI32o2i;
                } else
                    return status::unimplemented;
            }
        } else {
            if (is_3d) {
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOdhwi16o : Odhwi16o;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOdhwI16i16o4i : OdhwI16i16o4i;
                    else
                        wei_tag = with_groups ? gOdhwI16o4i : OdhwI16o4i;
                } else if (jcp.wei_dt == bf16) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOdhwI16i16o2i : OdhwI16i16o2i;
                    else
                        wei_tag = with_groups ? gOdhwI16o2i : OdhwI16o2i;
                } else
                    return status::unimplemented;
            } else if (is_1d) {
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOwi16o : Owi16o;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOwI16i16o4i : OwI16i16o4i;
                    else
                        wei_tag = with_groups ? gOwI16o4i : OwI16o4i;
                } else if (jcp.wei_dt == bf16) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOwI16i16o2i : OwI16i16o2i;
                    else
                        wei_tag = with_groups ? gOwI16o2i : OwI16o2i;
                } else
                    return status::unimplemented;
            } else {
                assert(is_2d);
                UNUSED(is_2d);

                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOhwi16o : Ohwi16o;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOhwI16i16o4i : OhwI16i16o4i;
                    else
                        wei_tag = with_groups ? gOhwI16o4i : OhwI16o4i;
                } else if (jcp.wei_dt == bf16) {
                    if (jcp.is_ic_padded)
                        wei_tag = with_groups ? gOhwI16i16o2i : OhwI16i16o2i;
                    else
                        wei_tag = with_groups ? gOhwI16o2i : OhwI16o2i;
                } else
                    return status::unimplemented;
            }
        }
    }

    src_tag = dst_tag;

    const bool any_eligible = (jcp.prop_kind == prop_kind::forward_inference
            || jcp.wei_dt == data_type::s8 || is_amx(jcp.isa));
    CHECK(init_tag(jcp.src_tag, src_md, src_d, src_tag, any_eligible));
    CHECK(init_tag(jcp.dst_tag, dst_md, dst_d, dst_tag, any_eligible));
    CHECK(init_tag(jcp.wei_tag, weights_md, weights_d, wei_tag, true));

    return status::success;
}

struct brg_blocking_t : public jit_brgemm_conv_conf_t {
    struct array_in_loop_t {
        dim_t itersize;
        float repeatn;
        float overlap;
        void set(dim_t iter_s, float rpt, float ovlp = 1.f) {
            itersize = iter_s;
            repeatn = rpt;
            overlap = ovlp;
        }
    };

    struct loop_t {
        array_in_loop_t src;
        array_in_loop_t wei;
        array_in_loop_t dst;
    };

    brg_blocking_t() : jit_brgemm_conv_conf_t() { init(); }
    brg_blocking_t(const jit_brgemm_conv_conf_t &jcp)
        : jit_brgemm_conv_conf_t(jcp) {
        init();
    }
    void init() {
        ur = 0;
        eff = 0.f;
        nb_kd = 0;
        nb_kh = 0;
        nb_kw = 0;
        sp = 0;
        sp_block = 0;
        nb_sp = 0;
        eff = 0;
    }

    int ur;
    int nb_kd, nb_kh, nb_kw;
    float eff;
    static unsigned L1;
    static unsigned L2;
    static unsigned L3;
    // These are rough estimates of the latency (relative) of access to various
    // cache levels. This is enough for an estimation of data access cost.
    // TODO: Improve memory access estimates
    static constexpr float L1_k = 1.f;
    static constexpr float L2_k = 3.f;
    static constexpr float L3_k = 15.f;
    // TODO: At the moment, we are primarily evaluating the fit of the data into
    // the L1/L2. Need to take into account the difference between the L3 and
    // memory.
    static constexpr float mem_k = 15.f;
    static constexpr int bench_iterations = 1;
    static constexpr int max_regs = 32;
    static constexpr int bcast_simd = 16;

    int sp, sp_block, nb_sp;
    static int last_ic_block_size;

    void get_from_jcp(const jit_brgemm_conv_conf_t &jcp) { *this = jcp; }
    void save_to_jcp(jit_brgemm_conv_conf_t &jcp) const { jcp = *this; }

    int estimate_brgemm_ur(int spb);
    int get_brgemm_ur(
            const primitive_attr_t *attr, const memory_desc_t &dst_md);

    float io_k(dim_t src, dim_t wei, dim_t dst, float n, float pk,
            bool is_broadcast, bool is_shared) const;

    float io_k(const loop_t loop, const array_in_loop_t arr, float pk,
            bool is_broadcast, bool is_shared) const;

    void select_ic_block();

    void update_blocks();
    bool fast_check_oc_block() const;
    float est_eff();
    void iterate_ker_block(brg_blocking_t &best_brgb, int kd_block,
            int kh_block, bool maybe_use_buffer, int max_ow_block_thr);
    void calc_blocks();

    bool fast_check_oc_block_1x1() const;
    float est_eff_1x1();
    void calc_blocks_1x1();

    // utils
    static int get_inp_size(
            int max_src_size, int dst_size, int k, int stride, int dilate) {
        auto adj_str = nstl::min(k, stride);
        const auto res = nstl::min(max_src_size,
                calculate_end_padding(0, dst_size, 0, adj_str,
                        calculate_extended_filter_size(k, dilate)));
        return res;
    }

    static float squeeze_val(float eff, float koeff) {
        if (koeff <= 0) return 1;
        if (koeff == 1) return eff;
        const auto k = 1.f / koeff;
        return (k > 1.f) ? (k - 1 + eff) / k : eff * koeff;
    }

    static int estimate_ur(int oc_block) {
        const auto est_ur = (oc_block == 64)
                ? 6
                : ((oc_block == 48) ? 9 : ((oc_block == 32) ? 14 : 28));
        return est_ur;
    }

    int inp_w(int out_w, int ker_w) const {
        return get_inp_size(iw, out_w, ker_w, stride_w, dilate_w);
    }

    int rnd_simd(int val) const { return rnd_up(val, simd_w); }

    int rnd_inp_simd(int out_w, int ker_w, int vic) const {
        const auto vsp = inp_w(out_w, ker_w);
        return ((stride_w == 1 && vic >= ic) ? rnd_up(vsp * vic, simd_w)
                                             : vsp * rnd_up(vic, simd_w));
    }

    static constexpr int MAXNLOOPS = 32;
    loop_t loop[MAXNLOOPS];
};

unsigned brg_blocking_t::L1;
unsigned brg_blocking_t::L2;
unsigned brg_blocking_t::L3;
int brg_blocking_t::last_ic_block_size;

float brg_blocking_t::io_k(dim_t src, dim_t wei, dim_t dst, float n, float pk,
        bool is_broadcast, bool is_shared) const {
    if (n < 1) return 0;
    if (n == 1) return pk;
    const auto amount = src * src_dsz + wei * wei_dsz + dst * dst_dsz
            + (use_buffer ? dst * acc_dsz : 0);
    const auto amount_L1 = is_broadcast ? src * src_dsz : amount;
    const auto k = is_broadcast
            ? ((amount_L1 < L1) ? L1_k
                                : ((amount < L2) ? L2_k
                                                 : (is_shared ? L3_k : mem_k)))
            : ((amount < L2) ? L2_k : (is_shared ? L3_k : mem_k));
    const auto cost = pk + k * (n - 1);
    return cost / n;
}

float brg_blocking_t::io_k(const loop_t loop, const array_in_loop_t arr,
        float pk, bool is_broadcast, bool is_shared) const {
    return io_k(loop.src.itersize, loop.wei.itersize, loop.dst.itersize,
            arr.repeatn * arr.overlap, pk, is_broadcast, is_shared);
}

void brg_blocking_t::select_ic_block() {
    auto nb_simd = utils::div_up(ic, simd_w);
    auto max_simd_blocks = nstl::min(5 * simd_w, nb_simd);
    const auto nb_icb_eff_threshold = 0.5f;
    const auto padded_ic = last_ic_block_size * (is_ic_padded ? 16 : 1);
    if (is_amx(isa)) {
        if (ic * kw_sets < simd_w) {
            // this is current requirement from brgemm kernel
            ic_block = rnd_up(ic, last_ic_block_size);
        } else {
            if (exec_type == exec_trans) {
                auto simd_blocks = 1;
                for (int nb_icb = max_simd_blocks; nb_icb >= 1; nb_icb--) {
                    auto nb_icb_eff = static_cast<float>(nb_simd)
                            / rnd_up(nb_simd, nb_icb);
                    if (nb_icb_eff >= nb_icb_eff_threshold) {
                        simd_blocks = nb_icb;
                        break;
                    }
                }
                ic_block = simd_blocks * simd_w;
            } else
                ic_block = simd_w;
        }
    } else {
        const auto est_ur = nstl::min(sp_block, estimate_ur(oc_block));
        const auto inp_ur = is_os_blocking ? est_ur : inp_w(est_ur, kw_block);

        if (kw_block > 1) {
            // try to fit src into L1
            const auto inp_per_ic = static_cast<unsigned int>(inp_ur) * src_dsz;
            max_simd_blocks = saturate(1, max_simd_blocks,
                    static_cast<int>(L1 / (inp_per_ic * simd_w)));
        }
        // try to fit all batch for ur into L2
        const auto wei_per_ic = static_cast<unsigned int>(kd_block) * kh_block
                * kw_block * oc_block * wei_dsz;
        const auto inp_per_ic = static_cast<unsigned int>(kd_block) * kh_block
                * inp_ur * src_dsz;
        const auto out_size
                = static_cast<unsigned int>(ur) * oc_block * dst_dsz;

        max_simd_blocks = saturate(1, max_simd_blocks,
                static_cast<int>((L2 - out_size)
                        / ((wei_per_ic + inp_per_ic) * simd_w)));

        auto simd_blocks = 1;
        for (int nb_icb = nstl::min(max_simd_blocks, nb_simd); nb_icb >= 1;
                nb_icb--) {
            auto nb_icb_eff
                    = static_cast<float>(nb_simd) / rnd_up(nb_simd, nb_icb);
            if (nb_icb_eff >= nb_icb_eff_threshold) {
                simd_blocks = nb_icb;
                break;
            }
        }

        ic_block = nstl::min(
                (exec_type == exec_trans) ? rnd_up(ic, padded_ic) : ic,
                simd_blocks * simd_w);
    }
    nb_ic = utils::div_up(ic, ic_block);
}

int brg_blocking_t::estimate_brgemm_ur(int spb) {
    // Simple simulation of brgemm_desc init
    if (sp_block <= 0) return 0;
    brgemm_t brg;
    LDA = (kh_sets > 1 ? kh_sets : 1) * (kw_sets > 1 ? kw_sets : stride_w)
            * (exec_type == exec_trans ? ic_block : ic_without_padding);
    LDB = oc_block;
    LDC = use_buffer ? oc_block : oc_without_padding;

    // Configure matrix sizes
    // for amx if ic_block != ic then we use exec_trans so K is ic_block
    const auto padded_ic = last_ic_block_size * (is_ic_padded ? 16 : 1);

    icp = rnd_up(ic, padded_ic);
    M = brgM = sp >= sp_block ? sp_block : 0;
    M_tail = brgM_tail = sp % sp_block;
    if (is_os_blocking) {
        if (!is_1x1) M_tail = brgM_tail = (oh * ow) % sp_block;
        oskip = ((ext_kw - 1) / stride_w) * stride_h + (stride_h - 1) * ow;

        brgM = sp_block + oskip * (div_up(M, ow) - 1);

        // round up brgM to help brgemm kernel use max amx_h as brgemm bd_block
        if (use_M_mask == 2) {
            int ibrgM = 0;
            const auto adj_ow = ow_block + oskip;
            while (ibrgM < brgM) {
                if (ibrgM % adj_ow < ow_block)
                    ibrgM += amx_h;
                else
                    ibrgM++;
            }
            brgM = ibrgM;
        } else
            brgM = rnd_up(brgM, amx_h);

        brgM_tail = brgM;
    }

    N = oc >= oc_block ? oc_block : 0;
    N_tail = oc % oc_block;
    K = kh_sets * kw_sets * (ic >= ic_block ? ic_block : 0);
    K_tail = kh_sets * kw_sets
            * (exec_type == exec_trans
                            ? ic_block
                            : rnd_up(ic % ic_block, last_ic_block_size));

    const auto vK = K > 0 ? K : K_tail;
    const auto vM = M > 0 ? M : M_tail;
    const auto vN = N > 0 ? N : N_tail;

    const float alpha = 1.0;
    const float beta = 0.0;
    const auto status = brgemm_desc_init(&brg, isa, brgemm_addr, src_dt, wei_dt,
            false, false, brgemm_row_major, alpha, beta, LDA, LDB, LDC, vM, vN,
            vK);
    return status == success
            ? (is_amx(isa) ? brg.bd_block * brg.bd_block2 : brg.bd_block)
            : 0;
}

int brg_blocking_t::get_brgemm_ur(
        const primitive_attr_t *attr, const memory_desc_t &dst_md) {
    // Detailed simulation of brgemm convolution init
    if (sp_block <= 0 || ic_block <= 0 || sp_block <= 0 || oc_block <= 0)
        return 0;
    brgemm_t brg;
    status_t status = success;
    int res_ur = estimate_brgemm_ur(is_os_blocking ? os_block : ow_block);

    LDD = oc_without_padding;

    const float alpha = 1.0;
    const float beta = 1.0;
    const float beta_init = 0.0;

    for (int i = 0; i < M; i++) {
        auto vM = i + 1;
        // init only needed brgemm descriptors
        if ((utils::one_of(exec_type, exec_trans, exec_vpad) || is_1x1)
                && vM != M && vM != M_tail)
            continue;
        for (int i_init = 0; i_init < 2; i_init++) {
            for (int i_N = 0; i_N < 2; i_N++) {
                for (int i_K = 0; i_K < 2; i_K++) {
                    auto vbeta = (i_init) ? beta_init : beta;
                    auto vN = (i_N) ? N_tail : N;
                    auto vK = (i_K) ? K_tail : K;
                    if (vN == 0 || vK == 0) continue;
                    brgemm_strides_t brg_strides;
                    brg_strides.stride_a
                            = ic_without_padding * (dilate_w + 1) * src_dsz;
                    //weights are padded by oc_block and last_ic_block
                    brg_strides.stride_b = rnd_up(ic, last_ic_block_size)
                            * rnd_up(oc, oc_block) * wei_dsz;
                    const auto strides_ptr = (brg_type == brgemm_strd)
                            ? &brg_strides
                            : nullptr;
                    status = brgemm_desc_init(&brg, isa, brg_type, src_dt,
                            wei_dt, false, false, brgemm_row_major, alpha,
                            vbeta, LDA, LDB, LDC, vM, vN, vK, strides_ptr);
                    if (status != success) break;

                    brgemm_attr_t brgattr;
                    brgattr.max_bs = max_batch;
                    const auto max_vpad = (exec_type == exec_vpad)
                            ? nstl::max(l_pad, r_pad)
                            : 0;
                    brgattr.max_top_vpad = max_vpad;
                    brgattr.max_bottom_vpad = max_vpad;
                    status = brgemm_desc_set_attr(&brg, brgattr);
                    if (status != success) break;

                    brg.with_sum = with_sum;
                    status = brgemm_desc_set_postops(
                            &brg, attr, &dst_md, LDD, bia_dt);
                    if (status != success) break;
                }
                if (status != success) break;
            }
            if (status != success) break;
        }
        if (status != success) break;
    }

    return status == success ? res_ur : 0;
}

void brg_blocking_t::update_blocks() {
    if (sp_block <= 0
            || utils::one_of(0, od_block, oh_block, ic_block, oc_block,
                    kd_block, kh_block, kw_block, os_block, ow_block))
        return;

    nb_od = div_up(od, od_block);
    nb_oh = div_up(oh, oh_block);
    nb_ic = div_up(ic, ic_block);
    nb_oc = div_up(oc, oc_block);
    nb_kd = div_up(kd, kd_block);
    nb_kh = div_up(kh, kh_block);
    nb_kw = div_up(kw, kw_block);
    nb_ow = div_up(ow, ow_block);
    if (is_os_blocking) {
        nb_os = div_up(os, os_block);
        sp = os;
        sp_block = os_block;
        nb_sp = nb_os;
    } else {
        sp = ow;
        sp_block = ow_block;
        nb_sp = nb_ow;
        iw_block = get_inp_size(iwp, ow_block, kw, stride_w, dilate_w);
    }
}

bool brg_blocking_t::fast_check_oc_block() const {
    // This function for reducing the number of blocking variants
    // TODO: eliminate heuristic in this function
    const auto rnd_oc = rnd_up(oc, 16);
    auto res = false;
    if (oc_block == 64) {
        res = (rnd_oc % oc_block == 0 && rnd_oc * wei_dsz < 192 * 4);
    } else if (oc_block == 48) {
        const bool big_spatial
                = id * ih * iw > 81 * stride_d * stride_h * stride_w;
        res = (rnd_oc % oc_block == 0 && rnd_oc * wei_dsz <= 384 * 4
                && big_spatial);
    } else
        res = true;

    return res;
}

float brg_blocking_t::est_eff() {
    const auto ocblock = oc_block / 16;

    const auto brgemm_microkernel_eff
            = (static_cast<float>(ocblock) * ur) / ((ur + ocblock) * max_regs);

    const auto ur_eff = static_cast<float>(sp_block) / rnd_up(sp_block, ur);
    const auto brgemm_eff = squeeze_val(ur
                    * (2.f - nstl::min(1.9f, static_cast<float>(ur) / sp_block))
                    / 64,
            0.5f);

    const auto sp_amount = nb_od * nb_oh * nb_sp;
    const auto work_amount = mb * ngroups * nb_oc * sp_amount;
    const auto sp_eff = (static_cast<float>(sp) / rnd_up(sp, sp_block));

    const auto thr_eff = static_cast<float>(work_amount)
            / utils::rnd_up(work_amount, nthr);

    const auto oc_block_eff = static_cast<float>(oc) / rnd_up(oc, oc_block);

    const auto job = div_up(work_amount, nthr);

    auto job_eff = 1.f;
    if (job < nthr) {
        std::vector<dim_t> thr_jobs(nthr);

        for (int ithr = 0; ithr < nthr; ithr++) {
            thr_jobs[ithr] = 0;
            if (ithr >= work_amount) continue;
            dim_t thr_job = 0;
            int start {0}, end {0};
            balance211(work_amount, nthr, ithr, start, end);
            int n {0}, g {0}, ocb {0}, odp {0}, ohp {0}, spb {0};
            if (loop_order == loop_ndhwgc)
                nd_iterator_init(start, n, mb, odp, od, ohp, oh, spb, nb_sp, g,
                        ngroups, ocb, nb_oc);
            else if (loop_order == loop_ngcdhw)
                nd_iterator_init(start, n, mb, g, ngroups, ocb, nb_oc, odp, od,
                        ohp, oh, spb, nb_sp);

            for (auto work = start; work < end; work++) {
                const int ocp = ocb * oc_block;
                const auto oc_sz = nstl::min(oc - ocp, oc_block);
                int sp_sz = 0;
                const int spp = spb * sp_block;
                sp_sz = nstl::min(sp - spp, sp_block);
                thr_job += sp_sz * oc_sz;

                if (loop_order == loop_ndhwgc)
                    nd_iterator_step(n, mb, odp, od, ohp, oh, spb, nb_sp, g,
                            ngroups, ocb, nb_oc);
                else if (loop_order == loop_ngcdhw)
                    nd_iterator_step(n, mb, g, ngroups, ocb, nb_oc, odp, od,
                            ohp, oh, spb, nb_sp);
            }
            thr_jobs[ithr] = thr_job;
        }

        dim_t max_job = 0;
        dim_t sum_job = 0;
        for (int ithr = 0; ithr < nthr; ithr++) {
            if (thr_jobs[ithr] > max_job) max_job = thr_jobs[ithr];
            sum_job += thr_jobs[ithr];
        }
        job_eff = max_job == 0 ? 1
                               : static_cast<float>(sum_job) / (max_job * nthr);

    } else {
        job_eff = thr_eff;
    }

    const auto ic_blocking_size = ic_block * nb_ic_blocking;
    const auto oc_blocking_size = oc_block * ic_blocking_size;

    int l = -1;

    // -- brgemm kernel: loop by simd_w  --
    l++;
    const auto inp_ur = inp_w(ur, kw_block);
    loop[l].src.set(inp_ur * simd_w, 1, bcast_simd);
    loop[l].dst.set(0, 1);
    loop[l].wei.set(oc_block, 1);

    // -- brgemm kernel: loop by kw in kw_block  --
    l++;
    auto src_is = rnd_inp_simd(ur, kw_block, ic_blocking_size);
    loop[l].src.set(src_is, 1, kw_block);
    loop[l].dst.set(0, 1);
    loop[l].wei.set(oc_blocking_size, 1);

    // -- brgemm kernel: loop by batch (grouped by kw_block) in ur  --
    l++;
    loop[l].src.set(src_is, 1);
    loop[l].dst.set(0, 1);
    auto wei_is = kw_block * oc_blocking_size;
    loop[l].wei.set(wei_is, 1);
    // -- brgemm kernel: loop by ur in sp_block --
    l++;
    const auto nb_ur = div_up(sp_block, ur);
    loop[l].src.set(kd_block * kh_block * src_is, 1);
    loop[l].dst.set(ur * oc_block, 1);
    wei_is = kd_block * kh_block * kw_block * oc_blocking_size;
    loop[l].wei.set(wei_is, nb_ur);

    // -- harness: loop by k_blocks in ks --
    l++;
    loop[l].src.set(kd_block * kh_block
                    * rnd_inp_simd(sp_block, kw_block, ic_blocking_size),
            1);
    loop[l].dst.set(sp_block * oc_block, nb_kd * nb_kh * nb_kw);
    loop[l].wei.set(wei_is, 1);

    // -- brgemm kernel: loop by ic_chunks --
    l++;
    const auto ic_chunks = div_up(nb_ic, nb_ic_blocking);
    loop[l].src.set(kd * kh * rnd_inp_simd(sp_block, kw, ic_blocking_size), 1);
    loop[l].dst.set(sp_block * oc_block, ic_chunks);
    wei_is = kd * kh * kw * oc_blocking_size;
    loop[l].wei.set(wei_is, 1);

    const auto dim_oc = (loop_order == loop_ndhwgc) ? 1 : sp_amount;
    const auto nb_oc_thr = nstl::min(nb_oc, div_up(job, dim_oc));
    const auto oc_thr = nstl::min(oc, nb_oc_thr * oc_block);
    const auto nsimd_oc_thr = div_up(oc_thr, simd_w);

    const auto dim_sp = (loop_order == loop_ndhwgc) ? ngroups * nb_oc : 1;
    const auto nb_sp_thr = nstl::min(nb_sp, div_up(job, dim_sp));
    const auto sp_thr = nstl::min(sp, nb_sp_thr * sp_block);

    const auto dim_oh = nb_sp * dim_sp;
    const auto nb_oh_thr = nstl::min(nb_oh, div_up(job, dim_oh));
    const auto oh_thr = nstl::min(oh, nb_oh_thr * oh_block);

    const auto dim_od = nb_oh * dim_oh;
    const auto nb_od_thr = nstl::min(nb_od, div_up(job, dim_od));
    const auto od_thr = nstl::min(od, nb_od_thr * od_block);

    src_is = kd * kh * rnd_inp_simd(sp_block, kw, ic);

    auto wei_op = kd * kh * kw * ocblock * ic;
    if (loop_order == loop_ndhwgc) {
        // -- harness: loop by oc_block --
        l++;
        loop[l].src.set(src_is, nb_oc_thr);
        loop[l].dst.set(sp_block * oc_block, 1);
        wei_is = kd * kh * kw * oc_block * ic;
        wei_op = kd * kh * kw * nsimd_oc_thr * ic;
        loop[l].wei.set(wei_is, 1);
    }

    // -- harness: loop by sp_blocks --
    l++;
    loop[l].src.set(src_is, 1);
    const auto rnd_oc_for_sp
            = simd_w * ((loop_order == loop_ndhwgc) ? nsimd_oc_thr : ocblock);
    loop[l].dst.set(sp_block * rnd_oc_for_sp, 1);
    loop[l].wei.set(wei_op * simd_w, nb_sp_thr);
    // oh_block almost all is 1. TODO: manage oh_block != 1
    // -- harness: loop by oh_blocks --
    l++;
    src_is = kd * kh * rnd_inp_simd(sp_thr, kw, ic);
    loop[l].src.set(oh_block * src_is, 1);
    loop[l].dst.set(sp_thr * rnd_oc_for_sp, 1);
    loop[l].wei.set(wei_op * simd_w, nb_oh_thr);
    // od_block almost all is 1. TODO: manage oh_block != 1
    // -- harness: loop by od_blocks --
    l++;
    loop[l].src.set(od_block * oh_thr * src_is, 1);
    loop[l].dst.set(oh_thr * sp_thr * rnd_oc_for_sp, 1);
    loop[l].wei.set(wei_op * simd_w, nb_od_thr);

    if (loop_order != loop_ndhwgc) {
        // -- harness: loop by oc_block --
        l++;
        loop[l].src.set(od_thr * oh_thr * src_is, nb_oc_thr);
        loop[l].dst.set(oc_block * od_thr * oh_thr * sp_thr, 1);
        loop[l].wei.set(kd * kh * kw * oc_block * ic, 1);
    }

    // -- harness: loop by mb --
    l++;
    const auto mb_thr = nstl::min(mb, div_up(job, sp_amount * ngroups * nb_oc));
    loop[l].src.set(od_thr * oh_thr * src_is, 1);
    loop[l].dst.set(od_thr * oh_thr * sp_thr * nsimd_oc_thr * simd_w, 1);
    loop[l].wei.set(kd * kh * kw * nsimd_oc_thr * simd_w * ic, mb_thr);

    const auto src_op = static_cast<dim_t>(mb_thr) * od_thr
            * (is_os_blocking ? 1 : oh_thr) * sp_thr * kd * kh * kw * ic;
    const auto dst_op = static_cast<dim_t>(mb_thr) * od_thr
            * (is_os_blocking ? 1 : oh_thr) * sp_thr * nsimd_oc_thr;
    wei_op = kd * kh * kw * nsimd_oc_thr * ic;

    // for "real" application set bench_iterations to 1
    const auto iterations = bench_iterations;
    l++;
    loop[l].src.set(src_op, iterations);
    loop[l].dst.set(dst_op * simd_w, iterations);
    loop[l].wei.set(wei_op * simd_w, iterations);

    auto src_mem_k = mem_k;
    auto dst_mem_k = mem_k;
    auto wei_mem_k = mem_k;
    float src_rp = 1;
    float dst_rp = 1;
    float wei_rp = 1;

    for (auto il = l; il >= 0; il--) {
        src_mem_k = io_k(loop[il], loop[il].src, src_mem_k, true,
                loop_order == loop_ndhwgc ? false : true);
        dst_mem_k = io_k(loop[il], loop[il].dst, dst_mem_k, false, false);
        wei_mem_k = io_k(loop[il], loop[il].wei, wei_mem_k, false,
                loop_order == loop_ndhwgc ? true : false);
        src_rp *= loop[il].src.repeatn;
        dst_rp *= loop[il].dst.repeatn;
        wei_rp *= loop[il].wei.repeatn;
    }
    const auto src_ops = (src_op * src_rp) / iterations;
    const auto dst_ops = (dst_op * dst_rp) / iterations;
    const auto wei_ops = (wei_op * wei_rp) / iterations;

    const auto src_cost = src_mem_k * src_ops;
    const auto dst_cost = dst_mem_k * dst_ops;
    const auto wei_cost = wei_mem_k * wei_ops;
    const auto call_kernel_cost
            = 1000.f * job * ic_chunks * nb_kd * nb_kh * nb_kw;

    const auto cache_eff = (static_cast<dim_t>(mb) * od * oh * sp * ic * oc)
            / (nthr * (src_cost + dst_cost + wei_cost + call_kernel_cost));
    const auto res_eff = oc_block_eff * brgemm_microkernel_eff * sp_eff
            * job_eff * ur_eff * cache_eff * brgemm_eff;
    return res_eff;
}

void brg_blocking_t::iterate_ker_block(brg_blocking_t &best_brgb, int kd_block_,
        int kh_block_, bool maybe_use_buffer, int max_ow_block_thr) {

    unsigned est_k_amount = ic * oc_block * wei_dsz;

    kd_block = kd_block_;
    kh_block = kh_block_;
    if (one_of(exec_type, exec_vpad, exec_trans)) {
        kw_block = kw;
        kd_block_pad = kd_block;
        kh_block_pad = kh_block;
        kw_block_pad = kw_block;
    } else {
        kw_block = (est_k_amount * kw < L2) ? kw : 1;
        kd_block_pad = kh_block >= kd ? kd : 1;
        kh_block_pad = kw_block >= kh ? kh : 1;
        kw_block_pad = kw;
    }

    if (exec_type == exec_vpad) {
        od_block = 1;
        oh_block = 1;
    } else if (exec_type == exec_trans) {
        const auto w_block_size
                = 2 * src_dsz * ic * iwp + dst_dsz * ow * oc_block;
        const auto other_size = wei_dsz * kd * kh * kw * ic * oc_block
                + acc_dsz * 2 * amx_h * oc_block;
        const auto L2_available = nstl::min(static_cast<size_t>(div_up(L2, 2)),
                other_size > L2 ? 0 : L2 - other_size);
        if (idp * ihp * w_block_size > L2_available) {
            od_block = utils::saturate(
                    1, od, int(L2_available / (ihp * w_block_size)));
            if (od_block == 1)
                oh_block = utils::saturate(
                        1, oh, int(L2_available / (w_block_size)));
            else
                oh_block = oh;
        } else {
            od_block = 1;
            oh_block = oh;
        }
        if (is_amx(isa)) {
            // try to fit into L1
            bool L1_fit_res = false;
            auto cur_od_block = od_block;
            auto cur_oh_block = oh_block;
            const auto src_w_block_size
                    = src_dsz * ic * iwp + dst_dsz * ow * oc_block;
            if (src_w_block_size < L1) {
                cur_od_block = utils::saturate(
                        1, od, int(L1 / (ihp * src_w_block_size)));
                if (cur_od_block == 1)
                    cur_oh_block = utils::saturate(
                            1, oh, int(L1 / (src_w_block_size)));
            }
            for (; cur_od_block > 1; cur_od_block--) {
                const auto sp_size = cur_od_block * cur_oh_block * iwp;
                if ((static_cast<float>(od) / rnd_up(od, cur_od_block)) > 0.9f
                        && static_cast<float>(sp_size) / rnd_up(sp, amx_h)
                                > 0.8f) {
                    L1_fit_res = true;
                    break;
                }
            }
            if (cur_od_block == 1) {
                for (; cur_oh_block > 1; cur_oh_block--) {
                    const auto sp_size = cur_oh_block * iwp;
                    if ((static_cast<float>(oh) / rnd_up(oh, cur_oh_block))
                                    > 0.9f
                            && sp_size > 128) {
                        L1_fit_res = true;
                        break;
                    }
                }
            }
            if (L1_fit_res) {
                od_block = cur_od_block;
                oh_block = cur_oh_block;
            }
        }

        // limit oh_block to have good threading
        auto thr_od_block = div_up(od, div_up(nthr, mb * div_up(oc, oc_block)));
        auto thr_oh_block = div_up(oh,
                div_up(nthr,
                        mb * div_up(oc, oc_block) * div_up(od, thr_od_block)));
        od_block = nstl::min(od_block, thr_od_block);
        oh_block = nstl::min(oh_block, thr_oh_block);
    } else {
        od_block = 1;
        oh_block = 1;
    }

    // --- Select ow_block ----
    const auto max_ow_block_L2 = ow;
    auto start_ow_block = nstl::min(max_ow_block_thr, max_ow_block_L2);

    sp = ow;
    const auto start_sp_block = is_os_blocking ? ow : start_ow_block;
    auto prev_spb = 0;
    for (auto ns = 1; ns <= sp; ns++) {
        const auto spb = div_up(sp, ns);
        if (spb == prev_spb || spb > start_sp_block) continue;
        if (is_os_blocking && spb != ow) continue;
        prev_spb = spb;
        ow_block = spb;
        sp_block = ow_block;

        select_ic_block();

        use_buffer = maybe_use_buffer
                && (ic_block * nb_ic_blocking < ic || kd_block != kd
                        || kh_block != kh || kw_block != kw
                        || kd_block_pad != kd || kh_block_pad != kh
                        || kw_block_pad != kw);
        if (exec_type == exec_base)
            use_buffer = use_buffer || (maybe_use_buffer && iwp != iw);
        if (is_amx(isa)) use_buffer = use_buffer || (use_M_mask > 0);

        ur = estimate_brgemm_ur(ow_block);
        if (ur == 0) continue;
        os_block = sp_block = ow_block;
        update_blocks();

        eff = est_eff();

        if (eff > best_brgb.eff || best_brgb.eff == 0) best_brgb = *this;
    }
}

void brg_blocking_t::calc_blocks() {
    sp = ow;

    nb_ic_blocking = 1;
    // --- Select kernel blocking ---
    // if dst_dt != acc_dt and we need to store intermediate
    // results then we need the out buffer
    const auto maybe_use_buffer = (dst_dt != acc_dt || with_sum);

    std::vector<int> kd_blocks(1), kh_blocks(1);
    kd_blocks[0] = kd;
    kh_blocks[0] = kh;
    if (kd != 1) {
        kd_blocks.resize(2);
        kd_blocks[1] = 1;
    }
    if (kh != 1) {
        kh_blocks.resize(2);
        kh_blocks[1] = 1;
    }

    const auto thr_eff_threshold = 0.9f;
    const auto max_ow_block_thr = utils::saturate(1, ow,
            static_cast<int>(div_up(
                    mb * ngroups * nb_oc * os, thr_eff_threshold * nthr)));

    ow_block = os_block = sp_block = -1;
    brg_blocking_t best_brgb = *this;
    for (const auto &kd_block : kd_blocks) {
        for (const auto &kh_block : kh_blocks) {
            iterate_ker_block(best_brgb, kd_block, kh_block, maybe_use_buffer,
                    max_ow_block_thr);
        }
    }
    *this = best_brgb;
    if (is_os_blocking) {
        ow_block = ow;
        os_block = ow * oh_block;
        sp_block = os_block;
        ow_tail = 0;
    } else {
        ow_block = os_block = sp_block;
        ow_tail = ow % ow_block;
    }
    update_blocks();
}

bool brg_blocking_t::fast_check_oc_block_1x1() const {
    // This function for reducing the number of blocking variants
    // TODO: eliminate heuristic in this function
    const auto rnd_oc = rnd_up(oc, 16);
    auto res = false;
    if (oc_block == 64) {
        const auto big_spatial
                = od * oh * ow >= 64 * stride_d * stride_h * stride_w;
        res = (rnd_oc % oc_block == 0 && big_spatial);
    } else if (oc_block == 48) {
        const auto oc_block_eff = static_cast<float>(oc) / rnd_up(oc, oc_block);
        res = (oc_block_eff >= 0.95);
    } else
        res = true;

    return res;
}

float brg_blocking_t::est_eff_1x1() {
    const auto ocblock = oc_block / 16;

    const auto brgemm_microkernel_eff
            = (static_cast<float>(ocblock) * ur) / ((ur + ocblock) * max_regs);
    const auto ur_eff = static_cast<float>(sp_block) / rnd_up(sp_block, ur);
    const auto brgemm_eff = squeeze_val(ur
                    * (2.f - nstl::min(1.9f, static_cast<float>(ur) / sp_block))
                    / 64,
            0.5f);

    const auto sp_amount = is_os_blocking ? div_up(nb_os, nb_os_blocking)
                                          : nb_od * nb_oh * nb_sp;
    const auto work_amount = mb * ngroups * nb_oc * sp_amount;

    const auto sp_eff = static_cast<float>(sp) / rnd_up(sp, sp_block);
    const auto thr_eff = static_cast<float>(work_amount)
            / utils::rnd_up(work_amount, nthr);
    const auto oc_block_eff = static_cast<float>(oc) / rnd_up(oc, oc_block);

    const auto job = div_up(work_amount, nthr);

    const auto dim_oc = (loop_order == loop_ndhwgc) ? 1 : sp_amount;
    const auto nb_oc_thr = nstl::min(nb_oc, div_up(job, dim_oc));
    const auto oc_thr = nstl::min(oc, nb_oc_thr * oc_block);
    const auto nsimd_oc_thr = div_up(oc_thr, simd_w);

    const auto dim_sp = (loop_order == loop_ndhwgc) ? ngroups * nb_oc : 1;
    const auto nb_sp_thr = nstl::min(nb_sp, div_up(job, dim_sp));
    const auto sp_thr = nstl::min(sp, nb_sp_thr * sp_block);

    const auto dim_oh = nb_sp * dim_sp;
    const auto nb_oh_thr = nstl::min(nb_oh, div_up(job, dim_oh));
    const auto oh_thr
            = is_os_blocking ? 1 : nstl::min(oh, nb_oh_thr * oh_block);

    const auto dim_od = nb_oh * dim_oh;
    const auto nb_od_thr = nstl::min(nb_od, div_up(job, dim_od));
    const auto od_thr
            = is_os_blocking ? 1 : nstl::min(od, nb_od_thr * od_block);

    auto job_eff = 1.f;
    if (job < nthr) {
        std::vector<dim_t> thr_jobs(nthr);
        for (int ithr = 0; ithr < nthr; ithr++) {
            thr_jobs[ithr] = 0;
            if (ithr >= work_amount) continue;
            dim_t thr_job = 0;
            int start {0}, end {0};
            balance211(work_amount, nthr, ithr, start, end);
            int n {0}, g {0}, ocb {0}, oss {0}, odp {0}, ohp {0}, spb {0};
            if (loop_order == loop_ndhwgc) {
                if (is_os_blocking)
                    nd_iterator_init(start, n, mb, oss, sp_amount, g, ngroups,
                            ocb, nb_oc);
                else
                    nd_iterator_init(start, n, mb, odp, od, ohp, oh, spb, nb_sp,
                            g, ngroups, ocb, nb_oc);
            } else if (loop_order == loop_ngcdhw) {
                if (is_os_blocking)
                    nd_iterator_init(start, n, mb, g, ngroups, ocb, nb_oc, oss,
                            sp_amount);
                else
                    nd_iterator_init(start, n, mb, g, ngroups, ocb, nb_oc, odp,
                            od, ohp, oh, spb, nb_sp);
            }

            for (auto work = start; work < end; work++) {
                const int ocp = ocb * oc_block;
                const auto oc_sz = nstl::min(oc - ocp, oc_block);
                int sp_sz = 0;
                if (is_os_blocking) {
                    const auto osb_start = oss * nb_os_blocking;
                    const auto osb_range
                            = nstl::min(nb_os - osb_start, nb_os_blocking);
                    for (int osb = 0; osb < osb_range; osb++) {
                        const int osp = (osb_start + osb) * sp_block;
                        sp_sz = nstl::min(os - osp, sp_block);
                    }
                } else {
                    const int spp = spb * sp_block;
                    sp_sz = nstl::min(sp - spp, sp_block);
                }
                thr_job += sp_sz * oc_sz;

                if (loop_order == loop_ndhwgc) {
                    if (is_os_blocking)
                        nd_iterator_step(
                                n, mb, oss, sp_amount, g, ngroups, ocb, nb_oc);
                    else
                        nd_iterator_step(n, mb, odp, od, ohp, oh, spb, nb_sp, g,
                                ngroups, ocb, nb_oc);
                } else if (loop_order == loop_ngcdhw) {
                    if (is_os_blocking)
                        nd_iterator_step(
                                n, mb, g, ngroups, ocb, nb_oc, oss, sp_amount);
                    else
                        nd_iterator_step(n, mb, g, ngroups, ocb, nb_oc, odp, od,
                                ohp, oh, spb, nb_sp);
                }
            }
            thr_jobs[ithr] = thr_job;
        }

        dim_t max_job = 0;
        dim_t sum_job = 0;
        for (int ithr = 0; ithr < nthr; ithr++) {
            if (thr_jobs[ithr] > max_job) max_job = thr_jobs[ithr];
            sum_job += thr_jobs[ithr];
        }

        job_eff = max_job == 0 ? 1
                               : static_cast<float>(sum_job) / (max_job * nthr);
    } else {
        job_eff = thr_eff;
    }

    const auto ic_blocking_size = ic_block * nb_ic_blocking;
    const auto oc_blocking_size = oc_block * ic_blocking_size;

    int l = -1;
    // -- brgemm kernel: loop by simd_w  --
    l++;
    loop[l].src.set(ur * simd_w, 1, bcast_simd);
    loop[l].dst.set(0, 1);
    loop[l].wei.set(oc_block, 1);

    // -- brgemm kernel: loop by ur in sp_block --
    l++;
    const auto nb_ur = div_up(sp_block, ur);
    loop[l].src.set(ur * rnd_simd(ic_blocking_size), 1);
    loop[l].dst.set(ur * oc_block, 1);
    loop[l].wei.set(oc_blocking_size, nb_ur);
    // -- brgemm kernel: loop by ic_chunks --
    l++;
    const auto ic_chunks = div_up(nb_ic, nb_ic_blocking);
    loop[l].src.set(sp_block * ic_blocking_size, 1);
    loop[l].dst.set(sp_block * oc_block, ic_chunks);
    auto wei_is = oc_blocking_size;
    auto wei_op = ocblock * ic;
    loop[l].wei.set(wei_is, 1);

    if (loop_order == loop_ndhwgc) {
        // -- harness: loop by oc_block --
        l++;
        loop[l].src.set(sp_block * rnd_simd(ic), nb_oc_thr);
        loop[l].dst.set(sp_block * oc_block, 1);
        wei_is = oc_block * ic;
        wei_op = nsimd_oc_thr * ic;
        loop[l].wei.set(wei_is, 1);
    }

    const auto rnd_oc_for_sp
            = simd_w * ((loop_order == loop_ndhwgc) ? nsimd_oc_thr : ocblock);
    if (is_os_blocking) {
        // -- harness: loop by os_blocks --
        l++;
        loop[l].src.set(sp_block * ic_blocking_size, 1);
        loop[l].dst.set(sp_block * rnd_oc_for_sp, 1);
        loop[l].wei.set(wei_op * simd_w, nb_sp_thr);
    } else {
        // -- harness: loop by sp_blocks --
        l++;
        loop[l].src.set(sp_block * ic_blocking_size, 1);
        loop[l].dst.set(sp_block * rnd_oc_for_sp, 1);
        loop[l].wei.set(wei_op * simd_w, nb_sp_thr);
        // -- harness: loop by oh_blocks --
        l++;
        loop[l].src.set(oh_block * sp_thr * rnd_simd(ic_blocking_size), 1);
        loop[l].dst.set(oh_block * sp_thr * rnd_oc_for_sp, 1);
        loop[l].wei.set(wei_op * simd_w, nb_oh_thr);
        // -- harness: loop by od_blocks --
        l++;
        loop[l].src.set(
                od_block * oh_thr * sp_thr * rnd_simd(ic_blocking_size), 1);
        loop[l].dst.set(od_block * oh_thr * sp_thr * rnd_oc_for_sp, 1);
        loop[l].wei.set(wei_op * simd_w, nb_od_thr);
    }

    if (loop_order != loop_ndhwgc) {
        // -- harness: loop by oc_block --
        l++;
        loop[l].src.set(od_thr * oh_thr * rnd_simd(sp_thr * ic_blocking_size),
                nb_oc_thr);
        loop[l].dst.set(oc_block * od_thr * oh_thr * sp_thr, 1);
        loop[l].wei.set(oc_block * ic, 1);
    }

    // -- harness: loop by mb --
    l++;
    const auto mb_thr = nstl::min(mb, div_up(job, sp_amount * ngroups * nb_oc));
    loop[l].src.set(od_thr * oh_thr * sp_thr * rnd_simd(ic_blocking_size), 1);
    loop[l].dst.set(nsimd_oc_thr * simd_w * od_thr * oh_thr * sp_thr, 1);
    loop[l].wei.set(nsimd_oc_thr * ic * simd_w, mb_thr);

    const auto src_op = static_cast<dim_t>(mb_thr) * od_thr
            * (is_os_blocking ? 1 : oh_thr) * sp_thr * ic_blocking_size;
    const auto dst_op = static_cast<dim_t>(mb_thr) * nsimd_oc_thr * od_thr
            * (is_os_blocking ? 1 : oh_thr) * sp_thr;
    wei_op = nsimd_oc_thr * ic;

    // for "real" application set bench_iterations to 1
    const auto iterations = bench_iterations;
    l++;
    loop[l].src.set(src_op, iterations);
    loop[l].dst.set(dst_op * simd_w, iterations);
    loop[l].wei.set(wei_op * simd_w, iterations);

    auto src_mem_k = mem_k;
    auto dst_mem_k = mem_k;
    auto wei_mem_k = mem_k;
    float src_rp = 1;
    float dst_rp = 1;
    float wei_rp = 1;

    for (auto il = l; il >= 0; il--) {
        src_mem_k = io_k(loop[il], loop[il].src, src_mem_k, true, false);
        dst_mem_k = io_k(loop[il], loop[il].dst, dst_mem_k, false, false);
        wei_mem_k = io_k(loop[il], loop[il].wei, wei_mem_k, false, true);
        src_rp *= loop[il].src.repeatn;
        dst_rp *= loop[il].dst.repeatn;
        wei_rp *= loop[il].wei.repeatn;
    }
    const auto src_ops = (src_op * src_rp) / iterations;
    const auto dst_ops = (dst_op * dst_rp) / iterations;
    const auto wei_ops = (wei_op * wei_rp) / iterations;

    const auto src_cost = src_mem_k * src_ops;
    const auto dst_cost = dst_mem_k * dst_ops;
    const auto wei_cost = wei_mem_k * wei_ops;
    const auto call_kernel_cost = 1000.f * job * ic_chunks;

    const auto up_sp_size = is_os_blocking ? 1 : od * oh;

    const auto cache_eff = (static_cast<dim_t>(mb) * up_sp_size * sp * ic * oc)
            / (nthr * (src_cost + dst_cost + wei_cost + call_kernel_cost));

    const auto res_eff = oc_block_eff * brgemm_microkernel_eff * sp_eff
            * job_eff * ur_eff * cache_eff * brgemm_eff;
    return res_eff;
}

void brg_blocking_t::calc_blocks_1x1() {
    const bool is_os_blocking_ok
            = utils::everyone_is(1, stride_d, stride_h) && iw % stride_w == 0;
    if (is_os_blocking_ok) {
        sp = os;
        is_os_blocking = true;
    } else {
        sp = ow;
        is_os_blocking = false;
    }

    od_block = 1;
    oh_block = 1;
    kd_block = kh_block = kw_block = 1;
    kd_block_pad = kh_block_pad = kw_block_pad = 1;
    nb_ic_blocking = 1;

    const auto thr_eff_threshold = 0.9f;

    const auto max_sp_block_L2 = os;
    // TODO: nb_os_blocking always is 1 for now. Update this code
    nb_os_blocking = 1;
    int start_sp_block = 0;

    if (stride_d == 1 && stride_h == 1) {
        ow_block = 0;

        const auto max_os_block_thr = nstl::max(div_up(2048, oc_block),
                static_cast<int>(div_up(mb * ngroups * os, nthr)));
        const auto max_os_block_L2 = max_sp_block_L2;

        auto max_os_block_aliasing = 1000000 / nthr;
        if ((oc_without_padding * os * dst_dsz) % 4096 == 0) {
            max_os_block_aliasing /= 1;
            for (auto cur_oc = oc_without_padding;
                    max_os_block_aliasing * dst_dsz > 400 && cur_oc % 2 == 0
                    && cur_oc * os * dst_dsz >= 4096;
                    cur_oc /= 2) {
                max_os_block_aliasing /= 2;
            }
            max_os_block_aliasing += max_os_block_aliasing % 2 ? 0 : 1;
        }
        max_os_block_aliasing
                = nstl::min(div_up(1001, dst_dsz), max_os_block_aliasing);

        start_sp_block = utils::saturate(1, os,
                nstl::min(nstl::min(max_os_block_thr, max_os_block_L2),
                        max_os_block_aliasing));

    } else {
        os_block = 0;

        const auto max_ow_block_thr = utils::saturate(1, ow,
                static_cast<int>(div_up(
                        mb * ngroups * nb_oc * os, thr_eff_threshold * nthr)));
        const auto max_ow_block_L2 = max_sp_block_L2;

        start_sp_block = utils::saturate(
                1, ow, nstl::min(max_ow_block_thr, max_ow_block_L2));
    }
    os_block = ow_block = sp_block = -1;
    brg_blocking_t best_brgb = *this;

    auto prev_spb = 0;
    for (auto ns = 1; ns <= sp; ns++) {
        const auto spb = div_up(sp, ns);
        if (spb == prev_spb || spb > start_sp_block) continue;
        prev_spb = spb;
        os_block = ow_block = sp_block = spb;
        select_ic_block();
        ur = estimate_brgemm_ur(spb);
        if (ur == 0) continue;
        update_blocks();

        use_buffer = (dst_dt != acc_dt || with_sum)
                && (ic_block * nb_ic_blocking < ic);

        eff = est_eff_1x1();
        if (eff > best_brgb.eff || best_brgb.eff == 0) best_brgb = *this;
    }
    *this = best_brgb;
    os_block = ow_block = sp_block;
    update_blocks();
}

status_t init_jcp(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads) {
    using namespace prop_kind;

    brg_blocking_t::L1 = platform::get_per_core_cache_size(1);
    brg_blocking_t::L2 = platform::get_per_core_cache_size(2);
    brg_blocking_t::L3 = platform::get_per_core_cache_size(2);

    if (!mayiuse(avx512_core)) return status::unimplemented;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();
    jcp.isa = isa;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc_without_padding = dst_d.dims()[1];
    jcp.oc = jcp.oc_without_padding / jcp.ngroups;
    jcp.ic_without_padding = src_d.dims()[1];
    jcp.ic = jcp.ic_without_padding / jcp.ngroups;
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

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    jcp.os = jcp.od * jcp.oh * jcp.ow;

    jcp.ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    jcp.ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);

    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, jcp.ext_kd);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, jcp.ext_kh);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, jcp.ext_kw);

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    jcp.src_dt = cd.src_desc.data_type;
    jcp.dst_dt = cd.dst_desc.data_type;
    jcp.wei_dt = cd.weights_desc.data_type;
    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;

    brg_blocking_t::last_ic_block_size
            = (jcp.wei_dt == f32) ? 1 : ((jcp.wei_dt == bf16) ? 2 : 4);

    // TODO: optimize depthwise convolutions (for now direct approach is faster)
    const bool is_depthwise
            = with_groups && jcp.ngroups > 1 && everyone_is(1, jcp.ic, jcp.oc);
    if (is_depthwise) return status::unimplemented;

    // TODO: optimize grouped convolutions with small ic
    const bool is_grouped_small_ic
            = with_groups && jcp.ngroups > 1 && jcp.ic <= 16;
    if (is_grouped_small_ic) return status::unimplemented;

    // TODO: support s8 in non-amx brgemm convolutions
    if (!IMPLICATION(jcp.src_dt == s8, is_amx(jcp.isa)))
        return status::unimplemented;

    if (!IMPLICATION(jcp.wei_dt == s8, mayiuse(avx512_core_vnni)))
        return status::unimplemented;
    if (!IMPLICATION(jcp.wei_dt == bf16, mayiuse(avx512_core_bf16)))
        return status::unimplemented;

    if (one_of(jcp.src_dt, u8, s8)) {
        jcp.acc_dt = s32;
    } else if (one_of(jcp.src_dt, f32, bf16)) {
        jcp.acc_dt = f32;
    } else
        return status::unimplemented;

    jcp.src_dsz = types::data_type_size(jcp.src_dt);
    jcp.wei_dsz = types::data_type_size(jcp.wei_dt);
    jcp.dst_dsz = types::data_type_size(jcp.dst_dt);
    jcp.acc_dsz = types::data_type_size(jcp.acc_dt);
    jcp.bia_dsz = jcp.with_bias ? types::data_type_size(jcp.bia_dt) : 0;

    if (!post_ops_ok(jcp, attr, dst_d)) return status::unimplemented;

    jcp.simd_w = cpu_isa_traits<avx512_common>::vlen / jcp.src_dsz;
    jcp.amx_h = 16;
    jcp.amx_w = 64 / jcp.src_dsz;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;

    const int binary_ind = p.find(primitive_kind::binary);
    jcp.with_binary = binary_ind != -1;

    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, x));
    }

    jcp.nthr = nthreads;
    jcp.kh_sets = 1;
    jcp.kw_sets = 1;
    jcp.copy_block_only = false;
    jcp.amx_tile_load_xx = false;
    jcp.use_M_mask = 0;
    jcp.is_os_blocking = false;
    jcp.oskip = 0;
    jcp.use_uker = false;
    jcp.use_interleave_stores = false;
    jcp.brgemm_bd_loop_innermost = false;
    jcp.is_1x1 = true && jcp.f_pad <= 0 && jcp.back_pad <= 0 && jcp.t_pad <= 0
            && jcp.b_pad <= 0 && jcp.l_pad <= 0 && jcp.r_pad <= 0 && jcp.kd == 1
            && jcp.kh == 1 && jcp.kw == 1;

    // fast check data layout before spending time for blocking selection
    format_tag_t src_tag = pick(jcp.ndims - 3, nwc, nhwc, ndhwc);
    const bool any_eligible = (jcp.prop_kind == prop_kind::forward_inference
            || jcp.wei_dt == data_type::s8 || is_amx(jcp.isa));
    CHECK(init_tag(jcp.src_tag, src_md, src_d, src_tag, any_eligible));

    const auto ic_padded_block = 16 * brg_blocking_t::last_ic_block_size;
    jcp.is_ic_padded = !jcp.is_1x1 && one_of(jcp.wei_dt, bf16, s8)
            && jcp.ic * jcp.kw_sets > ic_padded_block && is_amx(isa);

    return status::success;
}

status_t init_conf(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads) {

    using namespace prop_kind;
    if (!mayiuse(isa)) return status::unimplemented;

    CHECK(init_jcp(
            jcp, isa, cd, src_md, weights_md, dst_md, bias_md, attr, nthreads));

    if (jcp.is_1x1) return status::unimplemented;
    // TODO: check these restrictions
    if (is_amx(isa)) {
        // disabled for first convolutions excepting 3d
        const bool is_3d = jcp.ndims == 5;
        if (jcp.ic <= 4 && !is_3d) return status::unimplemented;

        if (jcp.f_pad >= jcp.kd || jcp.t_pad >= jcp.kh || jcp.r_pad >= jcp.kw)
            return status::unimplemented;
        if (jcp.dilate_d > 0 || jcp.dilate_h > 0 || jcp.dilate_w > 0)
            return status::unimplemented;
    }

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    jcp.idp = jcp.id + jcp.f_pad + jcp.back_pad;
    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;

    using namespace data_type;
    // ======================= blocking =================================

    auto bcast_amount
            = static_cast<size_t>(jcp.id) * jcp.ih * jcp.iw * jcp.src_dsz;
    auto wei_amount = static_cast<size_t>(jcp.oc) * jcp.kd * jcp.kh * jcp.kw
            * jcp.wei_dsz;

    jcp.loop_order = (bcast_amount < wei_amount) ? loop_ngcdhw : loop_ndhwgc;

    const int min_oc_block = 16;

    int selected_ur = 0;
    MAYBE_UNUSED(selected_ur);

    auto try_exec_type = [&]() {
        brg_blocking_t best_brgb = zero<decltype(best_brgb)>();
        best_brgb.oc_block = min_oc_block;
        brg_blocking_t cur_brgb = zero<decltype(best_brgb)>();
        cur_brgb.get_from_jcp(jcp);
        auto start_ocb = (is_amx(isa) && jcp.is_os_blocking) ? 2 : 4;
        if (jcp.wei_plain)
            start_ocb = nstl::min(jcp.ic > 128 ? (jcp.ic > 256 ? 8 : 16) : 32,
                    div_up(jcp.oc, 16));
        start_ocb = nstl::min(div_up(jcp.oc, 16), start_ocb);

        auto finish_ocb = 1;
        for (auto ocb = start_ocb; ocb >= finish_ocb; ocb--) {
            cur_brgb.oc_block = ocb * 16;
            cur_brgb.nb_oc = utils::div_up(jcp.oc, cur_brgb.oc_block);
            if (!cur_brgb.fast_check_oc_block()) continue;

            cur_brgb.calc_blocks();
            const auto ur = cur_brgb.get_brgemm_ur(&attr, dst_md);
            if (ur == 0) continue;
            cur_brgb.ur = ur;
            cur_brgb.eff = cur_brgb.est_eff();
            if (cur_brgb.eff > best_brgb.eff) best_brgb = cur_brgb;
        }
        if (best_brgb.oc_block == 0 || best_brgb.ic_block == 0
                || best_brgb.ow_block == 0)
            return false;
        best_brgb.save_to_jcp(jcp);
        selected_ur = best_brgb.ur;
        return true;
    };

    //-----------------------------------------------------------------------

    jcp.exec_type = exec_base;
    jcp.brg_type = brgemm_addr; // TODO: Choose right type of BRGEMM

    bool try_exec_vpad = false;
    bool try_exec_trans = false;
    bool try_exec_base = true;

    if (!is_amx(isa) && div_up(jcp.l_pad, jcp.stride_w) < jcp.kw
            && div_up(jcp.r_pad, jcp.stride_w) < jcp.kw) {
        try_exec_vpad = true;
    }

    const auto ic_padded_block = 16 * brg_blocking_t::last_ic_block_size;
    // TODO: remove this restriction
    const auto w_padding = jcp.l_pad > 0 || jcp.r_pad > 0;
    if (is_amx(isa)) {
        try_exec_base = !w_padding
                && IMPLICATION(jcp.ic <= ic_padded_block,
                        jcp.ic % brg_blocking_t::last_ic_block_size == 0)
                && IMPLICATION(
                        jcp.ic > ic_padded_block, jcp.ic % ic_padded_block == 0)
                && jcp.ow > 50 /*TODO: reinvestigate this heuristic */;
        try_exec_trans = !try_exec_base;
    }

    bool must_exec_vpad = false;

    // TODO: in future use (kd/kh/kw) and (kd/kh/kw)_pad blocks for more
    // precise calculation of jcp.max_batch
    jcp.max_batch = jcp.kd * jcp.kh * jcp.kw;

    //TODO: check wei plain
    jcp.wei_plain = false;
    jcp.wei_plain = jcp.exec_type == exec_vpad ? jcp.wei_plain : false;

    bool try_exec_type_res = false;

    if (try_exec_vpad) {
        jcp.exec_type = exec_vpad;
        try_exec_type_res = try_exec_type();
        // to avoid case when both top and bottom virtual padding are non-zero
        // TODO: remove this restriction
        const auto iw_block = (jcp.ow_block - 1) * jcp.stride_w + 1;
        if (!must_exec_vpad && (iw_block > jcp.iw)) try_exec_type_res = false;
    }
    if (try_exec_type_res == false && try_exec_trans) {
        jcp.exec_type = exec_trans;

        // try loop_ndhwgc always for exec_trans
        jcp.loop_order = loop_ndhwgc;

        // we read input block only once for loop_ndhwgc, so we don't need to
        // keep it memory
        if (jcp.loop_order == loop_ndhwgc) { jcp.copy_block_only = true; }

        jcp.is_ic_padded = one_of(jcp.wei_dt, bf16, s8)
                && jcp.ic * jcp.kw_sets > ic_padded_block;

        if (is_amx(isa) && (/* heuristic*/ jcp.kw_sets == 1 && jcp.ow < 256)) {
            jcp.is_os_blocking = jcp.f_pad < jcp.kd && jcp.back_pad < jcp.kd
                    && jcp.t_pad < jcp.kh && jcp.b_pad < jcp.kh
                    && jcp.r_pad < jcp.kw && jcp.l_pad < jcp.kw;
            jcp.use_M_mask = jcp.is_os_blocking ? 2 : 0;
            jcp.use_uker = true;
            jcp.use_interleave_stores = true;
            // assuming 2x2 decomposition in amx brgemm kernel
            // and overlap of input by kw
            const auto bd_blocking = 2 * jcp.amx_h;
            const auto ld_blocking = 2 * 16;
            const auto A_ds
                    = jcp.src_dsz * bd_blocking * jcp.ic * jcp.kd * jcp.kh;
            const auto B_ds = jcp.wei_dsz * ld_blocking * jcp.ic * jcp.kd
                    * jcp.kh * jcp.kw;
            const auto C_ds = jcp.acc_dsz * bd_blocking * ld_blocking;
            if (A_ds + B_ds + C_ds > brg_blocking_t::L1)
                jcp.amx_tile_load_xx = true;
        }

        try_exec_type_res = try_exec_type();
    }
    if (try_exec_base && try_exec_type_res == false) {
        jcp.exec_type = exec_base;
        try_exec_type_res = try_exec_type();
    }

    if (try_exec_type_res == false) return status::unimplemented;

    // ============ end blocking ===========================================
    if (jcp.exec_type == exec_vpad)
        jcp.max_vpad = nstl::max(jcp.l_pad, jcp.r_pad);
    else
        jcp.max_vpad = 0;

    if (jcp.ow_block == 0 || jcp.ic_block == 0 || jcp.oc_block == 0)
        return status::unimplemented;

    jcp.gemm_batch_size = jcp.nb_ic_blocking
            * nstl::max(jcp.kd_block * jcp.kh_block * jcp.kw_block,
                    jcp.kd_block_pad * jcp.kh_block_pad * jcp.kw_block_pad);
    // to avoid cache concurrent write access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jcp.adjusted_batch_size
            = div_up(rnd_up(jcp.gemm_batch_size * sc_size, 4096), sc_size);

    CHECK(pick_tags(jcp, src_md, weights_md, dst_md, bias_md));
    CHECK(attr.set_default_formats(&dst_md));

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    // only common and per-oc-channel scales are supported
    const bool oscales_ok = one_of(oscales.mask_, 0, 1 << 1);
    if (!oscales_ok) return status::unimplemented;

    jcp.buffer_size = jcp.LDC * jcp.M;

    jcp.nb_od = div_up(jcp.od, jcp.od_block);
    jcp.nb_oh = div_up(jcp.oh, jcp.oh_block);

    if (jcp.exec_type == exec_trans) {
        // TODO: this is rough estimation of buffer for transpose input
        dim_t ds = jcp.copy_block_only
                ? (brg_blocking_t::get_inp_size(jcp.idp, jcp.od_block, jcp.kd,
                           jcp.stride_d, jcp.dilate_d)
                        + nstl::max(0, jcp.f_pad) + nstl::max(0, jcp.back_pad))
                : jcp.idp;
        dim_t hs = jcp.copy_block_only
                ? (brg_blocking_t::get_inp_size(jcp.ihp, jcp.oh_block, jcp.kh,
                           jcp.stride_h, jcp.dilate_h)
                        + nstl::max(0, jcp.t_pad) + nstl::max(0, jcp.b_pad))
                : jcp.ihp;
        if (jcp.is_os_blocking)
            hs = div_up(rnd_up(hs * jcp.iwp, jcp.brgM), jcp.iwp);

        jcp.inp_buffer_size = rnd_up(ds * hs * jcp.iwp * jcp.ngroups * jcp.nb_ic
                        * jcp.ic_block * jcp.kh_sets * jcp.kw_sets,
                4096);
        jcp.inp_buffer_mask_size = rnd_up(static_cast<dim_t>(jcp.nb_od)
                        * jcp.nb_oh * jcp.nb_ow * jcp.ngroups * jcp.nb_ic,
                4096);
    }

    return status::success;
}

status_t init_1x1_conf(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads) {

    using namespace prop_kind;
    if (!mayiuse(isa)) return status::unimplemented;

    CHECK(init_jcp(
            jcp, isa, cd, src_md, weights_md, dst_md, bias_md, attr, nthreads));

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    if (!jcp.is_1x1) return status::unimplemented;

    using namespace data_type;
    // ===================== blocking =================================

    auto bcast_amount
            = static_cast<size_t>(jcp.id) * jcp.ih * jcp.iw * jcp.src_dsz;
    auto wei_amount = static_cast<size_t>(jcp.oc) * jcp.wei_dsz;

    jcp.loop_order = (bcast_amount < wei_amount) ? loop_ngcdhw : loop_ndhwgc;

    const auto min_oc_block = 16;

    jcp.brg_type = brgemm_addr; // TODO: Choose right type of BRGEMM

    // max_batch is 1 and max_vpad is 0 for 1x1 convolutions
    jcp.max_batch = 1;
    jcp.max_vpad = 0;

    jcp.wei_plain = false;

    brg_blocking_t best_brgb = zero<decltype(best_brgb)>();
    best_brgb.oc_block = min_oc_block;
    brg_blocking_t cur_brgb = zero<decltype(cur_brgb)>();
    cur_brgb.get_from_jcp(jcp);
    auto start_ocb = 4;
    if (jcp.wei_plain)
        start_ocb = nstl::min(jcp.ic > 128 ? (jcp.ic > 256 ? 8 : 16) : 32,
                div_up(jcp.oc, 16));
    start_ocb = nstl::min(div_up(jcp.oc, 16), start_ocb);

    auto finish_ocb = 1;
    for (auto ocb = start_ocb; ocb >= finish_ocb; ocb--) {
        cur_brgb.oc_block = ocb * min_oc_block;
        cur_brgb.nb_oc = utils::div_up(jcp.oc, cur_brgb.oc_block);

        if (!cur_brgb.fast_check_oc_block_1x1()) continue;

        cur_brgb.calc_blocks_1x1();
        const auto ur = cur_brgb.get_brgemm_ur(&attr, dst_md);
        if (ur == 0) continue;
        cur_brgb.ur = ur;
        cur_brgb.eff = cur_brgb.est_eff_1x1();
        if (cur_brgb.eff > best_brgb.eff) best_brgb = cur_brgb;
    }
    best_brgb.save_to_jcp(jcp);

    // =============== end blocking =================================
    jcp.brg_stride_a = jcp.ic_block * jcp.src_dsz;
    jcp.brg_stride_b = jcp.ic_block * jcp.oc * jcp.wei_dsz;

    if (jcp.ic_block == 0 || jcp.oc_block == 0) return status::unimplemented;

    // Configure matrix sizes

    if (best_brgb.is_os_blocking) {
        if (jcp.os_block == 0) return status::unimplemented;
        jcp.M = jcp.brgM = jcp.os_block;
        jcp.M_tail = jcp.brgM_tail = jcp.os % jcp.os_block;
    } else {
        if (jcp.ow_block == 0) return status::unimplemented;
        jcp.M = jcp.brgM = jcp.ow_block;
        jcp.M_tail = jcp.brgM_tail = jcp.ow % jcp.ow_block;
    }

    jcp.K = jcp.ic >= jcp.ic_block ? jcp.ic_block : 0;
    jcp.N = jcp.oc >= jcp.oc_block ? jcp.oc_block : 0;
    jcp.N_tail = jcp.oc % jcp.oc_block;
    jcp.K_tail = jcp.ic % jcp.ic_block;

    jcp.gemm_batch_size = jcp.nb_ic_blocking;
    // to avoid cache concurrent access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jcp.adjusted_batch_size
            = div_up(rnd_up(jcp.gemm_batch_size * sc_size, 4096), sc_size);

    jcp.LDA = jcp.stride_w * jcp.ic_without_padding;
    jcp.LDC = (jcp.use_buffer) ? jcp.oc_block : jcp.oc_without_padding;
    jcp.LDD = jcp.oc_without_padding;

    CHECK(pick_tags(jcp, src_md, weights_md, dst_md, bias_md));
    CHECK(attr.set_default_formats(&dst_md));

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    // only common and per-oc-channel scales are supported
    const bool oscales_ok = one_of(oscales.mask_, 0, 1 << 1);
    if (!oscales_ok) return status::unimplemented;

    // no inp buffer or brgemm_vpad for 1x1
    jcp.exec_type = exec_base;
    jcp.inp_buffer_size = 0;
    jcp.buffer_size = jcp.LDC * jcp.M;

#if 0
    printf("@@@ debug: nthreads = %d, IC = %d, OC = %d, ID = %d, IH = %d, IW = "
           "%d, OD = %d, OH = %d, OW = %d, KD = %d, "
           "KH = %d, KW = %d\n",
            nthreads, jcp.ic, jcp.oc, jcp.id, jcp.ih, jcp.iw, jcp.od, jcp.oh,
            jcp.ow, jcp.kd, jcp.kh, jcp.kw);

    printf("@@@ debug: blocking: ic_block = %d, nb_ic_blocking = %d, oc_block "
           "= %d, os_block = %d, ow_block = %d, nb_os_blocking = %d, "
           "loop_order = %d, "
           "wei_plain = %d, wei_tag = %d \n",
            jcp.ic_block, jcp.nb_ic_blocking, jcp.oc_block, jcp.os_block,
            jcp.ow_block, jcp.nb_os_blocking, jcp.loop_order, jcp.wei_plain,
            jcp.wei_tag);

    printf("@@@ debug: Matrix configuration: M = %d, N = %d, K = "
           "%d, M_tail = %d, N_tail = %d, K_tail = %d, LDA = %d, LDB = %d, LDC "
           "= %d ur = %d\n",
            jcp.M, jcp.N, jcp.K, jcp.M_tail, jcp.N_tail, jcp.K_tail, jcp.LDA,
            jcp.LDB, jcp.LDC, best_brgb.ur);
    printf("@@@ debug: brg_type = %d use_buffer = %d \n", jcp.brg_type,
            jcp.use_buffer);
    fflush(nullptr);
#endif
    return status::success;
}

void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const jit_brgemm_conv_conf_t &jcp) {
    if (jcp.brg_type == brgemm_addr || jcp.brg_type == brgemm_offs
            || (jcp.brg_type == brgemm_strd && jcp.exec_type == exec_vpad))
        scratchpad.book(key_brgemm_primitive_batch,
                static_cast<size_t>(jcp.nthr) * jcp.adjusted_batch_size,
                sizeof(brgemm_batch_element_t), 64);
    if (jcp.exec_type == exec_trans) {
        size_t inp_buffer_size
                = static_cast<size_t>(jcp.nthr) * jcp.inp_buffer_size;
        scratchpad.book(
                key_conv_brgemm_inp_buffer, inp_buffer_size, jcp.src_dsz);
        size_t inp_buffer_mask_size
                = static_cast<size_t>(jcp.nthr) * jcp.inp_buffer_mask_size;
        scratchpad.book(key_conv_brgemm_inp_buffer_mask, inp_buffer_mask_size,
                sizeof(uint8_t));
    }
    if (jcp.use_buffer) {
        scratchpad.book(key_brgemm_primitive_buffer, jcp.nthr * jcp.buffer_size,
                jcp.acc_dsz);
    }
    if (is_amx(jcp.isa)) {
        scratchpad.book(
                key_conv_amx_tile_buffer, jcp.nthr * 4 * 1024, sizeof(char));
    }
}

} // namespace brgemm_convolution_utils

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
