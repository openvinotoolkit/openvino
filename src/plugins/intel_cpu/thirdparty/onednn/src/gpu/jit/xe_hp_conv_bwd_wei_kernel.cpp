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

#include "gpu/jit/xe_hp_conv_bwd_wei_kernel.hpp"

#include "gpu/jit/jit_generator.hpp"
#include "gpu/jit/ngen/ngen_register_allocator.hpp"
#include "gpu/jit/ngen/ngen_utils.hpp"
#include "gpu/jit/ngen_type_bridge.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using namespace ngen;

template <gpu_gen_t hw>
class xe_hp_conv_bwd_wei_init_kernel_t : public jit_generator<hw> {
public:
    NGEN_FORWARD_OPENCL(hw);

    const conv_conf_t conf;
    RegisterAllocator ra;

public:
    xe_hp_conv_bwd_wei_init_kernel_t(const conv_conf_t &conf)
        : conf(conf), ra(hw) {
        newArgument("wei", ExternalArgumentType::GlobalPtr);
        newArgument("bia", ExternalArgumentType::GlobalPtr);
        requireGRF(128);
        requireSIMD(8);

        finalizeInterface();

        auto arg_wei_ptr = getArgument("wei");
        auto arg_bia_ptr = getArgument("bia");

        ra.claim(r0);
        ra.claim(arg_wei_ptr);
        ra.claim(arg_bia_ptr);

        Subregister thread_x = r0.ud(1); // IC * SP
        Subregister thread_y = r0.ud(6); // G * OC

        const uint32_t wei_sp = conf.kd * conf.kh * conf.kw;
        const uint32_t ic_sp = conf.icb * wei_sp;
        const uint32_t wei_blk_size = conf.ic_block * conf.oc_block * 4
                / 16; // 16i16o f32 in owords
        const uint32_t bia_blk_size
                = conf.oc_block * 4 / 16; // 16o f32 in owords

        auto tmp = ra.alloc_range(16);
        auto zero = ra.alloc_range(16);

        auto wei_header = ra.alloc_range(8);
        auto bia_header = ra.alloc();

        auto wei_offset = ra.alloc();
        auto bia_offset = ra.alloc();

        setDefaultNoMask();
        setDefaultAutoSWSB(true);

        prologue();

        auto group_x = tmp[0].ud(0);
        auto group_y = tmp[3].ud(0);

        mov(1, group_x.ud(), thread_x.ud());
        mov(1, group_y.ud(), thread_y.ud());

        const uint16_t ic_sp_lo = ic_sp & 0xffffu;

        mad(1, wei_offset.ud(), group_x.ud(), group_y.ud(), ic_sp_lo);
        if (ic_sp > 0xffff) {
            mach(1, tmp[1].ud(), group_y.ud(), uint32_t(ic_sp));
            add(1, wei_offset.ud(), wei_offset.ud(), acc0.ud());
        }

        mov(16, wei_header[0].f(), 0);
        mov(16, wei_header[2].f(), 0);
        mov(16, wei_header[4].f(), 0);
        mov(16, wei_header[6].f(), 0);

        if (conf.with_bias) {
            mov(8, bia_header.ud(), 0);
            cmp(16 | eq | f0[0], thread_x.ud(), 0);
        }

        mul(1, wei_offset.ud(), wei_offset.ud(), uint16_t(wei_blk_size));
        mul(1, bia_offset.ud(), group_y.ud(), uint16_t(bia_blk_size));

        for (int i = 0; i < 8; i += 2)
            mov(16, zero[i].f(), float(0.f));

        for (uint32_t i = 0, blk_offset = 0; blk_offset < wei_blk_size;
                i++, blk_offset += 8) {
            add(1, wei_header[i].ud(2), wei_offset.ud(), uint32_t(blk_offset));
        }

        mov(1, bia_header.ud(2), bia_offset.ud());

        auto wei_surf = Surface(getArgumentSurface("wei"));
        auto bia_surf = Surface(getArgumentSurface("bia"));

        sync(SyncFunction::nop, SWSB<AllPipes>(1));

        for (uint32_t i = 0, blk_offset = 0; blk_offset < wei_blk_size;
                i++, blk_offset += 8) {
            store(16, block_oword(8), wei_surf, wei_header[i], zero);
        }

        if (conf.with_bias) {
            Label skip_bias;
            if_(16 | f0[0], skip_bias);
            store(16, block_oword(4), bia_surf, bia_header, zero);
            mark(skip_bias);
            endif(16);
        }

        sync(SyncFunction::allwr, 0xffff);

        epilogue();
    };
};

template <gpu_gen_t hw>
class xe_hp_conv_bwd_wei_cvt_kernel_t : public jit_generator<hw> {
public:
    NGEN_FORWARD_OPENCL(hw);

    const conv_conf_t conf;
    RegisterAllocator ra;

public:
    xe_hp_conv_bwd_wei_cvt_kernel_t(const conv_conf_t &conf)
        : conf(conf), ra(hw) {
        newArgument("wei_bf16", ExternalArgumentType::GlobalPtr);
        newArgument("bia_bf16", ExternalArgumentType::GlobalPtr);
        newArgument("wei_f32", ExternalArgumentType::GlobalPtr);
        newArgument("bia_f32", ExternalArgumentType::GlobalPtr);
        requireGRF(128);
        requireSIMD(8);

        finalizeInterface();

        auto arg_wei_f32_ptr = getArgument("wei_f32");
        auto arg_bia_f32_ptr = getArgument("bia_f32");
        auto arg_wei_bf16_ptr = getArgument("wei_bf16");
        auto arg_bia_bf16_ptr = getArgument("bia_bf16");

        ra.claim(r0);
        ra.claim(arg_wei_f32_ptr);
        ra.claim(arg_bia_f32_ptr);
        ra.claim(arg_wei_bf16_ptr);
        ra.claim(arg_bia_bf16_ptr);

        Subregister thread_x = r0.ud(1); // IC * SP
        Subregister thread_y = r0.ud(6); // G * OC

        const bool do_weights = conf.weights_data_type == data_type::bf16;
        const bool do_bias
                = conf.with_bias && conf.bias_data_type == data_type::bf16;

        const uint32_t goc_stride
                = conf.ic_block * conf.icb * conf.kd * conf.kh * conf.kw;

        // weights 16i16o, bias 16o
        const uint32_t wei_blk_size = conf.oc_block;
        const uint32_t bia_blk_size = conf.oc_block;

        const uint32_t wei_f32_blk_sz_in_ow = wei_blk_size * sizeof(float) / 16;
        const uint32_t wei_bf16_blk_sz_in_ow
                = wei_blk_size * sizeof(uint16_t) / 16;

        const uint32_t bia_f32_blk_sz_in_ow = bia_blk_size * sizeof(float) / 16;
        const uint32_t bia_bf16_blk_sz_in_ow
                = bia_blk_size * sizeof(uint16_t) / 16;

        const uint32_t wei_f32_n_grfs = wei_blk_size * sizeof(float) / 32;
        const uint32_t wei_bf16_n_grfs = wei_blk_size * sizeof(uint16_t) / 32;
        const uint32_t bia_f32_n_grfs = bia_blk_size * sizeof(float) / 32;
        const uint32_t bia_bf16_n_grfs = bia_blk_size * sizeof(uint16_t) / 32;

        auto w_f32 = ra.alloc_range(wei_f32_n_grfs);
        auto w_bf16 = ra.alloc_range(wei_bf16_n_grfs);

        auto b_f32 = ra.alloc_range(bia_f32_n_grfs);
        auto b_bf16 = ra.alloc_range(bia_bf16_n_grfs);

        auto wei_f32_hdr = ra.alloc();
        auto wei_bf16_hdr = ra.alloc();

        auto bia_f32_hdr = ra.alloc();
        auto bia_bf16_hdr = ra.alloc();

        auto w_off = ra.alloc();

        auto w_f32_off = ra.alloc();
        auto b_f32_off = ra.alloc();
        auto w_bf16_off = ra.alloc();
        auto b_bf16_off = ra.alloc();

        setDefaultNoMask();
        setDefaultAutoSWSB(true);

        prologue();

        if (do_weights) {
            mul(1, w_off.ud(), thread_y.ud(), uint32_t(goc_stride));
            add(1, w_off.ud(), w_off.ud(), thread_x.ud());

            mul(1, w_f32_off.ud(), w_off.ud(), uint16_t(wei_f32_blk_sz_in_ow));
            mul(1, w_bf16_off.ud(), w_off.ud(),
                    uint16_t(wei_bf16_blk_sz_in_ow));
        }
        if (do_bias) {
            cmp(16 | eq | f0[0], thread_x.ud(), 0);
            mul(1, b_f32_off.ud(), thread_y.ud(),
                    uint16_t(bia_f32_blk_sz_in_ow));
            mul(1, b_bf16_off.ud(), thread_y.ud(),
                    uint16_t(bia_bf16_blk_sz_in_ow));
        }

        // init load/store headers
        if (do_weights) {
            mov(8, wei_f32_hdr.ud(), 0);
            mov(8, wei_bf16_hdr.ud(), 0);
        }
        if (do_bias) {
            mov(8, bia_f32_hdr.ud(), 0);
            mov(8, bia_bf16_hdr.ud(), 0);
        }

        if (do_weights) {
            mov(1, wei_f32_hdr.ud(2), w_f32_off.ud());
            mov(1, wei_bf16_hdr.ud(2), w_bf16_off.ud());
        }
        if (do_bias) {
            mov(1, bia_f32_hdr.ud(2), b_f32_off.ud());
            mov(1, bia_bf16_hdr.ud(2), b_bf16_off.ud());
        }

        auto wei_f32_surf = Surface(getArgumentSurface("wei_f32"));
        auto wei_bf16_surf = Surface(getArgumentSurface("wei_bf16"));
        auto bia_f32_surf = Surface(getArgumentSurface("bia_f32"));
        auto bia_bf16_surf = Surface(getArgumentSurface("bia_bf16"));

        sync(SyncFunction::nop, SWSB<AllPipes>(1));

        if (do_weights) {
            load(16, w_f32[0], block_oword(bia_f32_blk_sz_in_ow), wei_f32_surf,
                    wei_f32_hdr);
        }
        if (do_bias) {
            Label skip_bias;
            if_(16 | f0[0], skip_bias);
            load(16, b_f32[0], block_oword(bia_f32_blk_sz_in_ow), bia_f32_surf,
                    bia_f32_hdr);
            mark(skip_bias);
            endif(16);
        }

        if (do_weights) {
            for (uint32_t i = 0; i < wei_bf16_n_grfs; ++i) {
                mov(8, w_bf16[i].bf(0), w_f32[2 * i + 0].f());
                mov(8, w_bf16[i].bf(8), w_f32[2 * i + 1].f());
            }

            store(16, block_oword(wei_bf16_blk_sz_in_ow), wei_bf16_surf,
                    wei_bf16_hdr, w_bf16[0]);
        }

        if (do_bias) {
            Label skip_bias;
            if_(16 | f0[0], skip_bias);
            mov(8, b_bf16[0].bf(0), b_f32[0].f());
            mov(8, b_bf16[0].bf(8), b_f32[1].f());
            store(16, block_oword(bia_bf16_blk_sz_in_ow), bia_bf16_surf,
                    bia_bf16_hdr, b_bf16[0]);
            mark(skip_bias);
            endif(16);
        }

        sync(SyncFunction::allwr, 0xffff);

        epilogue();
    };
};

template <gpu_gen_t hw>
class xe_hp_conv_bwd_wei_conv_kernel_t : public jit_generator<hw> {
public:
    NGEN_FORWARD_OPENCL(hw);

    const conv_conf_t conf;
    RegisterAllocator ra;

    DataType src_type, wei_type, bia_type, dst_type;

    int src_blk_size, dst_blk_size, wei_blk_size, bia_blk_size;
    uint32_t src_sp, dst_sp, wei_sp;

    uint32_t slm_blk_sz_in_ow;
    uint32_t slm_buf_sz_in_ow;

    Subregister thread_x, thread_y, thread_z;
    Subregister arg_src_ptr, arg_wei_ptr, arg_bia_ptr, arg_dst_ptr;
    Subregister local_size;

    GRFRange a[2], b[2], c[4];
    GRFRange s0, d0, tmp_regs;

    GRF reg_thr_ids;
    Subregister thr_local_id, thr_local_size, thr_fused_id, thr_src_id;
    Subregister src_select, bia_select, load_next;
    Subregister isp, osp, mb_g;

    GRF reg_sp_ids[3], reg_sp_init, reg_sp_max;
    Subregister iw, ih, id;
    Subregister iw0, ih0, id0;
    Subregister iw_init, ih_init, id_init;
    Subregister iw_max, ih_max, id_max;
    Subregister ow, oh, od;
    Subregister ow_init, oh_init, od_init;
    Subregister kw, kh, kd, kdhw;

    GRF gmem_blk_off;
    GRF slm_blk_off;

    GRF block_ids;
    Subregister icb_idx, ocb_idx, g_idx, mb_idx;
    Subregister store_slm_off;

    GRF reg_tg_ptrs;
    Subregister src_tg_ptr, dst_tg_ptr, wei_tg_off, bia_tg_off;

    GRF src_blk_ptrs, dst_blk_ptrs;
    GRFRange gmem_hdrs;

    GRF reg_slm_local_offs, reg_slm_buf_offs;
    Subregister icb_local, ocb_local;
    Subregister slm_src_load_off, slm_dst_load_off;

    GRFRange store_slm_hdrs;

    GRFRange wei_headers;
    GRFRange w, bia;

    GRF reg_counters, reg_local_sp;
    Subregister count, blk_idx, max_count;
    Subregister ls_slm_buf_off;
    GRF r_sfence, r_mfence, r_bar;

public:
    xe_hp_conv_bwd_wei_conv_kernel_t(const conv_conf_t &conf)
        : conf(conf), ra(hw) {

        proto();
        allocate_registers();

        auto sg_local_id = getLocalID(0);

        setDefaultNoMask();
        setDefaultAutoSWSB(true);

        prologue();

        sync(SyncFunction::allwr, 0xffff); // sync load args

        emov_qq(src_tg_ptr.q(), arg_src_ptr.q());
        emov_qq(dst_tg_ptr.q(), arg_dst_ptr.q());

        mov(8, reg_sp_ids[0].d(), 0);
        mov(8, reg_sp_ids[1].d(), 0);
        mov(8, reg_sp_ids[2].d(), 0);

        // split work item ids
        const int sg_size_shift = ngen::utils::log2(conf.sub_group_size);
        shr(1, thr_local_id.uw(), sg_local_id.uw(), int(sg_size_shift));
        shr(1, thr_local_size.uw(), local_size.uw(), int(sg_size_shift));
        shr(1, thr_src_id.uw(), thr_local_id.uw(), 1);
        and_(1, src_select.uw(), thr_src_id.uw(), 1);
        and_(1, thr_src_id.uw(), thr_src_id.uw(), uint16_t(0xfffe));
        and_(1, thr_fused_id.uw(), thr_local_id.uw(), 1);
        or_(1, thr_src_id.uw(), thr_src_id.uw(), thr_fused_id.uw());
        mul(1, src_select.uw(), src_select.uw(), uint16_t(0xffff));

        allocate_registers_v2();

        mov(1, mb_g.ud(), thread_z.ud());

        {
            const uint32_t div_ic = conf.icb / (conf.ic_blk_wg * 2);
            const uint32_t div_oc = conf.ocb / (conf.oc_blk_wg * 2);
            const uint32_t div_ic_oc = div_ic * div_oc;
            const uint32_t div_khw = conf.kh * conf.kw;
            const uint32_t div_kw = conf.kw;

            auto tmp = tmp_regs;

            auto arg_icb_ocb = tmp[0].ud(0);
            auto arg_odh = tmp[1].ud(0);
            auto arg_kdhw = tmp[2].ud(0);
            auto arg_khw = tmp[3].ud(0);
            auto arg_kw = tmp[4].ud(0);
            auto arg_kh = tmp[5].ud(0);
            auto arg_kd = tmp[6].ud(0);

            auto r_div = tmp[7];

            mul(1, osp.ud(), thread_y.ud(), uint16_t(conf.sp_block));

            eidiv_udud(1, arg_kdhw.ud(), thread_x.ud(), r_div,
                    uint32_t(div_ic_oc));
            mad(1, arg_icb_ocb.ud(), thread_x.ud(), -abs(arg_kdhw.uw()),
                    uint16_t(div_ic_oc));

            eidiv_udud(
                    1, ocb_idx.ud(), arg_icb_ocb.ud(), r_div, uint32_t(div_ic));
            mad(1, icb_idx.ud(), arg_icb_ocb.ud(), -abs(ocb_idx.uw()),
                    uint16_t(div_ic));

            eidiv_udud(1, arg_kd.ud(), arg_kdhw.ud(), r_div, uint32_t(div_khw));
            mad(1, arg_khw.ud(), arg_kdhw.ud(), -abs(arg_kd.uw()),
                    uint16_t(div_khw));

            eidiv_udud(1, arg_kh.ud(), arg_khw.ud(), r_div, uint32_t(div_kw));
            mad(1, arg_kw.ud(), arg_khw.ud(), -abs(arg_kh.uw()),
                    uint16_t(div_kw));

            eidiv_udud(1, arg_odh.ud(), osp.ud(), r_div, uint32_t(conf.ow));
            mad(1, ow_init.ud(), osp.ud(), -abs(arg_odh.uw()),
                    uint16_t(conf.ow));

            eidiv_udud(1, od_init.ud(), arg_odh.ud(), r_div, uint32_t(conf.oh));
            mad(1, oh_init.ud(), arg_odh.ud(), -abs(od_init.uw()),
                    uint16_t(conf.oh));

            eidiv_udud(1, mb_idx.ud(), thread_z.ud(), r_div,
                    uint32_t(conf.ngroups));
            mad(1, g_idx.ud(), thread_z.ud(), -abs(mb_idx.uw()),
                    uint16_t(conf.ngroups));

            mov(1, kw.uw(), arg_kw.uw());
            mov(1, kh.uw(), arg_kh.uw());
            mov(1, kd.uw(), arg_kd.uw());
            mov(1, kdhw.uw(), arg_kdhw.uw());
        }

        if (conf.with_bias) {
            auto s_is_ic0 = tmp_regs[0].ud(0);
            auto s_is_kdhw0 = tmp_regs[6].uw(0);

            cmp(1 | eq | f0[0], s_is_ic0.ud(), icb_idx.ud(), 0);
            cmp(1 | eq | f0[1], s_is_kdhw0.uw(), kdhw.uw(), 0);
            and_(1, bia_select.uw(), s_is_ic0.uw(), s_is_kdhw0.uw(0));
            and_(1, bia_select.uw(), bia_select.uw(), ~src_select.uw());
        }

        const int icb_local_bits = ngen::utils::log2(conf.ic_blk_wg);
        const int ocb_local_bits = ngen::utils::log2(conf.oc_blk_wg);

        shl(1, icb_idx.ud(), icb_idx.ud(), int32_t(icb_local_bits + 1));
        shl(1, ocb_idx.ud(), ocb_idx.ud(), int32_t(ocb_local_bits + 1));

        const uint32_t icb_local_mask = ~(~0u << icb_local_bits);
        const uint32_t icb_local_shift = icb_local_bits;

        and_(1, icb_local.ud(), thr_local_id.uw(), uint32_t(icb_local_mask));
        shr(1, ocb_local.ud(), thr_local_id.uw(), uint32_t(icb_local_shift));

        shl(1, icb_local.ud(), icb_local.ud(), 1);
        shl(1, ocb_local.ud(), ocb_local.ud(), 1);

        or_(1, icb_idx.ud(), icb_idx.ud(), icb_local.uw());
        or_(1, ocb_idx.ud(), ocb_idx.ud(), ocb_local.uw());

        // block byte offsets
        mov(8, gmem_blk_off.uw(0), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
        mov(8, slm_blk_off.uw(0), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
        // gem blk : 256bytes in owords
        shl(8, gmem_blk_off.ud(), gmem_blk_off.uw(0)(1), 8);
        // slm : 256bytes in owords
        shl(8, slm_blk_off.ud(), slm_blk_off.uw(0)(1), 4);

        // reset slm headers
        init_slm_headers();

        // src address
        init_gmem_offsets_tg_32n16c();

        // main loop
        auto flag_loop = f1[1];

        mov(4, reg_counters.d(0), 0);

        mov(4, iw.d(0)(1), iw_init.d(0)(1));
        mov(4, ow.d(0)(1), ow_init.d(0)(1));

        const int32_t max_sp = conf.ow * conf.oh * conf.od;

        add(1, max_count.d(), -osp.d(), int32_t(max_sp));
        min_(1, max_count.d(), max_count.d(), int32_t(conf.sp_block));

        shl(1, max_count.d(), max_count.d(), 1);

        // 1st iteration
        load_gmem();
        zero_acc();
        store_slm();

        // 2nd iteration
        load_gmem();

        // skip multiply loop
        add(1, count.d(), count.d(), 2);
        cmp(16 | lt | flag_loop, count.d(), max_count.d());

        store_slm();

        // fence and signal
        slmfence(InstructionModifier() | sb12, r_sfence);
        mov(8, null.ud(), r_sfence.ud());
        barriersignal(InstructionModifier() | sb13, r_bar);

        Label reduction_loop, skip_loop;
        jmpi(1 | ~flag_loop, skip_loop);

        // reduction loop
        mark(reduction_loop);
        {
            // sync pipes at the start of the reduction loop
            sync(SyncFunction::nop, SWSB<AllPipes>(1));

            load_gmem();
            multiply_32i32o();
            store_slm();

            add(1, count.d(), count.d(), 1);
            cmp(16 | lt | flag_loop, count.d(), max_count.d());
        }
        while_(16 | flag_loop, reduction_loop);

        // last 2 iterations
        mark(skip_loop);
        multiply_32i32o();
        multiply_32i32o(true);

        sync(SyncFunction::nop, SWSB<AllPipes>(1));
        sync(SyncFunction::allwr, 0xffff);

        auto tmp_idx = tmp_regs[0];
        auto val_idx = tmp_regs[1];
        auto w_hdr0 = tmp_regs[4];
        auto b_hdr0 = tmp_regs[7];
        auto w_off = tmp_regs[11];

        mov(8, tmp_idx.uw(0), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
        mov(8, tmp_idx.uw(8), Immediate::uv(8, 9, 10, 11, 12, 13, 14, 15));
        shl(16, val_idx.ud(), tmp_idx.uw(), 2); // [0..15] f32
        add(16, w_off.ud(), val_idx.ud(), wei_tg_off.ud(0)(0));

        if (conf.with_bias) {
            Label skip_bias;

            mov(1, f0[0].uw(), bia_select.uw());
            if_(16 | f0[0], skip_bias);
            {
                auto bia_surf = Surface(getArgumentSurface("bia"));
                add(16, b_hdr0.ud(), val_idx.ud(), bia_tg_off.ud(0)(0));
                atomic(AtomicOp::fadd, 16 | sb4, scattered_dword(), bia_surf,
                        b_hdr0, bia[0]);
                sync(SyncFunction::allrd, 0x0008);
            }
            mark(skip_bias);
            endif(16);
        }

        auto wei_surf = Surface(getArgumentSurface("wei"));
        for (int acc_idx = 0; acc_idx < 4; ++acc_idx) {
            auto ic_idx = acc_idx & 1;
            auto oc_idx = acc_idx >> 1;

            const int32_t c_stride = conf.kd * conf.kh * conf.kw * conf.ic_block
                    * conf.oc_block * sizeof(float);
            const int32_t next_c = (ic_idx + oc_idx * conf.icb) * c_stride;

            add(16, w_hdr0.ud(), w_off.ud(), uint32_t(next_c));

            if (acc_idx > 0) {
                sync(SyncFunction::allrd, 0xffff); // flush all
            }

            for (int i = 0; i < 32; i += 2) {
                add(16, wei_headers[i].ud(), w_hdr0.ud(),
                        uint16_t((i * 8) << 2)); // [0..15][0..15] f32
            }

            reorder_ud_2o8i8o_to_8i16o(w[0], c[acc_idx][0]);
            reorder_ud_2o8i8o_to_8i16o(w[16], c[acc_idx][16]);

            for (int i = 0; i < 16; ++i) {
                atomic(AtomicOp::fadd, 16 | SBID(i), scattered_dword(), //
                        wei_surf, wei_headers[i << 1], w[i << 1]);
            }
        }

        epilogue();

        kernel_pad(); // Kernel padding for instruction prefetch.
    }

    void proto() {
        // declare interface
        newArgument("src", ExternalArgumentType::GlobalPtr);
        newArgument("wei", ExternalArgumentType::GlobalPtr);
        newArgument("bia", ExternalArgumentType::GlobalPtr);
        newArgument("dst", ExternalArgumentType::GlobalPtr);

        assert(conf.ic_block == conf.oc_block);

        const uint32_t slm_blk_sz
                = (conf.mb_block / 2) * conf.ic_block * sizeof(uint16_t);
        const uint32_t slm_buf_sz
                = (conf.ic_blk_wg + conf.oc_blk_wg) * 2 * slm_blk_sz;
        const uint32_t slm_total_sz = 3 * slm_buf_sz;

        slm_blk_sz_in_ow = slm_blk_sz / 16;
        slm_buf_sz_in_ow = slm_buf_sz / 16;

        externalName("xe_hp_conv_bwd_wei");
        requireLocalID(3);
        requireLocalSize();
        requireGRF(256);
        requireSIMD(8);
        requireBarrier();
        requireDPAS();
        requireSLM(slm_total_sz);

        finalizeInterface();

        src_type = convert_dnnl_type_to_ngen(conf.src_data_type);
        dst_type = convert_dnnl_type_to_ngen(conf.dst_data_type);
        wei_type = DataType::f;
        bia_type = DataType::f;

        src_sp = conf.iw * conf.ih * conf.id;
        dst_sp = conf.ow * conf.oh * conf.od;
        wei_sp = conf.kw * conf.kh * conf.kd;

        src_blk_size = conf.mb_block * conf.ic_block * 2; // 32x16 bf16
        dst_blk_size = conf.mb_block * conf.oc_block * 2; // 32x16 bf16
        wei_blk_size = conf.ic_block * conf.oc_block * 4; // 16x16 f32
        bia_blk_size = conf.oc_block * 4; // 16 f32
    }

    void kernel_pad() {
        for (int rep = 0; rep < 8; rep++)
            nop();
    }

    void reorder_uw_16n16c_to_2c8n8c2n(const GRF &dst, const GRF &src) {
        GRFRange s(src.getBase(), 16);
        GRFRange d(dst.getBase(), 16);

        for (int i = 0; i < 8; ++i) {
            mov(8, d[i + 0].uw(0)(2), s[2 * i].uw(0)(1));
            mov(8, d[i + 0].uw(1)(2), s[2 * i + 1].uw(0)(1));
        }
        for (int i = 0; i < 8; ++i) {
            mov(8, d[i + 8].uw(0)(2), s[2 * i].uw(8)(1));
            mov(8, d[i + 8].uw(1)(2), s[2 * i + 1].uw(8)(1));
        }
    }

    void reorder_uw_16n16c_to_16c16n(const GRF &dst, const GRF &src) {
        GRFRange s(src.getBase(), 16);
        GRFRange d(dst.getBase(), 16);

        // -> 8n16c2n
        for (int i = 0; i < 16; i += 2) {
            mov(16, tmp_regs[i].uw(0)(2), s[i + 0].uw(0)(1));
            mov(16, tmp_regs[i].uw(1)(2), s[i + 1].uw(0)(1));
        }
        // -> 4n16c4n
        for (int i = 0; i < 16; i += 4) {
            mov(8, d[i + 0].ud(0)(2), tmp_regs[i + 0].ud(0)(1));
            mov(8, d[i + 2].ud(0)(2), tmp_regs[i + 1].ud(0)(1));
            mov(8, d[i + 0].ud(1)(2), tmp_regs[i + 2].ud(0)(1));
            mov(8, d[i + 2].ud(1)(2), tmp_regs[i + 3].ud(0)(1));
        }
        // -> 2n16c4n2n
        for (int i = 0; i < 16; i += 8) {
            mov(8, tmp_regs[i + 0].ud(0)(2), d[i + 0].ud(0)(1));
            mov(8, tmp_regs[i + 2].ud(0)(2), d[i + 1].ud(0)(1));
            mov(8, tmp_regs[i + 4].ud(0)(2), d[i + 2].ud(0)(1));
            mov(8, tmp_regs[i + 6].ud(0)(2), d[i + 3].ud(0)(1));
            mov(8, tmp_regs[i + 0].ud(1)(2), d[i + 4].ud(0)(1));
            mov(8, tmp_regs[i + 2].ud(1)(2), d[i + 5].ud(0)(1));
            mov(8, tmp_regs[i + 4].ud(1)(2), d[i + 6].ud(0)(1));
            mov(8, tmp_regs[i + 6].ud(1)(2), d[i + 7].ud(0)(1));
        }
        // -> 16c16n
        for (int i = 0; i < 8; ++i) {
            mov(4, d[2 * i + 0].ud(0)(1), tmp_regs[i + 0].ud(0)(1, 2, 2));
            mov(4, d[2 * i + 0].ud(4)(1), tmp_regs[i + 8].ud(0)(1, 2, 2));
            mov(4, d[2 * i + 1].ud(0)(1), tmp_regs[i + 0].ud(4)(1, 2, 2));
            mov(4, d[2 * i + 1].ud(4)(1), tmp_regs[i + 8].ud(4)(1, 2, 2));
        }
    }

    void reorder_ud_2o8i8o_to_8i16o(const GRF &dst, const GRF &src) {
        GRFRange s(src.getBase(), 16);
        GRFRange d(dst.getBase(), 16);
        for (int i = 0; i < 8; ++i) {
            mov(8, d[2 * i + 0].ud(), s[i + 0].ud(0)(1));
            mov(8, d[2 * i + 1].ud(), s[i + 8].ud(0)(1));
        }
    }

    void multiply_half_block_32i32o(
            const SBID &mod0, const SBID &mod1, int ic_idx) {
        const int acc_idx = ic_idx;

        dpasw(8 | mod0.dst | Atomic, 8, 8, //
                c[acc_idx + 0][0].f(), c[acc_idx + 0][0].f(), //
                a[ic_idx][0].bf(), b[0][0].bf());
        dpasw(8 | Atomic, 8, 8, //
                c[acc_idx + 0][16].f(), c[acc_idx + 0][16].f(), //
                a[ic_idx][0].bf(), b[0][4].bf());
        dpasw(8 | Atomic, 8, 8, //
                c[acc_idx + 2][0].f(), c[acc_idx + 2][0].f(), //
                a[ic_idx][0].bf(), b[1][0].bf());
        dpasw(8 | mod0, 8, 8, //
                c[acc_idx + 2][16].f(), c[acc_idx + 2][16].f(), //
                a[ic_idx][0].bf(), b[1][4].bf());

        dpasw(8 | mod1.dst | Atomic, 8, 8, //
                c[acc_idx + 0][8].f(), c[acc_idx + 0][8].f(), //
                a[ic_idx][8].bf(), b[0][0].bf());
        dpasw(8 | Atomic, 8, 8, //
                c[acc_idx + 0][24].f(), c[acc_idx + 0][24].f(), //
                a[ic_idx][8].bf(), b[0][4].bf());
        dpasw(8 | Atomic, 8, 8, //
                c[acc_idx + 2][8].f(), c[acc_idx + 2][8].f(), //
                a[ic_idx][8].bf(), b[1][0].bf());
        dpasw(8 | mod1, 8, 8, //
                c[acc_idx + 2][24].f(), c[acc_idx + 2][24].f(), //
                a[ic_idx][8].bf(), b[1][4].bf());
    }

    void compute_bias(const GRF &dst) {
        GRFRange s(dst.getBase(), 16);

        auto tmp = tmp_regs;

        add(16, tmp[0].f(), s[0].bf(), s[7].bf());
        add(16, tmp[6].f(), s[1].bf(), s[6].bf());
        add(16, tmp[8].f(), s[2].bf(), s[13].bf());
        add(16, tmp[14].f(), s[3].bf(), s[12].bf());
        add(16, tmp[2].f(), s[4].bf(), s[11].bf());
        add(16, tmp[10].f(), s[5].bf(), s[10].bf());
        add(16, tmp[4].f(), s[8].bf(), s[15].bf());
        add(16, tmp[12].f(), s[9].bf(), s[14].bf());

        add(16, tmp[0].f(), tmp[0].f(), tmp[6].f());
        add(16, tmp[14].f(), tmp[8].f(), tmp[14].f());
        add(16, tmp[2].f(), tmp[2].f(), tmp[10].f());
        add(16, tmp[12].f(), tmp[4].f(), tmp[12].f());

        add(16, tmp[0].f(), tmp[0].f(), tmp[14].f());
        add(16, tmp[12].f(), tmp[2].f(), tmp[12].f());

        add(16, tmp[0].f(), tmp[0].f(), tmp[12].f());

        add(16, bia[0].f(), bia[0].f(), tmp[0].f());
    }

    void allocate_registers() {
        // r0 + local_id
        ra.claim(r0);
        thread_x = r0.ud(1);
        thread_y = r0.ud(6);
        thread_z = r0.ud(7);

        ra.claim(getLocalID(0));

        local_size = getLocalSize(0);
        ra.claim(local_size);

        // args
        arg_src_ptr = getArgument("src");
        arg_wei_ptr = getArgument("wei");
        arg_bia_ptr = getArgument("bia");
        arg_dst_ptr = getArgument("dst");
        ra.claim(arg_src_ptr);
        ra.claim(arg_wei_ptr);
        ra.claim(arg_bia_ptr);
        ra.claim(arg_dst_ptr);

        // dpas
        for (int i = 0; i < 4; ++i)
            c[i] = GRFRange(28 + i * 64, 32);

        for (int i = 0; i < 2; ++i)
            a[i] = GRFRange(60 + i * 16, 16);

        for (int i = 0; i < 2; ++i)
            b[i] = GRFRange(202 + i * 8, 8);

        for (int i = 0; i < 4; ++i)
            ra.claim(c[i]);
        for (int i = 0; i < 2; ++i)
            ra.claim(a[i]);
        for (int i = 0; i < 1; ++i)
            ra.claim(b[i]);

        // prefetch
        s0 = GRFRange(124, 16);
        d0 = GRFRange(140, 16);

        tmp_regs = GRFRange(12, 16);

        ra.claim(s0);
        ra.claim(d0);
        ra.claim(tmp_regs);

        // fences
        r_mfence = GRF(253); // reused
        r_sfence = GRF(254);
        r_bar = GRF(255);

        ra.claim(r_sfence);
        ra.claim(r_bar);

        // bias
        bia = ra.alloc_range(2);

        // slm
        store_slm_hdrs = ra.alloc_range(2);
        gmem_hdrs = ra.alloc_range(2);

        reg_slm_local_offs = ra.alloc();
        reg_slm_buf_offs = ra.alloc();

        icb_local = reg_slm_local_offs.d(0);
        ocb_local = reg_slm_local_offs.d(1);

        slm_src_load_off = reg_slm_buf_offs.ud(0);
        slm_dst_load_off = reg_slm_buf_offs.ud(1);

        // thread group ids
        reg_thr_ids = ra.alloc();

        thr_local_id = reg_thr_ids.uw(0);
        thr_local_size = reg_thr_ids.uw(1);
        thr_fused_id = reg_thr_ids.uw(2);
        thr_src_id = reg_thr_ids.uw(3);

        src_select = reg_thr_ids.uw(4);
        bia_select = reg_thr_ids.uw(5);
        load_next = reg_thr_ids.uw(6);
        kdhw = reg_thr_ids.uw(7);

        kw = reg_thr_ids.uw(8);
        kh = reg_thr_ids.uw(9);
        kd = reg_thr_ids.uw(10);

        osp = reg_thr_ids.ud(6);
        mb_g = reg_thr_ids.ud(7);

        reg_sp_ids[0] = ra.alloc();
        reg_sp_ids[1] = ra.alloc();
        reg_sp_ids[2] = ra.alloc();

        reg_sp_init = reg_sp_ids[0];
        reg_sp_max = reg_sp_ids[1];

        iw_init = reg_sp_init.d(0);
        ih_init = reg_sp_init.d(1);
        id_init = reg_sp_init.d(2);

        ow_init = reg_sp_init.d(4);
        oh_init = reg_sp_init.d(5);
        od_init = reg_sp_init.d(6);

        iw0 = reg_sp_max.d(0);
        ih0 = reg_sp_max.d(1);
        id0 = reg_sp_max.d(2);

        iw_max = reg_sp_max.d(4);
        ih_max = reg_sp_max.d(5);
        id_max = reg_sp_max.d(6);

        // r/w block offsets
        gmem_blk_off = ra.alloc();
        slm_blk_off = ra.alloc();

        // block ids
        block_ids = ra.alloc();
        icb_idx = block_ids.d(0);
        ocb_idx = block_ids.d(1);
        g_idx = block_ids.d(2);
        mb_idx = block_ids.d(3);

        store_slm_off = block_ids.d(4);

        // ptr offsets
        reg_tg_ptrs = ra.alloc();

        src_tg_ptr = reg_tg_ptrs.q(0);
        dst_tg_ptr = reg_tg_ptrs.q(1);
        wei_tg_off = reg_tg_ptrs.d(4);
        bia_tg_off = reg_tg_ptrs.d(5);

        src_blk_ptrs = ra.alloc();
        dst_blk_ptrs = ra.alloc();

        // weights
        wei_headers = GRFRange(a[0][0], 32); // reuse
        w = GRFRange(124, 32); // reuse
    }

    void allocate_registers_v2() {
        ra.release(getLocalID(0));

        reg_counters = ra.alloc();
        reg_local_sp = ra.alloc();

        count = reg_counters.d(0);
        max_count = reg_counters.d(1);
        blk_idx = reg_counters.d(2);

        ls_slm_buf_off = reg_counters.ud(4);

        iw = reg_local_sp.d(0);
        ih = reg_local_sp.d(1);
        id = reg_local_sp.d(2);

        ow = reg_local_sp.d(4);
        oh = reg_local_sp.d(5);
        od = reg_local_sp.d(6);
    }

    void init_gmem_offsets_tg_32n16c() {
        assert(src_blk_size == 1024);
        assert(dst_blk_size == 1024);
        assert(wei_blk_size == 1024);
        assert(bia_blk_size == 64);

        auto flag_sel = f0[0];

        auto tmp = tmp_regs;

        auto r_cb = tmp[0];
        auto s_icb = r_cb.ud(0);
        auto s_ocb = r_cb.ud(1);

        auto s_ng = tmp[1].ud(0);

        auto r_cb_max = tmp[2];
        auto s_icb_max = r_cb_max.ud(0);
        auto s_ocb_max = r_cb_max.ud(1);

        auto r_ngc = tmp[3];

        auto r_ngc_stride = tmp[4];
        auto s_ng_ic_stride = r_ngc_stride.ud(0);
        auto s_ng_oc_stride = r_ngc_stride.ud(1);

        auto r_tg_off = tmp[5];
        auto r_tg_off_lo = tmp[6];
        auto r_tg_off_hi = tmp[7];
        auto r_tg_ptr = GRF(src_tg_ptr.getBase());
        auto s_tg_src_ptr = r_tg_ptr.d(0);
        auto s_tg_dst_ptr = r_tg_ptr.d(2);

        auto r_pad = tmp[8];
        auto s_pad_x = r_pad.w(0);
        auto s_pad_y = r_pad.w(1);
        auto s_pad_z = r_pad.w(2);
        auto s_pad_a = r_pad.w(3);

        auto r_stride = tmp[9];
        auto s_stride_x = r_stride.w(0);
        auto s_stride_y = r_stride.w(2);
        auto s_stride_z = r_stride.w(4);
        auto s_stride_a = r_stride.w(6);

        auto r_dilate = tmp[10];
        auto s_dilate_x = r_dilate.w(0);
        auto s_dilate_y = r_dilate.w(1);
        auto s_dilate_z = r_dilate.w(2);
        auto s_dilate_a = r_dilate.w(3);

        auto r_k = tmp[11];
        auto r_sp0 = tmp[12];
        auto r_sp_init = tmp[13];

        auto r_sp_stride = tmp[14];
        auto s_src_sp_stride_x = r_sp_stride.d(0);
        auto s_src_sp_stride_y = r_sp_stride.d(1);
        auto s_src_sp_stride_z = r_sp_stride.d(2);
        auto s_src_sp_stride_a = r_sp_stride.d(3);
        auto s_dst_sp_stride_x = r_sp_stride.d(4);
        auto s_dst_sp_stride_y = r_sp_stride.d(5);
        auto s_dst_sp_stride_z = r_sp_stride.d(6);
        auto s_dst_sp_stride_a = r_sp_stride.d(7);

        auto s_go = tmp[15].d(0);
        auto s_goi = tmp[0].d(0);

        auto r_ptr = tmp[1];
        auto r_sp_off = tmp[2];

        auto r_carry = tmp[4];
        auto s_carry = tmp[4].ud(0);

        auto f_test_src_min = f0[0];
        auto f_test_src_max = f0[1];

        auto r_src_min = tmp[5];
        auto r_src_max = tmp[6];
        auto r_src_ok = tmp[7];

        emov_qq(s_tg_src_ptr.q(), arg_src_ptr.q());
        emov_qq(s_tg_dst_ptr.q(), arg_dst_ptr.q());

        const int icb_glob_bits = ngen::utils::log2(conf.ic_blk_wg) + 1;
        const int ocb_glob_bits = ngen::utils::log2(conf.oc_blk_wg) + 1;

        mad(1, s_ng.ud(), g_idx.uw(), mb_idx.ud(), uint16_t(conf.ngroups));
        and_(1, s_icb.ud(), icb_idx.ud(), uint32_t(~0u << icb_glob_bits));
        and_(1, s_ocb.ud(), ocb_idx.ud(), uint32_t(~0u << ocb_glob_bits));
        or_(2, r_cb.ud(), r_cb.ud(), thr_src_id.uw());

        mad(1, bia_tg_off.ud(), s_ocb.ud(), g_idx.uw(), uint16_t(conf.ocb));
        mul(1, bia_tg_off.ud(), bia_tg_off.ud(), uint16_t(bia_blk_size));

        mov(1, s_icb_max.ud(), uint32_t(conf.icb));
        mov(1, s_ocb_max.ud(), uint32_t(conf.ocb));

        mad(4, r_ngc.ud(), r_cb.ud(), s_ng.ud(), r_cb_max.uw(0)(2));

        mov(1, s_ng_ic_stride.ud(), uint32_t(src_sp * src_blk_size));
        mov(1, s_ng_oc_stride.ud(), uint32_t(dst_sp * dst_blk_size));

        mul(2, acc0.ud(), r_ngc.ud(), r_ngc_stride.uw(0)(2));
        mach(2, r_tg_off_hi.ud(), r_ngc.ud(), r_ngc_stride.ud());
        mov(2, r_tg_off_lo.ud(), acc0.ud());

        mov(2, r_tg_off.ud(0)(2), r_tg_off_lo.ud());
        mov(2, r_tg_off.ud(1)(2), r_tg_off_hi.ud());

        addc(4, r_tg_ptr.ud(), r_tg_ptr.ud(), r_tg_off.ud());
        mov(4, r_carry.ud(), acc0.ud(r_tg_ptr.ud().getOffset())(1));
        add(2, r_tg_ptr.ud(1)(2), r_tg_ptr.ud(1)(2), r_carry.ud(0)(2));

        mov(1, s_pad_x.w(), -int16_t(conf.l_pad));
        mov(1, s_pad_y.w(), -int16_t(conf.t_pad));
        mov(1, s_pad_z.w(), -int16_t(conf.f_pad));
        mov(1, s_pad_a.w(), 0);

        mov(1, s_dilate_x.w(), int16_t(1 + conf.dilate_w));
        mov(1, s_dilate_y.w(), int16_t(1 + conf.dilate_h));
        mov(1, s_dilate_z.w(), int16_t(1 + conf.dilate_d));
        mov(1, s_dilate_a.w(), 0);

        mov(4, r_k.w(), kw.w(0)(1));

        mov(1, s_stride_x.w(), int16_t(conf.stride_w));
        mov(1, s_stride_y.w(), int16_t(conf.stride_h));
        mov(1, s_stride_z.w(), int16_t(conf.stride_d));
        mov(1, s_stride_a.w(), 0);

        mad(4, r_sp0.w(), r_pad.w(), r_k.w(), r_dilate.w());
        mad(4, r_sp_init.d(0), r_sp0.w(), ow_init.d(0)(1), r_stride.w(0)(2));
        mov(4, r_sp_init.d(4), ow_init.d(0)(1));

        mov(1, s_src_sp_stride_x.d(), int32_t(src_blk_size));
        mov(1, s_src_sp_stride_y.d(), int32_t(conf.iw * src_blk_size));
        mov(1, s_src_sp_stride_z.d(),
                int32_t(conf.ih * conf.iw * src_blk_size));
        mov(1, s_src_sp_stride_a.d(), 0);

        mov(1, s_dst_sp_stride_x.d(), int32_t(dst_blk_size));
        mov(1, s_dst_sp_stride_y.d(), int32_t(conf.ow * dst_blk_size));
        mov(1, s_dst_sp_stride_z.d(),
                int32_t(conf.oh * conf.ow * dst_blk_size));
        mov(1, s_dst_sp_stride_a.d(), 0);

        mul(4, iw0.d(), r_sp_stride.d(), r_sp0.w());
        mul(8, acc0.d(), r_sp_stride.d(), r_sp_init.uw(0)(2));
        mach(8, null.d(), r_sp_stride.d(), r_sp_init.d());

        mov(8, r_sp_off.d(), acc0.d());
        mov(4, iw_init.d(), r_sp_off.d()); // save iw_init

        mov(1, flag_sel.uw(), src_select.uw());

        auto m_src = InstructionModifier() | flag_sel;
        auto m_dst = InstructionModifier() | ~flag_sel;

        add3(1 | m_src, r_sp_off.d(0), r_sp_off.d(0), r_sp_off.d(1),
                r_sp_off.d(2));
        add3(1 | m_dst, r_sp_off.d(0), r_sp_off.d(4), r_sp_off.d(5),
                r_sp_off.d(6));

        emov_qq(m_src, r_ptr.q(0), src_tg_ptr.q());
        emov_qq(m_dst, r_ptr.q(0), dst_tg_ptr.q());

        // weights
        mad(1, s_go.ud(), ocb_idx.uw(), g_idx.ud(), uint16_t(conf.ocb));
        mad(1, s_goi.ud(), icb_idx.uw(), s_go.ud(), uint16_t(conf.icb));

        mul(1, acc0.ud(), s_goi.ud(),
                uint32_t(conf.kw * conf.kh * conf.kd * wei_blk_size));
        mad(1, wei_tg_off.ud(), acc0.ud(), kdhw.uw(), uint32_t(wei_blk_size));

        // gmem headers
        mov(16, gmem_hdrs[0].ud(), 0);

        asr(1, r_sp_off.d(1), r_sp_off.d(0), 31);
        addc(2, r_ptr.ud(), r_ptr.ud(), r_sp_off.ud());
        mov(1, s_carry.ud(), acc0.ud());
        add(1, r_ptr.ud(1), r_ptr.ud(1), s_carry.ud());

        mov(1, iw_max.d(), int32_t(conf.iw * src_blk_size));
        mov(1, ih_max.d(), int32_t(conf.ih * conf.iw * src_blk_size));
        mov(1, id_max.d(), int32_t(conf.id * conf.ih * conf.iw * src_blk_size));

        emov_qq(gmem_hdrs[0].q(0), r_ptr.q(0));
        emov_qq(gmem_hdrs[1].q(0), r_ptr.q(0));
        or_(1, gmem_hdrs[1].uw(0), gmem_hdrs[1].uw(0), 256);

        cmp(4 | ge | f_test_src_min, r_src_min.ud(), iw_init.d(0)(1), 0);
        cmp(4 | lt | f_test_src_max, r_src_max.ud(), iw_init.d(0)(1),
                iw_max.d(0)(1));

        and_(4, r_src_ok.ud(), r_src_min.ud(), r_src_max.ud());
        and_(1, load_next.uw(), r_src_ok.uw(0), r_src_ok.uw(2));
        and_(1, load_next.uw(), load_next.uw(), r_src_ok.uw(4));
        or_(1, load_next.uw(), load_next.uw(), ~src_select.uw());
    }

    void init_slm_headers() {
        mov(1, f0[0].uw(), src_select.uw());

        mov(16, store_slm_hdrs[0].ud(), 0);

        auto tmp = tmp_regs;

        auto blk_sz = tmp[0];

        auto ld_icb = tmp[2];
        auto ld_ocb = tmp[3];

        auto fused_b = tmp[4];

        mov(1, blk_sz.ud(), uint16_t(slm_blk_sz_in_ow));

        mul(1, fused_b.ud(), thr_fused_id.uw(), 128 / 16); // 128b in ow

        const uint16_t slm_dst_off_in_ow
                = slm_blk_sz_in_ow * 2 * conf.ic_blk_wg;

        mul(1, icb_local.ud(), icb_local.uw(), blk_sz.uw());
        mad(1, ocb_local.ud(), uint16_t(slm_dst_off_in_ow), ocb_local.uw(),
                blk_sz.uw());

        mul(1, ld_icb.ud(), thr_src_id.uw(0)(0), blk_sz.uw());
        mad(1, ld_ocb.ud(), uint16_t(slm_dst_off_in_ow), thr_src_id.uw(0)(0),
                blk_sz.uw());

        add(1, ocb_local.ud(), ocb_local.ud(), fused_b.ud(0)(0));

        mov(1 | f0[0], store_slm_off.ud(), ld_icb.ud());
        mov(1 | ~f0[0], store_slm_off.ud(), ld_ocb.ud());

        mov(1, ls_slm_buf_off.ud(), uint32_t(2 * slm_buf_sz_in_ow));

        add(1, slm_src_load_off.d(), icb_local.d(),
                int32_t(2 * slm_buf_sz_in_ow));
        add(1, slm_dst_load_off.d(), ocb_local.d(),
                int32_t(2 * slm_buf_sz_in_ow));

        add3(1, store_slm_hdrs[0].ud(2), ls_slm_buf_off.ud(),
                store_slm_off.ud(), slm_blk_off.ud(0));
        add3(1, store_slm_hdrs[1].ud(2), ls_slm_buf_off.ud(),
                store_slm_off.ud(), slm_blk_off.ud(1));
    }

    void zero_acc() {
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 32; i += 2)
                mov(16, c[j][i].f(), 0.f);
        if (conf.with_bias) { mov(16, bia[0].f(), 0.f); }
    }

    void load_gmem() {
        auto f_test_src_min = f0[0];
        auto f_test_src_max = f0[1];
        auto f_do_load = f1[0];

        auto f_select = f0[0];
        auto f_next_x = f0[1];
        auto f_next_y = f1[0];
        auto f_next_z = f0[0];

        auto m_update_x = InstructionModifier() | f_next_x | any16h;
        auto m_update_y = InstructionModifier() | f_next_y | any16h;
        auto m_update_z = InstructionModifier() | f_next_z | any16h;

        auto m_is_src = InstructionModifier() | f_select;
        auto m_is_dst = InstructionModifier() | ~f_select;

        const int32_t src_next_x = conf.stride_w * src_blk_size;
        const int32_t src_next_y = conf.stride_h * conf.iw * src_blk_size;
        const int32_t src_next_z
                = conf.stride_d * conf.ih * conf.iw * src_blk_size;

        const int32_t dst_next_x = dst_blk_size;

        auto s_next_x = tmp_regs[0].ud(0);
        auto r_sp_off = tmp_regs[1];
        auto s_carry = tmp_regs[2].ud(0);

        auto r_src_min = tmp_regs[3];
        auto r_src_max = tmp_regs[4];
        auto r_src_ok = tmp_regs[5];

        Label skip_sp, skip_ih, skip_id;

        mov(1, f_do_load.uw(), load_next.uw());
        sync(SyncFunction::allwr, 0x0003); // clear sbid
        for (int i = 0; i < 16; i += 2) {
            mov(16 | ~f_do_load, s0[i].f(), 0);
        }

        sync(SyncFunction::nop, SWSB<float>(1));

        setDefaultAutoSWSB(false);

        load(16 | f_do_load | all16h | sb0, s0[0], block_hword(8), A64,
                gmem_hdrs[0]);
        load(16 | f_do_load | all16h | sb1, s0[8], block_hword(8), A64,
                gmem_hdrs[1]);

        setDefaultAutoSWSB(true);

        mov(1, f_select.uw(), src_select.uw());
        mov(1, f_next_x.uw(), 0);
        mov(1, f_next_y.uw(), 0);
        xor_(1 | eq | f_next_x, blk_idx.uw(), blk_idx.uw(), 1);
        xor_(1 | sb0.src, gmem_hdrs[0].uw(), gmem_hdrs[0].uw(), 512);
        xor_(1 | sb1.src, gmem_hdrs[1].uw(), gmem_hdrs[1].uw(), 512);

        add(1 | m_update_x, ow.d(), ow.d(), 1);
        add(1 | m_update_x, iw.d(), iw.d(), int32_t(src_next_x));
        mov(1 | m_is_src, s_next_x.ud(0), int32_t(src_next_x));
        mov(1 | m_is_dst, s_next_x.ud(0), int32_t(dst_next_x));
        cmp(16 | ge | f_next_y, ow.d(), int32_t(conf.ow));

        addc(1 | m_update_x, gmem_hdrs[0].ud(0), gmem_hdrs[0].ud(0),
                s_next_x.ud(0));
        mov(1 | m_update_x, s_carry.ud(), acc0.ud(0));
        add(1 | m_update_x, gmem_hdrs[0].ud(1), gmem_hdrs[0].ud(1),
                s_carry.ud());
        emov_qq(m_update_x, gmem_hdrs[1].q(0), gmem_hdrs[0].q(0));
        xor_(1 | m_update_x, gmem_hdrs[1].uw(0), gmem_hdrs[1].uw(0), 256);

        and_(1, f_next_y.uw(), f_next_y.uw(), src_select.uw());

        Label skip_y;
        if_(16 | m_update_y, skip_y);

        add(1, oh.d(), oh.d(), 1);
        add(1, ih.d(), ih.d(), int32_t(src_next_y));
        mov(1, ow.d(), 0);
        mov(1, iw.d(), iw0.d());

        cmp(16 | ge | f_next_z, oh.d(), int32_t(conf.oh));
        add(1 | m_update_z, od.d(), od.d(), 1);
        add(1 | m_update_z, id.d(), id.d(), int32_t(src_next_z));
        mov(1 | m_update_z, oh.d(), 0);
        mov(1 | m_update_z, ih.d(), ih0.d());

        add3(1, r_sp_off.d(0), iw.d(), ih.d(), id.d());
        asr(1, r_sp_off.d(1), r_sp_off.d(0), 31);
        addc(2, gmem_hdrs[0].ud(0), src_tg_ptr.ud(0)(1), r_sp_off.ud(0)(1));
        mov(1 | m_update_x, s_carry.ud(), acc0.ud(0));
        add(1, gmem_hdrs[0].ud(1), gmem_hdrs[0].ud(1), s_carry.ud(0));
        emov_qq(m_update_x, gmem_hdrs[1].q(0), gmem_hdrs[0].q(0));
        xor_(1 | m_update_x, gmem_hdrs[1].uw(0), gmem_hdrs[1].uw(0), 256);

        mark(skip_y);
        endif(16);

        cmp(4 | ge | f_test_src_min, r_src_min.ud(), iw.d(0)(1), 0);
        cmp(4 | lt | f_test_src_max, r_src_max.ud(), iw.d(0)(1),
                iw_max.d(0)(1));
        and_(4, r_src_ok.ud(), r_src_min.ud(), r_src_max.ud());
        and_(1, load_next.uw(), r_src_ok.uw(0), r_src_ok.uw(2));
        and_(1, load_next.uw(), load_next.uw(), r_src_ok.uw(4));
        or_(1, load_next.uw(), load_next.uw(), ~src_select.uw());
    }

    void multiply_32i32o(bool is_last = false) {
        auto f_reset = f0[1];

        auto src_hdrs = GRFRange(tmp_regs.getBase() + 0, 4);
        auto dst_hdrs = GRFRange(tmp_regs.getBase() + 4, 4);

        barrierwait();

        sync(SyncFunction::nop, SWSB<AllPipes>(1)); // flush all pipes
        sync(SyncFunction::allwr, 0x0ff0); // clear sbid

        mov(16, src_hdrs[0].ud(), 0);
        mov(16, src_hdrs[2].ud(), 0);
        mov(16, dst_hdrs[0].ud(), 0);
        mov(16, dst_hdrs[2].ud(), 0);

        mov(1, dst_hdrs[0].ud(2), slm_dst_load_off.ud());
        add(1, dst_hdrs[1].ud(2), slm_dst_load_off.ud(),
                uint32_t(1 * 256 / 16));
        add(1, dst_hdrs[2].ud(2), slm_dst_load_off.ud(),
                uint32_t(2 * 256 / 16));
        add(1, dst_hdrs[3].ud(2), slm_dst_load_off.ud(),
                uint32_t(3 * 256 / 16));

        load(16 | SWSB(sb4, 4), b[0][0], block_oword(8), SLM, dst_hdrs[0]);
        load(16 | SWSB(sb5, 3), b[0][4], block_oword(8), SLM, dst_hdrs[1]);
        load(16 | SWSB(sb6, 2), b[1][0], block_oword(8), SLM, dst_hdrs[2]);
        load(16 | SWSB(sb7, 1), b[1][4], block_oword(8), SLM, dst_hdrs[3]);

        mov(1, f_reset.uw(), 0);

        mov(1, src_hdrs[0].ud(2), slm_src_load_off.ud());
        add(1, src_hdrs[1].ud(2), slm_src_load_off.ud(),
                uint32_t(1 * 256 / 16));
        add(1, src_hdrs[2].ud(2), slm_src_load_off.ud(),
                uint32_t(2 * 256 / 16));
        add(1, src_hdrs[3].ud(2), slm_src_load_off.ud(),
                uint32_t(3 * 256 / 16));

        load(16 | SWSB(sb8, 4), a[0][0], block_oword(16), SLM, src_hdrs[0]);
        load(16 | SWSB(sb9, 3), a[0][8], block_oword(16), SLM, src_hdrs[1]);
        load(16 | SWSB(sb10, 2), a[1][0], block_oword(16), SLM, src_hdrs[2]);
        load(16 | SWSB(sb11, 1), a[1][8], block_oword(16), SLM, src_hdrs[3]);

        add(2 | lt | f_reset, slm_src_load_off.d(0)(1),
                slm_src_load_off.d(0)(1), -int16_t(slm_buf_sz_in_ow));
        add(1 | f_reset | any16h, slm_src_load_off.d(), icb_local.d(),
                int32_t(2 * slm_buf_sz_in_ow));
        add(1 | f_reset | any16h, slm_dst_load_off.d(), ocb_local.d(),
                int32_t(2 * slm_buf_sz_in_ow));

        // XXX: WA: disable auto scoreboard for dpas
        setDefaultAutoSWSB(false);

        sync(SyncFunction::allwr, 0x00f0); // wait b load
        multiply_half_block_32i32o(sb8, sb9, 0);
        multiply_half_block_32i32o(sb10, sb11, 1);

        setDefaultAutoSWSB(true);

        // skip last signal
        if (!is_last) { barriersignal(InstructionModifier() | sb13, r_bar); }
    }

    void store_slm() {
        Label end_src_reorder, end_dst_reorder;
        auto f_select = f1[0];
        auto f_reset = f0[1];

        mov(1, f_select.uw(), src_select.uw());

        add(1 | lt | f_reset, ls_slm_buf_off.d(), ls_slm_buf_off.d(),
                -int16_t(slm_buf_sz_in_ow));

        sync(SyncFunction::allwr, 0x0003); // wait gmem_load
        if_(16 | f_select, end_src_reorder, end_dst_reorder);

        reorder_uw_16n16c_to_2c8n8c2n(d0, s0);

        else_(16, end_dst_reorder, end_dst_reorder);
        mark(end_src_reorder);

        if (conf.with_bias) { compute_bias(s0); }
        reorder_uw_16n16c_to_16c16n(d0, s0);

        mark(end_dst_reorder);
        endif(16);

        sync(SyncFunction::allwr, 0x000c);
        store(16 | sb2, block_oword(16), SLM, store_slm_hdrs[0], d0[0]);
        store(16 | sb3, block_oword(16), SLM, store_slm_hdrs[1], d0[8]);

        mov(1 | f_reset, ls_slm_buf_off.d(), int16_t(2 * slm_buf_sz_in_ow));

        sync(SyncFunction::nop, SWSB<int>(1));

        add3(1 | sb2.src, store_slm_hdrs[0].ud(2), ls_slm_buf_off.ud(),
                store_slm_off.ud(), slm_blk_off.ud(0));
        add3(1 | sb3.src, store_slm_hdrs[1].ud(2), ls_slm_buf_off.ud(),
                store_slm_off.ud(), slm_blk_off.ud(1));
    }

    void eidiv_udud(const InstructionModifier &mod, const RegData &dst,
            const RegData &src, const RegData &tmp, const uint32_t divisor) {
        // unsigned dwords only
        if (src.getType() != DataType::ud || dst.getType() != DataType::ud)
            throw invalid_operand_exception();
        // SIMD16/32 is not supported
        if (mod.getExecSize() > 8) throw invalid_execution_size_exception();

        // only packed src/dst
        if (src.getHS() > 1
                || (src.getVS() > 0 && src.getWidth() != src.getVS()))
            throw invalid_region_exception();
        if (dst.getHS() > 1
                || (dst.getVS() > 0 && dst.getWidth() != dst.getVS()))
            throw invalid_region_exception();

        // division by 0
        if (divisor == 0) throw invalid_operand_exception();

        // division by 1
        if (divisor == 1) {
            mov(mod, dst, src);
            return;
        }

        // division by power of 2
        const bool is_power_of_2 = !(divisor & (divisor - 1));
        if (is_power_of_2) {
            const int log2_div = ngen::utils::bsf<uint32_t>(divisor);
            shr(mod, dst, src, log2_div);
            return;
        }

        // regular
        constexpr int max_prec = 32;

        uint64_t multiplier = 0;
        int sh_pre = 0, sh_post = 0;

        auto choose_multiplier = [&](uint32_t d, int prec) -> void {
            sh_post = ngen::utils::bsr<uint32_t>(d) + (is_power_of_2 ? 0 : 1);
            uint64_t k0 = 1ull << (sh_post + max_prec);
            uint64_t k1 = 1ull << (sh_post + max_prec - prec);
            uint64_t m_lo = k0 / d;
            uint64_t m_hi = (k0 + k1) / d;

            while ((m_lo >> 1) < (m_hi >> 1) && sh_post > 0) {
                m_lo >>= 1;
                m_hi >>= 1;
                sh_post--;
            }
            multiplier = m_hi;
            return;
        };

        auto mul_uh = [this](const InstructionModifier &mod, const RegData &dst,
                              const RegData &src, const uint32_t m) -> void {
            const uint16_t m_lo = m & 0xffffu;
            const int wo = dst.getOffset();

            mul(mod, acc0.ud(wo), src, m_lo);
            mach(mod, dst, src, m);
        };

        choose_multiplier(divisor, max_prec);

        const bool is_even = (divisor & 1) == 0;
        if (multiplier >= (1ull << 32) && is_even) {
            sh_pre = ngen::utils::bsf<uint32_t>(divisor & -int32_t(divisor));
            choose_multiplier(divisor >> sh_pre, max_prec - sh_pre);
        } else {
            sh_pre = 0;
        }
        const uint64_t uint_max_plus1 = 1ull << 32;
        if (multiplier >= uint_max_plus1) {
            assert(sh_pre == 0);
            multiplier -= uint_max_plus1;
            assert(multiplier < uint_max_plus1);

            GRF t1 = GRF(tmp.getBase()).ud();

            mul_uh(mod, t1, src, uint32_t(multiplier));
            add(mod, dst, src, -t1);
            shr(mod, dst, dst, 1);
            add(mod, dst, dst, t1);

            if (sh_post > 1) shr(mod, dst, dst, sh_post - 1);
        } else {
            if (sh_pre > 0) {
                shr(mod, dst, src, sh_pre);
                mul_uh(mod, dst, dst, uint32_t(multiplier));
            } else {
                mul_uh(mod, dst, src, uint32_t(multiplier));
            }
            shr(mod, dst, dst, sh_post);
        }
    }

    void emov_qq(const Subregister &dst, const Subregister &src) {
        emov_qq(InstructionModifier(), dst, src);
    }

    void emov_qq(const InstructionModifier &mod, const Subregister &dst,
            const Subregister &src) {
        if (src.getType() != DataType::q && src.getType() != DataType::uq)
            throw invalid_operand_exception();
        if (dst.getType() != DataType::q && dst.getType() != DataType::uq)
            throw invalid_operand_exception();

        mov(2 | mod, dst.d(0)(1), src.d(0)(1));
    }
};

status_t xe_hp_conv_bwd_weights_create_kernels(const conv_conf_t &conf,
        std::vector<compute::kernel_t> &kernels, gpu_primitive_t *primitive,
        engine_t *engine) {

    using namespace compute;

    std::unique_ptr<jit::jit_generator_base> jit_gen_wei_init;
    std::unique_ptr<jit::jit_generator_base> jit_gen_wei_cvt;
    std::unique_ptr<jit::jit_generator_base> jit_gen_convolution;

    auto compute_engine = utils::downcast<compute_engine_t *>(engine);
    auto device_info = compute_engine->device_info();

    switch (device_info->gpu_arch()) {
        case gpu_arch_t::xe_hp:
            jit_gen_wei_init.reset(
                    new xe_hp_conv_bwd_wei_init_kernel_t<gpu_xe_hp>(conf));
            jit_gen_wei_cvt.reset(
                    new xe_hp_conv_bwd_wei_cvt_kernel_t<gpu_xe_hp>(conf));
            jit_gen_convolution.reset(
                    new xe_hp_conv_bwd_wei_conv_kernel_t<gpu_xe_hp>(conf));
            break;
        case gpu_arch_t::xe_hpg:
            jit_gen_wei_init.reset(
                    new xe_hp_conv_bwd_wei_init_kernel_t<gpu_xe_hpg>(conf));
            jit_gen_wei_cvt.reset(
                    new xe_hp_conv_bwd_wei_cvt_kernel_t<gpu_xe_hpg>(conf));
            jit_gen_convolution.reset(
                    new xe_hp_conv_bwd_wei_conv_kernel_t<gpu_xe_hpg>(conf));
            break;
        default: return status::runtime_error;
    }

    kernels.resize(3);

    CHECK(primitive->create_kernel(engine, &kernels[0], *jit_gen_wei_init));
    CHECK(primitive->create_kernel(engine, &kernels[1], *jit_gen_convolution));
    if (conf.weights_data_type == data_type::bf16
            || conf.bias_data_type == data_type::bf16)
        CHECK(primitive->create_kernel(engine, &kernels[2], *jit_gen_wei_cvt));

    return status::success;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
