/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "jit_avx512_core_u8s8s32x_deconvolution.hpp"

#define GET_OFF(field) offsetof(jit_deconv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

using namespace nstl;

#define wht_blk_off(d, g, ...) \
        (conf_.with_groups() \
         ? (d).blk_off((g), __VA_ARGS__) \
         : (d).blk_off(__VA_ARGS__))

status_t jit_avx512_core_u8s8s32x_deconv_fwd_kernel::init_conf(jit_conv_conf_t &jcp,
        const deconvolution_desc_t &cd, cpu_memory_t::pd_t &src_pd,
        cpu_memory_t::pd_t &weights_pd, cpu_memory_t::pd_t &dst_pd,
        const bool with_bias, cpu_memory_t::pd_t &bias_pd,
        const primitive_attr_t &attr) {
    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper dst_d(&dst_pd);
    const memory_desc_wrapper weights_d(&weights_pd);
    const memory_desc_wrapper bias_d(&bias_pd);

    if (!(mayiuse(avx512_core) &&
            src_d.data_type() == data_type::u8
         && weights_d.data_type() == data_type::s8
         && one_of(dst_d.data_type(), data_type::f32, data_type::s32,
            data_type::s8, data_type::u8)))
        return status::unimplemented;

    jcp = zero<decltype(jcp)>();

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = src_d.dims()[1] / jcp.ngroups;
    jcp.is_depthwise = true && with_groups && utils::everyone_is(1,
            jcp.ic_without_padding, jcp.oc_without_padding);

    const auto w_format = with_groups
        ? (jcp.is_depthwise ? Goihw16g : gOIhw4i16o4i)
        : OIhw4i16o4i;

    if (dst_d.format() == any)
        CHECK(dst_pd.set_format(nhwc));
    if (dst_d.format() != nhwc)
        return status::unimplemented;
    if (src_d.format() == any)
        CHECK(src_pd.set_format(nhwc));
    if (src_d.format() != nhwc)
        return status::unimplemented;
    if (weights_d.format() == any)
        CHECK(weights_pd.set_format(w_format));
    if (weights_d.format() != w_format)
        return status::unimplemented;

    jcp.with_bias = with_bias;
    if (jcp.with_bias) {
        if (bias_d.format() == any)
            CHECK(bias_pd.set_format(x));
        if (bias_d.format() != x)
            return status::unimplemented;
    }

    jcp.ndims = dst_d.ndims();
    jcp.prop_kind = cd.prop_kind;
    jcp.mb = src_d.dims()[0];
    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2];
    jcp.ow = dst_d.dims()[3];
    jcp.kh = weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + 3];
    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];
    jcp.src_fmt = src_d.format();
    jcp.with_eltwise = false;/*TODO: support post-ops*/

    if (jcp.is_depthwise) {
        jcp.ch_block = 16;
        jcp.oc_block = 1;
        jcp.ic_block = 1;
    } else {
        jcp.ch_block = 1;
        jcp.oc_block = 16;
        jcp.ic_block = 16;

        if (jcp.ngroups == 1) {
            jcp.oc = utils::rnd_up(jcp.oc_without_padding, jcp.oc_block);
            jcp.ic = utils::rnd_up(jcp.ic_without_padding, jcp.ic_block);
        }
        if (jcp.ic % jcp.ic_block != 0)
            return status::unimplemented;
    }

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    if (!IMPLICATION(jcp.dilate_h, jcp.stride_h == 1)
            || !IMPLICATION(jcp.dilate_w, jcp.stride_w == 1))
            return status::unimplemented;

    /*bottom and right :padding*/
    jcp.b_pad = (jcp.ih - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.oh + jcp.t_pad - 1);
    jcp.r_pad = (jcp.iw - 1) * jcp.stride_w + (jcp.kw - 1) * (jcp.dilate_w + 1)
            - (jcp.ow + jcp.l_pad - 1);

    if (!attr.post_ops_.has_default_values())
        return status::unimplemented;

    jcp.ver = ver_avx512_core;
    if (mayiuse(avx512_core_vnni))
        jcp.ver = ver_vnni;
    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    jcp.dst_dt = dst_d.data_type();
    jcp.bia_dt = jcp.with_bias ? bias_d.data_type() : data_type::undef;
    jcp.typesize_bia = jcp.with_bias ? types::data_type_size(bias_d.data_type()) : 0;
    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());

    jcp.nb_ch = div_up(jcp.ngroups, jcp.ch_block);
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    /*kernel blocking params*/
    const int regs = jcp.ver == ver_vnni ? 31 : 29;
    jcp.nb_oc_blocking = nstl::min(4, jcp.nb_oc);
    for (; jcp.nb_oc_blocking > 1; jcp.nb_oc_blocking--)
        if (jcp.nb_oc % jcp.nb_oc_blocking == 0
                && jcp.l_pad <= regs / (jcp.nb_oc_blocking + 1))
            break;

    jcp.ur_w = regs / (jcp.nb_oc_blocking + 1);
    int l_overflow = max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.l_pad) / jcp.stride_w);
    int r_overflow = max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                     - max(0, jcp.r_pad)) / jcp.stride_w);
    if (jcp.ow < jcp.ur_w)
        jcp.ur_w = jcp.ow;
    for (; jcp.ur_w > 1; jcp.ur_w--)
        if (jcp.ur_w % jcp.stride_w == 0
                && max(l_overflow,
                    r_overflow - (jcp.ow % jcp.ur_w) / jcp.stride_w) * jcp.stride_w <= jcp.ur_w)
            break;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    jcp.loop_order = jcp.ngroups > 1 ? loop_ngc : loop_cgn;
    return status::success;
}

void jit_avx512_core_u8s8s32x_deconv_fwd_kernel::compute_ker(
        int ur_w, int l_overflow, int r_overflow, ker_block_t last_block) {

    int ch_block_all = jcp.ch_block * jcp.ic_block * jcp.oc_block;
    int shift_src_ih = jcp.typesize_in * (jcp.dilate_h + 1)
        * jcp.iw * jcp.ngroups * jcp.ic_without_padding;
    int shift_filt_kh = jcp.typesize_in *  jcp.kw * jcp.stride_h * ch_block_all;

    auto src_offset = [=] (int oj, int icb, int ki) {
         return jcp.typesize_in *
           (((oj + jcp.l_pad - ki * (jcp.dilate_w + 1)) / jcp.stride_w) * jcp.ngroups * jcp.ic_without_padding + icb * 4);
    };

    auto kernel_offset = [=] (int ocb, int icb, int ki) {
        return jcp.typesize_in *
            (ocb * jcp.nb_ic * jcp.kh * jcp.kw * ch_block_all + icb * jcp.oc_block * jcp.ic_block/4
             + ki * ch_block_all);
    };

    auto compute = [=](zmm_t vreg_acc, zmm_t vreg_wei, zmm_t vreg_src) {
        if (jcp.ver == ver_vnni) {
            vpdpbusd(vreg_acc, vreg_src, vreg_wei);
        } else if (jcp.is_depthwise) {
            vpmulld(zmm_tmp, vreg_src, vreg_wei);
            vpaddd(vreg_acc, vreg_acc, zmm_tmp);
        } else {
            vpmaddubsw(zmm_tmp, vreg_src, vreg_wei);
            vpmaddwd(zmm_tmp, zmm_tmp, zmm_one);
            vpaddd(vreg_acc, vreg_acc, zmm_tmp);
        }
    };

    mov(aux_reg_src, reg_src);
    mov(aux_reg_filt, reg_filt);
    mov(reg_kj, reg_kh);
    Xbyak::Label kh_loop_label;
    L(kh_loop_label); {
       for (int ki = 0; ki < jcp.kw; ki++) {
           int jj_start = get_ow_start(ki, l_overflow);
           int jj_end = get_ow_end(ur_w, ki, r_overflow);
           int tail_size = jcp.ic_without_padding % 4;
           int n_ic_blocks = jcp.is_depthwise
                           ? 1
                           : (last_block &  ~no_last_block
                                   ? div_up(jcp.ic_without_padding % jcp.ic_block, 4)
                                   : jcp.ic_block / 4);
           for (int icb1 = 0; icb1 < n_ic_blocks; icb1++) {
               for (int jj = jj_start; jj < jj_end; jj += jcp.stride_w) {
                    assert((jj + jcp.l_pad - ki) % jcp.stride_w == 0);

                   int aux_src_off = src_offset(jj, icb1, ki);
                   if (jcp.is_depthwise) {
                       vpmovzxbd(zmm_inp(jj, jcp.nb_oc_blocking),
                                   EVEX_compress_addr(aux_reg_src, aux_src_off));
                   } else if ((last_block & last_sp_block)
                           && tail_size != 0 && icb1 == n_ic_blocks - 1) {
                       xmm_t xmm_tmp = xmm_t(zmm_inp(jj, jcp.nb_oc_blocking).getIdx());
                       for (int r = 0; r < tail_size; ++r)
                           vpinsrb(xmm_tmp, xmm_tmp,
                                   ptr[aux_reg_src + aux_src_off + r], r);
                       vpbroadcastd(zmm_inp(jj, jcp.nb_oc_blocking), xmm_tmp);
                   } else {
                       vpbroadcastd(zmm_inp(jj, jcp.nb_oc_blocking),
                               EVEX_compress_addr(aux_reg_src, aux_src_off));
                   }
               }

               for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
                   int aux_filt_off = kernel_offset(ocb, icb1, ki);
                   if (jj_end - jj_start > 0) {
                       if (jcp.is_depthwise)
                           vpmovsxbd(zmm_wei,
                               EVEX_compress_addr(aux_reg_filt, aux_filt_off));
                       else
                           vmovups(zmm_wei,
                                   EVEX_compress_addr(aux_reg_filt, aux_filt_off));
                   }
                   for (int jj = jj_start; jj < jj_end; jj += jcp.stride_w) {
                       compute(zmm_out(jj, ocb),
                               zmm_wei, zmm_inp(jj, jcp.nb_oc_blocking));
                   }
               }
           }
       }
       sub(aux_reg_src, shift_src_ih);
       add(aux_reg_filt, shift_filt_kh);
       dec(reg_kj);
       cmp(reg_kj, 0);
       jg(kh_loop_label, T_NEAR);
    }
}

void jit_avx512_core_u8s8s32x_deconv_fwd_kernel::prepare_output(int ur_w) {
    for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
        for (int ur = 0; ur < ur_w; ur++) {
                zmm_t zmm = zmm_out(ur, ocb);
                vpxord(zmm, zmm, zmm);
        }
    }
}

void jit_avx512_core_u8s8s32x_deconv_fwd_kernel::cvt2ps(data_type_t type_in,
        zmm_t zmm_in, const Xbyak::Operand &op, bool mask_flag) {
    zmm_t zmm = mask_flag ? zmm_in | ktail_mask | T_z : zmm_in;
    switch (type_in) {
    case data_type::f32:
    case data_type::s32: vmovups(zmm, op); break;
    case data_type::s8: vpmovsxbd(zmm, op); break;
    case data_type::u8: vpmovzxbd(zmm, op); break;
    default: assert(!"unsupported data type");
    }
    if (type_in != data_type::f32)
        vcvtdq2ps(zmm_in, zmm_in);
}

void jit_avx512_core_u8s8s32x_deconv_fwd_kernel::store_output(int ur_w, bool last_oc_block) {
    mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
    mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);

    vpxord(zmm_zero, zmm_zero, zmm_zero);
    for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
        const bool mask_flag = last_oc_block && ocb == jcp.nb_oc_blocking - 1;
        int scale_offset = jcp.is_oc_scale * (sizeof(float) * ocb * jcp.oc_block);

        auto zmm_bias = zmm_tmp;
        if (jcp.with_bias) {
            int bias_offset = jcp.typesize_bia * ocb * jcp.oc_block;
            auto bias_addr = EVEX_compress_addr(reg_bias, bias_offset);
            cvt2ps(jcp.bia_dt, zmm_bias, bias_addr, mask_flag);
        }

        for (int ur = 0; ur < ur_w; ur++) {
            zmm_t zmm = zmm_out(ur, ocb);
            vcvtdq2ps(zmm, zmm);
            if (jcp.with_bias) vaddps(zmm, zmm, zmm_bias);
            zmm_t mask_zmm = mask_flag
                           ? zmm | ktail_mask | T_z
                           : zmm;
            vmulps(mask_zmm, zmm,
                    EVEX_compress_addr(reg_ptr_scales, scale_offset));

            if (jcp.dst_dt == data_type::u8) vmaxps(zmm, zmm_zero, zmm);

            if (jcp.dst_dt != data_type::f32) {
                if (attr_.round_mode_ == round_mode::nearest)
                    vcvtps2dq(zmm | T_rn_sae, zmm);
                else if (attr_.round_mode_ == round_mode::down)
                    vcvtps2dq(zmm | T_rd_sae, zmm);
                else
                    assert(!"unimplemented");
            }
        }
        for (int ur = 0; ur < ur_w; ur++) {
            int aux_dst_off = jcp.typesize_out
                * (ur * jcp.ngroups * jcp.oc_without_padding + ocb * jcp.oc_block);
            auto addr = EVEX_compress_addr(reg_dst, aux_dst_off);

            zmm_t zmm = zmm_out(ur, ocb);
            zmm_t r_zmm = mask_flag
                        ? zmm | ktail_mask
                        : zmm;
            switch (jcp.dst_dt) {
            case data_type::f32:
            case data_type::s32: vmovups(addr, r_zmm); break;
            case data_type::s8: vpmovsdb(addr, r_zmm); break;
            case data_type::u8: vpmovusdb(addr, r_zmm); break;
            default: assert(!"unknown dst_dt");
            }
        }
    }
}

void jit_avx512_core_u8s8s32x_deconv_fwd_kernel::compute_loop(
        int ur_w, int l_overflow, int r_overflow, bool is_last_sp_block) {

    int shift_src_icb = jcp.typesize_in * jcp.ic_block;
    int shift_filt_icb = jcp.typesize_in * jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block;

    prepare_output(ur_w);

    Xbyak::Label icb_loop_label;
    mov(reg_icb, jcp.nb_ic);
    L(icb_loop_label); {

        if (jcp.ic_without_padding != jcp.ic) {
            Xbyak::Label common_ker, end_ker;
            cmp(reg_icb, 1);
            jg(common_ker, T_NEAR);

            compute_ker(ur_w, l_overflow, r_overflow,
                    is_last_sp_block ? last_sp_block : last_ic_block);
            jmp(end_ker, T_NEAR);

            L(common_ker);
            compute_ker(ur_w, l_overflow, r_overflow, no_last_block);

            L(end_ker);
        } else {
            compute_ker(ur_w, l_overflow, r_overflow, no_last_block);
        }

        add(reg_src, shift_src_icb);
        add(reg_filt, shift_filt_icb);
        dec(reg_icb);
        cmp(reg_icb, 0);
        jg(icb_loop_label, T_NEAR);
    }
    sub(reg_src, jcp.nb_ic * shift_src_icb);
    sub(reg_filt, jcp.nb_ic * shift_filt_icb);

    if (jcp.ngroups % jcp.ch_block != 0 || jcp.oc_without_padding != jcp.oc) {
        Xbyak::Label common_store, end_store;
        mov(reg_oc_blocks, ptr[param1 + GET_OFF(oc_blocks)]);
        if (jcp.is_depthwise)
            cmp(reg_oc_blocks, jcp.nb_ch - 1);
        else
            cmp(reg_oc_blocks, jcp.nb_oc - jcp.nb_oc_blocking);
        jne(common_store, T_NEAR);

        store_output(ur_w, true);
        jmp(end_store, T_NEAR);

        L(common_store);
        store_output(ur_w, false);

        L(end_store);

    } else {
        store_output(ur_w, false);
    }
}

void jit_avx512_core_u8s8s32x_deconv_fwd_kernel::generate() {
    preamble();

    Xbyak::Reg16 _t = reg_scratch.cvt16();
    mov(_t, 0x1);
    vpbroadcastw(zmm_one, _t);

    if (jcp.ngroups % jcp.ch_block != 0 || jcp.oc_without_padding != jcp.oc) {
        int tail_size = jcp.is_depthwise
            ? jcp.ngroups % jcp.ch_block
            : jcp.oc_without_padding % jcp.oc_block;
        int mask = (1 << tail_size) - 1;
        Xbyak::Reg32 regw_tmp = reg_nur_w.cvt32();
        mov(regw_tmp, mask);
        kmovw(ktail_mask, regw_tmp);
    }

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_filt, ptr[param1 + GET_OFF(filt)]);
    mov(reg_dst, ptr[param1 + GET_OFF(dst)]);
    mov(reg_kh, ptr[param1 + GET_OFF(kh_padding)]);

    int dst_shift = jcp.typesize_out * jcp.ur_w * jcp.ngroups * jcp.oc_without_padding;
    int src_shift = jcp.typesize_in * (jcp.ur_w / jcp.stride_w) * jcp.ngroups * jcp.ic_without_padding;

    int l_overflow = max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.l_pad) / jcp.stride_w);
    int r_overflow = max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                     - max(0, jcp.r_pad)) / jcp.stride_w);

    int r_overflow1 = nstl::max(0, ((jcp.kw -1) * (jcp.dilate_w + 1)
                - nstl::max(0, jcp.r_pad) - jcp.ur_w_tail) / jcp.stride_w);
    int nur_w = jcp.ow / jcp.ur_w;
    if (r_overflow1 > 0) nur_w--;

    if (jcp.ur_w == jcp.ow) {
        compute_loop(jcp.ur_w, l_overflow, r_overflow, true);
    } else if (nur_w == 0) {
        compute_loop(jcp.ur_w, l_overflow, r_overflow1, jcp.ur_w_tail == 0);
        add(reg_src, src_shift);
        add(reg_dst, dst_shift);
        if (jcp.ur_w_tail != 0)
            compute_loop(jcp.ur_w_tail, 0, r_overflow, true);
    } else {
        xor_(reg_nur_w, reg_nur_w);
        if (l_overflow > 0) {
            compute_loop(jcp.ur_w, l_overflow, 0, false);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
            inc(reg_nur_w);
        }
        if ((l_overflow <= 0 && nur_w > 0)
                || (l_overflow > 0 && nur_w > 1)) {
            Xbyak::Label ow_loop_label;
            L(ow_loop_label); {
                compute_loop(jcp.ur_w, 0, 0, false);
                add(reg_src, src_shift);
                add(reg_dst, dst_shift);
                inc(reg_nur_w);
                cmp(reg_nur_w, nur_w);
                jl(ow_loop_label, T_NEAR);
            }
        }
        if (r_overflow1 > 0) {
            compute_loop(jcp.ur_w, 0, r_overflow1, jcp.ur_w_tail == 0);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
        }
        if (jcp.ur_w_tail != 0) {
            compute_loop(jcp.ur_w_tail, 0, r_overflow, true);
        }
    }
    postamble();
}

template <data_type_t dst_type>
void _jit_avx512_core_u8s8s32x_deconvolution_fwd_t<dst_type>::
execute_forward()
{
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    auto &jcp = kernel_->jcp;

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int nb_groups = jcp.nb_ch;

    size_t src_h_stride = src_d.blk_off(0, 0, 1);
    size_t dst_h_stride = dst_d.blk_off(0, 0, 1);
    size_t wht_kh_stride = wht_blk_off(weights_d, 0, 0, 0, 1);

    const auto &oscales = conf_.attr()->output_scales_;

    parallel(0,
            [&](const int ithr, const int nthr) {
            int start{0}, end{0};
            int work_amount = jcp.mb * nb_groups * oc_chunks * jcp.oh;
            balance211(work_amount, nthr, ithr, start, end);

            auto p = jit_deconv_call_s();

            /*loop order = cgn*/
            int n{0}, g{0}, occ{0}, oh_s{0};
            if (jcp.loop_order == loop_ngc)
                nd_iterator_init(start, n, jcp.mb, g, nb_groups, occ, oc_chunks,
                    oh_s, jcp.oh);
            else if (jcp.loop_order == loop_cgn)
                nd_iterator_init(start, occ, oc_chunks, g, nb_groups, n, jcp.mb,
                    oh_s, jcp.oh);
            else
                assert(!"unsupported loop order");
            while (start < end) {

                int ocb = occ * jcp.nb_oc_blocking;
                int g_oc = (g * jcp.ch_block * jcp.nb_oc + ocb) * jcp.oc_block;
                int g_ic = g * jcp.ch_block * jcp.ic;
                int work_rem = end - start;
                int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;

                auto dst_w = dst + dst_d.blk_off(n, g_oc);
                auto src_w = src + src_d.blk_off(n, g_ic);
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb, 0);
                auto bias_w = jcp.with_bias
                            ? bias + (bias_d.blk_off(g_oc) * jcp.typesize_bia)
                            : 0;

                auto scales = &oscales.scales_[jcp.is_oc_scale * g_oc];
                for (int oj = oh_s; oj < oh_e; oj++) {
                    int ih_max, kh_lo, kh_len;
                    if (jcp.dilate_h != 0 && jcp.stride_h == 1) {
                            int dilate_h = jcp.dilate_h + 1;
                            // Note: use div_up to account for "holes" in filter
                            int o_t_overflow
                                = div_up(max(0, (jcp.kh - 1) * dilate_h
                                        - oj - jcp.t_pad), dilate_h);
                            int o_b_overflow
                                = div_up(max(0, (jcp.kh - 1) * dilate_h + 1
                                        - jcp.ih + oj - jcp.b_pad), dilate_h);
                            kh_len = jcp.kh - o_t_overflow - o_b_overflow;
                            kh_lo = o_b_overflow;
                            ih_max = oj + jcp.t_pad - o_b_overflow * dilate_h;
                    } else {
                        int o_t_overflow = max(0,
                                (jcp.kh - (oj + 1 + jcp.t_pad)) / jcp.stride_h); 
                        int o_b_overflow = max(0,
                                ((oj + 1 + jcp.kh - 1)
                                 - (jcp.oh + jcp.b_pad)) / jcp.stride_h);
                        int overflow_kh_hi = jcp.kh - 1
                            - abs(jcp.oh + jcp.b_pad - (oj + 1)) % jcp.stride_h;
                        int overflow_kh_lo = ((oj + 1 + jcp.t_pad) - 1) % jcp.stride_h;

                        kh_len = (overflow_kh_hi - overflow_kh_lo) / jcp.stride_h
                            + 1 - o_t_overflow - o_b_overflow;
                        kh_lo = overflow_kh_lo + o_b_overflow * jcp.stride_h;
                        ih_max = (oj + jcp.t_pad - kh_lo) / jcp.stride_h;
                    }

                    p.src = src_w + ih_max * src_h_stride;
                    p.dst = dst_w + oj * dst_h_stride;
                    p.filt = wht_w + kh_lo * wht_kh_stride;
                    p.bias = bias_w;
                    p.kh_padding = kh_len;
                    p.scales = scales;
                    p.oc_blocks = jcp.is_depthwise ? g : ocb;
                    kernel_->jit_ker(&p);
                }
                if (jcp.loop_order == loop_ngc)
                    nd_iterator_jump(start, end,
                            n, jcp.mb, g, nb_groups, occ, oc_chunks, oh_s, jcp.oh);
                else if (jcp.loop_order == loop_cgn)
                    nd_iterator_jump(start, end,
                            occ, oc_chunks, g, nb_groups, n, jcp.mb, oh_s, jcp.oh);
                else
                    assert(!"unsupported loop order");
            }
    });
}

template struct _jit_avx512_core_u8s8s32x_deconvolution_fwd_t<data_type::u8>;
template struct _jit_avx512_core_u8s8s32x_deconvolution_fwd_t<data_type::s8>;
template struct _jit_avx512_core_u8s8s32x_deconvolution_fwd_t<data_type::f32>;
template struct _jit_avx512_core_u8s8s32x_deconvolution_fwd_t<data_type::s32>;
}
}
}
