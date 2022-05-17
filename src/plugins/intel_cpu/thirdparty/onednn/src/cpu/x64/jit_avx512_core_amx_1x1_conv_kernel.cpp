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

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_avx512_core_amx_1x1_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::utils;
using namespace Xbyak;

jit_avx512_core_amx_1x1_fwd_kernel_t::jit_avx512_core_amx_1x1_fwd_kernel_t(
        const jit_conv_conf_t &ajcp, const primitive_attr_t &attr,
        const memory_desc_t &dst_md)
    : jit_generator(nullptr, MAX_CODE_SIZE, true, avx512_core_amx)
    , jcp(ajcp)
    , attr_(attr) {
    if (jcp.with_eltwise || jcp.with_binary || jcp.with_sum || jcp.with_depthwise || jcp.with_quantization) {
        using namespace binary_injector;
        const auto &rhs_addr_reg = bin_injector_helper_reg_1;
        const auto &rhs_helper_reg = bin_injector_helper_reg_2;
        static constexpr bool preserve_gpr = false;
        static constexpr bool preserve_vmm = false;
        const size_t tail_size = jcp.oc_without_padding % isa_simd_width_;
        static constexpr bool use_exact_tail_scalar_bcast = true;

        const rhs_arg_static_params_t rhs_arg_static_params {31, rhs_addr_reg,
                rhs_helper_reg, preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec),
                memory_desc_wrapper(dst_md), tail_size, ktail_mask,
                use_exact_tail_scalar_bcast};
        const static_params_t static_params {
                this->param1, rhs_arg_static_params};
        quantization_injector::static_params_t quantization_static_params =
                {zmm_d_weights.getIdx(), zmm_d_bias.getIdx(), reg_d_weights, reg_d_bias};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<avx512_core>>(
                this, jcp.post_ops, static_params, quantization_static_params);
    }
}

// Tile register decomposition
int jit_avx512_core_amx_1x1_fwd_kernel_t::get_out_tensor(int h, int i) const {
    return C_BASE + h * jcp.nb_os_blocking + i;
}
int jit_avx512_core_amx_1x1_fwd_kernel_t::get_inp_tensor(int h) const {
    return I_BASE + h;
}
int jit_avx512_core_amx_1x1_fwd_kernel_t::get_wei_tensor(int i) const {
    return W_BASE + i;
}

bool jit_avx512_core_amx_1x1_fwd_kernel_t::is_bf16() const {
    return jcp.src_dt == data_type::bf16;
}

// Code generation
void jit_avx512_core_amx_1x1_fwd_kernel_t::init_runtime_counters() {
    row_count_ = 0;
    buf_count_ = 0;
    is_store_done_ = false;
    is_buffer_empty_ = true;
}

size_t jit_avx512_core_amx_1x1_fwd_kernel_t::out_h_shift() const {
    return (size_t)jcp.ow * jcp.ngroups * jcp.oc_without_padding;
}

size_t jit_avx512_core_amx_1x1_fwd_kernel_t::out_w_shift() const {
    return (size_t)jcp.ngroups * jcp.oc_without_padding;
}

size_t jit_avx512_core_amx_1x1_fwd_kernel_t::inp_offset(
        int h, int w, int icb) const {
    return (size_t)jcp.typesize_in
            * (h * jcp.iw * jcp.ngroups * jcp.ic_without_padding
                    + w * jcp.ngroups * jcp.ic_without_padding
                    + icb * jcp.ic_block_int_np);
}

size_t jit_avx512_core_amx_1x1_fwd_kernel_t::out_row_offset(
        int h, int w, int ocb) const {
    return (size_t)jcp.typesize_out
            * (h * jcp.ow * jcp.ngroups * jcp.oc_without_padding
                    + w * jcp.ngroups * jcp.oc_without_padding
                    + ocb * jcp.oc_block);
}

void jit_avx512_core_amx_1x1_fwd_kernel_t::update_buffer_pointers() {
    auto buffer_offset = [=](bool shift) { return ((buf_count_ + shift) % 2); };
    int wsp_shift = jcp.typesize_acc * (jcp.wsp_buffer_size / 2);

    int postop_shift = wsp_shift * buffer_offset(true);

    mov(reg_postop, wsp_ptr);
    add(reg_postop, postop_shift);

    buf_count_++;
}

void jit_avx512_core_amx_1x1_fwd_kernel_t::interleave_store() {
    int scnd_dim = jcp.nb_os_blocking * jcp.tile_width;

    for (int c = 0;
            c < jcp.per_one_pstore && !is_store_done_ && !is_buffer_empty_;
            c++) {
        int ocb = (row_count_ / scnd_dim);
        int osb = (row_count_ % scnd_dim) / jcp.tile_width;
        int row = (row_count_ % scnd_dim) % jcp.tile_width;

        const Zmm zmm_r = zmm_out(row);

        int oh = ((osb * jcp.tile_width + row) / jcp.ow);
        int ow = ((osb * jcp.tile_width + row) % jcp.ow);

        {
            // preserve registers used by binary post_ops injector
            const injector_utils::conditional_register_preserve_guard_t
                    cond_register_guard(jcp.with_binary, this,
                            {bin_injector_helper_reg_1,
                                    bin_injector_helper_reg_2});
            const int wsp_row_offset = jcp.typesize_acc
                    * (osb * jcp.nb_oc_blocking * jcp.max_width * jcp.oc_block
                            + ocb * jcp.max_width * jcp.oc_block
                            + row * jcp.oc_block);

            vmovups(zmm_r, ptr[reg_postop + wsp_row_offset]);
            store_output_vector(zmm_r, ocb, oh, ow);
            row_count_++;
        }

        int exp_row_count
                = jcp.tile_width * jcp.nb_oc_blocking * jcp.nb_os_blocking;
        if (row_count_ == exp_row_count) {
            int oh = ((jcp.nb_os_blocking * jcp.tile_width) / jcp.ow);
            int ow = ((jcp.nb_os_blocking * jcp.tile_width) % jcp.ow);
            size_t out_offset = jcp.typesize_out
                    * (oh * out_h_shift() + ow * out_w_shift());
            add(out_ptr, out_offset);
            row_count_ = 0;
            is_store_done_ = true;
        }
    }
}

Ymm jit_avx512_core_amx_1x1_fwd_kernel_t::ymm_mask(
        const Ymm &ymm_in, bool mask_flag, bool store) {
    return mask_flag ? (store ? ymm_in | ktail_mask : ymm_in | ktail_mask | T_z)
                     : ymm_in;
}

Zmm jit_avx512_core_amx_1x1_fwd_kernel_t::zmm_mask(
        const Zmm &zmm_in, bool mask_flag, bool store) {
    return mask_flag ? (store ? zmm_in | ktail_mask : zmm_in | ktail_mask | T_z)
                     : zmm_in;
}

void jit_avx512_core_amx_1x1_fwd_kernel_t::cvt2ps(data_type_t type_in,
        const Zmm &zmm_in, const Operand &op, bool mask_flag = false) {
    using namespace dnnl::impl::data_type;
    const Zmm zmm = zmm_mask(zmm_in, mask_flag);
    switch (type_in) {
        case bf16:
            vpmovzxwd(zmm, op);
            vpslld(zmm_in, zmm_in, 16);
            break;
        case f32:
        case s32: vmovups(zmm, op); break;
        case s8: vpmovsxbd(zmm, op); break;
        case u8: vpmovzxbd(zmm, op); break;
        default: assert(!"unsupported data type");
    }
    if (utils::one_of(type_in, s32, s8, u8)) vcvtdq2ps(zmm_in, zmm_in);
}

void jit_avx512_core_amx_1x1_fwd_kernel_t::apply_sum(const Zmm &zmm_out,
        const float *p_sum_scale, const int32_t *p_sum_zp,
        const Xbyak::Address &addr, const bool mask_flag) {
    if (p_sum_scale) {
        const auto p_sum_scale_val = *p_sum_scale;
        const auto p_sum_zp_val = *p_sum_zp;
        const auto sum_injector = [&, p_sum_scale_val, p_sum_zp_val,
                                          mask_flag]() {
            cvt2ps(jcp.sum_dt, zmm_prev_dst, addr, mask_flag);
            if (p_sum_zp_val != 0) {
                vcvtdq2ps(zmm_sum_zp, ptr_b[reg_ptr_sum_zp]);
                vsubps(zmm_prev_dst, zmm_sum_zp);
            }
            if (p_sum_scale_val == 1.f)
                vaddps(zmm_out, zmm_prev_dst);
            else
                vfmadd231ps(zmm_out, zmm_prev_dst, zword_b[reg_ptr_sum_scale]);
        };
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);
    }
}

void jit_avx512_core_amx_1x1_fwd_kernel_t::apply_postops(const Zmm &zmm_out,
        const float *p_sum_scale, const int32_t *p_sum_zp,
        const Xbyak::Address &addr, const bool mask_flag, const size_t off,
        const int ocb) {
    if (jcp.with_eltwise || jcp.with_binary
            || (jcp.with_sum && p_sum_scale != nullptr) || jcp.with_depthwise || jcp.with_quantization) {
        std::map<size_t, int> vmm_idx_off;
        vmm_idx_off.insert({zmm_out.getIdx(), ocb * jcp.oc_block * sizeof(float)});
        depthwise_injector::dynamic_params_t ddp {zmm_d_weights.getIdx(), zmm_d_bias.getIdx(), reg_d_weights, reg_d_bias,
                                                  ptr[this->param1 + GET_OFF(oc_off)], vmm_idx_off, this->rsp};
        quantization_injector::dynamic_params_t qdp {ptr[this->param1 + GET_OFF(oc_off)], vmm_idx_off, jcp.dst_dt, this->rsp};

        binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;

        apply_sum(zmm_out, p_sum_scale, p_sum_zp, addr, mask_flag);

        const auto vmm_idx = zmm_out.getIdx();
        if (jcp.with_binary) {
            const int oc_l_offset = ocb * jcp.oc_block;
            rhs_arg_params.vmm_idx_to_oc_elem_off_addr.emplace(
                    vmm_idx, ptr[param1 + GET_OFF(oc_l_off)]);
            rhs_arg_params.vmm_idx_to_oc_elem_off_val.emplace(
                    vmm_idx, oc_l_offset);
            rhs_arg_params.vmm_idx_to_out_off_oprnd.emplace(
                    vmm_idx, out_off_oprnd);
            rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                    vmm_idx, static_cast<int>(off) / jcp.typesize_out);
            if (mask_flag) rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);

            const injector_utils::register_preserve_guard_t register_guard(
                    this, {out_off_oprnd});
            mov(out_off_oprnd, out_ptr);
            sub(out_off_oprnd, ptr[param1 + GET_OFF(dst_orig)]);
            shr(out_off_oprnd, std::log2(types::data_type_size(jcp.dst_dt)));
        }

        postops_injector_->compute_vector_range({(size_t)vmm_idx}, rhs_arg_params, ddp, qdp);
    }
}

bool jit_avx512_core_amx_1x1_fwd_kernel_t::is_fast_postops(
        const jit_conv_conf_t &jcp) {
    const auto &p = jcp.post_ops;
    auto is_relu = [&](int idx) { return p.entry_[idx].is_relu(); };
    auto is_sum = [&](int idx) {
        const bool require_scale_one = jcp.src_dt == data_type::bf16;
        return p.entry_[idx].is_sum(require_scale_one);
    };
    switch (p.len()) {
        case 0: return true;
        case 1: return is_relu(0) || is_sum(0);
        case 2: return is_sum(0) && is_relu(1);
        default: return false;
    }
    return false;
}

void jit_avx512_core_amx_1x1_fwd_kernel_t::store_output_vectors_int8(
        int ocb, int osb) {
    const bool mask_flag
            = last_oc_block_flag_ && ocb == (jcp.nb_oc_blocking - 1);
    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const float *p_sum_scale = nullptr;
    const int32_t *p_sum_zp = nullptr;
    if (sum_idx != -1) {
        const auto &p_entry = p.entry_[sum_idx];
        p_sum_scale = &p_entry.sum.scale;
        p_sum_zp = &p_entry.sum.zero_point;
    }
    if (p_sum_scale) {
        if (*p_sum_scale != 1.f)
            mov(reg_ptr_sum_scale, reinterpret_cast<size_t>(p_sum_scale));
        if (*p_sum_zp != 0)
            mov(reg_ptr_sum_zp, reinterpret_cast<size_t>(p_sum_zp));
    }

    if (jcp.src_zero_point) {
        const int zp_offset = sizeof(int32_t) * ocb * jcp.oc_block;
        const Zmm zmm_zp_m = zmm_mask(zmm_zp, mask_flag);
        vpmulld(zmm_zp_m, zmm_src_zp,
                EVEX_compress_addr(reg_zp_compensation, zp_offset));
        for (int j = 0; j < jcp.tile_width; j++) {
            const Zmm zmm_r = zmm_out(j);
            vpaddd(zmm_r, zmm_r, zmm_zp_m);
        }
    }

    for (int j = 0; j < jcp.tile_width; j++) {
        const Zmm zmm_r = zmm_out(j);
        vcvtdq2ps(zmm_r, zmm_r);
    }

    if (jcp.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
        int bias_offset = jcp.typesize_bia * ocb * jcp.oc_block;
        auto bias_addr = EVEX_compress_addr(reg_bias, bias_offset);
        cvt2ps(jcp.bia_dt, zmm_bias, bias_addr, mask_flag);
        for (int j = 0; j < jcp.tile_width; j++) {
            const Zmm zmm_r = zmm_out(j);
            vaddps(zmm_r, zmm_r, zmm_bias);
        }
    }

    mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);
    for (int j = 0; j < jcp.tile_width; j++) {
        const int scale_offset
                = jcp.is_oc_scale * (sizeof(float) * ocb * jcp.oc_block);
        const Zmm zmm_r = zmm_out(j);
        const Zmm zmm_r_msk = zmm_mask(zmm_r, mask_flag);
        vmulps(zmm_r_msk, zmm_r,
                EVEX_compress_addr(reg_ptr_scales, scale_offset));
    }

    if (p_sum_zp && *p_sum_zp != 0)
        vcvtdq2ps(zmm_sum_zp, ptr_b[reg_ptr_sum_zp]);
    if (jcp.with_sum && p_sum_scale != nullptr) {
        const auto p_sum_scale_val = *p_sum_scale;
        const auto p_sum_zp_val = *p_sum_zp;
        for (int j = 0; j < jcp.tile_width; j++) {
            int h = ((osb * jcp.tile_width + j) / jcp.ow);
            int w = ((osb * jcp.tile_width + j) % jcp.ow);

            const auto off = out_row_offset(h, w, ocb);
            const auto addr = EVEX_compress_addr(out_ptr, off);

            const Zmm zmm_r = zmm_out(j);
            cvt2ps(jcp.sum_dt, zmm_prev_dst, addr, mask_flag);
            if (p_sum_zp_val != 0) vsubps(zmm_prev_dst, zmm_sum_zp);
            if (p_sum_scale_val == 1.f)
                vaddps(zmm_r, zmm_prev_dst);
            else
                vfmadd231ps(zmm_r, zmm_prev_dst, zword_b[reg_ptr_sum_scale]);
        }
    }
    if (jcp.with_eltwise) {
        vxorps(zmm_zero, zmm_zero, zmm_zero);
        for (int j = 0; j < jcp.tile_width; j++) {
            const Zmm zmm_r = zmm_out(j);
            vmaxps(zmm_r, zmm_r, zmm_zero);
        }
    }

    if (jcp.dst_zero_point) {
        for (int j = 0; j < jcp.tile_width; j++) {
            const Zmm zmm_r = zmm_out(j);
            vaddps(zmm_r, zmm_r, zmm_dst_zp);
        }
    }

    // Properly saturate the accumulators for integer datatypes
    if (one_of(jcp.dst_dt, u8, s8, s32)) {
        init_saturate_f32(
                zmm_zero, zmm_saturation, aux_reg_saturation, f32, jcp.dst_dt);
        for (int j = 0; j < jcp.tile_width; j++) {
            const Zmm zmm_r = zmm_out(j);
            saturate_f32(zmm_r, zmm_zero, zmm_saturation, jcp.dst_dt);
            vcvtps2dq(zmm_r, zmm_r);
        }
    }

    for (int j = 0; j < jcp.tile_width; j++) {
        const int h = ((osb * jcp.tile_width + j) / jcp.ow);
        const int w = ((osb * jcp.tile_width + j) % jcp.ow);
        const auto off = out_row_offset(h, w, ocb);
        const auto addr = EVEX_compress_addr(out_ptr, off);

        const Zmm zmm_out_store = zmm_mask(zmm_out(j), mask_flag, true);
        switch (jcp.dst_dt) {
            case data_type::f32:
            case data_type::s32: vmovups(addr, zmm_out_store); break;
            case data_type::s8: vpmovsdb(addr, zmm_out_store); break;
            case data_type::u8: vpmovusdb(addr, zmm_out_store); break;
            default: assert(!"unknown dst_dt");
        }
    }
}

void jit_avx512_core_amx_1x1_fwd_kernel_t::store_output_vector_int8(
        const Zmm &zmm_out, int ocb, int h, int w) {

    const auto off = out_row_offset(h, w, ocb);
    const auto addr = EVEX_compress_addr(out_ptr, off);

    const bool mask_flag
            = last_oc_block_flag_ && ocb == (jcp.nb_oc_blocking - 1);
    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const float *p_sum_scale = nullptr;
    const int32_t *p_sum_zp = nullptr;
    if (sum_idx != -1) {
        const auto &p_entry = p.entry_[sum_idx];
        p_sum_scale = &p_entry.sum.scale;
        p_sum_zp = &p_entry.sum.zero_point;
    }

    if (p_sum_scale) {
        if (*p_sum_scale != 1.f)
            mov(reg_ptr_sum_scale, reinterpret_cast<size_t>(p_sum_scale));
        if (*p_sum_zp != 0)
            mov(reg_ptr_sum_zp, reinterpret_cast<size_t>(p_sum_zp));
    }

    mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
    mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);

    int scale_offset = jcp.is_oc_scale * (sizeof(float) * ocb * jcp.oc_block);
    if (jcp.with_bias) {
        int bias_offset = jcp.typesize_bia * ocb * jcp.oc_block;
        auto bias_addr = EVEX_compress_addr(reg_bias, bias_offset);
        cvt2ps(jcp.bia_dt, zmm_bias, bias_addr, mask_flag);
    }
    if (jcp.src_zero_point) {
        const int zp_offset = sizeof(int32_t) * ocb * jcp.oc_block;
        const Zmm zmm_zp_m = zmm_mask(zmm_zp, mask_flag);
        vpmulld(zmm_zp_m, zmm_src_zp,
                EVEX_compress_addr(reg_zp_compensation, zp_offset));
        vpaddd(zmm_out, zmm_out, zmm_zp_m);
    }
    /* add to zmm_accum: compensation, bias and permute */
    vcvtdq2ps(zmm_out, zmm_out);

    if (jcp.with_bias) vaddps(zmm_out, zmm_out, zmm_bias);
    const Zmm zmm_out_msk = zmm_mask(zmm_out, mask_flag);
    vmulps(zmm_out_msk, zmm_out,
            EVEX_compress_addr(reg_ptr_scales, scale_offset));

    apply_postops(zmm_out, p_sum_scale, p_sum_zp, addr, mask_flag, off, ocb);

    if (jcp.dst_zero_point) { vaddps(zmm_out, zmm_out, zmm_dst_zp); }

    // Properly saturate the accumulators for integer datatypes
    if (one_of(jcp.dst_dt, u8, s8, s32)) {
        init_saturate_f32(
                zmm_zero, zmm_saturation, aux_reg_saturation, f32, jcp.dst_dt);
        saturate_f32(zmm_out, zmm_zero, zmm_saturation, jcp.dst_dt);
        vcvtps2dq(zmm_out, zmm_out);
    }

    const Zmm zmm_out_store = zmm_mask(zmm_out, mask_flag, true);
    switch (jcp.dst_dt) {
        case data_type::f32:
        case data_type::s32: vmovups(addr, zmm_out_store); break;
        case data_type::s8: vpmovsdb(addr, zmm_out_store); break;
        case data_type::u8: vpmovusdb(addr, zmm_out_store); break;
        default: assert(!"unknown dst_dt");
    }
}

void jit_avx512_core_amx_1x1_fwd_kernel_t::store_output_vectors_bf16(
        int ocb, int osb) {
    const bool mask_flag
            = last_oc_block_flag_ && ocb == (jcp.nb_oc_blocking - 1);

    if (jcp.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
        const int bias_offset = jcp.typesize_bia * ocb * jcp.oc_block;
        const auto bias_addr = EVEX_compress_addr(reg_bias, bias_offset);
        cvt2ps(jcp.bia_dt, zmm_bias, bias_addr, mask_flag);
        for (int j = 0; j < jcp.tile_width; j++) {
            const Zmm zmm_r = zmm_out(j);
            vaddps(zmm_r, zmm_r, zmm_bias);
        }
    }

    if (jcp.with_sum) {
        for (int j = 0; j < jcp.tile_width; j++) {
            int h = ((osb * jcp.tile_width + j) / jcp.ow);
            int w = ((osb * jcp.tile_width + j) % jcp.ow);
            const auto off = out_row_offset(h, w, ocb);
            const auto addr = EVEX_compress_addr(out_ptr, off);
            const Zmm zmm_r = zmm_out(j);
            cvt2ps(jcp.dst_dt, zmm_prev_dst, addr, mask_flag);
            vaddps(zmm_r, zmm_prev_dst);
        }
    }
    if (jcp.with_eltwise) {
        vxorps(zmm_zero, zmm_zero, zmm_zero);
        for (int j = 0; j < jcp.tile_width; j++) {
            const Zmm zmm_r = zmm_out(j);
            vmaxps(zmm_r, zmm_r, zmm_zero);
        }
    }

    for (int j = 0; j < jcp.tile_width; j++) {
        const int h = ((osb * jcp.tile_width + j) / jcp.ow);
        const int w = ((osb * jcp.tile_width + j) % jcp.ow);
        const auto off = out_row_offset(h, w, ocb);
        const auto addr = EVEX_compress_addr(out_ptr, off);
        const Zmm zmm_r = zmm_out(j);
        if (jcp.dst_dt == data_type::bf16) {
            Ymm ymm_r = Ymm(zmm_r.getIdx());
            vcvtneps2bf16(ymm_r, zmm_r);
            vmovdqu16(addr, ymm_mask(ymm_r, mask_flag, true));
        } else {
            vmovups(addr, zmm_mask(zmm_r, mask_flag, true));
        }
    }
}

void jit_avx512_core_amx_1x1_fwd_kernel_t::store_output_vector_bf16(
        const Zmm &zmm_out, int ocb, int h, int w) {
    const auto off = out_row_offset(h, w, ocb);
    const auto addr = EVEX_compress_addr(out_ptr, off);

    const bool mask_flag
            = last_oc_block_flag_ && ocb == (jcp.nb_oc_blocking - 1);

    const auto &p = attr_.post_ops_;

    mov(reg_bias, ptr[param1 + GET_OFF(bias)]);

    const int sum_idx = p.find(primitive_kind::sum);
    if (sum_idx != -1) {
        if (jcp.dst_dt == data_type::bf16) {
            vpmovzxwd(zmm_mask(zmm_prev_dst, mask_flag), addr);
            vpslld(zmm_prev_dst, zmm_prev_dst, 16);
            vaddps(zmm_out, zmm_prev_dst);
        } else {
            vmovups(zmm_mask(zmm_prev_dst, mask_flag), addr);
            vaddps(zmm_out, zmm_prev_dst);
        }
    }
    if (jcp.with_bias) {
        int bias_offset = jcp.typesize_bia * ocb * jcp.oc_block;
        auto bias_addr = EVEX_compress_addr(reg_bias, bias_offset);
        if (jcp.bia_dt == data_type::bf16) {
            vpmovzxwd(zmm_mask(zmm_bias, mask_flag), bias_addr);
            vpslld(zmm_bias, zmm_bias, 16);
            vaddps(zmm_out, zmm_bias);
        } else
            vaddps(zmm_mask(zmm_out, mask_flag), bias_addr);
    }

    static constexpr auto skip_sum_in_injection = nullptr;
    apply_postops(zmm_out, skip_sum_in_injection, skip_sum_in_injection, addr,
            mask_flag, off, ocb);

    if (jcp.dst_dt == data_type::bf16) {
        Ymm ymm_out = Ymm(zmm_out.getIdx());
        vcvtneps2bf16(ymm_out, zmm_out);
        vmovdqu16(addr, ymm_mask(ymm_out, mask_flag, true));
    } else {
        vmovups(addr, zmm_mask(zmm_out, mask_flag, true));
    }
}

// Store all rows of a tile
void jit_avx512_core_amx_1x1_fwd_kernel_t::store_output_vectors(
        int ocb, int osb) {
    if (is_bf16()) {
        store_output_vectors_bf16(ocb, osb);
    } else {
        store_output_vectors_int8(ocb, osb);
    }
}

// Store single row
void jit_avx512_core_amx_1x1_fwd_kernel_t::store_output_vector(
        const Zmm &zmm_out, int ocb, int h, int w) {
    if (is_bf16()) {
        store_output_vector_bf16(zmm_out, ocb, h, w);
    } else {
        store_output_vector_int8(zmm_out, ocb, h, w);
    }
}

void jit_avx512_core_amx_1x1_fwd_kernel_t::prepare_output() {
    for (int osb = 0; osb < jcp.nb_os_blocking; osb++)
        for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++)
            tilezero(Tmm(get_out_tensor(osb, ocb)));
}

void jit_avx512_core_amx_1x1_fwd_kernel_t::store_output(
        bool do_store, bool has_tail) {

    auto store_output_subblock = [=](int ocb, int osb) {
        const int wsp_offset = jcp.typesize_acc
                * (osb * jcp.nb_oc_blocking * jcp.max_width * jcp.oc_block
                        + ocb * jcp.max_width * jcp.oc_block);
        tilestored(ptr[wsp_ptr + stride_seq + wsp_offset],
                Tmm(get_out_tensor(osb, ocb)));

        // preserve registers used by binary post_ops injector
        const injector_utils::conditional_register_preserve_guard_t
                cond_register_guard(jcp.with_binary, this,
                        {bin_injector_helper_reg_1, bin_injector_helper_reg_2});
        is_buffer_empty_ = false;
        is_store_done_ = (do_store) ? true : false;
        for (int j = 0; j < jcp.tile_width && do_store; j++) {
            int oh_ = ((osb * jcp.tile_width + j) / jcp.ow);
            int ow_ = ((osb * jcp.tile_width + j) % jcp.ow);

            auto addr = ptr[wsp_ptr + jcp.typesize_acc * (j * jcp.oc_block)
                    + wsp_offset];
            const Zmm zmm_r = zmm_out(j);
            vmovups(zmm_r, addr);
            if (!jcp.is_fast_postops) store_output_vector(zmm_r, ocb, oh_, ow_);
        }
        if (do_store && jcp.is_fast_postops) store_output_vectors(ocb, osb);
    };

    auto store_output_block = [=](int os_b = 1) {
        if (jcp.src_zero_point) {
            mov(reg_src_zero_point, ptr[param1 + GET_OFF(src_zero_point)]);
            mov(reg_zp_compensation, ptr[param1 + GET_OFF(zp_compensation)]);
            vpbroadcastd(zmm_src_zp, EVEX_compress_addr(reg_src_zero_point, 0));
        }
        if (jcp.dst_zero_point) {
            mov(reg_dst_zero_point, ptr[param1 + GET_OFF(dst_zero_point)]);
            vcvtdq2ps(zmm_dst_zp,
                    EVEX_compress_addr(reg_dst_zero_point, 0, true));
        }
        for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++)
            for (int osb = 0; osb < os_b; osb++)
                store_output_subblock(ocb, osb);
    };

    Label label_oc_store, label_done;

    if (check_last_sb_) {
        mov(reg_last_h, ptr[param1 + GET_OFF(last_h)]);
        cmp(reg_last_h, 1);
        je(label_oc_store, T_NEAR);
    }

    store_output_block(jcp.nb_os_blocking);
    jmp(label_done, T_NEAR);

    L(label_oc_store);
    store_output_block();

    L(label_done);
    update_buffer_pointers();
}

void jit_avx512_core_amx_1x1_fwd_kernel_t::icb_loop(bool do_store) {
    enum tiles_cfg_t { cfg_tiles, cfg_tiles_tail };
    enum restore_tiles_t { write_tiles, read_tiles };

    auto tdpbxxd = [=](const Tmm &x1, const Tmm &x2, const Tmm &x3) {
        if (jcp.src_dt == data_type::bf16 && jcp.wei_dt == data_type::bf16) {
            tdpbf16ps(x1, x2, x3);
        } else if (jcp.src_dt == data_type::u8 && jcp.wei_dt == data_type::u8) {
            tdpbuud(x1, x2, x3);
        } else if (jcp.src_dt == data_type::u8 && jcp.wei_dt == data_type::s8) {
            tdpbusd(x1, x2, x3);
        } else if (jcp.src_dt == data_type::s8 && jcp.wei_dt == data_type::u8) {
            tdpbsud(x1, x2, x3);
        } else if (jcp.src_dt == data_type::s8 && jcp.wei_dt == data_type::s8) {
            tdpbssd(x1, x2, x3);
        } else {
            assert(!"unsupported combination");
        }
    };

    auto tileloadd_nt = [=](const Tmm &t1, int offset) {
        int ab_size = jcp.nb_os2_blocking * jcp.nb_os_blocking * jcp.tile_width
                * (jcp.nb_ic_int * jcp.ic_block_int_np
                        + jcp.nb_oc_blocking * jcp.oc_block);
        int c_size = (jcp.nb_ic_int * jcp.ic_block_int_np * jcp.nb_oc_blocking
                * jcp.oc_block);
        // If the size of  src + wei used in the kernel cannot fit into L1 cache,
        // use non-temporal load of weights to help keep src in L1 cache
        if (static_cast<size_t>(jcp.typesize_in * (ab_size + c_size))
                >= platform::get_per_core_cache_size(1))
            tileloaddt1(t1, ptr[wei_ptr + offset + stride_seq]);
        else
            tileloadd(t1, ptr[wei_ptr + offset + stride_seq]);
    };

    auto compute_block = [=](int icb, int os_b) {
        for (int osb = 0; osb < os_b; osb++) {
            int ih = ((osb * jcp.tile_width) / jcp.ow) * jcp.stride_h;
            int iw = ((osb * jcp.tile_width) % jcp.ow) * jcp.stride_w;
            tileloadd(Tmm(get_inp_tensor(osb)),
                    ptr[inp_ptr + stride_nhwc + inp_offset(ih, iw, icb)]);
        }
        for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
            const int wei_offset = jcp.typesize_in
                    * (ocb
                                    * utils::rnd_up(jcp.ic_without_padding,
                                            jcp.ic_block_int)
                                    * jcp.oc_block
                            + icb * jcp.ic_block_int_np * jcp.oc_block);
            tileloadd_nt(Tmm(get_wei_tensor(ocb)), wei_offset);
            for (int osb = 0; osb < os_b; osb++) {
                tdpbxxd(Tmm(get_out_tensor(osb, ocb)), Tmm(get_inp_tensor(osb)),
                        Tmm(get_wei_tensor(ocb)));
                interleave_store();
            }
        }
    };

    auto reconfig_tiles = [=](tiles_cfg_t cfg) {
        tilerelease();
        if (cfg == cfg_tiles) {
            mov(reg_scratch, ptr[param1 + GET_OFF(tile_cfg)]);
        } else if (cfg == cfg_tiles_tail) {
            mov(reg_scratch, ptr[param1 + GET_OFF(tile_cfg_tail)]);
        }
        ldtilecfg(ptr[reg_scratch]);
    };

    auto restore_output_tiles = [=](int os_b, restore_tiles_t restore) {
        mov(reg_tilebuff, ptr[param1 + GET_OFF(src_prf)]);
        for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++)
            for (int osb = 0; osb < os_b; osb++) {
                const int wsp_offset = jcp.typesize_acc
                        * (osb * jcp.nb_oc_blocking * jcp.max_width
                                        * jcp.oc_block
                                + ocb * jcp.max_width * jcp.oc_block);
                if (restore == write_tiles)
                    tilestored(ptr[reg_tilebuff + stride_seq + wsp_offset],
                            Tmm(get_out_tensor(osb, ocb)));
                else if (restore == read_tiles)
                    tileloadd(Tmm(get_out_tensor(osb, ocb)),
                            ptr[reg_tilebuff + stride_seq + wsp_offset]);
            }
    };

    auto reset_tiles = [=](int os_b, bool tail) {
        if (jcp.nb_ic_int != 1) {
            restore_output_tiles(os_b, write_tiles);
            reconfig_tiles((tail) ? cfg_tiles_tail : cfg_tiles);
            restore_output_tiles(os_b, read_tiles);
        }
    };

    auto compute_icb_loop = [=](int os_b = 1) {
        int shift = (get_ic_tail() && os_b == 1) ? 1 : 0;
        int nb_ic_int = jcp.nb_ic_int - shift;

        if (jcp.src_zero_point) {
            mov(reg_zp_compensation, ptr[param1 + GET_OFF(zp_compensation)]);
            mov(reg_src_zero_point, ptr[param1 + GET_OFF(src_zero_point)]);
            vpbroadcastd(zmm_src_zp, EVEX_compress_addr(reg_src_zero_point, 0));
        }
        if (jcp.dst_zero_point) {
            mov(reg_dst_zero_point, ptr[param1 + GET_OFF(dst_zero_point)]);
            vcvtdq2ps(zmm_dst_zp,
                    EVEX_compress_addr(reg_dst_zero_point, 0, true));
        }

        for (int icb = 0; icb < nb_ic_int; icb++)
            compute_block(icb, os_b);

        // Tail processing
        if (get_ic_tail() && os_b == 1) {
            reset_tiles(os_b, true);
            compute_block(nb_ic_int, os_b);
            reset_tiles(os_b, false);
        }
    };

    Label label_last_os, label_compute_done, label_tail, label_done;

    int stride_nhwc_ = jcp.typesize_in * jcp.ngroups * jcp.ic_without_padding
            * jcp.stride_w;
    mov(stride_nhwc, stride_nhwc_);

    prepare_output();
    { // Compute
        if (check_last_sb_) {
            mov(reg_last_h, ptr[param1 + GET_OFF(last_h)]);
            cmp(reg_last_h, 1);
            je(label_last_os, T_NEAR);
        }
        compute_icb_loop(jcp.nb_os_blocking);

        jmp(label_compute_done, T_NEAR);

        L(label_last_os);
        compute_icb_loop();
    }
    L(label_compute_done);
    { // Store
        if (jcp.tile_tail && check_last_sb_)
            store_output(do_store, true);
        else
            store_output(do_store, false);
    }
}

void jit_avx512_core_amx_1x1_fwd_kernel_t::osb_loop(int nb_os) {
    for (int osi = 0; osi < nb_os; osi++) {
        bool do_store = IMPLICATION(jcp.per_one_pstore, (osi == nb_os - 1));
        check_last_sb_ = do_store;

        icb_loop(do_store);

        int oh = (((osi + 1) * jcp.nb_os_blocking * jcp.tile_width) / jcp.ow);
        int ow = (((osi + 1) * jcp.nb_os_blocking * jcp.tile_width) % jcp.ow);
        if (do_store) {
            size_t out_offset = jcp.typesize_out
                    * (oh * out_h_shift() + ow * out_w_shift());
            add(out_ptr, out_offset);
        }

        int ih = oh * jcp.stride_h;
        int iw = ow * jcp.stride_w;
        add(inp_ptr, inp_offset(ih, iw, 0));
    }
}

int jit_avx512_core_amx_1x1_fwd_kernel_t::get_ic_tail() const {
    return (jcp.ic_without_padding % jcp.ic_block_int_np);
}

void jit_avx512_core_amx_1x1_fwd_kernel_t::generate() {
    preamble();

    if (postops_injector_)
        postops_injector_->push_post_ops_data_on_stack(param1, GET_OFF(post_ops_binary_rhs_arg_vec), inp_ptr, wei_ptr);

    last_oc_block_flag_ = (jcp.oc_without_padding != jcp.oc);
    if (last_oc_block_flag_) {
        Xbyak::Label mask_is_set;

        // Use mask 0xF by default for all output data and post-ops
        // loads / stores with block index
        // ocb = occ * jcp.nb_oc_blocking + (jcp.nb_oc_blocking - 1)
        // TODO: use masked loads / stores for the last occ only
        int mask = (1 << jcp.oc_block) - 1;
        Xbyak::Reg32 regw_tmp = reg_tmp.cvt32();
        mov(regw_tmp, mask);
        kmovw(ktail_mask, regw_tmp);
        mov(reg_oc_blocks, ptr[param1 + GET_OFF(oc_blocks)]);
        cmp(reg_oc_blocks, jcp.nb_oc - jcp.nb_oc_blocking);
        jne(mask_is_set, T_NEAR);

        // Reset the mask
        mask = (1 << (jcp.oc_without_padding % jcp.oc_block)) - 1;
        mov(regw_tmp, mask);
        kmovw(ktail_mask, regw_tmp);

        L(mask_is_set);
    }

    mov(inp_ptr, ptr[param1 + GET_OFF(src)]);
    mov(wei_ptr, ptr[param1 + GET_OFF(filt)]);
    mov(out_ptr, ptr[param1 + GET_OFF(dst)]);
    mov(wsp_ptr, ptr[param1 + GET_OFF(acc_s32)]);

    mov(reg_is_osb, ptr[param1 + GET_OFF(is_osb)]);

    constexpr int tile_mem_stride_in_bytes = 64;
    mov(stride_seq, tile_mem_stride_in_bytes);

    init_runtime_counters();
    update_buffer_pointers();

    Xbyak::Label label_no_osb, label_done;

    cmp(reg_is_osb, 0);
    je(label_no_osb, T_NEAR);

    osb_loop(jcp.nb_os2_blocking);
    jmp(label_done, T_NEAR);

    L(label_no_osb);
    osb_loop();

    L(label_done);

    if (postops_injector_)
        postops_injector_->reset_stack_pointer();

    postamble();

    if (jcp.with_eltwise) postops_injector_->prepare_table();
}

void jit_avx512_core_amx_1x1_fwd_kernel_t::tile_configure(char *tcfg_buff) {

    int tile_max_columns_in_bytes
            = amx::get_max_column_bytes(amx::get_max_palette());
    const int max_palette_size_in_bytes = 64;

    auto cfg_tiles = [=](palette_config_t *buff, int Ac) {
        char *_tc = (char *)buff;
        for (int i = 0; i < max_palette_size_in_bytes; i++)
            _tc[i] = 0;

        int Ar = jcp.tile_width;
        int Br = Ac / jcp.typesize_acc;
        int Cr = jcp.tile_width;

        int Bc = tile_max_columns_in_bytes;
        int Cc = tile_max_columns_in_bytes;

        for (int s = 0; s < jcp.nb_os_blocking; s++)
            tc_configure_tile(buff, get_inp_tensor(s), Ar, Ac);
        for (int i = 0; i < jcp.nb_oc_blocking; i++)
            tc_configure_tile(buff, get_wei_tensor(i), Br, Bc);

        for (int s = 0; s < jcp.nb_os_blocking; s++)
            for (int i = 0; i < jcp.nb_oc_blocking; i++) {
                tc_configure_tile(buff, get_out_tensor(s, i), Cr, Cc);
            }

        buff->palette_id = amx::get_max_palette();
    };

    int Ac = jcp.typesize_in
            * ((jcp.nb_ic_int == 1 && get_ic_tail()) ? get_ic_tail()
                                                     : jcp.ic_block_int_np);

    cfg_tiles((palette_config_t *)tcfg_buff, Ac);
    if (jcp.nb_ic_int > 1 && get_ic_tail()) {
        int Ac = jcp.typesize_in * get_ic_tail();
        char *_t = tcfg_buff + max_palette_size_in_bytes;
        cfg_tiles((palette_config_t *)(_t), Ac);
    }
}

status_t jit_avx512_core_amx_1x1_fwd_kernel_t::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads) {
    using namespace prop_kind;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();
    bool is_1d = ndims == 3;
    bool is_3d = ndims == 5;

    const bool is_bf16_convolution
            = everyone_is(true, src_d.data_type() == data_type::bf16,
                    weights_d.data_type() == data_type::bf16,
                    one_of(dst_d.data_type(), data_type::bf16, data_type::f32));
    const bool is_int8_convolution = everyone_is(true,
            (src_d.data_type() == data_type::u8
                    || src_d.data_type() == data_type::s8),
            weights_d.data_type() == data_type::s8,
            one_of(dst_d.data_type(), data_type::f32, data_type::s32,
                    data_type::s8, data_type::u8));

    bool supported = false
            || (is_bf16_convolution && mayiuse(avx512_core_bf16_amx_bf16))
            || (is_int8_convolution && mayiuse(avx512_core_bf16_amx_int8));
    if (!supported) return status::unimplemented;

    jcp = zero<decltype(jcp)>();
    jcp.isa = is_bf16_convolution ? avx512_core_bf16_amx_bf16
                                  : avx512_core_bf16_amx_int8;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
    jcp.id = is_3d ? src_d.dims()[2] : 1;
    jcp.ih = !is_1d ? src_d.dims()[ndims - 2] : 1;
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = is_3d ? dst_d.dims()[2] : 1;
    jcp.oh = !is_1d ? dst_d.dims()[ndims - 2] : 1;
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kd = is_3d ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = !is_1d ? weights_d.dims()[with_groups + ndims - 2] : 1;
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    jcp.f_pad = is_3d ? cd.padding[0][0] : 0;
    jcp.t_pad = !is_1d ? cd.padding[0][ndims - 4] : 0;
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_d = is_3d ? cd.strides[0] : 1;
    jcp.stride_h = !is_1d ? cd.strides[ndims - 4] : 1;
    jcp.stride_w = cd.strides[ndims - 3];
    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    if (!(jcp.kd == 1 && jcp.kh == 1 && jcp.kw == 1))
        return status::unimplemented;

    if (!(jcp.f_pad == 0 && jcp.t_pad == 0 && jcp.l_pad == 0))
        return status::unimplemented;

    jcp.dilate_d = is_3d ? cd.dilates[0] : 0;
    jcp.dilate_h = is_1d ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    jcp.is_depthwise = true && with_groups && everyone_is(1, jcp.ic, jcp.oc);

    if (jcp.dilate_d != 0 || jcp.dilate_h != 0 || jcp.dilate_w != 0)
        return status::unimplemented;
    if (jcp.is_depthwise)
        return status::unimplemented; // TODO: add support of DW convolution
    if (jcp.ngroups > 1)
        return status::unimplemented; // TODO: add support for non-unit groups

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;
    jcp.src_dt = cd.src_desc.data_type;
    jcp.wei_dt = cd.weights_desc.data_type;

    const auto zp = attr.zero_points_;
    jcp.dst_zero_point = !zp.has_default_values(DNNL_ARG_DST);
    jcp.src_zero_point = !zp.has_default_values(DNNL_ARG_SRC);
    jcp.zp_src_is_common = zp.common(
            DNNL_ARG_SRC); // otherwise, it's per-channel (not supported)
    if (!IMPLICATION(jcp.src_zero_point, jcp.zp_src_is_common)
            || !IMPLICATION(jcp.dst_zero_point || jcp.src_zero_point,
                    is_int8_convolution))
        return status::unimplemented;

    jcp.nthr = nthreads;

    jcp.ic_block = 16;
    jcp.ic_block_int = is_bf16_convolution ? 32 : 64;
    jcp.ic_block_int_np = jcp.ic_block_int;
    if (jcp.ic_block_int < jcp.ic_without_padding
            && jcp.ic_without_padding % jcp.ic_block_int != 0) {
        // Order of blocks comes from empirical observation
        static const int try_blocks[] = {32, 48, 40, 56};
        for (auto blk_size : try_blocks) {
            const int _blk_size = is_bf16_convolution ? blk_size / 2 : blk_size;
            if (jcp.ic_without_padding % _blk_size == 0) {
                jcp.ic_block_int_np = _blk_size;
                break;
            }
        }
    }
    jcp.oc_block = 16;

    bool args_ok = true && jcp.ic % 4 == 0
            && (jcp.ow == jcp.iw && jcp.stride_w == 1)
            && (jcp.oh == jcp.ih && jcp.stride_h == 1)
            && (jcp.od == jcp.id && jcp.stride_d == 1);
    if (!args_ok) return status::unimplemented;

    if (jcp.ngroups == 1) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }

    auto set_or_check_wei_format = [&]() {
        using namespace format_tag;
        using namespace memory_extra_flags;
        format_tag_t wei_tag;
        wei_tag = (is_bf16_convolution)
                ? pick(with_groups + 2 * (ndims - 3), OIw16i16o2i, gOIw16i16o2i,
                        OIhw16i16o2i, gOIhw16i16o2i, OIdhw16i16o2i,
                        gOIdhw16i16o2i)
                : pick(with_groups + 2 * (ndims - 3), OIw16i16o4i, gOIw16i16o4i,
                        OIhw16i16o4i, gOIhw16i16o4i, OIdhw16i16o4i,
                        gOIdhw16i16o4i);
        memory_desc_t want_wei_md = weights_md;
        memory_desc_init_by_tag(want_wei_md, wei_tag);

        if (jcp.src_zero_point) {
            want_wei_md.extra.flags |= compensation_conv_asymmetric_src;
            want_wei_md.extra.asymm_compensation_mask = (1 << 0)
                    + (with_groups && !jcp.is_depthwise ? (1 << 1) : 0);
        }
        if (weights_md.format_kind == format_kind::any) {
            weights_md = want_wei_md;
            return true;
        }
        return weights_md == want_wei_md;
    };

    if (!set_or_check_wei_format()) { return status::unimplemented; }

    format_tag_t dat_tag = utils::pick(
            ndims - 3, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);

    if (src_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, dat_tag));
        jcp.src_tag = dat_tag;
    } else {
        jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    }
    if (jcp.src_tag != dat_tag) { return status::unimplemented; }

    if (dst_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(dst_md, dat_tag));
        jcp.dst_tag = dat_tag;
    } else {
        jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag);
    }
    if (jcp.dst_tag != dat_tag) { return status::unimplemented; }

    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, format_tag::x));
    }

    CHECK(attr.set_default_formats(&dst_md));

    const auto &p = attr.post_ops_;

    const int sum_ind = p.find(primitive_kind::sum);
    jcp.with_sum = sum_ind != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    const int binary_ind = p.find(primitive_kind::binary);
    jcp.with_binary = binary_ind != -1;
    jcp.sum_dt = p.get_sum_dt(jcp.dst_dt);

    if (jcp.with_sum)
        jcp.sum_dt = p.entry_[sum_ind].sum.dt;

    jcp.with_depthwise = p.find(primitive_kind::depthwise) != -1;
    jcp.with_quantization = p.find(primitive_kind::quantization) != -1;

    jcp.post_ops = p;
    jcp.is_fast_postops = is_fast_postops(jcp);

    using namespace injector;
    const bool sum_at_pos_0_only = (jcp.src_dt == data_type::bf16);
    const bool sum_requires_scale_one = sum_at_pos_0_only;
    const bool sum_requires_zp_zero = sum_at_pos_0_only;
    const bool post_ops_ok_ = post_ops_ok({avx512_core, {eltwise, binary, sum, depthwise, quantization},
            jcp.post_ops, &dst_d, sum_at_pos_0_only, sum_requires_scale_one,
            sum_requires_zp_zero});
    if (!post_ops_ok_) return status::unimplemented;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_bia
            = jcp.with_bias ? types::data_type_size(bias_d.data_type()) : 0;
    jcp.typesize_acc = sizeof(int32_t);

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic_int = div_up(jcp.ic_without_padding, jcp.ic_block_int_np);

    jcp.max_width = amx::get_max_rows(amx::get_max_palette());
    const int size_treshold = 32;
    const int min_width
            = 1; // TODO: Possible optimizations: do not use small values
    const int spatial = jcp.od * jcp.oh;
    const int os = jcp.od * jcp.oh * jcp.ow;

    jcp.tile_width = 1;
    for (int s_size = jcp.max_width; s_size >= min_width; s_size--) {
        if ((spatial >= size_treshold && spatial % s_size == 0)
                || (spatial < size_treshold && os % s_size == 0)) {
            jcp.tile_width = s_size;
            break;
        }
    }
    if (jcp.tile_width == 1) {
        jcp.tile_width = nstl::min(jcp.max_width, os);
        jcp.tile_tail = os % jcp.max_width;
        for (int i = jcp.max_width; i >= min_width; i--) {
            int i_tail = os % i;
            if (i_tail > jcp.tile_tail || i_tail == 0) {
                jcp.tile_width = i;
                jcp.tile_tail = i_tail;
                if (i_tail == 0) break;
            }
        }
        if (jcp.tile_width < min_width && jcp.tile_tail < min_width)
            jcp.tile_tail = 0;
    }

    /* TODO: Add stride support !
    while ((jcp.stride_h != 1 || jcp.stride_w != 1)
        && (jcp.ow % jcp.tile_width != 0) || jcp.tile_width > 16) {
        jcp.tile_width = jcp.ow / 2;
    }
    */

    // TODO: Add support for spatial tails
    if (jcp.tile_tail != 0) return status::unimplemented;

    // TODO: Implement efficient tile tail processing. Now just go to common
    // case if we utilize half of tile or less.
    if (jcp.tile_width <= jcp.max_width / 2) return status::unimplemented;

    jcp.nb_oc_blocking = (jcp.nb_oc % 2 == 0) ? 2 : 1;
    jcp.nb_ic_blocking = 1;
    jcp.nb_os_blocking = (os / jcp.tile_width > 2) ? 2 : 1;
    jcp.nb_os2_blocking = (jcp.nb_os_blocking > 1)
            ? ((jcp.nb_os_blocking * jcp.tile_width) % 2 == 0) ? 2 : 1
            : 1;
    jcp.nb_os = os / jcp.tile_width;

    jcp.wsp_buffer_size = (size_t)2 * jcp.nb_os_blocking * jcp.nb_oc_blocking
            * jcp.max_width * jcp.oc_block;

    int ops_tile_store
            = jcp.nb_oc_blocking * jcp.nb_os_blocking * jcp.tile_width;
    int avaliable_ops = jcp.nb_ic_int * jcp.nb_oc_blocking * jcp.nb_os_blocking;
    jcp.per_one_pstore
            = (avaliable_ops) ? ops_tile_store / avaliable_ops + 1 : 0;
    if (jcp.per_one_pstore > 12) jcp.per_one_pstore = 0;
    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    return status::success;
}

void jit_avx512_core_amx_1x1_fwd_kernel_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp,
        const primitive_attr_t &attr) {
    scratchpad.book(key_conv_amx_wsp_buffer, jcp.nthr * jcp.wsp_buffer_size,
            jcp.typesize_acc);
    if (jcp.ic_without_padding % jcp.ic_block_int_np)
        scratchpad.book(key_conv_amx_tile_buffer,
                jcp.nthr * (jcp.wsp_buffer_size / 2), jcp.typesize_acc);
    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding) {
        assert(jcp.ngroups == 1);
        scratchpad.book(key_conv_padded_bias, jcp.oc, jcp.typesize_bia);
    }
    scratchpad.book(key_conv_amx_tilecfg, 2, 64); // 2 whole cachelines
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
