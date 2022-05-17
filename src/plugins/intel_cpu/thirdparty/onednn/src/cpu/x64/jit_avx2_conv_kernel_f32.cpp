/*******************************************************************************
* Copyright 2016-2021 Intel Corporation
* Copyright 2018 YANDEX LLC
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
#include "common/memory.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/x64/injectors/injector_utils.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_avx2_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace Xbyak;

jit_avx2_conv_fwd_kernel_f32::jit_avx2_conv_fwd_kernel_f32(
        const jit_conv_conf_t &ajcp, const primitive_attr_t &attr,
        const memory_desc_t &dst_md)
    : jit_generator(nullptr, MAX_CODE_SIZE, true, avx2)
    , jcp(ajcp)
    , attr_(attr) {
    if (jcp.with_eltwise || jcp.with_binary || jcp.with_depthwise || jcp.with_quantization) {
        using namespace binary_injector;
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = false;
        static constexpr size_t helper_vmm_idx = 15;
        static constexpr bool use_exact_tail_scalar_bcast = false;
        const size_t tail_size = jcp.oc_without_padding % isa_simd_width_;

        rhs_arg_static_params_t rhs_arg_static_params {helper_vmm_idx, r13, r14,
                preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec),
                memory_desc_wrapper(dst_md), tail_size,
                use_exact_tail_scalar_bcast};
        static_params_t static_params {this->param1, rhs_arg_static_params};
        quantization_injector::static_params_t quantization_static_params
                {ymm_d_weights.getIdx(), ymm_d_bias.getIdx(), reg_d_weights, reg_d_bias};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<avx2>>(
                this, jcp.post_ops, static_params, quantization_static_params);
    }
}

void jit_avx2_conv_fwd_kernel_f32::oh_step_unroll_kw(
        int ur_w, int pad_l, int pad_r, int oc_blocks) {
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int dilate_w = jcp.dilate_w + 1;
    int ic_block = jcp.ic_block;
    int ic_tail = jcp.ic_tail;

    for (int ki = 0; ki < kw; ki++) {
        int jj_start = nstl::max(0, div_up(pad_l - ki * dilate_w, stride_w));
        int jj_end = ur_w
                - nstl::max(0,
                        div_up(ki * dilate_w + pad_r - (kw - 1) * dilate_w,
                                stride_w));

        auto compute = [=](int cur_ic_blk) {
            for (int ifm2 = 0; ifm2 < cur_ic_blk; ifm2++) {
                for (int jj = jj_start; jj < jj_end; jj++) {
                    size_t inp_off = get_input_offset(
                            ifm2, filter_w_to_input(ki, jj, pad_l));
                    vbroadcastss(Ymm(oc_blocks * ur_w + jj),
                            make_safe_addr(
                                    aux_reg_input, inp_off, reg_long_offt));
                }

                for (int ii = 0; ii < oc_blocks; ii++) {
                    vmovups(ymm15,
                            make_safe_addr(aux_reg_kernel,
                                    get_kernel_offset(ii, ki, ifm2),
                                    reg_long_offt));
                    for (int jj = jj_start; jj < jj_end; jj++)
                        if (mayiuse(avx2))
                            vfmadd231ps(Ymm(ur_w * ii + jj),
                                    Ymm(oc_blocks * ur_w + jj), ymm15);
                        else { // Intel(R) Advanced Vector Extensions (Intel(R) AVX) support
                            vmulps(ytmp, ymm15, Ymm(oc_blocks * ur_w + jj));
                            vaddps(Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj),
                                    ytmp);
                        }
                }
            }
        };

        if (ic_tail) {
            if (jcp.ic == ic_tail)
                compute(ic_tail);
            else {
                Label ic_blk_tail, ic_blk_done;
                cmp(reg_channel, ic_block);
                jl(ic_blk_tail, T_NEAR);

                compute(ic_block);
                jmp(ic_blk_done, T_NEAR);

                L(ic_blk_tail);
                compute(ic_tail);

                L(ic_blk_done);
            }
        } else {
            compute(ic_block);
        }
    }
}

void jit_avx2_conv_fwd_kernel_f32::oh_step_nopad(
        int ur_w, int pad_l, int pad_r, int oc_blocks) {
    Label kw_loop;

    int kw = jcp.kw;
    int ic_blk = jcp.ic_block;

    xor_(ki_iter, ki_iter);
    L(kw_loop);
    {
        int jj_start = 0;
        int jj_end = ur_w;
        for (int ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (int jj = jj_start; jj < jj_end; jj++) {
                size_t inp_off = get_input_offset(
                        ifm2, filter_w_to_input(0, jj, pad_l));
                vbroadcastss(Ymm(oc_blocks * ur_w + jj),
                        make_safe_addr(aux_reg_input, inp_off, reg_long_offt));
            }
            for (int ii = 0; ii < oc_blocks; ii++) {
                vmovups(ymm15,
                        make_safe_addr(aux_reg_kernel,
                                get_kernel_offset(ii, 0, ifm2), reg_long_offt));
                for (int jj = jj_start; jj < jj_end; jj++)
                    if (mayiuse(avx2))
                        vfmadd231ps(Ymm(ur_w * ii + jj),
                                Ymm(oc_blocks * ur_w + jj), ymm15);
                    else { // Intel AVX support
                        vmulps(ytmp, ymm15, Ymm(oc_blocks * ur_w + jj));
                        vaddps(Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj), ytmp);
                    }
            }
        }
        safe_add(aux_reg_kernel, get_kernel_offset(0, 1, 0), reg_long_offt);
        safe_add(aux_reg_input, get_input_offset(0, filter_w_to_input(1)),
                reg_long_offt);

        inc(ki_iter);
        cmp(ki_iter, kw);
        jl(kw_loop, T_NEAR);
    }
}

static int get_ymm_idx(
        const int ur_w, const int oc_block_idx, const int ur_w_idx) {
    return (ur_w * oc_block_idx + ur_w_idx);
}

static Ymm get_ymm(const int ur_w, const int oc_block_idx, const int ur_w_idx) {
    return Ymm(get_ymm_idx(ur_w, oc_block_idx, ur_w_idx));
}

template <typename F>
void iterate(const int load_loop_blk, const int ur, const int load_dim_tail,
        const F &f) {
    for (int i = 0; i < load_loop_blk; ++i) {
        const bool mask_flag = (load_dim_tail > 0) && (i == load_loop_blk - 1);
        for (int j = 0; j < ur; ++j)
            f(mask_flag, i, j);
    }
}
template <typename F>
void iterate(const int load_loop_blk, const int ur, const F &f) {
    iterate(load_loop_blk, ur, 0, f);
}

void jit_avx2_conv_fwd_kernel_f32::apply_postops(
        const int oc_blocks, const int ur_w, const int oc_tail) {
    if (jcp.with_eltwise || jcp.with_binary || jcp.with_depthwise || jcp.with_quantization) {
        Label regular_store;
        test(reg_ci_flag, FLAG_IC_LAST);
        je(regular_store, T_NEAR);

        std::map<size_t, int> vmm_idx_off;
        iterate(oc_blocks, ur_w, [&](const bool, const int i, const int j) {
            vmm_idx_off.insert({get_ymm_idx(ur_w, i, j), i * jcp.oc_block * sizeof(float)});
        });
        depthwise_injector::dynamic_params_t ddp {ymm_d_weights.getIdx(), ymm_d_bias.getIdx(), reg_d_weights, reg_d_bias,
                                                  ptr[this->param1 + GET_OFF(oc_off)], vmm_idx_off,
                                                  this->rsp, base_post_ops_data_offset};
        quantization_injector::dynamic_params_t qdp {ptr[this->param1 + GET_OFF(oc_off)], vmm_idx_off, jcp.dst_dt,
                                                     this->rsp, base_post_ops_data_offset};

        injector_utils::vmm_index_set_t vmm_idxs;
        if (jcp.with_binary) {
            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params,
                    rhs_arg_params_tail;
            iterate(oc_blocks, ur_w, oc_tail,
                    [&](const bool mask_flag, const int i, const int j) {
                        const int aux_output_offset
                                = get_output_offset(i, j) / sizeof(float);
                        const auto vmm_idx = get_ymm_idx(ur_w, i, j);
                        vmm_idxs.emplace(vmm_idx);

                        rhs_arg_params_tail.vmm_idx_to_oc_elem_off_addr.emplace(
                                vmm_idx, ptr[param1 + GET_OFF(oc_l_off)]);
                        rhs_arg_params_tail.vmm_idx_to_oc_elem_off_val.emplace(
                                vmm_idx, i * jcp.oc_block);
                        rhs_arg_params_tail.vmm_idx_to_out_elem_off_val.emplace(
                                vmm_idx, aux_output_offset);
                        rhs_arg_params_tail.vmm_idx_to_out_off_oprnd.emplace(
                                vmm_idx, temp_offset_reg);
                        if (mask_flag)
                            rhs_arg_params_tail.vmm_tail_idx_.emplace(vmm_idx);
                    });
            rhs_arg_params = rhs_arg_params_tail;
            rhs_arg_params.vmm_tail_idx_.clear();

            const injector_utils::register_preserve_guard_t register_guard(
                    this, {temp_offset_reg});
            mov(temp_offset_reg, reg_output);
            sub(temp_offset_reg, ptr[param1 + GET_OFF(dst_orig)]);
            shr(temp_offset_reg, std::log2(sizeof(float)));

            Label postops_done;
            if (oc_tail) {
                Label postops_no_tail;
                test(reg_oc_flag, FLAG_OC_LAST);
                je(postops_no_tail, T_NEAR);
                postops_injector_->compute_vector_range(
                        vmm_idxs, rhs_arg_params_tail);
                jmp(postops_done, T_NEAR);
                L(postops_no_tail);
            }
            postops_injector_->compute_vector_range(vmm_idxs, rhs_arg_params, ddp, qdp);
            L(postops_done);

        } else {
            iterate(oc_blocks, ur_w, [&](const bool, const int i, const int j) {
                vmm_idxs.emplace(get_ymm_idx(ur_w, i, j));
            });
            postops_injector_->compute_vector_range(vmm_idxs, binary_injector::rhs_arg_dynamic_params_t(), ddp, qdp);
        }
        L(regular_store);
    }
}

void jit_avx2_conv_fwd_kernel_f32::width_blk_step(
        int ur_w, int pad_l, int pad_r, int oc_blocks) {
    int kw = jcp.kw;
    int oc_blk = jcp.oc_block;
    int oc_tail = jcp.oc_tail;

    if (oc_tail) {
        push(reg_oc_blocks);
        base_post_ops_data_offset += reg64_size;
        mov(reg_oc_flag, ptr[param1 + GET_OFF(oc_flag)]);
    }

    auto load_output_bias_and_add_bias = [=](bool is_tail) {
        Label init_done, init_first;

        if (!jcp.with_sum) {
            test(reg_ci_flag, FLAG_IC_FIRST);
            jne(init_first, T_NEAR);
        }

        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++) {
                const auto ymm = get_ymm(ur_w, ii, jj);
                if (is_tail && ii == oc_blocks - 1)
                    load_bytes(ymm, reg_output, get_output_offset(ii, jj),
                            oc_tail * sizeof(float));
                else
                    vmovups(ymm,
                            make_safe_addr(reg_output,
                                    get_output_offset(ii, jj), reg_long_offt));
            }

        if (jcp.with_sum && jcp.with_bias) {
            test(reg_ci_flag, FLAG_IC_FIRST);
            je(init_done, T_NEAR);

            for (int ii = 0; ii < oc_blocks; ii++)
                for (int jj = 0; jj < ur_w; jj++) {
                    const Ymm ymm = get_ymm(ur_w, ii, jj);
                    if (is_tail && ii == oc_blocks - 1) {
                        load_bytes(ytmp, reg_bias, sizeof(float) * ii * oc_blk,
                                oc_tail * sizeof(float));
                        vaddps(ymm, ymm, ytmp);
                    } else {
                        vaddps(ymm, ymm,
                                yword[reg_bias + sizeof(float) * ii * oc_blk]);
                    }
                }
        }
        jmp(init_done, T_NEAR);

        L(init_first);

        if (jcp.with_bias) {
            for (int ii = 0; ii < oc_blocks; ii++)
                for (int jj = 0; jj < ur_w; jj++) {
                    const Ymm ymm = get_ymm(ur_w, ii, jj);
                    if (is_tail && ii == oc_blocks - 1)
                        load_bytes(ymm, reg_bias, sizeof(float) * ii * oc_blk,
                                oc_tail * sizeof(float));
                    else
                        vmovups(ymm,
                                yword[reg_bias + sizeof(float) * ii * oc_blk]);
                }
        } else {
            for (int ii = 0; ii < oc_blocks; ii++)
                for (int jj = 0; jj < ur_w; jj++) {
                    const Ymm ymm = get_ymm(ur_w, ii, jj);
                    uni_vpxor(ymm, ymm, ymm);
                }
        }
        L(init_done);
    };

    if (oc_tail) {
        if (jcp.nb_oc > jcp.nb_oc_blocking) {
            Label load_tail, load_done;
            test(reg_oc_flag, FLAG_OC_LAST);
            jne(load_tail, T_NEAR);

            load_output_bias_and_add_bias(false);
            jmp(load_done, T_NEAR);

            L(load_tail);
            load_output_bias_and_add_bias(true);

            L(load_done);
        } else {
            load_output_bias_and_add_bias(true);
        }
    } else {
        load_output_bias_and_add_bias(false);
    }

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);
    }

    Label skip_kh_loop, skip_kd_loop, kd_loop;
    if (jcp.ndims == 5) {
        push(reg_output);
        push(oi_iter);

        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_ker_d, ptr[param1 + GET_OFF(filt)]);
        mov(aux_reg_inp_d, reg_input);

        if ((jcp.dilate_d >= jcp.id)
                || (jcp.kd - 1) * (jcp.dilate_d + 1) < jcp.f_pad) {
            cmp(reg_ki, 0);
            je(skip_kd_loop, T_NEAR);
        }
        L(kd_loop);
        mov(kj, ptr[param1 + GET_OFF(kh_padding)]);
    } else {
        mov(kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_input, aux_reg_inp_d);
        mov(aux_reg_kernel, aux_reg_ker_d);
    }

    if ((jcp.dilate_h >= jcp.ih)
            || (jcp.kh - 1) * (jcp.dilate_h + 1)
                    < nstl::max(jcp.t_pad, jcp.b_pad)) {
        cmp(kj, 0);
        je(skip_kh_loop, T_NEAR);
    }
    Label kh_loop;
    L(kh_loop);
    {
        if (jcp.kw >= 5 && pad_l == 0 && pad_r == 0) {
            oh_step_nopad(ur_w, pad_l, pad_r, oc_blocks);
            add(aux_reg_input,
                    get_input_offset(0, filter_h_to_input(1))
                            - get_input_offset(0, filter_w_to_input(kw)));
        } else {
            oh_step_unroll_kw(ur_w, pad_l, pad_r, oc_blocks);
            safe_add(
                    aux_reg_kernel, get_kernel_offset(0, kw, 0), reg_long_offt);
            safe_add(aux_reg_input, get_input_offset(0, filter_h_to_input(1)),
                    reg_long_offt);
        }

        dec(kj);
        cmp(kj, 0);
        jg(kh_loop, T_NEAR);
    }

    L(skip_kh_loop);

    if (jcp.ndims == 5) {
        safe_add(aux_reg_inp_d, get_input_offset(0, filter_d_to_input(1)),
                reg_long_offt);
        safe_add(aux_reg_ker_d, get_kernel_offset(0, jcp.kw * jcp.kh, 0),
                reg_long_offt);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_loop, T_NEAR);
        L(skip_kd_loop);

        pop(oi_iter);
        pop(reg_output);
    }

    apply_postops(oc_blocks, ur_w, oc_tail);

    auto store_output = [=](bool is_tail, int tail) {
        const auto is_padding = jcp.oc_without_padding != jcp.oc;
        if (is_padding) uni_vxorps(ytmp, ytmp, ytmp);
        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++) {
                Ymm reg_out = get_ymm(ur_w, ii, jj);
                if (is_tail && ii == oc_blocks - 1) {
                    if (is_padding && jcp.with_binary) {
                        vmovups(make_safe_addr(reg_output,
                                        get_output_offset(ii, jj),
                                        reg_long_offt),
                                ytmp);
                    }
                    store_bytes(reg_out, reg_output, get_output_offset(ii, jj),
                            tail * sizeof(float));
                } else
                    vmovups(make_safe_addr(reg_output,
                                    get_output_offset(ii, jj), reg_long_offt),
                            reg_out);
            }
    };

    if (oc_tail) {
        if (jcp.nb_oc > jcp.nb_oc_blocking) {
            Label store_tail, store_done;
            test(reg_oc_flag, FLAG_OC_LAST);
            jne(store_tail, T_NEAR);

            store_output(false, oc_tail);
            jmp(store_done, T_NEAR);

            L(store_tail);
            store_output(true, oc_tail);

            L(store_done);
        } else {
            store_output(true, oc_tail);
        }
    } else {
        Label regular_store;
        Label store_done;
        const int tail = jcp.oc_without_padding % jcp.oc_block;
        if (jcp.with_binary && tail) {
            test(reg_ci_flag, FLAG_IC_LAST);
            je(regular_store, T_NEAR);
            if (!oc_tail) mov(reg_oc_flag, ptr[param1 + GET_OFF(oc_flag)]);
            test(reg_oc_flag, FLAG_OC_LAST);
            je(regular_store, T_NEAR);
            store_output(true, tail);
            jmp(store_done, T_NEAR);
        }

        L(regular_store);
        store_output(false, oc_tail);

        L(store_done);
    }

    if (oc_tail) {
        pop(reg_oc_blocks);
        base_post_ops_data_offset -= reg64_size;
    }
}

inline void jit_avx2_conv_fwd_kernel_f32::solve_common(int oc_blocks) {
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int n_oi = jcp.ow / ur_w;
    int iw = jcp.iw;
    int kw = jcp.kw;
    int str_w = jcp.stride_w;

    int l_pad = jcp.l_pad;
    int r_pad = nstl::max(0, jcp.r_pad);
    int r_pad1 = calculate_end_padding(l_pad, ur_w * n_oi, iw, str_w,
            calculate_extended_filter_size(kw, jcp.dilate_w));
    if (r_pad1 > 0) n_oi--;

    if (l_pad > 0) {
        n_oi--;
        if (n_oi < 0 && r_pad1 > 0)
            width_blk_step(ur_w, l_pad, r_pad1, oc_blocks); // "lrpad"
        else
            width_blk_step(ur_w, l_pad, 0, oc_blocks); // "lpad"
        add(reg_input, get_input_offset(0, filter_w_to_input(0, ur_w, l_pad)));
        add(reg_output, get_output_offset(0, ur_w));
    }

    Label ow_loop;
    xor_(oi_iter, oi_iter);

    if (n_oi > 0) {
        L(ow_loop);

        width_blk_step(ur_w, 0, 0, oc_blocks); // "middle"
        add(reg_input, get_input_offset(0, filter_w_to_input(0, ur_w)));
        add(reg_output, get_output_offset(0, ur_w));

        inc(oi_iter);
        cmp(oi_iter, n_oi);
        jl(ow_loop, T_NEAR);
    }

    if (r_pad1 > 0 && n_oi >= 0) {
        width_blk_step(ur_w, 0, r_pad1, oc_blocks); // "rpad"
        add(reg_input, get_input_offset(0, filter_w_to_input(0, ur_w)));
        add(reg_output, get_output_offset(0, ur_w));
    }

    if (ur_w_tail != 0)
        width_blk_step(ur_w_tail, 0, r_pad, oc_blocks); // "tail"
}

void jit_avx2_conv_fwd_kernel_f32::generate() {
    this->preamble();

    if (postops_injector_)
        postops_injector_->push_post_ops_data_on_stack(this->param1, GET_OFF(post_ops_binary_rhs_arg_vec), reg_input, reg_output);

    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    if (jcp.with_bias) mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_ci_flag, ptr[this->param1 + GET_OFF(flags)]);
    mov(reg_oc_blocks, ptr[this->param1 + GET_OFF(oc_blocks)]);

    if (is_src_layout_nxc())
        mov(reg_channel, ptr[param1 + GET_OFF(reduce_work)]);

    int nb_oc_tail = jcp.nb_oc % jcp.nb_oc_blocking;

    Label tail, exit;

    if (jcp.nb_oc > jcp.nb_oc_blocking) {
        cmp(reg_oc_blocks, jcp.nb_oc_blocking);
        jne(nb_oc_tail ? tail : exit, T_NEAR);

        solve_common(jcp.nb_oc_blocking);
        jmp(exit, T_NEAR);

        if (nb_oc_tail) {
            L(tail);
            cmp(reg_oc_blocks, nb_oc_tail);
            jne(exit, T_NEAR);
            solve_common(nb_oc_tail);
        }

        L(exit);
    } else if (jcp.nb_oc == jcp.nb_oc_blocking) {
        solve_common(jcp.nb_oc_blocking);
    } else {
        solve_common(nb_oc_tail);
    }

    if (postops_injector_)
        postops_injector_->reset_stack_pointer();

    this->postamble();

    if (jcp.with_eltwise) postops_injector_->prepare_table();
}

status_t jit_avx2_conv_fwd_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr) {
    if (!mayiuse(avx)) return status::unimplemented;
    jcp.isa = mayiuse(avx2) ? avx2 : avx;

    jcp.nthr = dnnl_get_max_threads();

    jcp.prop_kind = cd.prop_kind;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();
    jcp.ndims = ndims;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

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

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd);
//    bool kernel_outside_src = false || ext_kw <= jcp.l_pad
//            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad || ext_kh <= jcp.b_pad
//            || ext_kd <= jcp.f_pad || ext_kd <= jcp.back_pad;
//    if (kernel_outside_src) return status::unimplemented;

    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_ncx = pick(ndims - 3, ncw, nchw, ncdhw);
    const auto dat_tag_nCx8c = pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    auto wei_tag_OIxio = with_groups
            ? pick(ndims - 3, gOIw8i8o, gOIhw8i8o, gOIdhw8i8o)
            : pick(ndims - 3, OIw8i8o, OIhw8i8o, OIdhw8i8o);
    auto wei_tag_Oxio = with_groups ? pick(ndims - 3, gOwi8o, gOhwi8o, gOdhwi8o)
                                    : pick(ndims - 3, Owi8o, Ohwi8o, Odhwi8o);

    jcp.src_tag
            = src_d.mb_stride_relaxed_match(dat_tag_ncx, dat_tag_nxc, dat_tag_nCx8c);
    jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag_OIxio, wei_tag_Oxio);
    jcp.dst_tag = dst_d.mb_stride_relaxed_match(dat_tag_nxc, dat_tag_nCx8c);

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());

    bool is_data_layout_nxc
            = everyone_is(dat_tag_nxc, jcp.src_tag, jcp.dst_tag);

    // Disable this kernel on high width 1d object as gemm performs better until
    // optimizations can be made to fix it.
    if (is_data_layout_nxc && ndims == 3 && jcp.ow > 11 * 1024)
        return status::unimplemented;

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    const auto &post_ops = attr.post_ops_;

    jcp.with_sum = post_ops.find(primitive_kind::sum) != -1;
    const int eltwise_ind = post_ops.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    const int binary_ind = post_ops.find(primitive_kind::binary);
    jcp.with_binary = binary_ind != -1;
    jcp.with_depthwise = post_ops.find(primitive_kind::depthwise) != -1;
    jcp.with_quantization = post_ops.find(primitive_kind::quantization) != -1;

    jcp.post_ops = post_ops;

    const int simd_w = 8;
    const bool flat = jcp.ic < simd_w;
    const bool mimo = !flat;

    /* Grouped channel offset to support 'non-blocked data' format for
     * convolution sizes with '(input_channel / ngroups) < simd' */
    jcp.nonblk_group_off
            = one_of(jcp.src_tag, ncw, nchw, ncdhw) && jcp.ngroups > 1 ? jcp.ic
                                                                       : 1;

    bool ok_to_pad_channels = true && !is_data_layout_nxc && jcp.ngroups == 1;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        if (mimo) jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    if (jcp.with_eltwise || jcp.with_binary || jcp.with_depthwise || jcp.with_quantization)
        if (!mayiuse(avx2)) return status::unimplemented;

    using namespace injector;
    static constexpr bool sum_at_pos_0_only = true;
    static constexpr bool sum_requires_scale_one = true;
    static constexpr bool sum_requires_zp_zero = true;
    const bool post_ops_ok_ = post_ops_ok({avx2, {eltwise, binary, sum, depthwise, quantization},
            jcp.post_ops, &dst_d, sum_at_pos_0_only, sum_requires_scale_one,
            sum_requires_zp_zero});
    if (!post_ops_ok_) return status::unimplemented;

    bool args_ok = true
            && IMPLICATION(flat,
                    jcp.wei_tag == wei_tag_Oxio
                            && ((jcp.src_tag == dat_tag_ncx
                                        && jcp.dst_tag == dat_tag_nCx8c)
                                    || (jcp.src_tag == dat_tag_nxc
                                            && jcp.dst_tag == dat_tag_nxc)))
            && IMPLICATION(mimo,
                    jcp.wei_tag == wei_tag_OIxio
                            && ((jcp.src_tag == dat_tag_nCx8c
                                        && jcp.dst_tag == dat_tag_nCx8c)
                                    || (jcp.src_tag == dat_tag_nxc
                                            && jcp.dst_tag == dat_tag_nxc)))
            && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= dst_d.padded_dims()[1];
    if (!args_ok) return status::unimplemented;

    jcp.ur_h = 1; /* no code-unrolling by h so far */
    jcp.ur_w = 3;

    jcp.oc_block = simd_w;
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);

    jcp.nb_oc_blocking = 4; /* the optimal value for the kernel */

    // Intel AVX and Intel AVX2 kernels need 2 and 1 temporary YMMs, respectively
    // Thus, we can only assign 14 or 15 YMMs for data storage
    const int num_avail_regs = mayiuse(avx2) ? 15 : 14;
    if (!mayiuse(avx2)) {
        if ((jcp.nb_oc_blocking + 1) * jcp.ur_w > num_avail_regs) {
            // current register assignment requires more YMMs than available
            // adjust one of nb_oc_block, ur_w preserving to ur_w >= l_pad
            if (jcp.ur_w > jcp.l_pad && jcp.ur_w > 1)
                jcp.ur_w -= 1;
            else {
                for (int b = 3; b > 1; b--) {
                    if (jcp.nb_oc % b == 0) {
                        jcp.nb_oc_blocking = b;
                        break;
                    }
                }
                if ((jcp.nb_oc_blocking + 1) * jcp.ur_w > num_avail_regs) {
                    // No optimal size for 'nb_oc_blocking' with regards to
                    // 'nb_oc', default to only unroll by 'ur_w'.
                    jcp.nb_oc_blocking = 1;
                }
            }
        }
    }

    if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    args_ok = true && IMPLICATION(!is_data_layout_nxc, jcp.oc % simd_w == 0)
            && jcp.l_pad <= jcp.ur_w
            && IMPLICATION(jcp.kw > 7,
                    (jcp.t_pad == 0 && jcp.l_pad == 0)
                            || (jcp.stride_w == 1 && jcp.stride_h == 1))
            && IMPLICATION(mimo && !is_data_layout_nxc, jcp.ic % simd_w == 0);
    if (!args_ok) return status::unimplemented;

    jcp.ic_tail = is_data_layout_nxc ? jcp.ic % simd_w : 0;
    jcp.oc_tail = is_data_layout_nxc
            ? jcp.oc % simd_w
            : (jcp.with_binary ? jcp.oc_without_padding % simd_w : 0);

    int r_pad_no_tail = nstl::max(0,
            calculate_end_padding(jcp.l_pad, jcp.ow - jcp.ur_w_tail, jcp.iw,
                    jcp.stride_w, ext_kw));

    if (r_pad_no_tail > jcp.ur_w * jcp.stride_w && jcp.ow / jcp.ur_w > 1) {
        /* recalculate ur_w, nb_oc_blocking and ur_w_tail */
        jcp.ur_w = nstl::min(r_pad_no_tail / jcp.stride_w + jcp.ur_w_tail,
                nstl::min(jcp.ow, num_avail_regs / 2));
        jcp.nb_oc_blocking = (num_avail_regs - jcp.ur_w) / jcp.ur_w;
        jcp.ur_w_tail = jcp.ow % jcp.ur_w;
        /* check again ... */
        r_pad_no_tail = nstl::max(0,
                calculate_end_padding(jcp.l_pad, jcp.ow - jcp.ur_w_tail, jcp.iw,
                        jcp.stride_w, ext_kw));
        if (jcp.ur_w < nstl::max(jcp.l_pad, r_pad_no_tail))
            return status::unimplemented;
    }
    assert(jcp.nb_oc_blocking > 0);
    assert(jcp.ur_w * (jcp.nb_oc_blocking + 1) <= num_avail_regs);

    jcp.ic_block = flat ? jcp.ic : simd_w;
    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);

    jcp.nb_ic_blocking = 12;
    jcp.nb_ic_blocking_max = 16;

    /* adjust the thread decomposition
     * to improve the perf for small problem size
     * the threshold L1_cache_size is empirical 
     * simply set the thread as 4 for now
     * TODO: Add get_thr_eff func to get the optimal thread number*/
    size_t wei_size = (size_t)sizeof(float) * jcp.ic * jcp.oc * jcp.kh * jcp.kw
            * jcp.kd;
    size_t inp_size = (size_t)jcp.typesize_in * jcp.mb * jcp.ic * jcp.ih
            * jcp.iw * jcp.id;
    size_t out_size = (size_t)jcp.typesize_out * jcp.mb * jcp.oc * jcp.oh
            * jcp.ow * jcp.od;
    size_t total_size = jcp.ngroups * (wei_size + inp_size + out_size);

    const unsigned int L1_cache_size = platform::get_per_core_cache_size(1);

    if (jcp.ngroups < jcp.nthr && total_size < L1_cache_size) {
        jcp.nthr = nstl::min(jcp.nthr, 4);
    }

    return status::success;
}

void jit_avx2_conv_fwd_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding)
        scratchpad.book<float>(key_conv_padded_bias, jcp.oc);
}

void jit_avx2_conv_bwd_data_kernel_f32::compute_loop(
        int ur_w, int l_overflow, int r_overflow) {
    int kw = jcp.kw;
    int ow = jcp.ow;

    int oc_block = jcp.oc_block;
    int nb_ic_block = jcp.nb_ic_blocking;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;
    int oc_tail = jcp.oc_tail;
    int ic_tail = jcp.ic_tail;

    Label kd_loop, skip_kd_loop;
    Label oc_loop, skip_oc_loop;

    for (int ii = 0; ii < nb_ic_block; ii++)
        for (int jj = 0; jj < ur_w; jj++) {
            uni_vpxor(Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj),
                    Ymm(ur_w * ii + jj));
        }

    if (oc_tail) {
        base_post_ops_data_offset += reg64_size;
        push(reg_long_offt);
        mov(reg_reduce_work, ptr[param1 + GET_OFF(reduce_work)]);
    }

    if (one_of(jcp.ndims, 3, 4)) {
        cmp(reg_channel_work, 0);
        jle(skip_oc_loop, T_NEAR);
        xor_(reg_channel, reg_channel);

        mov(aux_reg_ddst_oc_loop, reg_ddst);
        mov(aux_reg_kernel_oc_loop, reg_kernel);

        L(oc_loop);
        mov(aux_reg_ddst, aux_reg_ddst_oc_loop);
        mov(aux_reg_kernel, aux_reg_kernel_oc_loop);
    }

    if (jcp.ndims == 5) {
        assert(jcp.nb_oc_blocking == 1);
        base_post_ops_data_offset += reg64_size;
        push(oi_iter);

        mov(reg_ki, ptr[this->param1 + GET_OFF(kd_padding)]);
        cmp(reg_ki, 0);
        jle(skip_kd_loop, T_NEAR);

        mov(aux_reg_dst_d, reg_ddst);
        mov(aux_reg_ker_d, ptr[this->param1 + GET_OFF(filt)]);

        L(kd_loop);
        mov(kj, ptr[this->param1 + GET_OFF(kh_padding)]);
    } else {
        mov(kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_ddst, aux_reg_dst_d);
        mov(aux_reg_kernel, aux_reg_ker_d);
    }

    Label kh_loop, skip_kh_loop;
    cmp(kj, 0);
    jle(skip_kh_loop, T_NEAR);

    L(kh_loop);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = get_iw_start(ki, l_overflow); // 0;
            int jj_end = get_iw_end(ur_w, ki, r_overflow); // ur_w;

            auto compute = [=](int cur_oc_blk) {
                for (int ofm2 = 0; ofm2 < cur_oc_blk; ofm2++) {
                    for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                        int aux_output_offset = get_ddst_offset(
                                0, filter_w_to_ddst(ki, jj, jcp.l_pad), ofm2);
                        vbroadcastss(Ymm(nb_ic_block * ur_w + jj / stride_w),
                                ptr[aux_reg_ddst + aux_output_offset]);
                    }

                    for (int ii = 0; ii < nb_ic_block; ii++) {
                        vmovups(ymm15,
                                ptr[aux_reg_kernel
                                        + get_kernel_offset(0, ii, ki, ofm2)]);
                        for (int jj = jj_start; jj < jj_end; jj += stride_w)
                            vfmadd231ps(Ymm(ur_w * ii + jj),
                                    Ymm(nb_ic_block * ur_w + jj / stride_w),
                                    ymm15);
                    }
                }
            };

            if (oc_tail) {
                if (jcp.oc == oc_tail)
                    compute(oc_tail);
                else {
                    Label oc_blk_tail, oc_blk_done;
                    cmp(reg_reduce_work, oc_block);
                    jl(oc_blk_tail, T_NEAR);
                    compute(oc_block);
                    jmp(oc_blk_done, T_NEAR);

                    L(oc_blk_tail);
                    compute(oc_tail);

                    L(oc_blk_done);
                }
            } else {
                compute(oc_block);
            }
        }

        add(aux_reg_kernel, get_kernel_offset(0, 0, stride_h * kw, 0));
        sub(aux_reg_ddst, get_ddst_offset(0, (jcp.dilate_h + 1) * ow, 0));

        dec(kj);
        cmp(kj, 0);
        jg(kh_loop, T_NEAR);
    }
    L(skip_kh_loop);

    if (jcp.ndims == 5) {
        sub(aux_reg_dst_d,
                get_ddst_offset(0, (jcp.dilate_d + 1) * jcp.oh * ow, 0));
        add(aux_reg_ker_d, get_kernel_offset(0, 0, jcp.kw * jcp.kh, 0));

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_loop, T_NEAR);
        L(skip_kd_loop);

        pop(oi_iter);
        base_post_ops_data_offset -= reg64_size;
    }

    if (one_of(jcp.ndims, 3, 4)) {
        int ddst_oc_shift = get_ddst_offset(1, 0, 0);
        int kernel_oc_shift = get_kernel_offset(1, 0, 0, 0);

        add(aux_reg_ddst_oc_loop, ddst_oc_shift);
        add(aux_reg_kernel_oc_loop, kernel_oc_shift);

        if (oc_tail) sub(reg_reduce_work, jcp.oc_block);
        inc(reg_channel);
        cmp(reg_channel, reg_channel_work);
        jl(oc_loop, T_NEAR);

        L(skip_oc_loop);
        mov(reg_channel, ptr[param1 + GET_OFF(channel)]);
    }

    if (oc_tail) {
        pop(reg_long_offt);
        base_post_ops_data_offset -= reg64_size;
    }

    auto load_store_dsrc = [=](bool is_tail) {
        mov(reg_channel, ptr[param1 + GET_OFF(channel)]);
        Label no_update_label, skip_post_ops;
        cmp(reg_channel, 0);
        je(no_update_label, T_NEAR);

        for (int ii = 0; ii < nb_ic_block; ii++)
            for (int jj = 0; jj < ur_w; jj++) {
                if (is_tail && ii == nb_ic_block - 1)
                    load_bytes(Ymm(15), reg_dsrc, get_dsrc_offset(ii, jj),
                            ic_tail * sizeof(float));
                else
                    vmovups(Ymm(15),
                            make_safe_addr(reg_dsrc, get_dsrc_offset(ii, jj),
                                    reg_long_offt));
                vaddps(Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj), Ymm(15));
            }

        jmp(skip_post_ops, T_NEAR);

        L(no_update_label);

        const auto &p = attr_.post_ops_;
        std::size_t post_ops_data_offset = 0;
        int depthwise_inj_idx = 0;
        for (int i = 0; i < p.len(); i++) {
            auto& post_op = p.entry_[i];
            if (post_op.is_depthwise()) {
                base_post_ops_data_offset += reg64_size;
                push(reg_d_weights);

                mov(reg_d_weights, ptr[this->rsp + base_post_ops_data_offset + post_ops_data_offset]);
                add(reg_d_weights, ptr[this->param1 + GET_OFF(ic_off)]);

                for (int ii = 0; ii < nb_ic_block; ii++) {
                    depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                            ur_w * ii, ur_w * ii + ur_w, reg_d_weights, reg_d_weights);

                    add(reg_d_weights, jcp.ic_block * sizeof(float));
                }
                pop(reg_d_weights);
                base_post_ops_data_offset -= reg64_size;

                post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
                depthwise_inj_idx++;
            }
        }
        L(skip_post_ops);

        for (int ii = 0; ii < nb_ic_block; ii++)
            for (int jj = 0; jj < ur_w; jj++) {
                if (is_tail && ii == nb_ic_block - 1)
                    store_bytes(Ymm(ur_w * ii + jj), reg_dsrc,
                            get_dsrc_offset(ii, jj), ic_tail * sizeof(float));
                else
                    vmovups(make_safe_addr(reg_dsrc, get_dsrc_offset(ii, jj),
                                    reg_long_offt),
                            Ymm(ur_w * ii + jj));
            }
    };

    if (ic_tail) {
        Label load_store_tail, load_store_done;
        mov(reg_ci_flag, ptr[param1 + GET_OFF(flags)]);
        test(reg_ci_flag, FLAG_IC_LAST);
        jne(load_store_tail, T_NEAR);

        load_store_dsrc(false);
        jmp(load_store_done, T_NEAR);

        L(load_store_tail);
        load_store_dsrc(true);

        L(load_store_done);
    } else {
        load_store_dsrc(false);
    }
}

void jit_avx2_conv_bwd_data_kernel_f32::generate() {
    const auto &p = attr_.post_ops_;
    for (int i = 0; i < p.len(); i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_depthwise()) {
            depthwise_injectors.push_back(new jit_uni_depthwise_injector_f32<avx2>(
                    this,
                    post_op
            ));
        }
    }

    preamble();

    std::size_t post_ops_pointers_count = 0;
    for (int i = 0; i < p.len(); i++) {
        if (p.entry_[i].is_depthwise() || p.entry_[i].is_quantization()) {
            post_ops_pointers_count++;
        }
    }

    if (post_ops_pointers_count != 0) {
        sub(rsp, post_ops_pointers_count * sizeof(float *));

        auto aux_reg0 = reg_dsrc;
        auto aux_reg1 = reg_ddst;

        mov(aux_reg0, ptr[this->param1 + GET_OFF(post_ops_binary_rhs_arg_vec)]);
        for (size_t i = 0; i < post_ops_pointers_count; i++) {
            mov(aux_reg1, ptr[aux_reg0 + i * sizeof(float *)]);
            mov(ptr[rsp + i * sizeof(float *)], aux_reg1);
        }
    }

    mov(reg_dsrc, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_ddst, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_channel, ptr[param1 + GET_OFF(channel)]);
    mov(reg_channel_work, ptr[param1 + GET_OFF(ch_blocks)]);

    int ddst_shift = get_ddst_offset(0, filter_w_to_ddst(0, jcp.ur_w), 0);
    int dsrc_shift = get_dsrc_offset(0, jcp.ur_w);

    const int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);

    int l_overflow = nstl::max(0, (ext_kw - 1 - jcp.l_pad) / jcp.stride_w);
    int r_overflow = nstl::max(
            0, (ext_kw - 1 - nstl::max(0, jcp.r_pad)) / jcp.stride_w);
    int r_overflow1 = nstl::max(
            0, (ext_kw - 1 - jcp.r_pad - jcp.ur_w_tail) / jcp.stride_w);

    int n_oi = jcp.iw / jcp.ur_w;
    if (r_overflow1 > 0) n_oi--;

    if (jcp.ur_w == jcp.iw) {
        compute_loop(jcp.ur_w, l_overflow, r_overflow);
    } else if (n_oi == 0) {
        compute_loop(jcp.ur_w, l_overflow, r_overflow1);
        add(reg_dsrc, dsrc_shift);
        add(reg_ddst, ddst_shift);
        if (jcp.ur_w_tail != 0) compute_loop(jcp.ur_w_tail, 0, r_overflow);
    } else {
        xor_(oi_iter, oi_iter);
        if (l_overflow > 0) {
            compute_loop(jcp.ur_w, l_overflow, 0);
            add(reg_dsrc, dsrc_shift);
            add(reg_ddst, ddst_shift);
            inc(oi_iter);
        }

        if ((l_overflow <= 0 && n_oi > 0) || (l_overflow > 0 && n_oi > 1)) {
            Label ow_loop;
            L(ow_loop);
            {
                compute_loop(jcp.ur_w, 0, 0);
                add(reg_dsrc, dsrc_shift);
                add(reg_ddst, ddst_shift);
                inc(oi_iter);
                cmp(oi_iter, n_oi);
                jl(ow_loop, T_NEAR);
            }
        }

        if (r_overflow1 > 0) {
            compute_loop(jcp.ur_w, 0, r_overflow1);
            add(reg_dsrc, dsrc_shift);
            add(reg_ddst, ddst_shift);
        }

        if (jcp.ur_w_tail != 0) compute_loop(jcp.ur_w_tail, 0, r_overflow);
    }

    if (post_ops_pointers_count != 0) {
        add(rsp, post_ops_pointers_count * sizeof(float *));
    }

    this->postamble();
}

bool jit_avx2_conv_bwd_data_kernel_f32::post_ops_ok(const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;
    if (p.len() > 1)
        return false;

    auto all_post_ops_supported = [&]() {
        bool ok = true;

        for (int i = 0; i < p.len(); i++) {
            ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::depthwise);
        }
        return ok;
    };

    return all_post_ops_supported();
}

status_t jit_avx2_conv_bwd_data_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d,
        const primitive_attr_t &attr) {
    if (!mayiuse(avx2)) return status::unimplemented;

    jcp.nthr = dnnl_get_max_threads();

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;

    int ndims = diff_src_d.ndims();
    jcp.ndims = ndims;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? diff_src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : diff_src_d.dims()[ndims - 2];
    jcp.iw = diff_src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims - 2];
    jcp.ow = diff_dst_d.dims()[ndims - 1];

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

    if ((jcp.dilate_w != 0 && jcp.stride_w != 1)
            || (jcp.dilate_d != 0 && jcp.stride_d != 1)
            || (jcp.dilate_h != 0 && jcp.stride_h != 1))
        return status::unimplemented;

    const int simd_w = 8;

    if (!post_ops_ok(attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    if (!mayiuse(avx2)) {
        for (int i = 0; i < p.len(); i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_depthwise()) {
                return status::unimplemented;
            }
        }
    }
    jcp.post_ops = p;

    /* derivatives */
    jcp.idp = jcp.id + 2 * jcp.f_pad;
    jcp.ihp = jcp.ih + 2 * jcp.t_pad;
    jcp.iwp = jcp.iw + 2 * jcp.l_pad;
    jcp.ohp = jcp.oh; /* do we really need */
    jcp.owp = jcp.ow; /* padded output ??? */

    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_nCx8c = pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    auto wei_tag = with_groups
            ? pick(ndims - 3, gOIw8o8i, gOIhw8o8i, gOIdhw8o8i)
            : pick(ndims - 3, OIw8o8i, OIhw8o8i, OIdhw8o8i);

    jcp.src_tag = diff_src_d.mb_stride_relaxed_match(dat_tag_nxc, dat_tag_nCx8c);
    jcp.dst_tag = diff_dst_d.mb_stride_relaxed_match(dat_tag_nxc, dat_tag_nCx8c);
    jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);

    jcp.typesize_in = types::data_type_size(diff_src_d.data_type());
    jcp.typesize_out = types::data_type_size(diff_dst_d.data_type());

    bool is_data_layout_nxc
            = everyone_is(dat_tag_nxc, jcp.src_tag, jcp.dst_tag);
    bool ok_to_pad_channels = true && !is_data_layout_nxc && jcp.ngroups == 1;

    /* gemm-based convolution performs better in these cases */
    if (jcp.ic < simd_w && jcp.kw > 3 && jcp.stride_w > 1)
        return status::unimplemented;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    jcp.ic_block = (!is_data_layout_nxc && jcp.ic % simd_w) ? 1 : simd_w;
    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);

    jcp.ic_tail = is_data_layout_nxc ? jcp.ic % simd_w : 0;
    jcp.oc_tail = is_data_layout_nxc ? jcp.oc % simd_w : 0;

    jcp.oc_block = simd_w;
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);

    jcp.ur_h = 1; /* no code-unrolling by h so far */
    jcp.nb_ic_blocking = 1;
    jcp.nb_oc_blocking = 1;
    jcp.ur_w = 1;

    if (one_of(ndims, 3, 4) && jcp.ow < 40)
        jcp.nb_oc_blocking = jcp.ow < 15 ? 4 : 2;

    auto required_dat_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx8c;

    bool args_ok = true && jcp.stride_w == jcp.stride_h && jcp.stride_d == 1
            && IMPLICATION(!is_data_layout_nxc,
                    jcp.ic % simd_w == 0 && jcp.oc % simd_w == 0)
            && jcp.ic <= diff_src_d.padded_dims()[1]
            && jcp.oc <= diff_dst_d.padded_dims()[1]
            && jcp.dst_tag == required_dat_tag
            && jcp.src_tag == required_dat_tag && jcp.wei_tag == wei_tag;
    if (!args_ok) return status::unimplemented;

    const int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    const int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    const int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);

    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd);

    bool kernel_outside_src = false || ext_kw <= jcp.l_pad
            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad || ext_kh <= jcp.b_pad
            || ext_kd <= jcp.f_pad || ext_kd <= jcp.back_pad;
    if (kernel_outside_src) return status::unimplemented;

    int l_overflow = nstl::max(0, (ext_kw - 1 - jcp.l_pad) / jcp.stride_w);

    const int max_regs = 15; /* Maximum number of registers available for
                                result accumulation and delta dst data.
                                One additional register is reserved for weights
                                data. */

    /* Find the best blocking with maximum number of fma instructions
       per ur_w * nb_ic_blocking compute loops. Number of required registers
       is num_regs = ur_w * nb_ic_blocking + ur_w / stride_w <= max_regs.
       ur_w must be divisible by stride_w */
    if (jcp.stride_w + 1 > max_regs) /* Minimal possible registers
                                         distribution exceeds max_regs */
        return status::unimplemented;

    int best_nfmas = 0;
    for (int b = 1; b <= 4; b++) {
        if (jcp.nb_ic % b != 0) continue;

        for (int u = jcp.stride_w; u * b + u / jcp.stride_w <= max_regs
                && u < jcp.iw + jcp.stride_w;
                u += jcp.stride_w) {
            int ur_w = nstl::min(u, jcp.iw);
            /* maximum 1 step with l_overflow so far */
            if (l_overflow * jcp.stride_w > ur_w && ur_w != jcp.iw) continue;
            int nfmas = div_up(ur_w, jcp.stride_w) * b;
            if (nfmas > best_nfmas
                    || (nfmas == best_nfmas && jcp.ur_w < ur_w)) {
                jcp.ur_w = ur_w;
                jcp.nb_ic_blocking = b;
                best_nfmas = nfmas;
            }
        }
    }
    if (best_nfmas == 0) /* can't find appropriate blocking */
        return status::unimplemented;

    jcp.ur_w_tail = jcp.iw % jcp.ur_w;

    int r_overflow_no_tail = nstl::max(
            0, (ext_kw - 1 - jcp.r_pad - jcp.ur_w_tail) / jcp.stride_w);

    bool tails_not_ok = false
            /* maximum 1 ur_w block with r_overflow so far */
            || r_overflow_no_tail * jcp.stride_w > jcp.ur_w
            /* ur_w must be a multiple of stride */
            || ((jcp.iw > jcp.ur_w) && (jcp.ur_w % jcp.stride_w != 0))
            /* r_pad must not extend beyond ur_w_tail */
            || ((jcp.iw > jcp.ur_w) && (jcp.r_pad + jcp.ur_w_tail < 0));
    if (tails_not_ok) return status::unimplemented;

    /* adjust the thread decomposition
     * to improve the perf for small problem size
     * the threshold L1_cache_size is empirical 
     * simply set the thread to 4 for now
     * TODO: Add get_thr_eff func to get optimal thread number */
    size_t wei_size = (size_t)sizeof(float) * jcp.ic * jcp.oc * jcp.kh * jcp.kw
            * jcp.kd;
    size_t inp_size = (size_t)jcp.typesize_in * jcp.mb * jcp.ic * jcp.ih
            * jcp.iw * jcp.id;
    size_t out_size = (size_t)jcp.typesize_out * jcp.mb * jcp.oc * jcp.oh
            * jcp.ow * jcp.od;
    size_t total_size = jcp.ngroups * (wei_size + inp_size + out_size);
    const unsigned int L1_cache_size = platform::get_per_core_cache_size(1);

    if (jcp.ngroups < jcp.nthr && total_size < L1_cache_size) {
        jcp.nthr = nstl::min(jcp.nthr, 4);
    }

    return status::success;
}

void jit_avx2_conv_bwd_data_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    UNUSED(scratchpad);
    UNUSED(jcp);
}

void jit_avx2_conv_bwd_weights_kernel_f32::generate() {
    this->preamble();

    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    compute_oh_loop_common();
    this->postamble();
}

status_t jit_avx2_conv_bwd_weights_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &diff_weights_d,
        const memory_desc_wrapper &diff_dst_d) {
    if (!mayiuse(avx2)) return status::unimplemented;

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();
    jcp.ndims = ndims;

    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims - 2];
    jcp.ow = diff_dst_d.dims()[ndims - 1];

    jcp.kd = (ndims == 5) ? diff_weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : diff_weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = diff_weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_ncx = pick(ndims - 3, ncw, nchw, ncdhw);
    const auto dat_tag_nCx8c = pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    auto wei_tag_OIxio = with_groups
            ? pick(ndims - 3, gOIw8i8o, gOIhw8i8o, gOIdhw8i8o)
            : pick(ndims - 3, OIw8i8o, OIhw8i8o, OIdhw8i8o);
    auto wei_tag_Oxio = with_groups ? pick(ndims - 3, gOwi8o, gOhwi8o, gOdhwi8o)
                                    : pick(ndims - 3, Owi8o, Ohwi8o, Odhwi8o);

    jcp.src_tag
            = src_d.matches_one_of_tag(dat_tag_ncx, dat_tag_nxc, dat_tag_nCx8c);
    jcp.wei_tag
            = diff_weights_d.matches_one_of_tag(wei_tag_OIxio, wei_tag_Oxio);
    jcp.dst_tag = diff_dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx8c);

    bool is_data_layout_nxc
            = everyone_is(dat_tag_nxc, jcp.src_tag, jcp.dst_tag);

    jcp.with_bias = cd.diff_bias_desc.format_kind != format_kind::undef;

    const bool flat = jcp.ic == 3;
    const bool mimo = !flat;

    const int simd_w = 8;

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.r_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw));
    jcp.b_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh));
    jcp.back_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd));

    const int max_h_pad = ext_kh;
    const int max_w_pad = ext_kw;
    const bool boundaries_ok = true && jcp.t_pad < max_h_pad
            && jcp.b_pad < max_h_pad && jcp.l_pad < max_w_pad
            && jcp.r_pad < max_w_pad && jcp.f_pad == 0 && jcp.back_pad == 0;
    if (!boundaries_ok) return status::unimplemented;

    bool ok_to_pad_channels = true && !is_data_layout_nxc && jcp.ngroups == 1;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        if (mimo) jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    jcp.ic_tail = is_data_layout_nxc ? jcp.ic % simd_w : 0;
    jcp.oc_tail = is_data_layout_nxc ? jcp.oc % simd_w : 0;

    bool args_ok = true
            && IMPLICATION(flat,
                    jcp.wei_tag == wei_tag_Oxio
                            && ((jcp.src_tag == dat_tag_ncx
                                        && jcp.dst_tag == dat_tag_nCx8c)
                                    || (jcp.src_tag == dat_tag_nxc
                                            && jcp.dst_tag == dat_tag_nxc)))
            && IMPLICATION(mimo,
                    jcp.wei_tag == wei_tag_OIxio
                            && ((jcp.src_tag == dat_tag_nCx8c
                                        && jcp.dst_tag == dat_tag_nCx8c)
                                    || (jcp.src_tag == dat_tag_nxc
                                            && jcp.dst_tag == dat_tag_nxc)))
            && IMPLICATION(mimo && !is_data_layout_nxc, jcp.ic % simd_w == 0)
            && IMPLICATION(!is_data_layout_nxc, jcp.oc % simd_w == 0)
            && jcp.kw < 14 && jcp.kh <= jcp.t_pad + jcp.ih /* [bwd_w:r1] */
            && jcp.kh <= jcp.ih /* [bwd_w:r2] */
            && jcp.kd <= jcp.f_pad + jcp.id && jcp.kd <= jcp.id
            && jcp.t_pad < jcp.kh /* XXX: must fix the kernel! */
            && jcp.dilate_d == 0 && jcp.dilate_h == 0 && jcp.dilate_w == 0
            && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= diff_dst_d.padded_dims()[1];
    if (!args_ok) return status::unimplemented;

    jcp.ic_block = flat ? jcp.ic : simd_w;
    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);

    jcp.oc_block = simd_w;
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    return status::success;
}

void jit_avx2_conv_bwd_weights_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    if (jcp.with_bias && (jcp.oc_without_padding % jcp.oc_block != 0)) {
        const size_t nelems_padded_bias
                = jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block);
        scratchpad.book<float>(key_conv_padded_bias, nelems_padded_bias);
    }
}

inline void jit_avx2_conv_bwd_weights_kernel_f32::od_step_comeback_pointers() {
    Label kd_comeback_loop;
    mov(kj, jcp.kd); //FIXME (Anton): this works only if f_pad = back_pad = 0
    L(kd_comeback_loop);
    {
        sub(aux_reg_input, get_input_offset(0, jcp.iw * jcp.ih));
        sub(aux_reg_kernel, get_kernel_offset(jcp.kw * jcp.kh, 0));
        dec(kj);
        cmp(kj, 0);
        jg(kd_comeback_loop, T_NEAR);
    }
}

inline void jit_avx2_conv_bwd_weights_kernel_f32::oh_step_comeback_pointers() {
    mov(kj, reg_kh);
    Label kh_comeback_loop;
    L(kh_comeback_loop);
    {
        sub(reg_input, get_input_offset(0, jcp.iw));
        sub(reg_kernel, get_kernel_offset(jcp.kw, 0));
        dec(kj);
        cmp(kj, 0);
        jg(kh_comeback_loop, T_NEAR);
    }
}

inline void jit_avx2_conv_bwd_weights_kernel_f32::compute_ic_block_step(
        int ur_w, int pad_l, int pad_r, int ic_block_step, int input_offset,
        int kernel_offset, int output_offset) {

    if (ic_block_step <= 0) return;

    const int kw = jcp.kw;
    const int oc_tail = jcp.oc_tail;

    if (oc_tail) {
        push(reg_kh);
        mov(reg_ci_flag, ptr[param1 + GET_OFF(flags)]);
    }

    auto load_compute_store = [=](bool is_tail) {
        for (int i_kw = 0; i_kw < kw; i_kw++)
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                size_t off = get_kernel_offset(i_kw, i_ic) + kernel_offset;
                if (is_tail)
                    load_bytes(Ymm(i_kw * ic_block_step + i_ic), reg_kernel,
                            off, oc_tail * sizeof(float));
                else
                    vmovups(Ymm(i_kw * ic_block_step + i_ic),
                            yword[reg_kernel + off]);
            }

        for (int i_ur = 0; i_ur < ur_w; i_ur++) {
            if (is_tail)
                load_bytes(Ymm(kw * ic_block_step + 0), reg_output,
                        get_output_offset(0, i_ur) + output_offset,
                        oc_tail * sizeof(float));
            else
                vmovups(Ymm(kw * ic_block_step + 0),
                        yword[reg_output + get_output_offset(0, i_ur)
                                + output_offset]);

            for (int i_kw = 0; i_kw < kw; i_kw++) {
                int i_iw = i_ur * jcp.stride_w + i_kw;
                if (i_iw - pad_l < 0
                        || i_iw > (ur_w - 1) * jcp.stride_w + kw - 1 - pad_r)
                    continue;
                for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                    size_t i_off = get_input_offset(i_ic, i_iw - pad_l);
                    vbroadcastss(Ymm(kw * ic_block_step + 1),
                            make_safe_addr(reg_input, i_off, reg_long_offt));
                    vfmadd231ps(Ymm(i_kw * ic_block_step + i_ic),
                            Ymm(kw * ic_block_step + 0),
                            Ymm(kw * ic_block_step + 1));
                }
            }
        }

        for (int i_kw = 0; i_kw < kw; i_kw++)
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                size_t off = get_kernel_offset(i_kw, i_ic) + kernel_offset;
                if (is_tail)
                    store_bytes(Ymm(i_kw * ic_block_step + i_ic), reg_kernel,
                            off, oc_tail * sizeof(float));

                else
                    vmovups(yword[reg_kernel + off],
                            Ymm(i_kw * ic_block_step + i_ic));
            }
    };

    if (oc_tail) {
        Label load_tail, load_done;
        test(reg_ci_flag, FLAG_OC_LAST);
        jne(load_tail, T_NEAR);

        load_compute_store(false);
        jmp(load_done, T_NEAR);

        L(load_tail);
        load_compute_store(true);

        L(load_done);
    } else {
        load_compute_store(false);
    }

    if (oc_tail) pop(reg_kh);
}

inline void jit_avx2_conv_bwd_weights_kernel_f32::compute_oh_step_disp() {
    int ic_block_step;
    if (one_of(jcp.src_tag, ncw, nchw, ncdhw)) {
        ic_block_step = jcp.kw >= 5 ? 1 : jcp.ic_block;
    } else if (one_of(jcp.src_tag, nwc, nhwc, ndhwc)) {
        ic_block_step = jcp.kw > 7 ? 1 : jcp.kw > 3 ? 2 : jcp.kw > 1 ? 4 : 8;
        if (jcp.ic_block % ic_block_step != 0) {
            ic_block_step = jcp.ic_block < ic_block_step ? jcp.ic_block : 1;
        }
        if (jcp.ic < ic_block_step) ic_block_step = jcp.ic;
    } else {
        ic_block_step = jcp.kw > 7 ? 1 : jcp.kw > 3 ? 2 : jcp.kw > 1 ? 4 : 8;
    }

    const int max_ur_w = jcp.ow > 56 ? 14 : 28;

    if (jcp.ow <= max_ur_w || one_of(jcp.src_tag, nwc, nhwc, ndhwc))
        compute_oh_step_unroll_ow(ic_block_step, max_ur_w);
    else
        compute_oh_step_common(ic_block_step, max_ur_w);

    if (jcp.ndims == 5) {
        od_step_comeback_pointers();
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    } else {
        oh_step_comeback_pointers();
    }
}

inline void jit_avx2_conv_bwd_weights_kernel_f32::compute_oh_step_unroll_ow(
        int ic_block_step, int max_ur_w) {
    UNUSED(max_ur_w);

    const int r_pad = jcp.r_pad;
    const int ic_tail = jcp.ic_tail;
    const int ic_block = jcp.ic_block;
    const int ic_block_step_tail = jcp.ic % ic_block_step;
    const size_t inp_icblk_stride = get_input_offset(ic_block_step, 0);

    if (ic_tail) {
        push(reg_ih_count);
        mov(reg_channel, ptr[param1 + GET_OFF(channel)]);
    }

    Label kd_loop;
    if (jcp.ndims == 5) {
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);
        mov(ki, jcp.kd);
        L(kd_loop);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    Label kh_loop, kh_loop_ic_tail, kh_loop_done;
    if (ic_tail) {
        cmp(reg_channel, ic_block);
        jl(kh_loop_ic_tail, T_NEAR);
    }

    L(kh_loop);
    {
        xor_(b_ic, b_ic);
        Label ic_block_loop;
        L(ic_block_loop);
        {
            compute_ic_block_step(
                    jcp.ow, jcp.l_pad, r_pad, ic_block_step, 0, 0, 0);
            safe_add(reg_input, inp_icblk_stride, reg_long_offt);
            add(reg_kernel, get_kernel_offset(0, ic_block_step));
            add(b_ic, ic_block_step);
            cmp(b_ic, ic_block);
            jl(ic_block_loop, T_NEAR);
        }
        add(reg_input,
                get_input_offset(0, jcp.iw) - get_input_offset(ic_block, 0));
        add(reg_kernel, get_kernel_offset((jcp.kw - 1), 0));
        dec(kj);
        cmp(kj, 0);
        jg(kh_loop, T_NEAR);
    }
    jmp(kh_loop_done, T_NEAR);

    L(kh_loop_ic_tail);
    {
        Label ic_block_loop, ic_block_loop_done;

        cmp(reg_channel, ic_block_step);
        jl(ic_block_loop_done, T_NEAR);

        mov(b_ic, ic_tail);
        L(ic_block_loop);
        {
            compute_ic_block_step(
                    jcp.ow, jcp.l_pad, r_pad, ic_block_step, 0, 0, 0);
            safe_add(reg_input, inp_icblk_stride, reg_long_offt);
            add(reg_kernel, get_kernel_offset(0, ic_block_step));
            sub(b_ic, ic_block_step);
            cmp(b_ic, ic_block_step);
            jge(ic_block_loop, T_NEAR);
        }

        L(ic_block_loop_done);

        if (ic_block_step_tail) {
            compute_ic_block_step(
                    jcp.ow, jcp.l_pad, r_pad, ic_block_step_tail, 0, 0, 0);
            add(reg_input, get_input_offset(ic_block_step_tail, 0));
            add(reg_kernel, get_kernel_offset(0, ic_block_step_tail));
        }

        add(reg_input,
                get_input_offset(0, jcp.iw) - get_input_offset(ic_tail, 0));
        add(reg_kernel,
                get_kernel_offset(0, ic_block - ic_tail)
                        + get_kernel_offset((jcp.kw - 1), 0));
        dec(kj);
        cmp(kj, 0);
        jg(kh_loop_ic_tail, T_NEAR);
    }

    L(kh_loop_done);

    if (jcp.ndims == 5) {
        add(aux_reg_input, get_input_offset(0, jcp.ih * jcp.iw));
        add(aux_reg_kernel, get_kernel_offset(jcp.kh * jcp.kw, 0));
        dec(ki);
        cmp(ki, 0);
        jg(kd_loop, T_NEAR);
    }
    if (ic_tail) pop(reg_ih_count);
}

inline void jit_avx2_conv_bwd_weights_kernel_f32::compute_oh_step_common(
        int ic_block_step, int max_ur_w) {
    // TODO: suppport channel tails for nxc format

    const int ic_block = jcp.ic_block;
    const int stride_w = jcp.stride_w;
    Label kd_loop;

    const int r_pad = jcp.r_pad;

    int ur_w = nstl::min(jcp.ow, max_ur_w);
    int ur_w_trips = jcp.ow / ur_w;
    int ur_w_tail = jcp.ow % ur_w;
    if ((ur_w_tail == 0 && r_pad != 0) || r_pad >= ur_w_tail) {
        if (ur_w_trips > 1) {
            ur_w_tail += ur_w;
            ur_w_trips--;
        } else {
            ur_w_tail += (ur_w - ur_w / 2);
            ur_w = ur_w / 2;
        }
    }

    int input_comeback
            = get_input_offset(0, ur_w_trips * ur_w * stride_w - jcp.l_pad);
    int output_comeback = get_output_offset(0, ur_w_trips * ur_w);

    if (jcp.ndims == 5) {
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);
        mov(ki, jcp.kd);
        L(kd_loop);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    Label kh_loop;
    L(kh_loop);
    {
        xor_(b_ic, b_ic);
        Label ic_block_loop;
        L(ic_block_loop);
        {
            if (jcp.l_pad != 0) {
                ur_w_trips--;
                compute_ic_block_step(
                        ur_w, jcp.l_pad, 0, ic_block_step, 0, 0, 0);
                add(reg_input,
                        get_input_offset(0, ur_w * stride_w - jcp.l_pad));
                add(reg_output, get_output_offset(0, ur_w));
            }

            if (ur_w_trips > 0) {
                xor_(reg_ur_w_trips, reg_ur_w_trips);
                Label ow_block_loop;
                L(ow_block_loop);
                {
                    compute_ic_block_step(ur_w, 0, 0, ic_block_step, 0, 0, 0);
                    add(reg_output, get_output_offset(0, ur_w));
                    add(reg_input, get_input_offset(0, ur_w * stride_w));

                    inc(reg_ur_w_trips);
                    cmp(reg_ur_w_trips, ur_w_trips);
                    jl(ow_block_loop, T_NEAR);
                }
            }

            if (ur_w_tail > 0)
                compute_ic_block_step(
                        ur_w_tail, 0, r_pad, ic_block_step, 0, 0, 0);

            sub(reg_input, input_comeback);
            sub(reg_output, output_comeback);

            size_t inp_icblk_stride = get_input_offset(ic_block_step, 0);
            safe_add(reg_input, inp_icblk_stride, reg_long_offt);
            add(reg_kernel, get_kernel_offset(0, ic_block_step));

            add(b_ic, ic_block_step);
            cmp(b_ic, jcp.ic_block);
            jl(ic_block_loop, T_NEAR);
        }
        add(reg_input,
                get_input_offset(0, jcp.iw) - get_input_offset(ic_block, 0));
        add(reg_kernel, get_kernel_offset((jcp.kw - 1), 0));
        dec(kj);
        cmp(kj, 0);
        jg(kh_loop, T_NEAR);
    }

    if (jcp.ndims == 5) {
        add(aux_reg_input, get_input_offset(0, jcp.ih * jcp.iw));
        add(aux_reg_kernel, get_kernel_offset(jcp.kh * jcp.kw, 0));
        dec(ki);
        cmp(ki, 0);
        jg(kd_loop, T_NEAR);
    }
}

inline void jit_avx2_conv_bwd_weights_kernel_f32::compute_oh_loop_common() {
    const int t_pad = jcp.t_pad;
    const int stride_h = jcp.stride_h;
    int b_pad = jcp.b_pad;

    Label oh_tpad_loop, oh_loop, oh_loop_end;

    mov(reg_kh, jcp.kh);
    xor_(reg_ih_count, reg_ih_count);
    xor_(reg_oj, reg_oj);
    if (t_pad > 0) {
        assert(jcp.kh <= t_pad + jcp.ih); /* [bwd_w:r1] */
        mov(reg_kh, jcp.kh <= t_pad + jcp.ih ? jcp.kh - t_pad : jcp.ih);
        add(reg_kernel, get_kernel_offset(t_pad * jcp.kw, 0));

        L(oh_tpad_loop);
        {
            compute_oh_step_disp();
            add(reg_output, get_output_offset(0, jcp.ow));
            sub(reg_kernel, get_kernel_offset(stride_h * jcp.kw, 0));

            inc(reg_oj);
            add(reg_ih_count, stride_h);
            add(reg_kh, stride_h);

            /* the overlap between input and kernel may not reach kernel size.
             * so far we do not support that (until we put constant here) */
            const int final_inp_ker_overlap = jcp.kh; /* [bwd_w:r2] */
            cmp(reg_kh, final_inp_ker_overlap);
            jl(oh_tpad_loop, T_NEAR);
        }

        if (t_pad % stride_h != 0) {
            int inp_corr = stride_h - t_pad % stride_h;
            add(reg_kernel, get_kernel_offset(inp_corr * jcp.kw, 0));
            add(reg_input, get_input_offset(0, inp_corr * jcp.iw));
        }
    }
    cmp(reg_ih_count, jcp.ih + t_pad - jcp.kh + 1);
    jge(oh_loop_end, T_NEAR);
    cmp(reg_oj, jcp.oh);
    jge(oh_loop, T_NEAR);

    mov(reg_kh, jcp.kh);
    L(oh_loop);
    {
        compute_oh_step_disp();
        add(reg_input, get_input_offset(0, stride_h * jcp.iw));
        add(reg_output, get_output_offset(0, jcp.ow));

        inc(reg_oj);
        add(reg_ih_count, stride_h);

        cmp(reg_ih_count, jcp.ih + t_pad - jcp.kh + 1);
        jge(oh_loop_end, T_NEAR);

        cmp(reg_oj, jcp.oh);
        jl(oh_loop, T_NEAR);
    }
    L(oh_loop_end);
    if (b_pad > 0) {
        Label oh_bpad_loop, oh_bpad_loop_end;
        cmp(reg_oj, jcp.oh);
        jge(oh_bpad_loop_end, T_NEAR);

        mov(reg_kh, jcp.ih + t_pad);
        sub(reg_kh, reg_ih_count);
        L(oh_bpad_loop);
        {
            compute_oh_step_disp();
            add(reg_input, get_input_offset(0, stride_h * jcp.iw));
            add(reg_output, get_output_offset(0, jcp.ow));

            sub(reg_kh, stride_h);
            cmp(reg_kh, 0);
            jle(oh_bpad_loop_end, T_NEAR);

            inc(reg_oj);
            cmp(reg_oj, jcp.oh);
            jl(oh_bpad_loop, T_NEAR);
        }
        L(oh_bpad_loop_end);
    }
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
