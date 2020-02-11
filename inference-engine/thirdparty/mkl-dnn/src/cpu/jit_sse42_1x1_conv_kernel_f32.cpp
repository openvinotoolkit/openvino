/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"

#include "jit_sse42_1x1_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_1x1_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

using namespace Xbyak;

void jit_sse42_1x1_conv_kernel_f32::generate_bcast_loop(int load_loop_blk)
{
    mov(aux1_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_output_data, reg_output_data);
    mov(bcast_loop_iter, reg_bcast_loop_work);

    Label bcast_loop;
    Label bcast_loop_tail;

    cmp(bcast_loop_iter, jcp.ur);
    jl(bcast_loop_tail, T_NEAR);

    L(bcast_loop); {
        assert(jcp.bcast_block % jcp.ur == 0);
        int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        for (int i = 0; i < num_substeps; i++) {
            generate_reduce_loop(load_loop_blk, jcp.ur);
            if (i < num_substeps - 1) {
                add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_substep);
                add(aux_reg_output_data, jcp.bcast_loop_output_substep);
            } else {
                add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_step
                        - (num_substeps - 1) * jcp.bcast_loop_bcast_substep);
                add(aux_reg_output_data, jcp.bcast_loop_output_step
                        - (num_substeps - 1) * jcp.bcast_loop_output_substep);
            }
        }
        sub(bcast_loop_iter, jcp.bcast_block);
        cmp(bcast_loop_iter, jcp.bcast_block);
        jge(bcast_loop, T_NEAR);
    }

    L(bcast_loop_tail);
    if (jcp.ur_tail) {
        Label bcast_loop_tail_out;
        cmp(bcast_loop_iter, 0);
        jz(bcast_loop_tail_out, T_NEAR);
        generate_reduce_loop(load_loop_blk, jcp.ur_tail);
        L(bcast_loop_tail_out);
    }
}

void jit_sse42_1x1_conv_kernel_f32::generate_reduce_loop(
        int load_loop_blk, int ur)
{
    auto reg_load = [=](int i, int n) {
        return Xmm(2*ur * load_loop_blk + 2*i + n + 1);
    };

    auto reg_accum = [=](int i, int j, int n) {
        return Xmm(n*load_loop_blk*ur + i*ur + j + 1);
    };

    auto bias_ptr = [=](int i, int n) {
        return ptr[reg_bias_data + sizeof(float) * jcp.oc_block * i + n*4*sizeof(float)];
    };

    auto bcast_ptr = [=](int u, int j) {
        assert(j < jcp.ur);
        assert(u <= jcp.reduce_loop_unroll);
        size_t offt;
        if (one_of(jcp.prop_kind,
                    forward_training, forward_inference, backward_data)) {
            assert(jcp.reduce_loop_unroll == (jcp.prop_kind == backward_data)
                    ? jcp.oc_block : jcp.ic_block);
            auto height = (jcp.prop_kind == backward_data) ? jcp.os : jcp.is;
            offt = (u == jcp.reduce_loop_unroll)
                ? (height + j) * jcp.reduce_loop_unroll
                : j * jcp.reduce_loop_unroll + u;
        } else
            offt = u * jcp.ic_block + j;
        return ptr[aux_reg_bcast_data + sizeof(float) * offt];
    };

    auto load_ptr = [=](int u, int i, int n) {
        size_t offt;
        size_t u0 = u % jcp.reduce_loop_unroll;
        size_t u1 = u / jcp.reduce_loop_unroll;
        switch (jcp.prop_kind) {
        case backward_data:
            offt = (i * jcp.oc_block + u0) * jcp.ic_block;
            break;
        case backward_weights:
            offt = (i * jcp.os + u0) * jcp.oc_block;
            break;
        default:
            offt = (i * jcp.ic + u0) * jcp.oc_block;
        }
        return ptr[aux_reg_load_data
            + u1 * jcp.reduce_loop_load_step + sizeof(float) * offt + n * 4 * sizeof(float)];
    };

    auto output_ptr = [=](int i, int j, int n) {
        switch (jcp.prop_kind) {
        case backward_data:
            return ptr[aux_reg_output_data +
                (i * jcp.is + j) * jcp.ic_block * sizeof(float) + n * 4 * sizeof(float)];
        case backward_weights:
            return ptr[aux_reg_output_data
                + (i ? reg_output_stride * i : 0) // TODO: Xbyak should allow 0 scale
                + sizeof(float) * jcp.oc_block * j + n * 4 * sizeof(float)];
        default:
            if (jcp.with_dw_conv)
                return ptr[aux_reg_output_data +
                           (i * jcp_dw.kh * jcp.ow + j) * jcp.oc_block * sizeof(float) + n*4*sizeof(float)];
            else
                return ptr[aux_reg_output_data +
                    (i * jcp.os + j) * jcp.oc_block * sizeof(float) + n*4*sizeof(float)];
        }
    };

    auto init = [=]() {
        Label init_done;
        Label init_zero;

        if (jcp.with_bias && one_of(jcp.prop_kind, forward_training,
                    forward_inference)) {
            test(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            jz(init_zero);

            for (int i = 0; i < load_loop_blk; i++)
                for (int j = 0; j < ur; ++j) {
                    movups(reg_accum(i, j, 0), bias_ptr(i, 0));
                    movups(reg_accum(i, j, 1), bias_ptr(i, 1));
                }
            jmp(init_done);
        }

        L(init_zero);
        for (int i = 0; i < load_loop_blk; ++i)
            for (int j = 0; j < ur; ++j) {
                auto r0 = reg_accum(i, j, 0);
                auto r1 = reg_accum(i, j, 1);
                xorps(r0, r0);
                xorps(r1, r1);
            }

        L(init_done);

        // load weights
        for (int i = 0; i < load_loop_blk; ++i) {
            movups(reg_load(i, 0), load_ptr(0, i, 0));
            movups(reg_load(i, 1), load_ptr(0, i, 1));
        }

        movss(reg_bcast, bcast_ptr(0, 0));
        shufps(reg_bcast, reg_bcast, 0);
    }; // init()

    auto store = [=]() {
        Label store_noadd;

        if (!jcp.with_sum) {
            test(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            jnz(store_noadd, T_NEAR);
        }

        for (int j = 0; j < ur; ++j)
            for (int i = 0; i < load_loop_blk; ++i) {
                auto r0 = reg_accum(i, j, 0);
                auto r1 = reg_accum(i, j, 1);
                addps(r0, output_ptr(i, j, 0));
                addps(r1, output_ptr(i, j, 1));
            }

        L(store_noadd);

        Label store_no_postops;
        test(reg_reduce_pos_flag, FLAG_REDUCE_LAST);
        jz(store_no_postops, T_NEAR);

        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        const auto &p = attr_.post_ops_;

        int end_idx = jcp.with_dw_conv ? p.find(primitive_kind::convolution) : p.len_;
        for (int i = 0; i < end_idx; i++) {
            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors[eltwise_inj_idx]->compute_vector_range(1, 2 * ur * load_loop_blk + 1);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data));

                add(reg_d_weights, reg_oc_off);
                add(reg_d_bias, reg_oc_off);

                for (int j = 0; j < load_loop_blk; ++j) {
                    for (int k = 0; k < 2; k++) {
                        int start_idx = reg_accum(j, 0, k).getIdx();
                        int end_idx = reg_accum(j, ur, k).getIdx();

                        depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                                start_idx, end_idx, reg_d_weights, reg_d_bias);

                        add(reg_d_weights, 4 * sizeof(float));
                        add(reg_d_bias, 4 * sizeof(float));
                    }
                }

                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.crop_low_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.crop_high_data));

                add(reg_d_weights, reg_oc_off);
                add(reg_d_bias, reg_oc_off);

                for (int j = 0; j < load_loop_blk; ++j) {
                    for (int k = 0; k < 2; k++) {
                        uni_vmovups(xmm_d_weights, ptr[reg_d_weights + (j * jcp.oc_block + k * jcp.oc_block / 2) * sizeof(float)]);
                        uni_vmovups(xmm_d_bias, ptr[reg_d_bias + (j * jcp.oc_block + k * jcp.oc_block / 2) * sizeof(float)]);

                        for (int u = 0; u < ur; u++) {
                            Xmm xmm_dst = reg_accum(j, u, k);

                            uni_vmaxps(xmm_dst, xmm_dst, xmm_d_weights);
                            uni_vminps(xmm_dst, xmm_dst, xmm_d_bias);
                        }
                    }
                }

                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.input_scale_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.input_shift_data));

                add(reg_d_weights, reg_oc_off);
                add(reg_d_bias, reg_oc_off);

                for (int j = 0; j < load_loop_blk; ++j) {
                    for (int k = 0; k < 2; k++) {
                        uni_vmovups(xmm_d_weights, ptr[reg_d_weights + (j * jcp.oc_block + k * jcp.oc_block / 2) * sizeof(float)]);
                        uni_vmovups(xmm_d_bias, ptr[reg_d_bias + (j * jcp.oc_block + k * jcp.oc_block / 2) * sizeof(float)]);

                        for (int u = 0; u < ur; u++) {
                            Xmm xmm_dst = reg_accum(j, u, k);

                            uni_vfmadd213ps(xmm_dst, xmm_d_weights, xmm_d_bias);
                            uni_vroundps(xmm_dst, xmm_dst, 0);
                        }
                    }
                }

                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.output_scale_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.output_shift_data));

                add(reg_d_weights, reg_oc_off);
                add(reg_d_bias, reg_oc_off);

                for (int j = 0; j < load_loop_blk; ++j) {
                    for (int k = 0; k < 2; k++) {
                        uni_vmovups(xmm_d_weights, ptr[reg_d_weights + (j * jcp.oc_block + k * jcp.oc_block / 2) * sizeof(float)]);
                        uni_vmovups(xmm_d_bias, ptr[reg_d_bias + (j * jcp.oc_block + k * jcp.oc_block / 2) * sizeof(float)]);

                        for (int u = 0; u < ur; u++) {
                            Xmm xmm_dst = reg_accum(j, u, k);

                            uni_vfmadd213ps(xmm_dst, xmm_d_weights, xmm_d_bias);
                        }
                    }
                }
            }
        }

        L(store_no_postops);

        for (int j = 0; j < ur; ++j)
            for (int i = 0; i < load_loop_blk; ++i) {
                movups(output_ptr(i, j, 0), reg_accum(i, j, 0));
                movups(output_ptr(i, j, 1), reg_accum(i, j, 1));
            }
    };

    auto fma_block = [=](bool last_block) {
        for (int u = 0; u < jcp.reduce_loop_unroll; ++u) {
            for (int j = 0; j < ur; ++j) {
                for (int i = 0; i < load_loop_blk; ++i) {
                    mulps(reg_load(i, 0), reg_bcast);
                    mulps(reg_load(i, 1), reg_bcast);
                    addps(reg_accum(i, j, 0), reg_load(i, 0));
                    addps(reg_accum(i, j, 1), reg_load(i, 1));

                    if (j == ur - 1 && !(last_block && u == jcp.reduce_loop_unroll - 1)) {
                        movups(reg_load(i, 0), load_ptr(u + 1, i, 0));
                        movups(reg_load(i, 1), load_ptr(u + 1, i, 1));
                    }
                }
                if (j < ur - 1) {
                    movss(reg_bcast, bcast_ptr(u, j + 1));
                    shufps(reg_bcast, reg_bcast, 0);
                }
            } // for ur
            if (!last_block || u < jcp.reduce_loop_unroll - 1) {
                movss(reg_bcast, bcast_ptr(u + 1, 0));
                shufps(reg_bcast, reg_bcast, 0);
            }
        } // for reduce_loop_unroll
    };

    Label reduce_loop;
    Label reduce_loop_tail;

    mov(aux_reg_load_data, reg_load_data);
    mov(aux_reg_bcast_data, aux1_reg_bcast_data);

    init();

    mov(reduce_loop_iter, reg_reduce_loop_work);
    sub(reduce_loop_iter, jcp.reduce_loop_unroll);
    jle(reduce_loop_tail, T_NEAR);

    L(reduce_loop); {
        fma_block(false);
        add(aux_reg_bcast_data, jcp.reduce_loop_bcast_step);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        jg(reduce_loop, T_NEAR);
    }

    L(reduce_loop_tail);
    fma_block(true);

    store();
} // reduce_loop()

void jit_sse42_1x1_conv_kernel_f32::generate_diff_bias_loop(int load_loop_blk)
{
    if (!jcp.with_bias || jcp.prop_kind != backward_weights)
        return;

    Label diff_bias_loop, diff_bias_loop_out, diff_bias_init_out;
    Label diff_bias_load;

    auto diff_bias_ptr = [=](int i, int n) {
        return ptr[reg_diff_bias_data + i * jcp.oc_block * sizeof(float)+ 4*n*sizeof(float)];
    };

    auto load_ptr = [=](int u, int i, int n) {
        return ptr[aux_reg_load_data
            + (i * jcp.os + u) * jcp.oc_block * sizeof(float) + 4*n*sizeof(float)];
    };

    auto diff_bias_reg = [=](int i, int n) { return Xmm(2*i + n + 1); };

    mov(reg_diff_bias_data, ptr[rsp + reg_diff_bias_data_stack_offt]);
    cmp(reg_diff_bias_data, 0);
    je(diff_bias_loop_out, T_NEAR);

    test(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
    jz(diff_bias_load, T_NEAR);

    for (int i = 0; i < load_loop_blk; ++i) {
        auto r0 = diff_bias_reg(i, 0);
        auto r1 = diff_bias_reg(i, 1);
        xorps(r0, r0);
        xorps(r1, r1);
    }
    jmp(diff_bias_init_out, T_NEAR);

    L(diff_bias_load);
    for (int i = 0; i < load_loop_blk; ++i) {
        movups(diff_bias_reg(i, 0), diff_bias_ptr(i, 0));
        movups(diff_bias_reg(i, 1), diff_bias_ptr(i, 1));
    }

    L(diff_bias_init_out);
    mov(aux_reg_load_data, reg_load_data);
    mov(reduce_loop_iter, reg_reduce_loop_work);
    L(diff_bias_loop); {
        for(int u = 0; u < jcp.reduce_loop_unroll; ++u)
            for (int i = 0; i < load_loop_blk; ++i) {
                addps(diff_bias_reg(i, 0), load_ptr(u, i, 0));
                addps(diff_bias_reg(i, 1), load_ptr(u, i, 1));
            }
        assert(jcp.reduce_dim % jcp.reduce_loop_unroll == 0);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        jnz(diff_bias_loop, T_NEAR);
    }

    for (int i = 0; i < load_loop_blk; i++) {
        movups(diff_bias_ptr(i, 0), diff_bias_reg(i, 0));
        movups(diff_bias_ptr(i, 1), diff_bias_reg(i, 1));
    }

    add(reg_diff_bias_data, load_loop_blk * jcp.oc_block * sizeof(float));
    mov(ptr[rsp + reg_diff_bias_data_stack_offt], reg_diff_bias_data);

    L(diff_bias_loop_out);
}

void jit_sse42_1x1_conv_kernel_f32::generate()
{
    const auto &p = attr_.post_ops_;
    int end_idx = jcp.with_dw_conv ? p.find(primitive_kind::convolution) : p.len_;
    for (int i = 0; i < end_idx; i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors.push_back(new jit_uni_eltwise_injector_f32<sse42>(
                    this,
                    post_op.eltwise.alg,
                    post_op.eltwise.alpha,
                    post_op.eltwise.beta
            ));
        } else if (post_op.is_depthwise()) {
            depthwise_injectors.push_back(new jit_uni_depthwise_injector_f32<sse42>(
                    this,
                    post_op.depthwise.alg
            ));
        }
    }

    preamble();

    mov(reg_bcast_data, ptr[param1 + GET_OFF(bcast_data)]);
    mov(reg_load_data, ptr[param1 + GET_OFF(load_data)]);
    mov(reg_output_data, ptr[param1 + GET_OFF(output_data)]);
    if (jcp.with_bias) {
        if (jcp.prop_kind == backward_weights) {
            sub(rsp, stack_space_needed);
            mov(reg_diff_bias_data, ptr[param1 + GET_OFF(bias_data)]);
            mov(ptr[rsp + reg_diff_bias_data_stack_offt], reg_diff_bias_data);
        } else
            mov(reg_bias_data, ptr[param1 + GET_OFF(bias_data)]);
    }

    mov(reg_load_loop_work, ptr[param1 + GET_OFF(load_dim)]);
    mov(reg_bcast_loop_work, ptr[param1 + GET_OFF(bcast_dim)]);
    mov(reg_reduce_loop_work, ptr[param1 + GET_OFF(reduce_dim)]);
    mov(reg_reduce_pos_flag, ptr[param1 + GET_OFF(first_last_flag)]);
    if (jcp.prop_kind == backward_weights)
        mov(reg_output_stride, ptr[param1 + GET_OFF(output_stride)]);
    mov(reg_oc_off, ptr[param1 + GET_OFF(oc_off)]);

    auto generate_load_loop_body = [=] (int load_loop_blk) {
        generate_bcast_loop(load_loop_blk);
        add(reg_load_data, load_loop_blk * jcp.load_loop_load_step);
        switch (jcp.prop_kind) {
        case forward_training:
        case forward_inference:
            add(reg_bias_data, load_loop_blk * jcp.oc_block * sizeof(float));
            if (jcp.with_dw_conv)
                add(reg_output_data,
                    load_loop_blk * jcp.ow * jcp.oc_block * sizeof(float));
            else
                add(reg_output_data,
                        load_loop_blk * jcp.os * jcp.oc_block * sizeof(float));
            break;
        case backward_data:
            add(reg_output_data,
                    load_loop_blk * jcp.is * jcp.ic_block * sizeof(float));
            break;
        case backward_weights:
            for (int i = 0; i < load_loop_blk; i++)
                add(reg_output_data, reg_output_stride);
            break;
        default:
            assert(!"invalid prop_kind");
        }
        sub(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
        add(reg_oc_off, load_loop_blk * jcp.oc_block * sizeof(float));
    };

    Label load_loop_blk_8;
    Label load_loop_blk_16;
    Label load_loop_blk_24;
    Label load_loop_blk_end;

    cmp(reg_load_loop_work, 8);
    jle(load_loop_blk_8, T_NEAR);

    cmp(reg_load_loop_work, 32);
    je(load_loop_blk_16, T_NEAR);

    cmp(reg_load_loop_work, 16);
    jle(load_loop_blk_16, T_NEAR);

    L(load_loop_blk_24); {
        generate_diff_bias_loop(3);
        generate_load_loop_body(3);
        cmp(reg_load_loop_work, 32);
        je(load_loop_blk_16);
        cmp(reg_load_loop_work, 24);
        jge(load_loop_blk_24);
    }

    cmp(reg_load_loop_work, 8);
    jle(load_loop_blk_8, T_NEAR);

    L(load_loop_blk_16); {
        generate_diff_bias_loop(2);
        generate_load_loop_body(2);
        cmp(reg_load_loop_work, 16);
        jge(load_loop_blk_16);
    }

    L(load_loop_blk_8); {
        cmp(reg_load_loop_work, 0);
        je(load_loop_blk_end, T_NEAR);
        generate_diff_bias_loop(1);
        generate_load_loop_body(1);
    }

    L(load_loop_blk_end);

    if (jcp.with_bias && jcp.prop_kind == backward_weights)
        add(rsp, stack_space_needed);

    postamble();

    for (auto& inj : eltwise_injectors)
        inj->prepare_table();
}

bool jit_sse42_1x1_conv_kernel_f32::post_ops_ok(
        jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    int dw_conv_idx = p.find(primitive_kind::convolution);
    bool with_dw_conv = dw_conv_idx != -1;

    auto all_post_ops_supported = [&]() {
        bool ok = true;

        int end_idx = with_dw_conv ? dw_conv_idx : p.len_;
        for (int i = 0; i < end_idx; i++) {
            ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::sum, primitive_kind::eltwise, primitive_kind::depthwise,
                    primitive_kind::quantization);
        }
        return ok;
    };
    auto contain = [&](mkldnn::impl::primitive_kind_t kind) { return p.find(kind, 0, dw_conv_idx) != -1; };
    auto position = [&](mkldnn::impl::primitive_kind_t kind) { return p.find(kind, 0, dw_conv_idx); };
    auto count = [&](mkldnn::impl::primitive_kind_t kind) { return p.count(kind, 0, dw_conv_idx); };

    return all_post_ops_supported() &&
           count(primitive_kind::sum) <= 1 &&
           IMPLICATION(contain(primitive_kind::sum), position(primitive_kind::sum) == 0) &&
           IMPLICATION(with_dw_conv, !contain(primitive_kind::sum));
}

status_t jit_sse42_1x1_conv_kernel_f32::init_conf(jit_1x1_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr)
{
    if (!mayiuse(sse42))
        return status::unimplemented;

    // TODO (Roma): this code is duplicated from the generic kernel; maybe the
    // configuration struct could do some stuff below
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int ndims = src_d.ndims();

    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[2];
    jcp.ow = dst_d.dims()[ndims - 1];

    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][0];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[0];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.src_fmt = src_d.format();
    jcp.with_bias = cd.bias_desc.format != memory_format::undef;

    jcp.src_dt = cd.src_desc.data_type;
    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;

    int dw_conv_ind = p.find(primitive_kind::convolution);
    jcp.with_dw_conv = dw_conv_ind != -1;
    if (jcp.with_dw_conv) {
        jcp.dw_conv_oh = jcp.oh;
        jcp.dw_conv_ow = jcp.ow;
        jcp.oh = p.entry_[dw_conv_ind].dw_conv.in_h;
        jcp.ow = p.entry_[dw_conv_ind].dw_conv.in_w;

        jcp.dw_conv_dst_dt = jcp.dst_dt;
        jcp.dst_dt = p.entry_[dw_conv_ind].dw_conv.in_dt;
    }

    jcp.with_sum = p.find(primitive_kind::sum, 0, dw_conv_ind) != -1;

    jcp.os = jcp.oh * jcp.ow;
    jcp.is = jcp.ih * jcp.iw;

    const int is_bwd_d = jcp.prop_kind == backward_data;
    memory_format_t weights_format = with_groups
        ? utils::pick(2 * ndims - 6 + is_bwd_d, gOIw8i8o, gOIw8o8i, gOIhw8i8o,
            gOIhw8o8i)
        : utils::pick(2 * ndims - 6 + is_bwd_d, OIw8i8o, OIw8o8i, OIhw8i8o,
            OIhw8o8i);

    bool args_ok = true
        && jcp.ngroups == 1
        && one_of(src_d.format(), nCw8c, nChw8c)
        && weights_d.format() == weights_format
        && one_of(cd.bias_desc.format, memory_format::undef, any, x)
        && one_of(dst_d.format(), nCw8c, nChw8c);
    if (!args_ok) return status::unimplemented;

    const int simd_w = 4;

    jcp.oc = rnd_up(jcp.oc, simd_w*2);
    jcp.ic = rnd_up(jcp.ic, simd_w*2);

    jcp.ic_block = jcp.oc_block = simd_w*2;

    args_ok = true
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic % jcp.ic_block == 0
        && jcp.t_pad == 0 && jcp.l_pad == 0
        && jcp.stride_w == 1 && jcp.stride_h == 1 // TODO: support some strides
        && jcp.kh == 1 && jcp.kw == 1;
    if (!args_ok) return status::unimplemented;

    jcp.ur = 1;

    int load_blocking{ 0 };
    int load_blocking_max{ 0 };
    int bcast_blocking{ 0 };
    int bcast_blocking_max{ 0 };
    int reduce_blocking{ 0 };

    if (one_of(jcp.prop_kind, forward_training, forward_inference)) {
        jcp.reduce_dim = jcp.ic;
        jcp.reduce_block = jcp.ic_block;

        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.with_dw_conv ? jcp.iw : jcp.is;
        jcp.bcast_block = jcp.ur;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
            = jcp.reduce_loop_unroll * jcp.is * sizeof(float);
        jcp.reduce_loop_load_step
            = jcp.reduce_loop_unroll * jcp.oc_block * sizeof(float);

        jcp.bcast_loop_output_step = jcp.ur * jcp.oc_block * sizeof(float);
        jcp.bcast_loop_output_substep = -1; // unused
        jcp.bcast_loop_bcast_step = jcp.ur * jcp.ic_block * sizeof(float);
        jcp.bcast_loop_bcast_substep = -1; // unused

        jcp.load_loop_load_step = jcp.ic * jcp.oc_block * sizeof(float);
        jcp.load_loop_iter_step = jcp.oc_block;

        load_blocking = jcp.with_dw_conv ? nstl::min(3 * jcp.load_block, jcp.oc) : 120; // assumes the kernel is jcp.ur x 3
        load_blocking_max = jcp.with_dw_conv ? nstl::min(3 * jcp.load_block, jcp.oc) : 144;
        bcast_blocking = 128; // affects load balancing across threads
        bcast_blocking_max = 192;
        reduce_blocking = 128; // affects L1$ utilization
    } else if (jcp.prop_kind == backward_data) {
        jcp.reduce_dim = jcp.oc;
        jcp.reduce_block = jcp.oc_block;

        jcp.load_dim = jcp.ic;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.os;
        jcp.bcast_block = jcp.ur;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
            = jcp.reduce_loop_unroll * jcp.os * sizeof(float);
        jcp.reduce_loop_load_step
            = jcp.reduce_loop_unroll * jcp.ic * sizeof(float);

        jcp.bcast_loop_output_step = jcp.ur * jcp.ic_block * sizeof(float);
        jcp.bcast_loop_output_substep = -1; // unused
        jcp.bcast_loop_bcast_step = jcp.ur * jcp.oc_block * sizeof(float);
        jcp.bcast_loop_bcast_substep = -1; // unused

        jcp.load_loop_load_step = jcp.oc_block * jcp.ic_block * sizeof(float);
        jcp.load_loop_iter_step = jcp.ic_block;

        load_blocking = 96; // assumes the kernel is jcp.ur x 3
        load_blocking_max = 144;
        bcast_blocking = 128; // affects load balancing across threads
        bcast_blocking_max = 196;
        reduce_blocking = 64; // affects L1$ utilization
    } else if (jcp.prop_kind == backward_weights) {
        jcp.reduce_dim = jcp.os;
        jcp.reduce_block = 1;

        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.ic;
        jcp.bcast_block = jcp.ic_block;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
            = jcp.reduce_loop_unroll * jcp.ic_block * sizeof(float);
        jcp.reduce_loop_load_step
            = jcp.reduce_loop_unroll * jcp.oc_block * sizeof(float);

        jcp.bcast_loop_output_step = jcp.oc_block * jcp.ic_block * sizeof(float);
        jcp.bcast_loop_output_substep = jcp.oc_block * jcp.ur * sizeof(float);
        jcp.bcast_loop_bcast_step = jcp.ic_block * jcp.is * sizeof(float);
        jcp.bcast_loop_bcast_substep = jcp.ur * sizeof(float);

        jcp.load_loop_load_step = jcp.oc_block * jcp.os * sizeof(float);
        jcp.load_loop_iter_step = jcp.oc_block;

        /* --- */

        load_blocking = div_up(jcp.load_dim, jcp.load_block);
        while (true) {
            if (load_blocking <= 32) break;
            else if (load_blocking % 2 == 0) load_blocking /= 2;
            else if (load_blocking % 3 == 0) load_blocking /= 3;
            else break;
        }
        load_blocking *= jcp.load_block;
        load_blocking_max = load_blocking;
        assert(jcp.load_dim % load_blocking == 0);

        bcast_blocking = div_up(jcp.bcast_dim, jcp.bcast_block);
        while (true) {
            if (bcast_blocking <= 9) break;
            else if (bcast_blocking % 2 == 0) bcast_blocking /= 2;
            else if (bcast_blocking % 3 == 0) bcast_blocking /= 3;
            else break;
        }
        bcast_blocking *= jcp.bcast_block;
        bcast_blocking_max = bcast_blocking;
        assert(jcp.bcast_dim % bcast_blocking == 0);

        reduce_blocking = 128; // affects L1$ utilization
    } else
        return status::unimplemented;

    assert(load_blocking);
    assert(load_blocking_max);
    assert(bcast_blocking);
    assert(bcast_blocking_max);
    assert(reduce_blocking);

    assert(jcp.bcast_block % jcp.ur == 0);
    jcp.ur_tail = jcp.bcast_dim % jcp.ur;

    jcp.nb_bcast_blocking = bcast_blocking / jcp.bcast_block;
    jcp.nb_bcast_blocking_max = bcast_blocking_max / jcp.bcast_block;
    jcp.nb_load_blocking = load_blocking / jcp.load_block;
    jcp.nb_load_blocking_max = load_blocking_max / jcp.load_block;
    jcp.nb_reduce_blocking = reduce_blocking / jcp.reduce_block;

    jcp.nb_bcast = jcp.with_dw_conv ? jcp.ih : div_up(jcp.bcast_dim, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.load_dim, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    return status::success;
}

void jit_sse42_1x1_conv_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp, const jit_conv_conf_t &jcp_dw) {
    using namespace mkldnn::impl::memory_tracking::names;

    if (jcp.prop_kind != backward_data && jcp.oc != jcp.oc_without_padding)
        scratchpad.book(key_conv_padded_bias, sizeof(float) * jcp.oc);

    if (jcp.with_dw_conv) {
        const int nthreads = mkldnn_get_max_threads();
        size_t dw_conv_buffer_size_ = (size_t)jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block * (jcp.oc / jcp.oc_block);
        scratchpad.book(key_dw_conv_buffer, sizeof(float) * dw_conv_buffer_size_ * nthreads);

        if (jcp.oc != jcp.oc_without_padding)
            scratchpad.book(key_dw_conv_padded_bias, sizeof(float) * jcp.oc);
    }
}

}
}
}
