// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ie_parallel.hpp>
#include <mkldnn_extension_utils.h>
#include "jit_generator.hpp"
#include "jit_uni_eltwise.hpp"
#include "utils/bfloat16.hpp"
#include "permute_utils.h"

using namespace InferenceEngine;
using namespace MKLDNNPlugin;
using namespace mkldnn;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;

#define GET_OFF(field) offsetof(jit_args_permute, field)

template <cpu::cpu_isa_t isa>
struct jit_uni_permute_kernel_f32 : public jit_uni_permute_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_permute_kernel_f32)

    explicit jit_uni_permute_kernel_f32(jit_permute_config_params jcp_) : jit_uni_permute_kernel(jcp_), jit_generator() {
        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);

        loop(jcp_.n);

        this->postamble();

        ker_ = (decltype(ker_))this->getCode();
    }

    void load(const Xbyak::Xmm &xmm, const Xbyak::Address &addr) {
        switch (jcp.data_size) {
            case 16: movups(xmm, addr); break;
            case 8: movsd(xmm, addr); break;
            case 4: movss(xmm, addr); break;
            case 2: pinsrw(xmm, addr, 0x0); break;
            case 1: pinsrb(xmm, addr, 0x0); break;
        }
    }

    void store(const Xbyak::Address &addr, const Xbyak::Xmm &xmm) {
        switch (jcp.data_size) {
            case 16: movups(addr, xmm); break;
            case 8: movsd(addr, xmm); break;
            case 4: movss(addr, xmm); break;
            case 2: pextrw(addr, xmm, 0x0); break;
            case 1: pextrb(addr, xmm, 0x0); break;
        }
    }

    void loop(int n) {
        mov(reg_work_amount, jcp.dst_block_dims[n]);

        Xbyak::Label main_loop_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label exit_label;

        if (n + 1 == jcp.ndims) {
            if (jcp.src_strides[n] == jcp.dst_strides[n] == 1) {
                uint32_t step = vlen / jcp.data_size;

                L(main_loop_label);
                {
                    cmp(reg_work_amount, step);
                    jl(tail_loop_label, T_NEAR);

                    uni_vmovups(vmm, ptr[reg_src]);
                    uni_vmovups(ptr[reg_dst], vmm);

                    add(reg_src, step * jcp.data_size);
                    add(reg_dst, step * jcp.data_size);
                    sub(reg_work_amount, step);

                    jmp(main_loop_label, T_NEAR);
                }
            }
        }

        L(tail_loop_label); {
            cmp(reg_work_amount, 0);
            je(exit_label, T_NEAR);

            if (n + 1 == jcp.ndims) {
                load(xmm, ptr[reg_src]);
                store(ptr[reg_dst], xmm);
            } else {
                aux_reg_src = reg_src;
                aux_reg_dst = reg_dst;
                push(aux_reg_src);
                push(aux_reg_dst);
                push(reg_work_amount);
                loop(n + 1);
                pop(reg_work_amount);
                pop(reg_dst);
                pop(reg_src);
            }

            add(reg_src, jcp.src_strides[n] * jcp.data_size);
            add(reg_dst, jcp.dst_strides[n] * jcp.data_size);
            sub(reg_work_amount, 1);

            jmp(tail_loop_label, T_NEAR);
        }

        L(exit_label);
    }

private:
    using Vmm = typename conditional3<isa == cpu::sse42, Xbyak::Xmm, isa == cpu::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    uint32_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_work_amount = r10;
    Xbyak::Reg64 aux_reg_src = r11;
    Xbyak::Reg64 aux_reg_dst = r12;

    Xbyak::Reg64 reg_params = abi_param1;

    Vmm vmm = Vmm(0);
    Xbyak::Xmm xmm = Xbyak::Xmm(0);
};

void PermuteUtils::prepareConfigParams() {
    jit_permute_config_params jcp;

    SizeVector tmp_order;
    for (size_t i = 0; i < optimizedParams.dst_block_order.size(); i++) {
        tmp_order.push_back(order[optimizedParams.dst_block_order[i]]);
    }

    SizeVector new_dst_block_strides = optimizedParams.dst_block_strides;
    SizeVector new_dst_block_order = optimizedParams.dst_block_order;
    SizeVector new_dst_block_dims = optimizedParams.dst_block_dims;
    SizeVector new_src_block_strides(optimizedParams.dst_block_strides.size());
    SizeVector mask(optimizedParams.dst_block_strides.size());

    for (int i = tmp_order.size() - 1; i >= 0; i--) {
        int pos = std::distance(std::find(
                optimizedParams.src_block_order.rbegin(), optimizedParams.src_block_order.rend(), tmp_order[i]), optimizedParams.src_block_order.rend() - 1);
        if (pos != -1) {
            new_src_block_strides[i] = optimizedParams.src_block_strides[pos];
            optimizedParams.src_block_order.erase(optimizedParams.src_block_order.begin() + pos);
            optimizedParams.src_block_strides.erase(optimizedParams.src_block_strides.begin() + pos);
            mask[i] = 0;
        } else {
            new_src_block_strides[i] = new_src_block_strides[tmp_order.size() - 1] * optimizedParams.dst_block_dims[tmp_order.size() - 1];
            mask[i] = 1;
            mask[tmp_order.size() - 1] = 1;
        }
    }
    if (!optimizedParams.src_block_order.empty()) {
        int pos = std::distance(tmp_order.begin(), std::find(tmp_order.begin(), tmp_order.end(), optimizedParams.src_block_order[0]));
        new_src_block_strides.insert(new_src_block_strides.begin() + pos,
                                  optimizedParams.src_block_strides[0]);
        new_dst_block_strides.insert(new_dst_block_strides.begin() + pos,
                                  new_dst_block_strides[pos] * optimizedParams.src_block_dims[optimizedParams.src_block_dims.size() - 1]);
        new_dst_block_order.insert(new_dst_block_order.begin() + pos,
                                   new_dst_block_order[pos]);
        new_dst_block_dims.insert(new_dst_block_dims.begin() + pos + 1,
                                  optimizedParams.src_block_dims[optimizedParams.src_block_dims.size() - 1]);
        new_dst_block_dims[pos] = div_up(new_dst_block_dims[pos], new_dst_block_dims[pos + 1]);
        optimizedParams.src_block_order.erase(optimizedParams.src_block_order.begin());
        optimizedParams.src_block_strides.erase(optimizedParams.src_block_strides.begin());
        mask.insert(mask.begin() + pos + 1, 1);
        mask[pos] = 1;
    }

    SizeVector sorted_src_strides;
    SizeVector sorted_dst_strides;
    SizeVector sorted_order;
    SizeVector sorted_dst_dims;

    //  support dynamic batch
    int batch_ord = std::distance(order.begin(), std::find(order.begin(), order.end(), 0));
    int batch_count = 0;
    int batch_pos = 0;
    for (size_t i = 0; i < new_dst_block_order.size(); i++) {
        if (new_dst_block_order[i] == batch_ord) {
            batch_count++;
            batch_pos = i;
        }
    }
    if (batch_count == 1) {
        sorted_src_strides.push_back(new_src_block_strides[batch_pos]);
        sorted_dst_strides.push_back(new_dst_block_strides[batch_pos]);
        sorted_order.push_back(new_dst_block_order[batch_pos]);
        sorted_dst_dims.push_back(new_dst_block_dims[batch_pos]);
        jcp.supported_dynamic_batch = true;
    }

    int n2 = 0;
    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i] == 0) {
            n2++;
            if (batch_count == 1 && new_dst_block_order[i] == batch_ord) {
                continue;
            }
            sorted_src_strides.push_back(new_src_block_strides[i]);
            sorted_dst_strides.push_back(new_dst_block_strides[i]);
            sorted_order.push_back(new_dst_block_order[i]);
            sorted_dst_dims.push_back(new_dst_block_dims[i]);
        }
    }
    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i] == 1) {
            sorted_src_strides.push_back(new_src_block_strides[i]);
            sorted_dst_strides.push_back(new_dst_block_strides[i]);
            sorted_order.push_back(new_dst_block_order[i]);
            sorted_dst_dims.push_back(new_dst_block_dims[i]);
        }
    }

    int max_threads = mkldnn_get_max_threads();
    const int n_max = 3;    //  max count dims for parallel
    int n = 0;
    int work_amount = sorted_dst_dims[0];
    for (size_t i = 1; i < sorted_dst_dims.size() && n < n_max; i++) {
        n++;
        if (work_amount >= 4 * max_threads) {   //  4 * max_threads is a specially selected value for best performance
            break;
        }
        work_amount *= sorted_dst_dims[i];
    }

    jcp.src_strides = sorted_src_strides;
    jcp.dst_strides = sorted_dst_strides;
    jcp.dst_block_dims = sorted_dst_dims;
    jcp.n = std::min(n, n2);
    jcp.ndims = sorted_order.size();
    jcp.data_size = optimizedParams.data_size;

    if (mayiuse(cpu::avx512_common)) {
        permute_kernel.reset(new jit_uni_permute_kernel_f32<cpu::avx512_common>(jcp));
    } else if (mayiuse(cpu::avx2)) {
        permute_kernel.reset(new jit_uni_permute_kernel_f32<cpu::avx2>(jcp));
    } else if (mayiuse(cpu::sse42)) {
        permute_kernel.reset(new jit_uni_permute_kernel_f32<cpu::sse42>(jcp));
    }
}

void PermuteUtils::optimizedExecute(const uint8_t* src_data, uint8_t* dst_data) {
    const auto &jcp = (*permute_kernel).jcp;

    SizeVector dst_dims = jcp.dst_block_dims;
    SizeVector dst_strides = jcp.dst_strides;
    SizeVector src_strides = jcp.src_strides;

    switch (jcp.n) {
        case 1:
            parallel_for(dst_dims[0], [&](int i0) {
                auto arg = jit_args_permute();

                size_t dst_off = i0 * dst_strides[0];
                size_t src_off = i0 * src_strides[0];
                arg.src = &src_data[src_off * jcp.data_size];
                arg.dst = &dst_data[dst_off * jcp.data_size];

                (*permute_kernel)(&arg);
            });
            break;
        case 2:
            parallel_for2d(dst_dims[0], dst_dims[1], [&](int i0, int i1) {
                auto arg = jit_args_permute();

                size_t dst_off = i0 * dst_strides[0] + i1 * dst_strides[1];
                size_t src_off = i0 * src_strides[0] + i1 * src_strides[1];
                arg.src = &src_data[src_off * jcp.data_size];
                arg.dst = &dst_data[dst_off * jcp.data_size];

                (*permute_kernel)(&arg);
            });
            break;
        case 3:
            parallel_for3d(dst_dims[0], dst_dims[1], dst_dims[2], [&](int i0, int i1, int i2) {
                auto arg = jit_args_permute();

                size_t dst_off = i0 * dst_strides[0] + i1 * dst_strides[1] + i2 * dst_strides[2];
                size_t src_off = i0 * src_strides[0] + i1 * src_strides[1] + i2 * src_strides[2];
                arg.src = &src_data[src_off * jcp.data_size];
                arg.dst = &dst_data[dst_off * jcp.data_size];

                (*permute_kernel)(&arg);
            });
            break;
    }
    return;
}
