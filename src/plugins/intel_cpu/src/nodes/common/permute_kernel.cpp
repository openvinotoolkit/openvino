// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_kernel.h"

#include <memory>
#include <vector>

#include "common/primitive_hashing_utils.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu_memcpy.h"
#include "dnnl_extension_utils.h"
#include "dnnl_types.h"
#include "nodes/executors/common/ref_transpose.hpp"
#include "nodes/executors/transpose.hpp"
#include "openvino/core/except.hpp"
#include "utils/bfloat16.hpp"

using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_args_permute, field)

namespace ov::intel_cpu {

#if defined(OPENVINO_ARCH_X86_64)

template <cpu_isa_t isa>
struct jit_uni_permute_kernel_f32 : public jit_uni_permute_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_permute_kernel_f32)

    explicit jit_uni_permute_kernel_f32(jit_permute_config_params jcp_)
        : jit_uni_permute_kernel(jcp_),
          jit_generator(jit_name()) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);

        loop(jcp.n);

        this->postamble();
    }

    void load(const Xbyak::Xmm& xmm, const Xbyak::Address& addr) {
        switch (jcp.data_size) {
        case 16:
            uni_vmovups(xmm, addr);
            break;
        case 8:
            uni_vmovsd(xmm, addr);
            break;
        case 4:
            uni_vmovss(xmm, addr);
            break;
        case 2:
            uni_vpinsrw(xmm, xmm, addr, 0x0);
            break;
        case 1:
            uni_vpinsrb(xmm, xmm, addr, 0x0);
            break;
        }
    }

    void store(const Xbyak::Address& addr, const Xbyak::Xmm& xmm) {
        switch (jcp.data_size) {
        case 16:
            uni_vmovups(addr, xmm);
            break;
        case 8:
            uni_vmovsd(addr, xmm);
            break;
        case 4:
            uni_vmovss(addr, xmm);
            break;
        case 2:
            uni_vpextrw(addr, xmm, 0x0);
            break;
        case 1:
            uni_vpextrb(addr, xmm, 0x0);
            break;
        }
    }

    void loop(int n) {
        mov(reg_work_amount, jcp.dst_block_dims[n]);

        Xbyak::Label main_loop_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label exit_label;

        if (n + 1 == static_cast<int>(jcp.ndims)) {
            if (jcp.src_strides[n] == 1 && jcp.dst_strides[n] == 1) {
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

        L(tail_loop_label);
        {
            cmp(reg_work_amount, 0);
            je(exit_label, T_NEAR);

            if (n + 1 == static_cast<int>(jcp.ndims)) {
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
    using Vmm =
        typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    uint32_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_work_amount = r10;
    Xbyak::Reg64 aux_reg_src = r11;
    Xbyak::Reg64 aux_reg_dst = r12;

    Xbyak::Reg64 reg_params = abi_param1;

    Vmm vmm = Vmm(1);
    Xbyak::Xmm xmm = Xbyak::Xmm(1);
};

#endif  // OPENVINO_ARCH_X86_64

PermuteKernel::PermuteKernel(const PermuteParams& params) : params(params) {
    jcp = TransposeExecutor::prepareParams(params);
#if defined(OPENVINO_ARCH_X86_64)
    if (mayiuse(cpu::x64::avx512_core)) {
        permute_kernel = std::make_shared<jit_uni_permute_kernel_f32<cpu::x64::avx512_core>>(jcp);
    } else if (mayiuse(cpu::x64::avx2)) {
        permute_kernel = std::make_shared<jit_uni_permute_kernel_f32<cpu::x64::avx2>>(jcp);
    } else if (mayiuse(cpu::x64::sse41)) {
        permute_kernel = std::make_shared<jit_uni_permute_kernel_f32<cpu::x64::sse41>>(jcp);
    }
#endif  // OPENVINO_ARCH_X86_64

    if (permute_kernel) {
        permute_kernel->create_ker();
    }
}

void PermuteKernel::execute(const uint8_t* src_data, uint8_t* dst_data, const int mb) {
    if (permute_kernel) {
        optimizedExecute(src_data, dst_data, mb);
        return;
    }

    RefTransposeExecutor::referenceExecute(src_data, dst_data, jcp, mb);
}

void PermuteKernel::execute(const uint8_t* src_data, uint8_t* dst_data) {
    VectorDims dst_dims = jcp.dst_block_dims;
    if (permute_kernel) {
        optimizedExecute(src_data, dst_data, dst_dims[0]);
        return;
    }

    RefTransposeExecutor::referenceExecute(src_data, dst_data, jcp, dst_dims[0]);
}

void PermuteKernel::optimizedExecute(const uint8_t* src_data, uint8_t* dst_data, const int mb) {
    VectorDims dst_dims = jcp.dst_block_dims;
    const VectorDims dst_strides = jcp.dst_strides;
    const VectorDims src_strides = jcp.src_strides;

    if (static_cast<int>(dst_dims[0]) != mb) {
        dst_dims[0] = mb;
    }

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
    default:
        OPENVINO_THROW("Unsupported number of dimensions: " + std::to_string(jcp.n));
    }
    return;
}

size_t PermuteParams::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = get_vector_hash(seed, src_block_dims);
    seed = get_vector_hash(seed, dst_block_dims);
    seed = get_vector_hash(seed, src_block_order);
    seed = get_vector_hash(seed, dst_block_order);
    seed = get_vector_hash(seed, order);
    seed = hash_combine(seed, data_size);
    return seed;
}

bool PermuteParams::operator==(const PermuteParams& rhs) const {
    return (src_block_dims == rhs.src_block_dims) && (dst_block_dims == rhs.dst_block_dims) &&
           (src_block_order == rhs.src_block_order) && (dst_block_order == rhs.dst_block_order) &&
           (order == rhs.order) && (data_size == rhs.data_size);
}

}  // namespace ov::intel_cpu
