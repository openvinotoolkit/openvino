// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memcpy.h"

#include "cpu/x64/jit_generator.hpp"
#include "common/utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_args_memcpy, field)

namespace ov {
namespace intel_cpu {

template <cpu_isa_t isa>
struct jit_memcpy_kernel : public jit_uni_memcpy_kernel, public jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_memcpy_kernel)
    explicit jit_memcpy_kernel() : jit_uni_memcpy_kernel(), jit_generator(jit_name()) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_size, ptr[reg_params + GET_OFF(size)]);

        emit_common();

        postamble();
    }

private:
    void emit_common() {
        using namespace Xbyak;

        const int step = vlen;
        const int unroll_factor = 4;
        const int unrolled_step = step * unroll_factor;

        Label loop, remainder, end;
        test(reg_size, reg_size);
        jz(end);

        mov(reg_curr_size, reg_size);
        and_(reg_curr_size, ~(unrolled_step - 1));  // Align the size to the unrolled step

        L(loop);
        for (int i = 0; i < unroll_factor; ++i) {
            uni_vmovdqu(vmm, ptr[reg_src + step * i]);
            uni_vmovdqu(ptr[reg_dst + step * i], vmm);
        }
        add(reg_dst, unrolled_step);
        add(reg_src, unrolled_step);
        sub(reg_curr_size, unrolled_step);
        jnz(loop);

        // Handle remaining bytes
        // mov(reg_size, reg_size);
        and_(reg_size, unrolled_step - 1);  // Get the remaining size
        jz(end);  // If no remaining bytes, jump to the end

        L(remainder);
        mov(al, ptr[reg_src]);
        mov(ptr[reg_dst], al);
        inc(reg_dst);
        inc(reg_src);
        dec(reg_size);
        jnz(remainder);

        L(end);
        ret();
    }

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_size = r10;
    Xbyak::Reg64 reg_curr_size = r11;
    Xbyak::Reg64 reg_params = abi_param1;
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    uint32_t vlen = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen;
    Vmm vmm = Vmm(1);
};

MemCpy::MemCpy() {
    if (mayiuse(cpu::x64::avx512_core)) {
        memcpy_kernel.reset(new jit_memcpy_kernel<dnnl::impl::cpu::x64::avx512_core>());
    } else if (mayiuse(cpu::x64::avx2)) {
        memcpy_kernel.reset(new jit_memcpy_kernel<cpu::x64::avx2>());
    } else if (mayiuse(cpu::x64::sse41)) {
        memcpy_kernel.reset(new jit_memcpy_kernel<cpu::x64::sse41>());
    }

    if (memcpy_kernel)
        memcpy_kernel->create_ker();
}

void MemCpy::execute(const uint8_t* src_data, uint8_t* dst_data, std::size_t size) {
    // VectorDims dst_dims = jcp.dst_block_dims;
    auto args = jit_args_memcpy();

    args.src = src_data;
    args.dst = dst_data;
    args.size = size;

    if (memcpy_kernel) {
        return (*memcpy_kernel)(&args);
    }
}

}   // namespace intel_cpu
}   // namespace ov
