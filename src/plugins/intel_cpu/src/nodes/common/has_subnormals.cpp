// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/visibility.hpp"

#include "has_subnormals.h"
#include "cpu_memory.h"
#include "openvino/core/parallel.hpp"
#include "cpu/x64/jit_generator.hpp"

using namespace dnnl;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {

#if defined(OPENVINO_ARCH_X86_64)

struct jit_has_subnormals_base : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_has_subnormals_base)

    typedef struct {
        const float* src;
        const size_t count;
        bool hasSubnormals;
    } args_t;

    void (*ker_)(const args_t *);
    void operator()(const args_t* args) { assert(ker_); ker_(args); }

    jit_has_subnormals_base() : jit_generator(jit_name()) {
        jit_ker_ = nullptr;
    }

    virtual void create_ker() = 0;

protected:
    void foreach(const Xbyak::Reg64& idx,
                 size_t step,
                 const Xbyak::Reg64& end,
                 std::function<void(const Xbyak::Reg64&)> && fn) {
        Label loop, exit;

        L(loop);
        cmp(idx, end);
        jge(exit);

        fn(idx);

        add(idx, step);
        jmp(loop);
        L(exit);
    }

    void copy_floats(const Xbyak::Reg64& dst,
                     const Xbyak::Reg64& src,
                     const Xbyak::Reg64& size) {
        push(rsi);
        push(r15);

        xor_(rsi, rsi);

        foreach(rsi, 1, size, [&, this](const Xbyak::Reg64& idx) {
            mov(r15d, dword[src + idx * sizeof(float)]);
            mov(dword[dst + idx * sizeof(float)], r15d);
        });

        pop(r15);
        pop(rsi);
    }

    void check_subnormals(const Xbyak::Reg64& src, const Xbyak::Ymm &exponent_mask, const Xbyak::Ymm &mantissa_mask, const Xbyak::Ymm &zero) {
        auto a = ymm1;
        auto b = ymm2;
        auto c = ymm3;

        vmovdqu(a, yword[src]);         // load 8 floats
        vpand(b, a, mantissa_mask);     // b = a & 00000000011111111111111111111111
        vpcmpeqd(b, b, zero);           // if (b == 0) b = 1 else b = 0
        vpand(c, a, exponent_mask);     // c = a & 01111111100000000000000000000000
        vpcmpeqd(c, c, zero);           // if (c == 0) c = 1 else c = 0
        vptest(b, c);                   // if ((!b & c) == 0) CF = 1 else CF = 0
    }

    void check_subnormals(const Xbyak::Reg64& src, const Xbyak::Xmm &exponent_mask, const Xbyak::Xmm &mantissa_mask, const Xbyak::Xmm &zero) {
        auto a = xmm1;
        auto b = xmm2;
        auto c = xmm3;

        uni_vmovdqu(a, xword[src]);          // load 4 floats
        uni_vmovdqu(b, a);                   // b = a
        uni_vmovdqu(c, a);                   // c = a
        uni_vpand(b, b, mantissa_mask);      // b = a & 00000000011111111111111111111111
        uni_vpcmpeqd(b, b, zero);            // if (b == 0) b = 1 else b = 0
        uni_vpand(c, c, exponent_mask);      // c = a & 01111111100000000000000000000000
        uni_vpcmpeqd(c, c, zero);            // if (c == 0) c = 1 else c = 0
        uni_vtestps(b, c);                   // if ((!b & c) == 0) CF = 1 else CF = 0
    }

protected:
    Label exit, has_subnormals, no_subnormals;

    const Reg64 &reg_src = rax;
    const Reg64 &reg_dst = rbx;
    const Reg64 &reg_sz = rdx;
    const Reg64 &reg_idx = rsi;
    const Reg64 &reg_mask_addr = r15;

    static const uint32_t exponent_mask_data[8];
    static const uint32_t mantissa_mask_data[8];
};

const uint32_t jit_has_subnormals_base::exponent_mask_data[8] = {
    0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000,
    0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000
};

const uint32_t jit_has_subnormals_base::mantissa_mask_data[8] = {
    0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff,
    0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff
};

template<cpu_isa_t isa>
struct jit_has_subnormals : public jit_has_subnormals_base {
    using Vmm = typename dnnl::impl::utils::conditional<isa == sse41, Xbyak::Xmm, Xbyak::Ymm>::type;

    const Vmm rmm4 = Vmm(4);
    const Vmm rmm5 = Vmm(5);
    const Vmm rmm6 = Vmm(6);
    const int length = isa == sse41 ? 4 : 8;

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override final { // NOLINT
        size_t const vlen = length;
        const int sh_bits = std::ilogb(vlen);

        auto zero = rmm4;
        auto exponent_mask = rmm5;
        auto mantissa_mask = rmm6;

        preamble();

        // Get arguments addresses
        mov(reg_src, ptr[param1 + offsetof(args_t, src)]);
        lea(reg_dst, ptr[param1 + offsetof(args_t, hasSubnormals)]);
        mov(reg_sz, ptr[param1 + offsetof(args_t, count)]);

        // Initialize necessary consts
        uni_vpxor(zero, zero, zero);
        mov(reg_mask_addr, (size_t)exponent_mask_data);
        uni_vmovdqu(exponent_mask, ptr[reg_mask_addr]);
        mov(reg_mask_addr, (size_t)mantissa_mask_data);
        uni_vmovdqu(mantissa_mask, ptr[reg_mask_addr]);

        // Main loop
        xor_(reg_idx, reg_idx);
        mov(r8, reg_sz);
        shr(r8, sh_bits);

        foreach(reg_idx, 1, r8, [&, this](const Xbyak::Reg64& idx) {
            check_subnormals(reg_src, exponent_mask, mantissa_mask, zero);
            jnc(has_subnormals);
            add(reg_src, sizeof(float) * vlen);
        });

        // Tail
        shl(reg_idx, sh_bits);
        sub(reg_sz, reg_idx);
        test(reg_sz, reg_sz);
        jz(exit);

        // use space on stack for 4 or 8 floats
        sub(rsp, vlen * sizeof(float));
        mov(r8, rsp);

        uni_vmovdqu(ptr[r8], zero);

        copy_floats(r8, reg_src, reg_sz);
        check_subnormals(r8, exponent_mask, mantissa_mask, zero);
        jc(no_subnormals);
        add(rsp, vlen * sizeof(float));

        L(has_subnormals);

        mov(rax, 1);
        mov(byte[reg_dst], al);
        jmp(exit);

        L(no_subnormals);
        add(rsp, vlen * sizeof(float));

        L(exit);

        postamble();
    }
};

static std::shared_ptr<jit_has_subnormals_base> createKernel() {
    std::shared_ptr<jit_has_subnormals_base> kernel;
    if (mayiuse(cpu_isa_t::avx2)) {
        kernel = std::make_shared<jit_has_subnormals<cpu_isa_t::avx2>>();
    } else if (mayiuse(cpu_isa_t::sse41)) {
        kernel = std::make_shared<jit_has_subnormals<cpu_isa_t::sse41>>();
    }

    if (kernel) {
        kernel->create_ker();
        if (!kernel->jit_ker())
            kernel = nullptr;
    }

    return kernel;
}
#endif

bool HasSubnormals::execute(const IMemory& src) {
    const auto prec = src.getPrecision();
    const auto size = src.getShape().getElementsCount();

    if (size == 0)
        return false;

    if (prec != ov::element::f32)
        return false;

    const uint32_t* u32data = src.getDataAs<uint32_t>();

#if defined(OPENVINO_ARCH_X86_64)
    static std::shared_ptr<jit_has_subnormals_base> kernel = createKernel();
    if (kernel) {
        static const size_t batch_size = 2048;
        const size_t iterations_num = size / batch_size + 1;

        volatile bool has_subnormals = false;

        parallel_for(iterations_num, [&](int n) {
            auto ptr = u32data + n * batch_size;
            const jit_has_subnormals_base::args_t args = {reinterpret_cast<float const*>(ptr),
                                                          std::min(batch_size, (size_t)(u32data + size - ptr)),
                                                          false};
            (*kernel)(&args);
            // result is written to the input 'hasSubnormals' parameter
            if (args.hasSubnormals)
                has_subnormals = true;
        });

        return has_subnormals;
    }
#endif

    // @todo optimize for ARM and other architectures
    uint32_t mantissaMask = 0x007fffff;
    uint32_t exponentMask = 0x7f800000;
    for (size_t i = 0; i < size; ++i) {
        if ((u32data[i] & exponentMask) == 0 && (u32data[i] & mantissaMask) != 0) {
            return true;
        }
    }

    return false;
}

}   // namespace intel_cpu
}   // namespace ov
