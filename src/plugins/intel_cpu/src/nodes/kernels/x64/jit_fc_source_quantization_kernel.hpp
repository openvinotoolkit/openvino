// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace ov::intel_cpu {

using namespace dnnl::impl::cpu::x64;

struct FCSourceQuantizationKernelRuntimeParams {
    const float* src = nullptr;
    int8_t* dst = nullptr;
    float* scale = nullptr;
    size_t groupCount = 0;
};

struct FCSourceQuantizationKernelCompileParams {
    size_t groupSize = 0;
};

class FCSourceQuantizationKernelBase {
public:
    virtual ~FCSourceQuantizationKernelBase() = default;

    virtual void operator()(const FCSourceQuantizationKernelRuntimeParams* args) const = 0;

    [[nodiscard]] virtual size_t blockSize() const = 0;
};

template <cpu_isa_t isa>
class JitFCSourceQuantizationKernel : public FCSourceQuantizationKernelBase, public jit_generator_t {
public:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == cpu_isa_t::sse41,
                                                         Xbyak::Xmm,
                                                         isa == cpu_isa_t::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;

    explicit JitFCSourceQuantizationKernel(const FCSourceQuantizationKernelCompileParams& params);

    void operator()(const FCSourceQuantizationKernelRuntimeParams* args) const override;

    [[nodiscard]] size_t blockSize() const override;

private:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(JitFCSourceQuantizationKernel)

    static constexpr size_t offSrc() {
        return offsetof(FCSourceQuantizationKernelRuntimeParams, src);
    }

    static constexpr size_t offDst() {
        return offsetof(FCSourceQuantizationKernelRuntimeParams, dst);
    }

    static constexpr size_t offScale() {
        return offsetof(FCSourceQuantizationKernelRuntimeParams, scale);
    }

    static constexpr size_t offGroupCount() {
        return offsetof(FCSourceQuantizationKernelRuntimeParams, groupCount);
    }

    void reduceMax(Vmm value, Vmm temp);
    void storeQuantized(size_t offset);
    void zeroDstBlock(size_t offset);
    void generate() override;

    Vmm vmmValues() const {
        return Vmm(0);
    }

    Vmm vmmMax() const {
        return Vmm(1);
    }

    Vmm vmmSignMask() const {
        return Vmm(2);
    }

    Vmm vmmTmp() const {
        return Vmm(3);
    }

    Vmm vmmInt8Max() const {
        return Vmm(4);
    }

    Vmm vmmQScale() const {
        return Vmm(5);
    }

    Vmm vmmOne() const {
        return Vmm(6);
    }

    Vmm vmmScale() const {
        return Vmm(7);
    }

    Vmm vmmZero() const {
        return Vmm(8);
    }

    size_t m_groupSize = 0;
    size_t m_blockSize = 0;
    void (*m_kernel)(const FCSourceQuantizationKernelRuntimeParams*) = nullptr;
    Xbyak::Reg64 regSrc = r8;
    Xbyak::Reg64 regDst = r9;
    Xbyak::Reg64 regScale = r10;
    Xbyak::Reg64 regGroupCount = r11;
    Xbyak::Reg64 regTmp = r12;
};

}  // namespace ov::intel_cpu
