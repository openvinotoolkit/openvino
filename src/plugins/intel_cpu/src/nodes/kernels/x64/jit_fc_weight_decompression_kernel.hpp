// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

using namespace dnnl::impl::cpu::x64;

struct FCWeightDecompressionKernelRuntimeParams {
    const uint8_t* weights = nullptr;
    float* dst = nullptr;
    const float* scales = nullptr;
    const float* zeroPoints = nullptr;
};

struct FCWeightDecompressionKernelCompileParams {
    bool withScales = false;
    bool withZeroPoints = false;
    bool broadcastScales = false;
    bool broadcastZeroPoints = false;
    ov::element::Type weightsType = ov::element::dynamic;
};

class FCWeightDecompressionKernelBase {
public:
    virtual ~FCWeightDecompressionKernelBase() = default;

    virtual void operator()(const FCWeightDecompressionKernelRuntimeParams* args) const = 0;

    [[nodiscard]] virtual size_t blockSize() const = 0;
};

template <cpu_isa_t isa>
class JitFCWeightDecompressionKernel : public FCWeightDecompressionKernelBase, public jit_generator_t {
public:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == cpu_isa_t::sse41,
                                                         Xbyak::Xmm,
                                                         isa == cpu_isa_t::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;

    explicit JitFCWeightDecompressionKernel(const FCWeightDecompressionKernelCompileParams& params);

    void operator()(const FCWeightDecompressionKernelRuntimeParams* args) const override;

    [[nodiscard]] size_t blockSize() const override;

private:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(JitFCWeightDecompressionKernel)

    static constexpr size_t offWeights() {
        return offsetof(FCWeightDecompressionKernelRuntimeParams, weights);
    }

    static constexpr size_t offDst() {
        return offsetof(FCWeightDecompressionKernelRuntimeParams, dst);
    }

    static constexpr size_t offScales() {
        return offsetof(FCWeightDecompressionKernelRuntimeParams, scales);
    }

    static constexpr size_t offZeroPoints() {
        return offsetof(FCWeightDecompressionKernelRuntimeParams, zeroPoints);
    }

    void generate() override;

    Vmm vmmWeights() const {
        return Vmm(0);
    }

    Vmm vmmScales() const {
        return Vmm(1);
    }

    Vmm vmmZeroPoints() const {
        return Vmm(2);
    }

    FCWeightDecompressionKernelCompileParams m_params;
    size_t m_blockSize = 0;
    void (*m_kernel)(const FCWeightDecompressionKernelRuntimeParams*) = nullptr;
    Xbyak::Reg64 regWeights = r8;
    Xbyak::Reg64 regDst = r9;
    Xbyak::Reg64 regScales = r10;
    Xbyak::Reg64 regZeroPoints = r11;
};

}  // namespace ov::intel_cpu
