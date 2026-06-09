// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_fc_weight_decompression_kernel.hpp"

#include "openvino/core/except.hpp"

namespace ov::intel_cpu {

template <cpu_isa_t isa>
JitFCWeightDecompressionKernel<isa>::JitFCWeightDecompressionKernel(const FCWeightDecompressionKernelCompileParams& params)
    : jit_generator_t(jit_name()),
      m_params(params),
      m_blockSize(cpu_isa_traits_t<isa>::vlen / sizeof(float)) {
    create_kernel();
    m_kernel = reinterpret_cast<decltype(m_kernel)>(jit_ker());
}

template <cpu_isa_t isa>
void JitFCWeightDecompressionKernel<isa>::operator()(const FCWeightDecompressionKernelRuntimeParams* args) const {
    OPENVINO_ASSERT(m_kernel != nullptr, "JIT decompression kernel is not initialized");
    m_kernel(args);
}

template <cpu_isa_t isa>
size_t JitFCWeightDecompressionKernel<isa>::blockSize() const {
    return m_blockSize;
}

template <cpu_isa_t isa>
void JitFCWeightDecompressionKernel<isa>::generate() {
    preamble();

    mov(regWeights, ptr[param1 + offWeights()]);
    mov(regDst, ptr[param1 + offDst()]);

    if (m_params.withScales) {
        mov(regScales, ptr[param1 + offScales()]);
        if (m_params.broadcastScales) {
            uni_vbroadcastss(vmmScales(), ptr[regScales]);
        } else {
            uni_vmovups(vmmScales(), ptr[regScales]);
        }
    }

    if (m_params.withZeroPoints) {
        mov(regZeroPoints, ptr[param1 + offZeroPoints()]);
        if (m_params.broadcastZeroPoints) {
            uni_vbroadcastss(vmmZeroPoints(), ptr[regZeroPoints]);
        } else {
            uni_vmovups(vmmZeroPoints(), ptr[regZeroPoints]);
        }
    }

    switch (m_params.weightsType) {
    case ov::element::u8:
        uni_vpmovzxbd(vmmWeights(), ptr[regWeights]);
        uni_vcvtdq2ps(vmmWeights(), vmmWeights());
        break;
    case ov::element::i8:
        uni_vpmovsxbd(vmmWeights(), ptr[regWeights]);
        uni_vcvtdq2ps(vmmWeights(), vmmWeights());
        break;
    default:
        OPENVINO_THROW("Unsupported JIT decompression precision");
    }

    if (m_params.withZeroPoints) {
        uni_vsubps(vmmWeights(), vmmWeights(), vmmZeroPoints());
    }
    if (m_params.withScales) {
        uni_vmulps(vmmWeights(), vmmWeights(), vmmScales());
    }

    uni_vmovups(ptr[regDst], vmmWeights());
    postamble();
}

template class JitFCWeightDecompressionKernel<cpu_isa_t::avx2>;
template class JitFCWeightDecompressionKernel<cpu_isa_t::avx512_core>;

}  // namespace ov::intel_cpu
