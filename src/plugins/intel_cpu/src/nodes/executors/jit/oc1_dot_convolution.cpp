// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "oc1_dot_convolution.hpp"

#include <algorithm>
#include <array>
#include <numeric>

#include "cpu_parallel.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/arm_isa_support.h"

namespace ov::intel_cpu::aarch64 {

Oc1DotConvolution::Oc1DotConvolution(const ConvAttrs& attrs,
                                     const MemoryArgs& memory,
                                     const ExecutorContext::CPtr& context)
    : m_parallel(context->getCpuParallel()),
      m_channels(memory.at(ARG_SRC)->getStaticDims()[1]),
      m_unsigned_input(memory.at(ARG_SRC)->getPrecision() == ov::element::u8),
      m_zero_point(attrs.inputZeroPoint),
      m_pad_top(attrs.paddingL[0]),
      m_pad_left(attrs.paddingL[1]),
      m_scale(attrs.dqScales.empty() ? 1.F : attrs.dqScales.front()),
      m_with_bias(attrs.withBias),
      m_kernel(m_channels, m_unsigned_input) {}

bool Oc1DotConvolution::supportsGeometry(const VectorDims& src,
                                         const VectorDims& weights,
                                         const VectorDims& dst,
                                         const std::vector<size_t>& strides,
                                         const std::vector<size_t>& dilations,
                                         const std::vector<ptrdiff_t>& padsBegin,
                                         const std::vector<ptrdiff_t>& padsEnd) {
    const auto nonnegative = [](ptrdiff_t pad) {
        return pad >= 0;
    };
    return hasArmISASupport(ArmISA::DOTPROD) && src.size() == 4 && weights.size() == 4 && dst.size() == 4 &&
           std::all_of(src.begin(),
                       src.end(),
                       [](size_t dim) {
                           return dim > 0;
                       }) &&
           src[0] == dst[0] && src[1] >= 4 && src[1] <= 128 && src[1] % 4 == 0 && weights[0] == 1 &&
           weights[1] == src[1] && weights[2] == 3 && weights[3] == 3 && dst[1] == 1 && dst[3] >= 16 &&
           dst[2] * dst[3] >= 1024 && strides == std::vector<size_t>{1, 1} && dilations == std::vector<size_t>{0, 0} &&
           padsBegin.size() == 2 && padsEnd.size() == 2 &&
           std::all_of(padsBegin.begin(), padsBegin.end(), nonnegative) &&
           std::all_of(padsEnd.begin(), padsEnd.end(), nonnegative);
}

bool Oc1DotConvolution::supports(const ConvConfig& config) {
    const auto& attrs = config.attrs;
    const auto& src = *config.descs.at(ARG_SRC);
    const auto& weights = *config.descs.at(ARG_WEI);
    const auto& dst = *config.descs.at(ARG_DST);
    const bool signedInput = src.getPrecision() == ov::element::i8 &&
                             attrs.inputZeroPointsType == ZeroPointsType::None && attrs.inputZeroPoint == 0;
    const bool unsignedInput = src.getPrecision() == ov::element::u8 &&
                               attrs.inputZeroPointsType == ZeroPointsType::PerTensorUnsignedOc1Jit &&
                               attrs.inputZeroPoint >= 0 && attrs.inputZeroPoint <= 255;
    return (signedInput || unsignedInput) && !attrs.isGrouped && attrs.constantWeights && attrs.postOps.empty() &&
           attrs.dqScales.size() <= 1 && weights.getPrecision() == ov::element::i8 &&
           dst.getPrecision() == ov::element::f32 &&
           (!attrs.withBias || config.descs.at(ARG_BIAS)->getPrecision() == ov::element::f32) &&
           src.getShape().isStatic() && weights.getShape().isStatic() && dst.getShape().isStatic() &&
           supportsGeometry(src.getShape().getStaticDims(),
                            weights.getShape().getStaticDims(),
                            dst.getShape().getStaticDims(),
                            attrs.stride,
                            attrs.dilation,
                            attrs.paddingL,
                            attrs.paddingR);
}

void Oc1DotConvolution::execute(const MemoryArgs& memory) {
    const auto& srcMemory = memory.at(ARG_SRC);
    const auto& dstMemory = memory.at(ARG_DST);
    const auto* src = static_cast<const int8_t*>(srcMemory->getData());
    const auto* weights = memory.at(ARG_WEI)->getDataAs<const int8_t>();
    auto* dst = dstMemory->getDataAs<float>();
    const float bias = m_with_bias ? *memory.at(ARG_BIAS)->getDataAs<const float>() : 0.F;
    const auto& srcDims = srcMemory->getStaticDims();
    const auto& dstDims = dstMemory->getStaticDims();
    const auto inputWidth = srcDims[3];
    const auto inputHeight = srcDims[2];
    const auto outputWidth = dstDims[3];
    const auto outputHeight = dstDims[2];
    std::array<int8_t, 128> padding;
    padding.fill(static_cast<int8_t>(m_zero_point));
    // SDOT reads U8 bytes rebased by 128; restore (input - zero point) in the accumulator.
    const int32_t correction =
        m_unsigned_input ? (128 - m_zero_point) * std::accumulate(weights, weights + 9 * m_channels, 0) : 0;

    m_parallel->parallel_for(srcDims[0] * outputHeight, [&](size_t row) {
        constexpr size_t tile = 64;
        std::array<const int8_t*, tile * 9> taps;
        const auto y = row % outputHeight;
        const auto* image = src + (row / outputHeight) * inputHeight * inputWidth * m_channels;
        for (size_t x = 0; x < outputWidth; x += tile) {
            const auto count = std::min(tile, outputWidth - x);
            for (size_t ky = 0; ky < 3; ++ky) {
                const auto iy = y + ky - m_pad_top;
                const auto* line = iy < inputHeight ? image + iy * inputWidth * m_channels : nullptr;
                for (size_t dx = 0; dx < count; ++dx) {
                    for (size_t kx = 0; kx < 3; ++kx) {
                        const auto ix = x + dx + kx - m_pad_left;
                        taps[dx * 9 + ky * 3 + kx] = line && ix < inputWidth ? line + ix * m_channels : padding.data();
                    }
                }
            }
            m_kernel(taps.data(), weights, dst + row * outputWidth + x, count, correction, m_scale, bias);
        }
    });
}

}  // namespace ov::intel_cpu::aarch64
