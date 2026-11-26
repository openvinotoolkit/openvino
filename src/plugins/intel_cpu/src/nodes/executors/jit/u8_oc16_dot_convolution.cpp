// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "u8_oc16_dot_convolution.hpp"

#include <algorithm>
#include <array>
#include <cstring>

#include "cpu_parallel.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/arm_isa_support.h"

namespace ov::intel_cpu::aarch64 {
namespace {

constexpr auto oc16Channels = U8Oc16DotKernel::outputChannels;
constexpr auto oc16FilterElements = U8Oc16DotKernel::filterElements;
constexpr auto oc16PackedElements = U8Oc16DotKernel::packedElements;

}  // namespace

U8Oc16DotConvolution::U8Oc16DotConvolution(const ConvAttrs&, const MemoryArgs&, const ExecutorContext::CPtr& context)
    : m_parallel(context->getCpuParallel()) {}

bool U8Oc16DotConvolution::supports(const ConvConfig& config) {
    const auto& attrs = config.attrs;
    const auto& src = *config.descs.at(ARG_SRC);
    const auto& weights = *config.descs.at(ARG_WEI);
    const auto& dst = *config.descs.at(ARG_DST);
    if (!hasArmISASupport(ArmISA::DOTPROD) || attrs.isGrouped || !attrs.constantWeights || attrs.withBias ||
        !attrs.postOps.empty() || !attrs.dqScales.empty() || attrs.inputZeroPoint != 0 ||
        attrs.inputZeroPointsType != ZeroPointsType::None || src.getPrecision() != ov::element::u8 ||
        weights.getPrecision() != ov::element::i8 || dst.getPrecision() != ov::element::f32 ||
        !src.getShape().isStatic() || !weights.getShape().isStatic() || !dst.getShape().isStatic()) {
        return false;
    }

    const auto& input = src.getShape().getStaticDims();
    const auto& filter = weights.getShape().getStaticDims();
    const auto& output = dst.getShape().getStaticDims();
    return input.size() == 4 && output.size() == 4 && filter == VectorDims{oc16Channels, 3, 3, 3} && input[1] == 3 &&
           output[1] == oc16Channels && input[0] == output[0] && input[2] == output[2] && input[3] == output[3] &&
           output[3] >= 16 && output[2] * output[3] >= 1024 && attrs.stride == std::vector<size_t>{1, 1} &&
           attrs.dilation == std::vector<size_t>{0, 0} && attrs.paddingL == std::vector<ptrdiff_t>{1, 1} &&
           attrs.paddingR == std::vector<ptrdiff_t>{1, 1};
}

void U8Oc16DotConvolution::execute(const MemoryArgs& memory) {
    const auto& srcMemory = memory.at(ARG_SRC);
    const auto* src = srcMemory->getDataAs<const uint8_t>();
    const auto* weights = memory.at(ARG_WEI)->getDataAs<const int8_t>();
    auto* dst = memory.at(ARG_DST)->getDataAs<float>();
    const auto& dims = srcMemory->getStaticDims();
    const auto batch = dims[0];
    const auto height = dims[2];
    const auto width = dims[3];
    std::array<int8_t, oc16PackedElements * oc16Channels> packedWeights{};
    std::array<int32_t, oc16Channels> corrections{};
    // Pack four bytes per SDOT lane for four output channels at a time.
    for (size_t oc = 0; oc < oc16Channels; ++oc) {
        for (size_t k = 0; k < oc16FilterElements; ++k) {
            const auto weight = weights[oc * oc16FilterElements + k];
            packedWeights[(k / 4) * 64 + oc * 4 + k % 4] = weight;
            corrections[oc] += 128 * weight;
        }
    }
    m_parallel->parallel_for(batch * height, [&](size_t row) {
        constexpr size_t tile = 64;
        std::array<uint8_t, tile * oc16PackedElements> patches;
        const auto y = row % height;
        const auto* image = src + (row / height) * width * height * 3;
        for (size_t x = 0; x < width; x += tile) {
            const auto count = std::min(tile, width - x);
            for (size_t dx = 0; dx < count; ++dx) {
                auto* patch = patches.data() + dx * oc16PackedElements;
                const auto column = x + dx;
                for (size_t ky = 0; ky < 3; ++ky) {
                    const auto iy = y + ky - 1;
                    if (iy >= height) {
                        std::fill_n(patch + ky * 9, 9, 0);
                    } else {
                        const auto* line = image + iy * width * 3;
                        if (column > 0 && column + 1 < width) {
                            std::memcpy(patch + ky * 9, line + (column - 1) * 3, 9);
                        } else {
                            auto* patchRow = patch + ky * 9;
                            std::fill_n(patchRow, 9, 0);
                            const bool left = column == 0;
                            std::memcpy(patchRow + (left ? 3 : 0), line + (left ? 0 : (column - 1) * 3), 6);
                        }
                    }
                }
                patch[oc16FilterElements] = 0;
            }
            m_kernel(patches.data(),
                     packedWeights.data(),
                     dst + (row * width + x) * oc16Channels,
                     count,
                     corrections.data());
        }
    });
}

}  // namespace ov::intel_cpu::aarch64
