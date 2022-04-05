// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include <legacy/ie_layers.h>

namespace GNAPluginNS {
namespace GNAConvolutionLayer {
bool isMappableFrom2DTo1D(const uint32_t inHeight, const uint32_t inWidth, const uint32_t inChannels,
                          const uint32_t kernelHeight, const uint32_t kernelWidth,
                          const uint32_t strideHeight, const uint32_t strideWidth);

// 3D input or 2D kernel
bool isConv2D(const uint32_t inHeight, const uint32_t inWidth, const uint32_t inDepth,
    const uint32_t kernelHeight, const uint32_t kernelWidth);

double getWeightsReducer(InferenceEngine::ConvolutionLayer& conv);

uint32_t outputFromConv(const uint32_t in, const uint32_t flt, const uint32_t stride);

uint32_t outputFromPooling(const uint32_t in, const uint32_t window, const uint32_t stride, bool legacy = false);

uint32_t outputFromPoolingLegacy(const uint32_t in, const uint32_t stride);

} // namespace GNAConvolutionLayer
} // namespace GNAPluginNS
