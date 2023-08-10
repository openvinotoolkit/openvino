// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>

#include <cstdint>

namespace ov {
namespace intel_gna {
namespace gna_convolution_layer {

bool should_transpose_h_w(const uint32_t in_height,
                          const uint32_t kernel_height,
                          const uint32_t in_channels,
                          const uint32_t stride_height);

bool isMappableFrom2DTo1D(const uint32_t inHeight,
                          const uint32_t inWidth,
                          const uint32_t inChannels,
                          const uint32_t kernelHeight,
                          const uint32_t kernelWidth,
                          const uint32_t strideHeight,
                          const uint32_t strideWidth);

bool is3DInputOr2DKernel(const uint32_t inHeight,
                         const uint32_t inWidth,
                         const uint32_t inDepth,
                         const uint32_t kernelHeight,
                         const uint32_t kernelWidth);

double getWeightsReducer(InferenceEngine::ConvolutionLayer& conv);

uint32_t outputFromConv(const uint32_t in, const uint32_t flt, const uint32_t stride);

uint32_t outputFromPooling(const uint32_t in, const uint32_t window, const uint32_t stride, bool legacy = false);

uint32_t outputFromPoolingLegacy(const uint32_t in, const uint32_t stride);

}  // namespace gna_convolution_layer
}  // namespace intel_gna
}  // namespace ov
