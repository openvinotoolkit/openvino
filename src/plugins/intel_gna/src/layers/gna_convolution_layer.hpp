// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>

#include <cstdint>

namespace ov {
namespace intel_gna {
namespace gna_convolution_layer {

bool should_transpose_h_w(const size_t in_height,
                          const size_t kernel_height,
                          const size_t in_channels,
                          const size_t stride_height);

bool isMappableFrom2DTo1D(const size_t inHeight,
                          const size_t inWidth,
                          const size_t inChannels,
                          const size_t kernelHeight,
                          const size_t kernelWidth,
                          const size_t strideHeight,
                          const size_t strideWidth);

bool is3DInputOr2DKernel(const size_t inHeight,
                         const size_t inWidth,
                         const size_t inDepth,
                         const size_t kernelHeight,
                         const size_t kernelWidth);

double getWeightsReducer(InferenceEngine::ConvolutionLayer& conv);

size_t outputFromConv(const size_t in, const size_t flt, const size_t stride);

size_t outputFromPooling(const size_t in, const size_t window, const size_t stride, bool legacy = false);

size_t outputFromPoolingLegacy(const size_t in, const size_t stride);

}  // namespace gna_convolution_layer
}  // namespace intel_gna
}  // namespace ov
