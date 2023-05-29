// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_convolution_layer.hpp"

#include <legacy/ie_layers.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "gna_graph_tools.hpp"
#include "log/debug.hpp"

namespace ov {
namespace intel_gna {
namespace gna_convolution_layer {

bool should_transpose_h_w(const size_t in_height,
                          const size_t kernel_height,
                          const size_t in_channels,
                          const size_t stride_height) {
    return in_height == kernel_height && in_channels == 1 && stride_height == 1;
}

bool isMappableFrom2DTo1D(const size_t inHeight,
                          const size_t inWidth,
                          const size_t in_channels,
                          const size_t kernelHeight,
                          const size_t kernelWidth,
                          const size_t strideHeight,
                          const size_t strideWidth) {
    if (inHeight <= 1 || inWidth <= 1) {
        // Mapping not needed since input is already 1D
        return false;
    }
    return (inWidth == kernelWidth && strideWidth == 1) ||
           should_transpose_h_w(inHeight, kernelHeight, in_channels, strideHeight);
}

bool is3DInputOr2DKernel(const size_t inHeight,
                         const size_t inWidth,
                         const size_t inDepth,
                         const size_t kernelHeight,
                         const size_t kernelWidth) {
    return (kernelHeight > 1 && kernelWidth > 1) || (inHeight > 1 && inWidth > 1 && inDepth > 1);
}

double getWeightsReducer(InferenceEngine::ConvolutionLayer& conv) {
    using KRT = std::pair<uint32_t, double>;
    // Empirically determined weights reducers for 2D Convolution
    // i.e.:
    // for kernelSize >= 14      -> 1.7
    // for kernelSize >= 9       -> 1.3
    // for kernelSize in {7, 8}  -> 1.2
    const std::vector<KRT> reducers{{49, 3.0}, {36, 2.6}, {21, 2.3}, {14, 1.7}, {9, 1.3}, {7, 1.2}};
    auto reducer = 1.0;
    const auto inDepth =
        InferenceEngine::GetDataDimByName(conv.insData.front().lock(), InferenceEngine::DataDimName::C);
    const auto inHeight =
        InferenceEngine::GetDataDimByName(conv.insData.front().lock(), InferenceEngine::DataDimName::H);
    const auto inWidth =
        InferenceEngine::GetDataDimByName(conv.insData.front().lock(), InferenceEngine::DataDimName::W);
    if (is3DInputOr2DKernel(inHeight, inWidth, inDepth, conv._kernel_y, conv._kernel_x) &&
        !isMappableFrom2DTo1D(inHeight,
                              inWidth,
                              inDepth,
                              conv._kernel_y,
                              conv._kernel_x,
                              conv._stride_y,
                              conv._stride_x)) {
        const auto kernelSize = conv._kernel_x * conv._kernel_y;
        auto r =
            std::lower_bound(reducers.begin(), reducers.end(), kernelSize, [](const KRT& l, const KRT::first_type& r) {
                return l.first > r;
            });
        if (r != reducers.end())
            reducer = r->second;
    }
    return reducer;
}

size_t outputFromConv(const size_t in, const size_t flt, const size_t stride) {
    // floor[(in - flt)/stride] + 1, GNA Spec 1.24
    if (flt > in || flt == 0 || stride == 0) {
        THROW_GNA_EXCEPTION << "Invalid (input, filter, stride) = (" << in << "," << flt << "," << stride << ")";
    }
    return (in - flt) / stride + 1;
}

size_t outputFromPooling(const size_t in, const size_t window, const size_t stride, const bool legacy) {
    if (legacy) {
        return outputFromPoolingLegacy(in, stride);
    }
    // ceil[(in - window)/stride] + 1, GNA Spec 1.24
    if (window > in || window == 0 || stride == 0) {
        THROW_GNA_EXCEPTION << "Invalid (input, window, stride) = (" << in << "," << window << "," << stride << ")";
    }
    if (window == in)
        return 1;

    return (in - window - 1) / stride + 2;
}

size_t outputFromPoolingLegacy(const size_t in, const size_t stride) {
    // floor[(in - 1)/stride] + 1, GNA 1.0/2.0 HW Spec
    // See issue 50386 for details
    if (in == 0 || stride == 0) {
        THROW_GNA_EXCEPTION << "Invalid (input, stride) = (" << in << "," << stride << ")";
    }
    return (in - 1) / stride + 1;
}

}  // namespace gna_convolution_layer
}  // namespace intel_gna
}  // namespace ov
