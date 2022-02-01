// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_convolution_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include <legacy/ie_layers.h>
#include "gna_graph_tools.hpp"
#include "gna_plugin_log.hpp"

namespace GNAPluginNS {
namespace GNAConvolutionLayer {

bool isMappableFrom2DTo1D(const uint32_t inHeight, const uint32_t inWidth, const uint32_t in_channels,
                          const uint32_t kernelHeight, const uint32_t kernelWidth,
                          const uint32_t strideHeight, const uint32_t strideWidth) {
    return ((inHeight > 1 && inWidth > 1) &&
            ((inWidth == kernelWidth && strideWidth == 1) ||
             (inHeight == kernelHeight && strideHeight == 1 && in_channels == 1)));
}

// 3D input or 2D kernel
bool isConv2D(const uint32_t inHeight, const uint32_t inWidth, const uint32_t inDepth,
                 const uint32_t kernelHeight, const uint32_t kernelWidth) {
    return (kernelHeight > 1 && kernelWidth > 1) || (inHeight > 1 && inWidth > 1 && inDepth > 1);
}

double getWeightsReducer(InferenceEngine::ConvolutionLayer& conv) {
    using KRT = std::pair<uint32_t, double>;
    // Empirically determined weights reducers for 2D Convolution
    // i.e.:
    // for kernelSize >= 14      -> 1.7
    // for kernelSize >= 9       -> 1.3
    // for kernelSize in {7, 8}  -> 1.2
    const std::vector< KRT > reducers{ {49, 3.0}, {36, 2.6}, {21, 2.3}, {14, 1.7}, {9, 1.3}, {7, 1.2} };
    auto reducer = 1.0;
    const auto inDepth = GetDataDimSize(conv.insData.front().lock(), InferenceEngine::DataDimName::C);
    const auto inHeight = GetDataDimSize(conv.insData.front().lock(), InferenceEngine::DataDimName::H);
    const auto inWidth = GetDataDimSize(conv.insData.front().lock(), InferenceEngine::DataDimName::W);
    if (isConv2D(inHeight, inWidth, inDepth, conv._kernel_y, conv._kernel_x) &&
         !isMappableFrom2DTo1D(inHeight, inWidth, inDepth, conv._kernel_y, conv._kernel_x, conv._stride_y, conv._stride_x)) {
        const auto kernelSize = conv._kernel_x * conv._kernel_y;
        auto r = std::lower_bound(reducers.begin(), reducers.end(), kernelSize,
            [](const KRT& l, const KRT::first_type& r) {return l.first > r; });
        if (r != reducers.end())
            reducer = r->second;
    }
    return reducer;
}

uint32_t outputFromConv(const uint32_t in, const uint32_t flt, const uint32_t stride) {
    // floor[(in - flt)/stride] + 1, GNA Spec 1.24
    if (flt > in || flt == 0 || stride == 0) {
        THROW_GNA_EXCEPTION << "Invalid (input, filter, stride) = (" << in << "," << flt << "," << stride << ")";
    }
    return (in - flt) / stride + 1;
}

uint32_t outputFromPooling(const uint32_t in, const uint32_t window, const uint32_t stride, const bool legacy) {
    if (legacy) {
        return outputFromPoolingLegacy(in, stride);
    }
    // ceil[(in - window)/stride] + 1, GNA Spec 1.24
    if (window > in || window == 0 || stride == 0) {
        THROW_GNA_EXCEPTION << "Invalid (input, window, stride) = (" << in << "," << window << "," << stride << ")";
    }
    if (window == in) return 1;

    return (in - window - 1) / stride + 2;
}

uint32_t outputFromPoolingLegacy(const uint32_t in, const uint32_t stride) {
    // floor[(in - 1)/stride] + 1, GNA 1.0/2.0 HW Spec
    // See issue 50386 for details
    if (in == 0 || stride == 0) {
        THROW_GNA_EXCEPTION << "Invalid (input, stride) = (" << in << "," << stride << ")";
    }
    return (in - 1) / stride + 1;
}

} // namespace GNAConvolutionLayer
} // namespace GNAPluginNS
