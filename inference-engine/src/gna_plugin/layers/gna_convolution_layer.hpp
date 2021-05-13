// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <legacy/ie_layers.h>
#include "../gna_graph_tools.hpp"

namespace GNAPluginNS {
struct GNAConvolutionLayer {
    static bool isMappableFrom2DTo1D(const uint32_t inHeight, const uint32_t inWidth, const uint32_t kernelWidth, const uint32_t strideWidth) {
        return inHeight > 1 && inWidth > 1 && inWidth == kernelWidth && strideWidth == 1;
    }

    // 3D input or 2D kernel
    static bool isConv2D(const uint32_t inHeight, const uint32_t inWidth, const uint32_t inDepth,
                     const uint32_t kernelHeight, const uint32_t kernelWidth) {
        return (kernelHeight > 1 && kernelWidth > 1) || (inHeight > 1 && inWidth > 1 && inDepth > 1);
    }

    static double getWeightsReducer(InferenceEngine::ConvolutionLayer& conv) {
        using KRT = std::pair<uint32_t, double>;
        // Empirically determined weights reducers for 2D Convolution
        // i.e.:
        // for kernelSize >= 9       -> 1.3
        // for kernelSize in {7, 8}  -> 1.2
        const std::vector< KRT > reducers{ {9, 1.3}, {7, 1.2} };
        auto reducer = 1.0;
        const auto inDepth = GetDataDimSize(conv.insData.front().lock(), InferenceEngine::DataDimName::C);
        const auto inHeight = GetDataDimSize(conv.insData.front().lock(), InferenceEngine::DataDimName::H);
        const auto inWidth = GetDataDimSize(conv.insData.front().lock(), InferenceEngine::DataDimName::W);
        if (isConv2D(inHeight, inWidth, inDepth, conv._kernel_y, conv._kernel_x) &&
             !isMappableFrom2DTo1D(inHeight, inWidth, conv._kernel_x, conv._stride_x)) {
            const auto kernelSize = conv._kernel_x * conv._kernel_y;
            auto r = std::lower_bound(reducers.begin(), reducers.end(), kernelSize,
                [](const KRT& l, const KRT::first_type& r) {return l.first > r; });
            if (r != reducers.end())
                reducer = r->second;
        }
        return reducer;
    }
};
}  // namespace GNAPluginNS
