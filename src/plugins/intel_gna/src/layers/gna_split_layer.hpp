// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>

#include <vector>

#include "backend/gna_limitations.hpp"

namespace ov {
namespace intel_gna {

// Split, Slice
class GNASplitLayer {
    InferenceEngine::CNNLayerPtr splitLayer;

public:
    explicit GNASplitLayer(InferenceEngine::CNNLayerPtr layer) : splitLayer(layer) {}

    InferenceEngine::CNNLayerPtr getSplit() {
        return splitLayer;
    }
    /**
     * gna memory of this size is reserved for split
     */
    size_t reserved_size = 0;
    bool output_allocation_flag = false;
    /**
     * gna memory of this offset from gna_ptr
     */
    struct SplitConnectedLayerInfo {
        SplitConnectedLayerInfo() = default;
        SplitConnectedLayerInfo(InferenceEngine::CNNLayerPtr connectedTo, int insDataIdx, size_t o, size_t p)
            : connectedTo(connectedTo),
              insDataIdx(insDataIdx),
              offset(o),
              pure_size(p) {}

        InferenceEngine::CNNLayerPtr connectedTo;
        int insDataIdx = 0;
        size_t offset = 0;  // in number of elements of input layer
        size_t pure_size = 0;
    };
    std::vector<SplitConnectedLayerInfo> splitOutputLayers;
};

// @brief Returns sizes of split outputs to split the input tensor into aligned parts that are not greater than the
// specified split size or alignment, depending on which one is larger
inline std::vector<uint32_t> GetAlignedSplitSizes(uint32_t totalSize, uint32_t splitSize, uint32_t alignment) {
    std::vector<uint32_t> splitSizes;
    uint32_t maxAlignedSplitSize = std::max(splitSize - splitSize % alignment, alignment);
    uint32_t usedSize = 0;
    while (usedSize < totalSize) {
        uint32_t partSize = std::min(totalSize - usedSize, maxAlignedSplitSize);
        splitSizes.push_back(partSize);
        usedSize += partSize;
    }
    return splitSizes;
}

// @brief Returns pair of axis and sizes of split outputs to split the input tensor to aligned parts, taking into
// account GNA HW limitations
inline std::pair<int64_t, std::vector<uint32_t>> AlignedSplitSizesPerAxis(InferenceEngine::SizeVector dims) {
    std::vector<uint32_t> splitSizes = {};
    auto totalElementsSize = static_cast<uint32_t>(InferenceEngine::details::product(std::begin(dims), std::end(dims)));
    auto firstValuableDim = std::find_if(std::begin(dims), std::end(dims), [](size_t val) {
        return val > 1;
    });
    IE_ASSERT(firstValuableDim != std::end(dims));
    auto splittedElementsSize = static_cast<uint32_t>(*firstValuableDim);
    auto splittedDimIx = std::distance(std::begin(dims), firstValuableDim);
    auto alignment = static_cast<uint32_t>(limitations::Limitations::get_instance()->get_memory_alignment());

    // Split output size should be multiple of device memory alignment to avoid align filters insertion,
    // but we need to check if our input size to split exceeds alignment; if not we can always
    // split if the remaining size is aligned
    auto split_size = limitations::Limitations::kBufferMaxSize * splittedElementsSize / totalElementsSize;

    if (splittedElementsSize <= alignment || split_size < alignment) {
        if ((totalElementsSize / splittedElementsSize) % alignment == 0) {
            alignment = 1;
        } else {
            return {splittedDimIx, splitSizes};
        }
    }
    splitSizes = GetAlignedSplitSizes(splittedElementsSize, split_size, alignment);
    return {splittedDimIx, splitSizes};
}

}  // namespace intel_gna
}  // namespace ov
