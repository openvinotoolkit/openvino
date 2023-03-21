// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>

#include <cstdint>
#include <vector>

namespace ov {
namespace intel_gna {

class GNACropLayer {
    InferenceEngine::CNNLayerPtr cropLayer;

public:
    explicit GNACropLayer(InferenceEngine::CNNLayerPtr layer) : cropLayer(layer) {}

    InferenceEngine::CNNLayerPtr getCrop() {
        return cropLayer;
    }

    /**
     * pointer to gna croped memory beginning
     */
    void* gna_ptr = nullptr;
};

struct SimpleCrop {
    size_t start_offset;
    size_t crop_size;
};

/**
 * @brief returns parameters extracted from Crop layer: elements offset, elements output size and axes
 * @param cropLayer pointer to a Crop layer
 */
SimpleCrop get_crop_params(const std::vector<int32_t>& axis_in,
                           const std::vector<int32_t>& offset_in,
                           const std::vector<int32_t>& dim_in,
                           const std::vector<size_t>& input_dims);

SimpleCrop GetCropParams(InferenceEngine::CropLayer* cropLayer);

}  // namespace intel_gna
}  // namespace ov
