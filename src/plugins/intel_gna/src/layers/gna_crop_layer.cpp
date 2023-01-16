// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <vector>

#include "gna_crop_layer.hpp"
#include "log/log.hpp"
#include "log/debug.hpp"

using namespace ov::intel_gna;

namespace GNAPluginNS {

SimpleCrop get_crop_params(const std::vector<int32_t>& axis_in,
                           const std::vector<int32_t>& offset_in,
                           const std::vector<int32_t>& dim_in,
                           const std::vector<size_t>& input_dims) {
    const auto total_size_to_crop = ov::shape_size(input_dims);

    SimpleCrop ret{0, total_size_to_crop};

    auto crop_axis_detected = false;
    auto cropped_dim_size = total_size_to_crop;
    for (int n = 0; n < axis_in.size(); n++) {
        const auto axis = axis_in[n];
        if (axis < 0 || axis >= input_dims.size()) {
            log::warning() << "Crop axis outside of input shape size detected.\n";
            continue;
        }
        const auto input_dim = input_dims[axis];
        // Skip axis that is untouched
        if (offset_in[n] == 0 && dim_in[n] == input_dim) {
            continue;
        }
        if (crop_axis_detected) {
            THROW_GNA_EXCEPTION
                << "Crop layer does not support the number of (non-trivial) cropped dimensions more than 1.";
        }
        crop_axis_detected = true;
        ret.crop_size = dim_in[n];
        ret.start_offset = offset_in[n];
        cropped_dim_size = input_dim;
    }

    if (crop_axis_detected) {
        ret.start_offset *= (total_size_to_crop / cropped_dim_size);
        ret.crop_size *= (total_size_to_crop / cropped_dim_size);
    }

    return ret;
}

SimpleCrop GetCropParams(InferenceEngine::CropLayer* cropLayer) {
    auto input_dims = cropLayer->insData.begin()->lock()->getDims();

    const auto out_val = get_crop_params(cropLayer->axis, cropLayer->offset, cropLayer->dim, input_dims);
    return out_val;
}

}  // namespace GNAPluginNS
