// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <backend/dnn_types.hpp>
#include <ie_precision.hpp>
#include <memory>

#include "data_conversion_helpers.hpp"
#include "hw_accelerated_converter.hpp"

namespace ov {
namespace intel_gna {
namespace pre_post_processing {

class InputOutputDataHandler {
public:
    InputOutputDataHandler(std::shared_ptr<HwAcceleratedDataConverter> converter = nullptr);
    void import_frames(void* ptr_dst,
                       const void* ptr_src,
                       const InferenceEngine::Precision& input_precision,
                       float scale_factor,
                       intel_dnn_orientation_t orientation,
                       size_t num_frames,
                       size_t num_group,
                       size_t num_vector_elements,
                       size_t num_vector_stride,
                       bool input_low_precision,
                       bool is_gna_device);

    void export_scores(void* ptr_dst,
                       const void* ptr_src,
                       intel_dnn_orientation_t orientation,
                       size_t num_frames,
                       size_t num_group,
                       size_t num_vector_elements,
                       size_t num_active_elements,
                       size_t num_vector_stride,
                       const InferenceEngine::Precision& precision_in,
                       const InferenceEngine::Precision& precision_out,
                       const float scale_factor);

private:
    std::shared_ptr<HwAcceleratedDataConverter> m_hw_accelerated_converter;
};
}  // namespace pre_post_processing
}  // namespace intel_gna
}  // namespace ov