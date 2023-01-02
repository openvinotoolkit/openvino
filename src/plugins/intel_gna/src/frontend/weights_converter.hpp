// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "precision_utils.h"
#include "layers/gna_layer_info.hpp"

namespace ov {
namespace intel_gna {
namespace frontend {

/**
 * @brief Create an FP32 blob from an FP16 one
 * @param fp16_blob Pointer to an FP16 blob
 * @return Pointer to an FP32 blob
 */
InferenceEngine::Blob::Ptr make_fp32_blob(InferenceEngine::Blob::Ptr fp16_blob);

/**
 * @brief Convert all blobs of a layer from FP32 to FP16 precision
 * @param layer A layer which blobs are to be converted
 */
void convert_blobs_precision(InferenceEngine::CNNLayer& layer);

}  // namespace frontend
}  // namespace intel_gna
}  // namespace ov
