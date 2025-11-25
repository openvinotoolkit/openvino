// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/network_metadata.hpp"
#include "openvino/core/model.hpp"

namespace intel_npu {

bool isInitMetadata(const NetworkMetadata& networkMetadata);

/**
 * @brief Stores the information within the "WeightlessCacheAttribute" as runtime fields that persist upon
 * serialization.
 * @details Constant nodes (weights) may contain as medatadata the "WeightlessCacheAttribute", that is information
 * regarding the offset of the weights within the binary file, as well as the original size and precision. This
 * information is required within the "weights separation" flow, therefore this function is here to store it.
 * @note Not calling this function in the weights separation flow would lead to this information being lost upon
 * serialization. The "WeightlessCacheAttribute" information that is populated upon de-serialization would represent
 * metadata corresponding to the serialized stream, not the original weights file. Therefore the compiler would be
 * misinformed and lookups of weights offsets could fail.
 *
 * @param model Both source and target.
 */
void storeWeightlessCacheAttribute(const std::shared_ptr<ov::Model>& model);
}  // namespace intel_npu
