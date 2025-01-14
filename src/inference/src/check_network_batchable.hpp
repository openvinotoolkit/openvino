// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>

#include "openvino/core/model.hpp"

namespace ov {
namespace details {
/**
 * @brief Checks if the input model is batch-able (e.g. no dynamic inputs, inputs has the batch dimension, etc)
 * @param model A model to check for automatic-batching applicability
 * @return An enum value indicating whether the model can be safely batched (with HETERO or as is) or not
 */
enum class NetworkBatchAbility : uint32_t { NO = 0, AS_IS, WITH_HETERO };
NetworkBatchAbility is_model_batchable(const std::shared_ptr<const ov::Model>& model,
                                       const std::string& deviceNoBatch,
                                       bool strictly_track_dims);
/**
 * @brief Sets BATCH affinity for all the nodes except DetectionOutput
 * @param model_ A model to set affinity to
 * @param deviceNameWithoutBatch Device name to set for DetectionOutput node if any
 * @return A copy of the model with set affinity
 */
std::shared_ptr<const ov::Model> apply_batch_affinity(const std::shared_ptr<const ov::Model>& model_,
                                                      const std::string& deviceNameWithoutBatch);

}  // namespace details
}  // namespace ov
