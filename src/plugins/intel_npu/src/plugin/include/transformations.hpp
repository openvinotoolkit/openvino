// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {
namespace batch_helpers {

/**
 * @brief Detects if shape contains dynamic dimensions other than the batch dimension
 * Plugin-side batch handling can only be applied when batch is the sole dynamic dimension
 * @param shape The partial shape to check
 * @return true if there are dynamic dimensions other than batch, false otherwise
 */
bool hasOtherDynamicDims(const ov::PartialShape& shape);

/**
 * @brief Checks if model has dynamic dimensions other than batch in inputs or outputs
 * @param model The OpenVINO model to check
 * @return true if model has other dynamic dimensions, false otherwise
 */
bool checkModelDynamicDims(const std::shared_ptr<const ov::Model>& model);

/**
 * @brief Validates if the model is suitable for plugin-side batching
 * @param model The OpenVINO model to validate
 * @param logger Logger instance for diagnostic messages
 * @return true if model supports plugin batching, false otherwise
 */
bool validateModelBatch(const std::shared_ptr<const ov::Model>& model, Logger logger);

/**
 * @brief Attempts to debatch a model by setting batch dimension to specified value
 * @param model The model to debatch (modified in-place)
 * @param newBatch The new batch dimension value
 * @param originalBatch Output parameter to store the original batch dimension
 * @return true if debatching was successful, false otherwise
 */
bool deBatchModel(std::shared_ptr<ov::Model>& model,
                  ov::Dimension newBatch,
                  std::optional<ov::Dimension>& originalBatch);

/**
 * @brief Handles plugin-side batching logic including validation and model reshaping
 * @param model The model to process
 * @param localConfig Configuration
 * @param originalBatch Output parameter to store original batch dimension
 * @param logger Logger instance for diagnostic messages
 */
std::tuple<std::shared_ptr<ov::Model>, bool> handlePluginBatching(
    std::shared_ptr<const ov::Model> model,
    FilteredConfig& localConfig,
    const std::function<void(ov::intel_npu::BatchMode)>& updateBatchMode,
    std::optional<ov::Dimension>& originalBatch,
    Logger logger);

}  // namespace batch_helpers
}  // namespace intel_npu
