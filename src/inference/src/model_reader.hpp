// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "openvino/core/extension.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/aligned_buffer.hpp"

namespace ov {

namespace util {

/**
 * @brief Reads model
 * @param modelPath path to Model file
 * @param binPath optional path for model weights. If empty for IR we will find bin file with the model name.
 * if bin file with the same name was not found, will load IR without weights.
 * @param extensions vector with OpenVINO extensions
 * @param enable_mmap boolean to enable/disable `mmap` use in Frontend
 * @return Shared pointer to ov::Model
 */
std::shared_ptr<ov::Model> read_model(const std::string& modelPath,
                                      const std::string& binPath,
                                      const std::vector<ov::Extension::Ptr>& extensions,
                                      bool enable_mmap);

/**
 * @brief Reads model
 * @param model shared pointer to aligned buffer with IR.
 * @param weights shared pointer to aligned buffer with weights.
 * @param extensions vector with OpenVINO extensions
 * @return Shared pointer to ov::Model
 */
std::shared_ptr<ov::Model> read_model(const std::shared_ptr<ov::AlignedBuffer>& model,
                                      const std::shared_ptr<ov::AlignedBuffer>& weights,
                                      const std::vector<ov::Extension::Ptr>& extensions);

/**
 * @brief Reads model
 * @param model Serialized model representation
 * @param weights constant Tensor with weights
 * @param extensions vector with OpenVINO extensions
 * @param frontendMode read network without post-processing or other transformations
 * @return Shared pointer to ov::Model
 */
std::shared_ptr<ov::Model> read_model(const std::string& model,
                                      const ov::Tensor& weights,
                                      const std::vector<ov::Extension::Ptr>& extensions,
                                      bool frontendMode = false);

}  // namespace util
}  // namespace ov
