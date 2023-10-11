// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "cpp/ie_cnn_network.h"
#include "ie_blob.h"
#include "ie_iextension.h"
#include "openvino/core/extension.hpp"

namespace InferenceEngine {
namespace details {

/**
 * @brief Reads IR xml and bin files
 * @param modelPath path to IR file
 * @param binPath path to bin file, if path is empty, will try to read bin file with the same name as xml and
 * if bin file with the same name was not found, will load IR without weights.
 * @param ov_exts vector with OpenVINO extensions
 * @param enable_mmap boolean to enable/disable `mmap` use in Frontend
 * @return CNNNetwork
 */
CNNNetwork ReadNetwork(const std::string& modelPath,
                       const std::string& binPath,
                       const std::vector<ov::Extension::Ptr>& ov_exts,
                       bool is_new_api,
                       bool enable_mmap);
/**
 * @brief Reads IR xml and bin (with the same name) files
 * @param model string with IR
 * @param weights shared pointer to constant blob with weights
 * @param ov_exts vector with OpenVINO extensions
 * @param frontendMode read network without post-processing or other transformations
 * @return CNNNetwork
 */
CNNNetwork ReadNetwork(const std::string& model,
                       const Blob::CPtr& weights,
                       const std::vector<ov::Extension::Ptr>& ov_exts,
                       bool is_new_api,
                       bool frontendMode = false);

}  // namespace details
}  // namespace InferenceEngine
