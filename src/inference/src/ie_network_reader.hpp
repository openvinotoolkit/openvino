// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "cpp/ie_cnn_network.h"
#include "ie_blob.h"
#include "ie_iextension.h"
#include "openvino/core/extension.hpp"
#include "openvino/frontend/manager.hpp"

namespace InferenceEngine {
namespace details {

/**
 * @brief Reads IR xml and bin files
 * @param modelPath path to IR file
 * @param binPath path to bin file, if path is empty, will try to read bin file with the same name as xml and
 * if bin file with the same name was not found, will load IR without weights.
 * @param exts vector with extensions
 * @param ov_exts vector with OpenVINO extensions
 * @param newAPI Whether this function is called from OpenVINO 2.0 API
 * @param frontEndManager FrontEndManager to load corresponding frontend plugin
 * @return CNNNetwork
 */
CNNNetwork ReadNetwork(const std::string& modelPath,
                       const std::string& binPath,
                       const std::vector<IExtensionPtr>& exts,
                       const std::vector<ov::Extension::Ptr>& ov_exts,
                       bool newAPI,
                       ov::frontend::FrontEndManager& frontEndManager);
/**
 * @brief Reads IR xml and bin (with the same name) files
 * @param model string with IR
 * @param weights shared pointer to constant blob with weights
 * @param exts vector with extensions
 * @param ov_exts vector with OpenVINO extensions
 * @param newAPI Whether this function is called from OpenVINO 2.0 API
 * @param frontEndManager FrontEndManager to load corresponding frontend plugin
 * @return CNNNetwork
 */
CNNNetwork ReadNetwork(const std::string& model,
                       const Blob::CPtr& weights,
                       const std::vector<IExtensionPtr>& exts,
                       const std::vector<ov::Extension::Ptr>& ov_exts,
                       bool newAPI,
                       ov::frontend::FrontEndManager& frontEndManager);

}  // namespace details
}  // namespace InferenceEngine
