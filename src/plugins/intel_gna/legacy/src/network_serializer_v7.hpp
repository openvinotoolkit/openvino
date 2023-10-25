// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>

#include <string>
#include <vector>

#include "compilation_context.hpp"
#include "cpp/ie_cnn_network.h"

namespace InferenceEngine {
namespace Serialization {

/**
 * @brief Serialize network into IE IR XML file and binary weights file
 * @param xmlPath   Path to XML file
 * @param binPath   Path to BIN file
 * @param network   network to be serialized
 */
void Serialize(const std::string& xmlPath, const std::string& binPath, const InferenceEngine::CNNNetwork& network);

}  // namespace Serialization
}  // namespace InferenceEngine
