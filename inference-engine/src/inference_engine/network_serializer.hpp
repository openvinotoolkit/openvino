// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_icnn_network.hpp>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace Serialization {

/**
 * @brief Serializes a network into IE IR v10 XML file and binary weights file
 * @param xmlPath   Path to XML file
 * @param binPath   Path to BIN file
 * @param function  Ngraph function to be serialized
 */
void SerializeV10(const std::string &xmlPath, const std::string &binPath,
                  const ngraph::Function &function);

}  // namespace Serialization
}  // namespace InferenceEngine
