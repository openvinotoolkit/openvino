// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/core/model.hpp"
#include "openvino/frontend/extension.hpp"
#include "openvino/frontend/manager.hpp"

// Resolves different backend names to an internal device enumeration
inline std::string backend_name_to_device(const std::string& backend_name) {
    if (backend_name == "INTERPRETER")
        return ov::test::utils::DEVICE_TEMPLATE;
    if (backend_name == "IE_CPU")
        return ov::test::utils::DEVICE_CPU;
    if (backend_name == "IE_GPU")
        return ov::test::utils::DEVICE_GPU;
    throw "Unsupported backend name";
}

namespace ov {
namespace frontend {
namespace onnx {
namespace tests {

extern const std::string ONNX_FE;

// A wrapper to create ONNX Frontend and configure the conversion pipeline
std::shared_ptr<ov::Model> convert_model(const std::string& model_path,
                                         const ov::frontend::ConversionExtensionBase::Ptr& conv_ext = nullptr);
// A wrapper to create ONNX Frontend and configure the conversion pipeline
std::shared_ptr<ov::Model> convert_model(std::ifstream& model_stream);
// A wrapper to create ONNX Frontend and configure the conversion pipeline to get
// a model with possible Framework Nodes
std::shared_ptr<ov::Model> convert_partially(const std::string& model_path);
// Returns loaded InputModel for customizing before conversion
// If FrontEnd::Ptr has been passed - return a FrontEnd object which was used for loading model
InputModel::Ptr load_model(const std::string& model_path, ov::frontend::FrontEnd::Ptr* return_front_end = nullptr);
InputModel::Ptr load_model(const std::wstring& model_path, ov::frontend::FrontEnd::Ptr* return_front_end = nullptr);
// Returns path to a manifest file
std::string onnx_backend_manifest(const std::string& manifest);

// Duplicate implementation for tests - will be removed when ONNX_ITERATOR is removed
inline bool is_graph_iterator_enabled() {
    const char* env_value = std::getenv("ONNX_ITERATOR");
    if (env_value == nullptr) {
        return true;  // Enabled by default
    }

    std::string value(env_value);
    // Remove whitespace
    value.erase(std::remove_if(value.begin(),
                               value.end(),
                               [](unsigned char ch) {
                                   return std::isspace(ch);
                               }),
                value.end());
    // Convert to lowercase
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    static const std::unordered_map<std::string, bool> valid_values = {{"1", true},
                                                                       {"true", true},
                                                                       {"on", true},
                                                                       {"enable", true},
                                                                       {"0", false},
                                                                       {"false", false},
                                                                       {"off", false},
                                                                       {"disable", false}};

    auto it = valid_values.find(value);
    if (it != valid_values.end()) {
        return it->second;
    }

    throw std::runtime_error(std::string{"Unknown value for ONNX_ITERATOR environment variable: '"} + env_value +
                             "'. Expected 1 (enable) or 0 (disable).");
}

}  // namespace tests
}  // namespace onnx
}  // namespace frontend
}  // namespace ov

// For compatibility purposes, need to remove when will be unused
extern const std::string ONNX_FE;
