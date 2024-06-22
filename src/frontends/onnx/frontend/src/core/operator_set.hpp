// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <string>
#include <map>
#include <unordered_map>

#include "core/node.hpp"
#include "version_range.hpp"

namespace ov {
namespace frontend {
namespace onnx {

/// \brief      Function which transforms single ONNX operator to OV sub-graph.
using Operator = std::function<OutputVector(const Node&)>;

/// \brief      Map which contains ONNX operators accessible by std::string value as a key.
using OperatorSet = std::unordered_map<std::string, Operator>;

/// \brief      Map with map of versioned operators, accessible like map["Operation"][Version]
using DomainOpset = std::unordered_map<std::string, std::map<std::int64_t, Operator>>;

extern const char* OPENVINO_ONNX_DOMAIN;
extern const char* MICROSOFT_DOMAIN;
extern const char* PYTORCH_ATEN_DOMAIN;
extern const char* MMDEPLOY_DOMAIN;

/// \brief Registering a versions range of translator in global map of translators (preferred to use)
extern bool register_translator(const std::string name,
                                const VersionRange range,
                                const Operator fn,
                                const std::string domain = "");

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
