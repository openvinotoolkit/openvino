// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>
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

#define OPSET_RANGE(_in, _until) \
    VersionRange { _in, _until }
#define OPSET_SINCE(_since)         VersionRange::since(_since)
#define OPSET_IN(_in)               VersionRange::in(_in)
#define ONNX_OP_M(name, range, ...) register_translator(name, range, __VA_ARGS__)
#define ONNX_OP(name, range, ...)   static bool onnx_op_reg = ONNX_OP_M(name, range, __VA_ARGS__)

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
