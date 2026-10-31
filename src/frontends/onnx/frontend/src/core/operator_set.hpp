// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>
#include <unordered_map>

#include "core/node.hpp"
#include "version_range.hpp"

namespace ov::frontend::onnx {

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
extern const char* AIONNX_ML_DOMAIN;
extern const char* AIONNX_CONTRIB_DOMAIN;
// The default ONNX opset ("ai.onnx") is registered/looked up under an empty domain string.
extern const char* AIONNX_DOMAIN;

/// \brief Registering a versions range of translator in global map of translators (preferred to use)
extern bool register_translator(const std::string name,
                                const VersionRange range,
                                const Operator fn,
                                const std::string domain = "");

#define OPSET_RANGE(_in, _until) \
    VersionRange {               \
        _in, _until              \
    }
#define OPSET_SINCE(_since)         VersionRange::since(_since)
#define OPSET_IN(_in)               VersionRange::in(_in)
#define ONNX_OP_M(name, range, ...) register_translator(name, range, __VA_ARGS__)
// A unique variable name per registration keeps translation units mergeable (unity builds).
#define ONNX_OP_REG_CAT_(a, b) a##b
#define ONNX_OP_REG_CAT(a, b)  ONNX_OP_REG_CAT_(a, b)
#define ONNX_OP(name, range, ...) \
    static bool ONNX_OP_REG_CAT(onnx_op_reg_, __COUNTER__) = ONNX_OP_M(name, range, __VA_ARGS__)

}  // namespace ov::frontend::onnx
