// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

std::string jit_emitter_pretty_name(const std::string& pretty_func);

#ifdef __GNUC__
#    define OV_CPU_JIT_EMITTER_NAME jit_emitter_pretty_name(__PRETTY_FUNCTION__)
#else /* __GNUC__ */
#    define OV_CPU_JIT_EMITTER_NAME jit_emitter_pretty_name(__FUNCSIG__)
#endif /* __GNUC__ */

#define OV_CPU_JIT_EMITTER_THROW(...)        OPENVINO_THROW(OV_CPU_JIT_EMITTER_NAME, ": ", __VA_ARGS__)
#define OV_CPU_JIT_EMITTER_ASSERT(cond, ...) OPENVINO_ASSERT((cond), OV_CPU_JIT_EMITTER_NAME, ": ", __VA_ARGS__)

ov::element::Type get_arithmetic_binary_exec_precision(const std::shared_ptr<ov::Node>& n);

}  // namespace ov::intel_cpu
