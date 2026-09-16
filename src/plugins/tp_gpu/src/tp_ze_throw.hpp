// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ios>

#include "openvino/core/except.hpp"
#include "openvino/zero_api.hpp"

namespace ov {
namespace tp_gpu {

/// Turns a failed Level Zero call into an ov::Exception.
inline void ze_throw(ze_result_t result, const char* what) {
    if (result != ZE_RESULT_SUCCESS) {
        OPENVINO_THROW("[TP][L0] ", what, " failed: 0x", std::hex, result);
    }
}

}  // namespace tp_gpu
}  // namespace ov

#define ZE_THROW(expr) ::ov::tp_gpu::ze_throw((expr), #expr)
