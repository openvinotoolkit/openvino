// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

/**
 * @brief Retrieves the latest VCL error log associated with the given log handle.
 * @note Shared by every VCL caller (compiler, profiling decoder) so the log-retrieval protocol lives in one place.
 */
std::string getLatestVCLLog(const VCLFunctionTable& functions, vcl_log_handle_t logHandle);

}  // namespace intel_npu

/**
 * @brief Throws with the VCL error log appended when `ret` is not VCL_RESULT_SUCCESS.
 * @param functions The function table to fetch the error log through, passed explicitly rather than
 * captured from the enclosing scope so the macro is usable from free functions and member functions alike.
 */
#define THROW_ON_FAIL_FOR_VCL(functions, step, ret, logHandle)                    \
    do {                                                                          \
        const vcl_result_t vclResult_ = (ret);                                    \
        if (vclResult_ != VCL_RESULT_SUCCESS) {                                   \
            OPENVINO_THROW("Failed to call VCL API : ",                           \
                           step,                                                  \
                           " result: 0x",                                         \
                           std::hex,                                              \
                           vclResult_,                                            \
                           " - ",                                                 \
                           ::intel_npu::getLatestVCLLog((functions), logHandle)); \
        }                                                                         \
    } while (0)
