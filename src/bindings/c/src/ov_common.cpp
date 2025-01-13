// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_common.h"

/**
 * @variable global value for error info.
 * Don't change its order.
 */
char const* error_infos[] = {"success",
                             "general error",
                             "it's not implement",
                             "failed to network",
                             "input parameter mismatch",
                             "cannot find the value",
                             "out of bounds",
                             "run with unexpected error",
                             "request is busy",
                             "result is not ready",
                             "it is not allocated",
                             "inference start with error",
                             "network is not ready",
                             "inference is canceled",
                             "invalid c input parameters",
                             "unknown c error",
                             "not implement in c method",
                             "unknown exception"};

const char* ov_get_error_info(ov_status_e status) {
    auto index = -status;
    auto max_index = sizeof(error_infos) / sizeof(error_infos[0]) - 1;
    if (static_cast<size_t>(index) > max_index)
        return error_infos[max_index];
    return error_infos[index];
}

void ov_free(const char* content) {
    if (content)
        delete[] content;
}
