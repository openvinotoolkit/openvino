// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "query_wrapper.h"

#define NOMINMAX
#include <windows.h>

#include <system_error>

QueryWrapper::QueryWrapper() {
    PDH_STATUS status = PdhOpenQuery(NULL, NULL, &query);
    if (ERROR_SUCCESS != status) {
        throw std::system_error(status, std::system_category(), "PdhOpenQuery() failed");
    }
}
QueryWrapper::~QueryWrapper() {
    PdhCloseQuery(query);
}

QueryWrapper::operator PDH_HQUERY() const {
    return query;
}
