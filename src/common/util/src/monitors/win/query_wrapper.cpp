// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "query_wrapper.hpp"

#define NOMINMAX
#include <windows.h>

#include <string>
#include <system_error>

QueryWrapper::QueryWrapper() {
    PDH_STATUS status = PdhOpenQuery(NULL, NULL, &query);
    if (ERROR_SUCCESS != status) {
        throw std::runtime_error("PdhOpenQuery() failed. Error status: " + std::to_string(status));
    }
}
QueryWrapper::~QueryWrapper() {
    PdhCloseQuery(query);
}

QueryWrapper::operator PDH_HQUERY() const {
    return query;
}
