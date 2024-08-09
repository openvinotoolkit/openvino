// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define NOMINMAX
#include <pdh.h>
class QueryWrapper {
public:
    QueryWrapper();
    ~QueryWrapper();
    QueryWrapper(const QueryWrapper&) = delete;
    QueryWrapper& operator=(const QueryWrapper&) = delete;
    operator PDH_HQUERY() const;

private:
    PDH_HQUERY query;
};
