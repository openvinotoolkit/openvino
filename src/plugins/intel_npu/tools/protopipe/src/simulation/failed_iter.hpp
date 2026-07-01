//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>

struct FailedIter {
    size_t iter_idx;
    std::vector<std::string> reasons;
};
