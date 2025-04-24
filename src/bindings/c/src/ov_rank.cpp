// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_rank.h"

#include "common.h"

bool ov_rank_is_dynamic(const ov_rank_t rank) {
    if (rank.min == rank.max && rank.max > 0)
        return false;
    return true;
}
