// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_rank.h"

#include "common.h"

ov_status_e ov_rank_init_dynamic(ov_rank_t* rank, int64_t min_rank, int64_t max_rank) {
    if (!rank || min_rank < -1 || max_rank < -1) {
        return ov_status_e::INVALID_C_PARAM;
    }
    rank->max = max_rank;
    rank->min = min_rank;
    return ov_status_e::OK;
}

ov_status_e ov_rank_init(ov_rank_t* rank, int64_t rank_value) {
    if (!rank || rank_value <= 0) {
        return ov_status_e::INVALID_C_PARAM;
    }
    return ov_rank_init_dynamic(rank, rank_value, rank_value);
}

bool ov_rank_is_dynamic(const ov_rank_t* rank) {
    if (!rank) {
        PRINT_ERROR("null rank");
        return true;
    }
    if (rank->min == rank->max && rank->max > 0)
        return false;
    return true;
}