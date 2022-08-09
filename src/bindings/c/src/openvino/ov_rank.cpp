// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_rank.h"

#include "common.h"

ov_status_e ov_rank_create_dynamic(ov_rank_t** rank, int64_t min_dimension, int64_t max_dimension) {
    if (!rank || min_dimension < -1 || max_dimension < -1) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<ov_rank_t> _rank(new ov_rank_t);
        if (min_dimension != max_dimension) {
            _rank->object = ov::Dimension(min_dimension, max_dimension);
        } else {
            if (min_dimension > -1) {
                _rank->object = ov::Dimension(min_dimension);
            } else {
                _rank->object = ov::Dimension();
            }
        }
        *rank = _rank.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_rank_create(ov_rank** rank, int64_t rank_value) {
    if (!rank || rank_value <= 0) {
        return ov_status_e::INVALID_C_PARAM;
    }
    return ov_rank_create_dynamic(rank, rank_value, rank_value);
}

void ov_rank_free(ov_rank_t* rank) {
    if (rank)
        delete rank;
}
