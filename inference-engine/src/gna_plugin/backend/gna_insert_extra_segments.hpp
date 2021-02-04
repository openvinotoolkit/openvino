// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include "backend/gna_types.h"
#include <runtime/pwl.h>
#include <gna_slope_scale.h>


void gna_insert_extra_segments(const std::vector<pwl_t> &pwl,
                               std::vector<gna_pwl_segment_t> &gna_pwl,
                               const double in_scale,
                               const double out_scale);