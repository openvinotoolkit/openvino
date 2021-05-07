// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include "runtime/pwl.h"

void make_gna_pwl(const DnnActivation  fun,
                  const std::vector<pwl_t>& pwl,
                  const double l_bound,
                  const double u_bound,
                  const double in_scale,
                  const double out_scale,
                  const bool low_precision,
                  std::vector<gna_pwl_segment_t> &gna_pwl);
