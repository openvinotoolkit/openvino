// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <vector>

#include "runtime/pwl.h"

void make_gna_pwl(const DnnActivation& fun,
                  const std::vector<pwl_t>& pwl,
                  const double l_bound,
                  const double u_bound,
                  const double in_scale,
                  const double out_scale,
                  const bool low_precision,
                  const bool is_fused_with_conv2d,
                  std::vector<gna_pwl_segment_t>& gna_pwl);
void make_gna_pwl(const std::shared_ptr<ngraph::Node>& node,
                  const double in_scale,
                  const double out_scale,
                  const bool low_precision,
                  std::vector<gna_pwl_segment_t>& gna_pwl);
