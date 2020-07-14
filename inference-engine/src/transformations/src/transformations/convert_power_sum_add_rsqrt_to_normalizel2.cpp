// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_power_sum_add_rsqrt_to_normalizel2.hpp"

#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>

void ngraph::pass::ConvertPowerSumAddRsqrtToNormalizeL2::convert_to_normalize_l2() {
    auto data   = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto square = ngraph::op::Constant::create(element::f32, Shape{1, 1, 1, 1}, {2.0});
}
