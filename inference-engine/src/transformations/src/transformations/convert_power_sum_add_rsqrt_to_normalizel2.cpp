// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_power_sum_add_rsqrt_to_normalizel2.hpp"

#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>

void ngraph::pass::ConvertPowerSumAddRsqrtToNormalizeL2::convert_to_normalize_l2() {
    auto data       = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto square     = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto power_sqr  = std::make_shared<ngraph::opset1::Power>(data, square);
    auto axes       = std::make_shared<pattern::op::Label>(element::i64, Shape{4});
    auto reduce_sum = std::make_shared<ngraph::opset1::ReduceSum>(power_sqr, axes);
    auto epsilon    = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto add        = std::make_shared<ngraph::opset1::Add>(reduce_sum, epsilon);
    auto sqrt       = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto power_sqrt = std::make_shared<ngraph::opset1::Power>(add, sqrt);
    auto minus_one  = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto inv        = std::make_shared<ngraph::opset1::Power>(power_sqrt, minus_one);
    auto mul        = std::make_shared<ngraph::opset1::Multiply>(data, inv);
}
