// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <strided_slice_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;

TEST(StaticShapeInferenceTest, StridedSlice) {
    auto data = std::make_shared<op::v0::Parameter>(ngraph::element::f32, ov::PartialShape::dynamic());
    auto begin = op::v0::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {100});
    auto end = op::v0::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {-100});
    auto stride = op::v0::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {-1});

    std::vector<int64_t> begin_mask = {0, 0, 0, 0};
    std::vector<int64_t> end_mask = {0, 0, 0, 0};

    auto ss = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    check_static_shape(ss.get(),
                       {ov::StaticShape{3, 4, 5}, ov::StaticShape{3}, ov::StaticShape{3}, ov::StaticShape{3}},
                       {ov::StaticShape{3, 4, 5}});
}
