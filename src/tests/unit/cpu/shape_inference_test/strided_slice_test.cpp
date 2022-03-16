// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <strided_slice_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, StridedSlice1) {
    auto data = std::make_shared<op::v0::Parameter>(ngraph::element::f32, ov::PartialShape::dynamic());
    auto begin = op::v0::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {100});
    auto end = op::v0::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {-100});
    auto stride = op::v0::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {-1});

    std::vector<int64_t> begin_mask = {0, 0, 0, 0};
    std::vector<int64_t> end_mask = {0, 0, 0, 0};

    auto ss = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    check_static_shape(ss.get(),
                       {StaticShape{3, 4, 5}, StaticShape{3}, StaticShape{3}, StaticShape{3}},
                       {StaticShape{3, 4, 5}});
}

TEST(StaticShapeInferenceTest, StridedSlice2) {
    auto data = std::make_shared<op::v0::Parameter>(ngraph::element::f32, ov::PartialShape::dynamic());
    auto begin = std::make_shared<op::v0::Parameter>(ngraph::element::i64, ngraph::Shape{3});
    auto end = std::make_shared<op::v0::Parameter>(ngraph::element::i64, ngraph::Shape{3});
    auto stride = std::make_shared<op::v0::Parameter>(ngraph::element::i64, ngraph::Shape{3});

    std::vector<int64_t> begin_mask(3, 0);
    std::vector<int64_t> end_mask(3, 0);

    auto ss = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    check_static_shape(ss.get(),
                       {StaticShape{3, 2, 3}, {1, 0, 0}, {2, 1, 3}, {1, 1, 1}},
                       {StaticShape{1, 1, 3}});

    check_static_shape(ss.get(),
                       {StaticShape{3, 2, 3}, {1, 0, 0}, {2, 2, 3}, {1, 1, 1}},
                       {StaticShape{1, 2, 3}});

    check_static_shape(ss.get(),
                       {StaticShape{3, 2, 3}, {2, 0, 0}, {3, 2, 3}, {1, 1, 2}},
                       {StaticShape{1, 2, 2}});
}

TEST(StaticShapeInferenceTest, StridedSlice3) {
    auto data = std::make_shared<op::v0::Parameter>(ngraph::element::f32, ov::PartialShape::dynamic());
    auto begin = std::make_shared<op::v0::Parameter>(ngraph::element::i64, ngraph::Shape{3});
    auto end = std::make_shared<op::v0::Parameter>(ngraph::element::i64, ngraph::Shape{3});
    auto stride = std::make_shared<op::v0::Parameter>(ngraph::element::i64, ngraph::Shape{3});

    std::vector<int64_t> begin_mask{0, 1, 1};
    std::vector<int64_t> end_mask{1, 1, 1};

    auto ss = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    check_static_shape(ss.get(),
                       {StaticShape{3, 2, 3}, {1, 0, 0}, {0, 0, 0}, {1, 1, 1}},
                       {StaticShape{2, 2, 3}});
}

TEST(StaticShapeInferenceTest, StridedSlice4) {
    auto data = std::make_shared<op::v0::Parameter>(ngraph::element::f32, ov::PartialShape::dynamic());
    auto begin = std::make_shared<op::v0::Parameter>(ngraph::element::i64, ngraph::Shape{3});
    auto end = std::make_shared<op::v0::Parameter>(ngraph::element::i64, ngraph::Shape{3});
    auto stride = std::make_shared<op::v0::Parameter>(ngraph::element::i64, ngraph::Shape{3});

    std::vector<int64_t> begin_mask{1, 0, 1};
    std::vector<int64_t> end_mask{0, 1, 1};

    auto ss = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    check_static_shape(ss.get(),
                       {StaticShape{3, 2, 3}, {0, 1, 0}, {2, 0, 0}, {1, 1, 2}},
                       {StaticShape{2, 1, 2}});
}

TEST(StaticShapeInferenceTest, StridedSlice5) {
    auto data = std::make_shared<op::v0::Parameter>(ngraph::element::f32, ov::PartialShape::dynamic());
    auto begin = std::make_shared<op::v0::Parameter>(ngraph::element::i64, ngraph::Shape{3});
    auto end = std::make_shared<op::v0::Parameter>(ngraph::element::i64, ngraph::Shape{3});
    auto stride = std::make_shared<op::v0::Parameter>(ngraph::element::i64, ngraph::Shape{3});

    std::vector<int64_t> begin_mask{0, 1, 1};
    std::vector<int64_t> end_mask{0, 1, 1};

    auto ss = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    check_static_shape(ss.get(),
                       {StaticShape{3, 2, 3}, {0, 0, 0}, {1, 0, 0}, {1, 1, -1}},
                       {StaticShape{1, 2, 3}});
}