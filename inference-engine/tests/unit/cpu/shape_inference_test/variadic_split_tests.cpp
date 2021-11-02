// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <variadic_split_shape_inference.hpp>

#include "utils/shape_inference/static_shape.hpp"

using namespace ov;

static std::shared_ptr<op::v1::Split> build_split(PartialShape data_shape, int64_t axis_value, size_t num_splits) {
    std::shared_ptr<op::Op> axis;
    const auto data = std::make_shared<op::v0::Parameter>(element::i32, data_shape);
    if (axis_value >= 0)
        axis = std::static_pointer_cast<op::Op>(op::v0::Constant::create(element::i64, ov::Shape{}, {axis_value}));
    else
        axis = std::static_pointer_cast<op::Op>(std::make_shared<op::v0::Parameter>(element::i64, ov::PartialShape{}));

    return std::make_shared<op::v1::Split>(data, axis, num_splits);
}

TEST(StaticShapeInferenceTest, VariadicSplitV1) {
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});
    const auto axis = op::Constant::create<int64_t>(element::i64, Shape{}, {1});
    const auto splits = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 4});
    const auto split = make_shared<op::v1::VariadicSplit>(data, axis, splits);
    EXPECT_EQ(split->outputs().size(), 2);
    EXPECT_EQ(split->get_output_shape(0), (Shape{2, 2}));
    EXPECT_EQ(split->get_output_shape(1), (Shape{2, 4}));
    EXPECT_EQ(split->get_output_element_type(0), element::i32);
    EXPECT_EQ(split->get_output_element_type(1), element::i32);

    EXPECT_EQ(make_shared<op::v1::VariadicSplit>(make_shared<op::Parameter>(element::i32, Shape{12, 6}),
                                                 op::Constant::create<int64_t>(element::i64, Shape{}, {-2}),
                                                 op::Constant::create<int64_t>(element::i64, Shape{3}, {7, -1, 2}))
                  ->output(1)
                  .get_shape(),
              (Shape{3, 6}));

    EXPECT_EQ(make_shared<op::v1::VariadicSplit>(make_shared<op::Parameter>(element::i32, Shape{12, 6}),
                                                 op::Constant::create<int64_t>(element::i64, Shape{}, {-2}),
                                                 op::Constant::create<int64_t>(element::i64, Shape{3}, {-1, 7, 2}))
                  ->output(0)
                  .get_shape(),
              (Shape{3, 6}));

    EXPECT_EQ(make_shared<op::v1::VariadicSplit>(make_shared<op::Parameter>(element::i32, Shape{12, 1, 6}),
                                                 op::Constant::create<int64_t>(element::i64, Shape{1}, {2}),
                                                 op::Constant::create<int64_t>(element::i64, Shape{3}, {3, 1, 2}))
                  ->output(2)
                  .get_shape(),
              (Shape{12, 1, 2}));

    EXPECT_EQ(make_shared<op::v1::VariadicSplit>(make_shared<op::Parameter>(element::i32, Shape{12, 6}),
                                                 op::Constant::create<int64_t>(element::i64, Shape{1}, {1}),
                                                 op::Constant::create<int64_t>(element::i64, Shape{2}, {6, 0}))
                  ->output(1)
                  .get_shape(),
              (Shape{12, 0}));
}
