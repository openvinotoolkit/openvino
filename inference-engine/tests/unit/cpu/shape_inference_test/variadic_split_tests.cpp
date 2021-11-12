// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <variadic_split_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;

static std::shared_ptr<op::v1::VariadicSplit> build_variadic_split(PartialShape data_shape,
                                                                   std::initializer_list<int64_t> axis_value,
                                                                   std::initializer_list<int64_t> splits) {
    std::shared_ptr<op::Op> axis;
    std::shared_ptr<op::Op> splits_len;

    const auto data = std::make_shared<op::v0::Parameter>(element::i32, data_shape);
    if (axis_value.size())
        axis = std::static_pointer_cast<op::Op>(
            op::v0::Constant::create(element::i64, ov::Shape{}, {*axis_value.begin()}));
    else
        axis = std::static_pointer_cast<op::Op>(
            std::make_shared<op::v0::Parameter>(element::i64, ov::PartialShape::dynamic(ov::Rank(0))));

    if (splits.size())
        splits_len =
            std::static_pointer_cast<op::Op>(op::v0::Constant::create(element::i64, ov::Shape{splits.size()}, splits));
    else
        splits_len = std::static_pointer_cast<op::Op>(
            std::make_shared<op::v0::Parameter>(element::i64, ov::PartialShape::dynamic(ov::Rank(1))));

    return std::make_shared<op::v1::VariadicSplit>(data, axis, splits_len);
}

TEST(StaticShapeInferenceTest, VariadicSplitV1) {
    const auto split = build_variadic_split(Shape{2, 6}, {1}, {2, 4});
    check_output_shape(split, {{2, 2}, {2, 4}});

    check_output_shape(build_variadic_split(Shape{12, 6}, {-2}, {7, -1, 2}), {{7, 6}, {3, 6}, {2, 6}});
    check_output_shape(build_variadic_split(Shape{12, 6}, {-2}, {-1, 7, 2}), {{3, 6}, {7, 6}, {2, 6}});
    check_output_shape(build_variadic_split(Shape{12, 1, 6}, {2}, {3, 1, 2}), {{12, 1, 3}, {12, 1, 1}, {12, 1, 2}});
    check_output_shape(build_variadic_split(Shape{12, 6}, {1}, {6, 0}), {{12, 6}, {12, 0}});

    check_static_shape(split,
                       {StaticShape{12, 6}, {-2}, TestTensor(StaticShape{3}, {7, -1, 2})},
                       {{7, 6}, {3, 6}, {2, 6}});
}

TEST(StaticShapeInferenceTest, VariadicSplitV1_StaticWithConstMap) {
    check_static_shape(build_variadic_split(Shape{2, 6}, {}, {}),
                       {StaticShape{12, 6}, {-2}, {7, -1, 2}},
                       {{7, 6}, {3, 6}, {2, 6}});
}

TEST(StaticShapeInferenceTest, VariadicSplitV1_StaticNoConstMap) {
    check_static_shape(build_variadic_split(Shape{2, 6}, {}, {}),
                       {StaticShape{2, 6}, StaticShape{}, StaticShape{3}},
                       {},
                       true);
}

TEST(StaticShapeInferenceTest, VariadicSplitV1_StaticWrongConstMap1) {
    check_static_shape(build_variadic_split(Shape{2, 6}, {}, {}), {StaticShape{12, 6}, {-2}, {7, 99, 2}}, {}, true);
}

TEST(StaticShapeInferenceTest, VariadicSplitV1_StaticWrongConstMap2) {
    check_static_shape(build_variadic_split(Shape{2, 6}, {}, {}),
                       {StaticShape{12, 6}, {-2}, {StaticShape{2, 2}, std::initializer_list<int>{7, -1, 2, 1}}},
                       {},
                       true);
}
