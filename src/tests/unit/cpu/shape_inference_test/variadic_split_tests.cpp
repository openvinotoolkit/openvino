// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <variadic_split_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

static std::shared_ptr<op::v1::VariadicSplit> build_variadic_split(PartialShape data_shape,
                                                                   std::initializer_list<int64_t> axis_value,
                                                                   std::initializer_list<int64_t> splits) {
    std::shared_ptr<ov::Node> axis;
    std::shared_ptr<ov::Node> splits_len;

    const auto data = std::make_shared<op::v0::Parameter>(element::i32, data_shape);
    if (axis_value.size())
        axis = op::v0::Constant::create(element::i64, ov::Shape{}, {*axis_value.begin()});
    else
        axis = std::make_shared<op::v0::Parameter>(element::i64, ov::PartialShape::dynamic(ov::Rank(0)));

    if (splits.size())
        splits_len = op::v0::Constant::create(element::i64, ov::Shape{splits.size()}, splits);
    else
        splits_len = std::make_shared<op::v0::Parameter>(element::i64, ov::PartialShape::dynamic(ov::Rank(1)));

    return std::make_shared<op::v1::VariadicSplit>(data, axis, splits_len);
}

TEST(StaticShapeInferenceTest, VariadicSplitV1) {
    const auto split = build_variadic_split(ov::PartialShape::dynamic(), {}, {});

    check_static_shape(split.get(),
                       {StaticShape{12, 6}, {-2}, {7, -1, 2}},
                       {{7, 6}, {3, 6}, {2, 6}});
    check_static_shape(split.get(),
                       {StaticShape{12, 6}, {-2}, {-1, 7, 2}},
                       {{3, 6}, {7, 6}, {2, 6}});
    check_static_shape(split.get(),
                       {StaticShape{12, 1, 6}, {2}, {3, 1, 2}},
                       {{12, 1, 3}, {12, 1, 1}, {12, 1, 2}});
    check_static_shape(split.get(), {StaticShape{12, 6}, {1}, {6, 0}}, {{12, 6}, {12, 0}});
}

TEST(StaticShapeInferenceTest, VariadicSplitV1_StaticWithConstMap) {
    check_static_shape(build_variadic_split(ov::PartialShape{-1, -1}, {}, {}).get(),
                       {StaticShape{12, 6}, {-2}, {7, -1, 2}},
                       {{7, 6}, {3, 6}, {2, 6}});
}