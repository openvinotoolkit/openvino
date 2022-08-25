// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <split_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

static std::shared_ptr<op::v1::Split> build_split(PartialShape data_shape,
                                                  std::initializer_list<int64_t> axis_value,
                                                  size_t num_splits) {
    std::shared_ptr<ov::Node> axis;
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
    if (axis_value.size())
        axis = op::v0::Constant::create(element::i64, ov::Shape{}, {*axis_value.begin()});
    else
        axis = std::make_shared<op::v0::Parameter>(element::i64, ov::PartialShape{});

    return std::make_shared<op::v1::Split>(data, axis, num_splits);
}

TEST(StaticShapeInferenceTest, SplitV1) {
    const auto op = build_split(PartialShape{-1, -1, -1}, {}, 3);
    check_static_shape(op.get(), {StaticShape{2, 3, 4}, 1}, {{2, 1, 4}, {2, 1, 4}, {2, 1, 4}});
}

TEST(StaticShapeInferenceTest, SplitV1_Dynamic) {
    check_output_shape(build_split(PartialShape({2, 8, 4}), {}, 4).get(),
                       {ov::PartialShape::dynamic(ov::Rank(3)),
                        ov::PartialShape::dynamic(ov::Rank(3)),
                        ov::PartialShape::dynamic(ov::Rank(3)),
                        ov::PartialShape::dynamic(ov::Rank(3))});
}

TEST(StaticShapeInferenceTest, SplitV1_StaticWithConstMap) {
    check_static_shape(build_split(PartialShape({-1, -1, -1}), {}, 4).get(),
                       {StaticShape{2, 8, 4}, 2},
                       {{2, 8, 1}, {2, 8, 1}, {2, 8, 1}, {2, 8, 1}});
}
