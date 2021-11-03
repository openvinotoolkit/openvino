// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <split_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;

static std::shared_ptr<op::v1::Split> build_split(PartialShape data_shape,
                                                  std::initializer_list<int64_t> axis_value,
                                                  size_t num_splits) {
    std::shared_ptr<op::Op> axis;
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
    if (axis_value.size())
        axis = std::static_pointer_cast<op::Op>(
            op::v0::Constant::create(element::i64, ov::Shape{}, {*axis_value.begin()}));
    else
        axis = std::static_pointer_cast<op::Op>(std::make_shared<op::v0::Parameter>(element::i64, ov::PartialShape{}));

    return std::make_shared<op::v1::Split>(data, axis, num_splits);
}

TEST(StaticShapeInferenceTest, SplitV1) {
    const auto op = build_split(PartialShape({2, 3, 4}), {1}, 3);

    check_output_shape(op, {{2, 1, 4}, {2, 1, 4}, {2, 1, 4}});
    check_static_shape(op, {Shape{2, 3, 4}, Shape{}}, {{2, 1, 4}, {2, 1, 4}, {2, 1, 4}});
}

TEST(StaticShapeInferenceTest, SplitV1_Dynamic) {
    check_output_shape(build_split(PartialShape({2, 8, 4}), {}, 4),
                       {ov::PartialShape::dynamic(ov::Rank(3)),
                        ov::PartialShape::dynamic(ov::Rank(3)),
                        ov::PartialShape::dynamic(ov::Rank(3)),
                        ov::PartialShape::dynamic(ov::Rank(3))});
}

TEST(StaticShapeInferenceTest, SplitV1_StaticNoConstMap) {
    check_static_shape(build_split(PartialShape({2, 8, 4}), {}, 4),
                       {Shape{2, 8, 4}, Shape{}},
                       {{2, 8, 1}, {2, 8, 1}, {2, 8, 1}, {2, 8, 1}},
                       true);
}

TEST(StaticShapeInferenceTest, SplitV1_StaticWrongConstMap1) {
    check_static_shape(build_split(PartialShape({2, 8, 4}), {}, 4),
                       {Shape{2, 8, 4}, {2, 3}},
                       {{2, 8, 1}, {2, 8, 1}, {2, 8, 1}, {2, 8, 1}},
                       true);
}

TEST(StaticShapeInferenceTest, SplitV1_StaticWrongConstMap2) {
    check_static_shape(build_split(PartialShape({2, 8, 4}), {}, 4),
                       {Shape{2, 8, 4}, -4},
                       {{2, 8, 1}, {2, 8, 1}, {2, 8, 1}, {2, 8, 1}},
                       true);
}

TEST(StaticShapeInferenceTest, SplitV1_StaticWithConstMap) {
    check_static_shape(build_split(PartialShape({2, 8, 4}), {}, 4),
                       {Shape{2, 8, 4}, 2},
                       {{2, 8, 1}, {2, 8, 1}, {2, 8, 1}, {2, 8, 1}});
}
