// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <topk_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

static std::shared_ptr<op::v3::TopK> build_topk(PartialShape data_shape = PartialShape::dynamic(),
                                                int64_t axis = 1,
                                                int k_value = -1) {
    std::shared_ptr<ov::Node> k;
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
    if (k_value >= 0)
        k = op::v0::Constant::create(element::i64, ov::Shape{}, {2});
    else
        k = std::make_shared<op::v0::Parameter>(element::i64, ov::PartialShape{});
    return std::make_shared<op::v3::TopK>(data, k, axis, "max", "value");
}

TEST(StaticShapeInferenceTest, TopKv3) {
    const auto topk = build_topk(PartialShape::dynamic(), 1, 2);

    check_static_shape(topk.get(),
                       {StaticShape{1, 10, 100}, StaticShape{}},
                       {StaticShape({1, 2, 100}), StaticShape({1, 2, 100})});
}

TEST(StaticShapeInferenceTest, TopKv3_StaticNoConstMap) {
    const auto topk = build_topk();

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 10, 100}, StaticShape{}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}, StaticShape{}};
    EXPECT_THROW(shape_inference(topk.get(), static_input_shapes, static_output_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, TopKv3_StaticWithConstMap) {
    const auto topk = build_topk();

    check_static_shape(topk.get(), {StaticShape{1, 10, 100}, 2}, {StaticShape{1, 2, 100}, StaticShape{1, 2, 100}});
}
