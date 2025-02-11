// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/split.hpp"
#include "split_shape_inference.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

using SplitTestParams = std::tuple<StaticShapeVector,  // Input shapes
                                   int64_t,            // Split axis
                                   size_t,             // Number of splits
                                   StaticShape         // Expected output(s) shape
                                   >;

class SplitStaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v1::Split>,
                                      public WithParamInterface<SplitTestParams> {
protected:
    void SetUp() override {
        std::tie(input_shapes, axis, num_of_splits, exp_shape) = GetParam();

        output_shapes = StaticShapeVector();
        arg = std::make_shared<op::v0::Parameter>(element::f32, input_shapes.front().get_shape());
    }

    int64_t axis;
    size_t num_of_splits;
    std::shared_ptr<op::v0::Parameter> arg;
};

INSTANTIATE_TEST_SUITE_P(1d_shapes,
                         SplitStaticShapeInferenceTest,
                         Values(make_tuple(StaticShapeVector{{0}, {}}, 0, 1, StaticShape({0})),
                                make_tuple(StaticShapeVector{{1}, {}}, 0, 1, StaticShape({1})),
                                make_tuple(StaticShapeVector{{2}, {}}, -1, 1, StaticShape({2})),
                                make_tuple(StaticShapeVector{{2}, {}}, 0, 2, StaticShape({1})),
                                make_tuple(StaticShapeVector{{4}, {}}, -1, 2, StaticShape({2})),
                                make_tuple(StaticShapeVector{{9}, {}}, 0, 3, StaticShape({3})),
                                make_tuple(StaticShapeVector{{15}, {}}, -1, 5, StaticShape({3}))),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(multi_dim_shapes,
                         SplitStaticShapeInferenceTest,
                         Values(make_tuple(StaticShapeVector{{6, 12, 21}, {}}, 0, 6, StaticShape({1, 12, 21})),
                                make_tuple(StaticShapeVector{{6, 12, 21}, {}}, -1, 3, StaticShape({6, 12, 7})),
                                make_tuple(StaticShapeVector{{6, 12, 21}, {}}, -2, 2, StaticShape({6, 6, 21})),
                                make_tuple(StaticShapeVector{{6, 12, 21}, {}}, 2, 7, StaticShape({6, 12, 3})),
                                make_tuple(StaticShapeVector{{6, 12, 1, 14}, {}}, -1, 7, StaticShape({6, 12, 1, 2}))),
                         PrintToStringParamName());

TEST_P(SplitStaticShapeInferenceTest, shape_inference_empty_const_map) {
    const auto axis_node = std::make_shared<op::v0::Constant>(element::i64, ov::Shape{}, axis);
    op = make_op(arg, axis_node, num_of_splits);

    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), num_of_splits);
    EXPECT_THAT(output_shapes, Each(exp_shape));
}

TEST_P(SplitStaticShapeInferenceTest, shape_inference_with_const_map) {
    const auto axis_node = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{});
    op = make_op(arg, axis_node, num_of_splits);

    const auto axis_tensor = ov::Tensor(element::i64, ov::Shape{}, &axis);
    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, axis_tensor}};

    output_shapes = shape_inference(op.get(), input_shapes, constant_data);

    ASSERT_EQ(output_shapes.front(), exp_shape);
}
