// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/variadic_split.hpp"
#include "utils.hpp"
#include "variadic_split_shape_inference.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

using VariadicSplitTestParams = std::tuple<StaticShapeVector,     // Input shapes
                                           int64_t,               // Split axis
                                           std::vector<int64_t>,  // split lengths
                                           StaticShapeVector      // Expected shapes
                                           >;

class VariadicSplitStaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v1::VariadicSplit>,
                                              public WithParamInterface<VariadicSplitTestParams> {
protected:
    void SetUp() override {
        std::tie(input_shapes, axis, split_lengths, exp_shapes) = GetParam();

        data = std::make_shared<op::v0::Parameter>(element::f32, input_shapes.front().get_shape());
    }

    int64_t axis;
    std::vector<int64_t> split_lengths;
    StaticShapeVector exp_shapes;

    std::shared_ptr<op::v0::Parameter> data;
};

INSTANTIATE_TEST_SUITE_P(
    1d_shapes,
    VariadicSplitStaticShapeInferenceTest,
    Values(make_tuple(StaticShapeVector{{0}, {}, {1}}, 0, std::vector<int64_t>{0}, StaticShapeVector{{0}}),
           make_tuple(StaticShapeVector{{15}, {}, {3}},
                      -1,
                      std::vector<int64_t>{2, 3, 10},
                      StaticShapeVector{{2}, {3}, {10}}),
           make_tuple(StaticShapeVector{{15}, {}, {3}},
                      0,
                      std::vector<int64_t>{5, -1, 2},
                      StaticShapeVector{{5}, {8}, {2}})),
    PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(multi_dim_shapes,
                         VariadicSplitStaticShapeInferenceTest,
                         Values(make_tuple(StaticShapeVector{{2, 6, 5}, {}, {3}},
                                           2,
                                           std::vector<int64_t>{2, 1, 2},
                                           StaticShapeVector{{2, 6, 2}, {2, 6, 1}, {2, 6, 2}}),
                                make_tuple(StaticShapeVector{{2, 6, 5}, {}, {2}},
                                           -2,
                                           std::vector<int64_t>{2, 4},
                                           StaticShapeVector{{2, 2, 5}, {2, 4, 5}}),
                                make_tuple(StaticShapeVector{{4, 6, 5}, {}, {3}},
                                           0,
                                           std::vector<int64_t>{-1, 3, 1},
                                           StaticShapeVector{{0, 6, 5}, {3, 6, 5}, {1, 6, 5}}),
                                make_tuple(StaticShapeVector{{4, 6, 5}, {}, {3}},
                                           0,
                                           std::vector<int64_t>{3, -1, 1},
                                           StaticShapeVector{{3, 6, 5}, {0, 6, 5}, {1, 6, 5}}),
                                make_tuple(StaticShapeVector{{4, 6, 5}, {}, {3}},
                                           0,
                                           std::vector<int64_t>{3, 1, -1},
                                           StaticShapeVector{{3, 6, 5}, {1, 6, 5}, {0, 6, 5}})),
                         PrintToStringParamName());

TEST_P(VariadicSplitStaticShapeInferenceTest, shape_inference_empty_const_map) {
    const auto axis_node = std::make_shared<op::v0::Constant>(element::i64, ov::Shape{}, axis);
    const auto split_len_node =
        std::make_shared<op::v0::Constant>(element::i64, ov::Shape{split_lengths.size()}, split_lengths);
    op = make_op(data, axis_node, split_len_node);

    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), split_lengths.size());
    EXPECT_EQ(output_shapes, exp_shapes);
}

TEST_P(VariadicSplitStaticShapeInferenceTest, shape_inference_axis_in_const_map) {
    const auto axis_node = std::make_shared<op::v0::Parameter>(element::i64, ov::PartialShape::dynamic());
    const auto split_len_node =
        std::make_shared<op::v0::Constant>(element::i64, ov::Shape{split_lengths.size()}, split_lengths);
    op = make_op(data, axis_node, split_len_node);

    const auto axis_tensor = ov::Tensor(element::i64, ov::Shape{}, &axis);
    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, axis_tensor}};

    output_shapes = shape_inference(op.get(), input_shapes, constant_data);

    EXPECT_EQ(output_shapes.size(), split_lengths.size());
    EXPECT_EQ(output_shapes, exp_shapes);
}

TEST_P(VariadicSplitStaticShapeInferenceTest, shape_inference_all_const_in_map) {
    const auto axis_node = std::make_shared<op::v0::Parameter>(element::i64, ov::PartialShape::dynamic());
    const auto split_len_node = std::make_shared<op::v0::Parameter>(element::i64, ov::PartialShape::dynamic());
    op = make_op(data, axis_node, split_len_node);

    const auto axis_tensor = ov::Tensor(element::i64, ov::Shape{}, &axis);
    const auto split_len_tensor = ov::Tensor(element::i64, ov::Shape{split_lengths.size()}, split_lengths.data());

    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{2, split_len_tensor}, {1, axis_tensor}};

    output_shapes = shape_inference(op.get(), input_shapes, constant_data);

    EXPECT_EQ(output_shapes.size(), split_lengths.size());
    EXPECT_EQ(output_shapes, exp_shapes);
}
