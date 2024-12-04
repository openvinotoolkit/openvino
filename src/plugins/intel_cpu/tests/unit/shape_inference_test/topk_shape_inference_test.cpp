// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/opsets/opset10.hpp"
#include "topk_shape_inference.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace ov::opset10;
using namespace testing;

namespace topk_test {
using TopKTestParams = std::tuple<StaticShapeVector,  // Input shapes
                                  int64_t,            // axis
                                  int64_t,            // k value
                                  StaticShape         // Expected output shape
                                  >;
template <class TOp>
class TopKTest : public OpStaticShapeInferenceTest<TOp>, public WithParamInterface<TopKTestParams> {
protected:
    void SetUp() override {
        std::tie(this->input_shapes, this->axis, this->k, this->exp_shape) = GetParam();
        this->output_shapes.resize(2);
    }
    int64_t axis, k;
};

const auto TopkTestValues = Values(make_tuple(StaticShapeVector{{0}, {}}, 0, 1, StaticShape{1}),
                                   make_tuple(StaticShapeVector{{5, 2, 10, 0}, {}}, -1, 5, StaticShape{5, 2, 10, 5}),
                                   make_tuple(StaticShapeVector{{3, 5, 6}, {}}, 1, 2, StaticShape{3, 2, 6}));

namespace v1 {
using TopKV1AssertStaticShapeInferenceTest = OpStaticShapeInferenceTest<op::v1::TopK>;

TEST_F(TopKV1AssertStaticShapeInferenceTest, k_is_negative) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto k_node = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());

    const auto op = make_op(data, k_node, 0, "max", "value");

    input_shapes = StaticShapeVector{{5, 2}, {}};
    output_shapes = StaticShapeVector(2);

    int64_t k = -2;
    const auto const_map = std::unordered_map<size_t, ov::Tensor>{{1, {element::i64, ov::Shape{}, &k}}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, const_map),
                    ov::AssertFailure,
                    HasSubstr("The value of 'K' must be greater or equal to zero. (got " + std::to_string(k) + ")"));
}

using TopKV1Test = TopKTest<op::v1::TopK>;
INSTANTIATE_TEST_SUITE_P(StaticShapeInference, TopKV1Test, TopkTestValues, PrintToStringParamName());

TEST_P(TopKV1Test, no_constant_map) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto k_node = Constant::create(element::i64, ov::Shape{}, {k});

    const auto op = make_op(data, k_node, axis, "max", "value");

    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 2);
    EXPECT_THAT(output_shapes, Each(exp_shape));
}

TEST_P(TopKV1Test, k_as_param_no_const_map) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto k_node = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());

    const auto op = make_op(data, k_node, axis, "min", "value");

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Static shape inference lacks constant data on port 1"));
}

TEST_P(TopKV1Test, k_as_param_in_const_map) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto k_node = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());

    const auto const_map = std::unordered_map<size_t, ov::Tensor>{{1, {element::i64, ov::Shape{}, &k}}};

    const auto op = make_op(data, k_node, axis, "min", "value");

    output_shapes = shape_inference(op.get(), input_shapes, const_map);

    EXPECT_EQ(output_shapes.size(), 2);
    EXPECT_THAT(output_shapes, Each(exp_shape));
}
}  // namespace v1

namespace v3 {
using TopKV3AssertStaticShapeInferenceTest = OpStaticShapeInferenceTest<op::v3::TopK>;

TEST_F(TopKV3AssertStaticShapeInferenceTest, k_is_negative) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto k_node = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());

    const auto op = make_op(data, k_node, 0, "max", "value");

    input_shapes = StaticShapeVector{{5, 2}, {}};
    output_shapes = StaticShapeVector(2);

    int64_t k = -2;
    const auto const_map = std::unordered_map<size_t, ov::Tensor>{{1, {element::i64, ov::Shape{}, &k}}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, const_map),
                    ov::AssertFailure,
                    HasSubstr("The value of 'K' must be greater or equal to zero. (got " + std::to_string(k) + ")"));
}

using TopKV3Test = TopKTest<op::v3::TopK>;
INSTANTIATE_TEST_SUITE_P(StaticShapeInference, TopKV3Test, TopkTestValues, PrintToStringParamName());

TEST_P(TopKV3Test, k_as_constant) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto k_node = Constant::create(element::i64, ov::Shape{}, {k});

    const auto op = make_op(data, k_node, axis, "min", "value");

    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 2);
    EXPECT_THAT(output_shapes, Each(exp_shape));
}

TEST_P(TopKV3Test, k_as_param_no_const_map) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto k_node = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());

    const auto op = make_op(data, k_node, axis, "min", "value");

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Static shape inference lacks constant data on port 1"));
}

TEST_P(TopKV3Test, k_as_param_in_const_map) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto k_node = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());

    const auto const_map = std::unordered_map<size_t, ov::Tensor>{{1, {element::i64, ov::Shape{}, &k}}};

    const auto op = make_op(data, k_node, axis, "max", "value");

    output_shapes = shape_inference(op.get(), input_shapes, const_map);

    EXPECT_EQ(output_shapes.size(), 2);
    EXPECT_THAT(output_shapes, Each(exp_shape));
}
}  // namespace v3
}  // namespace topk_test
