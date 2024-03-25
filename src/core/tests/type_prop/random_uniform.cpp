// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace testing;
using namespace ov;

TEST(type_prop, random_uniform_default_ctor) {
    auto out_shape = opset8::Constant::create(element::i64, Shape{4}, {2, 3, 4, 5});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0.f);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 1.f);

    auto r = make_shared<opset8::RandomUniform>();
    r->set_arguments(OutputVector{out_shape, min_val, max_val});
    r->set_out_type(element::f32);
    r->set_global_seed(121);
    r->set_op_seed(100);
    r->validate_and_infer_types();

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), (PartialShape{2, 3, 4, 5}));
}

TEST(type_prop, random_uniform_type_shape) {
    auto out_shape = opset8::Constant::create(element::i64, Shape{4}, {2, 3, 4, 5});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0.f);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 1.f);

    auto r = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f32, 120, 100);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), (PartialShape{2, 3, 4, 5}));
}

TEST(type_prop, random_uniform_param_input) {
    auto out_shape = make_shared<opset8::Parameter>(element::i32, PartialShape{3});
    auto min_val = make_shared<opset8::Constant>(element::i64, Shape{}, 5);
    auto max_val = make_shared<opset8::Constant>(element::i64, Shape{}, 10);

    auto r = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::i64, 100, 200);

    EXPECT_EQ(r->get_output_element_type(0), element::i64);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape::dynamic(3));
}

TEST(type_prop, random_uniform_dynamic_shape) {
    auto out_shape = make_shared<opset8::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto min_val = make_shared<opset8::Constant>(element::i64, Shape{}, 5);
    auto max_val = make_shared<opset8::Constant>(element::i64, Shape{}, 10);

    auto r = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::i64, 100, 200);

    EXPECT_EQ(r->get_output_element_type(0), element::i64);
    EXPECT_EQ(r->get_output_partial_shape(0), (PartialShape::dynamic()));
}

TEST(type_prop, random_uniform_dynamic_shape_with_labels) {
    auto shape = PartialShape{{0, 10}, 4, {3, 7}, -1};
    auto symbols = set_shape_symbols(shape);
    auto param = make_shared<opset8::Parameter>(element::i32, shape);
    auto out_shape = make_shared<opset8::ShapeOf>(param);

    auto min_val = make_shared<opset8::Constant>(element::i64, Shape{}, 5);
    auto max_val = make_shared<opset8::Constant>(element::i64, Shape{}, 10);

    auto r = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::i64, 100, 200);

    EXPECT_EQ(r->get_output_element_type(0), element::i64);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape({{0, 10}, 4, {3, 7}, -1}));
    EXPECT_THAT(get_shape_symbols(r->get_output_partial_shape(0)), symbols);
}

TEST(type_prop, random_uniform_dynamic_rank) {
    auto out_shape = make_shared<opset8::Parameter>(element::i32, PartialShape::dynamic());
    auto min_val = make_shared<opset8::Constant>(element::f64, Shape{}, 5);
    auto max_val = make_shared<opset8::Constant>(element::f64, Shape{}, 10);

    auto r = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f64, 100, 200);

    EXPECT_EQ(r->get_output_element_type(0), element::f64);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, random_uniform_invalid_out_shape_type) {
    auto out_shape = opset8::Constant::create(element::f64, Shape{4}, {2, 3, 4, 5});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0.f);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 1.f);

    OV_EXPECT_THROW(ignore = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f32, 120, 100),
                    NodeValidationFailure,
                    HasSubstr("Type of the input should be int32 or int64."));
}

TEST(type_prop, random_uniform_invalid_out_shape_rank) {
    auto out_shape = make_shared<opset8::Parameter>(element::i32, Shape{3, 2});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0.f);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 1.f);

    OV_EXPECT_THROW(ignore = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f32, 120, 100),
                    NodeValidationFailure,
                    HasSubstr("The rank of the tensor defining output shape must be equal to 1."));
}

TEST(type_prop, random_uniform_invalid_min_val) {
    auto out_shape = opset8::Constant::create(element::i32, Shape{4}, {2, 3, 4, 5});
    auto min_val = opset8::Constant::create(element::f32, Shape{2}, {2, 3});
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 1.f);

    OV_EXPECT_THROW(ignore = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f32, 120, 100),
                    NodeValidationFailure,
                    HasSubstr("Min value must be a scalar or one element 1D tensor."));
}

TEST(type_prop, random_uniform_invalid_max_val) {
    auto out_shape = opset8::Constant::create(element::i32, Shape{4}, {2, 3, 4, 5});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0.f);
    auto max_val = opset8::Constant::create(element::f32, Shape{3}, {2, 3, 5});

    OV_EXPECT_THROW(ignore = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f32, 120, 100),
                    NodeValidationFailure,
                    HasSubstr("Max value must be a scalar or one element 1D tensor."));
}

TEST(type_prop, random_uniform_invalid_min_max_val_type_case1) {
    auto out_shape = opset8::Constant::create(element::i64, Shape{4}, {2, 3, 4, 5});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0.f);
    auto max_val = make_shared<opset8::Constant>(element::i32, Shape{}, 100);

    OV_EXPECT_THROW(ignore = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f32, 120, 100),
                    NodeValidationFailure,
                    HasSubstr("'min_val' should have the same type as 'max_val'."));
}

TEST(type_prop, random_uniform_invalid_min_max_val_type_case2) {
    auto out_shape = opset8::Constant::create(element::i64, Shape{4}, {2, 3, 4, 5});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0.f);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 1.f);

    OV_EXPECT_THROW(ignore = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::i32, 120, 100),
                    NodeValidationFailure,
                    HasSubstr("'min_val' and 'max_val' should have the same type as 'out_type' attribute."));
}

TEST(type_prop, random_uniform_invalid_min_max_values_case1) {
    auto out_shape = opset8::Constant::create(element::i64, Shape{4}, {2, 3, 4, 5});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 1.f);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0.f);

    OV_EXPECT_THROW(ignore = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f32, 120, 100),
                    NodeValidationFailure,
                    HasSubstr("Min value must be less than max value."));
}

TEST(type_prop, random_uniform_invalid_min_max_values_case2) {
    auto out_shape = opset8::Constant::create(element::i64, Shape{4}, {2, 3, 4, 5});
    auto min_val = make_shared<opset8::Constant>(element::i32, Shape{}, 100);
    auto max_val = make_shared<opset8::Constant>(element::i32, Shape{}, 100);

    OV_EXPECT_THROW(ignore = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::i32, 120, 100),
                    NodeValidationFailure,
                    HasSubstr("Min value must be less than max value."));
}

TEST(type_prop, random_uniform_min_max_1d_tensors) {
    auto out_shape = opset8::Constant::create(element::i64, Shape{4}, {2, 3, 4, 5});
    auto min_val = opset8::Constant::create(element::f32, Shape{1}, {-1.0});
    auto max_val = opset8::Constant::create(element::f32, Shape{1}, {2.0});

    auto r = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f32, 120, 100);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), (PartialShape{2, 3, 4, 5}));
}
