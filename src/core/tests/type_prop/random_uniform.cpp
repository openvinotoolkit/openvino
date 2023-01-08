// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset8.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, random_uniform_type_shape) {
    auto out_shape = opset8::Constant::create(element::i64, Shape{4}, {2, 3, 4, 5});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0.f);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 1.f);

    auto r = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f32, 120, 100);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 3, 4, 5}));
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
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
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

    try {
        auto r = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f32, 120, 100);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid output shape.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Type of the input should be int32 or int64."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason.";
    }
}

TEST(type_prop, random_uniform_invalid_out_shape_rank) {
    auto out_shape = make_shared<opset8::Parameter>(element::i32, Shape{3, 2});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0.f);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 1.f);
    try {
        auto r = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f32, 120, 100);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid output shape.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("The rank of the tensor defining output shape must be equal to 1."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason.";
    }
}

TEST(type_prop, random_uniform_invalid_min_val) {
    auto out_shape = opset8::Constant::create(element::i32, Shape{4}, {2, 3, 4, 5});
    auto min_val = opset8::Constant::create(element::f32, Shape{2}, {2, 3});
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 1.f);

    try {
        auto r = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f32, 120, 100);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid min value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'min_val' should have 1 element."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason.";
    }
}

TEST(type_prop, random_uniform_invalid_max_val) {
    auto out_shape = opset8::Constant::create(element::i32, Shape{4}, {2, 3, 4, 5});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0.f);
    auto max_val = opset8::Constant::create(element::f32, Shape{3}, {2, 3, 5});

    try {
        auto r = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f32, 120, 100);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid max value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'max_val' should have 1 element."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason.";
    }
}

TEST(type_prop, random_uniform_invalid_min_max_val_type_case1) {
    auto out_shape = opset8::Constant::create(element::i64, Shape{4}, {2, 3, 4, 5});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0.f);
    auto max_val = make_shared<opset8::Constant>(element::i32, Shape{}, 100);

    try {
        auto r = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f32, 120, 100);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid min value type.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'min_val' should have the same type as 'max_val'."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, random_uniform_invalid_min_max_val_type_case2) {
    auto out_shape = opset8::Constant::create(element::i64, Shape{4}, {2, 3, 4, 5});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0.f);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 1.f);

    try {
        auto r = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::i32, 120, 100);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid min and max value type.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("'min_val' and 'max_val' should have the same type as 'out_type' attribute."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, random_uniform_invalid_min_max_values_case1) {
    auto out_shape = opset8::Constant::create(element::i64, Shape{4}, {2, 3, 4, 5});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 1.f);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0.f);

    try {
        auto r = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f32, 120, 100);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid min and max values.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Min value must be less than max value."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, random_uniform_invalid_min_max_values_case2) {
    auto out_shape = opset8::Constant::create(element::i64, Shape{4}, {2, 3, 4, 5});
    auto min_val = make_shared<opset8::Constant>(element::i32, Shape{}, 100);
    auto max_val = make_shared<opset8::Constant>(element::i32, Shape{}, 100);

    try {
        auto r = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::i32, 120, 100);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid min and max values.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Min value must be less than max value."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, random_uniform_min_max_1d_tensors) {
    auto out_shape = opset8::Constant::create(element::i64, Shape{4}, {2, 3, 4, 5});
    auto min_val = opset8::Constant::create(element::f32, Shape{1}, {-1.0});
    auto max_val = opset8::Constant::create(element::f32, Shape{1}, {2.0});

    auto r = make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::f32, 120, 100);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 3, 4, 5}));
}
