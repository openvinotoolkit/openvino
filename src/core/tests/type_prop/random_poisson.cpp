// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/random_poisson.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace testing;
using namespace ov;

TEST(type_prop, random_poisson_f32) {
    auto input = make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 4});
    auto r = make_shared<op::v17::RandomPoisson>(input, 120, 100, op::PhiloxAlignment::PYTORCH);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), (PartialShape{2, 3, 4}));
}

TEST(type_prop, random_poisson_f64) {
    auto input = make_shared<op::v0::Parameter>(element::f64, Shape{5, 10});
    auto r = make_shared<op::v17::RandomPoisson>(input, 0, 0, op::PhiloxAlignment::PYTORCH);

    EXPECT_EQ(r->get_output_element_type(0), element::f64);
    EXPECT_EQ(r->get_output_partial_shape(0), (PartialShape{5, 10}));
}

TEST(type_prop, random_poisson_f16) {
    auto input = make_shared<op::v0::Parameter>(element::f16, Shape{7, 7, 7});
    auto r = make_shared<op::v17::RandomPoisson>(input, 42, 7, op::PhiloxAlignment::TENSORFLOW);

    EXPECT_EQ(r->get_output_element_type(0), element::f16);
    EXPECT_EQ(r->get_output_partial_shape(0), (PartialShape{7, 7, 7}));
}

TEST(type_prop, random_poisson_default_ctor) {
    auto input = make_shared<op::v0::Parameter>(element::f32, Shape{3, 4});

    auto r = make_shared<op::v17::RandomPoisson>();
    r->set_arguments(OutputVector{input});
    r->set_global_seed(121);
    r->set_op_seed(100);
    r->set_alignment(op::PhiloxAlignment::PYTORCH);
    r->validate_and_infer_types();

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), (PartialShape{3, 4}));
    EXPECT_EQ(r->get_global_seed(), 121);
    EXPECT_EQ(r->get_op_seed(), 100);
    EXPECT_EQ(r->get_alignment(), op::PhiloxAlignment::PYTORCH);
}

TEST(type_prop, random_poisson_dynamic_shape) {
    auto input =
        make_shared<op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 3, Dimension::dynamic()});
    auto r = make_shared<op::v17::RandomPoisson>(input, 0, 0, op::PhiloxAlignment::PYTORCH);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 3, Dimension::dynamic()}));
}

TEST(type_prop, random_poisson_dynamic_rank) {
    auto input = make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());
    auto r = make_shared<op::v17::RandomPoisson>(input, 10, 20, op::PhiloxAlignment::PYTORCH);

    EXPECT_EQ(r->get_output_element_type(0), element::f64);
    EXPECT_TRUE(r->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, random_poisson_dynamic_type) {
    auto input = make_shared<op::v0::Parameter>(element::dynamic, PartialShape{2, 3});
    auto r = make_shared<op::v17::RandomPoisson>(input, 0, 0, op::PhiloxAlignment::PYTORCH);

    EXPECT_EQ(r->get_output_element_type(0), element::dynamic);
    EXPECT_EQ(r->get_output_partial_shape(0), (PartialShape{2, 3}));
}

TEST(type_prop, random_poisson_invalid_integer_input) {
    auto input = make_shared<op::v0::Parameter>(element::i32, Shape{3, 4});

    OV_EXPECT_THROW(ignore = make_shared<op::v17::RandomPoisson>(input, 0, 0, op::PhiloxAlignment::PYTORCH),
                    NodeValidationFailure,
                    HasSubstr("Input tensor must be of type float"));
}

TEST(type_prop, random_poisson_invalid_scalar_input) {
    auto input = make_shared<op::v0::Parameter>(element::f32, Shape{});

    OV_EXPECT_THROW(ignore = make_shared<op::v17::RandomPoisson>(input, 0, 0, op::PhiloxAlignment::PYTORCH),
                    NodeValidationFailure,
                    HasSubstr("scalars (rank 0) are not supported"));
}
