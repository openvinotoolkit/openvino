// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/one_hot.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"

using namespace std;
using namespace ov;
using namespace testing;

TEST(type_prop, one_hot_v1_output_shape) {
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto depth = ov::op::v0::Constant::create(element::i64, Shape{}, {2});
    auto on_value = ov::op::v0::Constant::create(element::u32, Shape{}, {5});
    auto off_value = ov::op::v0::Constant::create(element::u32, Shape{}, {10});
    int64_t axis = -1;
    auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
    ASSERT_EQ(ont_hot->get_element_type(), element::u32);
    ASSERT_EQ(ont_hot->get_shape(), (Shape{3, 2}));

    auto dyn_indices = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{{1, 3}});
    auto dyn_ont_hot = make_shared<op::v1::OneHot>(dyn_indices, depth, on_value, off_value, axis);
    ASSERT_EQ(dyn_ont_hot->get_output_element_type(0), element::u32);
    ASSERT_EQ(dyn_ont_hot->get_output_partial_shape(0), (PartialShape{{1, 3}, 2}));
}

TEST(type_prop, one_hot_v1_output_shape_2) {
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1, 3, 2, 3});
    auto depth = ov::op::v0::Constant::create(element::i64, Shape{}, {4});
    auto on_value = ov::op::v0::Constant::create(element::f32, Shape{}, {1.0f});
    auto off_value = ov::op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    int64_t axis = 3;
    auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
    ASSERT_EQ(ont_hot->get_element_type(), element::f32);
    ASSERT_EQ(ont_hot->get_shape(), (Shape{1, 3, 2, 4, 3}));

    auto dyn_indices = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{1, {3, 5}, 2, 3});
    auto dyn_ont_hot = make_shared<op::v1::OneHot>(dyn_indices, depth, on_value, off_value, axis);
    ASSERT_EQ(dyn_ont_hot->get_output_element_type(0), element::f32);
    ASSERT_EQ(dyn_ont_hot->get_output_partial_shape(0), (PartialShape{1, {3, 5}, 2, 4, 3}));
}

TEST(type_prop, one_hot_v1_indices_symbols) {
    auto ind_shape = PartialShape{-1, {3, 5}, 2, 3};
    auto symbols = set_shape_symbols(ind_shape);

    auto dyn_indices = make_shared<ov::op::v0::Parameter>(element::i64, ind_shape);
    auto depth = ov::op::v0::Constant::create(element::i64, Shape{}, {4});
    auto on_value = ov::op::v0::Constant::create(element::f32, Shape{}, {1.0f});
    auto off_value = ov::op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    int64_t axis = 1;

    PartialShape expected_shape{-1, 4, {3, 5}, 2, 3};
    ov::TensorSymbol expected_symbols = {symbols[0], nullptr, symbols[1], symbols[2], symbols[3]};

    auto dyn_one_hot = make_shared<op::v1::OneHot>(dyn_indices, depth, on_value, off_value, axis);
    const auto& out_shape = dyn_one_hot->get_output_partial_shape(0);

    EXPECT_EQ(dyn_one_hot->get_output_element_type(0), element::f32);
    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, one_hot_v1_depth_shape_of_value) {
    auto ind_shape = PartialShape{-1, {3, 5}, 2, 3};
    set_shape_symbols(ind_shape);

    auto dyn_indices = make_shared<ov::op::v0::Parameter>(element::i64, ind_shape);

    PartialShape shape_for_depth = PartialShape{4};

    auto data = make_shared<ov::op::v0::Parameter>(element::i8, shape_for_depth);
    auto depth_dim = make_shared<op::v3::ShapeOf>(data);

    auto depth = make_shared<op::v0::Squeeze>(depth_dim);
    auto on_value = ov::op::v0::Constant::create(element::f32, Shape{}, {1.0f});
    auto off_value = ov::op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    int64_t axis = 1;

    PartialShape expected_shape{-1, 4, {3, 5}, 2, 3};

    auto dyn_one_hot = make_shared<op::v1::OneHot>(dyn_indices, depth, on_value, off_value, axis);
    const auto& out_shape = dyn_one_hot->get_output_partial_shape(0);

    EXPECT_EQ(dyn_one_hot->get_output_element_type(0), element::f32);
    EXPECT_EQ(out_shape, expected_shape);
}

TEST(type_prop, one_hot_v1_depth_value_symbol) {
    auto ind_shape = PartialShape{-1, {3, 5}, 2, 3};
    auto symbols = set_shape_symbols(ind_shape);

    auto dyn_indices = make_shared<ov::op::v0::Parameter>(element::i64, ind_shape);

    auto symboled_dim = Dimension(4, 6);
    auto depth_symbol = std::make_shared<Symbol>();
    symboled_dim.set_symbol(depth_symbol);
    PartialShape shape_for_depth = PartialShape{symboled_dim};

    auto data = make_shared<ov::op::v0::Parameter>(element::i8, shape_for_depth);
    auto depth_dim = make_shared<op::v3::ShapeOf>(data);

    auto depth = make_shared<op::v0::Squeeze>(depth_dim);
    auto on_value = ov::op::v0::Constant::create(element::f32, Shape{}, {1.0f});
    auto off_value = ov::op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    int64_t axis = 1;

    PartialShape expected_shape{-1, {4, 6}, {3, 5}, 2, 3};
    ov::TensorSymbol expected_symbols{symbols[0], depth_symbol, symbols[1], symbols[2], symbols[3]};

    auto dyn_one_hot = make_shared<op::v1::OneHot>(dyn_indices, depth, on_value, off_value, axis);
    const auto& out_shape = dyn_one_hot->get_output_partial_shape(0);

    EXPECT_EQ(dyn_one_hot->get_output_element_type(0), element::f32);
    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, one_hot_v1_output_symbols) {
    auto ind_shape = PartialShape{-1, {3, 5}, 2, 3};
    auto symbols = set_shape_symbols(ind_shape);

    auto dyn_indices = make_shared<ov::op::v0::Parameter>(element::i64, ind_shape);
    auto depth = ov::op::v0::Constant::create(element::i64, Shape{}, {4});
    auto on_value = ov::op::v0::Constant::create(element::f32, Shape{}, {1.0f});
    auto off_value = ov::op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    int64_t axis = 1;

    PartialShape expected_shape{-1, 4, {3, 5}, 2, 3};
    ov::TensorSymbol expected_symbols{symbols[0], nullptr, symbols[1], symbols[2], symbols[3]};

    auto dyn_one_hot = make_shared<op::v1::OneHot>(dyn_indices, depth, on_value, off_value, axis);
    const auto& out_shape = dyn_one_hot->get_output_partial_shape(0);

    EXPECT_EQ(dyn_one_hot->get_output_element_type(0), element::f32);
    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, one_hot_v1_default_constructor) {
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1, 3, 2, 3});
    auto depth = ov::op::v0::Constant::create(element::i64, Shape{}, {4});
    auto on_value = ov::op::v0::Constant::create(element::f32, Shape{}, {1.0f});
    auto off_value = ov::op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    int64_t axis = 3;
    auto ont_hot = make_shared<op::v1::OneHot>();

    ont_hot->set_argument(0, indices);
    ont_hot->set_argument(1, depth);
    ont_hot->set_argument(2, on_value);
    ont_hot->set_argument(3, off_value);

    ont_hot->set_axis(axis);
    EXPECT_EQ(ont_hot->get_axis(), axis);

    ont_hot->validate_and_infer_types();

    EXPECT_EQ(ont_hot->get_element_type(), element::f32);
    EXPECT_EQ(ont_hot->get_shape(), (Shape{1, 3, 2, 4, 3}));
}

TEST(type_prop, one_hot_v1_indices_elem_not_integral) {
    auto indices = make_shared<ov::op::v0::Parameter>(element::f16, Shape{2, 2});
    auto depth = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto on_value = make_shared<ov::op::v0::Parameter>(element::u32, Shape{});
    auto off_value = make_shared<ov::op::v0::Parameter>(element::u32, Shape{});
    int64_t axis = -1;
    try {
        auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices element type not detected";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Indices must be integral element type."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_v1_depth_elem_not_integral) {
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{2, 2});
    auto depth = make_shared<ov::op::v0::Parameter>(element::f16, Shape{});
    auto on_value = make_shared<ov::op::v0::Parameter>(element::u32, Shape{});
    auto off_value = make_shared<ov::op::v0::Parameter>(element::u32, Shape{});
    int64_t axis = -1;
    try {
        auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect depth element type not detected";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Depth must be integral element type."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_v1_negative_depth) {
    auto indices = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 2});
    auto depth = ov::op::v0::Constant::create(element::i64, Shape{}, {-4});
    auto on_value = ov::op::v0::Constant::create(element::f32, Shape{}, {1.0f});
    auto off_value = ov::op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    int64_t axis = -1;

    OV_EXPECT_THROW(auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis),
                    ov::Exception,
                    HasSubstr("can't be negative."));
}

TEST(type_prop, one_hot_v1_on_off_values_not_compatible) {
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{2, 2});
    auto depth = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto on_value = make_shared<ov::op::v0::Parameter>(element::bf16, Shape{});
    auto off_value = make_shared<ov::op::v0::Parameter>(element::f16, Shape{});
    int64_t axis = -1;
    try {
        auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible on/off element types not detected";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("on_value element type must be compatible with off_value element type."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_v1_depth_not_scalar) {
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{2, 2});
    auto depth = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1});
    auto on_value = make_shared<ov::op::v0::Parameter>(element::bf16, Shape{});
    auto off_value = make_shared<ov::op::v0::Parameter>(element::bf16, Shape{});
    int64_t axis = -1;
    try {
        auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Not scalar depth input not detected.";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("depth input must be scalar."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_v1_on_value_not_scalar) {
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{2, 2});
    auto depth = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto on_value = make_shared<ov::op::v0::Parameter>(element::bf16, Shape{2});
    auto off_value = make_shared<ov::op::v0::Parameter>(element::bf16, Shape{});
    int64_t axis = -1;
    try {
        auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Not scalar on_value input not detected.";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("on_value input must be scalar."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_v1_off_value_not_scalar) {
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{2, 2});
    auto depth = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto on_value = make_shared<ov::op::v0::Parameter>(element::bf16, Shape{});
    auto off_value = make_shared<ov::op::v0::Parameter>(element::bf16, Shape{3});
    int64_t axis = -1;
    try {
        auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Not scalar off_value input not detected.";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("off_value input must be scalar."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_v1_out_types_1) {
    auto indices = make_shared<ov::op::v0::Parameter>(element::i32, Shape{3, 2});
    auto depth = ov::op::v0::Constant::create(element::i32, Shape{}, {2});
    int64_t axis = -1;
    auto on_value = ov::op::v0::Constant::create(element::f32, Shape{}, {-3.3});
    auto off_value = ov::op::v0::Constant::create(element::f32, Shape{}, {-10.12});
    auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
    ASSERT_EQ(ont_hot->get_element_type(), element::f32);
}

TEST(type_prop, one_hot_v1_out_types_2) {
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 2});
    auto depth = ov::op::v0::Constant::create(element::i32, Shape{}, {2});
    int64_t axis = -1;
    auto on_value = ov::op::v0::Constant::create(element::i32, Shape{}, {-1});
    auto off_value = ov::op::v0::Constant::create(element::i32, Shape{}, {7});
    auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
    ASSERT_EQ(ont_hot->get_element_type(), element::i32);
}

TEST(type_prop, one_hot_v1_out_types_3) {
    auto indices = make_shared<ov::op::v0::Parameter>(element::i32, Shape{3, 2});
    auto depth = ov::op::v0::Constant::create(element::i32, Shape{}, {2});
    int64_t axis = -1;
    auto on_value = ov::op::v0::Constant::create(element::boolean, Shape{}, {true});
    auto off_value = ov::op::v0::Constant::create(element::boolean, Shape{}, {false});
    auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
    ASSERT_EQ(ont_hot->get_element_type(), element::boolean);
}
