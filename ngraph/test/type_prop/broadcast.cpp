// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "util/type_prop.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

// Because v3::Broadcast is backward compatible to v1::Broadcast all v1::Broadcast tests should pass
template <typename T>
class BroadcastTests : public ::testing::Test
{
};
TYPED_TEST_SUITE_P(BroadcastTests);

TYPED_TEST_P(BroadcastTests, broadcast_numpy)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1});
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 6});

    auto bc = make_shared<TypeParam>(param, target_shape);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{2, 3, 6}));
}

TYPED_TEST_P(BroadcastTests, broadcast_axes_mapping)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1});
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 1});
    auto axes_mapping = op::Constant::create<int64_t>(element::i64, Shape{2}, {1, 2});

    auto bc = make_shared<TypeParam>(param, target_shape, axes_mapping);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{2, 3, 1}));
}

TYPED_TEST_P(BroadcastTests, broadcast_target_shape_as_concat_with_constants)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{16});
    auto target_shape_constant_1 = op::Constant::create<int64_t>(element::i64, Shape{1}, {1});
    auto target_shape_constant_2 = op::Constant::create<int64_t>(element::i64, Shape{1}, {16});
    auto target_shape_constant_3 = op::Constant::create<int64_t>(element::i64, Shape{1}, {50});
    auto target_shape_constant_4 = op::Constant::create<int64_t>(element::i64, Shape{1}, {50});
    std::int64_t axis = 0;
    std::vector<std::shared_ptr<Node>> args{target_shape_constant_1,
                                            target_shape_constant_2,
                                            target_shape_constant_3,
                                            target_shape_constant_4};
    auto target_shape = make_shared<op::Concat>(args, axis);
    auto axes_mapping = op::Constant::create<int64_t>(element::i64, Shape{1}, {1});
    auto bc = make_shared<TypeParam>(param, target_shape, axes_mapping, "NONE");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_static());
    ASSERT_TRUE(bc->get_output_partial_shape(0).same_scheme(PartialShape{1, 16, 50, 50}));
}

TYPED_TEST_P(BroadcastTests, broadcast_target_shape_as_concat_with_node)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{16});
    auto target_shape_constant_1 = make_shared<op::Parameter>(element::i64, Shape{1});
    auto target_shape_constant_2 = op::Constant::create<int64_t>(element::i64, Shape{1}, {16});
    auto target_shape_constant_3 = op::Constant::create<int64_t>(element::i64, Shape{1}, {50});
    auto target_shape_constant_4 = op::Constant::create<int64_t>(element::i64, Shape{1}, {50});
    std::int64_t axis = 0;
    std::vector<std::shared_ptr<Node>> args{target_shape_constant_1,
                                            target_shape_constant_2,
                                            target_shape_constant_3,
                                            target_shape_constant_4};
    auto target_shape = make_shared<op::Concat>(args, axis);
    auto axes_mapping = op::Constant::create<int64_t>(element::i64, Shape{1}, {1});
    auto bc = make_shared<TypeParam>(param, target_shape, axes_mapping, "NONE");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_dynamic());
    ASSERT_EQ(bc->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 16, 50, 50}));
}

TYPED_TEST_P(BroadcastTests, broadcast_fail_rank)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1});
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 1});
    auto axes_mapping = op::Constant::create<int64_t>(element::i64, Shape{3}, {1, 2, 3});

    try
    {
        auto bc = make_shared<TypeParam>(param, target_shape, axes_mapping);
        FAIL() << "Broadcast: target shape mismatch with input rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Broadcast axes_mapping shape Shape{3} doesn't match rank of input tensor 2");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(BroadcastTests, broadcast_fail_transpose)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1});
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 1, 3});
    auto axes_mapping = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 1});

    try
    {
        auto bc = make_shared<TypeParam>(param, target_shape, axes_mapping);
        FAIL() << "Broadcast: transpose prohibition not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Broadcast doesn't permit transposes. axes_mapping AxisVector{2, 1} "
                             "not in sorted order");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(BroadcastTests, broadcast_fail_axes_map)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1});
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 1});
    auto axes_mapping = op::Constant::create<int64_t>(element::i64, Shape{2}, {1, 3});

    try
    {
        auto bc = make_shared<TypeParam>(param, target_shape, axes_mapping);
        FAIL() << "Broadcast: wrong axes_map not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Broadcast axes_mapping[1]: 3 exceeds target rank 3");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(BroadcastTests, broadcast_fail_axes_map_shape)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 2});
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 3});
    auto axes_mapping = op::Constant::create<int64_t>(element::i64, Shape{2}, {1, 2});

    try
    {
        auto bc = make_shared<TypeParam>(param, target_shape, axes_mapping);
        FAIL() << "Broadcast: wrong target shape not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Broadcast target[axes_mapping[1]] Expected 2. Got 3");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(BroadcastTests, broadcast_axes_wrong_rank)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc_shape = make_shared<op::Parameter>(element::i64, Shape{1});
    auto bc_axes = make_shared<op::Parameter>(element::i64, Shape{2, 2});

    try
    {
        auto bc = make_shared<TypeParam>(arg, bc_shape, bc_axes);
        FAIL() << "Broadcast: axes shape rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Broadcast axes rank must be 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(BroadcastTests, broadcast_fully_dynamic_target_shape)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto bc_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    auto bc = make_shared<TypeParam>(arg, bc_shape, bc_axes);
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_dynamic());

    bc_shape = make_shared<op::Parameter>(element::i64, Shape{1});
    bc = make_shared<TypeParam>(arg, bc_shape, bc_axes);
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_dynamic_values_of_target_shape)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{2});
    const auto target = make_shared<op::Parameter>(element::i32, PartialShape::dynamic(4));
    const auto target_shape = std::make_shared<ngraph::opset6::ShapeOf>(target);
    const auto axes_mapping = op::Constant::create(element::i64, Shape{1}, {1});

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping);
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_dynamic());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TYPED_TEST_P(BroadcastTests, broadcast_broadcast_shape_et_wrong)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    // wrong element type
    auto bc_shape = make_shared<op::Parameter>(element::boolean, Shape{1});
    auto bc_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    try
    {
        auto bc = make_shared<TypeParam>(arg, bc_shape, bc_axes);
        FAIL() << "Broadcast: did not detect shape element type not integral number";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Broadcast shape must be an integral number"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(BroadcastTests, broadcast_axes_et_wrong)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc_shape = make_shared<op::Parameter>(element::i64, Shape{1});
    // wrong element type
    auto bc_axes = make_shared<op::Parameter>(element::f32, Shape{2});

    try
    {
        auto bc = make_shared<TypeParam>(arg, bc_shape, bc_axes);
        FAIL() << "Broadcast: did not detect axes element type not integral numbers";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Broadcast axes must be integral numbers, but are:"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

// EXPLICIT MODE

TYPED_TEST_P(BroadcastTests, broadcast_explicit_all_inputs_dynamic)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    const auto axes_mapping = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // const axes mapping
    const auto axes_mapping_const =
        op::Constant::create(element::i64, Shape{3}, vector<int64_t>{0, 1, 2});
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_explicit_target_shape_static_rank)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    const auto axes_mapping = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // const axes mapping
    const auto axes_mapping_const =
        op::Constant::create(element::i64, Shape{3}, vector<int64_t>{0, 1, 2});
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_explicit_const_target_shape)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto target_shape =
        op::Constant::create(element::i64, Shape{3}, vector<int64_t>{1, 2, 3});
    const auto axes_mapping = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");

    ASSERT_TRUE(bc->get_output_partial_shape(0).is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 3);
    ASSERT_EQ(bc->get_shape(), (Shape{1, 2, 3}));

    // const axes mapping
    const auto axes_mapping_const =
        op::Constant::create(element::i64, Shape{3}, vector<int64_t>{0, 2, 1});
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");

    ASSERT_TRUE(bc->get_output_partial_shape(0).is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 3);
    ASSERT_EQ(bc->get_shape(), (Shape{1, 2, 3}));
}

TYPED_TEST_P(BroadcastTests, broadcast_explicit_input_rank_static)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    const auto axes_mapping = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // const axes mapping
    const auto axes_mapping_const =
        op::Constant::create(element::i64, Shape{3}, vector<int64_t>{0, 2, 1});
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_explicit_target_shape_and_input_data_rank_static)
{
    // static rank data
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto axes_mapping = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // const axes mapping
    const auto axes_mapping_const =
        op::Constant::create(element::i64, Shape{3}, vector<int64_t>{0, 2, 1});
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_explicit_const_target_shape_static_rank_input)
{
    const auto target_shape =
        op::Constant::create(element::i64, Shape{4}, vector<int64_t>{1, 1, 5, 10});
    // static rank data
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(3));
    auto axes_mapping = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_shape(), (Shape{1, 1, 5, 10}));

    // const axes mapping
    const auto axes_mapping_const =
        op::Constant::create(element::i64, Shape{4}, vector<int64_t>{0, 2, 1, 3});
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_shape(), (Shape{1, 1, 5, 10}));
}

TYPED_TEST_P(BroadcastTests, broadcast_explicit_static_input_shape)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3, 4});
    // dynamic target shape and axes mapping
    auto target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto axes_mapping = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // const axes mapping
    const auto axes_mapping_const =
        op::Constant::create(element::i64, Shape{4}, vector<int64_t>{0, 2, 1, 3});
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // static rank target shape
    target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // static rank target shape and const axes mapping
    target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_explicit_static_input_shape_const_target_shape)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{4});
    auto target_shape = op::Constant::create(element::i64, Shape{4}, vector<int64_t>{1, 4, 2, 3});
    // dynamic axes mapping
    const auto axes_mapping = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_shape(), (Shape{1, 4, 2, 3}));

    // const axes mapping
    const auto axes_mapping_const =
        op::Constant::create(element::i64, Shape{1}, vector<int64_t>{1});
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_shape(), (Shape{1, 4, 2, 3}));
}

TYPED_TEST_P(BroadcastTests, broadcast_explicit_static_target_shape)
{
    // dynamic input
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto target_shape = make_shared<op::Parameter>(element::i64, PartialShape{4});
    const auto axes_mapping = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_dynamic());

    // static rank input
    data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(2));
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_dynamic());
}

// NUMPY MODE

TYPED_TEST_P(BroadcastTests, broadcast_numpy_input_shape_dynamic)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    // dynamic output shape
    auto target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // static rank target shape
    target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_numpy_target_shape_constant)
{
    // dynamic data
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto target_shape =
        op::Constant::create(element::i64, Shape{3}, vector<int64_t>{1, 2, 3});

    auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 3);

    // static rank data
    data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(2));
    bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 3);
}

TYPED_TEST_P(BroadcastTests, broadcast_numpy_target_shape_dynamic)
{
    // static rank data
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // static shape data
    data = make_shared<op::Parameter>(element::f32, PartialShape{3, 4, 5, 6});
    bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_numpy_input_target_shape_static_rank)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));

    const auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_numpy_input_static_shape)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3});
    // static rank target_shape
    auto target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));

    auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // constant target_shape
    const auto target_shape_const =
        op::Constant::create(element::i64, Shape{3}, vector<int64_t>{3, 2, 3});
    bc = make_shared<TypeParam>(data, target_shape_const, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 3);
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0), (PartialShape{3, 2, 3}));
}

TYPED_TEST_P(BroadcastTests, broadcast_numpy_input_partially_dynamic)
{
    const Shape expected_target_shape{1, 2, 3, 4};
    const auto target_shape = op::Constant::create(
        element::i64,
        {expected_target_shape.size()},
        std::vector<int64_t>(expected_target_shape.begin(), expected_target_shape.end()));

    auto data = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, Dimension::dynamic()});
    auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), expected_target_shape);

    data = make_shared<op::Parameter>(element::f32,
                                      PartialShape{Dimension::dynamic(), 3, Dimension::dynamic()});
    bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), expected_target_shape);

    data = make_shared<op::Parameter>(element::f32,
                                      PartialShape{2, Dimension::dynamic(), Dimension::dynamic()});
    bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), expected_target_shape);

    data = make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), expected_target_shape);
}

TYPED_TEST_P(BroadcastTests, broadcast_numpy_static_dims_incorrect)
{
    const auto target_shape = op::Constant::create(element::i64, Shape{4}, {1, 2, 3, 4});

    auto data =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 999, 3, 4});
    try
    {
        auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Input shape dimension equal 999 cannot be broadcasted (numpy mode) "
                             "to 2. Allowed input dimension value would be 1 or 2");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    data = make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 888});
    try
    {
        auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Input shape dimension equal 888 cannot be broadcasted (numpy mode) "
                             "to 4. Allowed input dimension value would be 1 or 4");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    data = make_shared<op::Parameter>(
        element::f32,
        PartialShape{5, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    try
    {
        auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Input shape dimension equal 5 cannot be broadcasted (numpy mode) to "
                             "1. Allowed input dimension value would be 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

REGISTER_TYPED_TEST_SUITE_P(BroadcastTests,
                            broadcast_numpy,
                            broadcast_axes_mapping,
                            broadcast_target_shape_as_concat_with_constants,
                            broadcast_target_shape_as_concat_with_node,
                            broadcast_fail_rank,
                            broadcast_fail_transpose,
                            broadcast_fail_axes_map,
                            broadcast_fail_axes_map_shape,
                            broadcast_axes_wrong_rank,
                            broadcast_fully_dynamic_target_shape,
                            broadcast_dynamic_values_of_target_shape,
                            broadcast_broadcast_shape_et_wrong,
                            broadcast_axes_et_wrong,
                            broadcast_explicit_all_inputs_dynamic,
                            broadcast_explicit_target_shape_static_rank,
                            broadcast_explicit_const_target_shape,
                            broadcast_explicit_input_rank_static,
                            broadcast_explicit_target_shape_and_input_data_rank_static,
                            broadcast_explicit_const_target_shape_static_rank_input,
                            broadcast_explicit_static_input_shape,
                            broadcast_explicit_static_input_shape_const_target_shape,
                            broadcast_explicit_static_target_shape,
                            broadcast_numpy_input_shape_dynamic,
                            broadcast_numpy_target_shape_constant,
                            broadcast_numpy_target_shape_dynamic,
                            broadcast_numpy_input_target_shape_static_rank,
                            broadcast_numpy_input_static_shape,
                            broadcast_numpy_input_partially_dynamic,
                            broadcast_numpy_static_dims_incorrect);

typedef ::testing::Types<op::v1::Broadcast, op::v3::Broadcast> BroadcastTypes;
// the last empty argument resolves compiler warning on MAC:
// `must specify at least one argument for '...'` (variadic macro)
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, BroadcastTests, BroadcastTypes, );

// changing AutoBroadcastSpec to BroadcastModeSpec forces runing pdpd tests separately
TEST(type_prop, broadcast_v1_pdpd)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1});
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 6});

    auto bc = make_shared<op::v1::Broadcast>(
        param, target_shape, op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 1));
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{2, 3, 6}));
}

TEST(type_prop, broadcast_v3_pdpd)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1});
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 6});

    auto bc = make_shared<op::v3::Broadcast>(
        param, target_shape, op::BroadcastModeSpec(op::BroadcastType::PDPD, 1));
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{2, 3, 6}));
}

TEST(type_prop, broadcast_v3_bidirectional_mode_string)
{
    const auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1});
    const auto shape = make_shared<op::Parameter>(element::i32, Shape{2});

    const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, "BIDIRECTIONAL");

    ASSERT_EQ(broadcast_v3->get_broadcast_spec(), op::BroadcastType::BIDIRECTIONAL);
    ASSERT_EQ(broadcast_v3->get_version(), 3);
}

TEST(type_prop, broadcast_v3_shape_unexpected_axes_mapping_input)
{
    const auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1});
    const auto shape = make_shared<op::Parameter>(element::i16, Shape{2});
    const auto axes_mapping = make_shared<op::Parameter>(element::f32, Shape{3});
    const auto broadcast_spec = op::BroadcastType::BIDIRECTIONAL;

    try
    {
        const auto broadcast_v3 =
            make_shared<op::v3::Broadcast>(arg, shape, axes_mapping, broadcast_spec);
        FAIL() << "Unexpected axes mapping input exception not thrown";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("axes_mapping input should not be provided for mode other than explicit"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_v3_not_provided_axes_input_for_explicit_mode)
{
    const auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1});
    const auto shape = make_shared<op::Parameter>(element::i16, Shape{2});
    const auto broadcast_spec = op::BroadcastType::EXPLICIT;

    try
    {
        const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, broadcast_spec);
        FAIL() << "axes_mapping input should be provided if explicit mode is used";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("axes_mapping input should be provided if explicit mode is used"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_v3_shape)
{
    const auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1});
    const auto shape = op::Constant::create(element::i64, {2}, {1, 4});
    const auto broadcast_spec = op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, broadcast_spec);

    ASSERT_EQ(broadcast_v3->get_element_type(), element::f32);
    ASSERT_EQ(broadcast_v3->get_shape(), (Shape{1, 4, 4}));
    ASSERT_EQ(broadcast_v3->get_broadcast_axes(), (make_pair<bool, AxisSet>(true, AxisSet{2})));
}

TEST(type_prop, broadcast_v3_shape_2)
{
    const auto arg = make_shared<op::Parameter>(element::f32, Shape{3, 1});
    const auto shape = op::Constant::create(element::i64, {3}, {2, 1, 6});
    const auto broadcast_spec = op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, broadcast_spec);

    ASSERT_EQ(broadcast_v3->get_element_type(), element::f32);
    ASSERT_EQ(broadcast_v3->get_shape(), (Shape{2, 3, 6}));
    ASSERT_EQ(broadcast_v3->get_broadcast_axes(), (make_pair<bool, AxisSet>(true, AxisSet{0, 2})));
}

TEST(type_prop, broadcast_v3_shape_3)
{
    const auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 1});
    const auto shape = op::Constant::create(element::i64, {2}, {2, 4});
    const auto broadcast_spec = op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, broadcast_spec);

    ASSERT_EQ(broadcast_v3->get_element_type(), element::f32);
    ASSERT_EQ(broadcast_v3->get_shape(), (Shape{2, 4}));
    ASSERT_EQ(broadcast_v3->get_broadcast_axes(), (make_pair<bool, AxisSet>(true, AxisSet{1})));
}

TEST(type_prop, broadcast_v3_shape_4)
{
    const auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 3, 1});
    const auto shape = op::Constant::create(element::i64, {2}, {3, 1});
    const auto broadcast_spec = op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, broadcast_spec);

    ASSERT_EQ(broadcast_v3->get_element_type(), element::f32);
    ASSERT_EQ(broadcast_v3->get_shape(), (Shape{1, 3, 1}));
    ASSERT_EQ(broadcast_v3->get_broadcast_axes(), (make_pair<bool, AxisSet>(true, AxisSet{})));
}

TEST(type_prop, broadcast_v3_shape_5)
{
    const auto arg = make_shared<op::Parameter>(element::f32, Shape{16, 1, 1});
    const auto shape = op::Constant::create(element::i64, {4}, {1, 1, 50, 50});
    const auto broadcast_spec = op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, broadcast_spec);

    ASSERT_EQ(broadcast_v3->get_element_type(), element::f32);
    ASSERT_EQ(broadcast_v3->get_shape(), (Shape{1, 16, 50, 50}));
    ASSERT_EQ(broadcast_v3->get_broadcast_axes(),
              (make_pair<bool, AxisSet>(true, AxisSet{0, 2, 3})));
}

TEST(type_prop, broadcast_v3_shape_6)
{
    const auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 3, 1});
    const auto shape = op::Constant::create(element::i64, {3}, {3, 1, 3});
    const auto broadcast_spec = op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, broadcast_spec);

    ASSERT_EQ(broadcast_v3->get_element_type(), element::f32);
    ASSERT_EQ(broadcast_v3->get_shape(), (Shape{3, 3, 3}));
    ASSERT_EQ(broadcast_v3->get_broadcast_axes(), (make_pair<bool, AxisSet>(true, AxisSet{0, 2})));
}

TEST(type_prop, broadcast_v3_shape_6_type_infer)
{
    const auto arg = make_shared<op::Parameter>(element::u16, Shape{1, 3, 1});
    const auto shape = op::Constant::create(element::i64, {3}, {3, 1, 3});
    const auto broadcast_spec = op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, broadcast_spec);

    ASSERT_EQ(broadcast_v3->get_element_type(), element::u16);
    ASSERT_EQ(broadcast_v3->get_shape(), (Shape{3, 3, 3}));
    ASSERT_EQ(broadcast_v3->get_broadcast_axes(), (make_pair<bool, AxisSet>(true, AxisSet{0, 2})));
}

TEST(type_prop, broadcast_v3_incorrect_target_shape)
{
    const auto arg = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});
    const auto shape = op::Constant::create(element::i64, {3}, {8, 6, 4});
    const auto broadcast_spec = op::BroadcastType::BIDIRECTIONAL;

    try
    {
        const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, broadcast_spec);
        FAIL() << "Not applicable breadcast exception not thrown";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Broadcast incorrect target shape. Expecting either 1 or 4. Got 8"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_v3_incorrect_target_shape_2)
{
    const auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 1, 2});
    const auto shape = op::Constant::create(element::i64, {2}, {2, 3});
    const auto broadcast_spec = op::BroadcastType::BIDIRECTIONAL;

    try
    {
        const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, broadcast_spec);
        FAIL() << "Not applicable breadcast exception not thrown";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Broadcast incorrect target shape. Expecting either 1 or 2. Got 3"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_v3_output_rank_not_deduced)
{
    const auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    const auto broadcast_spec = op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, broadcast_spec);

    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, broadcast_v3_output_rank_deduced_from_arg)
{
    const auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto shape = op::Constant::create(element::i64, {3}, {8, 6, 4});
    const auto broadcast_spec = op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, broadcast_spec);
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 8, 6, 4}));
}

TEST(type_prop, broadcast_v3_output_rank_deduced_from_new_shape_input)
{
    const auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto shape = op::Constant::create(element::i64, {5}, {8, 6, 1, 5, 1});
    const auto broadcast_spec = op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, broadcast_spec);
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0).rank().get_length(), 5);
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).same_scheme(
        PartialShape{8, 6, Dimension::dynamic(), 5, Dimension::dynamic()}));
}

TEST(type_prop, broadcast_v3_bidirectional_dynamic_input)
{
    const auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    // dynamic target shape
    auto target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_dynamic());

    // static rank target shape
    target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    broadcast_v3 = make_shared<op::v3::Broadcast>(arg, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_dynamic());

    // constant target shape
    const auto target_shape_const = op::Constant::create(element::i64, {3}, {2, 4, 6});
    broadcast_v3 = make_shared<op::v3::Broadcast>(arg, target_shape_const, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, broadcast_v3_bidirectional_static_rank_input)
{
    const auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(4));

    // dynamic target shape
    auto target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_dynamic());

    // static rank target shape
    target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    broadcast_v3 = make_shared<op::v3::Broadcast>(arg, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_dynamic());

    // constant target shape
    const auto target_shape_const = op::Constant::create(element::i64, {3}, {2, 4, 6});
    broadcast_v3 = make_shared<op::v3::Broadcast>(arg, target_shape_const, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).is_dynamic());
}

TEST(type_prop, broadcast_v3_bidirectional_static_shape_input)
{
    const auto arg = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3, 1});

    // dynamic target shape
    auto target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_dynamic());

    // static rank target shape
    target_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    broadcast_v3 = make_shared<op::v3::Broadcast>(arg, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_dynamic());

    // constant target shape
    auto target_shape_const = op::Constant::create(element::i64, {4}, {2, 2, 3, 2});
    broadcast_v3 = make_shared<op::v3::Broadcast>(arg, target_shape_const, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).is_static());
    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0), (PartialShape{2, 2, 3, 2}));

    target_shape_const = op::Constant::create(element::i64, {4}, {5, 2, 3, 7});
    broadcast_v3 = make_shared<op::v3::Broadcast>(arg, target_shape_const, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).is_static());
    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0), (PartialShape{5, 2, 3, 7}));
}

TEST(type_prop, broadcast_v3_bidirectional_partially_dynamic_input)
{
    const auto target_shape =
        op::Constant::create(element::i64, Shape{4}, vector<int64_t>{1, 1, 50, 50});

    auto data = make_shared<op::Parameter>(element::f32, PartialShape{16, 1, Dimension::dynamic()});
    auto bc = make_shared<op::v3::Broadcast>(data, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), (PartialShape{1, 16, 50, 50}));

    data = make_shared<op::Parameter>(element::f32,
                                      PartialShape{Dimension::dynamic(), 1, Dimension::dynamic()});
    bc = make_shared<op::v3::Broadcast>(data, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), (PartialShape{1, Dimension::dynamic(), 50, 50}));

    data = make_shared<op::Parameter>(element::f32,
                                      PartialShape{16, Dimension::dynamic(), Dimension::dynamic()});
    bc = make_shared<op::v3::Broadcast>(data, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), (PartialShape{1, 16, 50, 50}));

    data = make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    bc = make_shared<op::v3::Broadcast>(data, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), (PartialShape{1, Dimension::dynamic(), 50, 50}));
}
