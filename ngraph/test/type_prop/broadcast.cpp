//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, broadcast_deduce)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    Shape bc_shape{2, 3, 4};
    auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), bc_shape);
}

TEST(type_prop, broadcast_axes_oob)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc_shape = Shape{2, 3, 4};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1, 3});
        FAIL() << "Broadcast axis out of bounds not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Broadcast axis index (3) exceeds specified output shape rank");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_shape_mismatch_wrong_rank)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc_shape = Shape{2, 3, 4, 5};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
        FAIL() << "Output shape mismatch (wrong rank) not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Broadcast argument shape, specified output shape, and axes are incompatible");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_shape_mismatch_wrong_size)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc_shape = Shape{2, 3, 5};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
        FAIL() << "Output shape mismatch (wrong size) not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Broadcast argument shape, specified output shape, and axes are incompatible");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_partial_rank_dynamic_ok)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    Shape bc_shape{2, 3, 4};
    auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), bc_shape);
}

TEST(type_prop, broadcast_partial_rank_dynamic_axes_oob)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto bc_shape = Shape{2, 3, 4};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1, 3});
        FAIL() << "Broadcast axis out of bounds not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Broadcast axis index (3) exceeds specified output shape rank");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_partial_rank_static_dynamic_ok)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});
    Shape bc_shape{2, 3, 4};
    auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), bc_shape);
}

TEST(type_prop, broadcast_partial_rank_static_dynamic_axes_oob)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});
    auto bc_shape = Shape{2, 3, 4};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1, 3});
        FAIL() << "Broadcast axis out of bounds not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Broadcast axis index (3) exceeds specified output shape rank");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_partial_rank_static_dynamic_shape_mismatch_wrong_rank)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});
    auto bc_shape = Shape{2, 3, 4, 5};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
        FAIL() << "Output shape mismatch (wrong rank) not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Broadcast argument shape, specified output shape, and axes are incompatible");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_partial_rank_static_dynamic_shape_mismatch_wrong_size)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});
    auto bc_shape = Shape{2, 3, 5};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
        FAIL() << "Output shape mismatch (wrong size) not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Broadcast argument shape, specified output shape, and axes are incompatible");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

// Because v3::Broadcast is backward compatible to v1::Broadcast all v1::Broadcast tests should pass
template <typename T>
class BroadcastTests : public ::testing::Test
{
};
TYPED_TEST_CASE_P(BroadcastTests);

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
    ASSERT_TRUE(bc->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 16, 50, 50}));
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
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1});
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 3});
    auto axes_mapping = op::Constant::create<int64_t>(element::i64, Shape{2}, {1, 2});

    try
    {
        auto bc = make_shared<TypeParam>(param, target_shape, axes_mapping);
        FAIL() << "Broadcast: wrong target shape not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Broadcast target[axes_mapping[1]] Expected 1. Got 3");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(BroadcastTests, broadcast_shape_wrong_rank)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc_shape = make_shared<op::Parameter>(element::i64, Shape{1, 1});
    auto bc_axes = make_shared<op::Parameter>(element::i64, Shape{1});

    try
    {
        auto bc = make_shared<TypeParam>(arg, bc_shape, bc_axes);
        FAIL() << "DynBroadcast: wrong shape rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Broadcast shape rank must be 1");
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

REGISTER_TYPED_TEST_CASE_P(BroadcastTests,
                           broadcast_numpy,
                           broadcast_axes_mapping,
                           broadcast_target_shape_as_concat_with_constants,
                           broadcast_target_shape_as_concat_with_node,
                           broadcast_fail_rank,
                           broadcast_fail_transpose,
                           broadcast_fail_axes_map,
                           broadcast_fail_axes_map_shape,
                           broadcast_shape_wrong_rank,
                           broadcast_axes_wrong_rank,
                           broadcast_fully_dynamic_target_shape,
                           broadcast_broadcast_shape_et_wrong,
                           broadcast_axes_et_wrong);

typedef ::testing::Types<op::v1::Broadcast, op::v3::Broadcast> BroadcastTypes;
// the last empty argument resolves compiler warning on MAC:
// `must specify at least one argument for '...'` (variadic macro)
INSTANTIATE_TYPED_TEST_CASE_P(type_prop, BroadcastTests, BroadcastTypes, );

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
