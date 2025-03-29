// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/broadcast.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/attr_types.hpp"

using namespace std;
using namespace testing;

// Because v3::Broadcast is backward compatible to v1::Broadcast all v1::Broadcast tests should pass
template <typename T>
class BroadcastTests : public ::testing::Test {};
TYPED_TEST_SUITE_P(BroadcastTests);

TYPED_TEST_P(BroadcastTests, broadcast_dynamic_value_propagation) {
    ov::Dimension marked = ov::Dimension(3);
    auto A = std::make_shared<ov::Symbol>();
    marked.set_symbol(A);
    ov::PartialShape target = ov::PartialShape{1, 2, marked, 4};

    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1});
    auto param_1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, target);
    auto shape = make_shared<ov::op::v3::ShapeOf>(param_1);

    auto indices = ov::op::v0::Constant::create(ov::element::i32, {}, {2});
    auto axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {0});
    auto gather = make_shared<ov::op::v1::Gather>(shape, indices, axis);
    auto unsqueeze = make_shared<ov::op::v0::Unsqueeze>(gather, axis);

    auto five = ov::op::v0::Constant::create(ov::element::i64, {1}, {5});
    auto target_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{unsqueeze, five}, 0);

    auto bc = make_shared<TypeParam>(param, target_shape);
    ASSERT_EQ(bc->get_element_type(), ov::element::f32);
    ASSERT_EQ(bc->get_shape(), (ov::Shape{3, 5}));
    ASSERT_TRUE(ov::symbol::are_equal(bc->get_output_partial_shape(0)[0].get_symbol(), A));
}

TYPED_TEST_P(BroadcastTests, broadcast_numpy) {
    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 1});
    auto target_shape = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{3}, {2, 3, 6});

    auto bc = make_shared<TypeParam>(param, target_shape);
    ASSERT_EQ(bc->get_element_type(), ov::element::f32);
    ASSERT_EQ(bc->get_shape(), (ov::Shape{2, 3, 6}));
}

TYPED_TEST_P(BroadcastTests, broadcast_axes_mapping) {
    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 1});
    auto target_shape = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{3}, {2, 3, 1});
    auto axes_mapping = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {1, 2});

    auto bc = make_shared<TypeParam>(param, target_shape, axes_mapping);
    ASSERT_EQ(bc->get_element_type(), ov::element::f32);
    ASSERT_EQ(bc->get_shape(), (ov::Shape{2, 3, 1}));
}

TYPED_TEST_P(BroadcastTests, broadcast_target_shape_as_concat_with_constants) {
    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{16});
    auto target_shape_constant_1 = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{1}, {1});
    auto target_shape_constant_2 = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{1}, {16});
    auto target_shape_constant_3 = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{1}, {50});
    auto target_shape_constant_4 = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{1}, {50});
    std::int64_t axis = 0;
    std::vector<std::shared_ptr<ov::Node>> args{target_shape_constant_1,
                                                target_shape_constant_2,
                                                target_shape_constant_3,
                                                target_shape_constant_4};
    auto target_shape = make_shared<ov::op::v0::Concat>(args, axis);
    auto axes_mapping = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{1}, {1});
    auto bc = make_shared<TypeParam>(param, target_shape, axes_mapping, "NONE");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank(), (ov::Rank{4}));
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0), (ov::PartialShape{1, 16, 50, 50}));
}

TYPED_TEST_P(BroadcastTests, broadcast_target_shape_as_concat_with_node) {
    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{16});
    auto target_shape_constant_1 = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
    auto target_shape_constant_2 = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{1}, {16});
    auto target_shape_constant_3 = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{1}, {50});
    auto target_shape_constant_4 = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{1}, {50});
    std::int64_t axis = 0;
    std::vector<std::shared_ptr<ov::Node>> args{target_shape_constant_1,
                                                target_shape_constant_2,
                                                target_shape_constant_3,
                                                target_shape_constant_4};
    auto target_shape = make_shared<ov::op::v0::Concat>(args, axis);
    auto axes_mapping = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{1}, {1});
    auto bc = make_shared<TypeParam>(param, target_shape, axes_mapping, "NONE");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank(), (ov::Rank{4}));
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_dynamic());
    ASSERT_EQ(bc->get_output_partial_shape(0), ov::PartialShape({ov::Dimension::dynamic(), 16, 50, 50}));
}

TYPED_TEST_P(BroadcastTests, broadcast_fail_rank) {
    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 1});
    auto target_shape = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{3}, {2, 3, 1});
    auto axes_mapping = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{3}, {1, 2, 3});

    try {
        auto bc = make_shared<TypeParam>(param, target_shape, axes_mapping);
        FAIL() << "Broadcast: target shape mismatch with input rank not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Broadcast axes_mapping shape [3] doesn't match rank of input tensor 2");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(BroadcastTests, broadcast_fail_transpose) {
    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 1});
    auto target_shape = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{3}, {2, 1, 3});
    auto axes_mapping = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {2, 1});

    try {
        auto bc = make_shared<TypeParam>(param, target_shape, axes_mapping);
        FAIL() << "Broadcast: transpose prohibition not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Broadcast doesn't permit transposes. axes_mapping AxisVector{2, 1} "
                             "not in sorted order");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(BroadcastTests, broadcast_fail_axes_map) {
    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 1});
    auto target_shape = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{3}, {2, 3, 1});
    auto axes_mapping = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {1, 3});

    try {
        auto bc = make_shared<TypeParam>(param, target_shape, axes_mapping);
        FAIL() << "Broadcast: wrong axes_map not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Broadcast axes_mapping[1]: 3 exceeds target rank 3");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(BroadcastTests, broadcast_fail_axes_map_shape) {
    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 2});
    auto target_shape = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{3}, {2, 3, 3});
    auto axes_mapping = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {1, 2});

    try {
        auto bc = make_shared<TypeParam>(param, target_shape, axes_mapping);
        FAIL() << "Broadcast: wrong target shape not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Broadcast target[axes_mapping[1]] Expected 2. Got 3");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(BroadcastTests, broadcast_axes_wrong_rank) {
    auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    auto bc_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
    auto bc_axes = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2, 2});

    try {
        auto bc = make_shared<TypeParam>(arg, bc_shape, bc_axes);
        FAIL() << "Broadcast: axes shape rank not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Broadcast axes rank must be 1");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(BroadcastTests, broadcast_target_shape_wrong_rank) {
    auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    auto bc_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{});

    try {
        auto bc = make_shared<TypeParam>(arg, bc_shape);
        FAIL() << "Broadcast: axes target shape rank not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Broadcast shape rank must be 1, but has");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(BroadcastTests, broadcast_fully_dynamic_target_shape) {
    auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    auto bc_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());
    auto bc_axes = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2});

    auto bc = make_shared<TypeParam>(arg, bc_shape, bc_axes);
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_dynamic());

    bc_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
    bc = make_shared<TypeParam>(arg, bc_shape, bc_axes);
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_dynamic_values_of_target_shape) {
    const auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2});
    const auto target = make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape::dynamic(4));
    const auto target_shape = std::make_shared<ov::op::v3::ShapeOf>(target);
    const auto axes_mapping = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping);
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_dynamic());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), ov::PartialShape::dynamic(4));
}

TYPED_TEST_P(BroadcastTests, broadcast_broadcast_shape_et_wrong) {
    auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    // wrong element type
    auto bc_shape = make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{1});
    auto bc_axes = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2});

    try {
        auto bc = make_shared<TypeParam>(arg, bc_shape, bc_axes);
        FAIL() << "Broadcast: did not detect shape element type not integral number";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Broadcast shape must be an integral number"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(BroadcastTests, broadcast_axes_et_wrong) {
    auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    auto bc_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
    // wrong element type
    auto bc_axes = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2});

    try {
        auto bc = make_shared<TypeParam>(arg, bc_shape, bc_axes);
        FAIL() << "Broadcast: did not detect axes element type not integral numbers";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Broadcast axes must be integral numbers, but are:"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

// EXPLICIT MODE

TYPED_TEST_P(BroadcastTests, broadcast_explicit_all_inputs_dynamic) {
    const auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());
    const auto axes_mapping = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // const axes mapping
    const auto axes_mapping_const =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, vector<int64_t>{0, 1, 2});
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_explicit_target_shape_static_rank) {
    const auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic(1));
    const auto axes_mapping = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // const axes mapping
    const auto axes_mapping_const =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, vector<int64_t>{0, 1, 2});
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_explicit_const_target_shape) {
    const auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto target_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, vector<int64_t>{1, 2, 3});
    const auto axes_mapping = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");

    ASSERT_TRUE(bc->get_output_partial_shape(0).is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 3);
    ASSERT_EQ(bc->get_shape(), (ov::Shape{1, 2, 3}));

    // const axes mapping
    const auto axes_mapping_const =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, vector<int64_t>{0, 2, 1});
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");

    ASSERT_TRUE(bc->get_output_partial_shape(0).is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 3);
    ASSERT_EQ(bc->get_shape(), (ov::Shape{1, 2, 3}));
}

TYPED_TEST_P(BroadcastTests, broadcast_explicit_input_rank_static) {
    const auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));
    const auto target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());
    const auto axes_mapping = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // const axes mapping
    const auto axes_mapping_const =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, vector<int64_t>{0, 2, 1});
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_explicit_target_shape_and_input_data_rank_static) {
    // static rank data
    const auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));
    const auto target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic(1));
    auto axes_mapping = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // const axes mapping
    const auto axes_mapping_const =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, vector<int64_t>{0, 2, 1});
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_explicit_const_target_shape_static_rank_input) {
    const auto target_shape =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, vector<int64_t>{1, 1, 5, 10});
    // static rank data
    const auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));
    auto axes_mapping = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_shape(), (ov::Shape{1, 1, 5, 10}));

    // const axes mapping
    const auto axes_mapping_const =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, vector<int64_t>{0, 2, 1, 3});
    try {
        auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");
        FAIL() << "Broadcast: Broadcast axes_mapping shape doesn't match rank of input tensor";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Broadcast axes_mapping shape [4] doesn't match rank of input tensor 3"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(BroadcastTests, broadcast_explicit_static_input_shape) {
    const auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 4});
    // dynamic target shape and axes mapping
    auto target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());
    auto axes_mapping = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // const axes mapping
    const auto axes_mapping_const =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, vector<int64_t>{0, 2, 1, 3});
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // static rank target shape
    target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic(1));
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // static rank target shape and const axes mapping
    target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic(1));
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_explicit_static_input_shape_const_target_shape) {
    const auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4});
    auto target_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, vector<int64_t>{1, 4, 2, 3});
    // dynamic axes mapping
    const auto axes_mapping = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_shape(), (ov::Shape{1, 4, 2, 3}));

    // const axes mapping
    const auto axes_mapping_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, vector<int64_t>{1});
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping_const, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_shape(), (ov::Shape{1, 4, 2, 3}));
}

TYPED_TEST_P(BroadcastTests, broadcast_explicit_static_target_shape) {
    // dynamic input
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{4});
    const auto axes_mapping = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_dynamic());

    // static rank input
    data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(2));
    bc = make_shared<TypeParam>(data, target_shape, axes_mapping, "EXPLICIT");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_dynamic());
}

// NUMPY MODE

TYPED_TEST_P(BroadcastTests, broadcast_numpy_input_shape_dynamic) {
    const auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    // dynamic output shape
    auto target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // static rank target shape
    target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic(1));
    bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_numpy_target_shape_constant) {
    // dynamic data
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto target_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, vector<int64_t>{1, 2, 3});

    auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 3);

    // static rank data
    data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(2));
    bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 3);
}

TYPED_TEST_P(BroadcastTests, broadcast_numpy_target_shape_dynamic) {
    // static rank data
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));
    const auto target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());

    auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // static shape data
    data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3, 4, 5, 6});
    bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_numpy_input_target_shape_static_rank) {
    const auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));
    const auto target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic(1));

    const auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BroadcastTests, broadcast_numpy_input_static_shape) {
    const auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3});
    // static rank target_shape
    auto target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic(1));

    auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_dynamic());

    // constant target_shape
    const auto target_shape_const =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, vector<int64_t>{3, 2, 3});
    bc = make_shared<TypeParam>(data, target_shape_const, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 3);
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0), (ov::PartialShape{3, 2, 3}));
}

TYPED_TEST_P(BroadcastTests, broadcast_numpy_input_partially_dynamic) {
    const ov::Shape expected_target_shape{1, 2, 3, 4};
    const auto target_shape =
        ov::op::v0::Constant::create(ov::element::i64,
                                     {expected_target_shape.size()},
                                     std::vector<int64_t>(expected_target_shape.begin(), expected_target_shape.end()));

    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 3, ov::Dimension::dynamic()});
    auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), expected_target_shape);

    data = make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                              ov::PartialShape{ov::Dimension::dynamic(), 3, ov::Dimension::dynamic()});
    bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), expected_target_shape);

    data = make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                              ov::PartialShape{2, ov::Dimension::dynamic(), ov::Dimension::dynamic()});
    bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), expected_target_shape);

    data = make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()});
    bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), expected_target_shape);
}

TYPED_TEST_P(BroadcastTests, broadcast_numpy_static_dims_incorrect) {
    const auto target_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 2, 3, 4});

    auto data =
        make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), 999, 3, 4});
    try {
        auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Input shape dimension equal 999 cannot be broadcasted (numpy mode) "
                             "to 2. Allowed input dimension value would be 1 or 2");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    data = make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic(), 888});
    try {
        auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Input shape dimension equal 888 cannot be broadcasted (numpy mode) "
                             "to 4. Allowed input dimension value would be 1 or 4");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    data = make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::PartialShape{5, ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()});
    try {
        auto bc = make_shared<TypeParam>(data, target_shape, "NUMPY");
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Input shape dimension equal 5 cannot be broadcasted (numpy mode) to "
                             "1. Allowed input dimension value would be 1");
    } catch (...) {
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
                            broadcast_target_shape_wrong_rank,
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
                            broadcast_numpy_static_dims_incorrect,
                            broadcast_dynamic_value_propagation);

typedef ::testing::Types<ov::op::v1::Broadcast, ov::op::v3::Broadcast> BroadcastTypes;
// the last empty argument resolves compiler warning on MAC:
// `must specify at least one argument for '...'` (variadic macro)
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, BroadcastTests, BroadcastTypes, );

// changing AutoBroadcastSpec to BroadcastModeSpec forces runing pdpd tests separately
TEST(type_prop, broadcast_v1_pdpd) {
    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 1});
    auto target_shape = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{3}, {2, 3, 6});

    auto bc = make_shared<ov::op::v1::Broadcast>(param,
                                                 target_shape,
                                                 ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 1));
    ASSERT_EQ(bc->get_element_type(), ov::element::f32);
    ASSERT_EQ(bc->get_shape(), (ov::Shape{2, 3, 6}));
}

TEST(type_prop, broadcast_v3_pdpd) {
    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 1});
    auto target_shape = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{3}, {2, 3, 6});

    auto bc = make_shared<ov::op::v3::Broadcast>(param,
                                                 target_shape,
                                                 ov::op::BroadcastModeSpec(ov::op::BroadcastType::PDPD, 1));
    ASSERT_EQ(bc->get_element_type(), ov::element::f32);
    ASSERT_EQ(bc->get_shape(), (ov::Shape{2, 3, 6}));
}

TEST(type_prop, broadcast_v3_bidirectional_mode_string) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 1});
    const auto shape = make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});

    const auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, shape, "BIDIRECTIONAL");

    ASSERT_EQ(broadcast_v3->get_broadcast_spec(), ov::op::BroadcastType::BIDIRECTIONAL);
}

TEST(type_prop, broadcast_v3_shape_unexpected_axes_mapping_input) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 1});
    const auto shape = make_shared<ov::op::v0::Parameter>(ov::element::i16, ov::Shape{2});
    const auto axes_mapping = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    const auto broadcast_spec = ov::op::BroadcastType::BIDIRECTIONAL;

    try {
        const auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, shape, axes_mapping, broadcast_spec);
        FAIL() << "Unexpected axes mapping input exception not thrown";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("axes_mapping input should not be provided for mode other than explicit"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_v3_not_provided_axes_input_for_explicit_mode) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 1});
    const auto shape = make_shared<ov::op::v0::Parameter>(ov::element::i16, ov::Shape{2});
    const auto broadcast_spec = ov::op::BroadcastType::EXPLICIT;

    try {
        const auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, shape, broadcast_spec);
        FAIL() << "axes_mapping input should be provided if explicit mode is used";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("axes_mapping input should be provided if explicit mode is used"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_v3_shape) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 1});
    const auto shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {1, 4});
    const auto broadcast_spec = ov::op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, shape, broadcast_spec);

    ASSERT_EQ(broadcast_v3->get_element_type(), ov::element::f32);
    ASSERT_EQ(broadcast_v3->get_shape(), (ov::Shape{1, 4, 4}));
    ASSERT_EQ(broadcast_v3->get_broadcast_axes(), (make_pair<bool, ov::AxisSet>(true, ov::AxisSet{2})));
}

TEST(type_prop, broadcast_v3_shape_2) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 1});
    const auto shape = ov::op::v0::Constant::create(ov::element::i64, {3}, {2, 1, 6});
    const auto broadcast_spec = ov::op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, shape, broadcast_spec);

    ASSERT_EQ(broadcast_v3->get_element_type(), ov::element::f32);
    ASSERT_EQ(broadcast_v3->get_shape(), (ov::Shape{2, 3, 6}));
    ASSERT_EQ(broadcast_v3->get_broadcast_axes(), (make_pair<bool, ov::AxisSet>(true, ov::AxisSet{0, 2})));
}

TEST(type_prop, broadcast_v3_shape_3) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 1});
    const auto shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 4});
    const auto broadcast_spec = ov::op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, shape, broadcast_spec);

    ASSERT_EQ(broadcast_v3->get_element_type(), ov::element::f32);
    ASSERT_EQ(broadcast_v3->get_shape(), (ov::Shape{2, 4}));
    ASSERT_EQ(broadcast_v3->get_broadcast_axes(), (make_pair<bool, ov::AxisSet>(true, ov::AxisSet{1})));
}

TEST(type_prop, broadcast_v3_shape_4) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 1});
    const auto shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {3, 1});
    const auto broadcast_spec = ov::op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, shape, broadcast_spec);

    ASSERT_EQ(broadcast_v3->get_element_type(), ov::element::f32);
    ASSERT_EQ(broadcast_v3->get_shape(), (ov::Shape{1, 3, 1}));
    ASSERT_EQ(broadcast_v3->get_broadcast_axes(), (make_pair<bool, ov::AxisSet>(true, ov::AxisSet{})));
}

TEST(type_prop, broadcast_v3_shape_5) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{16, 1, 1});
    const auto shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, 1, 50, 50});
    const auto broadcast_spec = ov::op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, shape, broadcast_spec);

    ASSERT_EQ(broadcast_v3->get_element_type(), ov::element::f32);
    ASSERT_EQ(broadcast_v3->get_shape(), (ov::Shape{1, 16, 50, 50}));
    ASSERT_EQ(broadcast_v3->get_broadcast_axes(), (make_pair<bool, ov::AxisSet>(true, ov::AxisSet{0, 2, 3})));
}

TEST(type_prop, broadcast_v3_shape_6) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 1});
    const auto shape = ov::op::v0::Constant::create(ov::element::i64, {3}, {3, 1, 3});
    const auto broadcast_spec = ov::op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, shape, broadcast_spec);

    ASSERT_EQ(broadcast_v3->get_element_type(), ov::element::f32);
    ASSERT_EQ(broadcast_v3->get_shape(), (ov::Shape{3, 3, 3}));
    ASSERT_EQ(broadcast_v3->get_broadcast_axes(), (make_pair<bool, ov::AxisSet>(true, ov::AxisSet{0, 2})));
}

TEST(type_prop, broadcast_v3_shape_6_type_infer) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::u16, ov::Shape{1, 3, 1});
    const auto shape = ov::op::v0::Constant::create(ov::element::i64, {3}, {3, 1, 3});
    const auto broadcast_spec = ov::op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, shape, broadcast_spec);

    ASSERT_EQ(broadcast_v3->get_element_type(), ov::element::u16);
    ASSERT_EQ(broadcast_v3->get_shape(), (ov::Shape{3, 3, 3}));
    ASSERT_EQ(broadcast_v3->get_broadcast_axes(), (make_pair<bool, ov::AxisSet>(true, ov::AxisSet{0, 2})));
}

TEST(type_prop, broadcast_v3_incorrect_target_shape) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 3, 2});
    const auto shape = ov::op::v0::Constant::create(ov::element::i64, {3}, {8, 6, 4});
    const auto broadcast_spec = ov::op::BroadcastType::BIDIRECTIONAL;

    try {
        const auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, shape, broadcast_spec);
        FAIL() << "Not applicable breadcast exception not thrown";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Broadcast incorrect target shape. Expecting either 1 or 4. Got 8"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_v3_incorrect_target_shape_2) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 2});
    const auto shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});
    const auto broadcast_spec = ov::op::BroadcastType::BIDIRECTIONAL;

    try {
        const auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, shape, broadcast_spec);
        FAIL() << "Not applicable breadcast exception not thrown";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Broadcast incorrect target shape. Expecting either 1 or 2. Got 3"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_v3_output_rank_not_deduced) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic(1));
    const auto broadcast_spec = ov::op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, shape, broadcast_spec);

    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0), (ov::PartialShape::dynamic()));
}

TEST(type_prop, broadcast_v3_output_rank_deduced_from_arg) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    const auto shape = ov::op::v0::Constant::create(ov::element::i64, {3}, {8, 6, 4});
    const auto broadcast_spec = ov::op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, shape, broadcast_spec);
    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0), (ov::PartialShape{ov::Dimension::dynamic(), 8, 6, 4}));
}

TEST(type_prop, broadcast_v3_output_rank_deduced_from_new_shape_input) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    const auto shape = ov::op::v0::Constant::create(ov::element::i64, {5}, {8, 6, 1, 5, 1});
    const auto broadcast_spec = ov::op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, shape, broadcast_spec);
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0).rank().get_length(), 5);
    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0),
              (ov::PartialShape{8, 6, ov::Dimension::dynamic(), 5, ov::Dimension::dynamic()}));
}

TEST(type_prop, broadcast_v3_bidirectional_dynamic_input) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());

    // dynamic target shape
    auto target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());
    auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_dynamic());

    // static rank target shape
    target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic(1));
    broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_dynamic());

    // constant target shape
    const auto target_shape_const = ov::op::v0::Constant::create(ov::element::i64, {3}, {2, 4, 6});
    broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, target_shape_const, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, broadcast_v3_bidirectional_static_rank_input) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));

    // dynamic target shape
    auto target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());
    auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_dynamic());

    // static rank target shape
    target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic(1));
    broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_dynamic());

    // constant target shape
    const auto target_shape_const = ov::op::v0::Constant::create(ov::element::i64, {3}, {2, 4, 6});
    broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, target_shape_const, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).is_dynamic());
}

TEST(type_prop, broadcast_v3_bidirectional_static_shape_input) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 1});

    // dynamic target shape
    auto target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic());
    auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_dynamic());

    // static rank target shape
    target_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape::dynamic(1));
    broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_dynamic());

    // constant target shape
    auto target_shape_const = ov::op::v0::Constant::create(ov::element::i64, {4}, {2, 2, 3, 2});
    broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, target_shape_const, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).is_static());
    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0), (ov::PartialShape{2, 2, 3, 2}));

    target_shape_const = ov::op::v0::Constant::create(ov::element::i64, {4}, {5, 2, 3, 7});
    broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, target_shape_const, "BIDIRECTIONAL");
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_TRUE(broadcast_v3->get_output_partial_shape(0).is_static());
    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0), (ov::PartialShape{5, 2, 3, 7}));
}

TEST(type_prop, broadcast_v3_bidirectional_partially_dynamic_input) {
    const auto target_shape =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, vector<int64_t>{1, 1, 50, 50});

    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{16, 1, ov::Dimension::dynamic()});
    auto bc = make_shared<ov::op::v3::Broadcast>(data, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), (ov::PartialShape{1, 16, 50, 50}));

    data = make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                              ov::PartialShape{ov::Dimension::dynamic(), 1, ov::Dimension::dynamic()});
    bc = make_shared<ov::op::v3::Broadcast>(data, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), (ov::PartialShape{1, ov::Dimension::dynamic(), 50, 50}));

    data = make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                              ov::PartialShape{16, ov::Dimension::dynamic(), ov::Dimension::dynamic()});
    bc = make_shared<ov::op::v3::Broadcast>(data, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), (ov::PartialShape{1, 16, 50, 50}));

    data = make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()});
    bc = make_shared<ov::op::v3::Broadcast>(data, target_shape, "BIDIRECTIONAL");
    ASSERT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    ASSERT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 4);
    ASSERT_EQ(bc->get_output_partial_shape(0), (ov::PartialShape{1, ov::Dimension::dynamic(), 50, 50}));
}

TEST(type_prop, broadcast_i32_shape_value) {
    const auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape({5, -1}));
    const auto shape = make_shared<ov::op::v3::ShapeOf>(arg, ov::element::i64);
    const auto broadcast_spec = ov::op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<ov::op::v3::Broadcast>(arg, shape, broadcast_spec);

    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0), ov::PartialShape({5, -1}));

    // shape type resetting
    shape->set_output_type(ov::element::i32);
    arg->revalidate_and_infer_types();
    shape->revalidate_and_infer_types();
    broadcast_v3->revalidate_and_infer_types();

    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0), ov::PartialShape({5, -1}));

    // broadcast type resetting
    broadcast_v3->set_broadcast_spec(ov::op::BroadcastType::NUMPY);
    arg->revalidate_and_infer_types();
    shape->revalidate_and_infer_types();
    broadcast_v3->revalidate_and_infer_types();

    ASSERT_EQ(broadcast_v3->get_output_partial_shape(0), ov::PartialShape({5, -1}));
}

TEST(type_prop, broadcast_v3_default_constructor) {
    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{5, 2, 3, 1});
    auto target_shape = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{3}, {1, 3, 6});

    auto op = make_shared<ov::op::v3::Broadcast>();

    EXPECT_EQ(op->get_broadcast_spec().m_type, ov::op::BroadcastType::NUMPY);

    op->set_broadcast_spec(ov::op::BroadcastType::BIDIRECTIONAL);
    EXPECT_EQ(op->get_broadcast_spec().m_type, ov::op::BroadcastType::BIDIRECTIONAL);

    op->set_argument(0, param);
    op->set_argument(1, target_shape);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_shape(), (ov::Shape{5, 2, 3, 6}));
}

TEST(type_prop, broadcast_v3_bidirectional_data_bigger_rank_numpy) {
    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{5, 2, 3, 1});
    auto target_shape = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{3}, {4, 3, 6});

    OV_EXPECT_THROW(auto b = make_shared<ov::op::v3::Broadcast>(param, target_shape),
                    ov::NodeValidationFailure,
                    HasSubstr("Broadcast target_shape has smaller rank"));
}

TEST(type_prop, broadcast_v3_symbols_in0_dynamic_mixed_dims_bidirectional) {
    // All dimensions of A have symbols, B without symbols
    ov::PartialShape pshape_a{-1, 2, 1, {4, 8}, -1, {4, 8}, -1, {1, 8}, {1, 10}, {4, 18}};
    ov::PartialShape pshape_b{-1, 2, {3, 9}, 1, {3, 9}, -1, {1, 9}, -1, {3, 19}, {1, 10}};

    ov::PartialShape expected_shape = {-1, 2, {3, 9}, {4, 8}, {3, 9}, {4, 8}, -1, -1, {3, 19}, {4, 18}};

    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>(), C = std::make_shared<ov::Symbol>(),
         D = std::make_shared<ov::Symbol>();
    auto E = std::make_shared<ov::Symbol>(), F = std::make_shared<ov::Symbol>(), G = std::make_shared<ov::Symbol>(),
         H = std::make_shared<ov::Symbol>();
    auto I = std::make_shared<ov::Symbol>(), J = std::make_shared<ov::Symbol>();
    const std::shared_ptr<ov::Symbol> NO = nullptr;

    ov::TensorSymbol expected_symbols{A, B, NO, D, NO, F, G, H, NO, J};

    set_shape_symbols(pshape_a, {A, B, C, D, E, F, G, H, I, J});

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_a);
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_b);
    auto shape_of = make_shared<ov::op::v3::ShapeOf>(target_shape);

    auto op = make_shared<ov::op::v3::Broadcast>(data, shape_of, "BIDIRECTIONAL");

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, broadcast_v3_symbols_in1_dynamic_mixed_dims_bidirectional) {
    // All dimensions of B have symbols, A without symbols
    ov::PartialShape pshape_a{-1, 2, 1, {4, 8}, -1, {4, 8}, -1, {1, 8}, {1, 10}, {4, 18}};
    ov::PartialShape pshape_b{-1, 2, {3, 9}, 1, {3, 9}, -1, {1, 9}, -1, {3, 19}, {1, 10}};

    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>(), C = std::make_shared<ov::Symbol>(),
         D = std::make_shared<ov::Symbol>();
    auto E = std::make_shared<ov::Symbol>(), F = std::make_shared<ov::Symbol>(), G = std::make_shared<ov::Symbol>(),
         H = std::make_shared<ov::Symbol>();
    auto I = std::make_shared<ov::Symbol>(), J = std::make_shared<ov::Symbol>();
    const std::shared_ptr<ov::Symbol> NO = nullptr;

    ov::PartialShape expected_shape = {-1, 2, {3, 9}, {4, 8}, {3, 9}, {4, 8}, -1, -1, {3, 19}, {4, 18}};
    ov::TensorSymbol expected_symbols{A, B, C, NO, E, NO, G, H, I, NO};

    set_shape_symbols(pshape_b, {A, B, C, D, E, F, G, H, I, J});

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_a);
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_b);
    auto shape_of = make_shared<ov::op::v3::ShapeOf>(target_shape);

    auto op = make_shared<ov::op::v3::Broadcast>(data, shape_of, "BIDIRECTIONAL");

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, broadcast_v3_symbols_different_dynamic_mixed_dims_broadcast_bidirectional) {
    // Both params have dimensions with different symbols
    ov::PartialShape pshape_a{-1, 2, 1, {4, 8}, -1, {4, 8}, -1, {1, 8}, {1, 10}, {4, 18}};
    ov::PartialShape pshape_b{-1, 2, {3, 9}, 1, {3, 9}, -1, {1, 9}, -1, {3, 19}, {1, 10}};

    ov::PartialShape expected_shape = {-1, 2, {3, 9}, {4, 8}, {3, 9}, {4, 8}, -1, -1, {3, 19}, {4, 18}};

    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>(), C = std::make_shared<ov::Symbol>(),
         D = std::make_shared<ov::Symbol>();
    auto E = std::make_shared<ov::Symbol>(), F = std::make_shared<ov::Symbol>(), G = std::make_shared<ov::Symbol>(),
         H = std::make_shared<ov::Symbol>();
    auto I = std::make_shared<ov::Symbol>(), J = std::make_shared<ov::Symbol>(), K = std::make_shared<ov::Symbol>(),
         L = std::make_shared<ov::Symbol>();
    auto M = std::make_shared<ov::Symbol>(), N = std::make_shared<ov::Symbol>(), O = std::make_shared<ov::Symbol>(),
         P = std::make_shared<ov::Symbol>();
    auto Q = std::make_shared<ov::Symbol>(), R = std::make_shared<ov::Symbol>(), S = std::make_shared<ov::Symbol>(),
         T = std::make_shared<ov::Symbol>();
    const std::shared_ptr<ov::Symbol> NO = nullptr;
    ov::TensorSymbol expected_symbols{NO, B, M, D, O, F, NO, NO, S, J};

    set_shape_symbols(pshape_a, {A, B, C, D, E, F, G, H, I, J});
    set_shape_symbols(pshape_b, {K, L, M, N, O, P, Q, R, S, T});

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_a);
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_b);
    auto shape_of = make_shared<ov::op::v3::ShapeOf>(target_shape);

    auto op = make_shared<ov::op::v3::Broadcast>(data, shape_of, "BIDIRECTIONAL");

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, broadcast_v3_symbols_same_dynamic_mixed_dims_broadcast_bidirectional) {
    // Both params have dimensions with the same symbols
    ov::PartialShape pshape_a{-1, 2, 1, {4, 8}, -1, {4, 8}, -1, {1, 8}, {1, 10}, {4, 18}};
    ov::PartialShape pshape_b{-1, 2, {3, 9}, 1, {3, 9}, -1, {1, 9}, -1, {3, 19}, {1, 10}};

    ov::PartialShape expected_shape = {-1, 2, {3, 9}, {4, 8}, {3, 9}, {4, 8}, -1, -1, {3, 19}, {4, 18}};
    auto expected_symbols = set_shape_symbols(expected_shape);

    set_shape_symbols(pshape_a, expected_symbols);
    set_shape_symbols(pshape_b, expected_symbols);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_a);
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_b);
    auto shape_of = make_shared<ov::op::v3::ShapeOf>(target_shape);

    auto op = make_shared<ov::op::v3::Broadcast>(data, shape_of, "BIDIRECTIONAL");

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, broadcast_v3_in0_interval_in1_param_rank_bigger_bidirectional) {
    ov::PartialShape pshape_a{{4, 8}, 1};
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::i32, pshape_a);
    auto target_shape_param = make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{3});
    auto broadcast = make_shared<ov::op::v3::Broadcast>(data, target_shape_param, ov::op::BroadcastType::BIDIRECTIONAL);

    EXPECT_EQ(broadcast->get_output_partial_shape(0), (ov::PartialShape{-1, {4, 8}, -1}));
}

TEST(type_prop, broadcast_v3_in0_interval_in1_param_rank_smaller_bidirectional) {
    ov::PartialShape pshape_a{-1, 2, {1, 10}, {4, 8}, 1};
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::i32, pshape_a);
    auto target_shape_param = make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{3});
    auto broadcast = make_shared<ov::op::v3::Broadcast>(data, target_shape_param, ov::op::BroadcastType::BIDIRECTIONAL);

    EXPECT_EQ(broadcast->get_output_partial_shape(0), (ov::PartialShape{-1, 2, -1, {4, 8}, -1}));
}

TEST(type_prop, broadcast_v3_symbols_in0_dims_in1_param_bidirectional) {
    ov::PartialShape pshape_a{-1, 2, 1, {4, 8}, {1, 10}};

    ov::PartialShape expected_shape{-1, 2, -1, {4, 8}, -1};
    auto expected_symbols = set_shape_symbols(expected_shape);
    set_shape_symbols(pshape_a, expected_symbols);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_a);
    auto target_shape_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{5});
    auto broadcast = make_shared<ov::op::v3::Broadcast>(data, target_shape_param, ov::op::BroadcastType::BIDIRECTIONAL);

    const auto& out_shape = broadcast->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, broadcast_v3_non_broadcastable_dims_numpy) {
    // Numpy mode for v3::Broadcast mode is one directional
    ov::PartialShape pshape_a{{4, 8}, {2, 4}};
    ov::PartialShape pshape_b{{1}, {5, 6}};

    // No validation for non-broadcastable dimensions pair
    ov::PartialShape expected_shape = {1, {5, 6}};

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_a);
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_b);
    auto shape_of = make_shared<ov::op::v3::ShapeOf>(target_shape);

    auto op = make_shared<ov::op::v3::Broadcast>(data, shape_of, "NUMPY");

    const auto out_shape = op->get_output_partial_shape(0);
    EXPECT_EQ(out_shape, expected_shape);
}

TEST(type_prop, broadcast_v3_symbols_in0_dynamic_mixed_dims_numpy) {
    // Numpy mode for v3::Broadcast mode is one directional
    // All dimensions of A have symbols, B without symbols
    ov::PartialShape pshape_a{-1, 2, 1, -1, {4, 8}, -1, {1, 8}, {1, 10}, {4, 18}};
    ov::PartialShape pshape_b{-1, 2, {3, 9}, {4, 10}, -1, {5, 11}, -1, {6, 20}, {1, 10}};

    ov::PartialShape expected_shape = {-1, 2, {3, 9}, {4, 10}, -1, {5, 11}, -1, {6, 20}, {1, 10}};

    set_shape_symbols(pshape_a);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_a);
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_b);
    auto shape_of = make_shared<ov::op::v3::ShapeOf>(target_shape);

    auto op = make_shared<ov::op::v3::Broadcast>(data, shape_of, "NUMPY");

    const auto out_shape = op->get_output_partial_shape(0);
    EXPECT_EQ(out_shape, expected_shape);
    // Output shape is a copy of the target shape value, the `A` symbols are not propagated
    EXPECT_THAT(get_shape_symbols(out_shape), Each(nullptr));
}

TEST(type_prop, broadcast_v3_symbols_in1_dynamic_mixed_dims_numpy) {
    // Numpy mode for v3::Broadcast mode is one directional
    // All dimensions of B have symbols, A without symbols
    ov::PartialShape pshape_a{-1, 2, 1, -1, {4, 8}, -1, {1, 8}, {1, 10}, {4, 18}};
    ov::PartialShape pshape_b{-1, 2, {3, 9}, {4, 10}, -1, {5, 11}, -1, {6, 20}, {1, 10}};

    ov::PartialShape expected_shape = {-1, 2, {3, 9}, {4, 10}, -1, {5, 11}, -1, {6, 20}, {1, 10}};
    // Output shape is a copy of the target shape, `B` symbols are propagated
    auto expected_symbols = set_shape_symbols(expected_shape);
    set_shape_symbols(pshape_b, expected_symbols);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_a);
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_b);
    auto shape_of = make_shared<ov::op::v3::ShapeOf>(target_shape);

    auto op = make_shared<ov::op::v3::Broadcast>(data, shape_of, "NUMPY");

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, broadcast_v3_symbols_both_inputs_dynamic_mixed_dims_numpy) {
    // Numpy mode for v3::Broadcast mode is one directional
    // All dimensions of A and B have symbols
    ov::PartialShape pshape_a{-1, 2, 1, -1, {4, 8}, -1, {1, 8}, {1, 10}, {4, 18}};
    ov::PartialShape pshape_b{-1, 2, {3, 9}, {4, 10}, -1, {5, 11}, -1, {6, 20}, {1, 10}};

    ov::PartialShape expected_shape = {-1, 2, {3, 9}, {4, 10}, -1, {5, 11}, -1, {6, 20}, {1, 10}};
    // Output shape is a copy of the target shape, `B` symbols are propagated

    set_shape_symbols(pshape_a);
    auto expected_symbols = set_shape_symbols(pshape_b);

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_a);
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_b);
    auto shape_of = make_shared<ov::op::v3::ShapeOf>(target_shape);

    auto op = make_shared<ov::op::v3::Broadcast>(data, shape_of, "NUMPY");

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, broadcast_v3_symbols_dynamic_mixed_dims_explicit) {
    ov::PartialShape pshape_a{2, {6, 8}, -1};
    ov::PartialShape pshape_b{2, -1, {6, 8}, -1, 5};

    ov::PartialShape expected_shape = {2, -1, {6, 8}, -1, 5};

    auto expected_symbols = set_shape_symbols(pshape_b);
    auto axis_map =
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{0, 2, 3});

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_a);
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_b);
    auto shape_of = make_shared<ov::op::v3::ShapeOf>(target_shape);

    auto op = make_shared<ov::op::v3::Broadcast>(data, shape_of, axis_map, "EXPLICIT");

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, broadcast_v3_eval_symbols_static_dims_numpy) {
    // Numpy mode for v3::Broadcast mode is one directional
    // All dimensions of A have symbols, B without symbols
    ov::PartialShape pshape_a{1, 1};
    ov::PartialShape pshape_b{2, 3};
    ov::PartialShape pshape_c{1, 3};

    ov::PartialShape expected_shape = {2, 3};

    auto expected_symbols = set_shape_symbols(pshape_b);

    auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_a);
    auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_b);
    auto shape_of_a = make_shared<ov::op::v3::ShapeOf>(a);
    auto shape_of_b = make_shared<ov::op::v3::ShapeOf>(b);

    auto broadcast_a = make_shared<ov::op::v3::Broadcast>(a, shape_of_b, "NUMPY");
    auto shape_of_broadcast_a = make_shared<ov::op::v3::ShapeOf>(broadcast_a);

    auto c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_c);
    auto broadcast_c = make_shared<ov::op::v3::Broadcast>(c, shape_of_broadcast_a, "NUMPY");

    const auto out_shape = broadcast_c->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, broadcast_v3_eval_symbols_static_dims_bidirectional) {
    ov::PartialShape pshape_a{1, 3};
    ov::PartialShape pshape_b{2, 1};
    ov::PartialShape pshape_c{1, 1};

    ov::PartialShape expected_shape = {2, 3};

    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>(), C = std::make_shared<ov::Symbol>(),
         D = std::make_shared<ov::Symbol>();
    auto E = std::make_shared<ov::Symbol>(), F = std::make_shared<ov::Symbol>();
    ov::TensorSymbol expected_symbols{C, B};

    set_shape_symbols(pshape_a, {A, B});
    set_shape_symbols(pshape_b, {C, D});
    set_shape_symbols(pshape_c, {E, F});

    auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_a);
    auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_b);
    auto shape_of_a = make_shared<ov::op::v3::ShapeOf>(a);
    auto shape_of_b = make_shared<ov::op::v3::ShapeOf>(b);

    auto broadcast_a = make_shared<ov::op::v3::Broadcast>(a, shape_of_b, "BIDIRECTIONAL");
    auto shape_of_broadcast_a = make_shared<ov::op::v3::ShapeOf>(broadcast_a);

    auto c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_c);
    auto broadcast_c = make_shared<ov::op::v3::Broadcast>(c, shape_of_broadcast_a, "BIDIRECTIONAL");

    const auto out_shape = broadcast_c->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TEST(type_prop, broadcast_v3_bidirectional_tricky_partial_value_case_and_equal_partial_value_propagation) {
    ov::PartialShape pshape_a{{0, 10}, 1, 4};
    ov::PartialShape pshape_b{{0, 10}, 1};

    ov::PartialShape expected_shape = ov::PartialShape{{0, 10}, 1, 4};

    auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_a);
    auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, pshape_b);
    auto shape_of_b = make_shared<ov::op::v3::ShapeOf>(b);
    auto concat = make_shared<ov::op::v0::Concat>(
        ov::OutputVector{shape_of_b, ov::op::v0::Constant::create(ov::element::i64, {1}, {4})},
        0);
    auto equal =
        make_shared<ov::op::v1::Equal>(concat, ov::op::v0::Constant::create(ov::element::i64, {3}, {-1, -1, -1}));
    auto select =
        make_shared<ov::op::v1::Select>(equal, ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 1, 1}), concat);

    ov::PartialShape shape;
    auto broadcast_a = make_shared<ov::op::v3::Broadcast>(a, select, "BIDIRECTIONAL");
    const auto out_shape = broadcast_a->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    {
        auto constant = ov::util::get_constant_from_source(equal->output(0));
        ASSERT_TRUE(constant != nullptr);
        std::vector<bool> expected{false, false, false}, calculated = constant->get_vector<bool>();
        EXPECT_EQ(calculated, expected);
    }
    {
        equal = make_shared<ov::op::v1::Equal>(concat, ov::op::v0::Constant::create(ov::element::i64, {3}, {5, 1, 4}));
        EXPECT_TRUE(ov::util::get_constant_from_source(equal->output(0)) == nullptr);
    }
    {
        equal = make_shared<ov::op::v1::Equal>(concat, ov::op::v0::Constant::create(ov::element::i64, {3}, {11, 1, 4}));
        auto constant = ov::util::get_constant_from_source(equal->output(0));
        ASSERT_TRUE(constant != nullptr);
        std::vector<bool> expected{false, true, true}, calculated = constant->get_vector<bool>();
        EXPECT_EQ(calculated, expected);
    }
}
