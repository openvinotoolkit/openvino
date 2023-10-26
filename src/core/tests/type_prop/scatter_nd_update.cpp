// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_nd_update.hpp"

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, scatter_nd_update_v3_fail_indices_element_type) {
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = make_shared<ov::op::v0::Parameter>(element::f32, ref_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::f16, indices_shape);
    auto U = make_shared<ov::op::v0::Parameter>(element::f32, updates_shape);
    try {
        auto G = make_shared<op::v3::ScatterNDUpdate>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices element type";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Indices element type must be i64 or i32"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_nd_update_v3_fail_updates_rank) {
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{1};
    Shape updates_shape{3, 3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<ov::op::v0::Parameter>(element::f32, ref_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i32, indices_shape);
    auto U = make_shared<ov::op::v0::Parameter>(element::f32, updates_shape);
    try {
        auto G = make_shared<op::v3::ScatterNDUpdate>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Rank of updates must be rank of inputs + rank of indices "
                                         "- last dimension of indices - 1"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_nd_update_fail_updates_element_type) {
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{1};
    Shape updates_shape{3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<ov::op::v0::Parameter>(element::f32, ref_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i32, indices_shape);
    auto U = make_shared<ov::op::v0::Parameter>(element::i32, updates_shape);
    try {
        auto G = make_shared<op::v3::ScatterNDUpdate>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Created ScatterND op with incorrect updates element type.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Updates element type must be the same as inputs"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_nd_update_fail_updates_shape) {
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{1};
    Shape updates_shape{2, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<ov::op::v0::Parameter>(element::f32, ref_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i32, indices_shape);
    auto U = make_shared<ov::op::v0::Parameter>(element::f32, updates_shape);
    try {
        auto G = make_shared<op::v3::ScatterNDUpdate>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates shape";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("updates_shape[indices_rank-1:] shape must be input_shape[indices_shape[-1]:]"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_nd_update_fail_indices_last_dim) {
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{2, 4};
    Shape updates_shape{2, 3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<ov::op::v0::Parameter>(element::f32, ref_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i32, indices_shape);
    auto U = make_shared<ov::op::v0::Parameter>(element::f32, updates_shape);
    try {
        auto G = make_shared<op::v3::ScatterNDUpdate>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices innermost dim";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Last dimension of indices can be at most the rank of inputs"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

using namespace ov::opset10;
using namespace testing;

class TypePropScatterUpdateNDV3Test : public TypePropOpTest<op::v3::ScatterNDUpdate> {
protected:
    void SetUp() override {
        set_shape_labels(data_3d_dynamic, 10);
    }
    PartialShape data_3d_dynamic{{2, 5}, 2, {4, 10}};
};

TEST_F(TypePropScatterUpdateNDV3Test, data_input_partial_shape_and_labels_propagation) {
    const auto d = std::make_shared<Parameter>(element::f32, data_3d_dynamic);
    const auto i = std::make_shared<Parameter>(element::i32, PartialShape{3, 2});
    const auto u = std::make_shared<Parameter>(element::f32, PartialShape{3, 5});

    const auto op = make_op(d, i, u);

    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), data_3d_dynamic);
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), ElementsAre(10, 11, 12));
}

TEST_F(TypePropScatterUpdateNDV3Test, indicies_input_is_dynamic) {
    const auto d = std::make_shared<Parameter>(element::f64, data_3d_dynamic);
    const auto i = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto u = std::make_shared<Parameter>(element::f64, PartialShape{3, 5});

    const auto op = make_op(d, i, u);

    EXPECT_EQ(op->get_output_element_type(0), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), data_3d_dynamic);
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), ElementsAre(10, 11, 12));
}

TEST_F(TypePropScatterUpdateNDV3Test, updates_input_is_dynamic) {
    const auto d = std::make_shared<Parameter>(element::f64, data_3d_dynamic);
    const auto i = std::make_shared<Parameter>(element::i32, PartialShape{3, 2});
    const auto u = std::make_shared<Parameter>(element::f64, PartialShape::dynamic());

    const auto op = make_op(d, i, u);

    EXPECT_EQ(op->get_output_element_type(0), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), data_3d_dynamic);
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), ElementsAre(10, 11, 12));
}

TEST_F(TypePropScatterUpdateNDV3Test, indicies_input_has_interval_dimensions) {
    const auto d = std::make_shared<Parameter>(element::i64, data_3d_dynamic);
    const auto i = std::make_shared<Parameter>(element::i32, PartialShape{{0, 3}, 1});
    const auto u = std::make_shared<Parameter>(element::i64, PartialShape{3, 2, {8, 10}});

    const auto op = make_op(d, i, u);

    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), data_3d_dynamic);
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), ElementsAre(10, 11, 12));
}

TEST_F(TypePropScatterUpdateNDV3Test, updates_input_is_scalar) {
    const auto d = std::make_shared<Parameter>(element::i8, data_3d_dynamic);
    const auto i = std::make_shared<Parameter>(element::i32, PartialShape{3});
    const auto u = std::make_shared<Parameter>(element::i8, PartialShape{});

    const auto op = make_op(d, i, u);

    EXPECT_EQ(op->get_output_element_type(0), element::i8);
    EXPECT_EQ(op->get_output_partial_shape(0), data_3d_dynamic);
}

TEST_F(TypePropScatterUpdateNDV3Test, default_ctor) {
    const auto d = std::make_shared<Parameter>(element::i64, PartialShape{2, 3, 5, 1});
    const auto i = std::make_shared<Parameter>(element::i32, PartialShape{1, 3});
    const auto u = std::make_shared<Parameter>(element::i64, PartialShape{1, 1});

    const auto op = make_op();
    op->set_arguments(OutputVector{d, i, u});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, 3, 5, 1}));
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), Each(ov::no_label));
}

TEST_F(TypePropScatterUpdateNDV3Test, preserve_partial_values_and_labels_via_evaluates_bounds) {
    const auto d = Constant::create(element::i64, Shape{4}, {2, 3, 15, 4});
    const auto i = Constant::create(element::i64, Shape{2, 1}, {2, 0});
    auto u_shape = PartialShape{{10, 20}, {3, 4}};
    set_shape_labels(u_shape, 20);

    const auto shape_of_u = std::make_shared<op::v0::ShapeOf>(std::make_shared<Parameter>(element::i64, u_shape));
    const auto op = make_op(d, i, shape_of_u);

    auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1});
    auto bc = std::make_shared<op::v3::Broadcast>(param, op, op::BroadcastType::BIDIRECTIONAL);

    EXPECT_EQ(bc->get_output_partial_shape(0), PartialShape({{3, 4}, 3, {10, 20}, 4}));
    EXPECT_THAT(get_shape_labels(bc->get_output_partial_shape(0)), ElementsAre(21, ov::no_label, 20, ov::no_label));
}

TEST_F(TypePropScatterUpdateNDV3Test, indices_dynamic_type) {
    const auto d = std::make_shared<Parameter>(element::f32, data_3d_dynamic);
    const auto i = std::make_shared<Parameter>(element::dynamic, PartialShape{3, 2});
    const auto u = std::make_shared<Parameter>(element::f32, PartialShape{3, 5});

    const auto op = make_op(d, i, u);

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), data_3d_dynamic);
}

TEST_F(TypePropScatterUpdateNDV3Test, updates_dynamic_type) {
    const auto d = std::make_shared<Parameter>(element::i64, data_3d_dynamic);
    const auto i = std::make_shared<Parameter>(element::i32, PartialShape{3, 2});
    const auto u = std::make_shared<Parameter>(element::dynamic, PartialShape{3, 5});

    const auto op = make_op(d, i, u);

    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), data_3d_dynamic);
}

TEST_F(TypePropScatterUpdateNDV3Test, all_dynamic_type) {
    const auto d = std::make_shared<Parameter>(element::dynamic, data_3d_dynamic);
    const auto i = std::make_shared<Parameter>(element::i64, PartialShape{3, 2});
    const auto u = std::make_shared<Parameter>(element::dynamic, PartialShape{3, 5});

    const auto op = make_op(d, i, u);

    EXPECT_EQ(op->get_output_element_type(0), element::dynamic);
    EXPECT_EQ(op->get_output_partial_shape(0), data_3d_dynamic);
}
