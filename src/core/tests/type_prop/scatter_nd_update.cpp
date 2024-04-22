// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_nd_update.hpp"

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"

using namespace ov;
using namespace testing;

template <class T>
class TypePropScatterNDUpdateTest : public TypePropOpTest<T> {
protected:
    void SetUp() override {
        set_shape_symbols(data_3d_dynamic);
    }
    PartialShape data_3d_dynamic{{2, 5}, 2, {4, 10}};
};

TYPED_TEST_SUITE_P(TypePropScatterNDUpdateTest);

TYPED_TEST_P(TypePropScatterNDUpdateTest, scatter_nd_update_v3_fail_indices_element_type) {
    Shape ref_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 2, 1, 4};
    auto R = std::make_shared<op::v0::Parameter>(element::f32, ref_shape);
    auto I = std::make_shared<op::v0::Parameter>(element::f16, indices_shape);
    auto U = std::make_shared<op::v0::Parameter>(element::f32, updates_shape);
    try {
        auto G = this->make_op(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices element type";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Indices element type must be i64 or i32"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(TypePropScatterNDUpdateTest, scatter_nd_update_v3_fail_updates_rank) {
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{1};
    Shape updates_shape{3, 3, 3};
    Shape out_shape{3, 3, 3};
    auto R = std::make_shared<op::v0::Parameter>(element::f32, ref_shape);
    auto I = std::make_shared<op::v0::Parameter>(element::i32, indices_shape);
    auto U = std::make_shared<op::v0::Parameter>(element::f32, updates_shape);
    try {
        auto G = this->make_op(R, I, U);
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

TYPED_TEST_P(TypePropScatterNDUpdateTest, scatter_nd_update_fail_updates_element_type) {
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{1};
    Shape updates_shape{3, 3};
    Shape out_shape{3, 3, 3};
    auto R = std::make_shared<op::v0::Parameter>(element::f32, ref_shape);
    auto I = std::make_shared<op::v0::Parameter>(element::i32, indices_shape);
    auto U = std::make_shared<op::v0::Parameter>(element::i32, updates_shape);
    try {
        auto G = this->make_op(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Created ScatterND op with incorrect updates element type.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Updates element type must be the same as inputs"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(TypePropScatterNDUpdateTest, scatter_nd_update_fail_updates_shape) {
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{1};
    Shape updates_shape{2, 3};
    Shape out_shape{3, 3, 3};
    auto R = std::make_shared<op::v0::Parameter>(element::f32, ref_shape);
    auto I = std::make_shared<op::v0::Parameter>(element::i32, indices_shape);
    auto U = std::make_shared<op::v0::Parameter>(element::f32, updates_shape);
    try {
        auto G = this->make_op(R, I, U);
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

TYPED_TEST_P(TypePropScatterNDUpdateTest, scatter_nd_update_fail_indices_last_dim) {
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{2, 4};
    Shape updates_shape{2, 3, 3};
    Shape out_shape{3, 3, 3};
    auto R = std::make_shared<op::v0::Parameter>(element::f32, ref_shape);
    auto I = std::make_shared<op::v0::Parameter>(element::i32, indices_shape);
    auto U = std::make_shared<op::v0::Parameter>(element::f32, updates_shape);
    try {
        auto G = this->make_op(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices innermost dim";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Last dimension of indices can be at most the rank of inputs"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(TypePropScatterNDUpdateTest, data_input_partial_shape_and_symbols_propagation) {
    const auto d = std::make_shared<op::v0::Parameter>(element::f32, this->data_3d_dynamic);
    const auto i = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto u = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 5});

    const auto op = this->make_op(d, i, u);

    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), this->data_3d_dynamic);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), get_shape_symbols(this->data_3d_dynamic));
}

TYPED_TEST_P(TypePropScatterNDUpdateTest, indicies_input_is_dynamic) {
    const auto d = std::make_shared<op::v0::Parameter>(element::f64, this->data_3d_dynamic);
    const auto i = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto u = std::make_shared<op::v0::Parameter>(element::f64, PartialShape{3, 5});

    const auto op = this->make_op(d, i, u);

    EXPECT_EQ(op->get_output_element_type(0), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), this->data_3d_dynamic);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), get_shape_symbols(this->data_3d_dynamic));
}

TYPED_TEST_P(TypePropScatterNDUpdateTest, updates_input_is_dynamic) {
    const auto d = std::make_shared<op::v0::Parameter>(element::f64, this->data_3d_dynamic);
    const auto i = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto u = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());

    const auto op = this->make_op(d, i, u);

    EXPECT_EQ(op->get_output_element_type(0), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), this->data_3d_dynamic);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), get_shape_symbols(this->data_3d_dynamic));
}

TYPED_TEST_P(TypePropScatterNDUpdateTest, indicies_input_has_interval_dimensions) {
    const auto d = std::make_shared<op::v0::Parameter>(element::i64, this->data_3d_dynamic);
    const auto i = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{{0, 3}, 1});
    const auto u = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{3, 2, {8, 10}});

    const auto op = this->make_op(d, i, u);

    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), this->data_3d_dynamic);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), get_shape_symbols(this->data_3d_dynamic));
}

TYPED_TEST_P(TypePropScatterNDUpdateTest, updates_input_is_scalar) {
    const auto d = std::make_shared<op::v0::Parameter>(element::i8, this->data_3d_dynamic);
    const auto i = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto u = std::make_shared<op::v0::Parameter>(element::i8, PartialShape{});

    const auto op = this->make_op(d, i, u);

    EXPECT_EQ(op->get_output_element_type(0), element::i8);
    EXPECT_EQ(op->get_output_partial_shape(0), this->data_3d_dynamic);
}

TYPED_TEST_P(TypePropScatterNDUpdateTest, default_ctor) {
    const auto d = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{2, 3, 5, 1});
    const auto i = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1, 3});
    const auto u = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1, 1});

    const auto op = this->make_op();
    op->set_arguments(OutputVector{d, i, u});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, 3, 5, 1}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TYPED_TEST_P(TypePropScatterNDUpdateTest, preserve_partial_values_and_symbols_via_evaluates_bounds) {
    const auto d = op::v0::Constant::create(element::i64, Shape{4}, {2, 3, 15, 4});
    const auto i = op::v0::Constant::create(element::i64, Shape{2, 1}, {2, 0});
    auto u_shape = PartialShape{{10, 20}, {3, 4}};
    auto symbols = set_shape_symbols(u_shape);

    const auto shape_of_u =
        std::make_shared<op::v0::ShapeOf>(std::make_shared<op::v0::Parameter>(element::i64, u_shape));
    const auto op = this->make_op(d, i, shape_of_u);

    auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1});
    auto bc = std::make_shared<op::v3::Broadcast>(param, op, op::BroadcastType::BIDIRECTIONAL);

    EXPECT_EQ(bc->get_output_partial_shape(0), PartialShape({{3, 4}, 3, {10, 20}, 4}));
    EXPECT_THAT(get_shape_symbols(bc->get_output_partial_shape(0)),
                ElementsAre(symbols[1], nullptr, symbols[0], nullptr));
}

TYPED_TEST_P(TypePropScatterNDUpdateTest, indices_dynamic_type) {
    const auto d = std::make_shared<op::v0::Parameter>(element::f32, this->data_3d_dynamic);
    const auto i = std::make_shared<op::v0::Parameter>(element::dynamic, PartialShape{3, 2});
    const auto u = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 5});

    const auto op = this->make_op(d, i, u);

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), this->data_3d_dynamic);
}

TYPED_TEST_P(TypePropScatterNDUpdateTest, updates_dynamic_type) {
    const auto d = std::make_shared<op::v0::Parameter>(element::i64, this->data_3d_dynamic);
    const auto i = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto u = std::make_shared<op::v0::Parameter>(element::dynamic, PartialShape{3, 5});

    const auto op = this->make_op(d, i, u);

    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), this->data_3d_dynamic);
}

TYPED_TEST_P(TypePropScatterNDUpdateTest, all_dynamic_type) {
    const auto d = std::make_shared<op::v0::Parameter>(element::dynamic, this->data_3d_dynamic);
    const auto i = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{3, 2});
    const auto u = std::make_shared<op::v0::Parameter>(element::dynamic, PartialShape{3, 5});

    const auto op = this->make_op(d, i, u);

    EXPECT_EQ(op->get_output_element_type(0), element::dynamic);
    EXPECT_EQ(op->get_output_partial_shape(0), this->data_3d_dynamic);
}

TEST(type_prop, scatter_nd_update_v15_default_attribute) {
    const auto d = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{2, 3, 5, 1});
    const auto i = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1, 3});
    const auto u = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1, 1});

    const auto op = std::make_shared<op::v15::ScatterNDUpdate>();
    op->set_arguments(OutputVector{d, i, u});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_reduction(), op::v15::ScatterNDUpdate::Reduction::NONE);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, 3, 5, 1}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, scatter_nd_update_v15_attribute_setter_enum) {
    const auto d = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{2, 3, 5, 1});
    const auto i = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1, 3});
    const auto u = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1, 1});

    const auto op = std::make_shared<op::v15::ScatterNDUpdate>();
    op->set_arguments(OutputVector{d, i, u});
    op->set_reduction(op::v15::ScatterNDUpdate::Reduction::PROD);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_reduction(), op::v15::ScatterNDUpdate::Reduction::PROD);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, 3, 5, 1}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, scatter_nd_update_v15_attribute_constructor) {
    const auto d = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{2, 3, 5, 1});
    const auto i = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1, 3});
    const auto u = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1, 1});

    const auto op = std::make_shared<op::v15::ScatterNDUpdate>(d, i, u, op::v15::ScatterNDUpdate::Reduction::MAX);

    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_reduction(), op::v15::ScatterNDUpdate::Reduction::MAX);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, 3, 5, 1}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

REGISTER_TYPED_TEST_SUITE_P(TypePropScatterNDUpdateTest,
                            default_ctor,
                            indices_dynamic_type,
                            indicies_input_has_interval_dimensions,
                            data_input_partial_shape_and_symbols_propagation,
                            indicies_input_is_dynamic,
                            preserve_partial_values_and_symbols_via_evaluates_bounds,
                            scatter_nd_update_fail_indices_last_dim,
                            scatter_nd_update_fail_updates_element_type,
                            scatter_nd_update_fail_updates_shape,
                            scatter_nd_update_v3_fail_indices_element_type,
                            scatter_nd_update_v3_fail_updates_rank,
                            updates_dynamic_type,
                            updates_input_is_dynamic,
                            updates_input_is_scalar,
                            all_dynamic_type);
using OpVersions = ::testing::Types<op::v3::ScatterNDUpdate, op::v15::ScatterNDUpdate>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, TypePropScatterNDUpdateTest, OpVersions);
