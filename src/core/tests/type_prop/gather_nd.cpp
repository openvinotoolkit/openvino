// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/ops.hpp"

using namespace std;
using namespace ov;
using namespace op;
using namespace testing;

template <typename T>
class gather_nd_type_prop : public TypePropOpTest<T> {};
TYPED_TEST_SUITE_P(gather_nd_type_prop);

// ------------------------------ V5 & V8 ----------------------------------------
// Output shape for V5 and V8 is the same, when batch_dims attribute is equal to 1

TYPED_TEST_P(gather_nd_type_prop, default_ctor) {
    PartialShape data_shape{8, 3, 11, 12};
    PartialShape indices_shape{8, 4, 2};
    PartialShape expected_shape{8, 4, 12};

    auto op = this->make_op();

    constexpr auto batch_dims = 1;
    op->set_batch_dims(batch_dims);
    EXPECT_EQ(op->get_batch_dims(), batch_dims);

    auto data_param = std::make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = std::make_shared<v0::Parameter>(element::i32, indices_shape);

    op->set_argument(0, data_param);
    op->set_argument(1, indices_param);

    op->validate_and_infer_types();

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_shape);
}

TYPED_TEST_P(gather_nd_type_prop, static_shape_batch_dims_1_ind_tuple_2) {
    PartialShape data_shape{8, 3, 11, 12};
    PartialShape indices_shape{8, 4, 2};
    PartialShape expected_shape{8, 4, 12};

    constexpr auto batch_dims = 1;
    auto data_param = std::make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = std::make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = this->make_op(data_param, indices_param, batch_dims);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_shape);
}

TYPED_TEST_P(gather_nd_type_prop, static_shape_batch_dims_1_ind_tuple_3) {
    PartialShape data_shape{8, 3, 11, 12};
    PartialShape indices_shape{8, 4, 3};
    PartialShape expected_shape{8, 4};

    constexpr auto batch_dims = 1;
    auto data_param = std::make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = std::make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = this->make_op(data_param, indices_param, batch_dims);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_shape);
}

TYPED_TEST_P(gather_nd_type_prop, static_shape_batch_dims_1_ind_tuple_dynamic) {
    PartialShape data_shape{8, 3, 11, 12};
    PartialShape indices_shape{8, 4, -1};
    PartialShape expected_shape = PartialShape::dynamic();

    constexpr auto batch_dims = 1;
    auto data_param = std::make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = std::make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = this->make_op(data_param, indices_param, batch_dims);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_shape);
}

TYPED_TEST_P(gather_nd_type_prop, interval_both_symboled_batch_dims_1_ind_tuple_2) {
    PartialShape data_shape{{2, 6}, {3, 7}, {8, 10}, {12, 14}};
    auto data_symbols = set_shape_symbols(data_shape);

    PartialShape indices_shape{{4, 8}, {6, 10}, 2};
    auto indices_symbols = set_shape_symbols(indices_shape);

    PartialShape expected_shape{{4, 6}, {6, 10}, {12, 14}};

    constexpr auto batch_dims = 1;
    auto data_param = std::make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = std::make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = this->make_op(data_param, indices_param, batch_dims);

    const auto& out_shape = op->get_output_partial_shape(0);
    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_THAT(get_shape_symbols(out_shape), ElementsAre(data_symbols[0], indices_symbols[1], data_symbols[3]));
}

TYPED_TEST_P(gather_nd_type_prop, interval_data_symboled_batch_dims_1_ind_tuple_2) {
    PartialShape data_shape{{2, 6}, {3, 7}, {8, 10}, {12, 14}};
    auto symbols = set_shape_symbols(data_shape);

    PartialShape indices_shape{{4, 8}, {6, 10}, 2};
    PartialShape expected_shape{{4, 6}, {6, 10}, {12, 14}};

    constexpr auto batch_dims = 1;
    auto data_param = std::make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = std::make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = this->make_op(data_param, indices_param, batch_dims);

    const auto& out_shape = op->get_output_partial_shape(0);
    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_THAT(get_shape_symbols(out_shape), ElementsAre(symbols[0], nullptr, symbols[3]));
}

TYPED_TEST_P(gather_nd_type_prop, interval_indices_symboled_batch_dims_1_ind_tuple_2) {
    PartialShape data_shape{{2, 6}, {3, 7}, {8, 10}, {12, 14}};
    PartialShape indices_shape{{4, 8}, {6, 10}, 2};
    auto symbols = set_shape_symbols(indices_shape);

    PartialShape expected_shape{{4, 6}, {6, 10}, {12, 14}};

    constexpr auto batch_dims = 1;
    auto data_param = std::make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = std::make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = this->make_op(data_param, indices_param, batch_dims);

    const auto& out_shape = op->get_output_partial_shape(0);
    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_THAT(get_shape_symbols(out_shape), ElementsAre(symbols[0], symbols[1], nullptr));
}

REGISTER_TYPED_TEST_SUITE_P(gather_nd_type_prop,
                            default_ctor,
                            static_shape_batch_dims_1_ind_tuple_2,
                            static_shape_batch_dims_1_ind_tuple_3,
                            static_shape_batch_dims_1_ind_tuple_dynamic,
                            interval_both_symboled_batch_dims_1_ind_tuple_2,
                            interval_data_symboled_batch_dims_1_ind_tuple_2,
                            interval_indices_symboled_batch_dims_1_ind_tuple_2);

typedef Types<v5::GatherND, v8::GatherND> GatherNDTypes;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, gather_nd_type_prop, GatherNDTypes);

// ------------------------------ V5 ------------------------------

TEST(type_prop, gather_nd_v5_slices_from_4d_batch_dims0) {
    Shape data_shape{2, 3, 11, 12};
    Shape indices_shape{2, 3, 2};
    Shape expected_shape{2, 3, 11, 12};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);
    auto op = make_shared<v5::GatherND>(data_param, indices_param, 0);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v5_scalars_from_4d_batch_dims2) {
    Shape data_shape{2, 3, 11, 12};
    Shape indices_shape{2, 3, 2};
    Shape expected_shape{6};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);
    auto op = make_shared<v5::GatherND>(data_param, indices_param, 2);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v5_slices_from_5d_batch_dims2) {
    Shape data_shape{7, 5, 11, 12, 32};
    Shape indices_shape{7, 5, 3, 1};
    Shape expected_shape{35, 3, 12, 32};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);
    auto op = make_shared<v5::GatherND>(data_param, indices_param, 2);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v5_batch_dim2_with_dyn_dim) {
    PartialShape data_shape{7, Dimension::dynamic(), 11, 12, 32};
    Shape indices_shape{7, 5, 3, 1};
    Shape expected_shape{35, 3, 12, 32};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);
    auto op = make_shared<v5::GatherND>(data_param, indices_param, 2);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v5_batch_dim2_with_dyn_dim2) {
    PartialShape data_shape{7, Dimension::dynamic(), Dimension::dynamic(), 12, 32};
    Shape indices_shape{7, 5, 3, 1};
    Shape expected_shape{35, 3, 12, 32};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);
    auto op = make_shared<v5::GatherND>(data_param, indices_param, 2);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v5_batch_dim2_with_dyn_dim3) {
    PartialShape data_shape{7, Dimension::dynamic(), Dimension::dynamic(), 12, Dimension::dynamic()};
    Shape indices_shape{7, 5, 3, 1};
    PartialShape expected_shape{35, 3, 12, Dimension::dynamic()};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);
    auto op = make_shared<v5::GatherND>(data_param, indices_param, 2);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(expected_shape));
}

TEST(type_prop, gather_nd_v5_batch_dim0_with_dyn_ind_dim) {
    PartialShape data_shape{7, Dimension::dynamic(), Dimension::dynamic(), 12, Dimension::dynamic()};
    PartialShape indices_shape{7, 5, 3, Dimension::dynamic()};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);
    auto op = make_shared<v5::GatherND>(data_param, indices_param, 0);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, gather_nd_v5_fail_batch_dims_greater_indices_rank) {
    Shape data_shape{2, 3, 4, 5};
    Shape indices_shape{2, 1};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    try {
        auto op = make_shared<v5::GatherND>(data_param, indices_param, 3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Number of batch dimensions must not exceed a rank of indices."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_v5_fail_unequal_batch_dims) {
    Shape data_shape{2, 3, 4, 5};
    Shape indices_shape{2, 1, 2};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    try {
        auto op = make_shared<v5::GatherND>(data_param, indices_param, 2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Batch dimensions of data and indices must be the same."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_v5_fail_indices_tuple_greater_data_rank_batch_dims2) {
    Shape data_shape{2, 1, 4, 5};
    Shape indices_shape{2, 1, 5, 3};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    try {
        auto op = make_shared<v5::GatherND>(data_param, indices_param, 2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Length of a tuple with indices must not exceed a rank of "
                                         "data tensor excluding batch dimensions."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_v5_scalar_from_2d) {
    Shape data_shape{2, 2};
    Shape indices_shape{2, 2};
    Shape expected_shape{2};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = make_shared<v5::GatherND>(data_param, indices_param);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v5_1d_from_2d) {
    Shape data_shape{2, 2};
    Shape indices_shape{2, 1};
    Shape expected_shape{2, 2};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = make_shared<v5::GatherND>(data_param, indices_param);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v5_scalar_from_3d) {
    Shape data_shape{2, 2, 2};
    Shape indices_shape{2, 3};
    Shape expected_shape{2};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = make_shared<v5::GatherND>(data_param, indices_param);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v5_1d_from_3d) {
    Shape data_shape{2, 2, 2};
    Shape indices_shape{2, 2};
    Shape expected_shape{2, 2};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = make_shared<v5::GatherND>(data_param, indices_param);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v5_2d_from_3d) {
    Shape data_shape{2, 2, 2};
    Shape indices_shape{1, 1};
    Shape expected_shape{1, 2, 2};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = make_shared<v5::GatherND>(data_param, indices_param);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v5_batch_scalar_from_2d) {
    Shape data_shape{2, 2};
    Shape indices_shape{2, 1, 2};
    Shape expected_shape{2, 1};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = make_shared<v5::GatherND>(data_param, indices_param);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v5_batch_1d_from_2d) {
    Shape data_shape{2, 2};
    Shape indices_shape{2, 1, 1};
    Shape expected_shape{2, 1, 2};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = make_shared<v5::GatherND>(data_param, indices_param);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v5_batch_scalar_from_3d) {
    Shape data_shape{2, 2, 2};
    Shape indices_shape{2, 2, 3};
    Shape expected_shape{2, 2};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = make_shared<v5::GatherND>(data_param, indices_param);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v5_batch_1d_from_3d) {
    Shape data_shape{2, 2, 2};
    Shape indices_shape{2, 2, 2};
    Shape expected_shape{2, 2, 2};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = make_shared<v5::GatherND>(data_param, indices_param);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v5_batch_2d_from_3d) {
    Shape data_shape{2, 2, 2};
    Shape indices_shape{2, 1, 1};
    Shape expected_shape{2, 1, 2, 2};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = make_shared<v5::GatherND>(data_param, indices_param);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v5_interval_both_symboled_batch_dims_2_ind_tuple_2) {
    PartialShape data_shape{{2, 6}, {3, 7}, {8, 10}, {12, 14}};
    set_shape_symbols(data_shape);

    PartialShape indices_shape{{4, 8}, {6, 10}, 2};
    set_shape_symbols(indices_shape);

    PartialShape expected_shape{{24, 42}};

    constexpr auto batch_dims = 2;
    auto data_param = std::make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = std::make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = make_shared<v5::GatherND>(data_param, indices_param, batch_dims);

    const auto& out_shape = op->get_output_partial_shape(0);
    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_THAT(get_shape_symbols(out_shape), ElementsAre(nullptr));
}

TEST(type_prop, gather_nd_v5_interval_both_symboled_batch_dims_2_ind_tuple_1) {
    PartialShape data_shape{{2, 6}, {3, 7}, {8, 10}, {12, 14}};
    auto symbols = set_shape_symbols(data_shape);

    PartialShape indices_shape{{4, 8}, {6, 10}, 1};
    set_shape_symbols(indices_shape);

    PartialShape expected_shape{{24, 42}, {12, 14}};

    constexpr auto batch_dims = 2;
    auto data_param = std::make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = std::make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = make_shared<v5::GatherND>(data_param, indices_param, batch_dims);

    const auto& out_shape = op->get_output_partial_shape(0);
    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_THAT(get_shape_symbols(out_shape), ElementsAre(nullptr, symbols[3]));
}

TEST(type_prop, gather_nd_v5_fail_params_rank) {
    Shape data_shape{};
    Shape indices_shape{2, 1, 1};
    Shape expected_shape{2, 1, 2, 2};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    try {
        auto op = make_shared<v5::GatherND>(data_param, indices_param);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect params rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data rank must be at least 1."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_v5_fail_indices_rank) {
    Shape data_shape{2, 2, 2};
    Shape indices_shape{};
    Shape expected_shape{2, 1, 2, 2};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    try {
        auto op = make_shared<v5::GatherND>(data_param, indices_param);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Indices rank must be at least 1."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_v5_fail_indices_element_type) {
    Shape data_shape{2, 2, 2};
    Shape indices_shape{2, 1, 1};
    Shape expected_shape{2, 1, 2, 2};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::f32, indices_shape);

    try {
        auto op = make_shared<v5::GatherND>(data_param, indices_param);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices element type";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("The indices type is expected to be an integer type."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

// ------------------------------ V8 ------------------------------

TEST(type_prop, gather_nd_v8_slices_from_4d_batch_dims0) {
    Shape data_shape{2, 3, 11, 12};
    Shape indices_shape{2, 3, 2};
    Shape expected_shape{2, 3, 11, 12};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);
    auto op = make_shared<v8::GatherND>(data_param, indices_param, 0);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v8_scalars_from_4d_batch_dims2) {
    Shape data_shape{2, 3, 11, 12};
    Shape indices_shape{2, 3, 2};
    Shape expected_shape{2, 3};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);
    auto op = make_shared<v8::GatherND>(data_param, indices_param, 2);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v8_slices_from_5d_batch_dims2) {
    Shape data_shape{7, 5, 11, 12, 32};
    Shape indices_shape{7, 5, 3, 1};
    Shape expected_shape{7, 5, 3, 12, 32};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);
    auto op = make_shared<v8::GatherND>(data_param, indices_param, 2);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v8_batch_dim2_with_dyn_dim) {
    PartialShape data_shape{7, Dimension::dynamic(), 11, 12, 32};
    Shape indices_shape{7, 5, 3, 1};
    Shape expected_shape{7, 5, 3, 12, 32};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);
    auto op = make_shared<v8::GatherND>(data_param, indices_param, 2);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v8_batch_dim2_with_dyn_dim2) {
    PartialShape data_shape{7, Dimension::dynamic(), Dimension::dynamic(), 12, 32};
    Shape indices_shape{7, 5, 3, 1};
    Shape expected_shape{7, 5, 3, 12, 32};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);
    auto op = make_shared<v8::GatherND>(data_param, indices_param, 2);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), expected_shape);
}

TEST(type_prop, gather_nd_v8_batch_dim2_with_dyn_dim3) {
    PartialShape data_shape{7, Dimension::dynamic(), Dimension::dynamic(), 12, Dimension::dynamic()};
    Shape indices_shape{7, 5, 3, 1};
    PartialShape expected_shape{7, 5, 3, 12, Dimension::dynamic()};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);
    auto op = make_shared<v8::GatherND>(data_param, indices_param, 2);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(expected_shape));
}

TEST(type_prop, gather_nd_v8_batch_dim0_with_dyn_ind_dim) {
    PartialShape data_shape{7, Dimension::dynamic(), Dimension::dynamic(), 12, Dimension::dynamic()};
    PartialShape indices_shape{7, 5, 3, Dimension::dynamic()};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);
    auto op = make_shared<v8::GatherND>(data_param, indices_param, 0);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, gather_nd_v8_interval_both_symboled_batch_dims_2_ind_tuple_2) {
    PartialShape data_shape{{2, 6}, {3, 7}, {8, 10}, {12, 14}};
    auto symbols = set_shape_symbols(data_shape);

    PartialShape indices_shape{{4, 8}, {6, 10}, 2};
    set_shape_symbols(indices_shape);

    PartialShape expected_shape{{4, 6}, {6, 7}};

    constexpr auto batch_dims = 2;
    auto data_param = std::make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = std::make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = make_shared<v8::GatherND>(data_param, indices_param, batch_dims);

    const auto& out_shape = op->get_output_partial_shape(0);
    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_THAT(get_shape_symbols(out_shape), ElementsAre(symbols[0], symbols[1]));
}

TEST(type_prop, gather_nd_v8_interval_both_symboled_batch_dims_2_ind_tuple_1) {
    PartialShape data_shape{{2, 6}, {3, 7}, {8, 10}, {12, 14}};
    auto data_symbols = set_shape_symbols(data_shape);

    PartialShape indices_shape{{4, 8}, {6, 10}, 1};
    auto indices_symbols = set_shape_symbols(indices_shape);

    PartialShape expected_shape{{4, 6}, {6, 7}, {12, 14}};

    constexpr auto batch_dims = 2;
    auto data_param = std::make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = std::make_shared<v0::Parameter>(element::i32, indices_shape);

    auto op = make_shared<v8::GatherND>(data_param, indices_param, batch_dims);

    const auto& out_shape = op->get_output_partial_shape(0);
    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_THAT(get_shape_symbols(out_shape), ElementsAre(data_symbols[0], data_symbols[1], data_symbols[3]));
}

TEST(type_prop, gather_nd_v8_fail_batch_dims_greater_indices_rank) {
    Shape data_shape{2, 3, 4, 5};
    Shape indices_shape{2, 1};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    try {
        auto op = make_shared<v8::GatherND>(data_param, indices_param, 3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Number of batch dimensions must not exceed a rank of indices."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_v8_fail_unequal_batch_dims) {
    Shape data_shape{2, 3, 4, 5};
    Shape indices_shape{2, 1, 2};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    try {
        auto op = make_shared<v8::GatherND>(data_param, indices_param, 2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Batch dimensions of data and indices must be the same."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_v8_fail_indices_tuple_greater_data_rank_batch_dims2) {
    Shape data_shape{2, 1, 4, 5};
    Shape indices_shape{2, 1, 5, 3};
    auto data_param = make_shared<v0::Parameter>(element::f32, data_shape);
    auto indices_param = make_shared<v0::Parameter>(element::i32, indices_shape);

    try {
        auto op = make_shared<v8::GatherND>(data_param, indices_param, 2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Length of a tuple with indices must not exceed a rank of "
                                         "data tensor excluding batch dimensions."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
