// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------ V5 ------------------------------

TEST(type_prop, gather_nd_slices_from_4d_batch_dims0) {
    Shape params_shape{2, 3, 11, 12};
    Shape indices_shape{2, 3, 2};
    Shape out_shape{2, 3, 11, 12};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G5 = make_shared<op::v5::GatherND>(P, I, 0);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_scalars_from_4d_batch_dims2) {
    Shape params_shape{2, 3, 11, 12};
    Shape indices_shape{2, 3, 2};
    Shape out_shape{6};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G5 = make_shared<op::v5::GatherND>(P, I, 2);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_slices_from_5d_batch_dims2) {
    Shape params_shape{7, 5, 11, 12, 32};
    Shape indices_shape{7, 5, 3, 1};
    Shape out_shape{35, 3, 12, 32};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G5 = make_shared<op::v5::GatherND>(P, I, 2);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_dim2_with_dyn_dim) {
    PartialShape params_shape{7, Dimension::dynamic(), 11, 12, 32};
    Shape indices_shape{7, 5, 3, 1};
    Shape out_shape{35, 3, 12, 32};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G5 = make_shared<op::v5::GatherND>(P, I, 2);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_dim2_with_dyn_dim2) {
    PartialShape params_shape{7, Dimension::dynamic(), Dimension::dynamic(), 12, 32};
    Shape indices_shape{7, 5, 3, 1};
    Shape out_shape{35, 3, 12, 32};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G5 = make_shared<op::v5::GatherND>(P, I, 2);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_dim2_with_dyn_dim3) {
    PartialShape params_shape{7, Dimension::dynamic(), Dimension::dynamic(), 12, Dimension::dynamic()};
    Shape indices_shape{7, 5, 3, 1};
    PartialShape out_shape{35, 3, 12, Dimension::dynamic()};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G5 = make_shared<op::v5::GatherND>(P, I, 2);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_TRUE(G5->get_output_partial_shape(0).same_scheme(out_shape));
}

TEST(type_prop, gather_nd_batch_dim0_with_dyn_ind_dim) {
    PartialShape params_shape{7, Dimension::dynamic(), Dimension::dynamic(), 12, Dimension::dynamic()};
    PartialShape indices_shape{7, 5, 3, Dimension::dynamic()};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G5 = make_shared<op::v5::GatherND>(P, I, 0);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_TRUE(G5->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, gather_nd_fail_batch_dims_greater_indices_rank) {
    Shape params_shape{2, 3, 4, 5};
    Shape indices_shape{2, 1};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    try {
        auto G5 = make_shared<op::v5::GatherND>(P, I, 3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Number of batch dimensions must not exceed a rank of indices."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_fail_unequal_batch_dims) {
    Shape params_shape{2, 3, 4, 5};
    Shape indices_shape{2, 1, 4};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    try {
        auto G5 = make_shared<op::v5::GatherND>(P, I, 2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Batch dimensions of data and indices must be the same."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_fail_indices_tuple_greater_data_rank_batch_dims2) {
    Shape params_shape{2, 1, 4, 5};
    Shape indices_shape{2, 1, 5, 3};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    try {
        auto G5 = make_shared<op::v5::GatherND>(P, I, 2);
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

// ------------------------------ V0 + V5 ------------------------------

TEST(type_prop, gather_nd_scalar_from_2d) {
    Shape params_shape{2, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_1d_from_2d) {
    Shape params_shape{2, 2};
    Shape indices_shape{2, 1};
    Shape out_shape{2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_scalar_from_3d) {
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 3};
    Shape out_shape{2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_1d_from_3d) {
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_2d_from_3d) {
    Shape params_shape{2, 2, 2};
    Shape indices_shape{1, 1};
    Shape out_shape{1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_scalar_from_2d) {
    Shape params_shape{2, 2};
    Shape indices_shape{2, 1, 2};
    Shape out_shape{2, 1};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_1d_from_2d) {
    Shape params_shape{2, 2};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_scalar_from_3d) {
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 2, 3};
    Shape out_shape{2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_1d_from_3d) {
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_2d_from_3d) {
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_fail_params_rank) {
    Shape params_shape{};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    try {
        auto G5 = make_shared<op::v5::GatherND>(P, I);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect params rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data rank must be at least 1."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_fail_indices_rank) {
    Shape params_shape{2, 2, 2};
    Shape indices_shape{};
    Shape out_shape{2, 1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    try {
        auto G5 = make_shared<op::v5::GatherND>(P, I);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Indices rank must be at least 1."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_fail_indices_element_type) {
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::f32, indices_shape);

    try {
        auto G5 = make_shared<op::v5::GatherND>(P, I);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices element type";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("The indices type is expected to be an integer type."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

// ------------------------------ V8 ------------------------------

TEST(type_prop, gather_nd_8_slices_from_4d_batch_dims0) {
    Shape params_shape{2, 3, 11, 12};
    Shape indices_shape{2, 3, 2};
    Shape out_shape{2, 3, 11, 12};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G5 = make_shared<op::v8::GatherND>(P, I, 0);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_8_scalars_from_4d_batch_dims2) {
    Shape params_shape{2, 3, 11, 12};
    Shape indices_shape{2, 3, 2};
    Shape out_shape{2, 3};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G5 = make_shared<op::v8::GatherND>(P, I, 2);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_8_slices_from_5d_batch_dims2) {
    Shape params_shape{7, 5, 11, 12, 32};
    Shape indices_shape{7, 5, 3, 1};
    Shape out_shape{7, 5, 3, 12, 32};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G5 = make_shared<op::v8::GatherND>(P, I, 2);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_8_batch_dim2_with_dyn_dim) {
    PartialShape params_shape{7, Dimension::dynamic(), 11, 12, 32};
    Shape indices_shape{7, 5, 3, 1};
    Shape out_shape{7, 5, 3, 12, 32};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G5 = make_shared<op::v8::GatherND>(P, I, 2);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_8_batch_dim2_with_dyn_dim2) {
    PartialShape params_shape{7, Dimension::dynamic(), Dimension::dynamic(), 12, 32};
    Shape indices_shape{7, 5, 3, 1};
    Shape out_shape{7, 5, 3, 12, 32};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G5 = make_shared<op::v8::GatherND>(P, I, 2);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_EQ(G5->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_8_batch_dim2_with_dyn_dim3) {
    PartialShape params_shape{7, Dimension::dynamic(), Dimension::dynamic(), 12, Dimension::dynamic()};
    Shape indices_shape{7, 5, 3, 1};
    PartialShape out_shape{7, 5, 3, 12, Dimension::dynamic()};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G5 = make_shared<op::v8::GatherND>(P, I, 2);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_TRUE(G5->get_output_partial_shape(0).same_scheme(out_shape));
}

TEST(type_prop, gather_nd_8_batch_dim0_with_dyn_ind_dim) {
    PartialShape params_shape{7, Dimension::dynamic(), Dimension::dynamic(), 12, Dimension::dynamic()};
    PartialShape indices_shape{7, 5, 3, Dimension::dynamic()};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G5 = make_shared<op::v8::GatherND>(P, I, 0);
    ASSERT_EQ(G5->get_element_type(), element::f32);
    ASSERT_TRUE(G5->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, gather_nd_8_fail_batch_dims_greater_indices_rank) {
    Shape params_shape{2, 3, 4, 5};
    Shape indices_shape{2, 1};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    try {
        auto G5 = make_shared<op::v8::GatherND>(P, I, 3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Number of batch dimensions must not exceed a rank of indices."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_8_fail_unequal_batch_dims) {
    Shape params_shape{2, 3, 4, 5};
    Shape indices_shape{2, 1, 4};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    try {
        auto G5 = make_shared<op::v8::GatherND>(P, I, 2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Batch dimensions of data and indices must be the same."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_8_fail_indices_tuple_greater_data_rank_batch_dims2) {
    Shape params_shape{2, 1, 4, 5};
    Shape indices_shape{2, 1, 5, 3};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);

    try {
        auto G5 = make_shared<op::v8::GatherND>(P, I, 2);
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
