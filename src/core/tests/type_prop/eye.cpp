// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset9.hpp"
#include "type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, eye_constant) {
    auto num_rows = op::v0::Constant::create(element::i64, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i64, Shape{}, {3});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {0});

    auto eye = std::make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::bf16);

    EXPECT_EQ(eye->get_output_element_type(0), element::bf16);
    EXPECT_EQ(eye->get_output_partial_shape(0), ov::PartialShape({6, 3}));
}

TEST(type_prop, eye_batchshape_constant) {
    auto num_rows = op::v0::Constant::create(element::i64, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i64, Shape{}, {3});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {0});
    auto batch_shape = op::v0::Constant::create(element::i64, Shape{1}, {2});

    auto eye = std::make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, batch_shape, element::bf16);

    EXPECT_EQ(eye->get_output_element_type(0), element::bf16);
    EXPECT_EQ(eye->get_output_partial_shape(0), PartialShape({2, 6, 3}));
}

TEST(type_prop, eye_rows_param) {
    auto num_rows = make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    auto num_columns = op::v0::Constant::create(element::i64, Shape{}, {10});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::f32);

    EXPECT_EQ(eye->get_output_element_type(0), element::f32);
    EXPECT_EQ(eye->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 10}));
}

TEST(type_prop, eye_rows_const) {
    auto num_rows = op::v0::Constant::create(element::i64, Shape{}, {10});
    auto num_columns = make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    auto diagonal_index = make_shared<op::v0::Parameter>(element::i64, PartialShape{1});

    auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::f32);

    EXPECT_EQ(eye->get_output_element_type(0), element::f32);
    EXPECT_EQ(eye->get_output_partial_shape(0), PartialShape({10, Dimension::dynamic()}));
}

TEST(type_prop, eye_batchshape_const) {
    auto num_rows = make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    auto num_columns = num_rows;
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {0});
    auto batch_shape = op::v0::Constant::create(element::i64, Shape{2}, {2, 3});

    auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, batch_shape, element::f32);

    EXPECT_EQ(eye->get_output_element_type(0), element::f32);
    EXPECT_EQ(eye->get_output_partial_shape(0), PartialShape({2, 3, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, eye_batchshape_params) {
    auto num_rows = make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    auto num_columns = make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    auto diagonal_index = make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    auto batch_shape = make_shared<op::v0::Parameter>(element::i64, PartialShape{2});

    auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, batch_shape, element::f64);

    EXPECT_EQ(eye->get_output_element_type(0), element::f64);
    EXPECT_EQ(eye->get_output_partial_shape(0), PartialShape().dynamic(4));
}

TEST(type_prop, eye_batchshape_shapeof) {
    auto num_rows = op::v0::Constant::create(element::i64, Shape{}, {10});
    auto num_columns = num_rows;
    auto diagonal_index = make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    auto batch_shape = make_shared<op::v0::Parameter>(element::i64, PartialShape({{1, 10}, {10, 25}}));
    auto shape_of = make_shared<op::v0::ShapeOf>(batch_shape);

    auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, shape_of, element::f64);

    EXPECT_EQ(eye->get_output_element_type(0), element::f64);
    EXPECT_EQ(eye->get_output_partial_shape(0), PartialShape({{1, 10}, {10, 25}, 10, 10}));
}

TEST(type_prop, eye_invalid_num_rows_value) {
    auto num_rows = op::v0::Constant::create(element::i64, Shape{1}, {-6});
    auto num_columns = op::v0::Constant::create(element::i64, Shape{}, {2});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    try {
        auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid num rows value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'num_rows' must be non-negative value. Got: -6"));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_num_columns_value) {
    auto num_rows = op::v0::Constant::create(element::i64, Shape{1}, {6});
    auto num_columns = op::v0::Constant::create(element::i64, Shape{}, {-6});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    try {
        auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid num columns value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'num_columns' must be non-negative value. Got: -6"));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_num_rows_type) {
    auto num_rows = op::v0::Constant::create(element::bf16, Shape{}, {1.2});
    auto num_columns = op::v0::Constant::create(element::i64, Shape{}, {2});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    try {
        auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid num rows value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Type of the 'num_rows' should be int32 or int64. Got: bf16"));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_num_columns_type) {
    auto num_rows = op::v0::Constant::create(element::i64, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::bf16, Shape{}, {6.5});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    try {
        auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid num columns value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Type of the 'num_columns' should be int32 or int64. Got: bf16"));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_diagonal_index_type) {
    auto num_rows = op::v0::Constant::create(element::i64, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i64, Shape{}, {6});
    auto diagonal_index = op::v0::Constant::create(element::bf16, Shape{}, {6.5});

    try {
        auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid diagonal index value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Type of the 'diagonal_index' should be int32 or int64. Got: bf16"));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_num_rows_shape) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{3}, {1, 1, 1});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    try {
        auto Eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid num rows value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'num_columns' value input should have 1 element."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_num_columns_shape) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{3}, {1, 1, 1});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    try {
        auto Eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid num columns value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'num_columns' value input should have 1 element."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_diagonal_index_shape) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto diagonal_index = op::v0::Constant::create(element::i32, Shape{3}, {1, 1, 1});

    try {
        auto Eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid diagonal index value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'diagonal_index' value input should have 1 element."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_num_rows_rank) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{1, 1}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{1}, {1});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    try {
        auto Eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid num rows value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'num_rows' value must be a scalar or 1D tensor."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_num_columns_rank) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{2, 1}, {1, 2});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});

    try {
        auto Eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid num columns value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'num_columns' value must be a scalar or 1D tensor."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_invalid_diagonal_index_rank) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{}, {2});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{1, 2, 1}, {2, 8});

    try {
        auto Eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid diagonal index value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'diagonal_index' value must be a scalar or 1D tensor."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, eye_dynamic_batch_shape_dyn_rank) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{}, {4});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});
    auto batch_shape = make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());

    auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, batch_shape, element::i32);

    EXPECT_EQ(eye->get_output_element_type(0), element::i32);
    EXPECT_EQ(eye->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, eye_dynamic_batch_shape_1D) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{}, {4});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});
    auto batch_shape = make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));

    auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, batch_shape, element::i32);

    EXPECT_EQ(eye->get_output_element_type(0), element::i32);
    EXPECT_EQ(eye->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, eye_dynamic_batch_shape_invalid_rank) {
    auto num_rows = op::v0::Constant::create(element::i32, Shape{}, {6});
    auto num_columns = op::v0::Constant::create(element::i32, Shape{}, {4});
    auto diagonal_index = op::v0::Constant::create(element::i64, Shape{}, {2});
    auto batch_shape = make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(2));

    try {
        auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, batch_shape, element::i32);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid 'batch_shape' value.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'batch_shape' input must be a 1D tensor."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}
