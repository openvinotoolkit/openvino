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

TEST(type_prop, batchmatmultranspose_deduce_3d)
{
    // Deduce type for matrix/matrix arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{5, 2, 3});
    auto bc = make_shared<op::BatchMatMulTranspose>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{5, 4, 3}));
}

TEST(type_prop, batchmatmultranspose_deduce_3d_transpose0)
{
    // Deduce type for matrix/matrix arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 2, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{5, 2, 3});
    auto bc = make_shared<op::BatchMatMulTranspose>(param1, param2, true, false);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{5, 4, 3}));
}

TEST(type_prop, batchmatmultranspose_deduce_3d_transpose1)
{
    // Deduce type for matrix/matrix arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{5, 3, 2});
    auto bc = make_shared<op::BatchMatMulTranspose>(param1, param2, false, true);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{5, 4, 3}));
}

TEST(type_prop, batchmatmultranspose_deduce_3d_transpose_both)
{
    // Deduce type for matrix/matrix arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 2, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{5, 3, 2});
    auto bc = make_shared<op::BatchMatMulTranspose>(param1, param2, true, true);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{5, 4, 3}));
}

TEST(type_prop, batchmatmultranspose_deduce_left_rank_wrong)
{
    // Type deduction fails due to element type mismatch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 5, 4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{5, 2, 5});
    try
    {
        auto bc = make_shared<op::BatchMatMulTranspose>(param1, param2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Element type mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("shape must have rank 3"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchmatmultranspose_deduce_right_rank_wrong)
{
    // Type deduction fails due to element type mismatch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 5, 2, 5});
    try
    {
        auto bc = make_shared<op::BatchMatMulTranspose>(param1, param2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Element type mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("shape must have rank 3"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchmatmultranspose_deduce_element_type_mismatch)
{
    // Type deduction fails due to element type mismatch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 4, 2});
    auto param2 = make_shared<op::Parameter>(element::i32, Shape{5, 2, 5});
    try
    {
        auto bc = make_shared<op::BatchMatMulTranspose>(param1, param2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Element type mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("compatible element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchmatmultranspose_deduce_reduction_axes_size_mismatch)
{
    // Type deduction fails due to reduction axes size mismatch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{6, 3, 5});
    try
    {
        auto bc = make_shared<op::BatchMatMulTranspose>(param1, param2);
        // Should have thrown, so fail if it didn't
        FAIL() << "BatchMatMulTranspose reduction axes size mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Product dimensions are not equal"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchmatmultranspose_partial_both_rank_dynamic_implicit)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto d = make_shared<op::BatchMatMulTranspose>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().same_scheme(3));
}

TEST(type_prop, batchmatmultranspose_partial_left_rank_dynamic_right_rank_static_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto d = make_shared<op::BatchMatMulTranspose>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().same_scheme(3));
}

TEST(type_prop, batchmatmultranspose_partial_left_rank_static_dynamic_right_rank_dynamic)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto d = make_shared<op::BatchMatMulTranspose>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().same_scheme(3));
}

TEST(type_prop, batchmatmultranspose_partial_left_rank_static_dynamic_right_rank_static)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape{3, 4, 5});
    auto d = make_shared<op::BatchMatMulTranspose>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).same_scheme(PartialShape{3, 2, 5}));
}

TEST(type_prop, batchmatmultranspose_partial_left_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto d = make_shared<op::BatchMatMulTranspose>(param0, param1);

    ASSERT_EQ(d->get_output_element_type(0), element::f32);
}

TEST(type_prop, batchmatmultranspose_partial_right_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto d = make_shared<op::BatchMatMulTranspose>(param0, param1);

    ASSERT_EQ(d->get_output_element_type(0), element::i32);
}

TEST(type_prop, batchmatmultranspose_partial_both_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto d = make_shared<op::BatchMatMulTranspose>(param0, param1);

    ASSERT_EQ(d->get_output_element_type(0), element::dynamic);
}
