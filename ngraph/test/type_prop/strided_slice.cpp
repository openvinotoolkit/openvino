//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <memory>

using namespace std;
using namespace ngraph;

TEST(type_prop, strided_slice_begin_incorrect_type)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::f16, Shape{4});
    auto end = make_shared<op::Parameter>(element::i64, Shape{4});
    try
    {
        auto strided_slice = make_shared<op::v1::StridedSlice>(
            data, begin, end, vector<int64_t>{1, 0, 1, 0}, vector<int64_t>{1, 0, 1, 0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect begin type exception not thrown.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Begin mask must be an integral number"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, strided_slice_end_incorrect_type)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::i64, Shape{4});
    auto end = make_shared<op::Parameter>(element::boolean, Shape{4});
    try
    {
        auto strided_slice = make_shared<op::v1::StridedSlice>(
            data, begin, end, vector<int64_t>{1, 0, 1, 0}, vector<int64_t>{1, 0, 1, 0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect end type exception not thrown.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("End mask must be an integral number"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, strided_slice_incompatible_size_of_masks_attr)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::i64, Shape{4});
    auto end = make_shared<op::Parameter>(element::i64, Shape{4});
    try
    {
        auto strided_slice = make_shared<op::v1::StridedSlice>(data,
                                                               begin,
                                                               end,
                                                               vector<int64_t>{1, 0, 1, 0},
                                                               vector<int64_t>{1, 0, 1, 0},
                                                               vector<int64_t>{1, 0, 1, 0, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible size od masks exception not thrown.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("All masks of StridedSlice must have the same size"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, strided_slice_mask_incorrect_value)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::i64, Shape{4, 5});
    auto end = make_shared<op::Parameter>(element::i64, Shape{4});
    try
    {
        auto strided_slice = make_shared<op::v1::StridedSlice>(
            data, begin, end, vector<int64_t>{1, 0, 1, 0}, vector<int64_t>{1, 0, 1, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect values of StridedSlice mask exception not thrown.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("All masks of StridedSlice must have be 0 or 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, strided_slice_begin_incorrect_shape)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::i64, Shape{4, 5});
    auto end = make_shared<op::Parameter>(element::i64, Shape{4});
    try
    {
        auto strided_slice = make_shared<op::v1::StridedSlice>(
            data, begin, end, vector<int64_t>{1, 0, 1, 0}, vector<int64_t>{1, 0, 1, 0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect shape of begin exception not thrown.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Begin input must be 1D (begin rank:"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, strided_slice_end_incorrect_shape)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::i64, Shape{4});
    auto end = make_shared<op::Parameter>(element::i64, Shape{4, 5});
    try
    {
        auto strided_slice = make_shared<op::v1::StridedSlice>(
            data, begin, end, vector<int64_t>{1, 0, 1, 0}, vector<int64_t>{1, 0, 1, 0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect shape of end exception not thrown.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("End input must be 1D (end rank:"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, strided_slice_default_stride_dynamic_shape_input)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto end = make_shared<op::Parameter>(element::i64, Shape{2});
    auto strided_slice = make_shared<op::v1::StridedSlice>(
        data, begin, end, vector<int64_t>{0, 0}, vector<int64_t>{0, 0});

    ASSERT_TRUE(strided_slice->input_value(3).get_partial_shape().compatible(PartialShape{2}));

    try
    {
        end = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
        strided_slice = make_shared<op::v1::StridedSlice>(
            data, begin, end, vector<int64_t>{0, 0}, vector<int64_t>{0, 0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Unknown data to calculate default strides exception not thrown.";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Begin input must be 1D"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, strided_slice_reverse_out_of_bounds)
{
    auto data = std::make_shared<op::Parameter>(ngraph::element::f32, ngraph::Shape{3, 4, 5});
    auto begin = op::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {100});
    auto end = op::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {-100});
    auto stride = op::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {-1});

    std::vector<int64_t> begin_mask = {0, 0, 0, 0};
    std::vector<int64_t> end_mask = {0, 0, 0, 0};

    auto ss =
        std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    Shape expected{3, 4, 5};
    EXPECT_EQ(ss->get_output_shape(0), expected);
}
