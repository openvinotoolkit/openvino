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

TEST(type_prop, dynreplaceslice_arg_static_replacement_static_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto replacement = make_shared<op::Parameter>(element::f32, Shape{2, 4, 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
}

TEST(type_prop, dynreplaceslice_arg_static_replacement_rank_static_dynamic_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto replacement =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
}

TEST(type_prop, dynreplaceslice_arg_static_replacement_rank_dynamic_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
}

TEST(type_prop, dynreplaceslice_arg_rank_static_dynamic_replacement_static_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto replacement = make_shared<op::Parameter>(element::f32, Shape{2, 4, 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(
        PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8}));
}

TEST(type_prop,
     dynreplaceslice_arg_rank_static_dynamic_replacement_rank_static_dynamic_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto replacement =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(
        PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8}));
}

TEST(type_prop, dynreplaceslice_arg_rank_static_dynamic_replacement_rank_dynamic_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(
        PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8}));
}

TEST(type_prop, dynreplaceslice_arg_static_replacement_static_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto replacement = make_shared<op::Parameter>(element::f32, Shape{2, 4, 2, 4});
    auto lower_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto upper_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
}

TEST(type_prop,
     dynreplaceslice_arg_static_replacement_rank_static_dynamic_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto replacement =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 2, 4});
    auto lower_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto upper_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
}

TEST(type_prop, dynreplaceslice_arg_static_replacement_rank_dynamic_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto lower_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto upper_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
}

TEST(type_prop,
     dynreplaceslice_arg_rank_static_dynamic_replacement_static_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 2, 4});
    auto lower_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto upper_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(
        PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8}));
}

TEST(
    type_prop,
    dynreplaceslice_arg_rank_static_dynamic_replacement_rank_static_dynamic_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto replacement =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 2, 4});
    auto lower_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto upper_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(
        PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8}));
}

TEST(type_prop,
     dynreplaceslice_arg_rank_static_dynamic_replacement_rank_dynamic_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto lower_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto upper_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(
        PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8}));
}

TEST(type_prop, dynreplaceslice_arg_rank_dynamic_replacement_static_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape{4});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    // TODO(amprocte): We should be able to infer PartialShape::dynamic(4) here.
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreplaceslice_arg_rank_dynamic_replacement_rank_static_dynamic_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape{4});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    // TODO(amprocte): We should be able to infer PartialShape::dynamic(4) here.
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreplaceslice_arg_rank_dynamic_replacement_rank_dynamic_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape{4});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    // TODO(amprocte): We should be able to infer PartialShape::dynamic(4) here.
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreplaceslice_arg_rank_dynamic_replacement_static_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    // TODO(amprocte): We should be able to infer PartialShape::dynamic(4) here.
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop,
     dynreplaceslice_arg_rank_dynamic_replacement_rank_static_dynamic_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    // TODO(amprocte): We should be able to infer PartialShape::dynamic(4) here.
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop,
     dynreplaceslice_arg_rank_dynamic_replacement_rank_dynamic_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreplaceslice_arg_rank_dynamic_replacement_static_params_rank_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    // TODO(amprocte): We should be able to infer PartialShape::dynamic(4) here.
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop,
     dynreplaceslice_arg_rank_dynamic_replacement_rank_static_dynamic_params_rank_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    // TODO(amprocte): We should be able to infer PartialShape::dynamic(4) here.
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreplaceslice_arg_rank_dynamic_replacement_rank_dynamic_params_rank_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreplaceslice_static_shape)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5, 6});
    auto replacement = make_shared<op::Parameter>(element::f32, Shape{1, 2, 1, 1, 3});
    auto lower_bounds = op::Constant::create(element::i64, Shape{5}, {0, 1, 2, 3, 1});
    auto upper_bounds = op::Constant::create(element::i64, Shape{5}, {1, 3, 3, 5, 6});
    auto strides = op::Constant::create(element::i64, Shape{5}, {1, 1, 1, 2, 2});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_shape(), (Shape{2, 3, 4, 5, 6}));
}

TEST(type_prop, dynreplaceslice_static_shape_replacement_inconsistent)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5, 6});
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 1, 1, 4});
    auto lower_bounds = op::Constant::create(element::i64, Shape{5}, {0, 1, 2, 3, 1});
    auto upper_bounds = op::Constant::create(element::i64, Shape{5}, {1, 3, 3, 5, 6});
    auto strides = op::Constant::create(element::i64, Shape{5}, {1, 1, 1, 2, 2});

    try
    {
        auto r =
            make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);
        FAIL() << "Did not detect mismatch of replacement shape";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), "Shape of the replacement is not compatible with the shape of the slice");
    }
}

struct DynReplaceSliceParams
{
    Shape arg_shape;
    Shape lower_bounds_shape;
    Shape upper_bounds_shape;
    Shape strides_shape;
    Shape replacement_shape;

    std::vector<int64_t> lower_bounds_val;
    std::vector<int64_t> upper_bounds_val;
    std::vector<int64_t> strides_val;

    AxisSet lower_bounds_mask;
    AxisSet upper_bounds_mask;
    AxisSet new_axis;
    AxisSet shrink_axis;
    AxisSet ellipsis_mask;
};

struct DeduceDynReplaceSliceTest : ::testing::TestWithParam<DynReplaceSliceParams>
{
};

TEST_P(DeduceDynReplaceSliceTest, output_shape)
{
    auto tp = GetParam();
    auto arg = make_shared<op::Parameter>(element::f32, tp.arg_shape);
    auto replacement = make_shared<op::Parameter>(element::f32, tp.replacement_shape);
    auto lower_bounds =
        op::Constant::create(element::i64, tp.lower_bounds_shape, tp.lower_bounds_val);
    auto upper_bounds =
        op::Constant::create(element::i64, tp.upper_bounds_shape, tp.upper_bounds_val);
    auto strides = op::Constant::create(element::i64, tp.strides_shape, tp.strides_val);

    auto r = make_shared<op::DynReplaceSlice>(arg,
                                              replacement,
                                              lower_bounds,
                                              upper_bounds,
                                              strides,
                                              tp.lower_bounds_mask,
                                              tp.upper_bounds_mask,
                                              tp.new_axis,
                                              tp.shrink_axis,
                                              tp.ellipsis_mask);

    EXPECT_EQ(r->get_shape(), tp.arg_shape);
}

INSTANTIATE_TEST_CASE_P(
    type_prop,
    DeduceDynReplaceSliceTest,
    ::testing::Values(
        DynReplaceSliceParams{{2, 3, 4, 5, 6},
                              {5},
                              {5},
                              {5},
                              {1, 2, 1, 1, 3},
                              {0, 1, 2, 3, 1},
                              {1, 3, 3, 5, 6},
                              {1, 1, 1, 2, 2},
                              {},
                              {},
                              {},
                              {},
                              {}},
        DynReplaceSliceParams{{10}, {0}, {0}, {0}, {10}, {}, {}, {}, {}, {}, {}, {}, {}},
        DynReplaceSliceParams{
            {10}, {1}, {1}, {0}, {10}, {0}, {0}, {}, {}, {0}, {}, {}, {}}, // end-mask
        DynReplaceSliceParams{
            {10}, {1}, {1}, {0}, {9}, {-1}, {-1}, {}, {0}, {}, {}, {}, {}}, // begin-mask
        DynReplaceSliceParams{{10}, {1}, {1}, {0}, {10}, {0}, {10}, {}, {}, {}, {}, {}, {}},
        DynReplaceSliceParams{{10}, {1}, {1}, {0}, {5}, {5}, {10}, {}, {}, {}, {}, {}, {}},
        DynReplaceSliceParams{{10}, {1}, {1}, {0}, {5}, {-5}, {10}, {}, {}, {}, {}, {}, {}},
        DynReplaceSliceParams{{10},
                              {1},
                              {1},
                              {1},
                              {6},
                              {-5},
                              {0},
                              {-1}, // negative-stride
                              {},
                              {0},
                              {},
                              {},
                              {}},
        DynReplaceSliceParams{{10}, {1}, {1}, {1}, {3}, {-5}, {2}, {-1}, {}, {}, {}, {}, {}},
        DynReplaceSliceParams{{10}, {1}, {1}, {1}, {5}, {0}, {0}, {2}, {}, {0}, {}, {}, {}},
        DynReplaceSliceParams{{10}, {1}, {1}, {1}, {5}, {1}, {0}, {2}, {}, {0}, {}, {}, {}},
        DynReplaceSliceParams{{10}, {1}, {1}, {1}, {10}, {-1}, {0}, {-1}, {}, {0}, {}, {}, {}},
        DynReplaceSliceParams{{10}, {1}, {1}, {1}, {5}, {-1}, {0}, {-2}, {}, {0}, {}, {}, {}},
        /* Axis Masks: New, Shrink, Ellipsis */
        DynReplaceSliceParams{{10}, {1}, {1}, {0}, {1, 10}, {0}, {10}, {}, {}, {}, {0}, {}, {}},
        DynReplaceSliceParams{
            {1, 2, 3}, {2}, {2}, {0}, {1, 2, 3}, {0, 0}, {1, 2}, {}, {}, {}, {}, {}, {1}},
        DynReplaceSliceParams{{1, 2, 3},
                              {4},
                              {4},
                              {0},
                              {1, 2, 1},
                              {0, 0, 0, 1},
                              {2, 3, 2, 2},
                              {},
                              {},
                              {},
                              {2},
                              {3},
                              {}},
        DynReplaceSliceParams{
            {1, 2, 3}, {3}, {3}, {0}, {1, 1, 2, 1}, {0, 0, 1}, {2, 2, 2}, {}, {}, {}, {0}, {}, {1}},
        DynReplaceSliceParams{
            {1, 2, 2, 2}, {1}, {1}, {1}, {0, 2, 2, 2}, {-1}, {0}, {-2}, {1}, {1}, {}, {1}, {}},
        DynReplaceSliceParams{{9, 10, 12, 2, 3}, /*arg_shape*/
                              {4},               /*lower_bounds_shape*/
                              {4},               /*upper_bounds_shape*/
                              {4},               /*strides_shape*/
                              {2, 10, 12, 2, 0}, /*replacement_shape*/

                              {2, 0, 0, 3},  /*lower_bounds_val*/
                              {6, 0, 0, 2},  /*upper_bounds_val*/
                              {2, 1, 1, -1}, /*strides_val*/

                              {2},  /*lower_bounds_mask*/
                              {2},  /*upper_bounds_mask*/
                              {},   /*new_axis*/
                              {},   /*shrink_axis*/
                              {1}}, /*ellipsis_mask*/

        DynReplaceSliceParams{{9, 10, 12, 2, 3}, /*arg_shape*/
                              {4},               /*lower_bounds_shape*/
                              {4},               /*upper_bounds_shape*/
                              {4},               /*strides_shape*/
                              {3, 10, 12, 1, 0}, /*replacement_shape*/

                              {6, 0, 1, 3},   /*lower_bounds_val*/
                              {1, 0, 2, 2},   /*upper_bounds_val*/
                              {-2, 1, 1, -1}, /*strides_val*/

                              {},                /*lower_bounds_mask*/
                              {},                /*upper_bounds_mask*/
                              {},                /*new_axis*/
                              {},                /*shrink_axis*/
                              {1}},              /*ellipsis_mask*/
        DynReplaceSliceParams{{9, 10, 12, 2, 3}, /*arg_shape*/
                              {3},               /*lower_bounds_shape*/
                              {3},               /*upper_bounds_shape*/
                              {3},               /*strides_shape*/
                              {9, 10, 12, 1, 0}, /*replacement_shape*/

                              {0, 1, 3},  /*lower_bounds_val*/
                              {0, 2, 2},  /*upper_bounds_val*/
                              {1, 1, -1}, /*strides_val*/

                              {},   /*lower_bounds_mask*/
                              {},   /*upper_bounds_mask*/
                              {},   /*new_axis*/
                              {},   /*shrink_axis*/
                              {0}}, /*ellipsis_mask*/
        DynReplaceSliceParams{{1, 2, 2, 2},
                              {4},
                              {4},
                              {0},
                              {1, 2, 2},
                              {0, 1, 0, 0},
                              {1, 2, 2, 2},
                              {},
                              {1},
                              {1},
                              {},
                              {1},
                              {}},
        DynReplaceSliceParams{
            {1, 2, 3}, {3}, {3}, {0}, {1, 1, 2}, {0, 0, 1}, {2, 2, 2}, {}, {}, {}, {0}, {2}, {1}}),
    PrintToDummyParamName());

void DynReplaceSlice_Test_Shape_Except(const shared_ptr<Node>& param_0,
                                       const shared_ptr<Node>& param_1,
                                       const shared_ptr<Node>& param_2,
                                       const shared_ptr<Node>& param_3,
                                       const shared_ptr<Node>& param_4)
{
    try
    {
        auto r = make_shared<op::DynReplaceSlice>(param_0, param_1, param_2, param_3, param_4);
        FAIL() << "Did not detect attributes not vector";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("shape must have rank 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dynreplaceslice_arg_static_replacement_static_params_rank_static_dynamic_not_vector)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    {
        lower_bounds =
            make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape{2, 2});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        arg = make_shared<op::Parameter>(
            element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
        lower_bounds =
            make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }

    {
        upper_bounds =
            make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape{2, 2});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        arg = make_shared<op::Parameter>(
            element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
        upper_bounds =
            make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }

    {
        strides = make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        strides = make_shared<op::Parameter>(element::i64, PartialShape{2, 2});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        arg = make_shared<op::Parameter>(
            element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
        strides = make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        replacement =
            make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 2, 4});
        strides = make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
}

TEST(type_prop, dynreplaceslice_params_et_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::dynamic, Shape{2, 4, 6, 8});
    auto replacement = make_shared<op::Parameter>(element::dynamic, Shape{2, 4, 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::dynamic, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::dynamic, Shape{4});
    auto strides = make_shared<op::Parameter>(element::dynamic, Shape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::dynamic);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
}

TEST(type_prop, dynreplaceslice_params_et_dynamic_inferrable_ok)
{
    auto arg = make_shared<op::Parameter>(element::dynamic, Shape{2, 4, 6, 8});
    auto replacement = make_shared<op::Parameter>(element::boolean, Shape{2, 4, 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::dynamic, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::dynamic, Shape{4});
    auto strides = make_shared<op::Parameter>(element::dynamic, Shape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::boolean);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
}

void DynReplaceSlice_Test_Type_Except(const shared_ptr<Node>& param_0,
                                      const shared_ptr<Node>& param_1,
                                      const shared_ptr<Node>& param_2,
                                      const shared_ptr<Node>& param_3,
                                      const shared_ptr<Node>& param_4)
{
    try
    {
        auto r = make_shared<op::DynReplaceSlice>(param_0, param_1, param_2, param_3, param_4);
        FAIL() << "Did not detect parameter element type not i64";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("must have element type i64."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dynreplaceslice_params_et_wrong)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto replacement = make_shared<op::Parameter>(element::f32, Shape{2, 4, 2, 4});

    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    {
        lower_bounds = make_shared<op::Parameter>(element::boolean, Shape{4});
        DynReplaceSlice_Test_Type_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        upper_bounds = make_shared<op::Parameter>(element::boolean, Shape{4});
        DynReplaceSlice_Test_Type_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        strides = make_shared<op::Parameter>(element::boolean, Shape{4});
        DynReplaceSlice_Test_Type_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
}
