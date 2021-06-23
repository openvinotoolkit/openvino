// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

struct ReduceParams
{
    PartialShape data_ps;
    element::Type data_et;
    Shape axes_ps;
    std::vector<int64_t> axes;
    element::Type axes_et;
    bool keep_dims;
};

template <class T>
static std::shared_ptr<Node> makeReduceOp(const ReduceParams& p, bool axes_as_param = false)
{
    auto in_data = make_shared<op::Parameter>(p.data_et, p.data_ps);
    shared_ptr<Node> in_axes;
    if (axes_as_param)
    {
        in_axes = make_shared<op::Parameter>(p.axes_et, p.axes_ps);
    }
    else
    {
        if (shape_size(p.axes_ps) != p.axes.size())
        {
            throw ngraph_error("Axes shape does not match with axes elements");
        }
        in_axes = make_shared<op::Constant>(p.axes_et, p.axes_ps, p.axes);
    }
    return make_shared<T>(in_data, in_axes, p.keep_dims);
}

template <class T>
class ReduceTest : public testing::Test
{
};

TYPED_TEST_SUITE_P(ReduceTest);

TYPED_TEST_P(ReduceTest, reduce_basic_shape_infer)
{
    PartialShape data_ps{3, 4, 5};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{1, 2};

    bool keep_dims = false;

    PartialShape out_ps{3};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    ASSERT_TRUE(reduce_op->get_output_partial_shape(0).same_scheme(out_ps));
}

TYPED_TEST_P(ReduceTest, reduce_basic_shape_infer_keep_dims)
{
    PartialShape data_ps{3, 4, 5};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{1, 2};

    bool keep_dims = true;

    PartialShape out_ps{3, 1, 1};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    ASSERT_TRUE(reduce_op->get_output_partial_shape(0).same_scheme(out_ps));
}

TYPED_TEST_P(ReduceTest, reduce_basic_shape_infer_scalar_axis)
{
    PartialShape data_ps{3, 4, 5};
    element::Type data_et = element::dynamic;

    Shape axes_ps{};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{1};

    bool keep_dims = false;

    PartialShape out_ps{3, 5};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    ASSERT_TRUE(reduce_op->get_output_partial_shape(0).same_scheme(out_ps));
}

TYPED_TEST_P(ReduceTest, reduce_basic_shape_infer_axes_as_param)
{
    PartialShape data_ps{3, 4, 5};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i32;
    std::vector<int64_t> axes;

    bool keep_dims = false;

    PartialShape out_ps{PartialShape::dynamic()};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    bool axes_as_param = true;
    auto reduce_op = makeReduceOp<TypeParam>(params, axes_as_param);
    ASSERT_TRUE(reduce_op->get_output_partial_shape(0).same_scheme(out_ps));
}

TYPED_TEST_P(ReduceTest, reduce_dynamic_shape_reduced_axes_static)
{
    PartialShape data_ps{3, 4, 5, Dimension::dynamic()};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{1, 2};

    bool keep_dims = false;

    PartialShape out_ps{3, Dimension::dynamic()};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    ASSERT_TRUE(reduce_op->get_output_partial_shape(0).same_scheme(out_ps));
}

TYPED_TEST_P(ReduceTest, reduce_dynamic_shape_reduced_axes_static_keep_dims)
{
    PartialShape data_ps{3, 4, 5, Dimension::dynamic()};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{1, 2};

    bool keep_dims = true;

    PartialShape out_ps{3, 1, 1, Dimension::dynamic()};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    ASSERT_TRUE(reduce_op->get_output_partial_shape(0).same_scheme(out_ps));
}

TYPED_TEST_P(ReduceTest, reduce_dynamic_shape_reduced_axes_not_static)
{
    PartialShape data_ps{Dimension::dynamic(), 4, 5, Dimension::dynamic()};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{2, 3};

    bool keep_dims = false;

    PartialShape out_ps{Dimension::dynamic(), 4};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    ASSERT_TRUE(reduce_op->get_output_partial_shape(0).same_scheme(out_ps));
}

TYPED_TEST_P(ReduceTest, reduce_dynamic_shape_reduced_axes_not_static_keep_dims)
{
    PartialShape data_ps{Dimension::dynamic(), 4, 5, Dimension::dynamic()};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{2, 3};

    bool keep_dims = true;

    PartialShape out_ps{Dimension::dynamic(), 4, 1, 1};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    ASSERT_TRUE(reduce_op->get_output_partial_shape(0).same_scheme(out_ps));
}

TYPED_TEST_P(ReduceTest, reduce_dynamic_shape_data)
{
    PartialShape data_ps{PartialShape::dynamic()};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{1, 2};

    bool keep_dims = false;

    PartialShape out_ps{PartialShape::dynamic()};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    ASSERT_TRUE(reduce_op->get_output_partial_shape(0).same_scheme(out_ps));
}

TYPED_TEST_P(ReduceTest, reduce_invalid_axis_out_of_range)
{
    PartialShape data_ps{1, 2, 3};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{2, 3};

    bool keep_dims = false;

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    try
    {
        auto reduce_op = makeReduceOp<TypeParam>(params);
        FAIL() << "Invalid axes values not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Reduction axis (");
    }
    catch (...)
    {
        FAIL() << "Axes input values validation check failed for unexpected reason";
    }
}

TYPED_TEST_P(ReduceTest, reduce_invalid_axes_shape)
{
    PartialShape data_ps{1, 2, 3};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2, 1};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{0, 1};

    bool keep_dims = true;

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    try
    {
        auto reduce_op = makeReduceOp<TypeParam>(params);
        FAIL() << "Invalid shape of axes input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Axes input must be a scalar or 1D input.");
    }
    catch (...)
    {
        FAIL() << "Axes input shape validation check failed for unexpected reason";
    }
}

TYPED_TEST_P(ReduceTest, reduce_invalid_axes_et)
{
    element::Type data_et = element::dynamic;
    PartialShape data_ps{1, 2, 3};

    element::Type axes_et = element::f32;
    Shape axes_ps{2};
    std::vector<int64_t> axes{0, 1};

    bool keep_dims = true;

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    try
    {
        auto reduce_op = makeReduceOp<TypeParam>(params);
        FAIL() << "Invalid element type of axes input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Element type of axes input must be integer.");
    }
    catch (...)
    {
        FAIL() << "Axes input element type validation check failed for unexpected reason";
    }
}

REGISTER_TYPED_TEST_SUITE_P(ReduceTest,
                            reduce_basic_shape_infer,
                            reduce_basic_shape_infer_keep_dims,
                            reduce_basic_shape_infer_scalar_axis,
                            reduce_basic_shape_infer_axes_as_param,
                            reduce_dynamic_shape_data,
                            reduce_dynamic_shape_reduced_axes_static,
                            reduce_dynamic_shape_reduced_axes_static_keep_dims,
                            reduce_dynamic_shape_reduced_axes_not_static,
                            reduce_dynamic_shape_reduced_axes_not_static_keep_dims,
                            reduce_invalid_axis_out_of_range,
                            reduce_invalid_axes_shape,
                            reduce_invalid_axes_et);

template <class T>
class ReduceArithmeticTest : public testing::Test
{
};

TYPED_TEST_SUITE_P(ReduceArithmeticTest);

TYPED_TEST_P(ReduceArithmeticTest, reduce_arithmetic_invalid_data_et)
{
    element::Type data_et = element::boolean;
    PartialShape data_ps{1, 2, 3};

    element::Type axes_et = element::i32;
    Shape axes_ps{2};
    std::vector<int64_t> axes{0, 1};

    bool keep_dims = true;

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    try
    {
        auto reduce_op = makeReduceOp<TypeParam>(params);
        FAIL() << "Invalid element type of data input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Element type of data input must be numeric.");
    }
    catch (...)
    {
        FAIL() << "Data input element type validation check failed for unexpected reason";
    }
}

REGISTER_TYPED_TEST_SUITE_P(ReduceArithmeticTest, reduce_arithmetic_invalid_data_et);

template <class T>
class ReduceLogicalTest : public testing::Test
{
};

TYPED_TEST_SUITE_P(ReduceLogicalTest);

TYPED_TEST_P(ReduceLogicalTest, reduce_logical_invalid_data_et)
{
    std::vector<element::Type> element_types{element::f32, element::i32, element::u32};
    PartialShape data_ps{1, 2, 3};

    element::Type axes_et = element::i32;
    Shape axes_ps{2};
    std::vector<int64_t> axes{0, 1};

    bool keep_dims = true;

    for (const auto& data_et : element_types)
    {
        const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
        try
        {
            auto reduce_op = makeReduceOp<TypeParam>(params);
            FAIL() << "Invalid element type of data input not detected";
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(), "Element type of data input must be boolean.");
        }
        catch (...)
        {
            FAIL() << "Data input element type validation check failed for unexpected reason";
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(ReduceLogicalTest, reduce_logical_invalid_data_et);
