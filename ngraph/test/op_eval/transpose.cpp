// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/op/transpose.hpp"
#include "ngraph/runtime/reference/transpose.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/test_tools.hpp"
#include "util/all_close_f.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

template <element::Type_t T>
void test_tranpose_eval(shared_ptr<Function> fun,
               const vector<vector<float>>& inputs,
               vector<Shape>& x_shapes,
               vector<Shape>& result_shapes,
               vector<vector<float>>& results)
{
    using IN_T = typename element_type_traits<T>::value_type;
    const std::vector<std::vector<IN_T>> perms{{0, 1}, {1, 0}, {}, {2, 1, 0}};
    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        auto result_tensor = make_shared<HostTensor>();
        ASSERT_TRUE(fun->evaluate({result_tensor},
                                  {make_host_tensor<element::Type_t::f32>(x_shapes[i], inputs[i]),
                                   make_host_tensor<T>(Shape{perms[i].size()}, perms[i])}));

        auto actual_results = read_vector<float>(result_tensor);
        ASSERT_EQ(actual_results, results[i]);

        { // Temporary test for legacy reference function template
        std::vector<float> ref_results(inputs[i].size());
        runtime::reference::transpose<float, IN_T>(inputs[i].data(), ref_results.data(), x_shapes[i],  perms[i].data());
        ASSERT_EQ(ref_results, results[i]);
        }
    }
}

TEST(op_eval, eval_transpose)
{
    auto x = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    vector<shared_ptr<op::Parameter>> axes;
    axes.push_back(make_shared<op::Parameter>(element::i8, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::i16, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()}));

    axes.push_back(make_shared<op::Parameter>(element::u8, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::u16, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::u32, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::u64, PartialShape{Dimension::dynamic()}));

    std::vector<Shape> x_shapes{Shape{2, 3}, Shape{2, 3}, Shape{2, 3}, Shape{2, 2, 3}};

    const std::vector<std::vector<float>> inputs{
        {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6},{1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    std::vector<Shape> result_shapes{Shape{2, 3}, Shape{3, 2}, Shape{3, 2}, {3, 2, 2}};
    std::vector<std::vector<float>> results{
        {1, 2, 3, 4, 5, 6}, {1, 4, 2, 5, 3, 6},{1, 4, 2, 5, 3, 6}, {1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12}};

    for (auto& axis : axes)
    {
        auto x_transpose = make_shared<op::v1::Transpose>(x, axis);
        auto fun = make_shared<Function>(NodeVector{x_transpose}, ParameterVector{x, axis});

        switch (axis->get_element_type())
        {
        case element::Type_t::i8:
            test_tranpose_eval<element::Type_t::i8>(fun, inputs, x_shapes, result_shapes, results);
            break;
        case element::Type_t::i16:
            test_tranpose_eval<element::Type_t::i16>(fun, inputs, x_shapes, result_shapes, results);
            break;
        case element::Type_t::i32:
            test_tranpose_eval<element::Type_t::i32>(fun, inputs, x_shapes, result_shapes, results);
            break;
        case element::Type_t::i64:
            test_tranpose_eval<element::Type_t::i64>(fun, inputs, x_shapes, result_shapes, results);
            break;
        case element::Type_t::u8:
            test_tranpose_eval<element::Type_t::u8>(fun, inputs, x_shapes, result_shapes, results);
            break;
        case element::Type_t::u16:
            test_tranpose_eval<element::Type_t::u16>(fun, inputs, x_shapes, result_shapes, results);
            break;
        case element::Type_t::u32:
            test_tranpose_eval<element::Type_t::u32>(fun, inputs, x_shapes, result_shapes, results);
            break;
        case element::Type_t::u64:
            test_tranpose_eval<element::Type_t::u64>(fun, inputs, x_shapes, result_shapes, results);
            break;
        default: NGRAPH_CHECK(false, "Invalid type"); break;
        }
    }
}

TEST(op_eval, eval_duplicated_axes_transpose)
{
    auto data_param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes_order = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    auto x_transpose = make_shared<op::v1::Transpose>(data_param, axes_order);
    auto function = make_shared<Function>(NodeVector{x_transpose}, ParameterVector{data_param, axes_order});

    const std::vector<float> data{1, 2, 3, 4, 5, 6};
    std::vector<size_t> data_shape{2, 3, 1};
    const std::vector<int32_t> perm{1, 2, 2};
    std::vector<float> expected_result{1, 4, 2, 5, 3, 6};

    try
    {
        auto result_tensor = make_shared<HostTensor>();
        function->evaluate({result_tensor},
                                {make_host_tensor<element::Type_t::f32>(data_shape, data),
                                make_host_tensor<element::Type_t::i32>(Shape{perm.size()}, perm)});

        auto actual_results = read_vector<float>(result_tensor);

        ASSERT_EQ(actual_results, expected_result);
        FAIL() << "Duplicated axes values not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("must be unique"));
    }
    catch (...)
    {
        FAIL() << "Failed for unexpected reason";
    }
}

TEST(op_eval, eval_out_of_shape_axes_transpose)
{
    auto data_param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes_order = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    auto x_transpose = make_shared<op::v1::Transpose>(data_param, axes_order);
    auto function = make_shared<Function>(NodeVector{x_transpose}, ParameterVector{data_param, axes_order});

    const std::vector<float> data{1, 2, 3, 4, 5, 6};
    std::vector<size_t> data_shape{2, 3, 1};
    const std::vector<int32_t> perm{0, 1, 3};

    try
    {
        auto result_tensor = make_shared<HostTensor>();
        function->evaluate({result_tensor},
                                {make_host_tensor<element::Type_t::f32>(data_shape, data),
                                make_host_tensor<element::Type_t::i32>(Shape{perm.size()}, perm)});

        FAIL() << "Out of shape axes not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("out of shape"));
    }
    catch (...)
    {
        FAIL() << "Failed for unexpected reason";
    }
}

TEST(op_eval, eval_negative_axes_transpose)
{
    auto data_param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes_order = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    auto x_transpose = make_shared<op::v1::Transpose>(data_param, axes_order);
    auto function = make_shared<Function>(NodeVector{x_transpose}, ParameterVector{data_param, axes_order});

    const std::vector<float> data{1, 2, 3, 4, 5, 6};
    std::vector<size_t> data_shape{2, 3, 1};
    const std::vector<int32_t> perm{-1, -2, -3};
    std::vector<float> expected_result{1, 4, 2, 5, 3, 6};

    try
    {
        auto result_tensor = make_shared<HostTensor>();
        function->evaluate({result_tensor},
                                {make_host_tensor<element::Type_t::f32>(data_shape, data),
                                make_host_tensor<element::Type_t::i32>(Shape{perm.size()}, perm)});

        auto actual_results = read_vector<float>(result_tensor);

        ASSERT_EQ(actual_results, expected_result);
        FAIL() << "Negative axes for Transpose were not supported before.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("not supported"));
    }
    catch (...)
    {
        FAIL() << "Failed for unexpected reason";
    }
}
