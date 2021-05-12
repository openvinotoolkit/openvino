// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, transpose_basic)
{
    //
    // Create a graph for f(x,perm) = Transpose(x,Convert<i64>(perm)). We'll do the permutation in
    // i32 and cast it to i64, just for fun (and to mirror the TensorFlow test I am porting here).
    //
    auto x = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto perm = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto perm_i64 = make_shared<op::Convert>(perm, element::i64);

    auto x_transpose = make_shared<op::Transpose>(x, perm_i64);

    auto f = make_shared<Function>(NodeVector{x_transpose}, ParameterVector{x, perm});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    std::vector<Shape> x_shapes{Shape{2, 3}, Shape{2, 3}, Shape{2, 2, 3}};
    std::vector<std::vector<int32_t>> perms{{0, 1}, {1, 0}, {2, 1, 0}};
    std::vector<std::vector<float>> inputs{
        {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    std::vector<Shape> expected_result_shapes{Shape{2, 3}, Shape{3, 2}, {3, 2, 2}};
    // Generated with numpy, so don't worry. :)
    std::vector<std::vector<float>> expected_results{
        {1, 2, 3, 4, 5, 6}, {1, 4, 2, 5, 3, 6}, {1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12}};

    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        auto t_x = backend->create_tensor(element::f32, x_shapes[i]);
        auto t_perm = backend->create_tensor(element::i32, Shape{perms[i].size()});

        copy_data(t_x, inputs[i]);
        copy_data(t_perm, perms[i]);

        ex->call_with_validate({t_r}, {t_x, t_perm});

        ASSERT_EQ(t_r->get_shape(), expected_result_shapes[i]);

        auto results = read_vector<float>(t_r);

        ASSERT_TRUE(test::all_close_f(results, expected_results[i], MIN_FLOAT_TOLERANCE_BITS));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, transpose_axes_constant)
{
    const auto data_shape = Shape{2, 1, 3, 4};
    const auto axes_shape = Shape{4};
    const auto output_shape = Shape{3, 4, 2, 1};

    auto data_param = make_shared<op::Parameter>(element::f32, data_shape);
    auto axes_const = op::Constant::create(element::i64, axes_shape, {2, 3, 0, 1});
    auto transpose = make_shared<op::Transpose>(data_param, axes_const);
    auto function = make_shared<ngraph::Function>(NodeVector{transpose}, ParameterVector{data_param});

    std::vector<float> data(shape_size(data_shape));
    std::iota(data.begin(), data.end(), 1);
    std::vector<float> expected_result{ 1, 13,  2, 14,  3, 15,  4, 16,  5, 17,  6, 18,  7, 19,  8, 20,  9,
       21, 10, 22, 11, 23, 12, 24};

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(data_shape, data);
    test_case.add_expected_output<float>(output_shape, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, transpose_axes_empty_constant)
{
    const auto data_shape = Shape{2, 1, 3, 4};
    const auto axes_shape = Shape{0};
    const auto output_shape = Shape{4, 3, 1, 2};

    auto data_param = make_shared<op::Parameter>(element::f32, data_shape);
    auto axes_const = op::Constant::create(element::i64, axes_shape, std::vector<int>{});
    auto transpose = make_shared<op::Transpose>(data_param, axes_const);
    auto function = make_shared<ngraph::Function>(NodeVector{transpose}, ParameterVector{data_param});

    std::vector<float> data(shape_size(data_shape));
    std::iota(data.begin(), data.end(), 1);
    std::vector<float> expected_result{ 1, 13,  5, 17,  9, 21,  2, 14,  6, 18, 10, 22,  3, 15,  7, 19, 11,
       23,  4, 16,  8, 20, 12, 24};
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(data_shape, data);
    test_case.add_expected_output<float>(output_shape, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, transpose_axes_parameter_static_shapes)
{
    const auto data_shape = Shape{2, 1, 3, 4};
    const auto axes_shape = Shape{4};
    const auto output_shape = Shape{3, 4, 2, 1};

    auto data_param = make_shared<op::Parameter>(element::f32, data_shape);
    auto axes_param = make_shared<op::Parameter>(element::i32, axes_shape);

    auto transpose = make_shared<op::Transpose>(data_param, axes_param);
    auto function = make_shared<ngraph::Function>(NodeVector{transpose}, ParameterVector{data_param, axes_param});

    std::vector<float> data(shape_size(data_shape));
    std::iota(data.begin(), data.end(), 1);

    std::vector<int> axes{2, 3, 0, 1};
    std::vector<float> expected_result{ 1, 13,  2, 14,  3, 15,  4, 16,  5, 17,  6, 18,  7, 19,  8, 20,  9,
       21, 10, 22, 11, 23, 12, 24};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    test_case.add_input<float>(data_shape, data);
    test_case.add_input<int>(axes_shape, axes);
    test_case.add_expected_output<float>(output_shape, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, transpose_axes_parameter_dynamic_shapes)
{
    const auto data_shape = Shape{2, 1, 3, 4};
    const auto axes_shape = Shape{4};
    const auto output_shape = Shape{3, 4, 2, 1};

    auto data_param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes_param = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    auto transpose = make_shared<op::Transpose>(data_param, axes_param);
    auto function = make_shared<ngraph::Function>(NodeVector{transpose}, ParameterVector{data_param, axes_param});

    std::vector<float> data(shape_size(data_shape));
    std::iota(data.begin(), data.end(), 1);

    std::vector<int> axes{2, 3, 0, 1};
    std::vector<float> expected_result{ 1, 13,  2, 14,  3, 15,  4, 16,  5, 17,  6, 18,  7, 19,  8, 20,  9,
       21, 10, 22, 11, 23, 12, 24};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    test_case.add_input<float>(data_shape, data);
    test_case.add_input<int>(axes_shape, axes);
    test_case.add_expected_output<float>(output_shape, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, transpose_int_data_axes_constant)
{
    const auto data_shape = Shape{2, 1, 3, 4};
    const auto axes_shape = Shape{4};
    const auto output_shape = Shape{3, 4, 2, 1};

    auto data_param = make_shared<op::Parameter>(element::i32, data_shape);
    auto axes_const = op::Constant::create(element::i64, axes_shape, {2, 3, 0, 1});
    auto transpose = make_shared<op::Transpose>(data_param, axes_const);
    auto function = make_shared<ngraph::Function>(NodeVector{transpose}, ParameterVector{data_param});

    std::vector<int32_t> data(shape_size(data_shape));
    std::iota(data.begin(), data.end(), 1);
    std::vector<int32_t> expected_result{ 1, 13,  2, 14,  3, 15,  4, 16,  5, 17,  6, 18,  7, 19,  8, 20,  9,
       21, 10, 22, 11, 23, 12, 24};

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<int32_t>(data_shape, data);
    test_case.add_expected_output<int32_t>(output_shape, expected_result);
    test_case.run();
}
