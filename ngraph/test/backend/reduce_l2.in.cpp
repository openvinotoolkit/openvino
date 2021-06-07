// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, reduce_l2_one_axis_keep_dims)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 2, 2});
    auto axes = op::Constant::create(element::i32, Shape{1}, {2});
    auto reduce_l2 = make_shared<op::v4::ReduceL2>(data, axes, true);
    auto f = make_shared<Function>(OutputVector{reduce_l2}, ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    std::vector<float> input{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    std::vector<float> expected_result{
        2.23606798, 5.0, 7.81024968, 10.63014581, 13.45362405, 16.2788206};

    auto data_tensor = backend->create_tensor(element::f32, Shape{3, 2, 2});
    copy_data(data_tensor, input);

    auto result_tensor = backend->create_tensor(element::f32, Shape{3, 2, 1});

    auto handle = backend->compile(f);
    handle->call_with_validate({result_tensor}, {data_tensor});
    EXPECT_TRUE(test::all_close_f((expected_result), read_vector<float>(result_tensor)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_l2_one_axis_do_not_keep_dims)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 2, 2});
    auto axes = op::Constant::create(element::i32, Shape{1}, {2});
    auto reduce_l2 = make_shared<op::v4::ReduceL2>(data, axes, false);
    auto f = make_shared<Function>(OutputVector{reduce_l2}, ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    std::vector<float> input{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    std::vector<float> expected_result{
        2.23606798, 5.0, 7.81024968, 10.63014581, 13.45362405, 16.2788206};

    auto data_tensor = backend->create_tensor(element::f32, Shape{3, 2, 2});
    copy_data(data_tensor, input);

    auto result_tensor = backend->create_tensor(element::f32, Shape{3, 2});

    auto handle = backend->compile(f);
    handle->call_with_validate({result_tensor}, {data_tensor});
    EXPECT_TRUE(test::all_close_f((expected_result), read_vector<float>(result_tensor)));
}
