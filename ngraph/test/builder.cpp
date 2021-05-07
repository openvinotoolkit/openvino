// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

shared_ptr<runtime::Tensor>
    make_reduce_result(function<shared_ptr<Node>(const shared_ptr<Node>&, const AxisSet&)> func)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(func(A, {0}), ParameterVector{A});
    auto backend = runtime::Backend::create("INTERPRETER");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    return result;
}

shared_ptr<runtime::Tensor> make_reduce_result_true(
    function<shared_ptr<Node>(const shared_ptr<Node>&, const AxisSet&, bool)> func)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(func(A, {0}, true), ParameterVector{A});
    auto backend = runtime::Backend::create("INTERPRETER");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    return result;
}

shared_ptr<runtime::Tensor> make_reduce_result_false(
    function<shared_ptr<Node>(const shared_ptr<Node>&, const AxisSet&, bool)> func)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(func(A, {0}, false), ParameterVector{A});
    auto backend = runtime::Backend::create("INTERPRETER");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    return result;
}
