// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "runtime/interpreter/int_executable.hpp"
#include "util/test_tools.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

TEST(INTERPRETER, nan_check_input)
{
    Shape shape{4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Divide>(A, B), ParameterVector{A, B});

    shared_ptr<runtime::Backend> backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, NAN, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 1, 8});
    auto result = backend->create_tensor(element::f32, shape);

    shared_ptr<runtime::Executable> handle = backend->compile(f);

    shared_ptr<runtime::interpreter::INTExecutable> ihandle =
        static_pointer_cast<runtime::interpreter::INTExecutable>(handle);
    ihandle->set_nan_check(true);
    EXPECT_ANY_THROW(handle->call_with_validate({result}, {a, b}));
}

TEST(INTERPRETER, nan_check_output)
{
    Shape shape{4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Divide>(A, B), ParameterVector{A, B});

    shared_ptr<runtime::Backend> backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 0, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 0, 8});
    auto result = backend->create_tensor(element::f32, shape);

    shared_ptr<runtime::Executable> handle = backend->compile(f);
    shared_ptr<runtime::interpreter::INTExecutable> ihandle =
        static_pointer_cast<runtime::interpreter::INTExecutable>(handle);
    ihandle->set_nan_check(true);
    EXPECT_ANY_THROW(handle->call_with_validate({result}, {a, b}));
}
