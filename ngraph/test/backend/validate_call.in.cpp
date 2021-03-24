// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_count)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(A, B), ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(auto handle = backend->compile(f); handle->call_with_validate({c}, {a}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_type)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(A, B), ParameterVector{A, B});

    auto a = backend->create_tensor(element::i32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(auto handle = backend->compile(f); handle->call_with_validate({c}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_shape)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(A, B), ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, {2, 3});
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(auto handle = backend->compile(f); handle->call_with_validate({c}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_count)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(A, B), ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);
    auto d = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(auto handle = backend->compile(f); handle->call_with_validate({c, d}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_type)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(A, B), ParameterVector{A, B});

    auto a = backend->create_tensor(element::i32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(auto handle = backend->compile(f); handle->call_with_validate({a}, {b, c}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_shape)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(A, B), ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, {2, 3});
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(auto handle = backend->compile(f); handle->call_with_validate({a}, {c, b}));
}
