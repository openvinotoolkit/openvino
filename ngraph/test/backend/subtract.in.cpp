// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, subtract)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Subtract>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 2, 4, 8}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, subtract_overload)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(std::make_shared<op::v1::Subtract>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 2, 4, 8}), read_vector<float>(result)));
}

namespace
{
    template <typename Value>
    void subtract_broadcst()
    {
        const auto element_type = ngraph::element::from<Value>();
        const Shape shape_a{3, 2, 1};
        const Shape shape_b{1, 6};
        const Shape shape_o{3, 2, 6};
        std::vector<Value> in_a{12, 24, 36, 48, 60, 72};
        std::vector<Value> in_b{1, 2, 3, 4, 6, 1};
        // clang-format off
        std::vector<Value> out{11, 10,  9,  8,  6, 11,
                               23, 22, 21, 20, 18, 23,

                               35, 34, 33, 32, 30, 35,
                               47, 46, 45, 44, 42, 47,

                               59, 58, 57, 56, 54, 59,
                               71, 70, 69, 68, 66, 71};
        // clang-format on

        auto A = make_shared<op::Parameter>(element_type, shape_a);
        auto B = make_shared<op::Parameter>(element_type, shape_b);
        auto f = make_shared<Function>(make_shared<op::v1::Subtract>(A, B), ParameterVector{A, B});

        auto backend = runtime::Backend::create("${BACKEND_NAME}");

        // Create some tensors for input/output
        auto a = backend->create_tensor(element_type, shape_a, in_a.data());
        auto b = backend->create_tensor(element_type, shape_b, in_b.data());
        auto result = backend->create_tensor(element_type, shape_o);

        auto handle = backend->compile(f);
        handle->call_with_validate({result}, {a, b});
        EXPECT_EQ(out, read_vector<Value>(result));
    }
} // namespace

NGRAPH_TEST(${BACKEND_NAME}, subtract_int32_broadcast)
{
    subtract_broadcst<int32_t>();
}

NGRAPH_TEST(${BACKEND_NAME}, subtract_f32_broadcast)
{
    subtract_broadcst<float>();
}

NGRAPH_TEST(${BACKEND_NAME}, subtract_int32_scalar)
{
    Shape shape{};

    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Subtract>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{2});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{8});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ(vector<int32_t>{-6}, read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, subtract_f32_scalar)
{
    Shape shape{};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Subtract>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{3.1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{8});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{-4.9}), read_vector<float>(result)));
}
