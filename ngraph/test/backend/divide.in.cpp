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
#include "util/type_prop.hpp"
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

NGRAPH_TEST(${BACKEND_NAME}, divide)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Divide>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 2, 2, 2}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_int32)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Divide>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{0x40000140, 0x40000001, 8, 16});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{2, 5, 4, 8});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int32_t>{536871072, 214748365, 2, 2}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_cpp_rounding_int32)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Divide>(A, B, false), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{-10, -10, 10, 10});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{-3, 3, -3, 3});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int32_t>{3, -3, -3, 3}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_python_rounding_int32)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Divide>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{-10, -10, 10, 10});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{-3, 3, -3, 3});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int32_t>{3, -4, -4, 3}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_overload)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Divide>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 2, 2, 2}), read_vector<float>(result)));
}

namespace
{
    template <typename Value>
    void divide_broadcast()
    {
        const auto element_type = ngraph::element::from<Value>();
        const Shape shape_a{3, 2, 1};
        const Shape shape_b{1, 6};
        const Shape shape_o{3, 2, 6};
        std::vector<Value> in_a{12, 24, 36, 48, 60, 72};
        std::vector<Value> in_b{1, 2, 3, 4, 6, 1};
        // clang-format off
        std::vector<Value> out{12,  6,  4,  3,  2,  12,
                               24, 12,  8,  6,  4,  24,

                               36, 18, 12, 9,  6,  36,
                               48, 24, 16, 12, 8,  48,

                               60, 30, 20, 15, 10, 60,
                               72, 36, 24, 18, 12, 72};
        // clang-format on

        auto A = make_shared<op::Parameter>(element_type, shape_a);
        auto B = make_shared<op::Parameter>(element_type, shape_b);
        auto f = make_shared<Function>(make_shared<op::v1::Divide>(A, B), ParameterVector{A, B});

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

NGRAPH_TEST(${BACKEND_NAME}, divide_int32_broadcast)
{
    divide_broadcast<int32_t>();
}

NGRAPH_TEST(${BACKEND_NAME}, divide_f32_broadcast)
{
    divide_broadcast<float>();
}

NGRAPH_TEST(${BACKEND_NAME}, divide_int32_scalar)
{
    Shape shape{};

    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Divide>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{18});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{8});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ(vector<int32_t>{2}, read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_f32_scalar)
{
    Shape shape{};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Divide>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{18});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{8});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{2.25}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_by_zero_float32)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Divide>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{0, 0, 0, 0});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));
}
