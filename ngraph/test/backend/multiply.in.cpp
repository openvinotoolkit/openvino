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
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, multiply)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Multiply>(A, B), ParameterVector{A, B});

    std::vector<float> a{1, 2, 3, 4};
    std::vector<float> b{5, 6, 7, 8};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {5, 12, 21, 32});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_overload)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Multiply>(A, B), ParameterVector{A, B});

    std::vector<float> a{1, 2, 3, 4};
    std::vector<float> b{5, 6, 7, 8};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {5, 12, 21, 32});
    test_case.run();
}

namespace
{
    template <typename Value>
    void multiply_broadcst()
    {
        const auto element_type = ngraph::element::from<Value>();
        const Shape shape_a{3, 2, 1};
        const Shape shape_b{1, 6};
        const Shape shape_o{3, 2, 6};
        std::vector<Value> in_a{12, 24, 36, 48, 60, 72};
        std::vector<Value> in_b{1, 2, 3, 4, 6, 1};
        // clang-format off
        std::vector<Value> out{12,  24,  36,  48,  72,  12,
                               24,  48,  72,  96, 144,  24,

                               36,  72, 108, 144, 216,  36,
                               48,  96, 144, 192, 288,  48,

                               60, 120, 180, 240, 360,  60,
                               72, 144, 216, 288, 432,  72};
        // clang-format on

        auto A = make_shared<op::Parameter>(element_type, shape_a);
        auto B = make_shared<op::Parameter>(element_type, shape_b);
        auto f = make_shared<Function>(make_shared<op::v1::Multiply>(A, B), ParameterVector{A, B});

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

NGRAPH_TEST(${BACKEND_NAME}, multiply_int32_broadcast)
{
    multiply_broadcst<int32_t>();
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_f32_broadcast)
{
    multiply_broadcst<float>();
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_int32_scalar)
{
    Shape shape{};

    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Multiply>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{2});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{8});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ(vector<int32_t>{16}, read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_f32_scalar)
{
    Shape shape{};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Multiply>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{3.1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{8});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{24.8}), read_vector<float>(result)));
}
