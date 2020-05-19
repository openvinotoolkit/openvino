//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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

NGRAPH_TEST(${BACKEND_NAME}, divide)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), ParameterVector{A, B});

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
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), ParameterVector{A, B});

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
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B, false), ParameterVector{A, B});

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
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), ParameterVector{A, B});

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
    auto f = make_shared<Function>(A / B, ParameterVector{A, B});

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

NGRAPH_TEST(${BACKEND_NAME}, divide_adjoint_stability)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto make_external = [&]() {
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto B = make_shared<op::Parameter>(element::f32, shape);
        auto f = make_shared<Function>(make_shared<op::Divide>(A, B), ParameterVector{A, B});

        auto Y_out = f->output(0);
        auto Xs = f->get_parameters();
        auto C = std::make_shared<op::Parameter>(Y_out.get_element_type(), Y_out.get_shape());
        ngraph::autodiff::Adjoints adjoints(OutputVector{Y_out}, OutputVector{C});
        std::vector<Output<Node>> dYdXs(Xs.size());
        transform(
            Xs.begin(), Xs.end(), dYdXs.begin(), [C, &adjoints](const std::shared_ptr<Node>& X) {
                return adjoints.backprop_output(X);
            });
        std::vector<std::shared_ptr<op::Parameter>> params(Xs);
        params.push_back(C);

        auto bf = std::make_shared<Function>(dYdXs, params);

        return bf;
    };

    auto bf = make_external();

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0, 0, 1, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2, 2, 2, 2});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{1, 1, 1, 1});

    auto resulta = backend->create_tensor(element::f32, shape);
    auto resultb = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(bf);
    handle->call_with_validate({resulta, resultb}, {a, b, c});
    EXPECT_TRUE(
        test::all_close_f((vector<float>{0.5, 0.5, 0.5, 0.5}), read_vector<float>(resulta)));
    EXPECT_TRUE(
        test::all_close_f((vector<float>{-0.0, -0.0, -0.25, -0.25}), read_vector<float>(resultb)));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_by_zero_float32)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), ParameterVector{A, B});

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
