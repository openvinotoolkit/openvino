//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
#include <numeric>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, matmul_2x0_0x2)
{
    Shape shape_a{2, 0};
    Shape shape_b{0, 2};
    Shape shape_r{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::MatMul>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the
    // right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{0, 0, 0, 0}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_0x2_2x0)
{
    Shape shape_a{0, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 0};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{0, 0};
    auto f = make_shared<Function>(make_shared<op::MatMul>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_3x2_2x0)
{
    Shape shape_a{3, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 0};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{3, 0};
    auto f = make_shared<Function>(make_shared<op::MatMul>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_2x2_2x2)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{2, 2};
    auto f = make_shared<Function>(make_shared<op::MatMul>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{5, 6, 7, 8});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{19, 22, 43, 50}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_2x3_3x3)
{
    Shape shape_in1{2, 3};
    Shape shape_in2{3, 3};
    Shape shape_out{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_in1);
    auto B = make_shared<op::Parameter>(element::f32, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, false, false);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape_in2);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape_out);

    copy_data(a, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    copy_data(b, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});

    EXPECT_TRUE(test::all_close_f(read_vector<float>(result),
                                  vector<float>{30.f, 36.f, 42.f, 66.f, 81.f, 96.f}));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_2x3_3x3_int64)
{
    Shape shape_in1{2, 3};
    Shape shape_in2{3, 3};
    Shape shape_out{2, 3};
    auto A = make_shared<op::Parameter>(element::i64, shape_in1);
    auto B = make_shared<op::Parameter>(element::i64, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, false, false);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i64, shape_in1);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i64, shape_in2);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i64, shape_out);

    copy_data(a, vector<int64_t>{1, 2, 3, 4, 5, 6});
    copy_data(b, vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});

    EXPECT_TRUE(
        test::all_close(read_vector<int64_t>(result), vector<int64_t>{30, 36, 42, 66, 81, 96}));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_3x2_3x3_transpose)
{
    Shape shape_in1{3, 2};
    Shape shape_in2{3, 3};
    Shape shape_out{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_in1);
    auto B = make_shared<op::Parameter>(element::f32, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, true, false);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape_in2);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape_out);

    copy_data(a, vector<float>{1.f, 4.f, 2.f, 5.f, 3.f, 6.f});
    copy_data(b, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});

    EXPECT_TRUE(test::all_close_f(read_vector<float>(result),
                                  vector<float>{30.f, 36.f, 42.f, 66.f, 81.f, 96.f}));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_3x2_2x3_transpose)
{
    Shape shape_in1{3, 2};
    Shape shape_in2{2, 3};
    Shape shape_out{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_in1);
    auto B = make_shared<op::Parameter>(element::f32, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, true, true);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape_in2);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape_out);

    copy_data(a, vector<float>{1.f, 4.f, 2.f, 5.f, 3.f, 6.f});
    copy_data(b, vector<float>{1.f, 3.f, 5.f, 2.f, 4.f, 6.f});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});

    EXPECT_TRUE(
        test::all_close_f(read_vector<float>(result), vector<float>{22.f, 28.f, 49.f, 64.f}));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_2x3x2_3x3_transpose)
{
    Shape shape_in1{2, 3, 2};
    Shape shape_in2{3, 3};
    Shape shape_out{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_in1);
    auto B = make_shared<op::Parameter>(element::f32, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, true, false);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape_in2);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape_out);

    copy_data(a, vector<float>{1.f, 4.f, 2.f, 5.f, 3.f, 6.f, 3.f, 2.f, 1.f, 4.f, 5.f, 6.f});
    copy_data(b, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});

    EXPECT_TRUE(test::all_close_f(
        read_vector<float>(result),
        vector<float>{30.f, 36.f, 42.f, 66.f, 81.f, 96.f, 42.f, 51.f, 60.f, 60.f, 72.f, 84.f}));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_2_2)
{
    Shape shape_in1{2};
    Shape shape_in2{2};
    Shape shape_out{};
    auto A = make_shared<op::Parameter>(element::f32, shape_in1);
    auto B = make_shared<op::Parameter>(element::f32, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, false, false);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape_in2);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape_out);

    copy_data(a, vector<float>{1.f, 2.f});
    copy_data(b, vector<float>{1.f, 2.f});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});

    EXPECT_TRUE(test::all_close_f(read_vector<float>(result), vector<float>{5.f}));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_2x2x3_2x1x3_transpose)
{
    Shape shape_in1{2, 2, 3};
    Shape shape_in2{2, 1, 3};
    Shape shape_out{2, 2, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_in1);
    auto B = make_shared<op::Parameter>(element::f32, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, false, true);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape_in2);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape_out);

    vector<float> in1_vec(shape_size(shape_in1));
    vector<float> in2_vec(shape_size(shape_in2));
    // in1_vec is 1.f, 2.f, 3.f, ..., 12.f
    iota(in1_vec.begin(), in1_vec.end(), 1.f);
    // in2_vec is 1.f, 2.f, 3.f, ..., 6.f
    iota(in2_vec.begin(), in2_vec.end(), 1.f);
    copy_data(a, in1_vec);
    copy_data(b, in2_vec);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});

    EXPECT_TRUE(
        test::all_close(read_vector<float>(result), vector<float>{14.f, 32.f, 122.f, 167.f}));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_2x2x3_2x1x3_transpose_int64)
{
    Shape shape_in1{2, 2, 3};
    Shape shape_in2{2, 1, 3};
    Shape shape_out{2, 2, 1};
    auto A = make_shared<op::Parameter>(element::i64, shape_in1);
    auto B = make_shared<op::Parameter>(element::i64, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, false, true);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i64, shape_in1);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i64, shape_in2);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i64, shape_out);

    vector<int64_t> in1_vec(shape_size(shape_in1));
    vector<int64_t> in2_vec(shape_size(shape_in2));
    // in1_vec is 1, 2, 3, ..., 12
    iota(in1_vec.begin(), in1_vec.end(), 1);
    // in2_vec is 1, 2, 3, ..., 6
    iota(in2_vec.begin(), in2_vec.end(), 1);
    copy_data(a, in1_vec);
    copy_data(b, in2_vec);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});

    EXPECT_TRUE(test::all_close(read_vector<int64_t>(result), vector<int64_t>{14, 32, 122, 167}));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_2x2x3_2x3x1_int64)
{
    Shape shape_in1{2, 2, 3};
    Shape shape_in2{2, 3, 1};
    Shape shape_out{2, 2, 1};
    auto A = make_shared<op::Parameter>(element::i64, shape_in1);
    auto B = make_shared<op::Parameter>(element::i64, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, false, false);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i64, shape_in1);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i64, shape_in2);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i64, shape_out);

    vector<int64_t> in1_vec(shape_size(shape_in1));
    vector<int64_t> in2_vec(shape_size(shape_in2));
    // in1_vec is 1, 2, 3, ..., 12
    iota(in1_vec.begin(), in1_vec.end(), 1);
    // in2_vec is 1, 2, 3, ..., 6
    iota(in2_vec.begin(), in2_vec.end(), 1);
    copy_data(a, in1_vec);
    copy_data(b, in2_vec);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});

    EXPECT_TRUE(test::all_close(read_vector<int64_t>(result), vector<int64_t>{14, 32, 122, 167}));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_1x2x3_1x4x3x2)
{
    Shape shape_in1{1, 2, 3};
    Shape shape_in2{1, 4, 3, 2};
    Shape shape_out{1, 4, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_in1);
    auto B = make_shared<op::Parameter>(element::f32, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, false, false);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape_in2);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape_out);

    vector<float> in1_vec(shape_size(shape_in1));
    vector<float> in2_vec(shape_size(shape_in2));
    // Assign in1_vec value with 0.f, 1.f, 2.f, ..., 5.f
    iota(in1_vec.begin(), in1_vec.end(), 0.f);
    // Assign in2_vec value with 0.f, 1.f, 2.f, ..., 23.f
    iota(in2_vec.begin(), in2_vec.end(), 0.f);
    copy_data(a, in1_vec);
    copy_data(b, in2_vec);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});

    EXPECT_TRUE(test::all_close_f(read_vector<float>(result),
                                  vector<float>{10.f,
                                                13.f,
                                                28.f,
                                                40.f,
                                                28.f,
                                                31.f,
                                                100.f,
                                                112.f,
                                                46.f,
                                                49.f,
                                                172.f,
                                                184.f,
                                                64.f,
                                                67.f,
                                                244.f,
                                                256.f}));
}

// 2D x 1D
NGRAPH_TEST(${BACKEND_NAME}, matmul_1_3_x_3_false_false_param)
{
    Shape shape_in1{1, 3};
    Shape shape_in2{3};
    Shape shape_out{1};

    bool transpose_a = false;
    bool transpose_b = false;

    std::vector<float> inputs_a{1, 2, 3};
    std::vector<float> inputs_b{1, 2, 3};
    std::vector<float> expected_result{14.};

    auto A = make_shared<op::Parameter>(element::f32, shape_in1);
    auto B = make_shared<op::Parameter>(element::f32, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, transpose_a, transpose_b);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(inputs_a);
    test_case.add_input<float>(inputs_b);

    test_case.add_expected_output<float>(shape_out, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_3_1_x_3_true_false_param)
{
    Shape shape_in1{3, 1};
    Shape shape_in2{3};
    Shape shape_out{1};

    bool transpose_a = true;
    bool transpose_b = false;

    std::vector<float> inputs_a{1, 2, 3};
    std::vector<float> inputs_b{1, 2, 3};
    std::vector<float> expected_result{14.};

    auto A = make_shared<op::Parameter>(element::f32, shape_in1);
    auto B = make_shared<op::Parameter>(element::f32, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, transpose_a, transpose_b);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(inputs_a);
    test_case.add_input<float>(inputs_b);

    test_case.add_expected_output<float>(shape_out, expected_result);
    test_case.run();
}

// 1D x 2D
NGRAPH_TEST(${BACKEND_NAME}, matmul_3_x_3_1_false_false_param)
{
    Shape shape_in1{3};
    Shape shape_in2{3, 1};
    Shape shape_out{1};

    bool transpose_a = false;
    bool transpose_b = false;

    std::vector<float> inputs_a{1, 2, 3};
    std::vector<float> inputs_b{1, 2, 3};
    std::vector<float> expected_result{14.};

    auto A = make_shared<op::Parameter>(element::f32, shape_in1);
    auto B = make_shared<op::Parameter>(element::f32, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, transpose_a, transpose_b);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(inputs_a);
    test_case.add_input<float>(inputs_b);

    test_case.add_expected_output<float>(shape_out, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_3_x_1_3_false_true_param)
{
    Shape shape_in1{3};
    Shape shape_in2{1, 3};
    Shape shape_out{1};

    bool transpose_a = false;
    bool transpose_b = true;

    std::vector<float> inputs_a{1, 2, 3};
    std::vector<float> inputs_b{1, 2, 3};
    std::vector<float> expected_result{14.};

    auto A = make_shared<op::Parameter>(element::f32, shape_in1);
    auto B = make_shared<op::Parameter>(element::f32, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, transpose_a, transpose_b);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(inputs_a);
    test_case.add_input<float>(inputs_b);

    test_case.add_expected_output<float>(shape_out, expected_result);
    test_case.run();
}
