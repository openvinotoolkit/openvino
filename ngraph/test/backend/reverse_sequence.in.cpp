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

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, reverse_sequence_n2c3h4w2)
{
    Shape shape{2, 3, 4, 2};
    Shape seq_len_shape{4};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, seq_len_shape);

    size_t batch_axis = 2;
    size_t sequence_axis = 1;
    auto rs = std::make_shared<op::ReverseSequence>(A, B, batch_axis, sequence_axis);

    auto f = make_shared<Function>(rs, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, seq_len_shape);

    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    std::vector<int> input{
        0,  0, 3,  0, 6,  0, 9,  0, 1,  0, 4,  0, 7,  0, 10, 0, 2,  0, 5,  0, 8,  0, 11, 0,
        12, 0, 15, 0, 18, 0, 21, 0, 13, 0, 16, 0, 19, 0, 22, 0, 14, 0, 17, 0, 20, 0, 23, 0,
    };

    std::vector<int> seq_lengths{1, 2, 1, 2};
    copy_data(b, seq_lengths);

    std::vector<int> expected{
        0,  0, 4,  0, 6,  0, 10, 0, 1,  0, 3,  0, 7,  0, 9,  0, 2,  0, 5,  0, 8,  0, 11, 0,

        12, 0, 16, 0, 18, 0, 22, 0, 13, 0, 15, 0, 19, 0, 21, 0, 14, 0, 17, 0, 20, 0, 23, 0};

    copy_data(a, input);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ(read_vector<int>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_sequence_n4c3h2w2)
{
    Shape shape{4, 3, 2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    Shape seq_len_shape{4};
    auto B = make_shared<op::Parameter>(element::i32, seq_len_shape);

    size_t batch_axis = 0;
    size_t sequence_axis = 1;

    auto rs = std::make_shared<op::ReverseSequence>(A, B, batch_axis, sequence_axis);

    auto f = make_shared<Function>(rs, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, seq_len_shape);

    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    std::vector<int> seq_lengths{1, 2, 3, 3};
    copy_data(b, seq_lengths);

    std::vector<int> input{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                           32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};

    std::vector<int> expected{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 16, 17, 18, 19,
                              12, 13, 14, 15, 20, 21, 22, 23, 32, 33, 34, 35, 28, 29, 30, 31,
                              24, 25, 26, 27, 44, 45, 46, 47, 40, 41, 42, 43, 36, 37, 38, 39};

    copy_data(a, input);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ(read_vector<int>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_sequence_n4d2c3h2w2)
{
    Shape shape{4, 2, 3, 2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    Shape seq_len_shape{4};
    auto B = make_shared<op::Parameter>(element::i32, seq_len_shape);

    size_t batch_axis = 0;
    size_t sequence_axis = 2;

    auto rs = std::make_shared<op::ReverseSequence>(A, B, batch_axis, sequence_axis);

    auto f = make_shared<Function>(rs, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, seq_len_shape);

    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    std::vector<int> input{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                           32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                           48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                           64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                           80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};

    std::vector<int> expected{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                              16, 17, 18, 19, 20, 21, 22, 23, 28, 29, 30, 31, 24, 25, 26, 27,
                              32, 33, 34, 35, 40, 41, 42, 43, 36, 37, 38, 39, 44, 45, 46, 47,
                              48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                              64, 65, 66, 67, 68, 69, 70, 71, 76, 77, 78, 79, 72, 73, 74, 75,
                              80, 81, 82, 83, 88, 89, 90, 91, 84, 85, 86, 87, 92, 93, 94, 95};

    copy_data(a, input);

    std::vector<int> seq_lengths{1, 2, 1, 2};
    copy_data(b, seq_lengths);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ(read_vector<int>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_sequence_negative_axes)
{
    Shape shape{2, 3, 4, 2};
    Shape seq_len_shape{4};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, seq_len_shape);

    int64_t batch_axis = -2;
    int64_t sequence_axis = -3;
    auto rs = std::make_shared<op::ReverseSequence>(A, B, batch_axis, sequence_axis);

    auto f = make_shared<Function>(rs, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, seq_len_shape);

    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    std::vector<int> input{
        0,  0, 3,  0, 6,  0, 9,  0, 1,  0, 4,  0, 7,  0, 10, 0, 2,  0, 5,  0, 8,  0, 11, 0,
        12, 0, 15, 0, 18, 0, 21, 0, 13, 0, 16, 0, 19, 0, 22, 0, 14, 0, 17, 0, 20, 0, 23, 0,
    };

    std::vector<int> seq_lengths{1, 2, 1, 2};
    copy_data(b, seq_lengths);

    std::vector<int> expected{
        0,  0, 4,  0, 6,  0, 10, 0, 1,  0, 3,  0, 7,  0, 9,  0, 2,  0, 5,  0, 8,  0, 11, 0,

        12, 0, 16, 0, 18, 0, 22, 0, 13, 0, 15, 0, 19, 0, 21, 0, 14, 0, 17, 0, 20, 0, 23, 0};

    copy_data(a, input);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ(read_vector<int>(result), expected);
}
