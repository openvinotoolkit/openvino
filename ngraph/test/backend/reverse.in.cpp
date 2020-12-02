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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, reverse_1d)
{
    Shape shape{8};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(
        make_shared<op::v1::Reverse>(
            A, op::Constant::create(element::Type_t::i64, {1}, {0}), op::v1::Reverse::Mode::INDEX),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7});
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{7, 6, 5, 4, 3, 2, 1, 0}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_0)
{
    Shape shape{4, 3};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(
        make_shared<op::v1::Reverse>(
            A, op::Constant::create(element::Type_t::i64, {1}, {0}), op::v1::Reverse::Mode::INDEX),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector());
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (test::NDArray<float, 2>({{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}}).get_vector()),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_1)
{
    Shape shape{4, 3};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(
        make_shared<op::v1::Reverse>(
            A, op::Constant::create(element::Type_t::i64, {1}, {1}), op::v1::Reverse::Mode::INDEX),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector());
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (test::NDArray<float, 2>({{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}}).get_vector()),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_1_mask)
{
    Shape shape{4, 3};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(
        make_shared<op::v1::Reverse>(
            A,
            op::Constant::create(element::Type_t::boolean, {2}, {false, true}),
            op::v1::Reverse::Mode::MASK),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector());
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (test::NDArray<float, 2>({{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}}).get_vector()),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_01)
{
    Shape shape{4, 3};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(
        make_shared<op::v1::Reverse>(A,
                                     op::Constant::create(element::Type_t::i64, {2}, {0, 1}),
                                     op::v1::Reverse::Mode::INDEX),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector());
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (test::NDArray<float, 2>({{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}}).get_vector()),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_01_mask)
{
    Shape shape{4, 3};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::v1::Reverse>(
                                  A,
                                  op::Constant::create(element::Type_t::boolean, {2}, {true, true}),
                                  op::v1::Reverse::Mode::MASK),
                              ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector());
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (test::NDArray<float, 2>({{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}}).get_vector()),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_0)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(
        make_shared<op::v1::Reverse>(
            A, op::Constant::create(element::Type_t::i64, {1}, {0}), op::v1::Reverse::Mode::INDEX),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (test::NDArray<float, 3>({{{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}},
                                  {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}})
             .get_vector()),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_1)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(
        make_shared<op::v1::Reverse>(
            A, op::Constant::create(element::Type_t::i64, {1}, {1}), op::v1::Reverse::Mode::INDEX),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (test::NDArray<float, 3>({{{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}},
                                  {{21, 22, 23}, {18, 19, 20}, {15, 16, 17}, {12, 13, 14}}})
             .get_vector()),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_2)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(
        make_shared<op::v1::Reverse>(
            A, op::Constant::create(element::Type_t::i64, {1}, {2}), op::v1::Reverse::Mode::INDEX),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (test::NDArray<float, 3>({{{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}},
                                  {{14, 13, 12}, {17, 16, 15}, {20, 19, 18}, {23, 22, 21}}})
             .get_vector()),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_01)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(
        make_shared<op::v1::Reverse>(A,
                                     op::Constant::create(element::Type_t::i64, {2}, {0, 1}),
                                     op::v1::Reverse::Mode::INDEX),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (test::NDArray<float, 3>({{{21, 22, 23}, {18, 19, 20}, {15, 16, 17}, {12, 13, 14}},
                                  {{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}}})
             .get_vector()),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_02)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(
        make_shared<op::v1::Reverse>(A,
                                     op::Constant::create(element::Type_t::i64, {2}, {0, 2}),
                                     op::v1::Reverse::Mode::INDEX),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (test::NDArray<float, 3>({{{14, 13, 12}, {17, 16, 15}, {20, 19, 18}, {23, 22, 21}},
                                  {{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}}})
             .get_vector()),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_12)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(
        make_shared<op::v1::Reverse>(A,
                                     op::Constant::create(element::Type_t::i64, {2}, {1, 2}),
                                     op::v1::Reverse::Mode::INDEX),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (test::NDArray<float, 3>({{{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}},
                                  {{23, 22, 21}, {20, 19, 18}, {17, 16, 15}, {14, 13, 12}}})
             .get_vector()),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_012)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto f = make_shared<Function>(
        make_shared<op::v1::Reverse>(A,
                                     op::Constant::create(element::Type_t::i64, {3}, {0, 1, 2}),
                                     op::v1::Reverse::Mode::INDEX),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::Type_t::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (test::NDArray<float, 3>({{{23, 22, 21}, {20, 19, 18}, {17, 16, 15}, {14, 13, 12}},
                                  {{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}}})
             .get_vector()),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_v1_incorrect_rev_axes_rank_index_mode)
{
    const auto Data = make_shared<op::Parameter>(element::Type_t::f32, Shape{2, 2, 2});
    const auto Rev_Axes =
        make_shared<op::Parameter>(element::Type_t::i64, Shape{1, 1}); // correct: 1D

    EXPECT_THROW(make_shared<Function>(
                     make_shared<op::v1::Reverse>(Data, Rev_Axes, op::v1::Reverse::Mode::INDEX),
                     ParameterVector{Data, Rev_Axes}),
                 ngraph::NodeValidationFailure);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_v1_incorrect_rev_axes_elems_mask_mode)
{
    const auto Data = make_shared<op::Parameter>(element::Type_t::f32, Shape{2, 2, 2});
    const auto Rev_Axes =
        make_shared<op::Parameter>(element::Type_t::boolean, Shape{2}); // correct: 3

    EXPECT_THROW(make_shared<op::v1::Reverse>(Data, Rev_Axes, op::v1::Reverse::Mode::MASK),
                 ngraph::NodeValidationFailure);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_v1_axes_out_of_bounds)
{
    const auto Data = make_shared<op::Parameter>(element::Type_t::f32, Shape{2, 2, 2});
    const auto Rev_Axes = op::Constant::create(element::Type_t::i64, Shape{2}, {1, 10});

    EXPECT_THROW(make_shared<op::v1::Reverse>(Data, Rev_Axes, op::v1::Reverse::Mode::INDEX),
                 ngraph::NodeValidationFailure);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_v1_too_many_axes)
{
    const auto Data = make_shared<op::Parameter>(element::Type_t::f32, Shape{2, 2, 2});
    const auto Rev_Axes = op::Constant::create(element::Type_t::i64, Shape{4}, {0, 1, 2, 3});

    EXPECT_THROW(make_shared<op::v1::Reverse>(Data, Rev_Axes, op::v1::Reverse::Mode::INDEX),
                 ngraph::NodeValidationFailure);
}
