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
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_1d)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{1}, {4});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{1}, {5});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::Type_t::f32, Shape{15});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(
        test::all_close_f({2112, 2112, 2112, 2112, 1, 2, 3, 4, 5, 6, 2112, 2112, 2112, 2112, 2112},
                          read_vector<float>(result),
                          MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_1d)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{1}, {4});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{1}, {-2});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::Type_t::f32, Shape{8});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f({2112, 2112, 2112, 2112, 1, 2, 3, 4},
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_1d_check_limits)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{1}, {4});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{1}, {-7});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::Type_t::f32, Shape{3});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        {2112, 2112, 2112}, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{1}, {2});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{1}, {3});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::EDGE),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::Type_t::f32, Shape{11});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        {1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6}, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d_top_neg)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{1}, {2});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{1}, {-3});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::EDGE),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::Type_t::f32, Shape{5});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(
        test::all_close_f({1, 1, 1, 2, 3}, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d_top_neg_bigger_than_tensor)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{1}, {2});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{1}, {-7});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::EDGE),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::Type_t::f32, Shape{1});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f({1}, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d_bottom_neg)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{1}, {-2});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{1}, {3});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::EDGE),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::Type_t::f32, Shape{7});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        {3, 4, 5, 6, 6, 6, 6}, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d_bottom_neg_bigger_than_tensor)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{1}, {-7});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{1}, {3});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::EDGE),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::Type_t::f32, Shape{2});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f({6, 6}, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_2d)
{
    const Shape data_shape{3, 4};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{2}, {2, 3});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{2}, {1, 2});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::EDGE),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
    auto result = backend->create_tensor(element::Type_t::f32, Shape{6, 9});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 2>({{1, 1, 1, 1, 2, 3, 4, 4, 4},
                                                           {1, 1, 1, 1, 2, 3, 4, 4, 4},
                                                           {1, 1, 1, 1, 2, 3, 4, 4, 4},
                                                           {5, 5, 5, 5, 6, 7, 8, 8, 8},
                                                           {9, 9, 9, 9, 10, 11, 12, 12, 12},
                                                           {9, 9, 9, 9, 10, 11, 12, 12, 12}})
                                      .get_vector(),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_2d_with_neg)
{
    const Shape data_shape{3, 4};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{2}, {2, -1});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{2}, {1, 2});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::EDGE),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
    auto result = backend->create_tensor(element::Type_t::f32, Shape{6, 5});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 2>({{2, 3, 4, 4, 4},
                                                           {2, 3, 4, 4, 4},
                                                           {2, 3, 4, 4, 4},
                                                           {6, 7, 8, 8, 8},
                                                           {10, 11, 12, 12, 12},
                                                           {10, 11, 12, 12, 12}})
                                      .get_vector(),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{1}, {2});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{1}, {3});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::REFLECT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::Type_t::f32, Shape{11});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(std::vector<float>({3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_top_neg)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{1}, {2});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{1}, {-3});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::REFLECT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::Type_t::f32, Shape{5});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        std::vector<float>({3, 2, 1, 2, 3}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_top_neg_bigger_than_tensor)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{1}, {2});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{1}, {-7});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::REFLECT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::Type_t::f32, Shape{1});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        std::vector<float>({3}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_bottom_neg)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{1}, {-2});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{1}, {3});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::REFLECT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::Type_t::f32, Shape{7});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(std::vector<float>({3, 4, 5, 6, 5, 4, 3}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_bottom_neg_bigger_than_tensor)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{1}, {-7});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{1}, {3});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::REFLECT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::Type_t::f32, Shape{2});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        std::vector<float>({4, 3}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_multi_reflect)
{
    const Shape data_shape{3};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{1}, {10});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{1}, {9});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::REFLECT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3}));
    auto result = backend->create_tensor(element::Type_t::f32, Shape{22});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        std::vector<float>({3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2}),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_2d)
{
    const Shape data_shape{3, 4};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{2}, {2, 3});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{2}, {1, 2});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::REFLECT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a,
              test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}).get_vector());
    auto result = backend->create_tensor(element::Type_t::f32, Shape{6, 9});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 2>({{12, 11, 10, 9, 10, 11, 12, 11, 10},
                                                           {8, 7, 6, 5, 6, 7, 8, 7, 6},
                                                           {4, 3, 2, 1, 2, 3, 4, 3, 2},
                                                           {8, 7, 6, 5, 6, 7, 8, 7, 6},
                                                           {12, 11, 10, 9, 10, 11, 12, 11, 10},
                                                           {8, 7, 6, 5, 6, 7, 8, 7, 6}})
                                      .get_vector(),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_2d_with_neg)
{
    const Shape data_shape{3, 4};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{2}, {2, -1});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{2}, {1, 2});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::REFLECT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a,
              test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}).get_vector());
    auto result = backend->create_tensor(element::Type_t::f32, Shape{6, 5});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 2>({{10, 11, 12, 11, 10},
                                                           {6, 7, 8, 7, 6},
                                                           {2, 3, 4, 3, 2},
                                                           {6, 7, 8, 7, 6},
                                                           {10, 11, 12, 11, 10},
                                                           {6, 7, 8, 7, 6}})
                                      .get_vector(),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_2d)
{
    const Shape data_shape{2, 3};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{2}, {1, -1});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{2}, {2, 0});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {9});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
    auto result = backend->create_tensor(element::Type_t::f32, Shape{5, 2});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        test::NDArray<float, 2>({{9, 9}, {2, 3}, {5, 6}, {9, 9}, {9, 9}}).get_vector(),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_2d_all_negative)
{
    const Shape data_shape{3, 3};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{2}, {-1, -1});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{2}, {-1, -1});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {9});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}).get_vector());
    auto result = backend->create_tensor(element::Type_t::f32, Shape{1, 1});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 2>({{5}}).get_vector(),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_0x0)
{
    const Shape data_shape{0, 0};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{2}, {2, 3});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{2}, {3, 2});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    auto result = backend->create_tensor(element::Type_t::f32, Shape{5, 5});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112}})
                                      .get_vector(),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_0x3)
{
    const Shape data_shape{0, 3};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{2}, {2, 1});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{2}, {3, 1});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    auto result = backend->create_tensor(element::Type_t::f32, Shape{5, 5});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112}})
                                      .get_vector(),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_3x0)
{
    const Shape data_shape{3, 0};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{2}, {1, 3});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{2}, {1, 2});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    auto result = backend->create_tensor(element::Type_t::f32, Shape{5, 5});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112}})
                                      .get_vector(),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_4d_1x2x2x2)
{
    const Shape data_shape{1, 2, 2, 2};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{4}, {0, 0, 1, 1});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{4}, {0, 0, 1, 1});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {42});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    // clang-format off
    copy_data(a, test::NDArray<float, 4>(
        {
            {
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                },
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                }
            }
        }).get_vector());
    // clang-format on
    auto result = backend->create_tensor(element::Type_t::f32, Shape{1, 2, 4, 4});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    // clang-format off
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 4>(
        {
            {
                {
                    {42.0f, 42.0f, 42.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 42.0f, 42.0f, 42.0f}
                },
                {
                    {42.0f, 42.0f, 42.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 42.0f, 42.0f, 42.0f}
                }
            }
        }).get_vector()),
        read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
    // clang-format on
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_4d)
{
    const Shape data_shape{1, 3, 2, 2};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{4}, {0, -1, 1, 1});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{4}, {0, -1, 1, 1});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {42});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    // clang-format off
    copy_data(a, test::NDArray<float, 4>(
        {
            {
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                },
                {
                    {1.0f, 1.0f},
                    {1.0f, 1.0f}
                },
                {
                    {2.0f, 2.0f},
                    {2.0f, 2.0f}
                }
            }
        }).get_vector());
    // clang-format on

    auto result = backend->create_tensor(element::Type_t::f32, Shape{1, 1, 4, 4});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    // clang-format off
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 4>(
        {
            {
                {
                    {42.0f, 42.0f, 42.0f, 42.0f},
                    {42.0f, 1.0f, 1.0f, 42.0f},
                    {42.0f, 1.0f, 1.0f, 42.0f},
                    {42.0f, 42.0f, 42.0f, 42.0f}
                }
            }
        }).get_vector()),
        read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
    // clang-format on
}

// This test covers the case with multiple image and with asymetric pad
// bug has been found on nvGPU side now covered by this test
NGRAPH_TEST(${BACKEND_NAME}, pad_2channel_2image_asym)
{
    const Shape data_shape{2, 2, 4, 4};
    const auto window_movement_strides = Strides{2, 2};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{4}, {0, 0, 0, 0});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{4}, {0, 0, 2, 2});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {42});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2}, // img 0 chan 0
                                         {0, 3, 2, 0},
                                         {2, 0, 0, 0},
                                         {0, 2, 1, 0}},

                                        {{0, 0, 0, 2}, // img 0 chan 1
                                         {0, 2, 3, 0},
                                         {2, 0, 1, 0},
                                         {2, 0, 0, 0}}},

                                       {{{0, 2, 1, 1}, // img 1 chan 0
                                         {0, 0, 2, 0},
                                         {0, 0, 1, 2},
                                         {0, 0, 0, 0}},

                                        {{2, 1, 0, 0}, // img 1 chan 1
                                         {0, 2, 0, 0},
                                         {1, 1, 2, 0},
                                         {1, 0, 0, 0}}}})
                  .get_vector());

    auto result = backend->create_tensor(element::Type_t::f32, Shape{2, 2, 6, 6});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 4>({{{{0, 1, 0, 2, 42, 42}, // img 0 chan 0
                                                              {0, 3, 2, 0, 42, 42},
                                                              {2, 0, 0, 0, 42, 42},
                                                              {0, 2, 1, 0, 42, 42},
                                                              {42, 42, 42, 42, 42, 42},
                                                              {42, 42, 42, 42, 42, 42}},

                                                             {{0, 0, 0, 2, 42, 42}, // img 1 chan 0
                                                              {0, 2, 3, 0, 42, 42},
                                                              {2, 0, 1, 0, 42, 42},
                                                              {2, 0, 0, 0, 42, 42},
                                                              {42, 42, 42, 42, 42, 42},
                                                              {42, 42, 42, 42, 42, 42}}},

                                                            {{{0, 2, 1, 1, 42, 42}, // img 1 chan 0
                                                              {0, 0, 2, 0, 42, 42},
                                                              {0, 0, 1, 2, 42, 42},
                                                              {0, 0, 0, 0, 42, 42},
                                                              {42, 42, 42, 42, 42, 42},
                                                              {42, 42, 42, 42, 42, 42}},

                                                             {{2, 1, 0, 0, 42, 42}, // img 1 chan 1
                                                              {0, 2, 0, 0, 42, 42},
                                                              {1, 1, 2, 0, 42, 42},
                                                              {1, 0, 0, 0, 42, 42},
                                                              {42, 42, 42, 42, 42, 42},
                                                              {42, 42, 42, 42, 42, 42}}}})
                                       .get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_symmetric)
{
    const Shape data_shape{2, 3};
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::Type_t::i64, Shape{2}, {1, 2});
    const auto pads_end = op::Constant::create(element::Type_t::i64, Shape{2}, {1, 2});
    const auto pad_val = op::Constant::create(element::Type_t::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::SYMMETRIC),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, data_shape);
    copy_data(a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
    auto result = backend->create_tensor(element::Type_t::f32, Shape{4, 7});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 2>({{2, 1, 1, 2, 3, 3, 2},
                                                            {2, 1, 1, 2, 3, 3, 2},
                                                            {5, 4, 4, 5, 6, 6, 5},
                                                            {5, 4, 4, 5, 6, 6, 5}})
                                       .get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}
