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

struct FlattenTestParams
{
    // Shape of input tensor to feed to flatten.
    Shape in_shape;

    // Parallel arrays (lengths must be same).
    // - expected_out_shapes[i] is the expected shape of
    //     "flatten(some_tensor_of_in_shape, axis=in_axes[i])"
    vector<size_t> in_axes;
    vector<Shape> expected_out_shapes;
};

struct FlattenTest : ::testing::TestWithParam<FlattenTestParams>
{
};

NGRAPH_TEST_P(${BACKEND_NAME}, FlattenTest, flatten)
{
    FlattenTestParams p = GetParam();

    auto value = make_shared<op::Parameter>(element::i32, p.in_shape);
    auto axis = make_shared<op::Parameter>(element::i64, Shape{});
    auto flattened = builder::flatten(value, axis);
    auto f = make_shared<Function>(NodeVector{flattened}, ParameterVector{value, axis});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);
    auto ex = backend->compile(f);

    auto t_value = backend->create_tensor(element::i32, p.in_shape);

    vector<int32_t> value_data(shape_size(p.in_shape));
    std::iota(value_data.begin(), value_data.end(), 0);
    copy_data(t_value, value_data);

    auto t_axis = backend->create_tensor(element::i64, Shape{});
    auto t_result = backend->create_dynamic_tensor(element::i32, PartialShape::dynamic(2));

    ASSERT_EQ(p.in_axes.size(), p.expected_out_shapes.size());

    for (size_t i = 0; i < p.in_axes.size(); i++)
    {
        copy_data(t_axis, vector<int64_t>{static_cast<int64_t>(p.in_axes[i])});

        ex->call_with_validate({t_result}, {t_value, t_axis});

        ASSERT_EQ(t_result->get_shape(), p.expected_out_shapes[i]);
        ASSERT_EQ(read_vector<int32_t>(t_result), value_data);
    }
}

NGRAPH_INSTANTIATE_TEST_CASE_P(
    ${BACKEND_NAME},
    flatten,
    FlattenTest,
    (::testing::Values(
        FlattenTestParams{
            Shape{2, 3, 4}, {0, 1, 2, 3}, {Shape{1, 24}, Shape{2, 12}, Shape{6, 4}, Shape{24, 1}}},
        FlattenTestParams{
            Shape{38}, {0, 1, 2, 3}, {Shape{1, 38}, Shape{38, 1}, Shape{38, 1}, Shape{38, 1}}},
        FlattenTestParams{Shape{0, 0, 0},
                          {0, 1, 2, 3, 4},
                          {Shape{1, 0}, Shape{0, 0}, Shape{0, 0}, Shape{0, 1}, Shape{0, 1}}},
        FlattenTestParams{Shape{},
                          {0, 1, 2, 3, 4},
                          {Shape{1, 1}, Shape{1, 1}, Shape{1, 1}, Shape{1, 1}, Shape{1, 1}}})));
