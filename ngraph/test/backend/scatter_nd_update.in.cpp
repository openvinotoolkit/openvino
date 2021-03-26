//*****************************************************************************
// Copyright 2021 Intel Corporation
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

#include <memory>
#include <string>

#include "ngraph/opsets/opset7.hpp"
#include "ngraph/type/element_type.hpp"

#include "runtime/backend.hpp"

#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

static std::string s_manifest = "${MANIFEST}";

namespace
{
    template <typename ValueType>
    struct Array
    {
        using StorageType = ngraph::test::NDArrayBase<ValueType>;
        static ngraph::element::Type element_type() { return ngraph::element::from<ValueType>(); }
        StorageType data;
    };
    struct Params
    {
        Array<float> input;
        Array<int32_t> indices;
        Array<float> updates;
        Array<float> expected_output;
    };

    void execute_test(const Params& p)
    {
        using namespace ngraph;
        using namespace opset7;

        auto inputs = std::make_shared<Parameter>(p.input.element_type(), p.input.data.get_shape());
        auto indices = Constant::create(
            p.indices.element_type(), p.indices.data.get_shape(), p.indices.data.get_vector());
        auto updates = Constant::create(
            p.updates.element_type(), p.updates.data.get_shape(), p.updates.data.get_vector());

        auto scatter = std::make_shared<ScatterNDUpdate>(inputs, indices, updates);

        auto function =
            std::make_shared<Function>(scatter, ParameterVector{inputs}, "ScatterNDUpdate");

        auto backend = runtime::Backend::create("${BACKEND_NAME}");

        // Create some tensors for input/output
        auto inputs_tensor =
            backend->create_tensor(p.input.element_type(), p.input.data.get_shape());
        copy_data(inputs_tensor, p.input.data.get_vector());

        auto result =
            backend->create_tensor(p.input.element_type(), p.expected_output.data.get_shape());

        auto handle = backend->compile(function);
        handle->call_with_validate({result}, {inputs_tensor});

        EXPECT_TRUE(test::all_close_f(p.expected_output.data.get_vector(),
                                      read_vector<float>(result),
                                      MIN_FLOAT_TOLERANCE_BITS));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, scatter_nd_update_1x1)
{
    using namespace ngraph::test;
    execute_test(Params{NDArray<float, 1>{1},
                        NDArray<int32_t, 2>{{0}},
                        NDArray<float, 1>{20},
                        NDArray<float, 1>{20}});
}

NGRAPH_TEST(${BACKEND_NAME}, scatter_nd_update_2x2_by_1)
{
    using namespace ngraph::test;
    execute_test(Params{NDArray<float, 2>{
                            {1, 2},
                            {3, 4},
                        },
                        NDArray<int32_t, 2>{{1}, {0}},
                        NDArray<float, 2>{{10, 20}, {30, 40}},
                        NDArray<float, 2>{
                            {30, 40},
                            {10, 20},
                        }});
}

NGRAPH_TEST(${BACKEND_NAME}, scatter_nd_update_2x2_by_2)
{
    using namespace ngraph::test;
    execute_test(Params{NDArray<float, 2>{
                            {1, 2},
                            {3, 4},
                        },
                        NDArray<int32_t, 2>{
                            {0, 0},
                            {1, 1},
                        },
                        NDArray<float, 1>{10, 40},
                        NDArray<float, 2>{
                            {10, 2},
                            {3, 40},
                        }});
}

NGRAPH_TEST(${BACKEND_NAME}, scatter_nd_update_3x3_by_1)
{
    using namespace ngraph::test;
    execute_test(Params{NDArray<float, 3>{
                            {
                                {11, 12, 13},
                                {14, 15, 16},
                                {17, 18, 19},
                            },
                            {
                                {21, 22, 23},
                                {24, 25, 26},
                                {27, 28, 29},
                            },
                            {
                                {31, 32, 33},
                                {34, 35, 36},
                                {37, 38, 39},
                            },
                        },
                        NDArray<int32_t, 2>{{0}, {2}},
                        NDArray<float, 3>{
                            {
                                {91, 92, 93},
                                {94, 95, 96},
                                {97, 98, 99},
                            },
                            {
                                {81, 82, 83},
                                {84, 85, 86},
                                {87, 88, 89},
                            },
                        },
                        NDArray<float, 3>{
                            {
                                {91, 92, 93},
                                {94, 95, 96},
                                {97, 98, 99},
                            },
                            {
                                {21, 22, 23},
                                {24, 25, 26},
                                {27, 28, 29},
                            },
                            {
                                {81, 82, 83},
                                {84, 85, 86},
                                {87, 88, 89},
                            },
                        }});
}

NGRAPH_TEST(${BACKEND_NAME}, scatter_nd_update_3x3_by_2v2)
{
    using namespace ngraph::test;
    execute_test(Params{NDArray<float, 3>{
                            {
                                {11, 12, 13},
                                {14, 15, 16},
                                {17, 18, 19},
                            },
                            {
                                {21, 22, 23},
                                {24, 25, 26},
                                {27, 28, 29},
                            },
                            {
                                {31, 32, 33},
                                {34, 35, 36},
                                {37, 38, 39},
                            },
                        },
                        NDArray<int32_t, 3>{
                            {
                                {0, 0, 0},
                                {2, 2, 2},
                            },
                            {
                                {1, 0, 0},
                                {1, 2, 2},
                            },
                        },
                        NDArray<float, 2>{
                            {91, 92},
                            {81, 82},
                        },
                        NDArray<float, 3>{
                            {
                                {91, 12, 13},
                                {14, 15, 16},
                                {17, 18, 19},
                            },
                            {
                                {81, 22, 23},
                                {24, 25, 26},
                                {27, 28, 82},
                            },
                            {
                                {31, 32, 33},
                                {34, 35, 36},
                                {37, 38, 92},
                            },
                        }});
}

NGRAPH_TEST(${BACKEND_NAME}, scatter_nd_update_3x3_by_2)
{
    using namespace ngraph::test;
    execute_test(Params{NDArray<float, 3>{
                            {
                                {11, 12, 13},
                                {14, 15, 16},
                                {17, 18, 19},
                            },
                            {
                                {21, 22, 23},
                                {24, 25, 26},
                                {27, 28, 29},
                            },
                            {
                                {31, 32, 33},
                                {34, 35, 36},
                                {37, 38, 39},
                            },
                        },
                        NDArray<int32_t, 2>{{0, 0}, {2, 2}},
                        NDArray<float, 2>{
                            {91, 92, 93},
                            {87, 88, 89},
                        },
                        NDArray<float, 3>{
                            {
                                {91, 92, 93},
                                {14, 15, 16},
                                {17, 18, 19},
                            },
                            {
                                {21, 22, 23},
                                {24, 25, 26},
                                {27, 28, 29},
                            },
                            {
                                {31, 32, 33},
                                {34, 35, 36},
                                {87, 88, 89},
                            },
                        }});
}

NGRAPH_TEST(${BACKEND_NAME}, scatter_nd_update_3x3_by_3)
{
    using namespace ngraph::test;
    execute_test(Params{NDArray<float, 3>{
                            {
                                {11, 12, 13},
                                {14, 15, 16},
                                {17, 18, 19},
                            },
                            {
                                {21, 22, 23},
                                {24, 25, 26},
                                {27, 28, 29},
                            },
                            {
                                {31, 32, 33},
                                {34, 35, 36},
                                {37, 38, 39},
                            },
                        },
                        NDArray<int32_t, 2>{{0, 0, 0}, {2, 2, 2}},
                        NDArray<float, 1>{91, 99},
                        NDArray<float, 3>{
                            {
                                {91, 12, 13},
                                {14, 15, 16},
                                {17, 18, 19},
                            },
                            {
                                {21, 22, 23},
                                {24, 25, 26},
                                {27, 28, 29},
                            },
                            {
                                {31, 32, 33},
                                {34, 35, 36},
                                {37, 38, 99},
                            },
                        }});
}

NGRAPH_TEST(${BACKEND_NAME}, scatter_nd_update_1d_from_examples)
{
    using namespace ngraph::test;
    execute_test(Params{NDArray<float, 1>{1, 2, 3, 4, 5, 6, 7, 8},
                        NDArray<int32_t, 2>{{4}, {3}, {1}, {7}},
                        NDArray<float, 1>{9, 10, 11, 12},
                        NDArray<float, 1>{1, 11, 3, 10, 9, 6, 7, 12}});
}

NGRAPH_TEST(${BACKEND_NAME}, scatter_nd_update_4x4_shape_from_examples)
{
    using namespace ngraph::test;
    execute_test(Params{NDArray<float, 3>{{
                                              {1, 2, 3, 4},
                                              {5, 6, 7, 8},
                                              {8, 7, 6, 5},
                                              {4, 3, 2, 1},
                                          },
                                          {
                                              {1, 2, 3, 4},
                                              {5, 6, 7, 8},
                                              {8, 7, 6, 5},
                                              {4, 3, 2, 1},
                                          },
                                          {
                                              {8, 7, 6, 5},
                                              {4, 3, 2, 1},
                                              {1, 2, 3, 4},
                                              {5, 6, 7, 8},
                                          },
                                          {
                                              {8, 7, 6, 5},
                                              {4, 3, 2, 1},
                                              {1, 2, 3, 4},
                                              {5, 6, 7, 8},
                                          }},
                        NDArray<int32_t, 2>{{0}, {2}},
                        NDArray<float, 3>{
                            {
                                {5, 5, 5, 5},
                                {6, 6, 6, 6},
                                {7, 7, 7, 7},
                                {8, 8, 8, 8},
                            },
                            {
                                {1, 1, 1, 1},
                                {2, 2, 2, 2},
                                {3, 3, 3, 3},
                                {4, 4, 4, 4},
                            },
                        },
                        NDArray<float, 3>{
                            {
                                {5, 5, 5, 5},
                                {6, 6, 6, 6},
                                {7, 7, 7, 7},
                                {8, 8, 8, 8},
                            },
                            {
                                {1, 2, 3, 4},
                                {5, 6, 7, 8},
                                {8, 7, 6, 5},
                                {4, 3, 2, 1},
                            },
                            {
                                {1, 1, 1, 1},
                                {2, 2, 2, 2},
                                {3, 3, 3, 3},
                                {4, 4, 4, 4},
                            },
                            {
                                {8, 7, 6, 5},
                                {4, 3, 2, 1},
                                {1, 2, 3, 4},
                                {5, 6, 7, 8},
                            },
                        }});
}

NGRAPH_TEST(${BACKEND_NAME}, scatter_nd_update_4x4_v2)
{
    using namespace ngraph::test;
    execute_test(Params{NDArray<float, 3>{
                            {
                                {1, 2, 3, 4},
                                {5, 6, 7, 8},
                                {8, 7, 6, 5},
                                {4, 3, 2, 1},
                            },
                            {
                                {1, 2, 3, 4},
                                {5, 6, 7, 8},
                                {8, 7, 6, 5},
                                {4, 3, 2, 1},
                            },
                            {
                                {8, 7, 6, 5},
                                {4, 3, 2, 1},
                                {1, 2, 3, 4},
                                {5, 6, 7, 8},
                            },
                            {
                                {8, 7, 6, 5},
                                {4, 3, 2, 1},
                                {1, 2, 3, 4},
                                {5, 6, 7, 8},
                            },
                        },
                        NDArray<int32_t, 3>{
                            {
                                {0, 0},
                                {2, 2},
                            },
                            {
                                {1, 1},
                                {3, 3},
                            },
                        },
                        NDArray<float, 3>{
                            {
                                {15, 16, 17, 18},
                                {25, 26, 27, 28},
                            },
                            {
                                {35, 36, 37, 38},
                                {45, 46, 47, 58},
                            },
                        },
                        NDArray<float, 3>{
                            {
                                {15, 16, 17, 18},
                                {5, 6, 7, 8},
                                {8, 7, 6, 5},
                                {4, 3, 2, 1},

                            },
                            {
                                {1, 2, 3, 4},
                                {35, 36, 37, 38},
                                {8, 7, 6, 5},
                                {4, 3, 2, 1},
                            },
                            {
                                {8, 7, 6, 5},
                                {4, 3, 2, 1},
                                {25, 26, 27, 28},
                                {5, 6, 7, 8},
                            },
                            {
                                {8, 7, 6, 5},
                                {4, 3, 2, 1},
                                {1, 2, 3, 4},
                                {45, 46, 47, 58},
                            },
                        }});
}
