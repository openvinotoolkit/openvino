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
#include "util/autodiff/numeric_compare.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, partial_slice_static)
{
    Shape shape_x{2, 3, 2};
    auto x = make_shared<op::Parameter>(element::f32, shape_x);
    AxisVector axes{0, 1};
    vector<int64_t> lower_bounds{1, 0};
    vector<int64_t> upper_bounds{2, 2};
    AxisVector decrease_axes{};
    auto f = make_shared<Function>(
        make_shared<op::PartialSlice>(x, axes, lower_bounds, upper_bounds, decrease_axes),
        ParameterVector{x});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto t_x = backend->create_tensor(element::f32, shape_x);
    vector<float> v_x{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f};
    copy_data(t_x, v_x);
    auto t_r = backend->create_tensor(element::f32, Shape{1, 2, 2});

    auto handle = backend->compile(f);
    handle->call_with_validate({t_r}, {t_x});
    vector<float> v_r{6.f, 7.f, 8.f, 9.f};
    ASSERT_EQ(t_r->get_shape(), (Shape{1, 2, 2}));
    EXPECT_TRUE(test::all_close_f(v_r, read_vector<float>(t_r)));
}

NGRAPH_TEST(${BACKEND_NAME}, partial_slice_partial_shape)
{
    auto pshape_x = PartialShape{Dimension::dynamic(), 3, Dimension::dynamic()};
    auto x = make_shared<op::Parameter>(element::f32, pshape_x);
    AxisVector axes{0, 1};
    vector<int64_t> lower_bounds{1, 0};
    vector<int64_t> upper_bounds{2, 2};
    AxisVector decrease_axes{};
    auto f = make_shared<Function>(
        make_shared<op::PartialSlice>(x, axes, lower_bounds, upper_bounds, decrease_axes),
        ParameterVector{x});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    Shape shape_x{2, 3, 2};
    auto t_x = backend->create_tensor(element::f32, shape_x);
    vector<float> v_x{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f};
    copy_data(t_x, v_x);
    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    auto handle = backend->compile(f);
    handle->call_with_validate({t_r}, {t_x});
    vector<float> v_r{6.f, 7.f, 8.f, 9.f};
    ASSERT_EQ(t_r->get_shape(), (Shape{1, 2, 2}));
    EXPECT_TRUE(test::all_close_f(v_r, read_vector<float>(t_r)));
}

NGRAPH_TEST(${BACKEND_NAME}, partial_slice_unkown_rank)
{
    auto pshape_x = PartialShape::dynamic();
    auto x = make_shared<op::Parameter>(element::f32, pshape_x);
    AxisVector axes{0, 1};
    vector<int64_t> lower_bounds{1, 0};
    vector<int64_t> upper_bounds{2, 2};
    AxisVector decrease_axes{};
    auto f = make_shared<Function>(
        make_shared<op::PartialSlice>(x, axes, lower_bounds, upper_bounds, decrease_axes),
        ParameterVector{x});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    Shape shape_x{2, 3, 2};
    auto t_x = backend->create_tensor(element::f32, shape_x);
    vector<float> v_x{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f};
    copy_data(t_x, v_x);
    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    auto handle = backend->compile(f);
    handle->call_with_validate({t_r}, {t_x});
    vector<float> v_r{6.f, 7.f, 8.f, 9.f};
    ASSERT_EQ(t_r->get_shape(), (Shape{1, 2, 2}));
    EXPECT_TRUE(test::all_close_f(v_r, read_vector<float>(t_r)));
}

NGRAPH_TEST(${BACKEND_NAME}, partial_slice_bprop_unkown_rank)
{
    auto pshape_x = PartialShape::dynamic();
    auto pshape_dout = PartialShape::dynamic();
    auto x = make_shared<op::Parameter>(element::f32, pshape_x);
    auto dout = make_shared<op::Parameter>(element::f32, pshape_dout);
    AxisVector axes{0, 1};
    vector<int64_t> lower_bounds{1, 0};
    vector<int64_t> upper_bounds{2, 2};
    auto f = make_shared<Function>(
        make_shared<op::PartialSliceBackprop>(x, dout, axes, lower_bounds, upper_bounds),
        ParameterVector{x, dout});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    Shape shape_x{2, 3, 2};
    Shape shape_dout{1, 2, 2};
    auto t_x = backend->create_tensor(element::f32, shape_x);
    auto t_dout = backend->create_tensor(element::f32, shape_dout);
    vector<float> v_x{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f};
    vector<float> v_dout{6.f, 7.f, 8.f, 9.f};
    copy_data(t_x, v_x);
    copy_data(t_dout, v_dout);
    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    auto handle = backend->compile(f);
    handle->call_with_validate({t_r}, {t_x, t_dout});
    vector<float> v_r{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 6.f, 7.f, 8.f, 9.f, 0.f, 0.f};
    ASSERT_EQ(t_r->get_shape(), (Shape{2, 3, 2}));
    EXPECT_TRUE(test::all_close_f(v_r, read_vector<float>(t_r)));
}
