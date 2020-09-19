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

#include "util/type_prop.hpp"

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif

#ifndef RTOL
#define RTOL 1e-4
#endif

#ifndef ATOL
#define ATOL 1e-4
#endif

// clang-format on

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

template <typename optype, typename itype, typename otype>
void check_auto_bcast(
    const std::vector<std::vector<itype>>& inputs,
    const std::vector<otype> output,
    const op::AutoBroadcastSpec& autob = op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY),
    bool set_tolerance = false)
{
    auto iet = element::from<itype>();
    auto oet = element::from<otype>();

    if (std::is_same<itype, char>::value)
    {
        iet = element::boolean;
    }
    if (std::is_same<otype, char>::value)
    {
        oet = element::boolean;
    }
    auto A = make_shared<op::Parameter>(iet, Shape{2, 3});
    auto B = make_shared<op::Parameter>(iet, Shape{3});
    auto f = make_shared<Function>(make_shared<optype>(A, B, autob), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(iet, Shape{2, 3});
    shared_ptr<runtime::Tensor> b = backend->create_tensor(iet, Shape{3});
    shared_ptr<runtime::Tensor> result = backend->create_tensor(oet, Shape{2, 3});

    copy_data(a, inputs[0]);
    copy_data(b, inputs[1]);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    if (set_tolerance)
    {
        EXPECT_TRUE(test::all_close(read_vector<otype>(result),
                                    output,
                                    static_cast<otype>(RTOL),
                                    static_cast<otype>(ATOL)));
    }
    else
    {
        EXPECT_TRUE(test::all_close(read_vector<otype>(result), output));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, auto_bcast_binary_elementwise_pdpd_dynamic)
{
    auto pshape_a = PartialShape::dynamic();
    auto pshape_b = PartialShape::dynamic();
    auto a = make_shared<op::Parameter>(element::f32, pshape_a);
    auto b = make_shared<op::Parameter>(element::f32, pshape_b);

    op::AutoBroadcastSpec autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, -1);
    auto f = make_shared<Function>(make_shared<op::Add>(a, b, autob), ParameterVector{a, b});
    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);
    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto t_a = backend->create_tensor(element::f32, Shape{2, 3});
    auto t_b = backend->create_tensor(element::f32, Shape{3});
    copy_data(t_a, vector<float>{1, 2, 3, 4, 5, 6});
    copy_data(t_b, vector<float>{5, 6, 7});
    ex->call_with_validate({t_r}, {t_a, t_b});
    ASSERT_EQ(t_r->get_shape(), (Shape{2, 3}));

    auto results = read_vector<float>(t_r);
    vector<float> expected_values{6, 8, 10, 9, 11, 13};
    EXPECT_TRUE(test::all_close_f(results, expected_values));

    // a shape {2, 3, 4, 5}, b shape {3, 4} axis = 1
    autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 1);
    f = make_shared<Function>(make_shared<op::Add>(a, b, autob), ParameterVector{a, b});
    ex = backend->compile(f);
    t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    t_a = backend->create_tensor(element::f32, Shape{2, 3, 4, 5});
    t_b = backend->create_tensor(element::f32, Shape{3, 4});
    copy_data(t_a, vector<float>(2 * 3 * 4 * 5, 1));
    copy_data(t_b, vector<float>(3 * 4, 1));
    ex->call_with_validate({t_r}, {t_a, t_b});
    ASSERT_EQ(t_r->get_shape(), (Shape{2, 3, 4, 5}));

    // a shape {2, 3, 4, 5}, b shape {3, 1} axis = 1
    t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    t_a = backend->create_tensor(element::f32, Shape{2, 3, 4, 5});
    t_b = backend->create_tensor(element::f32, Shape{3, 1});
    copy_data(t_a, vector<float>(2 * 3 * 4 * 5, 1));
    copy_data(t_b, vector<float>(3, 1));
    ex->call_with_validate({t_r}, {t_a, t_b});
    ASSERT_EQ(t_r->get_shape(), (Shape{2, 3, 4, 5}));
}

NGRAPH_TEST(${BACKEND_NAME}, auto_bcast_string_cast)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{1});
    auto b = make_shared<op::Parameter>(element::f32, Shape{1});

    auto add = make_shared<op::Add>(a, b, "NUMPY");
    ASSERT_EQ(add->get_autob(), op::AutoBroadcastType::NUMPY);

    add = make_shared<op::Add>(a, b, "NONE");
    ASSERT_EQ(add->get_autob(), op::AutoBroadcastType::NONE);

    add = make_shared<op::Add>(a, b, "PDPD");
    ASSERT_EQ(add->get_autob(), op::AutoBroadcastType::PDPD);

    add = make_shared<op::Add>(a, b, "EXPLICIT");
    ASSERT_EQ(add->get_autob(), op::AutoBroadcastType::EXPLICIT);

    try
    {
        add = make_shared<op::Add>(a, b, "UNKNOWN");
        FAIL() << "Unknown AutoBroadcastType not detected.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Invalid 'type' value passed in."));
    }
    catch (...)
    {
        FAIL() << "AutoBroadcastType checking failed for unexpected reason";
    }
}
