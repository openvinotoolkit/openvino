// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

// Multiple retrive values
NGRAPH_TEST(${BACKEND_NAME}, multiple_result)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto A_add_B = make_shared<op::v1::Add>(A, B);
    auto A_add_B_mul_C = make_shared<op::v1::Multiply>(A_add_B, C);

    auto f = make_shared<Function>(NodeVector{A_add_B, A_add_B_mul_C}, ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{5, 6, 7, 8});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{9, 10, 11, 12});

    auto r0 = backend->create_tensor(element::f32, shape);
    auto r1 = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({r0, r1}, {a, b, c});

    EXPECT_TRUE(test::all_close_f((vector<float>{6, 8, 10, 12}), read_vector<float>(r0)));
    EXPECT_TRUE(test::all_close_f((vector<float>{54, 80, 110, 144}), read_vector<float>(r1)));
}
