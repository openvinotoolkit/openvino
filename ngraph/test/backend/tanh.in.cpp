// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, tanh_f32)
{
    test::make_unary_test<TestEngine, op::Tanh, element::f32>(Shape{8}).test(
        {2.f, 1.f, 0.5f, 0.f, -0.f, -0.5f, -1.f, -2.f}, std::tanh);
}

NGRAPH_TEST(${BACKEND_NAME}, tanh_int32)
{
    test::make_unary_test<TestEngine, op::Tanh, element::i32>(Shape{5}).test({2, 1, 0, -1, -2},
                                                                             {1, 1, 0, -1, -1});
}
