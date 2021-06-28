// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, sinh)
{
    test::make_unary_test<TestEngine, op::Sinh, element::f32>(Shape{6}).test(
        {1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f}, std::sinh);
}
