// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, elu)
{
    test::make_unary_test<TestEngine, op::Elu, element::f32>(Shape{3, 2}, 0.5f)
        .test({-2.f, 3.f, -2.f, 1.f, -1.f, 0.f},
              {-0.432332358f, 3.f, -0.432332358f, 1.f, -0.316060279f, 0.f});
}

NGRAPH_TEST(${BACKEND_NAME}, elu_negative_alpha)
{
    test::make_unary_test<TestEngine, op::Elu, element::f32>(Shape{3, 2}, -1.f)
        .test({-2.f, 3.f, -2.f, 1.f, -1.f, 0.f},
              {0.864664717f, 3.f, 0.864664717f, 1.f, 0.632120559f, 0.f});
}