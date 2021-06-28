// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

template <ngraph::element::Type_t et>
test::unary_test<TestEngine, et>
    make_hardsigmoid_test(const ngraph::PartialShape& pshape, float f_alpha, float f_beta)
{
    auto A = make_shared<op::Parameter>(et, pshape);

    auto alpha = op::Constant::create(et, Shape{}, {f_alpha});
    auto beta = op::Constant::create(et, Shape{}, {f_beta});

    auto R = make_shared<op::v0::HardSigmoid>(A, alpha, beta);
    auto function = make_shared<Function>(R, ParameterVector{A});

    return test::unary_test<TestEngine, et>(function);
}

NGRAPH_TEST(${BACKEND_NAME}, hard_sigmoid_1d)
{
    make_hardsigmoid_test<element::f32>(Shape{3}, 0.5f, 0.6f)
        .test({-1.0f, 0.0f, 1.0f}, {0.1f, 0.6f, 1.f});
}

NGRAPH_TEST(${BACKEND_NAME}, hard_sigmoid_2d)
{
    make_hardsigmoid_test<element::f32>(Shape{2, 5}, 0.2f, 0.5f)
        .test({-3.0f, -1.0f, 0.0f, 1.0f, 3.0f, 0.5f, -0.2f, 6.0f, 8.0f, 0.1f},
              {0.0f, 0.3f, 0.5f, 0.7f, 1.0f, 0.6f, 0.46f, 1.0f, 1.0f, 0.52f});
}
