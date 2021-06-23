// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

namespace
{
    struct HardSigmoid
    {
        float m_alpha;
        float m_beta;
        HardSigmoid(float alpha, float beta)
            : m_alpha(alpha)
            , m_beta(beta)
        {
        }

        std::shared_ptr<Function> operator()(const ngraph::element::Type& ele_type,
                                             const PartialShape& pshape)
        {
            auto A = make_shared<op::Parameter>(ele_type, pshape);

            auto alpha = op::Constant::create(ele_type, Shape{}, {m_alpha});
            auto beta = op::Constant::create(ele_type, Shape{}, {m_beta});

            auto R = make_shared<op::v0::HardSigmoid>(A, alpha, beta);
            return make_shared<Function>(R, ParameterVector{A});
        }
    };
}

NGRAPH_TEST(${BACKEND_NAME}, hard_sigmoid_1d)
{
    test_unary<TestEngine, element::f32>(
        HardSigmoid(0.5f, 0.6f), {-1.0f, 0.0f, 1.0f}, {0.1f, 0.6f, 1.f});
}

NGRAPH_TEST(${BACKEND_NAME}, hard_sigmoid_2d)
{
    test_unary<TestEngine, element::f32>(
        HardSigmoid(0.2f, 0.5f),
        {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f, 0.5f, -0.2f, 6.0f, 8.0f, 0.1f},
        {0.0f, 0.3f, 0.5f, 0.7f, 1.0f, 0.6f, 0.46f, 1.0f, 1.0f, 0.52f},
        Shape{2, 5},
        Shape{2, 5});
}
