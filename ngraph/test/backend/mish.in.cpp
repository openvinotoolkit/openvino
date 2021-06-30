// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

template <typename T>
T f_mish(T x)
{
    return x * std::tanh(std::log(1.0 + std::exp(x)));
}

template <ngraph::element::Type_t et>
test::unary_test<TestEngine, et> make_mish_test(const ngraph::PartialShape& pshape)
{
    return test::make_unary_test<TestEngine, op::v4::Mish, et>(pshape);
}

struct rand_generator
{
    std::mt19937 gen{0};
    std::normal_distribution<> d{0, 20};
    float operator()() { return d(gen); }
};

NGRAPH_TEST(${BACKEND_NAME}, mish_f32)
{
    Shape shape1{2, 5};
    Shape shape2{2, 3, 4, 5};

    make_mish_test<element::f32>(shape1).test({rand_generator(), shape1}, f_mish, 1e-5f);
    make_mish_test<element::f32>(shape2).test({rand_generator(), shape2}, f_mish, 1e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, mish_f16)
{
    Shape shape1{2, 5};
    Shape shape2{2, 3, 4, 5};

    make_mish_test<element::f16>(shape1).test({rand_generator(), shape1}, f_mish);
    make_mish_test<element::f16>(shape2).test({rand_generator(), shape2}, f_mish);
}

NGRAPH_TEST(${BACKEND_NAME}, mish_dynamic)
{
    make_mish_test<element::f32>(PartialShape::dynamic())
        .test({rand_generator(), Shape{{2, 3, 4, 5}}}, f_mish);

    make_mish_test<element::f32>({2, Dimension::dynamic(), 4, 5})
        .test({rand_generator(), Shape{2, 3, 4, 5}}, f_mish);
}