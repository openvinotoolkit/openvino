// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

template <ngraph::element::Type_t et>
test::unary_test<TestEngine, et> abc_test(const ngraph::PartialShape& shape)
{
    auto A = make_shared<op::Parameter>(et, shape);
    auto B = make_shared<op::Parameter>(et, shape);
    auto C = make_shared<op::Parameter>(et, shape);
    auto arg = make_shared<op::v1::Multiply>(make_shared<op::v1::Add>(A, B), C);
    auto f = make_shared<Function>(arg, ParameterVector{A, B, C});

    return test::unary_test<TestEngine, et>(f);
}

NGRAPH_TEST(${BACKEND_NAME}, abc)
{
    constexpr auto et = element::Type_t::f32;
    Shape shape{2, 2};

    test::Data<et> a{1, 2, 3, 4};
    test::Data<et> b{5, 6, 7, 8};
    test::Data<et> c{9, 10, 11, 12};

    abc_test<et>(shape).test({a, b, c}, {54, 80, 110, 144});
    abc_test<et>(shape).test({b, a, c}, {54, 80, 110, 144});
    abc_test<et>(shape).test({a, c, b}, {50, 72, 98, 128});
}

NGRAPH_TEST(${BACKEND_NAME}, abc_int64)
{
    constexpr auto et = element::Type_t::i64;
    Shape shape{2, 2};

    test::Data<et> a{1, 2, 3, 4};
    test::Data<et> b{5, 6, 7, 8};
    test::Data<et> c{9, 10, 11, 12};

    abc_test<et>(shape).test({a, b, c}, {54, 80, 110, 144});
    abc_test<et>(shape).test({b, a, c}, {54, 80, 110, 144});
    abc_test<et>(shape).test({a, c, b}, {50, 72, 98, 128});
}
