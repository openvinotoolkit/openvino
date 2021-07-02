#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, equal_f32)
{
    Shape shape{4};
    auto A = op::Constant::create(element::f32, shape, {-2.5f, 25.5f, 2.25f, 6.0f});
    auto B = op::Constant::create(element::f32, shape, {10.0f, 5.0f, 2.25f, 10.0f});
    auto f = make_shared<Function>(make_shared<op::v1::Equal>(A, B), ParameterVector{});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_expected_output<bool>(shape, {false, false, true, false});
    test_case.run();
}