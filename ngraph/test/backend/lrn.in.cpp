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

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include <numeric>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, lrn_across_channel)
{
    Shape shape{2, 3, 2, 1};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    double alpha = 3;
    double beta = 0.5;
    double bias = 1;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    std::vector<float> a{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(shape, a);
    test_case.add_expected_output<float>(shape,
                                         {0.0000000f,
                                          0.3015113f,
                                          0.4364358f,
                                          0.5000000f,
                                          0.8728716f,
                                          0.8451542f,
                                          0.5970223f,
                                          0.6115928f,
                                          0.5642765f,
                                          0.5669467f,
                                          0.7784989f,
                                          0.7720487f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lrn_across_h)
{
    Shape shape{2, 3, 2, 1};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto axes = make_shared<op::Constant>(element::Type_t::i64, Shape{1}, vector<int64_t>{2});
    double alpha = 3;
    double beta = 0.5;
    double bias = 1;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, axes, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    std::vector<float> a{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(shape, a);
    test_case.add_expected_output<float>(shape,
                                         {0.0000000f,
                                          0.7071068f,
                                          0.5345225f,
                                          0.8017837f,
                                          0.6172134f,
                                          0.7715167f,
                                          0.6469966f,
                                          0.7548294f,
                                          0.6620847f,
                                          0.7448453f,
                                          0.6711560f,
                                          0.7382717f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lrn_across_hw)
{
    Shape shape{2, 3, 2, 1};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto axes = make_shared<op::Constant>(element::Type_t::i64, Shape{2}, vector<int64_t>{2, 3});
    double alpha = 3;
    double beta = 0.5;
    double bias = 1;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, axes, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    std::vector<float> a{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(shape, a);
    test_case.add_expected_output<float>(shape,
                                         {0.0000000f,
                                          0.8660254f,
                                          0.8660254f,
                                          1.2990381f,
                                          1.0444659f,
                                          1.3055824f,
                                          1.1078234f,
                                          1.2924607f,
                                          1.1389896f,
                                          1.2813632f,
                                          1.1572751f,
                                          1.2730026f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lrn_across_all_dims)
{
    Shape shape{2, 3, 2, 1};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto axes =
        make_shared<op::Constant>(element::Type_t::i64, Shape{4}, vector<int64_t>{0, 1, 2, 3});
    double alpha = 3;
    double beta = 0.5;
    double bias = 1;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, axes, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    std::vector<float> a{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(shape, a);
    test_case.add_expected_output<float>(shape,
                                         {0.0000000f,
                                          0.3156438f,
                                          0.4501407f,
                                          0.6752110f,
                                          0.9830783f,
                                          1.2288479f,
                                          1.8938627f,
                                          2.2095065f,
                                          1.8005627f,
                                          2.0256331f,
                                          2.4576957f,
                                          2.7034652f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, lrn_across_nw)
{
    Shape shape{2, 3, 2, 1};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto axes = make_shared<op::Constant>(element::Type_t::i64, Shape{2}, vector<int64_t>{0, 3});
    double alpha = 3;
    double beta = 0.5;
    double bias = 1;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, axes, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    std::vector<float> a{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(shape, a);
    test_case.add_expected_output<float>(shape,
                                         {0.0000000f,
                                          0.2379155f,
                                          0.4111132f,
                                          0.5388159f,
                                          0.6351073f,
                                          0.7094756f,
                                          1.6641006f,
                                          1.6654084f,
                                          1.6444529f,
                                          1.6164477f,
                                          1.5877683f,
                                          1.5608464f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lrn_across_empty)
{
    Shape shape{2, 3, 2, 1};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto axes = make_shared<op::Constant>(element::Type_t::i64, Shape{0}, vector<int64_t>{});
    double alpha = 3;
    double beta = 0.5;
    double bias = 1;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, axes, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    std::vector<float> a{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(shape, a);
    test_case.add_expected_output<float>(shape,
                                         {0.0000000f,
                                          0.5000000f,
                                          0.5547002f,
                                          0.5669467f,
                                          0.5714286f,
                                          0.5735393f,
                                          0.5746958f,
                                          0.5753965f,
                                          0.5758526f,
                                          0.5761660f,
                                          0.5763904f,
                                          0.5765567f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lrn_6D_across_2_axes)
{
    Shape shape{2, 3, 2, 2, 1, 1};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto axes = make_shared<op::Constant>(element::Type_t::i64, Shape{2}, vector<int64_t>{2, 3});
    double alpha = 3;
    double beta = 0.5;
    double bias = 1;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, axes, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    std::vector<float> a(24);
    std::iota(std::begin(a), std::end(a), 0);

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(shape, a);
    test_case.add_expected_output<float>(
        shape, {0.0000000f, 0.4200840f, 0.8401681f, 1.2602521f, 0.6099943f, 0.7624928f,
                0.9149914f, 1.0674900f, 0.7213357f, 0.8115027f, 0.9016696f, 0.9918366f,
                0.7656109f, 0.8294119f, 0.8932127f, 0.9570137f, 0.7892218f, 0.8385482f,
                0.8878745f, 0.9372009f, 0.8038679f, 0.8440613f, 0.8842546f, 0.9244481f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lrn_2d_across_empty)
{
    Shape shape{12};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto axes = make_shared<op::Constant>(element::Type_t::i64, Shape{0}, vector<int64_t>{});
    double alpha = 3;
    double beta = 0.5;
    double bias = 1;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, axes, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    std::vector<float> a{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(shape, a);
    test_case.add_expected_output<float>(shape,
                                         {0.0000000f,
                                          0.5000000f,
                                          0.5547002f,
                                          0.5669467f,
                                          0.5714286f,
                                          0.5735393f,
                                          0.5746958f,
                                          0.5753964f,
                                          0.5758526f,
                                          0.5761660f,
                                          0.5763904f,
                                          0.5765566f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lrn_2d_across_outermost_axis)
{
    Shape shape{6, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape);
    auto axes = make_shared<op::Constant>(element::Type_t::i64, Shape{1}, vector<int64_t>{0});
    double alpha = 0.0002;
    double beta = 0.5;
    double bias = 2.0;
    size_t size = 3;
    auto lrn = make_shared<op::LRN>(A, axes, alpha, beta, bias, size);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    std::vector<float> a{0.64915806f,
                         0.21213771f,
                         -1.48256505f,
                         -1.41040838f,
                         0.58189541f,
                         0.11432108f,
                         -0.22993855f,
                         -0.13325502f,
                         -0.03083259f,
                         -0.48450908f,
                         0.50342429f,
                         -0.99551708f};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(shape, a);
    test_case.add_expected_output<float>(shape,
                                         {0.4590040f,
                                          0.1499989f,
                                          -1.0482801f,
                                          -0.9972753f,
                                          0.4114444f,
                                          0.0808345f,
                                          -0.1625900f,
                                          -0.0942251f,
                                          -0.0218018f,
                                          -0.3425926f,
                                          0.3559732f,
                                          -0.7039225f});

    test_case.run(23);
}
