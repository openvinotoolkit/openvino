// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/test_case.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

template<typename T>
struct BatchNormTestParams
{
    std::vector<T> in;
    Shape in_shape;
    std::vector<T> in_g;
    std::vector<T> in_b;
    std::vector<T> in_m;
    std::vector<T> in_v;
    float epsilon;
    std::vector<T> out;
};

template <typename T>
static void BatchNormInferenceTest(const BatchNormTestParams<T>& p)
{
    const Shape ch_shape{p.in_shape.at(1)};
    auto input = make_shared<op::Parameter>(element::from<T>(), p.in_shape);
    auto gamma = make_shared<op::Parameter>(element::from<T>(), ch_shape);
    auto beta = make_shared<op::Parameter>(element::from<T>(), ch_shape);
    auto mean = make_shared<op::Parameter>(element::from<T>(), ch_shape);
    auto variance = make_shared<op::Parameter>(element::from<T>(), ch_shape);
    auto batch_norm = make_shared<op::v5::BatchNormInference>(
        input,
        gamma,
        beta,
        mean,
        variance,
        p.epsilon);
    auto f = make_shared<Function>(
        batch_norm, ParameterVector{input, gamma, beta, mean, variance});
    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<T>(p.in_shape, p.in);
    test_case.add_input<T>(ch_shape, p.in_g);
    test_case.add_input<T>(ch_shape, p.in_b);
    test_case.add_input<T>(ch_shape, p.in_m);
    test_case.add_input<T>(ch_shape, p.in_v);
    test_case.add_expected_output<T>(p.in_shape, p.out);
    test_case.run_with_tolerance_as_fp(1e-4f);
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_inference_2d_f32)
{
    const std::vector<BatchNormTestParams<float>> batch_norm_tests{
        BatchNormTestParams<float>{
                {1.0, 2.0, 3.0, -1.0, -2.0, -3.0},
                Shape{2, 3},
                {2.0, 3.0, 4.0},
                {0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0},
                {0.75, 0.75, 0.75},
                0.25,
                {2.0, 6.0, 12.0, -2.0, -6.0, -12.0}},

        BatchNormTestParams<float>{
                {1.0, 2.0, 3.0, -1.0, -2.0, -3.0},
                Shape{2, 3},
                {1.0, 1.0, 1.0},
                {2.0, -2.0, 3.0},
                {0.0, 0.0, 0.0},
                {0.75, 0.75, 0.75},
                0.25,
                {3.0, 0.0, 6.0, 1.0, -4.0, 0.0}},

        BatchNormTestParams<float>{
                {1.0, 2.0, 3.0, -1.0, -2.0, -3.0},
                Shape{2, 3},
                {1.0, 1.0, 1.0},
                {0.0, 0.0, 0.0},
                {-2.0, 2.0, -3.0},
                {0.75, 0.75, 0.75},
                0.25,
                {3.0, 0.0, 6.0, 1.0, -4.0, 0.0}},

        BatchNormTestParams<float>{
                {3.0, 5.0, 1.0, -3.0, -5.0, -1.0},
                Shape{2, 3},
                {1.0, 1.0, 1.0},
                {0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0},
                {2.0, 6.0, 0.0},
                0.25,
                {2.0, 2.0, 2.0, -2.0, -2.0, -2.0}},
    };

    for(const auto& test_case : batch_norm_tests)
    {
        BatchNormInferenceTest(test_case);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_inference_4d_f32)
{
    float eps = 0.001;
    Shape in_shape{2, 2, 2, 1};

    std::vector<float> in{0.54881352f,
                          0.71518934f,
                          0.60276335f,
                          0.54488319f,
                          0.42365479f,
                          0.64589411f,
                          0.4375872f,
                          0.89177299f};

    std::vector<std::vector<float>> ch_in_1{{1.0, 1.0},
                                            {1.0, 1.0},
                                            {1.0, 1.0},
                                            {1.0, 1.0}};
    std::vector<float> out_1{0.54903894f,
                             0.71533161f,
                             0.60296183f,
                             0.54511058f,
                             0.42394274f,
                             0.64607101f,
                             0.43786817f,
                             0.89182704f};

    std::vector<std::vector<float>> ch_in_2{{1.0, 1.0},
                                            {0.0f, 0.0f},
                                            {0.583388f, 0.619252f},
                                            {0.0119972f, 0.0282681f}};
    std::vector<float> out_2{-0.30327f,
                             1.1561f,
                             -0.096382f,
                             -0.434702f,
                             -1.4011f,
                             0.548275f,
                             -1.06187f,
                             1.59295f};

    const std::vector<BatchNormTestParams<float>> batch_norm_tests{
        BatchNormTestParams<float>{in, in_shape, ch_in_1[0], ch_in_1[1], ch_in_1[2], ch_in_1[3], eps, out_1},
        BatchNormTestParams<float>{in, in_shape, ch_in_2[0], ch_in_2[1], ch_in_2[2], ch_in_2[3], eps, out_2}
    };
    
    for(const auto& test_case : batch_norm_tests)
    {
        BatchNormInferenceTest(test_case);
    }
}
