// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/check.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_zero_bias_default_attrs)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size});

    const auto lstm_cell = make_shared<opset4::LSTMCell>(
        X,
        H_t,
        C_t,
        op::util::convert_lstm_node_format(W, op::util::LSTMWeightsFormat::IOFC),
        op::util::convert_lstm_node_format(R, op::util::LSTMWeightsFormat::IOFC),
        op::util::convert_lstm_node_format(B, op::util::LSTMWeightsFormat::IOFC),
        hidden_size);

    auto ht_function = make_shared<Function>(OutputVector{lstm_cell->output(0)},
                                             ParameterVector{X, H_t, C_t, W, R, B});
    auto ht_test_case = test::TestCase<TestEngine>(ht_function);

    // X
    vector<float> in_X{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f};
    // W
    vector<float> in_W{3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f,
                       7.3950343e-02f, 3.8063636e-01f, 9.6921772e-01f, 9.6897459e-01f,
                       6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                       6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f,
                       4.0472135e-01f, 6.8342745e-01f, 8.3432144e-01f, 4.4928190e-01f,
                       7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                       5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f,
                       2.1716513e-01f, 2.7473119e-01f, 3.3999152e-02f, 9.6835363e-01f,
                       3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f};
    // R
    vector<float> in_R{
        0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
        0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
        0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
        0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
        0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
        0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f};
    // Ht
    vector<float> in_Ht{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f};
    // Ct
    vector<float> in_Ct{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f};
    // B
    vector<float> in_B(gates_count * hidden_size, 0.f);

    ht_test_case.add_multiple_inputs(vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B});
    ht_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.81457126f, 0.61109227f, 0.769522f, 0.52239674f, 0.4324641f, 0.63183f});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(OutputVector{lstm_cell->output(1)},
                                             ParameterVector{X, H_t, C_t, W, R, B});
    auto ct_test_case = test::TestCase<TestEngine>(ct_function);
    ct_test_case.add_multiple_inputs(vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B});
    ct_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {1.4444952f, 0.9635685f, 1.2875274f, 0.8053419f, 0.7184521f, 0.95803297f});
    ct_test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_bias_default_attrs)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size});

    const auto lstm_cell = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, B, hidden_size);

    auto ht_function = make_shared<Function>(OutputVector{lstm_cell->output(0)},
                                             ParameterVector{X, H_t, C_t, W, R, B});
    auto ht_test_case = test::TestCase<TestEngine>(ht_function);

    // X
    vector<float> in_X{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f};
    // W
    vector<float> in_W{3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f,
                       7.3950343e-02f, 3.8063636e-01f, 9.6921772e-01f, 9.6897459e-01f,
                       6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                       6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f,
                       4.0472135e-01f, 6.8342745e-01f, 8.3432144e-01f, 4.4928190e-01f,
                       7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                       5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f,
                       2.1716513e-01f, 2.7473119e-01f, 3.3999152e-02f, 9.6835363e-01f,
                       3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f};
    // R
    vector<float> in_R{
        0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
        0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
        0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
        0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
        0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
        0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f};
    // Ht
    vector<float> in_Ht{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f};
    // Ct
    vector<float> in_Ct{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f};
    // B
    vector<float> in_B{1.07393714f,
                       1.15248052f,
                       1.16671345f,
                       0.21450312f,
                       1.2380678f,
                       1.51688835f,
                       0.46718366f,
                       0.91810346f,
                       1.1274234f,
                       0.51022074f,
                       1.11389844f,
                       0.74174305f};

    ht_test_case.add_multiple_inputs(vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B});
    ht_test_case.add_expected_output<float>(Shape{batch_size, hidden_size},
                                            {0.81014400720596313,
                                             0.76665538549423218,
                                             0.82509011030197144,
                                             0.6479143500328064,
                                             0.66586339473724365,
                                             0.74838578701019287});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(OutputVector{lstm_cell->output(1)},
                                             ParameterVector{X, H_t, C_t, W, R, B});
    auto ct_test_case = test::TestCase<TestEngine>(ct_function);
    ct_test_case.add_multiple_inputs(vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B});
    ct_test_case.add_expected_output<float>(Shape{batch_size, hidden_size},
                                            {1.6800162792205811,
                                             1.1150213479995728,
                                             1.4578367471694946,
                                             1.0649888515472412,
                                             0.93761754035949707,
                                             1.3659683465957642});
    ct_test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_bias_clip)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;
    const float clip_threshold = 3.5f;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size});

    const auto lstm_cell = make_shared<opset4::LSTMCell>(X,
                                                         H_t,
                                                         C_t,
                                                         W,
                                                         R,
                                                         B,
                                                         hidden_size,
                                                         vector<string>{"sigmoid", "tanh", "tanh"},
                                                         vector<float>{},
                                                         vector<float>{},
                                                         clip_threshold);
    auto ht_function = make_shared<Function>(OutputVector{lstm_cell->output(0)},
                                             ParameterVector{X, H_t, C_t, W, R, B});
    auto ht_test_case = test::TestCase<TestEngine>(ht_function);

    // X
    vector<float> in_X{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f};
    // W
    vector<float> in_W{3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f,
                       7.3950343e-02f, 3.8063636e-01f, 9.6921772e-01f, 9.6897459e-01f,
                       6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                       6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f,
                       4.0472135e-01f, 6.8342745e-01f, 8.3432144e-01f, 4.4928190e-01f,
                       7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                       5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f,
                       2.1716513e-01f, 2.7473119e-01f, 3.3999152e-02f, 9.6835363e-01f,
                       3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f};
    // R
    vector<float> in_R{
        0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
        0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
        0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
        0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
        0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
        0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f};
    // Ht
    vector<float> in_Ht{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f};
    // Ct
    vector<float> in_Ct{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f};
    // B
    vector<float> in_B{1.07393714f,
                       1.15248052f,
                       1.16671345f,
                       0.21450312f,
                       1.2380678f,
                       1.51688835f,
                       0.46718366f,
                       0.91810346f,
                       1.1274234f,
                       0.51022074f,
                       1.11389844f,
                       0.74174305f};

    ht_test_case.add_multiple_inputs(vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B});
    ht_test_case.add_expected_output<float>(Shape{batch_size, hidden_size},
                                            {0.81014400720596313,
                                             0.76665538549423218,
                                             0.82387429475784302,
                                             0.6479143500328064,
                                             0.66586339473724365,
                                             0.74838578701019287});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(OutputVector{lstm_cell->output(1)},
                                             ParameterVector{X, H_t, C_t, W, R, B});
    auto ct_test_case = test::TestCase<TestEngine>(ct_function);
    ct_test_case.add_multiple_inputs(vector<vector<float>>{in_X, in_Ht, in_Ct, in_W, in_R, in_B});
    ct_test_case.add_expected_output<float>(Shape{batch_size, hidden_size},
                                            {1.6800162792205811,
                                             1.1150213479995728,
                                             1.4510968923568726,
                                             1.0649888515472412,
                                             0.93761754035949707,
                                             1.3659683465957642});
    ct_test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, rnn_cell_zero_bias_default_attrs)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W = make_shared<op::Parameter>(element::f32, Shape{hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{hidden_size});

    const auto rnn_cell = make_shared<opset4::RNNCell>(X, H_t, W, R, B, hidden_size);
    auto function = make_shared<Function>(rnn_cell, ParameterVector{X, H_t, W, R, B});

    auto test_case = test::TestCase<TestEngine>(function);
    // X
    test_case.add_input<float>(
        {0.3432185f, 0.612268f, 0.20272376f, 0.9513413f, 0.30585995f, 0.7265472f});
    // Ht
    test_case.add_input<float>(
        {0.12444675f, 0.52055854f, 0.46489045f, 0.4983964f, 0.7730452f, 0.28439692f});
    // W
    test_case.add_input<float>({0.41930267f,
                                0.7872176f,
                                0.89940447f,
                                0.23659843f,
                                0.24676207f,
                                0.17101714f,
                                0.3147149f,
                                0.6555601f,
                                0.4559603f});
    // R
    test_case.add_input<float>({0.8374871f,
                                0.86660194f,
                                0.82114047f,
                                0.71549815f,
                                0.18775631f,
                                0.3182116f,
                                0.25392973f,
                                0.38301638f,
                                0.85531586f});
    // B
    test_case.add_input<float>({0.0f, 0.0f, 0.0f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.9408395f, 0.53823817f, 0.84270686f, 0.98932856f, 0.768665f, 0.90461975f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, rnn_cell_bias_clip)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    float clip = 2.88f;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W = make_shared<op::Parameter>(element::f32, Shape{hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{hidden_size});

    const auto rnn_cell = make_shared<opset4::RNNCell>(X,
                                                       H_t,
                                                       W,
                                                       R,
                                                       B,
                                                       hidden_size,
                                                       vector<string>{"tanh"},
                                                       vector<float>{},
                                                       vector<float>{},
                                                       clip);
    auto function = make_shared<Function>(rnn_cell, ParameterVector{X, H_t, W, R, B});

    auto test_case = test::TestCase<TestEngine>(function);
    // X
    test_case.add_input<float>(
        {0.3432185f, 0.612268f, 0.20272376f, 0.9513413f, 0.30585995f, 0.7265472f});
    // Ht
    test_case.add_input<float>(
        {0.12444675f, 0.52055854f, 0.46489045f, 0.4983964f, 0.7730452f, 0.28439692f});
    // W
    test_case.add_input<float>({0.41930267f,
                                0.7872176f,
                                0.89940447f,
                                0.23659843f,
                                0.24676207f,
                                0.17101714f,
                                0.3147149f,
                                0.6555601f,
                                0.4559603f});
    // R
    test_case.add_input<float>({0.8374871f,
                                0.86660194f,
                                0.82114047f,
                                0.71549815f,
                                0.18775631f,
                                0.3182116f,
                                0.25392973f,
                                0.38301638f,
                                0.85531586f});
    // B
    test_case.add_input<float>({1.0289404f, 1.6362579f, 0.4370661f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.9922437f, 0.97749525f, 0.9312212f, 0.9937176f, 0.9901317f, 0.95906746f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, rnn_cell_sigmoid_activation_function)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    float clip = 2.88f;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W = make_shared<op::Parameter>(element::f32, Shape{hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{hidden_size});

    const auto rnn_cell = make_shared<opset4::RNNCell>(X,
                                                       H_t,
                                                       W,
                                                       R,
                                                       B,
                                                       hidden_size,
                                                       vector<string>{"sigmoid"},
                                                       vector<float>{},
                                                       vector<float>{},
                                                       clip);
    auto function = make_shared<Function>(rnn_cell, ParameterVector{X, H_t, W, R, B});

    auto test_case = test::TestCase<TestEngine>(function);
    // X
    test_case.add_input<float>(
        {0.3432185f, 0.612268f, 0.20272376f, 0.9513413f, 0.30585995f, 0.7265472f});
    // Ht
    test_case.add_input<float>(
        {0.12444675f, 0.52055854f, 0.46489045f, 0.4983964f, 0.7730452f, 0.28439692f});
    // W
    test_case.add_input<float>({0.41930267f,
                                0.7872176f,
                                0.89940447f,
                                0.23659843f,
                                0.24676207f,
                                0.17101714f,
                                0.3147149f,
                                0.6555601f,
                                0.4559603f});
    // R
    test_case.add_input<float>({0.8374871f,
                                0.86660194f,
                                0.82114047f,
                                0.71549815f,
                                0.18775631f,
                                0.3182116f,
                                0.25392973f,
                                0.38301638f,
                                0.85531586f});
    // B
    test_case.add_input<float>({1.0289404f, 1.6362579f, 0.4370661f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.94126844f, 0.9036043f, 0.841243f, 0.9468489f, 0.934215f, 0.873708f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gru_cell_bias_clip)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;
    float clip = 2.88f;
    bool linear_before_reset = false;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size});

    const auto gru_cell = make_shared<opset4::GRUCell>(X,
                                                       H_t,
                                                       W,
                                                       R,
                                                       B,
                                                       hidden_size,
                                                       vector<string>{"sigmoid", "tanh"},
                                                       vector<float>{},
                                                       vector<float>{},
                                                       clip,
                                                       linear_before_reset);
    auto function = make_shared<Function>(gru_cell, ParameterVector{X, H_t, W, R, B});

    auto test_case = test::TestCase<TestEngine>(function);
    // X
    test_case.add_input<float>(
        {0.52421564f, 0.78845507f, 0.9372873f, 0.59783894f, 0.18278378f, 0.2084126f});

    // Ht
    test_case.add_input<float>(
        {0.45738035f, 0.996877f, 0.82882977f, 0.47492632f, 0.88471466f, 0.57833236f});

    // W
    test_case.add_input<float>(
        {0.5815369f, 0.16559383f, 0.08464007f, 0.843122f,   0.73968244f, 0.11359601f, 0.8295078f,
         0.9240567f, 0.10007995f, 0.20573162f, 0.09002485f, 0.2839569f,  0.3096991f,  0.5638341f,
         0.5787327f, 0.84552664f, 0.16263747f, 0.7243242f,  0.8049057f,  0.43966424f, 0.46294412f,
         0.9833361f, 0.31369713f, 0.1719934f,  0.4937093f,  0.6353004f,  0.77982515f});

    // R
    test_case.add_input<float>(
        {0.16510165f, 0.52435565f, 0.2788478f,  0.99427545f, 0.1623331f,  0.01389796f, 0.99669236f,
         0.53901845f, 0.8737506f,  0.9254788f,  0.21172932f, 0.11634306f, 0.40111724f, 0.37497616f,
         0.2903471f,  0.6796794f,  0.65131867f, 0.78163475f, 0.12058706f, 0.45591718f, 0.791677f,
         0.76497287f, 0.9895242f,  0.7845312f,  0.51267904f, 0.49030215f, 0.08498167f});

    // B (the sum of biases for W and R)
    test_case.add_input<float>({
        0.8286678f + 0.9175602f,
        0.9153158f + 0.14958014f,
        0.9581612f + 0.49230585f,
        0.6639213f + 0.63162816f,
        0.84239805f + 0.4161903f,
        0.5282445f + 0.22148274f,
        0.14153397f + 0.50496656f,
        0.22404431f + 0.34798595f,
        0.6549655f + 0.6699164f,
    });

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.48588726f, 0.99670005f, 0.83759373f, 0.5023099f, 0.89410484f, 0.60011315f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gru_cell_linear_before_reset)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;
    float clip = 2.88f;
    bool linear_before_reset = true;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{(gates_count + 1) * hidden_size});

    const auto gru_cell = make_shared<opset4::GRUCell>(X,
                                                       H_t,
                                                       W,
                                                       R,
                                                       B,
                                                       hidden_size,
                                                       vector<string>{"sigmoid", "tanh"},
                                                       vector<float>{},
                                                       vector<float>{},
                                                       clip,
                                                       linear_before_reset);
    auto function = make_shared<Function>(gru_cell, ParameterVector{X, H_t, W, R, B});

    auto test_case = test::TestCase<TestEngine>(function);
    // X
    test_case.add_input<float>(
        {0.12249453f, 0.6127907f, 0.5001741f, 0.5124603f, 0.04329684f, 0.023834f});
    // Ht
    test_case.add_input<float>(
        {0.8598948f, 0.41189128f, 0.72824323f, 0.53940123f, 0.31485787f, 0.04053852f});

    // W
    test_case.add_input<float>(
        {0.72259396f, 0.11561195f, 0.9457856f,  0.19037509f, 0.6964006f,  0.33459795f, 0.5468904f,
         0.85646594f, 0.5101311f,  0.9712257f,  0.3687071f,  0.60280246f, 0.56943774f, 0.7475505f,
         0.2490578f,  0.86977345f, 0.85542053f, 0.29660386f, 0.49717373f, 0.7473479f,  0.53454477f,
         0.15974349f, 0.5804805f,  0.14303213f, 0.07514781f, 0.5865731f,  0.76409274f});
    // R
    test_case.add_input<float>(
        {0.91382647f, 0.41527033f, 0.28040004f, 0.23601337f, 0.04471736f, 0.03888785f, 0.06308217f,
         0.44844428f, 0.29384327f, 0.49037653f, 0.50421673f, 0.7366393f,  0.63143945f, 0.00277612f,
         0.37198433f, 0.06966069f, 0.4613444f,  0.10999731f, 0.78273284f, 0.21453214f, 0.10751773f,
         0.18332677f, 0.1326976f,  0.9998985f,  0.19263928f, 0.10979804f, 0.52575564f});

    // B (the sum of biases for W and R for z and r gates, and separately for W and R for h gate)
    test_case.add_input<float>({0.61395123f, // 0.09875853f + 0.5151927f,
                                1.08667738f, // 0.37801138f + 0.708666f,
                                1.32600244f, // 0.7729636f + 0.55303884f,
                                0.81917698f, // 0.78493553f + 0.03424145f,
                                1.37736335f, // 0.5662702f + 0.81109315f,
                                0.42931147f, // 0.12406381f + 0.30524766f,
                                0.66729516f,
                                0.7752771f,
                                0.78819966f,
                                0.6606634f,
                                0.99040645f,
                                0.21112025f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.8709214f, 0.48411977f, 0.74495184f, 0.6074972f, 0.44572943f, 0.1467715f});

    test_case.run();
}

// Hard Sigmoid activation function is unsupprted with v3::GRUCell
NGRAPH_TEST(${BACKEND_NAME}, gru_cell_hardsigmoid_activation_function)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;
    float clip = 2.88f;
    bool linear_before_reset = true;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{(gates_count + 1) * hidden_size});

    const auto gru_cell = make_shared<opset4::GRUCell>(X,
                                                       H_t,
                                                       W,
                                                       R,
                                                       B,
                                                       hidden_size,
                                                       vector<string>{"hardsigmoid", "hardsigmoid"},
                                                       vector<float>{1.8345f, 1.8345f},
                                                       vector<float>{3.05f, 3.05f},
                                                       clip,
                                                       linear_before_reset);
    auto function = make_shared<Function>(gru_cell, ParameterVector{X, H_t, W, R, B});

    auto test_case = test::TestCase<TestEngine>(function);
    // X
    test_case.add_input<float>(
        {0.12249453f, 0.6127907f, 0.5001741f, 0.5124603f, 0.04329684f, 0.023834f});

    // Ht
    test_case.add_input<float>(
        {0.8598948f, 0.41189128f, 0.72824323f, 0.53940123f, 0.31485787f, 0.04053852f});

    // W
    test_case.add_input<float>(
        {0.72259396f, 0.11561195f, 0.9457856f,  0.19037509f, 0.6964006f,  0.33459795f, 0.5468904f,
         0.85646594f, 0.5101311f,  0.9712257f,  0.3687071f,  0.60280246f, 0.56943774f, 0.7475505f,
         0.2490578f,  0.86977345f, 0.85542053f, 0.29660386f, 0.49717373f, 0.7473479f,  0.53454477f,
         0.15974349f, 0.5804805f,  0.14303213f, 0.07514781f, 0.5865731f,  0.76409274f});
    // R
    test_case.add_input<float>(
        {0.91382647f, 0.41527033f, 0.28040004f, 0.23601337f, 0.04471736f, 0.03888785f, 0.06308217f,
         0.44844428f, 0.29384327f, 0.49037653f, 0.50421673f, 0.7366393f,  0.63143945f, 0.00277612f,
         0.37198433f, 0.06966069f, 0.4613444f,  0.10999731f, 0.78273284f, 0.21453214f, 0.10751773f,
         0.18332677f, 0.1326976f,  0.9998985f,  0.19263928f, 0.10979804f, 0.52575564f});

    // B (the sum of biases for W and R for z and r gates, and separately for W and R for h gate)
    test_case.add_input<float>({0.09875853f + 0.5151927f,
                                0.37801138f + 0.708666f,
                                0.7729636f + 0.55303884f,
                                0.78493553f + 0.03424145f,
                                0.5662702f + 0.81109315f,
                                0.12406381f + 0.30524766f,
                                0.66729516f,
                                0.7752771f,
                                0.78819966f,
                                0.6606634f,
                                0.99040645f,
                                0.21112025f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.8598948f, 0.41189128f, 0.72824323f, 0.53940123f, 0.31485787f, 0.04053852f});

    test_case.run();
}
