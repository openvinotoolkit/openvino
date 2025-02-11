// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "common_test_utils/all_close.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ndarray.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/test_tools.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/rnn_sequence.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");
static std::string s_device = backend_name_to_device("${BACKEND_NAME}");

// ONNX LSTM tests (implemented by OpenVINO LSTMCell and LSTMSequence)
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lstm_fwd_default_const) {
    auto model = convert_model("lstm_fwd_default_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0.68172926f, 1.1405563f, -0.03931177f, -0.03759607f});  // X

    test_case.add_expected_output<float>(Shape{2, 1, 1, 2},
                                         {-0.063373f, -0.20347191f, -0.07230289f, -0.13298286f});  // Y_data
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {-0.07230289f, -0.13298286f});            // Y_h_data
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {-0.1557954f, -0.24502525f});             // Y_c_data

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lstm_reverse_const) {
    auto model = convert_model("lstm_reverse_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0.68172926f, 1.1405563f, -0.03931177f, -0.03759607f});  // X

    test_case.add_expected_output<float>(Shape{2, 1, 1, 2},
                                         {-0.06082131f, -0.19985214f, 0.00860566f, 0.00920492f});  // Y_data
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {-0.06082131f, -0.19985214f});            // Y_h_data
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {-0.25917438f, -0.3832652f});             // Y_c_data

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lstm_bidir_const) {
    auto model = convert_model("lstm_bidir_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0.68172926f, 1.1405563f, -0.03931177f, -0.03759607f});  // X

    test_case.add_expected_output<float>(Shape{2, 2, 1, 2},
                                         {-0.063373f,
                                          -0.20347191f,
                                          -0.06082131f,
                                          -0.19985214f,
                                          -0.07230289f,
                                          -0.13298286f,
                                          0.00860566f,
                                          0.00920492f});  // Y_data
    test_case.add_expected_output<float>(Shape{2, 1, 2},
                                         {-0.07230289f, -0.13298286f, -0.06082131f, -0.19985214f});  // Y_h_data
    test_case.add_expected_output<float>(Shape{2, 1, 2},
                                         {-0.1557954f, -0.24502525f, -0.25917438f, -0.3832652f});  // Y_c_data

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lstm_fwd_with_clip_const) {
    auto model = convert_model("lstm_fwd_clip_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0.68172926f, 1.1405563f, -0.03931177f, -0.03759607f});  // X

    test_case.add_expected_output<float>(Shape{2, 1, 1, 2},
                                         {-0.02391884f, -0.02744377f, -0.01024176f, -0.01188637f});  // Y_data
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {-0.01024176f, -0.01188637f});              // Y_h_data
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {-0.02039271f, -0.02353566f});              // Y_c_data

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lstm_fwd_mixed_seq_const) {
    auto model = convert_model("lstm_fwd_mixed_seq_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0.68172926f, 1.1405563f, -0.03931177f, -0.03759607f});  // X

    test_case.add_expected_output<float>(Shape{2, 1, 2, 3},
                                         {0.13528088f,
                                          -0.1779867f,
                                          -0.07448981f,
                                          0.14769037f,
                                          -0.16327181f,
                                          -0.10419653f,
                                          0.,
                                          0.,
                                          0.,
                                          0.08759661f,
                                          -0.04002844f,
                                          -0.08617793f});  // Y_data
    test_case.add_expected_output<float>(
        Shape{1, 2, 3},
        {0.13528088f, -0.1779867f, -0.07448981f, 0.08759661f, -0.04002844f, -0.08617793f});  // Y_h_data
    test_case.add_expected_output<float>(
        Shape{1, 2, 3},
        {0.367563f, -0.43762812f, -0.20435227f, 0.17330585f, -0.0732716f, -0.18809439f});  // Y_c_data

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lstm_reverse_mixed_seq_const) {
    auto model = convert_model("lstm_reverse_mixed_seq_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0.68172926f, 1.1405563f, -0.03931177f, -0.03759607f});  // X

    test_case.add_expected_output<float>(Shape{2, 1, 2, 3},
                                         {0.13528088f,
                                          -0.1779867f,
                                          -0.07448981f,
                                          0.14696799f,
                                          -0.15571019f,
                                          -0.10270946f,
                                          0.,
                                          0.,
                                          0.,
                                          -0.01110403f,
                                          0.0228607f,
                                          0.00397353f});  // Y_data
    test_case.add_expected_output<float>(
        Shape{1, 2, 3},
        {0.13528088f, -0.1779867f, -0.07448981f, 0.14696799f, -0.15571019f, -0.10270946f});  // Y_h_data
    test_case.add_expected_output<float>(
        Shape{1, 2, 3},
        {0.367563f, -0.43762812f, -0.20435227f, 0.50598085f, -0.42627674f, -0.3641275f});  // Y_c_data

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lstm_bidir_mixed_seq_const) {
    auto model = convert_model("lstm_bidir_mixed_seq_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(
        {0.68172926f, 1.1405563f, -0.03931177f, -0.03759607f, 1.1397027f, 0.60444903f, 1.3246384f, -0.28191715f});  // X

    test_case.add_expected_output<float>(Shape{2, 2, 2, 2},
                                         {-0.063373f,
                                          -0.20347191f,
                                          0.00860566f,
                                          0.00920492f,
                                          -0.063373f,
                                          -0.20347191f,
                                          -0.12004475f,
                                          -0.12800421f,
                                          0.,
                                          0.,
                                          -0.19095606f,
                                          -0.12459831f,
                                          0.,
                                          0.,
                                          -0.1911628f,
                                          -0.12813942f});  // Y_data
    test_case.add_expected_output<float>(Shape{2, 2, 2},
                                         {-0.063373f,
                                          -0.20347191f,
                                          -0.19095606f,
                                          -0.12459831f,
                                          -0.063373f,
                                          -0.20347191f,
                                          -0.12004475f,
                                          -0.12800421f});  // Y_h_data
    test_case.add_expected_output<float>(Shape{2, 2, 2},
                                         {-0.2732999f,
                                          -0.38956356f,
                                          -0.48170844f,
                                          -0.34701264f,
                                          -0.2732999f,
                                          -0.38956356f,
                                          -0.27130172f,
                                          -0.253659f});  // Y_c_data

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lstm_fwd_mixed_seq) {
    auto model = convert_model("lstm_fwd_mixed_seq.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    int hidden_size{3};
    test_case.add_input<float>({1.f, 2.f, 10.f, 11.f});                                                // X
    test_case.add_input<float>({0.1f, 0.2f, 0.3f, 0.4f, 1.f, 2.f, 3.f, 4.f, 10.f, 11.f, 12.f, 13.f});  // W
    test_case.add_input(std::vector<float>(4 * hidden_size * hidden_size, 0.1f));                      // R
    test_case.add_input(std::vector<float>(8 * hidden_size, 0.0f));                                    // B
    test_case.add_input<int>({1, 2});                                                                  // seq_lengths

    test_case.add_expected_output<float>(Shape{2, 1, 2, 3},
                                         {0.28828835f,
                                          0.36581863f,
                                          0.45679406f,
                                          0.34526032f,
                                          0.47220859f,
                                          0.55850911f,
                                          0.f,
                                          0.f,
                                          0.f,
                                          0.85882828f,
                                          0.90703777f,
                                          0.92382453f});  // Y_data
    test_case.add_expected_output<float>(
        Shape{1, 2, 3},
        {0.28828835f, 0.36581863f, 0.45679406f, 0.85882828f, 0.90703777f, 0.92382453f});  // Y_h_data
    test_case.add_expected_output<float>(
        Shape{1, 2, 3},
        {0.52497941f, 0.54983425f, 0.5744428f, 1.3249796f, 1.51063104f, 1.61451544f});  // Y_c_data

    // We have to enlarge tolerance bits to 3 - it's only one bit more than default value.
    // The discrepancies may occur at most on 7th decimal position.
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lstm_fwd_hardsigmoid_activation) {
    auto model = convert_model("lstm_fwd_hardsigmoid_activation.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // X
    test_case.add_input<float>({-0.455351f, -0.276391f, -0.185934f, -0.269585f});
    // W
    test_case.add_input<float>({-0.494659f,
                                0.0453352f,
                                -0.487793f,
                                0.417264f,
                                -0.0175329f,
                                0.489074f,
                                -0.446013f,
                                0.414029f,
                                -0.0091708f,
                                -0.255364f,
                                -0.106952f,
                                -0.266717f,
                                -0.0888852f,
                                -0.428709f,
                                -0.283349f,
                                0.208792f});
    // R
    test_case.add_input<float>({0.146626f,
                                -0.0620289f,
                                -0.0815302f,
                                0.100482f,
                                -0.219535f,
                                -0.306635f,
                                -0.28515f,
                                -0.314112f,
                                -0.228172f,
                                0.405972f,
                                0.31576f,
                                0.281487f,
                                -0.394864f,
                                0.42111f,
                                -0.386624f,
                                -0.390225f});

    // Y
    test_case.add_expected_output<float>(Shape{2, 1, 1, 2}, {0.09086666f, 0.04378549f, 0.12914555f, 0.00257774f});
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {0.12914555f, 0.00257774f});
    // Y_c
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {0.19017234f, 0.00356848f});

    // The discrepancies occur at most at 18th mantissa bit - 8th decimal position.
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 4);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lstm_fwd_large_batch_no_clip) {
    auto model = convert_model("lstm_fwd_large_batch_no_clip.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    std::size_t seq_length = 2;
    std::size_t batch_size = 32;
    std::size_t input_size = 1;
    std::size_t hidden_size = 3;

    std::vector<float> in_X(seq_length * batch_size * input_size);
    std::iota(std::begin(in_X), std::end(in_X), 1.f);
    std::vector<float> in_R(4 * hidden_size * hidden_size, 0.1f);

    // X
    test_case.add_input<float>(in_X);
    // W
    test_case.add_input<float>({0.1f, 0.2f, 0.3f, 0.4f, 1.f, 2.f, 3.f, 4.f, 10.f, 11.f, 12.f, 13.f});
    // R
    test_case.add_input<float>(in_R);

    // Y_h_data
    test_case.add_expected_output<float>(
        Shape{1, batch_size, hidden_size},
        {0.90387899f, 0.9135572f,  0.91772245f, 0.90897038f, 0.92132433f, 0.92825467f, 0.91365823f, 0.92815113f,
         0.93676105f, 0.91799162f, 0.93406357f, 0.94344562f, 0.92199681f, 0.93912057f, 0.94859476f, 0.92569357f,
         0.94340185f, 0.95250664f, 0.92909964f, 0.94699686f, 0.95545127f, 0.93223207f, 0.94999634f, 0.95765468f,
         0.93510761f, 0.9524867f,  0.95929726f, 0.93774272f, 0.9545467f,  0.96051891f, 0.9401536f,  0.95624603f,
         0.96142619f, 0.94235605f, 0.95764499f, 0.96209939f, 0.94436539f, 0.95879495f, 0.96259862f, 0.94619635f,
         0.95973921f, 0.96296872f, 0.94786299f, 0.96051397f, 0.96324302f, 0.94937864f, 0.96114929f, 0.96344629f,
         0.95075587f, 0.96167006f, 0.96359692f, 0.95200645f, 0.96209679f, 0.96370852f, 0.95314133f, 0.9624464f,
         0.9637912f,  0.95417069f, 0.96273278f, 0.96385246f, 0.95510395f, 0.96296733f, 0.96389785f, 0.95594975f,
         0.96315942f, 0.96393147f, 0.95671607f, 0.96331673f, 0.96395638f, 0.9574102f,  0.96344554f, 0.96397483f,
         0.9580388f,  0.96355102f, 0.9639885f,  0.95860795f, 0.96363739f, 0.96399863f, 0.95912322f, 0.96370811f,
         0.96400613f, 0.95958963f, 0.96376601f, 0.96401169f, 0.96001179f, 0.96381342f, 0.96401581f, 0.96039386f,
         0.96385224f, 0.96401886f, 0.96073964f, 0.96388402f, 0.96402112f, 0.96105254f, 0.96391004f, 0.96402279f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lstm_mixed_seq_reverse) {
    auto model = convert_model("lstm_mixed_seq_reverse.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    size_t hidden_size = 3;

    // X
    test_case.add_input<float>({1.f, 2.f, 10.f, 11.f});
    // W
    test_case.add_input<float>({0.1f, 0.2f, 0.3f, 0.4f, 1.f, 2.f, 3.f, 4.f, 10.f, 11.f, 12.f, 13.f});
    // R
    test_case.add_input<float>(std::vector<float>(4 * hidden_size * hidden_size, 0.1f));
    // sequence_lens
    test_case.add_input<int>({1, 2});

    // Y
    test_case.add_expected_output<float>(Shape{2, 1, 2, 3},
                                         {0.28828844f,
                                          0.36581877f,
                                          0.45679423f,
                                          0.64046413f,
                                          0.82303363f,
                                          0.91610711f,
                                          0.f,
                                          0.f,
                                          0.f,
                                          0.62759886f,
                                          0.71640738f,
                                          0.74624585f});
    // Y_h
    test_case.add_expected_output<float>(
        Shape{1, 2, 3},
        {0.28828844f, 0.36581877f, 0.45679423f, 0.64046413f, 0.82303363f, 0.91610711f});
    // Y_c
    test_case.add_expected_output<float>(Shape{1, 2, 3},
                                         {0.52497941f, 0.54983425f, 0.5744428f, 1.34960834f, 1.54772296f, 1.65633056f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_import_only_lstm_dynamic_batch_seq_all_inputs) {
    auto model = convert_model("dynamic_shapes/lstm_dyn_batch_seq.onnx");

    auto batch_size = Dimension::dynamic();
    auto seq_length = Dimension::dynamic();
    int64_t hidden_size = 3;
    int64_t num_directions = 1;
    auto Y_expected_output = PartialShape{batch_size, num_directions, seq_length, hidden_size};
    auto Y_h_expected_output = PartialShape{num_directions, batch_size, hidden_size};
    auto Y_c_expected_output = PartialShape{num_directions, batch_size, hidden_size};

    EXPECT_EQ(model->get_output_size(), 3);
    EXPECT_EQ(model->get_output_partial_shape(0), Y_expected_output);
    EXPECT_EQ(model->get_output_partial_shape(1), Y_h_expected_output);
    EXPECT_EQ(model->get_output_partial_shape(2), Y_c_expected_output);

    EXPECT_EQ(count_ops_of_type<op::v5::LSTMSequence>(model), 1);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_import_only_lstm_dynamic_batch_seq_3_inputs) {
    auto model = convert_model("dynamic_shapes/lstm_dyn_batch_seq_3_inputs.onnx");

    auto batch_size = Dimension::dynamic();
    auto seq_length = Dimension::dynamic();
    int64_t hidden_size = 3;
    int64_t num_directions = 1;
    auto Y_expected_output = PartialShape{batch_size, num_directions, seq_length, hidden_size};
    auto Y_h_expected_output = PartialShape{num_directions, batch_size, hidden_size};
    auto Y_c_expected_output = PartialShape{num_directions, batch_size, hidden_size};

    EXPECT_EQ(model->get_output_size(), 3);
    EXPECT_EQ(model->get_output_partial_shape(0), Y_expected_output);
    EXPECT_EQ(model->get_output_partial_shape(1), Y_h_expected_output);
    EXPECT_EQ(model->get_output_partial_shape(2), Y_c_expected_output);

    EXPECT_EQ(count_ops_of_type<op::v5::LSTMSequence>(model), 1);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lstm_dynamic_batch_size_and_seq_len) {
    auto model = convert_model("lstm_dynamic_batch_size_and_seq_len.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1, 2, 3, 4, 5, 6});

    test_case.add_expected_output<float>(Shape{1, 1, 3, 2},
                                         {0.761594f, 0.761594f, 0.761594f, 0.761594f, 0.761594f, 0.761594f});  // Y
    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {0.761594f, 0.761594f, 0.761594f, 0.761594f, 0.761594f, 0.761594f});  // Y_c
    test_case.add_expected_output<float>(Shape{1, 3, 2}, {1, 1, 1, 1, 1, 1});                                  // Y_h

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

// RNNLikeSequenceOp test fixture for test setup reuse
class GRUSequenceOp : public testing::Test {
public:
    std::vector<float> in_X{0.68172926f, 1.1405563f,   -0.03931177f, -0.03759607f, 0.22778925f, 1.2471468f,
                            0.2785642f,  0.5198979f,   0.3712886f,   -0.3194908f,  0.8448233f,  -0.62065625f,
                            1.2968333f,  -0.20370148f, 0.40204826f,  -0.23721986f, 0.3629822f,  -0.3819832f,
                            -0.7766345f, 0.19374144f,  1.1397027f,   0.60444903f,  1.3246384f,  -0.28191715f};

    std::vector<float> in_W{
        0.31403765f,  -0.16793324f, 1.388258f,    -0.6902954f,  -0.3994045f,  -0.7833511f,  -0.30992958f, 0.3557573f,
        -0.4682631f,  1.1741459f,   -2.414789f,   -0.42783254f, -0.82199496f, -0.03900861f, -0.43670088f, -0.53810567f,
        -0.10769883f, 0.75242394f,  -0.2507971f,  1.0447186f,   -1.4777364f,  0.19993274f,  0.925649f,    -2.282516f,
        0.95039636f,  1.5379831f,   -0.88576007f, 0.28566247f,  0.79292643f,  -0.04261953f,
    };

    std::vector<float> in_R{
        0.8490583f,   0.45121244f, -1.1799014f,  0.13536449f,  0.81328654f,  0.6017516f,   0.48475724f, -1.2136037f,
        0.16383322f,  1.5106261f,  1.1177503f,   0.23582461f,  0.5754652f,   0.43879887f,  0.7399294f,  0.4517558f,
        1.3536783f,   -0.4843166f, -1.1503736f,  -0.2458678f,  0.54523313f,  -0.08649993f, -0.6936281f, 1.002422f,
        -1.770847f,   -0.94642f,   -1.8135757f,  1.8819852f,   -0.10852333f, -0.26120332f, 1.0223165f,  -0.7468837f,
        0.28566906f,  0.92321056f, 0.22521864f,  1.1123824f,   -0.9298287f,  1.2141289f,   1.3470556f,  -0.32972014f,
        -1.6552197f,  -1.0998285f, 0.71901864f,  0.962846f,    -0.1366851f,  -2.6534476f,  -1.4992771f, -0.45793465f,
        0.4290477f,   0.9893151f,  0.2511034f,   0.12906462f,  0.7491512f,   0.3316756f,   1.0576645f,  -0.04618666f,
        1.3556088f,   1.2842374f,  0.7103014f,   0.52889013f,  0.30327162f,  1.5069056f,   0.16591893f, 1.5719851f,
        -2.099427f,   -1.010277f,  -0.52800924f, -0.22292352f, -0.55177474f, 1.3432894f,   0.8731192f,  -0.01055307f,
        -0.01138215f, 0.85698843f, -1.2615703f,
    };

    std::vector<float> in_B{
        0.5336702f,   1.6593654f,   -1.150011f,  0.00342217f,  0.799371f,   0.43780383f, -0.55082625f, 1.0774187f,
        -0.6065135f,  0.6434064f,   -1.5693754f, 1.4923384f,   1.1554348f,  -1.328159f,  0.24995533f,  0.15112682f,
        -0.34698758f, -0.10088819f, -0.2931625f, -0.47319615f, 0.66167855f, -1.1646721f, -0.09588219f, 0.5212928f,
        0.37182367f,  0.27342287f,  1.1613405f,  -0.75196224f, -1.5143642f, 0.20604452f,
    };

    std::vector<int32_t> in_sequence_lens{2, 3, 4};

    std::vector<float> in_initial_h{
        -0.4840693f,
        -1.4054376f,
        0.84533644f,
        -0.1160888f,
        -1.3724717f,
        1.978259f,
        -0.8500094f,
        -2.0120409f,
        0.89959633f,
        -0.5367942f,
        0.21188478f,
        1.7603784f,
        0.38752958f,
        -0.06706902f,
        -1.4290836f,
    };

    std::vector<float> in_bdir_W{
        0.31403765f,  -0.16793324f, 1.388258f,    -0.6902954f,  -0.3994045f,  -0.7833511f,  -0.30992958f, 0.3557573f,
        -0.4682631f,  1.1741459f,   -2.414789f,   -0.42783254f, -0.82199496f, -0.03900861f, -0.43670088f, -0.53810567f,
        -0.10769883f, 0.75242394f,  -0.2507971f,  1.0447186f,   -1.4777364f,  0.19993274f,  0.925649f,    -2.282516f,
        0.95039636f,  1.5379831f,   -0.88576007f, 0.28566247f,  0.79292643f,  -0.04261953f, 0.8490583f,   0.45121244f,
        -1.1799014f,  0.13536449f,  0.81328654f,  0.6017516f,   0.48475724f,  -1.2136037f,  0.16383322f,  1.5106261f,
        1.1177503f,   0.23582461f,  0.5754652f,   0.43879887f,  0.7399294f,   0.4517558f,   1.3536783f,   -0.4843166f,
        -1.1503736f,  -0.2458678f,  0.54523313f,  -0.08649993f, -0.6936281f,  1.002422f,    -1.770847f,   -0.94642f,
        -1.8135757f,  1.8819852f,   -0.10852333f, -0.26120332f,
    };

    std::vector<float> in_bdir_R{
        1.02231646e+00f,  -7.46883690e-01f, 2.85669059e-01f,  9.23210561e-01f,  2.25218639e-01f,  1.11238241e+00f,
        -9.29828703e-01f, 1.21412885e+00f,  1.34705555e+00f,  -3.29720140e-01f, -1.65521967e+00f, -1.09982848e+00f,
        7.19018638e-01f,  9.62845981e-01f,  -1.36685103e-01f, -2.65344763e+00f, -1.49927711e+00f, -4.57934648e-01f,
        4.29047704e-01f,  9.89315093e-01f,  2.51103401e-01f,  1.29064620e-01f,  7.49151170e-01f,  3.31675589e-01f,
        1.05766451e+00f,  -4.61866595e-02f, 1.35560882e+00f,  1.28423738e+00f,  7.10301399e-01f,  5.28890133e-01f,
        3.03271621e-01f,  1.50690556e+00f,  1.65918931e-01f,  1.57198513e+00f,  -2.09942698e+00f, -1.01027703e+00f,
        -5.28009236e-01f, -2.22923517e-01f, -5.51774740e-01f, 1.34328938e+00f,  8.73119175e-01f,  -1.05530666e-02f,
        -1.13821477e-02f, 8.56988430e-01f,  -1.26157033e+00f, 5.33670187e-01f,  1.65936542e+00f,  -1.15001094e+00f,
        3.42216762e-03f,  7.99371004e-01f,  4.37803835e-01f,  -5.50826252e-01f, 1.07741868e+00f,  -6.06513500e-01f,
        6.43406391e-01f,  -1.56937540e+00f, 1.49233842e+00f,  1.15543485e+00f,  -1.32815897e+00f, 2.49955326e-01f,
        1.51126817e-01f,  -3.46987575e-01f, -1.00888193e-01f, -2.93162495e-01f, -4.73196149e-01f, 6.61678553e-01f,
        -1.16467214e+00f, -9.58821923e-02f, 5.21292806e-01f,  3.71823668e-01f,  2.73422867e-01f,  1.16134048e+00f,
        -7.51962245e-01f, -1.51436424e+00f, 2.06044525e-01f,  -4.84069288e-01f, -1.40543759e+00f, 8.45336437e-01f,
        -1.16088800e-01f, -1.37247169e+00f, 1.97825897e+00f,  -8.50009382e-01f, -2.01204085e+00f, 8.99596334e-01f,
        -5.36794186e-01f, 2.11884782e-01f,  1.76037836e+00f,  3.87529582e-01f,  -6.70690164e-02f, -1.42908359e+00f,
        8.20716441e-01f,  7.34144002e-02f,  2.08005775e-02f,  -3.74185145e-01f, 2.27367446e-01f,  -4.54556733e-01f,
        -2.24295408e-01f, 3.42533922e+00f,  -3.13701063e-01f, 1.25070000e+00f,  -1.29529154e+00f, -4.87530619e-01f,
        6.51176691e-01f,  -8.81322920e-02f, -1.84014812e-01f, -6.68743193e-01f, -2.83598930e-01f, 1.24322104e+00f,
        -1.03440486e-01f, -4.63501781e-01f, 1.72944975e+00f,  -2.54249543e-01f, -1.60864544e+00f, 4.86483961e-01f,
        7.00442135e-01f,  -1.71952701e+00f, -2.44922549e-01f, -5.80028534e-01f, 6.99496418e-02f,  3.74598980e-01f,
        -1.19728017e+00f, 9.30128455e-01f,  -2.42379427e-01f, 6.40181661e-01f,  2.04856300e+00f,  -1.27523863e+00f,
        -4.75532770e-01f, 3.02047610e-01f,  -2.54939228e-01f, -1.33242559e+00f, -8.23140562e-01f, -1.09450793e+00f,
        -1.70845091e-01f, 1.31205237e+00f,  2.28988096e-01f,  -5.51795721e-01f, -9.49851334e-01f, 1.28619313e+00f,
        1.28273416e+00f,  2.92767227e-01f,  -3.92974496e-01f, 2.09084296e+00f,  -1.28314102e+00f, -1.19076264e+00f,
        -3.52258608e-02f, -4.47186083e-02f, 6.82157278e-01f,  -2.59570718e-01f, 1.50172567e+00f,  -2.76523419e-02f,
    };

protected:
    void SetUp() override {}
};

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_defaults_fwd_const) {
    auto model = convert_model("gru_defaults_fwd_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.3224981f,  -0.44282594f, 0.7499796f,   -0.12240417f, 0.12079421f,  0.02534253f,  0.02504562f,
            -0.0463777f,  0.01204534f,  -0.01497037f, -0.04651929f, -0.6264307f,  0.7236632f,   0.06250653f,
            0.02594197f,  -0.06868916f, -0.5412897f,  0.49794048f,  0.22239858f,  -0.11257736f, -0.23071964f,
            0.26079988f,  -0.07375772f, -0.21816255f, 0.18764113f,  -0.5228772f,  0.00575754f,  0.2514028f,
            -0.58864325f, 0.49843538f,  -0.6129046f,  -0.10794663f, 0.6544055f,   -0.70105773f, 0.5397687f,
            -0.35791716f, 0.3885092f,   -0.15291792f, -0.22324723f, 0.11557932f,  -0.42112932f, 0.26772985f,
            -0.38304564f, -0.05039781f, -0.5057976f,  0.5775348f,   -0.6736855f,  -0.20032284f, 0.03698462f,
            -0.7693824f,  -0.5831348f,  0.25767964f,  0.7121098f,   -0.35951245f, 0.39223647f,  -0.6645166f,
            0.37950075f,  0.59931314f,  -0.4741001f,  0.21156166f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             0.5775348f,
                                             -0.6736855f,
                                             -0.20032284f,
                                             0.03698462f,
                                             -0.7693824f,
                                             -0.5831348f,
                                             0.25767964f,
                                             0.7121098f,
                                             -0.35951245f,
                                             0.39223647f,
                                             -0.6645166f,
                                             0.37950075f,
                                             0.59931314f,
                                             -0.4741001f,
                                             0.21156166f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 7);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_defaults_fwd) {
    auto model = convert_model("gru_defaults_fwd.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.3224981f,  -0.44282594f, 0.7499796f,   -0.12240417f, 0.12079421f,  0.02534253f,  0.02504562f,
            -0.0463777f,  0.01204534f,  -0.01497037f, -0.04651929f, -0.6264307f,  0.7236632f,   0.06250653f,
            0.02594197f,  -0.06868916f, -0.5412897f,  0.49794048f,  0.22239858f,  -0.11257736f, -0.23071964f,
            0.26079988f,  -0.07375772f, -0.21816255f, 0.18764113f,  -0.5228772f,  0.00575754f,  0.2514028f,
            -0.58864325f, 0.49843538f,  -0.6129046f,  -0.10794663f, 0.6544055f,   -0.70105773f, 0.5397687f,
            -0.35791716f, 0.3885092f,   -0.15291792f, -0.22324723f, 0.11557932f,  -0.42112932f, 0.26772985f,
            -0.38304564f, -0.05039781f, -0.5057976f,  0.5775348f,   -0.6736855f,  -0.20032284f, 0.03698462f,
            -0.7693824f,  -0.5831348f,  0.25767964f,  0.7121098f,   -0.35951245f, 0.39223647f,  -0.6645166f,
            0.37950075f,  0.59931314f,  -0.4741001f,  0.21156166f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             0.5775348f,
                                             -0.6736855f,
                                             -0.20032284f,
                                             0.03698462f,
                                             -0.7693824f,
                                             -0.5831348f,
                                             0.25767964f,
                                             0.7121098f,
                                             -0.35951245f,
                                             0.39223647f,
                                             -0.6645166f,
                                             0.37950075f,
                                             0.59931314f,
                                             -0.4741001f,
                                             0.21156166f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 7);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_fwd_activations_const) {
    // activations: relu, sigmoid
    auto model = convert_model("gru_fwd_activations_relu_sigmoid_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{0.30736187f,  0.10271017f,  0.91698503f, 0.3471303f,  -0.0123809f, 0.51264125f, 0.51235366f,
                           0.45471948f,  0.50601995f,  0.49260828f, 0.4781971f,  0.0668709f,  0.89421916f, 0.33762455f,
                           -0.19021586f, 0.6881336f,   0.7331965f,  0.8887774f,  0.34048334f, 0.38408905f, 0.49962956f,
                           0.2948451f,   0.3651103f,   0.33406913f, 0.57418096f, 0.49882296f, 0.4321446f,  0.97142136f,
                           0.20714557f,  0.66270787f,  0.53192705f, 0.46424377f, 0.9647801f,  0.19583187f, 0.7362316f,
                           0.48205143f,  -0.04748845f, 0.27395952f, 0.35897565f, 0.5801568f,  0.5889811f,  0.36110958f,
                           1.3433081f,   0.29702073f,  0.5709667f,  0.936689f,   0.84129435f, 1.1782551f,  0.23925206f,
                           0.57521456f,  0.43502977f,  -0.5664091f, 0.6758457f,  0.2958132f,  0.70932186f, 0.4411352f,
                           -0.1717428f,  1.7761463f,   0.14413449f, 0.73801273f});
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{0.936689f,
                                                            0.84129435f,
                                                            1.1782551f,
                                                            0.23925206f,
                                                            0.57521456f,
                                                            0.43502977f,
                                                            -0.5664091f,
                                                            0.6758457f,
                                                            0.2958132f,
                                                            0.70932186f,
                                                            0.4411352f,
                                                            -0.1717428f,
                                                            1.7761463f,
                                                            0.14413449f,
                                                            0.73801273f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 5);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_fwd_activations_relu_hardsigmoid) {
    // activations: relu, hardsigmoid
    auto model = convert_model("gru_fwd_activations_relu_hardsigmoid.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            0.33636323f, 0.08874974f, 0.9804139f,  0.35797057f, -0.01193848f, 0.51011515f, 0.50988495f,  0.4592467f,
            0.5048162f,  0.4940862f,  0.4825466f,  0.f,         0.9269162f,   0.3328298f,  -0.18795171f, 0.69541144f,
            0.7612694f,  0.937299f,   0.3463983f,  0.38764104f, 0.49957055f,  0.27359068f, 0.38423678f,  0.3618936f,
            0.55977404f, 0.5223568f,  0.46266305f, 1.016379f,   0.22654215f,  0.6347567f,  0.53541327f,  0.46684968f,
            1.0639775f,  0.21325049f, 0.70507824f, 0.48425108f, -0.05370265f, 0.3055008f,  0.38166368f,  0.5645658f,
            0.5998517f,  0.42573926f, 1.4539189f,  0.31789488f, 0.5567502f,   1.f,         0.92153484f,  1.4015231f,
            0.24147032f, 0.5783859f,  0.42785907f, -0.5690068f, 0.69624555f,  0.32291538f, 0.68179333f,  0.50179297f,
            0.0067991f,  2.043301f,   0.12669492f, 0.7062868f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             1.f,
                                             0.92153484f,
                                             1.4015231f,
                                             0.24147032f,
                                             0.5783859f,
                                             0.42785907f,
                                             -0.5690068f,
                                             0.69624555f,
                                             0.32291538f,
                                             0.68179333f,
                                             0.50179297f,
                                             0.0067991f,
                                             2.043301f,
                                             0.12669492f,
                                             0.7062868f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 5);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_fwd_mixed_seq_len) {
    auto model = convert_model("gru_fwd_mixed_seq_len.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);
    test_case.add_input<float>(in_B);
    test_case.add_input<int>(in_sequence_lens);
    test_case.add_input<float>(in_initial_h);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.9559332f,  0.4372494f,   0.9967716f,   -0.9079381f,  -1.2538278f,  1.9265908f,   -0.8437393f,
            -1.2057271f,  -0.25887525f, -0.52679026f, -0.3619178f,  0.67928517f,  0.9486744f,   -0.12006134f,
            -1.3862017f,  -0.98941356f, 0.80389524f,  0.97586197f,  -0.9343586f,  -0.74858856f, 1.797039f,
            -0.7873732f,  -0.72469383f, -0.5866635f,  -0.42103744f, -0.8406298f,  0.85877097f,  0.6349921f,
            -0.55897295f, -0.6168443f,  0.f,          0.f,          0.f,          0.f,          0.f,
            1.577129f,    -0.6935871f,  -0.304804f,   -0.75392795f, -0.20703818f, -0.93796504f, 0.9220495f,
            0.36017662f,  -0.7007159f,  0.06962098f,  0.f,          0.f,          0.f,          0.f,
            0.f,          0.f,          0.f,          0.f,          0.f,          0.f,          -0.96323603f,
            0.9265786f,   0.54976916f,  -0.8037839f,  0.73501444f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             -0.98941356f,
                                             0.80389524f,
                                             0.97586197f,
                                             -0.9343586f,
                                             -0.74858856f,
                                             1.577129f,
                                             -0.6935871f,
                                             -0.304804f,
                                             -0.75392795f,
                                             -0.20703818f,
                                             -0.96323603f,
                                             0.9265786f,
                                             0.54976916f,
                                             -0.8037839f,
                                             0.73501444f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 3);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_fwd_mixed_seq_len_const) {
    auto model = convert_model("gru_fwd_mixed_seq_len_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{-0.9559332f,  0.4372494f,   0.9967716f,   -0.9079381f,  -1.2538278f,  1.9265908f,
                           -0.8437393f,  -1.2057271f,  -0.25887525f, -0.52679026f, -0.3619178f,  0.67928517f,
                           0.9486744f,   -0.12006134f, -1.3862017f,  -0.98941356f, 0.80389524f,  0.97586197f,
                           -0.9343586f,  -0.74858856f, 1.797039f,    -0.7873732f,  -0.72469383f, -0.5866635f,
                           -0.42103744f, -0.8406298f,  0.85877097f,  0.6349921f,   -0.55897295f, -0.6168443f,
                           0.,           0.,           0.,           0.,           0.,           1.577129f,
                           -0.6935871f,  -0.304804f,   -0.75392795f, -0.20703818f, -0.93796504f, 0.9220495f,
                           0.36017662f,  -0.7007159f,  0.06962098f,  0.,           0.,           0.,
                           0.,           0.,           0.,           0.,           0.,           0.,
                           0.,           -0.96323603f, 0.9265786f,   0.54976916f,  -0.8037839f,  0.73501444f});
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{-0.98941356f,
                                                            0.80389524f,
                                                            0.97586197f,
                                                            -0.9343586f,
                                                            -0.74858856f,
                                                            1.577129f,
                                                            -0.6935871f,
                                                            -0.304804f,
                                                            -0.75392795f,
                                                            -0.20703818f,
                                                            -0.96323603f,
                                                            0.9265786f,
                                                            0.54976916f,
                                                            -0.8037839f,
                                                            0.73501444f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 3);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_reverse_mixed_seq_len_const) {
    auto model = convert_model("gru_reverse_mixed_seq_len_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{-0.9917215f,  0.07583051f,  0.997975f,    -0.9315585f,  -0.7483002f,  1.536813f,
                           -0.59922504f, -0.33637103f, -0.7565539f,  -0.23930266f, -0.7844553f,  1.0393485f,
                           0.73516595f,  -0.5616293f,  -0.09489207f, -0.9501128f,  0.7905356f,   0.9928266f,
                           -0.9153729f,  -1.1781745f,  1.7955453f,   -0.77754307f, -0.6831806f,  -0.6266324f,
                           -0.39791372f, -0.8030517f,  1.3107346f,   0.3700709f,   -0.49808976f, 0.52939236f,
                           0.,           0.,           0.,           0.,           0.,           1.9345565f,
                           -0.83817405f, -1.1433047f,  -0.35640514f, -0.5191339f,  -0.655544f,   1.3520991f,
                           0.42289692f,  -0.3171452f,  -0.3922639f,  0.,           0.,           0.,
                           0.,           0.,           0.,           0.,           0.,           0.,
                           0.,           -0.24612205f, 1.6415757f,   0.79883975f,  -0.18640287f, -1.0134869f});
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{-0.9917215f,
                                                            0.07583051f,
                                                            0.997975f,
                                                            -0.9315585f,
                                                            -0.7483002f,
                                                            1.536813f,
                                                            -0.59922504f,
                                                            -0.33637103f,
                                                            -0.7565539f,
                                                            -0.23930266f,
                                                            -0.7844553f,
                                                            1.0393485f,
                                                            0.73516595f,
                                                            -0.5616293f,
                                                            -0.09489207f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 3);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_bidir_mixed_seq_len_const) {
    auto model = convert_model("gru_bidir_mixed_seq_len_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 2, 3, 5},
        std::vector<float>{
            -0.3224981f,  -0.44282594f, 0.7499796f,   -0.12240417f, 0.12079421f,  0.02534254f,  0.02504561f,
            -0.0463777f,  0.01204535f,  -0.01497037f, -0.04651929f, -0.6264307f,  0.7236632f,   0.06250653f,
            0.02594197f,  0.0595789f,   0.40258542f,  -0.40646964f, 0.70320284f,  -0.02962421f, 0.10372428f,
            -0.38378227f, -0.4331268f,  -0.15696645f, -0.3451503f,  0.20918667f,  -0.59024405f, -0.845524f,
            0.60705113f,  -0.6336088f,  -0.0833023f,  -0.40062034f, 0.7579466f,   -0.12340625f, 0.04415433f,
            -0.24662054f, 0.27420586f,  -0.09122991f, -0.22768986f, 0.19980887f,  -0.218649f,   -0.5560231f,
            0.56177044f,  -0.25098884f, 0.15462328f,  0.0409361f,   0.17866893f,  -0.2782218f,  0.27396634f,
            -0.04992082f, 0.15353821f,  -0.4497267f,  -0.44631857f, -0.478926f,   -0.23017275f, 0.25369287f,
            -0.7369056f,  -0.73285f,    -0.5750758f,  -0.533177f,   0.,           0.,           0.,
            0.,           0.,           -0.45753813f, 0.5987347f,   -0.07046632f, -0.35819566f, 0.3916747f,
            -0.18096107f, -0.24415034f, 0.38435352f,  -0.29881003f, 0.07738188f,  0.,           0.,
            0.,           0.,           0.,           0.10390212f,  -0.29646862f, -0.20532897f, -0.31521815f,
            0.01049522f,  0.19370168f,  -0.6386781f,  -0.42919028f, -0.47081998f, -0.2954276f,  0.,
            0.,           0.,           0.,           0.,           0.,           0.,           0.,
            0.,           0.,           -0.50112087f, -0.11085765f, 0.5155622f,   -0.5635352f,  0.54762024f,
            0.,           0.,           0.,           0.,           0.,           0.,           0.,
            0.,           0.,           0.,           0.17058733f,  -0.6941011f,  -0.27862304f, -0.27050856f,
            -0.03864266f});
    // Y_h
    test_case.add_expected_output<float>(
        Shape{2, 3, 5},
        std::vector<float>{-0.0833023f, -0.40062034f, 0.7579466f,   -0.12340625f, 0.04415433f,  -0.45753813f,
                           0.5987347f,  -0.07046632f, -0.35819566f, 0.3916747f,   -0.50112087f, -0.11085765f,
                           0.5155622f,  -0.5635352f,  0.54762024f,  0.0595789f,   0.40258542f,  -0.40646964f,
                           0.70320284f, -0.02962421f, 0.10372428f,  -0.38378227f, -0.4331268f,  -0.15696645f,
                           -0.3451503f, 0.20918667f,  -0.59024405f, -0.845524f,   0.60705113f,  -0.6336088f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 4);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_rev_clip) {
    auto model = convert_model("gru_rev_clip.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.50679326f, -0.8251296f,  0.7804218f,   -0.1813852f,  0.00147036f,  -0.18647355f, 0.38888037f,
            -0.07898733f, -0.05150563f, -0.23335457f, -0.21705005f, -0.2966391f,  0.67461425f,  -0.1695634f,
            -0.09241624f, -0.10538863f, -0.6444952f,  -0.01815936f, -0.09695458f, -0.15107796f, -0.5036379f,
            0.56125206f,  0.12785181f,  -0.22290717f, 0.08662428f,  -0.5849108f,  0.4789885f,   -0.03569929f,
            -0.42043984f, 0.33464667f,  -0.01091215f, -0.42090097f, 0.24428985f,  -0.6002675f,  0.27305228f,
            -0.35063627f, 0.3717615f,   -0.00495788f, -0.00491725f, -0.27061304f, -0.3190831f,  0.3542383f,
            -0.17784928f, -0.12995736f, -0.30778408f, 0.47168806f,  -0.6330014f,  -0.1905269f,  0.26708886f,
            -0.19741398f, -0.3995853f,  -0.07459997f, 0.6749513f,   -0.36566192f, 0.32173023f,  -0.36364347f,
            0.13916425f,  0.3908174f,   -0.53085154f, 0.56740737f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             -0.50679326f,
                                             -0.8251296f,
                                             0.7804218f,
                                             -0.1813852f,
                                             0.00147036f,
                                             -0.18647355f,
                                             0.38888037f,
                                             -0.07898733f,
                                             -0.05150563f,
                                             -0.23335457f,
                                             -0.21705005f,
                                             -0.2966391f,
                                             0.67461425f,
                                             -0.1695634f,
                                             -0.09241624f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 8);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_rev_clip_const) {
    auto model = convert_model("gru_rev_clip_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.50679326f, -0.8251296f,  0.7804218f,   -0.1813852f,  0.00147036f,  -0.18647355f, 0.38888037f,
            -0.07898733f, -0.05150563f, -0.23335457f, -0.21705005f, -0.2966391f,  0.67461425f,  -0.1695634f,
            -0.09241624f, -0.10538863f, -0.6444952f,  -0.01815936f, -0.09695458f, -0.15107796f, -0.5036379f,
            0.56125206f,  0.12785181f,  -0.22290717f, 0.08662428f,  -0.5849108f,  0.4789885f,   -0.03569929f,
            -0.42043984f, 0.33464667f,  -0.01091215f, -0.42090097f, 0.24428985f,  -0.6002675f,  0.27305228f,
            -0.35063627f, 0.3717615f,   -0.00495788f, -0.00491725f, -0.27061304f, -0.3190831f,  0.3542383f,
            -0.17784928f, -0.12995736f, -0.30778408f, 0.47168806f,  -0.6330014f,  -0.1905269f,  0.26708886f,
            -0.19741398f, -0.3995853f,  -0.07459997f, 0.6749513f,   -0.36566192f, 0.32173023f,  -0.36364347f,
            0.13916425f,  0.3908174f,   -0.53085154f, 0.56740737f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             -0.50679326f,
                                             -0.8251296f,
                                             0.7804218f,
                                             -0.1813852f,
                                             0.00147036f,
                                             -0.18647355f,
                                             0.38888037f,
                                             -0.07898733f,
                                             -0.05150563f,
                                             -0.23335457f,
                                             -0.21705005f,
                                             -0.2966391f,
                                             0.67461425f,
                                             -0.1695634f,
                                             -0.09241624f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 8);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_reverse_const) {
    auto model = convert_model("gru_reverse_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.51097775f, -0.85767376f, 0.8065842f,   -0.1832461f,  -0.00109532f, -0.18766233f, 0.3910985f,
            -0.0617601f,  -0.05733761f, -0.23259571f, -0.22787738f, -0.3715533f,  0.70320934f,  -0.17635077f,
            -0.0972611f,  -0.11218601f, -0.660165f,   -0.03494868f, -0.07503931f, -0.15422714f, -0.5053969f,
            0.5710621f,   0.1448728f,   -0.225453f,   0.07250313f,  -0.5988957f,  0.48768237f,  0.00665835f,
            -0.42196327f, 0.2749501f,   -0.02106231f, -0.44533628f, 0.24044508f,  -0.5907899f,  0.26883256f,
            -0.3462156f,  0.3782666f,   0.00699124f,  -0.00378288f, -0.2990779f,  -0.32031405f, 0.3363319f,
            -0.1877775f,  -0.10781199f, -0.40970552f, 0.47168806f,  -0.6330014f,  -0.1905269f,  0.26708886f,
            -0.19741398f, -0.3995853f,  -0.07459997f, 0.691666f,    -0.36566192f, 0.32173023f,  -0.37267625f,
            0.1103513f,   0.3908174f,   -0.53085154f, 0.56740737f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             -0.51097775f,
                                             -0.85767376f,
                                             0.8065842f,
                                             -0.1832461f,
                                             -0.00109532f,
                                             -0.18766233f,
                                             0.3910985f,
                                             -0.0617601f,
                                             -0.05733761f,
                                             -0.23259571f,
                                             -0.22787738f,
                                             -0.3715533f,
                                             0.70320934f,
                                             -0.17635077f,
                                             -0.0972611f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 8);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_reverse) {
    auto model = convert_model("gru_reverse.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.51097775f, -0.85767376f, 0.8065842f,   -0.1832461f,  -0.00109532f, -0.18766233f, 0.3910985f,
            -0.0617601f,  -0.05733761f, -0.23259571f, -0.22787738f, -0.3715533f,  0.70320934f,  -0.17635077f,
            -0.0972611f,  -0.11218601f, -0.660165f,   -0.03494868f, -0.07503931f, -0.15422714f, -0.5053969f,
            0.5710621f,   0.1448728f,   -0.225453f,   0.07250313f,  -0.5988957f,  0.48768237f,  0.00665835f,
            -0.42196327f, 0.2749501f,   -0.02106231f, -0.44533628f, 0.24044508f,  -0.5907899f,  0.26883256f,
            -0.3462156f,  0.3782666f,   0.00699124f,  -0.00378288f, -0.2990779f,  -0.32031405f, 0.3363319f,
            -0.1877775f,  -0.10781199f, -0.40970552f, 0.47168806f,  -0.6330014f,  -0.1905269f,  0.26708886f,
            -0.19741398f, -0.3995853f,  -0.07459997f, 0.691666f,    -0.36566192f, 0.32173023f,  -0.37267625f,
            0.1103513f,   0.3908174f,   -0.53085154f, 0.56740737f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             -0.51097775f,
                                             -0.85767376f,
                                             0.8065842f,
                                             -0.1832461f,
                                             -0.00109532f,
                                             -0.18766233f,
                                             0.3910985f,
                                             -0.0617601f,
                                             -0.05733761f,
                                             -0.23259571f,
                                             -0.22787738f,
                                             -0.3715533f,
                                             0.70320934f,
                                             -0.17635077f,
                                             -0.0972611f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 8);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_fwd_bias_initial_h_const) {
    auto model = convert_model("gru_fwd_bias_initial_h_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.9559332f,  0.4372494f,   0.9967716f,   -0.9079381f,  -1.2538278f,  1.9265908f,   -0.8437393f,
            -1.2057271f,  -0.25887525f, -0.52679026f, -0.3619178f,  0.67928517f,  0.9486744f,   -0.12006134f,
            -1.3862017f,  -0.98941356f, 0.80389524f,  0.97586197f,  -0.9343586f,  -0.74858856f, 1.797039f,
            -0.7873732f,  -0.72469383f, -0.5866635f,  -0.42103744f, -0.8406298f,  0.85877097f,  0.6349921f,
            -0.55897295f, -0.6168443f,  -0.99686503f, 0.87408733f,  0.87070423f,  -0.9564345f,  0.52932394f,
            1.577129f,    -0.6935871f,  -0.304804f,   -0.75392795f, -0.20703818f, -0.93796504f, 0.9220495f,
            0.36017662f,  -0.7007159f,  0.06962098f,  -0.22581682f, 0.9119905f,   -0.64628327f, -0.79374063f,
            -0.82321495f, 1.2853851f,   -0.6176347f,  0.6865668f,   -0.85147655f, 0.0379298f,   -0.96323603f,
            0.9265786f,   0.54976916f,  -0.8037839f,  0.73501444f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             -0.22581682f,
                                             0.9119905f,
                                             -0.64628327f,
                                             -0.79374063f,
                                             -0.82321495f,
                                             1.2853851f,
                                             -0.6176347f,
                                             0.6865668f,
                                             -0.85147655f,
                                             0.0379298f,
                                             -0.96323603f,
                                             0.9265786f,
                                             0.54976916f,
                                             -0.8037839f,
                                             0.73501444f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 5);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_fwd_bias_initial_h) {
    auto model = convert_model("gru_fwd_bias_initial_h.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);
    test_case.add_input<float>(in_B);
    test_case.add_input<float>(in_initial_h);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.9559332f,  0.4372494f,   0.9967716f,   -0.9079381f,  -1.2538278f,  1.9265908f,   -0.8437393f,
            -1.2057271f,  -0.25887525f, -0.52679026f, -0.3619178f,  0.67928517f,  0.9486744f,   -0.12006134f,
            -1.3862017f,  -0.98941356f, 0.80389524f,  0.97586197f,  -0.9343586f,  -0.74858856f, 1.797039f,
            -0.7873732f,  -0.72469383f, -0.5866635f,  -0.42103744f, -0.8406298f,  0.85877097f,  0.6349921f,
            -0.55897295f, -0.6168443f,  -0.99686503f, 0.87408733f,  0.87070423f,  -0.9564345f,  0.52932394f,
            1.577129f,    -0.6935871f,  -0.304804f,   -0.75392795f, -0.20703818f, -0.93796504f, 0.9220495f,
            0.36017662f,  -0.7007159f,  0.06962098f,  -0.22581682f, 0.9119905f,   -0.64628327f, -0.79374063f,
            -0.82321495f, 1.2853851f,   -0.6176347f,  0.6865668f,   -0.85147655f, 0.0379298f,   -0.96323603f,
            0.9265786f,   0.54976916f,  -0.8037839f,  0.73501444f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             -0.22581682f,
                                             0.9119905f,
                                             -0.64628327f,
                                             -0.79374063f,
                                             -0.82321495f,
                                             1.2853851f,
                                             -0.6176347f,
                                             0.6865668f,
                                             -0.85147655f,
                                             0.0379298f,
                                             -0.96323603f,
                                             0.9265786f,
                                             0.54976916f,
                                             -0.8037839f,
                                             0.73501444f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 4);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_bidirectional_const) {
    auto model = convert_model("gru_bidirectional_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 2, 3, 5},
        std::vector<float>{
            -0.3224981f,  -0.44282594f, 0.7499796f,   -0.12240417f, 0.12079421f,  0.02534253f,  0.02504562f,
            -0.0463777f,  0.01204534f,  -0.01497037f, -0.04651929f, -0.6264307f,  0.7236632f,   0.06250653f,
            0.02594197f,  0.06575559f,  0.34565696f,  -0.3178988f,  0.6183835f,   -0.02136152f, 0.11640755f,
            -0.45138f,    -0.64678776f, -0.09675756f, -0.37742358f, 0.20918667f,  -0.59024405f, -0.845524f,
            0.60705113f,  -0.6336088f,  -0.0833023f,  -0.40062034f, 0.7579466f,   -0.12340625f, 0.04415433f,
            -0.24662055f, 0.27420586f,  -0.09122991f, -0.22768986f, 0.19980885f,  -0.218649f,   -0.5560231f,
            0.56177044f,  -0.25098884f, 0.15462328f,  0.02859182f,  0.22456945f,  -0.16747908f, -0.10665483f,
            0.06054133f,  0.18795699f,  -0.49318847f, -0.6660372f,  -0.5589901f,  -0.42696574f, 0.25369287f,
            -0.7369056f,  -0.73285f,    -0.5750758f,  -0.533177f,   -0.34549737f, -0.33324608f, 0.74590445f,
            -0.48038307f, 0.40253335f,  -0.45753813f, 0.5987347f,   -0.07046633f, -0.35819566f, 0.3916747f,
            -0.18096107f, -0.24415034f, 0.38435352f,  -0.29881003f, 0.07738188f,  -0.04626282f, -0.34389234f,
            0.2419839f,   -0.01195046f, 0.12158976f,  0.1648429f,   -0.4124067f,  -0.4792929f,  -0.498473f,
            -0.28167045f, 0.19370168f,  -0.6386781f,  -0.42919028f, -0.47081998f, -0.2954276f,  0.47018337f,
            0.01509789f,  0.43945605f,  -0.31491262f, 0.14951898f,  -0.7645583f,  0.2566264f,   0.7295435f,
            -0.5008343f,  0.57549477f,  -0.50112087f, -0.11085765f, 0.5155622f,   -0.5635352f,  0.54762024f,
            -0.26451954f, 0.17519262f,  0.5203082f,   0.6119683f,   0.01544304f,  0.11548323f,  -0.14230084f,
            -0.2133323f,  -0.3981219f,  -0.06852704f, 0.17058733f,  -0.6941011f,  -0.27862304f, -0.27050856f,
            -0.03864266f,
        });
    // Y_h
    test_case.add_expected_output<float>(
        Shape{2, 3, 5},
        std::vector<float>{
            0.47018337f,  0.01509789f, 0.43945605f,  -0.31491262f, 0.14951898f, -0.7645583f, 0.2566264f,   0.7295435f,
            -0.5008343f,  0.57549477f, -0.50112087f, -0.11085765f, 0.5155622f,  -0.5635352f, 0.54762024f,  0.06575559f,
            0.34565696f,  -0.3178988f, 0.6183835f,   -0.02136152f, 0.11640755f, -0.45138f,   -0.64678776f, -0.09675756f,
            -0.37742358f, 0.20918667f, -0.59024405f, -0.845524f,   0.60705113f, -0.6336088f,
        });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 6);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_bidirectional) {
    auto model = convert_model("gru_bidirectional.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_bdir_W);
    test_case.add_input<float>(in_bdir_R);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 2, 3, 5},
        std::vector<float>{
            -0.3224981f,  -0.44282594f, 0.7499796f,   -0.12240417f, 0.12079421f,  0.02534253f,  0.02504562f,
            -0.0463777f,  0.01204534f,  -0.01497037f, -0.04651929f, -0.6264307f,  0.7236632f,   0.06250653f,
            0.02594197f,  0.06575559f,  0.34565696f,  -0.3178988f,  0.6183835f,   -0.02136152f, 0.11640755f,
            -0.45138f,    -0.64678776f, -0.09675756f, -0.37742358f, 0.20918667f,  -0.59024405f, -0.845524f,
            0.60705113f,  -0.6336088f,  -0.0833023f,  -0.40062034f, 0.7579466f,   -0.12340625f, 0.04415433f,
            -0.24662055f, 0.27420586f,  -0.09122991f, -0.22768986f, 0.19980885f,  -0.218649f,   -0.5560231f,
            0.56177044f,  -0.25098884f, 0.15462328f,  0.02859182f,  0.22456945f,  -0.16747908f, -0.10665483f,
            0.06054133f,  0.18795699f,  -0.49318847f, -0.6660372f,  -0.5589901f,  -0.42696574f, 0.25369287f,
            -0.7369056f,  -0.73285f,    -0.5750758f,  -0.533177f,   -0.34549737f, -0.33324608f, 0.74590445f,
            -0.48038307f, 0.40253335f,  -0.45753813f, 0.5987347f,   -0.07046633f, -0.35819566f, 0.3916747f,
            -0.18096107f, -0.24415034f, 0.38435352f,  -0.29881003f, 0.07738188f,  -0.04626282f, -0.34389234f,
            0.2419839f,   -0.01195046f, 0.12158976f,  0.1648429f,   -0.4124067f,  -0.4792929f,  -0.498473f,
            -0.28167045f, 0.19370168f,  -0.6386781f,  -0.42919028f, -0.47081998f, -0.2954276f,  0.47018337f,
            0.01509789f,  0.43945605f,  -0.31491262f, 0.14951898f,  -0.7645583f,  0.2566264f,   0.7295435f,
            -0.5008343f,  0.57549477f,  -0.50112087f, -0.11085765f, 0.5155622f,   -0.5635352f,  0.54762024f,
            -0.26451954f, 0.17519262f,  0.5203082f,   0.6119683f,   0.01544304f,  0.11548323f,  -0.14230084f,
            -0.2133323f,  -0.3981219f,  -0.06852704f, 0.17058733f,  -0.6941011f,  -0.27862304f, -0.27050856f,
            -0.03864266f,
        });
    // Y_h
    test_case.add_expected_output<float>(
        Shape{2, 3, 5},
        std::vector<float>{
            0.47018337f,  0.01509789f, 0.43945605f,  -0.31491262f, 0.14951898f, -0.7645583f, 0.2566264f,   0.7295435f,
            -0.5008343f,  0.57549477f, -0.50112087f, -0.11085765f, 0.5155622f,  -0.5635352f, 0.54762024f,  0.06575559f,
            0.34565696f,  -0.3178988f, 0.6183835f,   -0.02136152f, 0.11640755f, -0.45138f,   -0.64678776f, -0.09675756f,
            -0.37742358f, 0.20918667f, -0.59024405f, -0.845524f,   0.60705113f, -0.6336088f,
        });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 6);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_fwd_linear_before_reset_const) {
    auto model = convert_model("gru_fwd_linear_before_reset_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.32330805f, -0.06708707f, 0.9148428f,   -0.5182527f, 0.15030569f,  -0.29070354f, 0.20353599f,
            0.36028495f,  -0.5524303f,  0.15076958f,  -0.3330416f, -0.2412689f,  0.90464234f,  -0.46817362f,
            0.08000847f,  -0.63514394f, 0.25109228f,  0.7674645f,  -0.7781104f,  -0.07633221f, -0.5679979f,
            0.32793444f,  0.18232828f,  -0.756521f,   0.07898282f, -0.7205035f,  -0.02278003f, -0.14991446f,
            -0.86801296f, 0.4434091f,   -0.8497459f,  0.35516143f, 0.8932138f,   -0.8957482f,  0.4693949f,
            -0.74337614f, 0.43600178f,  0.51654255f,  -0.8376663f, -0.18606272f, -0.8050637f,  0.06592449f,
            0.13366115f,  -0.8945458f,  -0.66395104f, 0.140306f,   0.42112982f,  -0.15852913f, -0.74940586f,
            -0.7907575f,  -0.89268315f, 0.5274858f,   0.97432905f, -0.89276016f, 0.15256537f,  -0.91766477f,
            0.22483218f,  0.9143838f,   -0.9442929f,  0.33684915f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             0.140306f,
                                             0.42112982f,
                                             -0.15852913f,
                                             -0.74940586f,
                                             -0.7907575f,
                                             -0.89268315f,
                                             0.5274858f,
                                             0.97432905f,
                                             -0.89276016f,
                                             0.15256537f,
                                             -0.91766477f,
                                             0.22483218f,
                                             0.9143838f,
                                             -0.9442929f,
                                             0.33684915f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 4);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_fwd_linear_before_reset) {
    auto model = convert_model("gru_fwd_linear_before_reset.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);
    test_case.add_input<float>(in_B);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.32330805f, -0.06708707f, 0.9148428f,   -0.5182527f, 0.15030569f,  -0.29070354f, 0.20353599f,
            0.36028495f,  -0.5524303f,  0.15076958f,  -0.3330416f, -0.2412689f,  0.90464234f,  -0.46817362f,
            0.08000847f,  -0.63514394f, 0.25109228f,  0.7674645f,  -0.7781104f,  -0.07633221f, -0.5679979f,
            0.32793444f,  0.18232828f,  -0.756521f,   0.07898282f, -0.7205035f,  -0.02278003f, -0.14991446f,
            -0.86801296f, 0.4434091f,   -0.8497459f,  0.35516143f, 0.8932138f,   -0.8957482f,  0.4693949f,
            -0.74337614f, 0.43600178f,  0.51654255f,  -0.8376663f, -0.18606272f, -0.8050637f,  0.06592449f,
            0.13366115f,  -0.8945458f,  -0.66395104f, 0.140306f,   0.42112982f,  -0.15852913f, -0.74940586f,
            -0.7907575f,  -0.89268315f, 0.5274858f,   0.97432905f, -0.89276016f, 0.15256537f,  -0.91766477f,
            0.22483218f,  0.9143838f,   -0.9442929f,  0.33684915f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             0.140306f,
                                             0.42112982f,
                                             -0.15852913f,
                                             -0.74940586f,
                                             -0.7907575f,
                                             -0.89268315f,
                                             0.5274858f,
                                             0.97432905f,
                                             -0.89276016f,
                                             0.15256537f,
                                             -0.91766477f,
                                             0.22483218f,
                                             0.9143838f,
                                             -0.9442929f,
                                             0.33684915f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 4);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_gru_defaults_fwd_const_dynamic) {
    auto model = convert_model("dynamic_shapes/gru_defaults_fwd_const_dynamic.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{4, 3, 2}, in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.3224981f,  -0.44282594f, 0.7499796f,   -0.12240417f, 0.12079421f,  0.02534253f,  0.02504562f,
            -0.0463777f,  0.01204534f,  -0.01497037f, -0.04651929f, -0.6264307f,  0.7236632f,   0.06250653f,
            0.02594197f,  -0.06868916f, -0.5412897f,  0.49794048f,  0.22239858f,  -0.11257736f, -0.23071964f,
            0.26079988f,  -0.07375772f, -0.21816255f, 0.18764113f,  -0.5228772f,  0.00575754f,  0.2514028f,
            -0.58864325f, 0.49843538f,  -0.6129046f,  -0.10794663f, 0.6544055f,   -0.70105773f, 0.5397687f,
            -0.35791716f, 0.3885092f,   -0.15291792f, -0.22324723f, 0.11557932f,  -0.42112932f, 0.26772985f,
            -0.38304564f, -0.05039781f, -0.5057976f,  0.5775348f,   -0.6736855f,  -0.20032284f, 0.03698462f,
            -0.7693824f,  -0.5831348f,  0.25767964f,  0.7121098f,   -0.35951245f, 0.39223647f,  -0.6645166f,
            0.37950075f,  0.59931314f,  -0.4741001f,  0.21156166f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             0.5775348f,
                                             -0.6736855f,
                                             -0.20032284f,
                                             0.03698462f,
                                             -0.7693824f,
                                             -0.5831348f,
                                             0.25767964f,
                                             0.7121098f,
                                             -0.35951245f,
                                             0.39223647f,
                                             -0.6645166f,
                                             0.37950075f,
                                             0.59931314f,
                                             -0.4741001f,
                                             0.21156166f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 7);
}

OPENVINO_TEST_F(${BACKEND_NAME}, GRUSequenceOp, onnx_model_import_only_gru_defaults_fwd_const_dynamic) {
    auto model = convert_model("dynamic_shapes/gru_defaults_fwd_const_dynamic.onnx");

    auto batch_size = Dimension::dynamic();
    auto seq_length = Dimension::dynamic();
    int64_t hidden_size = 5;
    int64_t num_directions = 1;
    auto Y_expected_output = PartialShape{batch_size, num_directions, seq_length, hidden_size};
    auto Y_h_expected_output = PartialShape{num_directions, batch_size, hidden_size};

    EXPECT_EQ(model->get_output_size(), 2);
    EXPECT_EQ(model->get_output_partial_shape(0), Y_expected_output);
    EXPECT_EQ(model->get_output_partial_shape(1), Y_h_expected_output);

    EXPECT_EQ(count_ops_of_type<op::v5::GRUSequence>(model), 1);
}

// RNNLikeSequenceOp test fixture for test setup reuse
class RNNSequenceOp : public testing::Test {
public:
    std::vector<float> in_X{
        0.68172926f, 1.1405563f,  -0.03931177f, -0.03759607f, 0.22778925f, 1.2471468f,   0.2785642f,  0.5198979f,
        0.3712886f,  -0.3194908f, 0.8448233f,   -0.62065625f, 1.2968333f,  -0.20370148f, 0.40204826f, -0.23721986f,
        0.3629822f,  -0.3819832f, -0.7766345f,  0.19374144f,  1.1397027f,  0.60444903f,  1.3246384f,  -0.28191715f,
    };

    std::vector<float> in_W{
        0.31403765f,
        -0.16793324f,
        1.388258f,
        -0.6902954f,
        -0.3994045f,
        -0.7833511f,
        -0.30992958f,
        0.3557573f,
        -0.4682631f,
        1.1741459f,
    };

    std::vector<float> in_R{
        -2.414789f,  -0.42783254f, -0.82199496f, -0.03900861f, -0.43670088f, -0.53810567f, -0.10769883f,
        0.75242394f, -0.2507971f,  1.0447186f,   -1.4777364f,  0.19993274f,  0.925649f,    -2.282516f,
        0.95039636f, 1.5379831f,   -0.88576007f, 0.28566247f,  0.79292643f,  -0.04261953f, 0.8490583f,
        0.45121244f, -1.1799014f,  0.13536449f,  0.81328654f,
    };

    std::vector<float> in_B{
        0.6017516f,
        0.48475724f,
        -1.2136037f,
        0.16383322f,
        1.5106261f,
        1.1177503f,
        0.23582461f,
        0.5754652f,
        0.43879887f,
        0.7399294f,
    };

    std::vector<int32_t> in_sequence_lens{2, 3, 4};

    std::vector<float> in_initial_h{
        0.4517558f,
        1.3536783f,
        -0.4843166f,
        -1.1503736f,
        -0.2458678f,
        0.54523313f,
        -0.08649993f,
        -0.6936281f,
        1.002422f,
        -1.770847f,
        -0.94642f,
        -1.8135757f,
        1.8819852f,
        -0.10852333f,
        -0.26120332f,
    };

    std::vector<float> in_bdir_W{
        0.31403765f,  -0.16793324f, 1.388258f,    -0.6902954f, -0.3994045f,  -0.7833511f,  -0.30992958f,
        0.3557573f,   -0.4682631f,  1.1741459f,   -2.414789f,  -0.42783254f, -0.82199496f, -0.03900861f,
        -0.43670088f, -0.53810567f, -0.10769883f, 0.75242394f, -0.2507971f,  1.0447186f,
    };

    std::vector<float> in_bdir_R{
        -1.4777364f, 0.19993274f,  0.925649f,   -2.282516f,  0.95039636f, 1.5379831f,  -0.88576007f, 0.28566247f,
        0.79292643f, -0.04261953f, 0.8490583f,  0.45121244f, -1.1799014f, 0.13536449f, 0.81328654f,  0.6017516f,
        0.48475724f, -1.2136037f,  0.16383322f, 1.5106261f,  1.1177503f,  0.23582461f, 0.5754652f,   0.43879887f,
        0.7399294f,  0.4517558f,   1.3536783f,  -0.4843166f, -1.1503736f, -0.2458678f, 0.54523313f,  -0.08649993f,
        -0.6936281f, 1.002422f,    -1.770847f,  -0.94642f,   -1.8135757f, 1.8819852f,  -0.10852333f, -0.26120332f,
        1.0223165f,  -0.7468837f,  0.28566906f, 0.92321056f, 0.22521864f, 1.1123824f,  -0.9298287f,  1.2141289f,
        1.3470556f,  -0.32972014f,
    };

protected:
    void SetUp() override {}
};

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_rnn_defaults_fwd_const) {
    auto model = convert_model("rnn_defaults_fwd_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            0.02254748f,  0.15776646f, -0.8229023f,  0.19205809f,  0.76984656f,  -0.00603169f, -0.02861464f,
            0.04512155f,  -0.0011912f, -0.02572936f, -0.13703543f, -0.49651444f, -0.78868157f, 0.3566854f,
            0.8758509f,   0.20788848f, 0.13481987f,  -0.756822f,   -0.121436f,   0.97542346f,  0.16959739f,
            0.63496053f,  0.1245538f,  -0.1970138f,  -0.56581646f, 0.8225869f,   0.9611373f,   -0.42990375f,
            -0.22925597f, 0.2226491f,  0.08246052f,  0.9798831f,   -0.13415998f, -0.5567714f,  0.78594816f,
            -0.34759718f, 0.11376679f, -0.07107389f, -0.5420871f,  -0.58504283f, -0.96065646f, 0.18588805f,
            -0.4870671f,  -0.1475982f, 0.82456505f,  -0.80264574f, -0.46370947f, 0.9719335f,   -0.7374159f,
            0.94937694f,  0.8814341f,  0.67015004f,  0.21958017f,  -0.8332769f,  -0.487742f,   0.9918536f,
            0.99563396f,  0.94866276f, -0.98504806f, -0.42824882f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             -0.80264574f,
                                             -0.46370947f,
                                             0.9719335f,
                                             -0.7374159f,
                                             0.94937694f,
                                             0.8814341f,
                                             0.67015004f,
                                             0.21958017f,
                                             -0.8332769f,
                                             -0.487742f,
                                             0.9918536f,
                                             0.99563396f,
                                             0.94866276f,
                                             -0.98504806f,
                                             -0.42824882f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 4);
}

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_rnn_defaults_fwd) {
    auto model = convert_model("rnn_defaults_fwd.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            0.02254748f,  0.15776646f, -0.8229023f,  0.19205809f,  0.76984656f,  -0.00603169f, -0.02861464f,
            0.04512155f,  -0.0011912f, -0.02572936f, -0.13703543f, -0.49651444f, -0.78868157f, 0.3566854f,
            0.8758509f,   0.20788848f, 0.13481987f,  -0.756822f,   -0.121436f,   0.97542346f,  0.16959739f,
            0.63496053f,  0.1245538f,  -0.1970138f,  -0.56581646f, 0.8225869f,   0.9611373f,   -0.42990375f,
            -0.22925597f, 0.2226491f,  0.08246052f,  0.9798831f,   -0.13415998f, -0.5567714f,  0.78594816f,
            -0.34759718f, 0.11376679f, -0.07107389f, -0.5420871f,  -0.58504283f, -0.96065646f, 0.18588805f,
            -0.4870671f,  -0.1475982f, 0.82456505f,  -0.80264574f, -0.46370947f, 0.9719335f,   -0.7374159f,
            0.94937694f,  0.8814341f,  0.67015004f,  0.21958017f,  -0.8332769f,  -0.487742f,   0.9918536f,
            0.99563396f,  0.94866276f, -0.98504806f, -0.42824882f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             -0.80264574f,
                                             -0.46370947f,
                                             0.9719335f,
                                             -0.7374159f,
                                             0.94937694f,
                                             0.8814341f,
                                             0.67015004f,
                                             0.21958017f,
                                             -0.8332769f,
                                             -0.487742f,
                                             0.9918536f,
                                             0.99563396f,
                                             0.94866276f,
                                             -0.98504806f,
                                             -0.42824882f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 4);
}

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_rnn_fwd_activations_const) {
    auto model = convert_model("rnn_fwd_activations_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            0.02255133f, 0.15909529f, 0.f,         0.19447318f, 1.019951f,   0.f,         0.f,         0.04515222f,
            0.f,         0.f,         0.f,         0.f,         0.f,         0.37308297f, 1.3576671f,  0.f,
            1.015355f,   0.00543064f, 0.10311858f, 1.426765f,   0.13313684f, 0.769961f,   0.14377424f, 0.f,
            0.f,         0.f,         2.9260807f,  0.5875195f,  0.f,         0.030334f,   0.f,         3.300393f,
            0.97026074f, 0.f,         0.7796261f,  0.f,         0.6755121f,  0.1155303f,  0.f,         0.f,
            0.f,         0.92621297f, 1.3119358f,  0.f,         0.03326398f, 0.f,         0.f,         2.4573548f,
            0.f,         1.5695758f,  0.f,         1.1791289f,  0.f,         0.f,         0.34451577f, 0.f,
            2.9556773f,  1.12296f,    0.f,         0.f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             0.f,
                                             0.f,
                                             2.4573548f,
                                             0.f,
                                             1.5695758f,
                                             0.f,
                                             1.1791289f,
                                             0.f,
                                             0.f,
                                             0.34451577f,
                                             0.f,
                                             2.9556773f,
                                             1.12296f,
                                             0.f,
                                             0.f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 5);
}

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_rnn_fwd_activations) {
    auto model = convert_model("rnn_fwd_activations.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            0.02255133f, 0.15909529f, 0.f,         0.19447318f, 1.019951f,   0.f,         0.f,         0.04515222f,
            0.f,         0.f,         0.f,         0.f,         0.f,         0.37308297f, 1.3576671f,  0.f,
            1.015355f,   0.00543064f, 0.10311858f, 1.426765f,   0.13313684f, 0.769961f,   0.14377424f, 0.f,
            0.f,         0.f,         2.9260807f,  0.5875195f,  0.f,         0.030334f,   0.f,         3.300393f,
            0.97026074f, 0.f,         0.7796261f,  0.f,         0.6755121f,  0.1155303f,  0.f,         0.f,
            0.f,         0.92621297f, 1.3119358f,  0.f,         0.03326398f, 0.f,         0.f,         2.4573548f,
            0.f,         1.5695758f,  0.f,         1.1791289f,  0.f,         0.f,         0.34451577f, 0.f,
            2.9556773f,  1.12296f,    0.f,         0.f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             0.f,
                                             0.f,
                                             2.4573548f,
                                             0.f,
                                             1.5695758f,
                                             0.f,
                                             1.1791289f,
                                             0.f,
                                             0.f,
                                             0.34451577f,
                                             0.f,
                                             2.9556773f,
                                             1.12296f,
                                             0.f,
                                             0.f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 5);
}

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_rnn_fwd_mixed_seq_len_const) {
    auto model = convert_model("rnn_fwd_mixed_seq_len_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            0.55277014f,  0.15672898f,  -0.25152922f, -0.63345766f, 0.99974346f,  0.94002223f, -0.97647303f,
            -0.9999884f,  0.9752002f,   0.97388494f,  0.9967754f,   0.96745205f,  0.7899921f,  0.92003024f,
            -0.43116868f, 0.11219919f,  0.895327f,    0.21749747f,  0.6617017f,   0.99962795f, 0.37670398f,
            0.7918401f,   -0.99966455f, 0.9961897f,   0.9995159f,   -0.84224236f, 0.92083716f, -0.99834263f,
            0.9435711f,   0.8485148f,   0.f,          0.f,          0.f,          0.f,         0.f,
            0.75459063f,  0.8326433f,   -0.99705976f, 0.62511444f,  0.99979305f,  0.99925995f, 0.94032586f,
            -0.86841005f, -0.8692311f,  0.9974319f,   0.f,          0.f,          0.f,         0.f,
            0.f,          0.f,          0.f,          0.f,          0.f,          0.f,         -0.30979204f,
            0.99138904f,  -0.10645419f, -0.18203181f, 0.9996245f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             0.11219919f,
                                             0.895327f,
                                             0.21749747f,
                                             0.6617017f,
                                             0.99962795f,
                                             0.75459063f,
                                             0.8326433f,
                                             -0.99705976f,
                                             0.62511444f,
                                             0.99979305f,
                                             -0.30979204f,
                                             0.99138904f,
                                             -0.10645419f,
                                             -0.18203181f,
                                             0.9996245f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 4);
}

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_rnn_fwd_mixed_seq_len) {
    auto model = convert_model("rnn_fwd_mixed_seq_len.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);
    test_case.add_input<float>(in_B);
    test_case.add_input<int>(in_sequence_lens);
    test_case.add_input<float>(in_initial_h);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            0.55277014f,  0.15672898f,  -0.25152922f, -0.63345766f, 0.99974346f,  0.94002223f, -0.97647303f,
            -0.9999884f,  0.9752002f,   0.97388494f,  0.9967754f,   0.96745205f,  0.7899921f,  0.92003024f,
            -0.43116868f, 0.11219919f,  0.895327f,    0.21749747f,  0.6617017f,   0.99962795f, 0.37670398f,
            0.7918401f,   -0.99966455f, 0.9961897f,   0.9995159f,   -0.84224236f, 0.92083716f, -0.99834263f,
            0.9435711f,   0.8485148f,   0.f,          0.f,          0.f,          0.f,         0.f,
            0.75459063f,  0.8326433f,   -0.99705976f, 0.62511444f,  0.99979305f,  0.99925995f, 0.94032586f,
            -0.86841005f, -0.8692311f,  0.9974319f,   0.f,          0.f,          0.f,         0.f,
            0.f,          0.f,          0.f,          0.f,          0.f,          0.f,         -0.30979204f,
            0.99138904f,  -0.10645419f, -0.18203181f, 0.9996245f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             0.11219919f,
                                             0.895327f,
                                             0.21749747f,
                                             0.6617017f,
                                             0.99962795f,
                                             0.75459063f,
                                             0.8326433f,
                                             -0.99705976f,
                                             0.62511444f,
                                             0.99979305f,
                                             -0.30979204f,
                                             0.99138904f,
                                             -0.10645419f,
                                             -0.18203181f,
                                             0.9996245f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 4);
}

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_rnn_reverse_mixed_seq_len_const) {
    auto model = convert_model("rnn_reverse_mixed_seq_len_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.27398264f, 0.96948624f, 0.26404798f,  0.8068119f,  0.99935544f, 0.73694086f,  0.44305325f, -0.9964632f,
            0.7063714f,   0.9999049f,  -0.7241098f,  0.08538079f, -0.785874f,  0.60833323f,  0.99999666f, 0.53703666f,
            0.0267657f,   0.37151086f, -0.68740594f, 0.9992448f,  0.3254757f,  0.7716811f,   -0.9996745f, 0.9957807f,
            0.9995338f,   0.9997339f,  0.9888724f,   -0.8992324f, -0.797282f,  0.98666525f,  0.,          0.,
            0.,           0.,          0.,           0.95711637f, -0.8986079f, -0.99998885f, 0.96265936f, 0.9380511f,
            -0.86523867f, 0.3528558f,  -0.99675506f, 0.946875f,   0.79539406f, 0.,           0.,          0.,
            0.,           0.,          0.,           0.,          0.,          0.,           0.,          0.99903f,
            0.9998094f,   0.9499353f,  0.6077225f,   -0.9921822f});
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{-0.27398264f,
                                                            0.96948624f,
                                                            0.26404798f,
                                                            0.8068119f,
                                                            0.99935544f,
                                                            0.73694086f,
                                                            0.44305325f,
                                                            -0.9964632f,
                                                            0.7063714f,
                                                            0.9999049f,
                                                            -0.7241098f,
                                                            0.08538079f,
                                                            -0.785874f,
                                                            0.60833323f,
                                                            0.99999666f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 4);
}

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_rnn_bidir_mixed_seq_len_const) {
    auto model = convert_model("rnn_bidir_mixed_seq_len_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 2, 3, 5},
        std::vector<float>{
            0.02254748f,  0.15776646f,  -0.8229023f,  0.19205809f,  0.76984656f,  -0.00603169f, -0.0286147f,
            0.04512155f,  -0.0011912f,  -0.02572936f, -0.13703543f, -0.49651444f, -0.78868157f, 0.3566854f,
            0.8758509f,   -0.99602485f, -0.8151508f,  -0.5803147f,  0.4985683f,   0.30210292f,  0.11550081f,
            -0.30236644f, 0.99622667f,  -0.8732492f,  -0.43772405f, -0.9284624f,  -0.5595875f,  0.9986867f,
            -0.18373811f, 0.8451735f,   -0.43823165f, -0.1904698f,  0.8320786f,   0.9830735f,   0.61861455f,
            0.19109797f,  0.64407f,     0.00962067f,  -0.32752877f, -0.5050589f,  -0.23455954f, 0.9517933f,
            0.9050665f,   0.91091585f,  -0.77941567f, -0.71390504f, -0.24422187f, -0.38115412f, 0.3462553f,
            0.44084883f,  -0.81455964f, -0.23556596f, 0.85043025f,  -0.7840209f,  -0.82087713f, -0.8349008f,
            -0.7880142f,  0.99017143f,  -0.9816452f,  -0.93827677f, 0.,           0.,           0.,
            0.,           0.,           0.28117967f,  0.20685148f,  0.01166701f,  -0.5441828f,  -0.5463747f,
            -0.85301256f, 0.52109087f,  -0.8317892f,  -0.9676957f,  -0.30258918f, 0.,           0.,
            0.,           0.,           0.,           -0.7010546f,  -0.3106169f,  -0.04788882f, -0.21822351f,
            -0.33518708f, -0.9073148f,  0.16276085f,  0.9518349f,   -0.8635942f,  -0.92539954f, 0.,
            0.,           0.,           0.,           0.,           0.,           0.,           0.,
            0.,           0.,           0.9948462f,   -0.6242633f,  -0.19065344f, -0.36072153f, -0.99407107f,
            0.,           0.,           0.,           0.,           0.,           0.,           0.,
            0.,           0.,           0.,           -0.9957684f,  -0.7924f,     -0.40261805f, -0.34061068f,
            -0.55580306f});
    // Y_h
    test_case.add_expected_output<float>(
        Shape{2, 3, 5},
        std::vector<float>{-0.43823165f, -0.1904698f,  0.8320786f,   0.9830735f,   0.61861455f,  0.28117967f,
                           0.20685148f,  0.01166701f,  -0.5441828f,  -0.5463747f,  0.9948462f,   -0.6242633f,
                           -0.19065344f, -0.36072153f, -0.99407107f, -0.99602485f, -0.8151508f,  -0.5803147f,
                           0.4985683f,   0.30210292f,  0.11550081f,  -0.30236644f, 0.99622667f,  -0.8732492f,
                           -0.43772405f, -0.9284624f,  -0.5595875f,  0.9986867f,   -0.18373811f, 0.8451735f});

    // loosest match @ mantissa bit:
    // 16 or next bit (0.01166688557714223862 vs 0.01166701037436723709)
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 7);
}

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_rnn_rev_clip_const) {
    auto model = convert_model("rnn_rev_clip_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            0.9416027f,   0.6461365f,   -0.8407804f,  -0.33646506f, 0.92833483f,  -0.9416027f,  0.65075886f,
            0.9416027f,   -0.33576548f, -0.10364902f, -0.9416027f,  -0.832458f,   -0.18187332f, 0.5103179f,
            0.5227027f,   -0.9416027f,  -0.90681225f, -0.9416027f,  0.5091027f,   0.8053496f,   0.6005076f,
            0.92147183f,  0.9416027f,   -0.8985506f,  0.28120112f,  0.9416027f,   0.9416027f,   0.9416027f,
            -0.92463756f, -0.9416027f,  0.79248047f,  0.9416027f,   -0.1611281f,  0.11231542f,  -0.8230629f,
            -0.2566173f,  0.16398644f,  -0.36077273f, -0.70470357f, 0.8011706f,   -0.59314847f, -0.41942674f,
            -0.20039755f, -0.6877927f,  -0.13850075f, -0.26959598f, -0.8372509f,  0.15711153f,  0.3000977f,
            0.53072214f,  0.25092757f,  0.82264745f,  -0.72998637f, -0.13731742f, 0.17423475f,  0.43279397f,
            0.9416027f,   -0.2988227f,  -0.4705984f,  -0.74036705f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             0.9416027f,
                                             0.6461365f,
                                             -0.8407804f,
                                             -0.33646506f,
                                             0.92833483f,
                                             -0.9416027f,
                                             0.65075886f,
                                             0.9416027f,
                                             -0.33576548f,
                                             -0.10364902f,
                                             -0.9416027f,
                                             -0.832458f,
                                             -0.18187332f,
                                             0.5103179f,
                                             0.5227027f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 3);
}

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_rnn_rev_clip) {
    auto model = convert_model("rnn_rev_clip.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            0.9416027f,   0.6461365f,   -0.8407804f,  -0.33646506f, 0.92833483f,  -0.9416027f,  0.65075886f,
            0.9416027f,   -0.33576548f, -0.10364902f, -0.9416027f,  -0.832458f,   -0.18187332f, 0.5103179f,
            0.5227027f,   -0.9416027f,  -0.90681225f, -0.9416027f,  0.5091027f,   0.8053496f,   0.6005076f,
            0.92147183f,  0.9416027f,   -0.8985506f,  0.28120112f,  0.9416027f,   0.9416027f,   0.9416027f,
            -0.92463756f, -0.9416027f,  0.79248047f,  0.9416027f,   -0.1611281f,  0.11231542f,  -0.8230629f,
            -0.2566173f,  0.16398644f,  -0.36077273f, -0.70470357f, 0.8011706f,   -0.59314847f, -0.41942674f,
            -0.20039755f, -0.6877927f,  -0.13850075f, -0.26959598f, -0.8372509f,  0.15711153f,  0.3000977f,
            0.53072214f,  0.25092757f,  0.82264745f,  -0.72998637f, -0.13731742f, 0.17423475f,  0.43279397f,
            0.9416027f,   -0.2988227f,  -0.4705984f,  -0.74036705f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             0.9416027f,
                                             0.6461365f,
                                             -0.8407804f,
                                             -0.33646506f,
                                             0.92833483f,
                                             -0.9416027f,
                                             0.65075886f,
                                             0.9416027f,
                                             -0.33576548f,
                                             -0.10364902f,
                                             -0.9416027f,
                                             -0.832458f,
                                             -0.18187332f,
                                             0.5103179f,
                                             0.5227027f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 3);
}

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_rnn_reverse_const) {
    auto model = convert_model("rnn_reverse_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            0.9963336f,   0.63758683f,  -0.82404625f, -0.38524252f, 0.9350034f,   -0.9918621f,  0.67038023f,
            0.9884596f,   -0.32398474f, -0.15730727f, -0.9970634f,  -0.831641f,   -0.19750828f, 0.5491314f,
            0.5148814f,   -0.9517943f,  -0.9077764f,  -0.9906229f,  0.4751265f,   0.81323147f,  0.6005076f,
            0.92147183f,  0.9878793f,   -0.8985506f,  0.28120112f,  0.97769725f,  0.95308435f,  0.9777889f,
            -0.9270168f,  -0.9459193f,  0.79248047f,  0.99223363f,  -0.1611281f,  0.11231542f,  -0.8230629f,
            -0.2566173f,  0.16398644f,  -0.36077273f, -0.70470357f, 0.8011706f,   -0.59996057f, -0.42161822f,
            -0.19564903f, -0.6991576f,  -0.12754434f, -0.26959598f, -0.8372509f,  0.15711153f,  0.3000977f,
            0.53072214f,  0.25092757f,  0.82264745f,  -0.72998637f, -0.13731742f, 0.17423475f,  0.43279397f,
            0.96632254f,  -0.2988227f,  -0.4705984f,  -0.74036705f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             0.9963336f,
                                             0.63758683f,
                                             -0.82404625f,
                                             -0.38524252f,
                                             0.9350034f,
                                             -0.9918621f,
                                             0.67038023f,
                                             0.9884596f,
                                             -0.32398474f,
                                             -0.15730727f,
                                             -0.9970634f,
                                             -0.831641f,
                                             -0.19750828f,
                                             0.5491314f,
                                             0.5148814f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 3);
}

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_rnn_reverse) {
    auto model = convert_model("rnn_reverse.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            0.9963336f,   0.63758683f,  -0.82404625f, -0.38524252f, 0.9350034f,   -0.9918621f,  0.67038023f,
            0.9884596f,   -0.32398474f, -0.15730727f, -0.9970634f,  -0.831641f,   -0.19750828f, 0.5491314f,
            0.5148814f,   -0.9517943f,  -0.9077764f,  -0.9906229f,  0.4751265f,   0.81323147f,  0.6005076f,
            0.92147183f,  0.9878793f,   -0.8985506f,  0.28120112f,  0.97769725f,  0.95308435f,  0.9777889f,
            -0.9270168f,  -0.9459193f,  0.79248047f,  0.99223363f,  -0.1611281f,  0.11231542f,  -0.8230629f,
            -0.2566173f,  0.16398644f,  -0.36077273f, -0.70470357f, 0.8011706f,   -0.59996057f, -0.42161822f,
            -0.19564903f, -0.6991576f,  -0.12754434f, -0.26959598f, -0.8372509f,  0.15711153f,  0.3000977f,
            0.53072214f,  0.25092757f,  0.82264745f,  -0.72998637f, -0.13731742f, 0.17423475f,  0.43279397f,
            0.96632254f,  -0.2988227f,  -0.4705984f,  -0.74036705f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             0.9963336f,
                                             0.63758683f,
                                             -0.82404625f,
                                             -0.38524252f,
                                             0.9350034f,
                                             -0.9918621f,
                                             0.67038023f,
                                             0.9884596f,
                                             -0.32398474f,
                                             -0.15730727f,
                                             -0.9970634f,
                                             -0.831641f,
                                             -0.19750828f,
                                             0.5491314f,
                                             0.5148814f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 3);
}

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_rnn_fwd_bias_initial_h_const) {
    auto model = convert_model("rnn_fwd_bias_initial_h_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            0.55277014f,  0.15672898f,  -0.25152922f, -0.63345766f, 0.99974346f,  0.94002223f, -0.97647303f,
            -0.9999884f,  0.9752002f,   0.97388494f,  0.9967754f,   0.96745205f,  0.7899921f,  0.92003024f,
            -0.43116868f, 0.11219919f,  0.895327f,    0.21749747f,  0.6617017f,   0.99962795f, 0.37670398f,
            0.7918401f,   -0.99966455f, 0.9961897f,   0.9995159f,   -0.84224236f, 0.92083716f, -0.99834263f,
            0.9435711f,   0.8485148f,   0.699257f,    0.9983405f,   -0.87222385f, 0.05191362f, 0.9878634f,
            0.75459063f,  0.8326433f,   -0.99705976f, 0.62511444f,  0.99979305f,  0.99925995f, 0.94032586f,
            -0.86841005f, -0.8692311f,  0.9974319f,   -0.37055743f, -0.54580235f, -0.8618355f, 0.6927968f,
            0.99997866f,  0.15482295f,  0.90996563f,  -0.9992051f,  0.784014f,    0.9999677f,  -0.30979204f,
            0.99138904f,  -0.10645419f, -0.18203181f, 0.9996245f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             -0.37055743f,
                                             -0.54580235f,
                                             -0.8618355f,
                                             0.6927968f,
                                             0.99997866f,
                                             0.15482295f,
                                             0.90996563f,
                                             -0.9992051f,
                                             0.784014f,
                                             0.9999677f,
                                             -0.30979204f,
                                             0.99138904f,
                                             -0.10645419f,
                                             -0.18203181f,
                                             0.9996245f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 5);
}

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_rnn_fwd_bias_initial_h) {
    auto model = convert_model("rnn_fwd_bias_initial_h.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);
    test_case.add_input<float>(in_B);
    test_case.add_input<float>(in_initial_h);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            0.55277014f,  0.15672898f,  -0.25152922f, -0.63345766f, 0.99974346f,  0.94002223f, -0.97647303f,
            -0.9999884f,  0.9752002f,   0.97388494f,  0.9967754f,   0.96745205f,  0.7899921f,  0.92003024f,
            -0.43116868f, 0.11219919f,  0.895327f,    0.21749747f,  0.6617017f,   0.99962795f, 0.37670398f,
            0.7918401f,   -0.99966455f, 0.9961897f,   0.9995159f,   -0.84224236f, 0.92083716f, -0.99834263f,
            0.9435711f,   0.8485148f,   0.699257f,    0.9983405f,   -0.87222385f, 0.05191362f, 0.9878634f,
            0.75459063f,  0.8326433f,   -0.99705976f, 0.62511444f,  0.99979305f,  0.99925995f, 0.94032586f,
            -0.86841005f, -0.8692311f,  0.9974319f,   -0.37055743f, -0.54580235f, -0.8618355f, 0.6927968f,
            0.99997866f,  0.15482295f,  0.90996563f,  -0.9992051f,  0.784014f,    0.9999677f,  -0.30979204f,
            0.99138904f,  -0.10645419f, -0.18203181f, 0.9996245f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             -0.37055743f,
                                             -0.54580235f,
                                             -0.8618355f,
                                             0.6927968f,
                                             0.99997866f,
                                             0.15482295f,
                                             0.90996563f,
                                             -0.9992051f,
                                             0.784014f,
                                             0.9999677f,
                                             -0.30979204f,
                                             0.99138904f,
                                             -0.10645419f,
                                             -0.18203181f,
                                             0.9996245f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 5);
}

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_rnn_bidirectional) {
    auto model = convert_model("rnn_bidirectional.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_bdir_W);
    test_case.add_input<float>(in_bdir_R);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 2, 3, 5},
        std::vector<float>{
            0.02254748f,  0.15776646f,  -0.8229023f,  0.19205809f,  0.76984656f,  -0.00603169f, -0.02861464f,
            0.04512155f,  -0.0011912f,  -0.02572936f, -0.13703543f, -0.49651444f, -0.78868157f, 0.3566854f,
            0.8758509f,   -0.9964788f,  -0.15236056f, 0.5478349f,   0.14500666f,  0.61871886f,  0.03722596f,
            -0.81331265f, 0.99774206f,  -0.888188f,   -0.5575663f,  -0.9284624f,  -0.5595875f,  0.9986867f,
            -0.18373811f, 0.8451735f,   -0.43823165f, -0.1904698f,  0.8320786f,   0.9830735f,   0.61861455f,
            0.19109797f,  0.6440699f,   0.00962079f,  -0.32752872f, -0.5050589f,  -0.23455954f, 0.9517933f,
            0.9050665f,   0.91091585f,  -0.77941567f, -0.9915407f,  -0.23976672f, 0.04337811f,  0.2958206f,
            -0.3979709f,  -0.9083327f,  -0.21814531f, 0.9981259f,   -0.8650538f,  -0.4886601f,  -0.8349008f,
            -0.7880142f,  0.99017143f,  -0.9816452f,  -0.93827677f, 0.16374564f,  0.98451114f,  -0.821692f,
            -0.6319715f,  -0.01324981f, 0.28117967f,  0.20685172f,  0.01166677f,  -0.5441829f,  -0.5463746f,
            -0.85301256f, 0.52109087f,  -0.8317892f,  -0.9676957f,  -0.30258918f, -0.9810498f,  -0.83153796f,
            -0.9676579f,  0.5483788f,   0.42533123f,  -0.9851954f,  -0.5354376f,  0.6905062f,   -0.46665573f,
            -0.851916f,   -0.9073148f,  0.16276085f,  0.9518349f,   -0.8635942f,  -0.92539954f, 0.33436012f,
            -0.988292f,   0.9238765f,   0.94239855f,  0.24151397f,  0.5482547f,   0.76547384f,  -0.81047577f,
            -0.6625802f,  -0.09694612f, 0.9948462f,   -0.6242633f,  -0.19065344f, -0.36072153f, -0.99407107f,
            0.94602585f,  0.55862486f,  0.2306763f,   0.22547626f,  0.37753606f,  -0.9951596f,  -0.74445903f,
            -0.6766813f,  0.32036817f,  0.33250773f,  -0.9957684f,  -0.7924f,     -0.40261805f, -0.34061068f,
            -0.55580306f,
        });
    // Y_h
    test_case.add_expected_output<float>(
        Shape{2, 3, 5},
        std::vector<float>{
            0.33436012f,  -0.988292f,   0.9238765f,   0.94239855f,  0.24151397f,  0.5482547f,
            0.76547384f,  -0.81047577f, -0.6625802f,  -0.09694612f, 0.9948462f,   -0.6242633f,
            -0.19065344f, -0.36072153f, -0.99407107f, -0.9964788f,  -0.15236056f, 0.5478349f,
            0.14500666f,  0.61871886f,  0.03722596f,  -0.81331265f, 0.99774206f,  -0.888188f,
            -0.5575663f,  -0.9284624f,  -0.5595875f,  0.9986867f,   -0.18373811f, 0.8451735f,
        });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 6);
}

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_rnn_bidirectional_const) {
    auto model = convert_model("rnn_bidirectional_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>(in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 2, 3, 5},
        std::vector<float>{
            0.02254748f,  0.15776646f,  -0.8229023f,  0.19205809f,  0.76984656f,  -0.00603169f, -0.02861464f,
            0.04512155f,  -0.0011912f,  -0.02572936f, -0.13703543f, -0.49651444f, -0.78868157f, 0.3566854f,
            0.8758509f,   -0.9964788f,  -0.15236056f, 0.5478349f,   0.14500666f,  0.61871886f,  0.03722596f,
            -0.81331265f, 0.99774206f,  -0.888188f,   -0.5575663f,  -0.9284624f,  -0.5595875f,  0.9986867f,
            -0.18373811f, 0.8451735f,   -0.43823165f, -0.1904698f,  0.8320786f,   0.9830735f,   0.61861455f,
            0.19109797f,  0.6440699f,   0.00962079f,  -0.32752872f, -0.5050589f,  -0.23455954f, 0.9517933f,
            0.9050665f,   0.91091585f,  -0.77941567f, -0.9915407f,  -0.23976672f, 0.04337811f,  0.2958206f,
            -0.3979709f,  -0.9083327f,  -0.21814531f, 0.9981259f,   -0.8650538f,  -0.4886601f,  -0.8349008f,
            -0.7880142f,  0.99017143f,  -0.9816452f,  -0.93827677f, 0.16374564f,  0.98451114f,  -0.821692f,
            -0.6319715f,  -0.01324981f, 0.28117967f,  0.20685172f,  0.01166677f,  -0.5441829f,  -0.5463746f,
            -0.85301256f, 0.52109087f,  -0.8317892f,  -0.9676957f,  -0.30258918f, -0.9810498f,  -0.83153796f,
            -0.9676579f,  0.5483788f,   0.42533123f,  -0.9851954f,  -0.5354376f,  0.6905062f,   -0.46665573f,
            -0.851916f,   -0.9073148f,  0.16276085f,  0.9518349f,   -0.8635942f,  -0.92539954f, 0.33436012f,
            -0.988292f,   0.9238765f,   0.94239855f,  0.24151397f,  0.5482547f,   0.76547384f,  -0.81047577f,
            -0.6625802f,  -0.09694612f, 0.9948462f,   -0.6242633f,  -0.19065344f, -0.36072153f, -0.99407107f,
            0.94602585f,  0.55862486f,  0.2306763f,   0.22547626f,  0.37753606f,  -0.9951596f,  -0.74445903f,
            -0.6766813f,  0.32036817f,  0.33250773f,  -0.9957684f,  -0.7924f,     -0.40261805f, -0.34061068f,
            -0.55580306f,
        });
    // Y_h
    test_case.add_expected_output<float>(
        Shape{2, 3, 5},
        std::vector<float>{
            0.33436012f,  -0.988292f,   0.9238765f,   0.94239855f,  0.24151397f,  0.5482547f,
            0.76547384f,  -0.81047577f, -0.6625802f,  -0.09694612f, 0.9948462f,   -0.6242633f,
            -0.19065344f, -0.36072153f, -0.99407107f, -0.9964788f,  -0.15236056f, 0.5478349f,
            0.14500666f,  0.61871886f,  0.03722596f,  -0.81331265f, 0.99774206f,  -0.888188f,
            -0.5575663f,  -0.9284624f,  -0.5595875f,  0.9986867f,   -0.18373811f, 0.8451735f,
        });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 6);
}

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_rnn_defaults_fwd_const_dynamic) {
    auto model = convert_model("dynamic_shapes/rnn_defaults_fwd_const_dynamic.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>(Shape{4, 3, 2}, in_X);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            0.02254748f,  0.15776646f, -0.8229023f,  0.19205809f,  0.76984656f,  -0.00603169f, -0.02861464f,
            0.04512155f,  -0.0011912f, -0.02572936f, -0.13703543f, -0.49651444f, -0.78868157f, 0.3566854f,
            0.8758509f,   0.20788848f, 0.13481987f,  -0.756822f,   -0.121436f,   0.97542346f,  0.16959739f,
            0.63496053f,  0.1245538f,  -0.1970138f,  -0.56581646f, 0.8225869f,   0.9611373f,   -0.42990375f,
            -0.22925597f, 0.2226491f,  0.08246052f,  0.9798831f,   -0.13415998f, -0.5567714f,  0.78594816f,
            -0.34759718f, 0.11376679f, -0.07107389f, -0.5420871f,  -0.58504283f, -0.96065646f, 0.18588805f,
            -0.4870671f,  -0.1475982f, 0.82456505f,  -0.80264574f, -0.46370947f, 0.9719335f,   -0.7374159f,
            0.94937694f,  0.8814341f,  0.67015004f,  0.21958017f,  -0.8332769f,  -0.487742f,   0.9918536f,
            0.99563396f,  0.94866276f, -0.98504806f, -0.42824882f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             -0.80264574f,
                                             -0.46370947f,
                                             0.9719335f,
                                             -0.7374159f,
                                             0.94937694f,
                                             0.8814341f,
                                             0.67015004f,
                                             0.21958017f,
                                             -0.8332769f,
                                             -0.487742f,
                                             0.9918536f,
                                             0.99563396f,
                                             0.94866276f,
                                             -0.98504806f,
                                             -0.42824882f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 4);
}

OPENVINO_TEST_F(${BACKEND_NAME}, RNNSequenceOp, onnx_model_import_only_rnn_defaults_fwd_const_dynamic) {
    auto model = convert_model("dynamic_shapes/rnn_defaults_fwd_const_dynamic.onnx");

    auto batch_size = Dimension::dynamic();
    auto seq_length = Dimension::dynamic();
    int64_t hidden_size = 5;
    int64_t num_directions = 1;
    auto Y_expected_output = PartialShape{batch_size, num_directions, seq_length, hidden_size};
    auto Y_h_expected_output = PartialShape{num_directions, batch_size, hidden_size};

    EXPECT_EQ(model->get_output_size(), 2);
    EXPECT_EQ(model->get_output_partial_shape(0), Y_expected_output);
    EXPECT_EQ(model->get_output_partial_shape(1), Y_h_expected_output);

    EXPECT_EQ(count_ops_of_type<op::v5::RNNSequence>(model), 1);
}
