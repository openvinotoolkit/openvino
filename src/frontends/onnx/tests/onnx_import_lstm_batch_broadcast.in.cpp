// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Tests for LSTM batch broadcasting feature (CVS-162986)
//
// These tests verify that LSTM correctly handles the case where initial_h/initial_c
// have batch_size=1 but input X has batch_size>1. The implementation should broadcast
// the initial states using the Tile operation.

#include <cmath>
#include <vector>

#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");
static std::string s_device = backend_name_to_device("${BACKEND_NAME}");

// Minimal test: batch=2, hidden=2
// Tests basic batch broadcasting functionality
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lstm_batch_broadcast_minimal) {
    auto model = convert_model("lstm_batch_broadcast_minimal.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // X: [seq_length=1, batch_size=2, input_size=2]
    // Use diverse values instead of constant 1.0f (as per PR #32608 guidelines)
    test_case.add_input<float>({-0.5f, 0.3f, 0.8f, -0.2f});

    // initial_h: [num_directions=1, batch_size=1, hidden_size=2]
    // This will be broadcast to batch_size=2 via Tile operation
    test_case.add_input<float>({0.1f, -0.1f});

    // initial_c: [num_directions=1, batch_size=1, hidden_size=2]
    test_case.add_input<float>({0.2f, -0.2f});

    // Use tolerance for FP accumulated errors with diverse data
    test_case.run_with_tolerance_as_fp(1e-4f);
}

// Large batch test: batch=128, like silero_vad model
// Tests that broadcasting works correctly with large batches
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lstm_batch_broadcast_large_batch) {
    auto model = convert_model("lstm_batch_broadcast_large_batch.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // X: [seq_length=2, batch_size=128, input_size=4]
    std::vector<float> X_data(2 * 128 * 4);
    // Fill with diverse values (sin wave pattern for variation)
    for (size_t i = 0; i < X_data.size(); ++i) {
        X_data[i] = static_cast<float>(std::sin(i * 0.1) * 2.0);
    }
    test_case.add_input<float>(X_data);

    // initial_h: [num_directions=1, batch_size=1, hidden_size=128]
    std::vector<float> h_data(128);
    for (size_t i = 0; i < h_data.size(); ++i) {
        h_data[i] = static_cast<float>(std::cos(i * 0.05));
    }
    test_case.add_input<float>(h_data);

    // initial_c: [num_directions=1, batch_size=1, hidden_size=128]
    std::vector<float> c_data(128);
    for (size_t i = 0; i < c_data.size(); ++i) {
        c_data[i] = static_cast<float>(std::sin(i * 0.03) * 0.5);
    }
    test_case.add_input<float>(c_data);

    // Higher tolerance for large batch with diverse data
    test_case.run_with_tolerance_as_fp(1e-4f);
}

// Standard sizes: batch=4, seq=3, input=8, hidden=16
// Tests with typical network dimensions
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lstm_batch_broadcast_standard) {
    auto model = convert_model("lstm_batch_broadcast_standard.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // X: [seq_length=3, batch_size=4, input_size=8]
    std::vector<float> X_data(3 * 4 * 8);
    for (size_t i = 0; i < X_data.size(); ++i) {
        X_data[i] = static_cast<float>((i % 17) / 10.0 - 0.8);  // Diverse values
    }
    test_case.add_input<float>(X_data);

    // initial_h: [num_directions=1, batch_size=1, hidden_size=16]
    std::vector<float> h_data(16);
    for (size_t i = 0; i < h_data.size(); ++i) {
        h_data[i] = static_cast<float>((i % 7) / 20.0);
    }
    test_case.add_input<float>(h_data);

    // initial_c: [num_directions=1, batch_size=1, hidden_size=16]
    std::vector<float> c_data(16);
    for (size_t i = 0; i < c_data.size(); ++i) {
        c_data[i] = static_cast<float>((i % 5) / 15.0 - 0.1);
    }
    test_case.add_input<float>(c_data);

    test_case.run_with_tolerance_as_fp(1e-4f);
}

// Corner case: batch_size=1 in both X and initial states
// Should work without broadcasting (no Tile needed)
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lstm_batch_broadcast_no_broadcast_needed) {
    // This test uses the standard lstm model where batch matches
    auto model = convert_model("lstm_fwd_default_const.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0.68172926f, 1.1405563f, -0.03931177f, -0.03759607f});

    test_case.add_expected_output<float>(Shape{2, 1, 1, 2}, {-0.063373f, -0.20347191f, -0.07230289f, -0.13298286f});
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {-0.07230289f, -0.13298286f});
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {-0.1557954f, -0.24502525f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}
