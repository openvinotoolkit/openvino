// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "gna_matcher.hpp"
#include "matchers/input_data_matcher.hpp"
#include "test_irs.hpp"

using namespace std;
using namespace InferenceEngine;
using namespace ::testing;
using namespace GNATestIRs;

class GNAInputPrecisionTest : public GNATest<> {
};

TEST_F(GNAInputPrecisionTest, CanProcessU8Input) {
    std::vector<float> input_init = {128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
    double scale = 1.f / 128;
    std::vector<int16_t> input_processed = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    assert_that().onInferModel(Fc2DOutputModel())
            .inNotCompactMode().gna().propagate_forward().called_with()
            .preprocessed_input_data(input_init, input_processed, Precision::U8)
            .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), scale);
}

TEST_F(GNAInputPrecisionTest, CanProcessFP32Input) {
    std::vector<float> input_init = {1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280};
    double scale = 1.f / 1280;
    std::vector<int16_t> input_processed = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    assert_that().onInferModel(Fc2DOutputModel())
            .inNotCompactMode().gna().propagate_forward().called_with()
            .preprocessed_input_data(input_init, input_processed, Precision::FP32)
            .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), scale);
}
