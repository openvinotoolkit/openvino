// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <gtest/gtest.h>
#include "gna_matcher.hpp"

using namespace std;
using namespace InferenceEngine;
using namespace ::testing;

class GNAHWPrecisionTest : public GNATest<> {

};

TEST_F(GNAHWPrecisionTest, defaultPrecisionIsInt16) {
    assert_that().onInfer1AFModel().gna().propagate_forward().called_with().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).
        nnet_input_precision(Precision::I16).
        nnet_ouput_precision(Precision::I32).
        nnet_weights_precision(Precision::I16).
        nnet_biases_precision(Precision::I32);
}

TEST_F(GNAHWPrecisionTest, canPassInt8Precision) {
    assert_that().onInfer1AFModel().withConfig(PRECISION, Precision::I8).
        gna().propagate_forward().called_with().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).
            nnet_input_precision(Precision::I16).
            nnet_ouput_precision(Precision::I32).
            nnet_weights_precision(Precision::I8).
            nnet_biases_precision(Precision::fromType<gna_compound_bias_t>());
}

TEST_F(GNAHWPrecisionTest, canPassInt16Precision) {
    assert_that().onInfer1AFModel().withConfig(PRECISION, Precision::I16).withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna().propagate_forward().called_with().
        nnet_input_precision(Precision::I16).
        nnet_ouput_precision(Precision::I32).
        nnet_weights_precision(Precision::I16).
        nnet_biases_precision(Precision::I32);
}

TEST_F(GNAHWPrecisionTest, failToCreatePluginWithUnsuportedPrecision) {
    assert_that().creating().gna_plugin().withConfig(PRECISION, Precision::FP32).withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).throws();
}