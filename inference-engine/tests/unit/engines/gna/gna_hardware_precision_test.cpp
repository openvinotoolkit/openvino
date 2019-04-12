//
// Copyright 2016-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <vector>

#include <gtest/gtest.h>
#include <mock_icnn_network.hpp>
#include <cpp/ie_cnn_net_reader.h>
#include <gmock/gmock-generated-actions.h>
#include "gna_matcher.hpp"

using namespace std;
using namespace InferenceEngine;
using namespace ::testing;

class GNAHWPrecisionTest : public GNATest {

};

TEST_F(GNAHWPrecisionTest, defaultPrecisionIsInt16) {
    assert_that().onInfer1AFModel().gna().propagate_forward().called_with().
        nnet_input_precision(Precision::I16).
        nnet_ouput_precision(Precision::I32).
        nnet_weights_precision(Precision::I16).
        nnet_biases_precision(Precision::I32);
}

TEST_F(GNAHWPrecisionTest, canPassInt8Precision) {
    assert_that().onInfer1AFModel().withConfig(PRECISION, Precision::I8).
        gna().propagate_forward().called_with().
            nnet_input_precision(Precision::I16).
            nnet_ouput_precision(Precision::I32).
            nnet_weights_precision(Precision::I8).
            nnet_biases_precision(Precision::fromType<intel_compound_bias_t>());
}

TEST_F(GNAHWPrecisionTest, canPassInt16Precision) {
    assert_that().onInfer1AFModel().withConfig(PRECISION, Precision::I16).
        gna().propagate_forward().called_with().
        nnet_input_precision(Precision::I16).
        nnet_ouput_precision(Precision::I32).
        nnet_weights_precision(Precision::I16).
        nnet_biases_precision(Precision::I32);
}

TEST_F(GNAHWPrecisionTest, failToCreatePluginWithUnsuportedPrecision) {
    assert_that().creating().gna_plugin().withConfig(PRECISION, Precision::FP32).throws();
}