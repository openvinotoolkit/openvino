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

#include <gtest/gtest.h>
#include <gmock/gmock-generated-actions.h>
#include "gna_matcher.hpp"
#include "matchers/input_data_matcher.hpp"
#include "test_irs.hpp"

using namespace std;
using namespace InferenceEngine;
using namespace ::testing;
using namespace GNATestIRs;

class GNAInputPrecisionTest : public GNATest {
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
