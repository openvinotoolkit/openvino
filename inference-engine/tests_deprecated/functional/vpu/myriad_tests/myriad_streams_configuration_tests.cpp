// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_layers_tests.hpp"

#include <gtest/gtest.h>
#include <ngraph_functions/subgraph_builders.hpp>

namespace {

class myriadCorrectStreamsConfiguration_nightly : public vpuLayersTests, public testing::WithParamInterface<std::uint32_t> {};
TEST_P(myriadCorrectStreamsConfiguration_nightly, InfersWithConfiguredStreams) {
    _config[InferenceEngine::MYRIAD_THROUGHPUT_STREAMS] = std::to_string(GetParam());
    _irVersion = IRVersion::v10;

    auto fn_ptr = ngraph::builder::subgraph::makeSplitMultiConvConcat();
    ASSERT_NO_THROW(_cnnNetwork = InferenceEngine::CNNNetwork(fn_ptr));
    ASSERT_NO_THROW(_inputsInfo = _cnnNetwork.getInputsInfo());
    ASSERT_NO_THROW(_outputsInfo = _cnnNetwork.getOutputsInfo());

    createInferRequest(NetworkInitParams{}.useHWOpt(true));

    ASSERT_TRUE(Infer());
}

INSTANTIATE_TEST_SUITE_P(StreamsConfiguration, myriadCorrectStreamsConfiguration_nightly, testing::Values(1, 2, 3));

}
