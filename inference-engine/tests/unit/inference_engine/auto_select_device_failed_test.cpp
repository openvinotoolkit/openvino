// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include <ie_core.hpp>
#include <ngraph_functions/subgraph_builders.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

class AutoLoadFailedTest: public ::testing::Test {
protected:
    std::shared_ptr<ngraph::Function> function;
    InferenceEngine::CNNNetwork cnnNet;
    std::shared_ptr<MockICore> core;

    void SetUp() override {
        core  = std::shared_ptr<MockICore>(new MockICore());
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        cnnNet = InferenceEngine::CNNNetwork(function);
    }
};

TEST_F(AutoLoadFailedTest, canContinueIfGpuFailed) {
    ON_CALL(core, LoadNetwork(MatcherCast<const CNNNetwork&>(_), "GPU").WillByDefault(Throw(InferenceEngine::GeneralError{""}));
    ASSERT_NO_THROW(core->LoadNetwork(cnnNet, "AUTO:CPU,GPU"));
}


