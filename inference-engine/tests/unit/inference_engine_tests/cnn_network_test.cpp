// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "cpp/ie_cnn_net_reader.h"
#include "cpp/ie_cnn_network.h"

using namespace InferenceEngine;

class CNNNetworkTests : public ::testing::Test {
protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
    }

public:

};

TEST_F(CNNNetworkTests, throwsOnInitWithNull) {
    std::shared_ptr<ICNNNetwork> nlptr = nullptr;
    ASSERT_THROW(CNNNetwork network(nlptr), InferenceEngine::details::InferenceEngineException);
}
