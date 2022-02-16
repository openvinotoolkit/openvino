// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine.hpp>
#include <backend/am_intel_dnn.hpp>

using namespace testing;

class GNA_AmIntelDnn_test : public ::testing::Test {
protected:
    GNAPluginNS::backend::AMIntelDNN amIntelDnn;
    Gna2Model desc = {};
};


TEST_F(GNA_AmIntelDnn_test, intel_nnet_type_tSecondInitNotAllowed) {
    desc.Operations = nullptr;
    amIntelDnn.component.resize(1);
    amIntelDnn.component[0].operation = kDnnAffineOp;
    // First init is ok
    ASSERT_NO_THROW(amIntelDnn.InitGNAStruct(&desc));
    // Second init would cause memory leak, so it's prohibited
    ASSERT_THROW(amIntelDnn.InitGNAStruct(&desc), InferenceEngine::Exception);
    amIntelDnn.DestroyGNAStruct(&desc);
}

TEST_F(GNA_AmIntelDnn_test, intel_nnet_type_t_ptrIsNullptr) {
    ASSERT_THROW(amIntelDnn.InitGNAStruct(nullptr), InferenceEngine::Exception);
}

TEST_F(GNA_AmIntelDnn_test, intel_nnet_type_t_pLayersIsNotNullptr) {
    ASSERT_THROW(amIntelDnn.InitGNAStruct(&desc), InferenceEngine::Exception);
}

TEST_F(GNA_AmIntelDnn_test, ComponentIsEmpty) {
    desc.Operations = nullptr;
    ASSERT_THROW(amIntelDnn.InitGNAStruct(&desc), InferenceEngine::Exception);
}
