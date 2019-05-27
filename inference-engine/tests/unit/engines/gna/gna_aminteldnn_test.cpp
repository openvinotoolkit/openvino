// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "gna_matcher.hpp"
#include "inference_engine.hpp"
#include "dnn.h"

using namespace testing;
using namespace InferenceEngine;

class GNA_AmIntelDnn_test : public GNATest {
protected:
    AmIntelDnn amIntelDnn;
    intel_nnet_type_t  desc = {};
};

TEST_F(GNA_AmIntelDnn_test, intel_nnet_type_tDoesNotFreeHisMemory) {
    desc.pLayers = nullptr;
    amIntelDnn.component.resize(1);
    amIntelDnn.component[0].operation = kDnnAffineOp;
    ASSERT_NO_THROW(amIntelDnn.InitGNAStruct(&desc));  // thirst init is ok
    ASSERT_THROW(amIntelDnn.InitGNAStruct(&desc), InferenceEngine::details::InferenceEngineException);  // second init involves memory leak
}

TEST_F(GNA_AmIntelDnn_test, intel_nnet_type_t_ptrIsNullptr) {
    ASSERT_THROW(amIntelDnn.InitGNAStruct(nullptr), InferenceEngine::details::InferenceEngineException);
}

TEST_F(GNA_AmIntelDnn_test, intel_nnet_type_t_pLayersIsNotNullptr) {
    ASSERT_THROW(amIntelDnn.InitGNAStruct(&desc), InferenceEngine::details::InferenceEngineException);
}

TEST_F(GNA_AmIntelDnn_test, ComponentIsEmpty) {
    desc.pLayers = nullptr;
    ASSERT_THROW(amIntelDnn.InitGNAStruct(&desc), InferenceEngine::details::InferenceEngineException);
}