// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <inference_engine.hpp>
#include <backend/am_intel_dnn.hpp>

#include "gna_matcher.hpp"

using namespace testing;

class GNA_AmIntelDnn_test : public GNATest<> {
protected:
    GNAPluginNS::backend::AMIntelDNN amIntelDnn;
#if GNA_LIB_VER == 2
    Gna2Model desc = {};
#else
    intel_nnet_type_t  desc = {};
#endif
};

TEST_F(GNA_AmIntelDnn_test, intel_nnet_type_tSecondInitNotAllowed) {
#if GNA_LIB_VER == 2
    desc.Operations = nullptr;
#else
    desc.pLayers = nullptr;
#endif
    amIntelDnn.component.resize(1);
    amIntelDnn.component[0].operation = kDnnAffineOp;
    // First init is ok
    ASSERT_NO_THROW(amIntelDnn.InitGNAStruct(&desc));
    // Second init would cause memory leak, so it's prohibited
    ASSERT_THROW(amIntelDnn.InitGNAStruct(&desc), InferenceEngine::details::InferenceEngineException);
    amIntelDnn.DestroyGNAStruct(&desc);
}

TEST_F(GNA_AmIntelDnn_test, intel_nnet_type_t_ptrIsNullptr) {
    ASSERT_THROW(amIntelDnn.InitGNAStruct(nullptr), InferenceEngine::details::InferenceEngineException);
}

TEST_F(GNA_AmIntelDnn_test, intel_nnet_type_t_pLayersIsNotNullptr) {
    ASSERT_THROW(amIntelDnn.InitGNAStruct(&desc), InferenceEngine::details::InferenceEngineException);
}

TEST_F(GNA_AmIntelDnn_test, ComponentIsEmpty) {
#if GNA_LIB_VER == 2
    desc.Operations = nullptr;
#else
    desc.pLayers = nullptr;
#endif
    ASSERT_THROW(amIntelDnn.InitGNAStruct(&desc), InferenceEngine::details::InferenceEngineException);
}
