// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "gna_api_wrapper.hpp"

using namespace testing;
using namespace InferenceEngine;

class GNA_CPPWrapper_test : public ::testing::Test {};

TEST_F(GNA_CPPWrapper_test, CPPWrapperConstructorCannotWorkWithInputEqualToZero) {
#if GNA_LIB_VER == 2
    ASSERT_THROW(GNAPluginNS::CPPWrapper<Gna2Model>(0), InferenceEngine::Exception);
#else
    ASSERT_THROW(GNAPluginNS::CPPWrapper<intel_nnet_type_t>(0), InferenceEngine::Exception);
#endif
}

TEST_F(GNA_CPPWrapper_test, CPPWrapperConstructorCanWorkWithInputNotEqualToZero) {
#if GNA_LIB_VER == 2
    ASSERT_NO_THROW(GNAPluginNS::CPPWrapper<Gna2Model>(3));
    #else
    ASSERT_NO_THROW(GNAPluginNS::CPPWrapper<intel_nnet_type_t>(3));
#endif
}

TEST_F(GNA_CPPWrapper_test, CPPWrapperConstructorCanWorkWithoutAnyInput) {
#if GNA_LIB_VER == 2
    ASSERT_NO_THROW(GNAPluginNS::CPPWrapper<Gna2Model>());
#else
    ASSERT_NO_THROW(GNAPluginNS::CPPWrapper<intel_nnet_type_t>());
#endif
}

