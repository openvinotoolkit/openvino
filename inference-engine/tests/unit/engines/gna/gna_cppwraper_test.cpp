// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _WIN32
#include <mm_malloc.h>
#endif
#include "gna_api_wrapper.hpp"
#include <gtest/gtest.h>

using namespace testing;
using namespace InferenceEngine;

class GNA_CPPWrapper_test : public ::testing::Test {};

TEST_F(GNA_CPPWrapper_test, CPPWrapperConstructorCannotWorkWithInputEqualToZero) {
    ASSERT_THROW(GNAPluginNS::CPPWrapper<intel_nnet_type_t>(0), InferenceEngine::details::InferenceEngineException);
}

TEST_F(GNA_CPPWrapper_test, CPPWrapperConstructorCanWorkWithInputNotEqualToZero) {
    ASSERT_NO_THROW(GNAPluginNS::CPPWrapper<intel_nnet_type_t>(3));
}

TEST_F(GNA_CPPWrapper_test, CPPWrapperConstructorCanWorkWithoutAnyInput) {
    ASSERT_NO_THROW(GNAPluginNS::CPPWrapper<intel_nnet_type_t>());
}

