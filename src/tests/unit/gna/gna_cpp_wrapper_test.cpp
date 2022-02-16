// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "gna_api_wrapper.hpp"

using namespace testing;
using namespace InferenceEngine;

class GNA_CPPWrapper_test : public ::testing::Test {};

TEST_F(GNA_CPPWrapper_test, CPPWrapperConstructorCannotWorkWithInputEqualToZero) {
    ASSERT_THROW(GNAPluginNS::CPPWrapper<Gna2Model>(0), InferenceEngine::Exception);
}

TEST_F(GNA_CPPWrapper_test, CPPWrapperConstructorCanWorkWithInputNotEqualToZero) {
    ASSERT_NO_THROW(GNAPluginNS::CPPWrapper<Gna2Model>(3));
}

TEST_F(GNA_CPPWrapper_test, CPPWrapperConstructorCanWorkWithoutAnyInput) {
    ASSERT_NO_THROW(GNAPluginNS::CPPWrapper<Gna2Model>());
}
