// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <request/worker_pool_impl.hpp>

//using namespace testing;
//using namespace InferenceEngine;

// TODO implement real test with differnt file name.
 class GNA_Request_test : public ::testing::Test {};

 TEST_F(GNA_Request_test, RequestConstructor) {
     std::cout << "cpp: " << __cplusplus << std::endl;
     ASSERT_NO_THROW(GNAPluginNS::request::WorkerPoolImpl());
 }


//class GNA_CPPWrapper_test : public ::testing::Test {};
//
//TEST_F(GNA_CPPWrapper_test, CPPWrapperConstructorCannotWorkWithInputEqualToZero) {
//    ASSERT_THROW(GNAPluginNS::CPPWrapper<Gna2Model>(0), InferenceEngine::Exception);
//}
//
//TEST_F(GNA_CPPWrapper_test, CPPWrapperConstructorCanWorkWithInputNotEqualToZero) {
//    ASSERT_NO_THROW(GNAPluginNS::CPPWrapper<Gna2Model>(3));
//}
//
//TEST_F(GNA_CPPWrapper_test, CPPWrapperConstructorCanWorkWithoutAnyInput) {
//    ASSERT_NO_THROW(GNAPluginNS::CPPWrapper<Gna2Model>());
//}
