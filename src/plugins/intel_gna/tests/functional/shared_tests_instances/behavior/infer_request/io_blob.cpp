// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/io_blob.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferRequestIOBBlobTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         InferRequestIOBBlobTest::getTestCaseName);

}  // namespace
