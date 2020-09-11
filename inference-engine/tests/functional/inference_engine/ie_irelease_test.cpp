// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include <gtest/gtest.h>
#include <details/ie_irelease.hpp>

#include "common_test_utils/test_common.hpp"

using IReleaseTests = CommonTestUtils::TestsCommon;

/**
 * @brief Testing that callback with Release() from  shared_from_irelease(...)
 * won't be applied for nullptr.
 */
TEST_F(IReleaseTests, sharedFromIReleaseWithNull) {
    InferenceEngine::details::IRelease *irelease = nullptr;
    std::shared_ptr<InferenceEngine::details::IRelease> ptr = InferenceEngine::details::shared_from_irelease(irelease);
    ptr.reset();
}