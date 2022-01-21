// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/cancellation.hpp"
#include "api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestCancellationTests,
        ::testing::Combine(
            ::testing::Values(ConformanceTests::targetDevice),
            ::testing::ValuesIn(emptyConfig)),
        OVInferRequestCancellationTests::getTestCaseName);
}  // namespace
