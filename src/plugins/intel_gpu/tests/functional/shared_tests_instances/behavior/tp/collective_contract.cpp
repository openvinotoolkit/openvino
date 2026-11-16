// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/tp/collective_contract.hpp"

#include "common_test_utils/test_constants.hpp"

namespace {

using ov::test::behavior::TPCollectiveContractTest;

INSTANTIATE_TEST_SUITE_P(smoke_TP,
                         TPCollectiveContractTest,
                         ::testing::Combine(::testing::Values(std::string(ov::test::utils::DEVICE_GPU)),
                                            ::testing::Values(2u, 3u, 4u)),
                         TPCollectiveContractTest::get_test_case_name);

}  // namespace
