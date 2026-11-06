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
                                            ::testing::Values(size_t{2}, size_t{3}, size_t{4})),
                         TPCollectiveContractTest::get_test_case_name);

}  // namespace
