// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/broadcast_eltwise_eliminated.hpp"

using namespace ov::test;

namespace {
INSTANTIATE_TEST_SUITE_P(smoke_BroadcastEltwise, BroadcastEltwiseEliminated,
                         ::testing::Values(ov::test::utils::DEVICE_GPU),
                         BroadcastEltwiseEliminated::getTestCaseName);

}  // namespace
