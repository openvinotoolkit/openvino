// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/broadcast_power.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(BroadcastPowerTest, CompareWithRefImpl) {
    Run();
};

}  // namespace SubgraphTestsDefinitions
