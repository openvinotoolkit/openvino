// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/two_fake_quantize_to_fullyconnected.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(FakeQuantizeSubgraphTest, CompareWithRefs) {
    Run();
}

}  // namespace SubgraphTestsDefinitions
