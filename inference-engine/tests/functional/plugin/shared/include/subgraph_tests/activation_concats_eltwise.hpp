// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/activation_concats_eltwise.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ActivationConcatsEltwise, CompareWithRefs) {
    Run();
}

}  // namespace SubgraphTestsDefinitions
