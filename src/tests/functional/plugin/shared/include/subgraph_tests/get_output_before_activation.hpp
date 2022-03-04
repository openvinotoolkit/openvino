// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/get_output_before_activation.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(OutputBeforeActivation, CompareWithRefs) {
    Run();
};

} // namespace SubgraphTestsDefinitions
