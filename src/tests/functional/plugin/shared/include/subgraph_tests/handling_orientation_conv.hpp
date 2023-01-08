// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/handling_orientation_conv.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(HandlingOrientationClass, CompareWithRefs){
    Run();
};

}  // namespace SubgraphTestsDefinitions
