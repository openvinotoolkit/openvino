// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/variadic_split_pad.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(VariadicSplitPad, CompareWithRefs){
    Run();
};

}  // namespace SubgraphTestsDefinitions
