// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/perm_conv_perm_concat.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(PermConvPermConcat, CompareWithRefs) {
    Run();
}

}  // namespace SubgraphTestsDefinitions
