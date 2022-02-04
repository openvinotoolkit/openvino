// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/conv_strides_opt.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ConvStridesOpt, CompareWithRefs) {
    Run();
}
} // namespace SubgraphTestsDefinitions
