// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/stridedslice_concat.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(SliceConcatTest, CompareWithRefImpl) {
    Run();
};

}  // namespace SubgraphTestsDefinitions