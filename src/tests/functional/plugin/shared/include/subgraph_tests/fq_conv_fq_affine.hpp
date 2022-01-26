// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/fq_conv_fq_affine.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(FqConvFqAffineTest, CompareWithRefs) {
    Run();
}

}  // namespace SubgraphTestsDefinitions
