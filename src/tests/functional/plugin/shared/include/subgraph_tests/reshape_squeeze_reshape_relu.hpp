// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/reshape_squeeze_reshape_relu.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ReshapeSqueezeReshapeRelu, CompareWithRefs){
    Run();
};

} // namespace SubgraphTestsDefinitions
