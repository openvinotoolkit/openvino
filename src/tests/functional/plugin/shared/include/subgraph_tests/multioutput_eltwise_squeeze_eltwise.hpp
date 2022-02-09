// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/multioutput_eltwise_squeeze_eltwise.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MultioutputEltwiseReshapeEltwise, CompareWithRefs){
    Run();
};

} // namespace SubgraphTestsDefinitions
