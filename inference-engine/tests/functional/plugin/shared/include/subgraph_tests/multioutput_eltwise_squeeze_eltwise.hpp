// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "shared_test_classes/subgraph/multioutput_eltwise_squeeze_eltwise.hpp"

namespace LayerTestsDefinitions {

TEST_P(MultioutputEltwiseReshapeEltwise, CompareWithRefs){
    Run();
};

} // namespace LayerTestsDefinitions
