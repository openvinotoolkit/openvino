// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/concat_multi_input.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ConcatMultiInput, CompareWithRefStridedSlice) {
    GenerateStridedSliceModel();
    Run();
};

TEST_P(ConcatMultiInput, CompareWithRefConstOnly) {
    GenerateConstOnlyModel();
    Run();
};

}  // namespace SubgraphTestsDefinitions
