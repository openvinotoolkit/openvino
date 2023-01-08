// Copyright (C) 2018-2022 Intel Corporation
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

TEST_P(ConcatMultiInput, CompareWithRefMemory) {
    GenerateMemoryModel();
    LoadNetwork();
    GenerateInputs();
    Infer();
};

}  // namespace SubgraphTestsDefinitions
