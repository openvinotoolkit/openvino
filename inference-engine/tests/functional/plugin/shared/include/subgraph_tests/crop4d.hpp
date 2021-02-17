// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "shared_test_classes/subgraph/crop4d.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(Crop4dTest, CompareWithRefs){
    Run();
};
}  // namespace SubgraphTestsDefinitions
