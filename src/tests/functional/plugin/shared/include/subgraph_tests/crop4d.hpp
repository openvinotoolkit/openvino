// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/crop4d.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(Crop4dTest, CompareWithRefs){
    Run();
};
}  // namespace SubgraphTestsDefinitions
