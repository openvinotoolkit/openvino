// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/multi_input_scale.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MultipleInputScaleTest, CompareWithRefs) {
    Run();
};

} // namespace SubgraphTestsDefinitions
