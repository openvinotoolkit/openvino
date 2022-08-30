// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/multi_crops_to_concat.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MultiCropsToConcatTest, CompareWithRefs) {
    Run();
};

} // namespace SubgraphTestsDefinitions
