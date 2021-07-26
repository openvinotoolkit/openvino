// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef PLUGIN_SHARED_MEMORY_CONCAT_HPP
#define PLUGIN_SHARED_MEMORY_CONCAT_HPP

#include "shared_test_classes/subgraph/memory_concat.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MemoryConcat, CompareWithRefs){
    Run();
};

}  // namespace SubgraphTestsDefinitions

#endif // PLUGIN_SHARED_MEMORY_CONCAT_HPP
