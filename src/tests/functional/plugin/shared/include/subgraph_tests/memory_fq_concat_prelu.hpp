// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef PLUGIN_SHARED_MEMORY_FQ_CONCAT_PRELU_HPP
#define PLUGIN_SHARED_MEMORY_FQ_CONCAT_PRELU_HPP

#include "shared_test_classes/subgraph/memory_fq_concat_prelu.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MemoryFqConcatPrelu, CompareWithRefs){
    Run();
};

}  // namespace SubgraphTestsDefinitions

#endif // PLUGIN_SHARED_MEMORY_FQ_CONCAT_PRELU_HPP
