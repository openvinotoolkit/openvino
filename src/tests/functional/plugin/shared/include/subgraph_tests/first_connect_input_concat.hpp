// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <shared_test_classes/subgraph/first_connect_input_concat.hpp>

namespace SubgraphTestsDefinitions {

TEST_P(ConcatFirstInputTest, CompareWithRefImpl) {
    Run();
};

}  // namespace SubgraphTestsDefinitions
