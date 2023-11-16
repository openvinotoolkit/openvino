// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "shared_test_classes/single_layer/multinomial.hpp"

namespace ov {
namespace test {
namespace subgraph {

TEST_P(MultinomialTest, CompareWithRefs) {
    run();
}

TEST_P(MultinomialTest, CompareQueryModel) {
    query_model();
}

}  // namespace subgraph
}  // namespace test
}  // namespace ov
