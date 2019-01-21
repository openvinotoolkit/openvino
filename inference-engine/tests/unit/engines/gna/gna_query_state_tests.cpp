// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include "gna_matcher.hpp"

class QueryStateTest : public GNATest {
 protected:
    void SetUp() override  {
    }
};
using namespace GNATestIRs;

// Recursive Algorithm
// Precision Threshold

TEST_F(QueryStateTest, returnEmptyCollectionOfStatesIfNoMemoryInIR) {
    assert_that().afterLoadingModel(TanhActivationModel()).queryState().isEmpty();
}

TEST_F(QueryStateTest, returnNonEmptyCollectionOfStatesForMemoryIR) {
    assert_that().afterLoadingModel(affineToMemoryModel()).queryState().isNotEmpty();
}
