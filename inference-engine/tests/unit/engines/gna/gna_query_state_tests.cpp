// Copyright (C) 2018-2019 Intel Corporation
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
    assert_that().afterLoadingModel(TanhActivationModel()).withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).queryState().isEmpty();
}

TEST_F(QueryStateTest, returnNonEmptyCollectionOfStatesForMemoryIR) {
    assert_that().afterLoadingModel(affineToMemoryModel()).withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).queryState().isNotEmpty();
}
