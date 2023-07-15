// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "matchers/subgraph/repeat_pattern.hpp"
#include "utils/model.hpp"

#include "test_models.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;


// ======================= ExtractorsManagerTest Unit tests =======================
class RepeatPatternExtractorTest : public RepeatPatternExtractor,
                                   public ::testing::Test {};

TEST_F(RepeatPatternExtractorTest, extract) {
    auto models = this->extract(generate_abs_relu_add());
    auto a = 0;
}

}  // namespace
