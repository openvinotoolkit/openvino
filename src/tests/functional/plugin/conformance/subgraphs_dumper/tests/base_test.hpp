// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gtest/gtest.h"
#include "matchers/subgraph/subgraph.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "common_test_utils/test_assertions.hpp"

using namespace ov::tools::subgraph_dumper;

// ======================= ExtractorsManagerTest Unit tests =======================
class SubgraphsDumperBaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
    }

    bool is_match(const std::vector<SubgraphExtractor::ExtractedPattern>& models,
                  const std::vector<std::shared_ptr<ov::Model>>& ref_models) {
        size_t match_numbers = 0;
        for (const auto& model : models) {
            bool is_match = false;
            for (const auto& ref_model : ref_models) {
                if (ov::util::ModelComparator::get()->match(std::get<0>(model), ref_model)) {
                    is_match = true;
                    ++match_numbers;
                    break;
                }
            }
            if (!is_match) {
                return false;
            }
        }
        return match_numbers == models.size();
    }
};
