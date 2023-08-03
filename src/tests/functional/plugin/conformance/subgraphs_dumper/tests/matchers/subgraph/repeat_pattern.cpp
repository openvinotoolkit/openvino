// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "matchers/subgraph/repeat_pattern.hpp"
#include "utils/model.hpp"

#include "base_test.hpp"
#include "test_models/model_0.hpp"
#include "test_models/model_1.hpp"
#include "test_models/model_2.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;


// ======================= ExtractorsManagerTest Unit tests =======================
class RepeatPatternExtractorTest : public RepeatPatternExtractor,
                                   public SubgraphsDumperBaseTest {
protected:
    bool is_match(const std::list<ExtractedPattern>& models,
                  const std::vector<std::shared_ptr<ov::Model>>& ref_models) {
        size_t match_numbers = 0;
        for (const auto& model : models) {
            bool is_match = false;
            for (const auto& ref_model : ref_models) {
                if (this->match(model.first, ref_model)) {
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

TEST_F(RepeatPatternExtractorTest, extract_0) {
    auto test_model = Model_0();
    auto models = this->extract(test_model.get());
    auto ref = test_model.get_repeat_pattern_ref();
    ASSERT_TRUE(is_match(models, ref));
}

TEST_F(RepeatPatternExtractorTest, extract_1) {
    auto test_model = Model_1();
    auto models = this->extract(test_model.get());
    auto ref = test_model.get_repeat_pattern_ref();
    ASSERT_TRUE(is_match(models, ref));
}

TEST_F(RepeatPatternExtractorTest, extract_2) {
    auto test_model = Model_2();
    auto models = this->extract(test_model.get());
    auto ref = test_model.get_repeat_pattern_ref();
    ASSERT_TRUE(is_match(models, ref));
}

}  // namespace
