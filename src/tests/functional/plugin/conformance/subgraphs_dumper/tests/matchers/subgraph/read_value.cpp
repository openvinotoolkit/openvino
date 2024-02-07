// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>

#include "gtest/gtest.h"

#include "matchers/subgraph/read_value_assign.hpp"
#include "utils/model.hpp"
#include "utils/model_comparator.hpp"

#include "test_models/model_0.hpp"
#include "test_models/model_4.hpp"
#include "test_models/model_5.hpp"
#include "base_test.hpp"

#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;


// ======================= ExtractorsManagerTest Unit tests =======================
class ReadValueAssignExtractorTest : public SubgraphsDumperBaseTest {
protected:
    ReadValueAssignExtractor extractor = ReadValueAssignExtractor();

    bool is_match(const std::vector<ReadValueAssignExtractor::ExtractedPattern>& models,
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

TEST_F(ReadValueAssignExtractorTest, extract_0) {
    auto test_model = Model_0();
    auto models = extractor.extract(test_model.get());
    ASSERT_EQ(models.size(), 0);
}

TEST_F(ReadValueAssignExtractorTest, extract_1) {
    auto test_model = Model_4();
    auto models = extractor.extract(test_model.get());
    ASSERT_EQ(models.size(), 4);
    auto ref_models = test_model.get_ref_models();
    ASSERT_TRUE(is_match(models, ref_models));
}

TEST_F(ReadValueAssignExtractorTest, extract_2) {
    auto test_model = Model_5();
    auto models = extractor.extract(test_model.get());
    ASSERT_EQ(models.size(), 1);
    auto ref_models = test_model.get_ref_models();
    ASSERT_TRUE(is_match(models, ref_models));
}

}  // namespace
