// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "matchers/subgraph/fused_names.hpp"
#include "utils/model.hpp"

#include "test_models/model_0.hpp"
#include "test_models/model_1.hpp"
#include "test_models/model_2.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;


// ======================= ExtractorsManagerTest Unit tests =======================
class FusedNamesExtractorTest : public FusedNamesExtractor,
                                   public ::testing::Test {
protected:
    bool is_match(const std::shared_ptr<ov::Model>& model) {
        auto compiled_names = extract_compiled_model_names(model);
        std::set<std::string> diff;
        for (const auto& op : model->get_ordered_ops()) {
            auto op_name = op->get_friendly_name();
            if (!compiled_names.count(op_name)) {
                diff.insert(op_name);
            }
        }
        auto models = this->extract(model);
        return diff.size() == 0 ? true : models.size() + 2 == diff.size();
    }
};

TEST_F(FusedNamesExtractorTest, extract_0) {
    auto test_model = Model_0();
    auto model = test_model.get();
    ASSERT_TRUE(is_match(model));
}

TEST_F(FusedNamesExtractorTest, extract_1) {
    auto test_model = Model_1();
    auto model = test_model.get();
    ASSERT_TRUE(is_match(model));
}

TEST_F(FusedNamesExtractorTest, extract_2) {
    auto test_model = Model_2();
    auto model = test_model.get();
    ASSERT_TRUE(is_match(model));
}

}  // namespace
