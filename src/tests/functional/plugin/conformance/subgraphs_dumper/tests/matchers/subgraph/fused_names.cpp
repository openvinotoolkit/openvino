// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/subgraph/fused_names.hpp"
#include "utils/model.hpp"

#include "test_models/model_0.hpp"
#include "test_models/model_1.hpp"
#include "test_models/model_2.hpp"
#include "base_test.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;


// ======================= ExtractorsManagerTest Unit tests =======================
class FusedNamesExtractorTest : public FusedNamesExtractor,
                                public SubgraphsDumperBaseTest {
protected:
    bool is_match(const std::shared_ptr<ov::Model>& model) {
        size_t graph_cnt = 0;
        {
            auto compiled_names = extract_compiled_model_names(model);
            std::vector<size_t> op_cnt;
            for (const auto& op : model->get_ordered_ops()) {
                if (this->is_node_to_skip(op)) {
                    op_cnt.push_back(1);
                    continue;
                }
                auto op_name = op->get_friendly_name();
                if (!compiled_names.count(op_name)) {
                    op_cnt.push_back(1);
                } else if (op_cnt.size() > 0) {
                    ++op_cnt[op_cnt.size() - 1];
                }
            }
            for (const auto& cnt : op_cnt) {
                if (cnt > 1) {
                    ++graph_cnt;
                }
            }
        }
        auto models = this->extract(model);
        return models.size() == graph_cnt;
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
