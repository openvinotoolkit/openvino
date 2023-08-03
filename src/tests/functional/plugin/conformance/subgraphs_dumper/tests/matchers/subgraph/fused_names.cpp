// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>

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
    void is_match_smoke(const std::shared_ptr<ov::Model>& model) {
        size_t graph_cnt = 0;
        {
            auto compiled_names = extract_compiled_model_names(model);
            std::vector<size_t> op_cnt;
            size_t current_op_cnt = 0;
            for (const auto& op : model->get_ordered_ops()) {
                if (this->is_node_to_skip(op)) {
                    continue;
                }
                auto op_name = op->get_friendly_name();
                if (!compiled_names.count(op_name)) {
                    ++current_op_cnt;
                } else {
                    if (current_op_cnt > 1) {
                        op_cnt.push_back(current_op_cnt);
                    }
                    current_op_cnt = 0;
                }
            }
            graph_cnt = op_cnt.size();
        }
        auto models = this->extract(model);
        ASSERT_EQ(models.size(), graph_cnt);
    }

    void is_match_2_runs(const std::shared_ptr<ov::Model>& model) {
        auto models_1 = this->extract(model);
        auto models_2 = this->extract(model);
        ASSERT_EQ(models_1.size(), models_2.size());
        auto it_model_1 = models_1.begin();
        auto it_model_2 = models_2.begin();
        while (it_model_1 != models_1.end() || it_model_2 != models_2.end()) {
            SubgraphExtractor extractor;
            ASSERT_TRUE(extractor.match(std::get<0>(*it_model_1), std::get<0>(*it_model_2)));
            auto in_info_1 = std::get<1>(*it_model_1);
            auto in_info_2 = std::get<1>(*it_model_2);
            for (const auto& in_info : in_info_1) {
                ASSERT_TRUE(in_info_2.count(in_info.first));
                ASSERT_EQ(in_info_2[in_info.first], in_info.second);
            }
            ASSERT_EQ(std::get<2>(*it_model_1), std::get<2>(*it_model_2));
            ++it_model_1;
            ++it_model_2;
        }
    }

    void is_match(const std::shared_ptr<ov::Model>& model) {
        is_match_smoke(model);
        is_match_2_runs(model);
    }
};

TEST_F(FusedNamesExtractorTest, extract_0) {
    auto test_model = Model_0();
    auto model = test_model.get();
    is_match(model);
}

TEST_F(FusedNamesExtractorTest, extract_1) {
    auto test_model = Model_1();
    auto model = test_model.get();
    is_match(model);
}

TEST_F(FusedNamesExtractorTest, extract_2) {
    auto test_model = Model_2();
    auto model = test_model.get();
    is_match(model);
}

}  // namespace
