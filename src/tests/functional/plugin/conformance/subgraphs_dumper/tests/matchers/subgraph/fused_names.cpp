// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>

#include "matchers/subgraph/fused_names.hpp"
#include "utils/model.hpp"
#include "utils/model_comparator.hpp"

#include "test_models/model_0.hpp"
#include "test_models/model_1.hpp"
#include "test_models/model_2.hpp"
#include "base_test.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;


// ======================= ExtractorsManagerTest Unit tests =======================
class FusedNamesExtractorTest : public SubgraphsDumperBaseTest {
    FusedNamesExtractor extractor = FusedNamesExtractor("TEMPLATE");

protected:
    void is_match(const std::shared_ptr<ov::Model>& model) {
        auto models_1 = extractor.extract(model);
        auto models_2 = extractor.extract(model);
        ASSERT_EQ(models_1.size(), models_2.size());
        auto it_model_1 = models_1.begin();
        auto it_model_2 = models_2.begin();
        while (it_model_1 != models_1.end() || it_model_2 != models_2.end()) {
            SubgraphExtractor extractor;
            ASSERT_TRUE(ov::util::ModelComparator::get()->match(std::get<0>(*it_model_1),
                                                                std::get<0>(*it_model_2)));
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
