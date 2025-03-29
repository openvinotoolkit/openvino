// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "matchers/subgraph/repeat_pattern.hpp"
#include "utils/model.hpp"
#include "utils/model_comparator.hpp"

#include "base_test.hpp"
#include "test_models/model_0.hpp"
#include "test_models/model_1.hpp"
#include "test_models/model_2.hpp"
#include "test_models/model_3.hpp"

#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;


// ======================= ExtractorsManagerTest Func tests =======================
class RepeatPatternExtractorFuncTest : public SubgraphsDumperBaseTest {
protected:
    RepeatPatternExtractor extractor;

    void sort_node_vec(std::vector<std::vector<ov::NodeVector>>& pattern_vec) {
        for (auto& pattern : pattern_vec) {
            for (auto& node_vec : pattern) {
                std::sort(node_vec.begin(), node_vec.end());
            }
            std::sort(pattern.begin(), pattern.end());
        }
        std::sort(pattern_vec.begin(), pattern_vec.end());
    }

    void
    is_equal_borders(const std::vector<std::vector<RepeatPatternExtractor::PatternBorders>>& pattern_vec_orig,
                     const std::vector<std::vector<RepeatPatternExtractor::PatternBorders>>& pattern_vec_ref) {
        ASSERT_EQ(pattern_vec_orig.size(), pattern_vec_ref.size());
        size_t orig_borders_cnt = 0, ref_borderd_cnt = 0, eq_borders = 0;
        for (const auto& pattern_orig : pattern_vec_orig) {
            orig_borders_cnt += pattern_orig.size();
            ref_borderd_cnt = 0;
            for (const auto& pattern_ref : pattern_vec_ref) {
                ref_borderd_cnt += pattern_ref.size();
                if (pattern_ref.size() != pattern_orig.size()) {
                    continue;
                }
                for (const auto& node_vec_orig : pattern_orig) {
                    // size_t eq_pattens = 0;
                    for (const auto& node_vec_ref : pattern_ref) {
                        if (node_vec_orig == node_vec_ref) {
                            ++eq_borders;
                            break;
                        }
                    }
                }
            }
        }
        ASSERT_EQ(orig_borders_cnt, ref_borderd_cnt);
        ASSERT_EQ(orig_borders_cnt, eq_borders);
    }
};

TEST_F(RepeatPatternExtractorFuncTest, extract_0) {
    auto test_model = Model_0();
    auto models = extractor.extract(test_model.get());
    auto ref = test_model.get_repeat_pattern_ref();
    ASSERT_TRUE(is_match(models, ref));
}

TEST_F(RepeatPatternExtractorFuncTest, extract_1) {
    auto test_model = Model_1();
    auto models = extractor.extract(test_model.get());
    auto ref = test_model.get_repeat_pattern_ref();
    ASSERT_TRUE(is_match(models, ref));
}

TEST_F(RepeatPatternExtractorFuncTest, extract_2) {
    auto test_model = Model_2();
    auto models = extractor.extract(test_model.get());
    auto ref = test_model.get_repeat_pattern_ref();
    ASSERT_TRUE(is_match(models, ref));
}

TEST_F(RepeatPatternExtractorFuncTest, get_repeat_node_vectors_model_0) {
    auto test_model = Model_0();
    auto node_vector = extractor.get_repeat_node_vectors(test_model.get());
    auto ref = test_model.get_ref_node_vector();
    sort_node_vec(node_vector);
    sort_node_vec(ref);
    ASSERT_EQ(node_vector, ref);
}

TEST_F(RepeatPatternExtractorFuncTest, get_repeat_node_vectors_model_1) {
    auto test_model = Model_1();
    auto node_vector = extractor.get_repeat_node_vectors(test_model.get());
    auto ref = test_model.get_ref_node_vector();
    sort_node_vec(node_vector);
    sort_node_vec(ref);
    ASSERT_EQ(node_vector, ref);
}

TEST_F(RepeatPatternExtractorFuncTest, get_repeat_node_vectors_model_2) {
    auto test_model = Model_2();
    auto node_vector = extractor.get_repeat_node_vectors(test_model.get());
    auto ref = test_model.get_ref_node_vector();
    sort_node_vec(node_vector);
    sort_node_vec(ref);
    ASSERT_EQ(node_vector, ref);
}

TEST_F(RepeatPatternExtractorFuncTest, get_repeat_pattern_borders_model_0) {
    auto test_model = Model_0();
    auto extracted_borders = extractor.get_repeat_pattern_borders(test_model.get());
    auto ref_borders = test_model.get_ref_node_borders();
    is_equal_borders(extracted_borders, ref_borders);
}

TEST_F(RepeatPatternExtractorFuncTest, get_repeat_pattern_borders_model_1) {
    auto test_model = Model_1();
    auto extracted_borders = extractor.get_repeat_pattern_borders(test_model.get());
    auto ref_borders = test_model.get_ref_node_borders();
    is_equal_borders(extracted_borders, ref_borders);
}

TEST_F(RepeatPatternExtractorFuncTest, get_repeat_pattern_borders_model_2) {
    auto test_model = Model_2();
    auto extracted_borders = extractor.get_repeat_pattern_borders(test_model.get());
    auto ref_borders = test_model.get_ref_node_borders();
    is_equal_borders(extracted_borders, ref_borders);
}
}  // namespace
