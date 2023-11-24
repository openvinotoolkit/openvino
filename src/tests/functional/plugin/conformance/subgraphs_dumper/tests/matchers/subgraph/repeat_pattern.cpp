// Copyright (C) 2018-2023 Intel Corporation
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

    bool is_match(const std::vector<RepeatPatternExtractor::ExtractedPattern>& models,
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

    void sort_node_vec(std::vector<std::vector<ov::NodeVector>>& pattern_vec) {
        for (auto& pattern : pattern_vec) {
            for (auto& node_vec : pattern) {
                std::sort(node_vec.begin(), node_vec.end());
            }
            std::sort(pattern.begin(), pattern.end());
        }
        std::sort(pattern_vec.begin(), pattern_vec.end());
    }

    // not allowed to sort inputs/outputs according there are not copy constructor
    // void sort_borders(std::vector<std::vector<RepeatPatternExtractor::PatternBorders>>& pattern_vec) {
    //     for (auto& pattern : pattern_vec) {
    //         for (auto& node_vec : pattern) {
    //             std::sort(node_vec.first.begin(), node_vec.first.end());
    //             std::sort(node_vec.second.begin(), node_vec.second.end());
    //         }
    //         std::sort(pattern.begin(), pattern.end());
    //     }
    //     std::sort(pattern_vec.begin(), pattern_vec.end());
    // }
};

TEST_F(RepeatPatternExtractorFuncTest, extract_0) {
    auto test_model = Model_0();
    auto models = extractor.extract(test_model.get());
    auto ref = test_model.get_repeat_pattern_ref();
    ASSERT_TRUE(is_match(models, ref));
}

TEST_F(RepeatPatternExtractorFuncTest, extract_1) {
    auto test_model = Model_1();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(
    "/Users/iefode/repo/openvino/src/tests/functional/plugin/conformance/subgraphs_dumper/tests/test_models/test.xml",
    "/Users/iefode/repo/openvino/src/tests/functional/plugin/conformance/subgraphs_dumper/tests/test_models/test.bin");
    manager.run_passes(test_model.get());
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
    // sort_borders(extracted_borders);
    // sort_borders(ref_borders);
    ASSERT_EQ(extracted_borders, ref_borders);
}

TEST_F(RepeatPatternExtractorFuncTest, get_repeat_pattern_borders_model_1) {
    auto test_model = Model_1();
    auto extracted_borders = extractor.get_repeat_pattern_borders(test_model.get());
    auto ref_borders = test_model.get_ref_node_borders();
    // sort_borders(extracted_borders);
    // sort_borders(ref_borders);
    ASSERT_EQ(extracted_borders, ref_borders);
}

TEST_F(RepeatPatternExtractorFuncTest, get_repeat_pattern_borders_model_2) {
    auto test_model = Model_2();
    auto extracted_borders = extractor.get_repeat_pattern_borders(test_model.get());
    auto ref_borders = test_model.get_ref_node_borders();
    // sort_borders(extracted_borders);
    // sort_borders(ref_borders);
    ASSERT_EQ(extracted_borders, ref_borders);
}

class RepeatPatternExtractorUnitTest : public SubgraphsDumperBaseTest,
                                       public RepeatPatternExtractor {};

TEST_F(RepeatPatternExtractorUnitTest, get_ordered_nodes) {
    auto test_model = Model_3();
    auto start_ops = test_model.get_start_ops();
    auto ordered_pattern = this->get_ordered_nodes(start_ops);
    auto ordered_pattern_ref = test_model.get_ordered_patterns();
    auto s_external = ordered_pattern.size();
    ASSERT_EQ(ordered_pattern, ordered_pattern_ref);
    auto repeats = this->post_process_patterns(ordered_pattern);
    auto repeats_ref = test_model.get_repeats();
    ASSERT_EQ(repeats, repeats_ref);
}

TEST_F(RepeatPatternExtractorUnitTest, get_node_queque) {
    auto test_model = Model_3();
    auto start_ops = test_model.get_start_ops();
    ov::NodeVector nodes_queue;
    this->get_node_queque(nodes_queue, start_ops.front());
    auto nodes_queue_ref = test_model.get_queue();
    ASSERT_EQ(nodes_queue, nodes_queue_ref);
}
}  // namespace
