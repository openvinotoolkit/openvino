// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "openvino/core/visibility.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/op/util/op_types.hpp"

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"

#include "cache/graph_cache.hpp"
#include "utils/node.hpp"
#include "test_models/model_0.hpp"
#include "test_models/model_1.hpp"
#include "test_models/model_2.hpp"
#include "base_test.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;

// ====================== Graph Cache Functional tests ==============================

class GraphCacheFuncTest : public SubgraphsDumperBaseTest {
protected:
    std::shared_ptr<ov::Model> test_model;
    std::string test_artifacts_dir, test_model_name, test_model_path;

    void SetUp() override {
        SubgraphsDumperBaseTest::SetUp();
        test_model_name = "test_model_name";
        test_artifacts_dir = ov::util::path_join({ov::test::utils::getCurrentWorkingDir(), "test_artifacts"});
        test_model_path = ov::util::path_join({test_artifacts_dir, test_model_name + ".xml"});
        ov::util::create_directory_recursive(test_artifacts_dir);
        {
            Model_0 test;
            test_model = test.get();
            test_model->set_friendly_name(test_model_name);
        }
    };

    void TearDown() override {
        ov::test::utils::removeDir(test_artifacts_dir);
        GraphCache::reset();
    }
};

TEST_F(GraphCacheFuncTest, get_graph_cache) {
    std::shared_ptr<ov::tools::subgraph_dumper::GraphCache> graph_cache = nullptr;
    EXPECT_NO_THROW(graph_cache = ov::tools::subgraph_dumper::GraphCache::get());
    ASSERT_NE(graph_cache, nullptr);
}

TEST_F(GraphCacheFuncTest, get_graph_cache_twice) {
    std::shared_ptr<ov::tools::subgraph_dumper::GraphCache> graph_cache_0 = nullptr, graph_cache_1 = nullptr;
    graph_cache_0 = ov::tools::subgraph_dumper::GraphCache::get();
    graph_cache_1 = ov::tools::subgraph_dumper::GraphCache::get();
    ASSERT_EQ(graph_cache_0, graph_cache_1);
}

#if (defined OPENVINO_ARCH_ARM && defined(__linux__))
// Ticket: 153168
TEST_F(GraphCacheFuncTest, DISABLED_update_cache) {
#else
TEST_F(GraphCacheFuncTest, update_cache) {
#endif
    auto graph_cache = ov::tools::subgraph_dumper::GraphCache::get();
    graph_cache->update_cache(test_model, test_model_path, true);
    OV_ASSERT_NO_THROW(graph_cache->update_cache(test_model, test_model_path, true));
}

TEST_F(GraphCacheFuncTest, serialize_cache) {
    auto graph_cache = ov::tools::subgraph_dumper::GraphCache::get();
    graph_cache->set_serialization_dir(test_artifacts_dir);
    OV_ASSERT_NO_THROW(graph_cache->serialize_cache());
}

// ====================== Graph Cache Unit tests ==============================

class GraphCacheUnitTest : public GraphCacheFuncTest,
                           public virtual GraphCache {
protected:
    std::shared_ptr<ov::op::v0::Convert> convert_node;
    ov::conformance::MetaInfo test_meta;

    void SetUp() override {
        GraphCacheFuncTest::SetUp();
    }
};

TEST_F(GraphCacheUnitTest, update_cache_by_graph) {
    Model_2 test;
    auto model_to_cache = test.get();
    std::map<std::string, ov::conformance::InputInfo> in_info;
    for (const auto& op : model_to_cache->get_ordered_ops()) {
        if (ov::op::util::is_parameter(op)) {
            in_info.insert({ op->get_friendly_name(), ov::conformance::InputInfo()});
        }
    }
    this->update_cache(model_to_cache, test_model_path, in_info, "test_extractor", model_to_cache->get_ordered_ops().size());
    ASSERT_EQ(m_graph_cache.size(), 1);
}
}  // namespace
