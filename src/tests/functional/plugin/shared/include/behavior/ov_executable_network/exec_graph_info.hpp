// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifcorer: Apache-2.0
//
#pragma once

#include <fstream>

#include "exec_graph_info.hpp"
#include "base/ov_behavior_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "pugixml.hpp"

namespace ov {
namespace test {
namespace behavior {

typedef std::tuple<
        ov::element::Type_t,                // Element type
        std::string,                        // Device name
        ov::AnyMap                          // Config
> OVExecGraphImportExportTestParams;

class OVExecGraphImportExportTest : public testing::WithParamInterface<OVExecGraphImportExportTestParams>,
                                    public OVCompiledNetworkTestBase {
    public:
    static std::string getTestCaseName(testing::TestParamInfo<OVExecGraphImportExportTestParams> obj);

    void SetUp() override;

    void TearDown() override;

    protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    ov::element::Type_t elementType;
    std::shared_ptr<ov::Model> function;
};

class OVExecGraphUniqueNodeNames : public testing::WithParamInterface<ov::test::BasicParams>,
                                   public OVCompiledNetworkTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::BasicParams> obj);
    void SetUp() override;

protected:
    std::shared_ptr<ov::Model> fnPtr;
};

class OVExecGraphSerializationTest : public testing::WithParamInterface<std::string>,
                                     public OVCompiledNetworkTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
    void SetUp() override;
    void TearDown() override;

private:
    // walker traverse (DFS) xml document and store layer & data nodes in
    // vector which is later used for comparison
    struct exec_graph_walker : pugi::xml_tree_walker {
        std::vector<pugi::xml_node> nodes;
        bool for_each(pugi::xml_node &node) override;
    };

    // compare_docs() helper
    std::pair<bool, std::string> compare_nodes(const pugi::xml_node &node1,
                                               const pugi::xml_node &node2);

protected:
    // checks if two exec graph xml's are equivalent:
    // - the same count of <layer> and <data> nodes
    // - the same count of attributes of each node
    // - the same name of each attribute (value is not checked, since it can differ
    // beetween different devices)
    std::pair<bool, std::string> compare_docs(const pugi::xml_document &doc1,
                                              const pugi::xml_document &doc2);

    std::string m_out_xml_path, m_out_bin_path;
};
}  // namespace behavior
}  // namespace test
}  // namespace ov
