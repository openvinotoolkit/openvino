// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "base/behavior_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

#include "pugixml.hpp"

namespace ExecutionGraphTests {

class ExecGraphUniqueNodeNames : public testing::WithParamInterface<LayerTestsUtils::basicParams>,
                                 public CommonTestUtils::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LayerTestsUtils::basicParams> obj);
    void SetUp() override;
    void TearDown() override;

protected:
    std::string targetDevice;
    std::shared_ptr<ngraph::Function> fnPtr;
};

class ExecGraphSerializationTest : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<std::string> {
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

    std::string deviceName, m_out_xml_path, m_out_bin_path;
};
}  // namespace ExecutionGraphTests
