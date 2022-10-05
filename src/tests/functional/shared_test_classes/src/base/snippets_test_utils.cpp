// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/snippets_test_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "exec_graph_info.hpp"

namespace ov {
namespace test {
void SnippetsTestsCommon::validateNumSubgraphs() {
    bool isCurrentTestDisabled = FuncTestUtils::SkipTestsConfig::currentTestIsDisabled();
    if (isCurrentTestDisabled)
        GTEST_SKIP() << "Disabled test due to configuration" << std::endl;

    const auto& compiled_model = compiledModel.get_runtime_model();
    size_t num_subgraphs = 0;
    size_t num_nodes = 0;
    for (const auto &op : compiled_model->get_ops()) {
        auto layer_type = op->get_rt_info().at(ExecGraphInfoSerialization::LAYER_TYPE).as<std::string>();
        // todo: Ignore reorders only after (Const or Inputs) or before outputs.
        //  Alternatively, force plain layouts for convolutions, matmuls, FCs, etc., so reorders won't be inserted.
        if (layer_type == "Const" ||
            layer_type == "Input" ||
            layer_type == "Output")
            continue;
        auto &rt = op->get_rt_info();
        const auto rinfo = rt.find("layerType");
        ASSERT_NE(rinfo, rt.end()) << "Failed to find layerType in " << op->get_friendly_name()
                                   << "rt_info.";
        const std::string layerType = rinfo->second.as<std::string>();
        num_subgraphs += layerType == "Subgraph";
        num_nodes++;
    }
    ASSERT_EQ(ref_num_nodes, num_nodes) << "Compiled model contains invalid number of nodes.";
    ASSERT_EQ(ref_num_subgraphs, num_subgraphs) << "Compiled model contains invalid number of subgraphs.";
}

void SnippetsTestsCommon::validateOriginalLayersNamesByType(const std::string& layerType, const std::string& originalLayersNames) {
    const auto& compiled_model = compiledModel.get_runtime_model();
    for (const auto& op : compiled_model->get_ops()) {
        const auto& rtInfo = op->get_rt_info();

        const auto& typeIt = rtInfo.find("layerType");
        const auto type = typeIt->second.as<std::string>();
        if (type == layerType) {
            const auto& nameIt = rtInfo.find("originalLayersNames");
            const auto name = nameIt->second.as<std::string>();
            ASSERT_EQ(originalLayersNames, name);
            return;
        }
    }

    ASSERT_TRUE(false) << "Layer type '" << layerType << "' was not found in compiled model";
}

}  // namespace test
}  // namespace ov
