// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/snippets_test_utils.hpp"

#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/runtime/exec_model_info.hpp"

namespace ov {
namespace test {
namespace snippets {
void SnippetsTestsCommon::validate() {
    ov::test::SubgraphBaseTest::validate();
    validateNumSubgraphs();
}

void SnippetsTestsCommon::validateNumSubgraphs() {
    // Some derived tests don't set expected counts, so skip validation for them.
    if (ref_num_nodes == 0 && ref_num_subgraphs == 0)
        return;

    const auto& compiled_model = compiledModel.get_runtime_model();
    size_t num_subgraphs = 0;
    size_t num_nodes = 0;
    for (const auto &op : compiled_model->get_ops()) {
        auto layer_type = op->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
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

void SnippetsTestsCommon::setInferenceType(ov::element::Type type) {
    configuration.emplace(ov::hint::inference_precision(type));
}

void SnippetsTestsCommon::setIgnoreCallbackMode() {
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
