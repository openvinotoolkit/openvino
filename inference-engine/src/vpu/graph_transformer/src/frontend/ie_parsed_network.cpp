// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/ie_parsed_network.hpp>

#include <string>

#include <legacy/details/ie_cnn_network_tools.h>
#include <caseless.hpp>

#include <vpu/compile_env.hpp>

namespace vpu {

IeParsedNetwork parseNetwork(const ie::CNNNetwork& network) {
    VPU_PROFILE(parseNetwork);

    const auto& env = CompileEnv::get();

    env.log->trace("Parse IE network : %s", network.getName());
    VPU_LOGGER_SECTION(env.log);

    IeParsedNetwork out;
    out.networkInputs = network.getInputsInfo();
    out.networkOutputs = network.getOutputsInfo();
    env.log->trace("Got %d inputs and %d outputs", out.networkInputs.size(), out.networkOutputs.size());
    IE_ASSERT(!out.networkInputs.empty());
    IE_ASSERT(!out.networkOutputs.empty());

    env.log->trace("Perform topological sort");
    const auto sortedNodes = network.getFunction()->get_ordered_ops();
    IE_ASSERT(!sortedNodes.empty());
    for (const auto& node : sortedNodes) {
        VPU_LOGGER_SECTION(env.log);
        IE_ASSERT(node != nullptr);
        if (ngraph::as_type_ptr<ngraph::op::Parameter>(node)) {
            env.log->trace("Found Parameter node : %s", node->get_friendly_name());
            out.networkParameters.emplace_back(node);
            continue;
        }
        if (ngraph::as_type_ptr<ngraph::op::Result>(node)) {
            env.log->trace("Found Result node : %s", node->get_friendly_name());
            out.networkResults.emplace_back(node);
            continue;
        }

        if (ngraph::as_type_ptr<ngraph::op::Constant>(node)) {
            env.log->trace("Found Const node : %s", node->get_friendly_name());
            if (node->get_output_size() != 1) {
                VPU_THROW_FORMAT(
                    "Const node %v has unsupported number of outputs %v",
                    node->get_friendly_name(), node->get_output_size());
            }

            out.networkConstants.emplace_back(node);

            continue;
        }

        env.log->trace("Found plain layer : %s", node->get_friendly_name());
        out.orderedOps.push_back(node);
    }

    return out;
}

}  // namespace vpu
