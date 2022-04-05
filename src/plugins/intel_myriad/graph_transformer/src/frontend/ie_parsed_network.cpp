// Copyright (C) 2018-2022 Intel Corporation
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
    ie::details::CaselessEq<std::string> cmp;

    env.log->trace("Parse IE network : %s", network.getName());
    VPU_LOGGER_SECTION(env.log);

    IeParsedNetwork out;
    out.networkInputs = network.getInputsInfo();
    out.networkOutputs = network.getOutputsInfo();

    env.log->trace("Got %d inputs and %d outputs", out.networkInputs.size(), out.networkOutputs.size());
    IE_ASSERT(!out.networkOutputs.empty());

    env.log->trace("Perform topological sort");
    const auto sortedLayers = ie::details::CNNNetSortTopologically(network);
    IE_ASSERT(!sortedLayers.empty());

    for (const auto& layer : sortedLayers) {
        VPU_LOGGER_SECTION(env.log);

        IE_ASSERT(layer != nullptr);

        if (cmp(layer->type, "Input")) {
            env.log->trace("Found Input layer : %s", layer->name);
            continue;
        }

        if (cmp(layer->type, "Const")) {
            env.log->trace("Found Const layer : %s", layer->name);

            if (layer->outData.size() != 1) {
                VPU_THROW_FORMAT(
                    "Const layer %v has unsupported number of outputs %v",
                    layer->name, layer->outData.size());
            }

            if (layer->blobs.size() != 1) {
                VPU_THROW_FORMAT(
                    "Const layer %v has unsupported number of blobs %v",
                    layer->name, layer->blobs.size());
            }

            const auto constData = layer->outData[0];
            IE_ASSERT(constData != nullptr);

            const auto constBlob = layer->blobs.begin()->second;
            IE_ASSERT(constBlob != nullptr);

            out.constDatas.emplace_back(constData, constBlob);

            continue;
        }

        env.log->trace("Found plain layer : %s", layer->name);
        out.orderedLayers.push_back(layer);
    }

    return out;
}

}  // namespace vpu
