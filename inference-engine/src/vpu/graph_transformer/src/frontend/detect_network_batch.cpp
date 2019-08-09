// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <string>
#include <set>

#include <details/caseless.hpp>
#include <cpp/ie_cnn_network.h>
#include <graph_tools.hpp>

#include <vpu/compile_env.hpp>

namespace vpu {

ie::CNNNetwork FrontEnd::detectNetworkBatch(
        const ie::ICNNNetwork& origNetwork,
        const Model::Ptr& model) {
    VPU_PROFILE(detectNetworkBatch);

    const auto& env = CompileEnv::get();

    env.log->debug("Detect network batch");
    VPU_LOGGER_SECTION(env.log);

    ie::details::CaselessEq<std::string> cmp;

    auto batchSize = origNetwork.getBatchSize();
    env.log->debug("Batch size = %d", batchSize);

    if (batchSize == 1 || !env.config.detectBatch) {
        env.log->debug("Keep original network");

        IE_SUPPRESS_DEPRECATED_START
        return ie::CNNNetwork(const_cast<ie::ICNNNetwork*>(&origNetwork));
        IE_SUPPRESS_DEPRECATED_END
    }

    model->setBatchSize(batchSize);

    //
    // Create a copy of the original network to reshape it.
    //

    ie::CNNNetwork reshapedNetwork(ie::CNNNetCopy(origNetwork));

    auto inputsInfo = reshapedNetwork.getInputsInfo();
    auto outputsInfo = reshapedNetwork.getOutputsInfo();

    //
    // Collect input shapes and remove batch from them.
    //

    env.log->debug("Remove batch from inputs");

    ie::ICNNNetwork::InputShapes inputShapes;

    for (const auto& p : inputsInfo) {
        VPU_LOGGER_SECTION(env.log);

        auto info = p.second;
        IE_ASSERT(info != nullptr);

        auto ieData = info->getInputData();
        IE_ASSERT(ieData != nullptr);

        auto ieShapes = ieData->getTensorDesc().getDims();
        env.log->debug("Input [%s] : %v", p.first, ieShapes);

        switch (ieData->getLayout()) {
            case ie::Layout::NCHW:
            case ie::Layout::NHWC:
            case ie::Layout::NC:
                ieShapes[0] = 1;
                break;
            case ie::Layout::CN:
                ieShapes[1] = 1;
                break;
            default:
                VPU_THROW_EXCEPTION << "Unexpected input layout : " << ieData->getLayout();
        }

        inputShapes[ieData->getName()] = ieShapes;
    }

    //
    // Special case for DetectionOutput.
    //

    for (const auto& layer : reshapedNetwork) {
        if (!cmp(layer->type, "DetectionOutput"))
            continue;

        env.log->debug("Found DetectionOutput layer [%s]", layer->name);

        if (layer->outData.empty()) {
            VPU_LOG_AND_THROW(env.log, "Unsupported layer configuration for %s", layer->name);
        }

        // 1. Don't support if DetectionOutput is not the last layer in network
        if (!layer->outData.front()->getInputTo().empty()) {
            VPU_LOG_AND_THROW(env.log, "Unsupported configuration : layer %s is not a network output", layer->name);
        }

        // 2. Don't support if there multiple outputs as well
        if (outputsInfo.size() != 1) {
            VPU_LOG_AND_THROW(env.log, "Unsupported configuration : layer %s must be the only output of the network", layer->name);
        }

        model->attrs().set<bool>("withDetectionOutput", true);
    }

    //
    // Gathering output shapes before reshaping.
    //

    ie::ICNNNetwork::InputShapes outputShapes;

    for (const auto& pair : outputsInfo) {
        auto ieData = pair.second;
        IE_ASSERT(ieData != nullptr);

        outputShapes[pair.first] = ieData->getDims();
    }

    //
    // Reshape the network.
    //

    env.log->debug("Reshape the network");

    reshapedNetwork.reshape(inputShapes);

    //
    // Checks outputs that doesn't change their shape.
    //

    env.log->debug("Checks for unbatched outputs");

    outputsInfo = reshapedNetwork.getOutputsInfo();

    for (const auto& p : outputsInfo) {
        VPU_LOGGER_SECTION(env.log);

        auto ieData = p.second;
        IE_ASSERT(ieData != nullptr);

        auto origShape = outputShapes[p.first];
        auto newShape = ieData->getDims();

        if (origShape == newShape) {
            env.log->debug("Output [%s] is unbatched", p.first);

            _unbatchedOutputs.insert(ieData);
        }
    }

    return reshapedNetwork;
}

}  // namespace vpu
