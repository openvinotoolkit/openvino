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

    ie::details::CaselessEq<std::string> cmp;

    auto batchSize = origNetwork.getBatchSize();

    if (batchSize == 1 || !env.config.detectBatch) {
        // Keep original network.
        return ie::CNNNetwork(const_cast<ie::ICNNNetwork*>(&origNetwork));
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

    ie::ICNNNetwork::InputShapes inputShapes;

    for (const auto& p : inputsInfo) {
        auto info = p.second;
        IE_ASSERT(info != nullptr);

        auto ieData = info->getInputData();
        IE_ASSERT(ieData != nullptr);

        inputShapes[ieData->name] = ieData->getTensorDesc().getDims();
        switch (ieData->getLayout()) {
            case ie::Layout::NCHW:
            case ie::Layout::NHWC:
            case ie::Layout::NC:
                inputShapes[ieData->name][0] = 1;
                break;
            case ie::Layout::CN:
                inputShapes[ieData->name][1] = 1;
                break;
            default:
                VPU_THROW_EXCEPTION << "Unexpected input layout : " << ieData->getLayout();
        }
    }

    //
    // Special case for DetectionOutput.
    //

    for (const auto& layer : reshapedNetwork) {
        if (!cmp(layer->type, "DetectionOutput"))
            continue;

        if (layer->outData.empty()) {
            VPU_THROW_EXCEPTION << "Unsupported layer configuration for " << layer->name;
        }

        // 1. Don't support if DetectionOutput is not the last layer in network
        if (!layer->outData.front()->getInputTo().empty()) {
            VPU_THROW_EXCEPTION << "Unsupported configuration : layer "<< layer->name << " is not a network output";
        }

        // 2. Don't support if there multiple outputs as well
        if (outputsInfo.size() != 1) {
            VPU_THROW_EXCEPTION << "Unsupported configuration : layer "<< layer->name << " must be the only output of the network";
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

    reshapedNetwork.reshape(inputShapes);

    //
    // Checks outputs that doesn't change their shape.
    //

    outputsInfo = reshapedNetwork.getOutputsInfo();

    for (const auto& pair : outputsInfo) {
        auto ieData = pair.second;
        IE_ASSERT(ieData != nullptr);

        auto origShape = outputShapes[pair.first];
        auto newShape = ieData->getDims();

        if (origShape == newShape) {
            _unbatchedOutputs.insert(ieData);
        }
    }

    return reshapedNetwork;
}

}  // namespace vpu
