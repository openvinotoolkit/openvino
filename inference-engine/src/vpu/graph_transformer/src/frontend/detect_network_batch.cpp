// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <string>
#include <set>
#include <map>

#include <legacy/details/ie_cnn_network_iterator.hpp>
#include <cpp/ie_cnn_network.h>
#include <legacy/graph_tools.hpp>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset3.hpp>

#include <vpu/compile_env.hpp>

namespace vpu {

void FrontEnd::detectNetworkBatch(
        ie::ICNNNetwork& network,
        const Model& model) {
    VPU_PROFILE(detectNetworkBatch);

    const auto& env = CompileEnv::get();

    auto checkForDeprecatedCnn = [&network, &env]() {
        return !network.getFunction()
               && !env.config.forceDeprecatedCnnConversion
               && !dynamic_cast<const ie::details::CNNNetworkImpl*>(&network);
    };

    if (!env.config.detectBatch || checkForDeprecatedCnn()) {
        // skip batch extraction step and go as is
        return;
    }

    env.log->trace("Detect network batch");
    VPU_LOGGER_SECTION(env.log);

    //
    // Get information about Network inputs and outputs.
    //

    ie::InputsDataMap inputsInfo;
    ie::OutputsDataMap outputsInfo;

    network.getInputsInfo(inputsInfo);
    network.getOutputsInfo(outputsInfo);

    //
    // Collect input shapes and remove batch from them.
    //

    env.log->trace("Remove batch from inputs");

    ie::ICNNNetwork::InputShapes inputShapes;
    size_t inputBatch = 0;

    for (const auto& p : inputsInfo) {
        VPU_LOGGER_SECTION(env.log);

        const auto info = p.second;
        IE_ASSERT(info != nullptr);

        const auto ieData = info->getInputData();
        IE_ASSERT(ieData != nullptr);

        auto ieShapes = ieData->getTensorDesc().getDims();
        env.log->trace("Input [%s] : %v", p.first, ieShapes);

        if (ieShapes.size() == 4 || ieShapes.size() == 5) {
            auto batch = ieShapes[0];
            if (!inputBatch)
                inputBatch = batch;

            if (inputBatch != batch) {
                env.log->trace("Network has inputs with different batch dim, aborting batch detection");
                return;
            }

            ieShapes[0] = 1;
        } else {
            // assume only 4D and 5D inputs have batch
        }

        inputShapes[ieData->getName()] = ieShapes;
    }

    if (inputBatch == 1) {
        env.log->trace("Network is batch 1. Not need to reshape");
        return;
    }

    if (inputBatch == 0) {
        env.log->trace("Unable to decide on network batch size.");
        return;
    }

    //
    // Special case for DetectionOutput.
    //

    const auto layers = network.getFunction()->get_ops();
    for (const auto& layer : layers) {
        if (layer->get_type_name() != std::string("DetectionOutput"))
            continue;

        env.log->trace("Found DetectionOutput layer [%s]", layer->get_name());

        if (layer->get_output_size() == 0)
            VPU_THROW_FORMAT("Unsupported layer %s configuration: no outputs", layer->get_name());

        // 1. Don't support if DetectionOutput is not the last layer in network
        // for (const auto& outputHandle : layer->outputs()) {
        //     for (const auto& inputHandle : outputHandle.get_target_inputs()) {
        //         auto outNode = inputHandle.get_node();
        //         if (dynamic_cast<::ngraph::opset3::Result *>(outNode)) {
        //             continue;
        //         }
        //         VPU_THROW_FORMAT("Unsupported layer %s configuration : it is not a network output", layer->get_name());
        //     }
        // }

        // 2. Don't support if there multiple outputs as well
        if (outputsInfo.size() != 1) {
            env.log->trace("Unsupported network configuration : layer %s must be the only output of the network", layer->get_name());
            return;
        }
        model->attrs().set<bool>("withDetectionOutput", true);
    }

    //
    // Gathering output shapes and precisions before reshaping.
    //

    ie::ICNNNetwork::InputShapes outputShapes;
    std::map<std::string, ie::Precision> outputsPresisions;

    for (const auto& pair : outputsInfo) {
        const auto ieData = pair.second;
        IE_ASSERT(ieData != nullptr);

        outputShapes[pair.first] = ieData->getDims();
        outputsPresisions[pair.first] = ieData->getPrecision();
    }

    //
    // Reshape the network.
    //

    env.log->trace("Reshape the network");

    ie::ResponseDesc desc;
    auto status = ie::StatusCode::OK;
    try {
        status = network.reshape(inputShapes, &desc);
    }
    catch (...) {
        status = ie::StatusCode::GENERAL_ERROR;
    }

    if (status != ie::StatusCode::OK) {
        env.log->trace("Failed to reshape Network: %v", desc.msg);
        return;
    }

    env.log->trace("Batch %d is successfully removed from the model", inputBatch);
    model->setBatchSize(checked_cast<int>(inputBatch));

    //
    // Checks outputs that doesn't change their shape.
    //

    env.log->trace("Checks for unbatched outputs");

    network.getOutputsInfo(outputsInfo);

    for (const auto& p : outputsInfo) {
        VPU_LOGGER_SECTION(env.log);

        const auto ieData = p.second;
        IE_ASSERT(ieData != nullptr);

        const auto origShape = outputShapes[p.first];
        const auto newShape = ieData->getDims();

        if (origShape == newShape) {
            env.log->trace("Output [%s] is unbatched", p.first);

            _unbatchedOutputs.insert(ieData);
        }

        // preserve overriden output precision.
        ieData->setPrecision(outputsPresisions[p.first]);
    }
}

}  // namespace vpu
