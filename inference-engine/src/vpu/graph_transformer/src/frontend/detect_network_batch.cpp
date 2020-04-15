// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <string>
#include <set>
#include <map>

#include <details/caseless.hpp>
#include <details/ie_cnn_network_iterator.hpp>
#include <cpp/ie_cnn_network.h>
#include <graph_tools.hpp>

#include <ngraph/function.hpp>

#include <vpu/compile_env.hpp>

namespace vpu {

void FrontEnd::detectNetworkBatch(
        ie::ICNNNetwork& network,
        const Model& model) {
    VPU_PROFILE(detectNetworkBatch);

    const auto& env = CompileEnv::get();
    ie::details::CaselessEq<std::string> cmp;

    env.log->trace("Detect network batch");
    VPU_LOGGER_SECTION(env.log);

    const auto batchSize = network.getBatchSize();
    env.log->trace("Batch size = %d", batchSize);

    auto checkForDeprecatedCnn = [&network, &env]() {
        return !network.getFunction()
               && !env.config.forceDeprecatedCnnConversion
               && !dynamic_cast<const ie::details::CNNNetworkImpl*>(&network);
    };
    VPU_THROW_UNLESS(!checkForDeprecatedCnn(), "Unexpected CNNNetwork format: it was converted to deprecated format prior plugin's call");

    if (batchSize == 1 || !env.config.detectBatch) {
        env.log->trace("Keep original network");
        return;
    }

    model->setBatchSize(checked_cast<int>(batchSize));

    //
    // Get information about Network inputs and outputs.
    //

    ie::InputsDataMap inputsInfo;
    ie::OutputsDataMap outputsInfo;

    network.getInputsInfo(inputsInfo);
    network.getOutputsInfo(outputsInfo);

    std::map<std::string, ie::Precision> outputsPresisions;
    for (const auto& pair : outputsInfo)
        outputsPresisions[pair.first] = pair.second->getPrecision();

    //
    // Collect input shapes and remove batch from them.
    //

    env.log->trace("Remove batch from inputs");

    ie::ICNNNetwork::InputShapes inputShapes;

    for (const auto& p : inputsInfo) {
        VPU_LOGGER_SECTION(env.log);

        const auto info = p.second;
        IE_ASSERT(info != nullptr);

        const auto ieData = info->getInputData();
        IE_ASSERT(ieData != nullptr);

        auto ieShapes = ieData->getTensorDesc().getDims();
        env.log->trace("Input [%s] : %v", p.first, ieShapes);

        switch (ieData->getLayout()) {
        case ie::Layout::NCDHW:
        case ie::Layout::NDHWC:
        case ie::Layout::NCHW:
        case ie::Layout::NHWC:
        case ie::Layout::NC:
        case ie::Layout::CN:
            ieShapes[0] = 1;
            break;
        default:
            VPU_THROW_FORMAT("Input %v has unexpected layout %v", ieData->getName(), ieData->getLayout());
        }

        inputShapes[ieData->getName()] = ieShapes;
    }

    //
    // Special case for DetectionOutput.
    //

    if (!network.getFunction()) {
        for (auto it = ie::details::CNNNetworkIterator(&network); it != ie::details::CNNNetworkIterator(); ++it) {
            const auto& layer = *it;

            if (!cmp(layer->type, "DetectionOutput")) {
                continue;
            }

            env.log->trace("Found DetectionOutput layer [%s]", layer->name);

            if (layer->outData.empty()) {
                VPU_THROW_FORMAT("Unsupported layer %s configuration: no outputs", layer->name);
            }

            // 1. Don't support if DetectionOutput is not the last layer in network
            if (!layer->outData.front()->getInputTo().empty()) {
                VPU_THROW_FORMAT("Unsupported layer %s configuration : it is not a network output", layer->name);
            }

            // 2. Don't support if there multiple outputs as well
            if (outputsInfo.size() != 1) {
                VPU_THROW_FORMAT("Unsupported network configuration : layer %s must be the only output of the network", layer->name);
            }

            model->attrs().set<bool>("withDetectionOutput", true);
        }
    } else {
        const auto layers = network.getFunction()->get_ops();
        for (const auto& layer : layers) {
            if (layer->get_type_name() != std::string("DetectionOutput"))
                continue;

            env.log->trace("Found DetectionOutput layer [%s]", layer->get_name());

            if (layer->get_output_size() == 0)
                VPU_THROW_FORMAT("Unsupported layer %s configuration: no outputs", layer->get_name());

            // 1. Don't support if DetectionOutput is not the last layer in network
            for (const auto& outputHandle : layer->get_outputs()) {
                for (const auto& inputHandle : outputHandle.get_inputs()) {
                    auto outNode = inputHandle->get_node();
                    if (std::dynamic_pointer_cast<::ngraph::op::Result>(outNode)) {
                        continue;
                    }
                    VPU_THROW_FORMAT("Unsupported layer %s configuration : it is not a network output", layer->get_name());
                }
            }

            // 2. Don't support if there multiple outputs as well
            if (outputsInfo.size() != 1) {
                VPU_THROW_FORMAT("Unsupported network configuration : layer %s must be the only output of the network", layer->get_name());
            }
            model->attrs().set<bool>("withDetectionOutput", true);
        }
    }

    //
    // Gathering output shapes before reshaping.
    //

    ie::ICNNNetwork::InputShapes outputShapes;

    for (const auto& pair : outputsInfo) {
        const auto ieData = pair.second;
        IE_ASSERT(ieData != nullptr);

        outputShapes[pair.first] = ieData->getDims();
    }

    //
    // Reshape the network.
    //

    env.log->trace("Reshape the network");

    ie::ResponseDesc desc;
    const auto status = network.reshape(inputShapes, &desc);

    VPU_THROW_UNLESS(
        status == ie::StatusCode::OK,
        "Failed to reshape Network: %v", desc.msg);

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
