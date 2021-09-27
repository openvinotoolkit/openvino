// Copyright (C) 2018-2021 Intel Corporation
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
#include <vpu/configuration/options/detect_network_batch.hpp>

namespace vpu {

void FrontEnd::detectNetworkBatch(
        ie::CNNNetwork& network,
        const Model& model) {
    VPU_PROFILE(detectNetworkBatch);
    using ShapesMap = std::map<std::string, InferenceEngine::SizeVector>;
    using PrecisionsMap = std::map<std::string, ie::Precision>;
    const auto& env = CompileEnv::get();

    if (!env.config.get<DetectNetworkBatchOption>()) {
        // skip batch extraction step and go as is
        return;
    }

    env.log->trace("Detect network batch");
    VPU_LOGGER_SECTION(env.log);

    //
    // Get information about Network inputs and outputs.
    //

    ie::InputsDataMap inputsInfo = network.getInputsInfo();
    ie::OutputsDataMap outputsInfo = network.getOutputsInfo();

    //
    // Collect input shapes and remove batch from them.
    //

    env.log->trace("Remove batch from inputs");

    size_t inputBatch = 0;

    for (const auto& inputInfo : inputsInfo) {
        VPU_LOGGER_SECTION(env.log);
        const auto info = inputInfo.second;
        IE_ASSERT(info != nullptr);

        const auto ieData = info->getInputData();
        IE_ASSERT(ieData != nullptr);
        auto ieShapes = ieData->getTensorDesc().getDims();
        env.log->trace("Input [%s] : %v", inputInfo.first, ieShapes);
        // assume only 4D and 5D inputs have batch
        if (ieShapes.size() == 4 || ieShapes.size() == 5) {
            auto batch = ieShapes[0];
            if (!inputBatch)
                inputBatch = batch;

            if (inputBatch != batch) {
                env.log->trace("Network input with name %s has different batch dim compared to previous inputs, aborting batch detection", inputInfo.first);
                return;
            }
        }
    }

    if (inputBatch == 1) {
        env.log->trace("Network is batch 1. Not need to reshape");
        return;
    }

    if (inputBatch == 0) {
        env.log->trace("Unable to decide on network batch size.");
        return;
    }

    ShapesMap inputShapes;

    for (const auto& inputInfo : inputsInfo) {
        const auto info = inputInfo.second;
        const auto ieData = info->getInputData();
        auto ieShapes = ieData->getTensorDesc().getDims();
        if (ieShapes[0] != inputBatch) {
            env.log->trace("Network input with name %s has different batch dim compared to previous inputs, aborting batch detection", inputInfo.first);
            return;
        }
        ieShapes[0] = 1;
        inputShapes[ieData->getName()] = ieShapes;
    }

    model->setBatchSize(checked_cast<int>(inputBatch));

    //
    // Special case for DetectionOutput.
    //

    const auto operations = network.getFunction()->get_ordered_ops();
    for (const auto& op : operations) {
        if (!ngraph::as_type_ptr<ngraph::op::DetectionOutput>(op))
            continue;
        env.log->trace("Found DetectionOutput layer [%s]", op->get_name());

        VPU_THROW_UNLESS(op->get_output_size() == 1, "Layer {} with type {} must have only 1 output but {} provided",
                         op->get_name(), op->get_type_name(), op->get_output_size());
        model->attrs().set<bool>("withDetectionOutput", true);
    }

    //
    // Gathering output shapes and precisions before reshaping.
    //

    ShapesMap outputShapes;
    PrecisionsMap outputsPresisions;

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

    network.reshape(inputShapes);

    //
    // Checks outputs that doesn't change their shape.
    //

    env.log->trace("Checks for unbatched outputs");

    outputsInfo = network.getOutputsInfo();

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
