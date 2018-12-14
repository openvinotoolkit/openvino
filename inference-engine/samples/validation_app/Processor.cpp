// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <algorithm>

#include <samples/common.hpp>

#include "Processor.hpp"

using namespace InferenceEngine;

Processor::Processor(const std::string& flags_m, const std::string& flags_d, const std::string& flags_i, int flags_b,
        InferencePlugin plugin, CsvDumper& dumper, const std::string& approach, PreprocessingOptions preprocessingOptions)

    : targetDevice(flags_d), modelFileName(flags_m), imagesPath(flags_i), batch(flags_b),
      plugin(plugin), dumper(dumper), approach(approach), preprocessingOptions(preprocessingOptions) {

    // --------------------Load network (Generated xml/bin files)-------------------------------------------
    slog::info << "Loading network files" << slog::endl;

    loadDuration = getDurationOf([&]() {
        /** Read network model **/
        networkReader.ReadNetwork(modelFileName);
        if (!networkReader.isParseSuccess()) THROW_IE_EXCEPTION << "cannot load a failed Model";

        /** Extract model name and load weights **/
        std::string binFileName = fileNameNoExt(modelFileName) + ".bin";
        networkReader.ReadWeights(binFileName.c_str());
    });
    // -----------------------------------------------------------------------------------------------------

    // -----------------------------Prepare input blobs-----------------------------------------------------
    slog::info << "Preparing input blobs" << slog::endl;

    /** Taking information about all topology inputs **/
    inputInfo = InputsDataMap(networkReader.getNetwork().getInputsInfo());

    /** Stores all input blobs data **/

    // TODO Check if it's necessary
    if (!targetDevice.empty()) {
        networkReader.getNetwork().setTargetDevice(getDeviceFromStr(targetDevice));
    }

    if (batch == 0) {
        // Zero means "take batch value from the IR"
        batch = networkReader.getNetwork().getBatchSize();
    } else {
        // Not zero means "use the specified value"
        networkReader.getNetwork().setBatchSize(batch);
    }

    if (inputInfo.size() != 1) {
        THROW_IE_EXCEPTION << "This app accepts networks having only one input";
    }

    for (auto & item : inputInfo) {
        inputDims = item.second->getDims();
        slog::info << "Batch size is " << std::to_string(networkReader.getNetwork().getBatchSize()) << slog::endl;
    }

    outInfo = networkReader.getNetwork().getOutputsInfo();
    DataPtr outData = outInfo.begin()->second;

    // Set the precision of output data provided by the user, should be called before load of the network to the plugin
    if (!outData) {
        throw std::logic_error("output data pointer is not valid");
    }
    outData->setPrecision(Precision::FP32);
    if (outInfo.size() != 1) {
        THROW_IE_EXCEPTION << "This app accepts networks having only one output";
    }
    if (!outData) {
        THROW_IE_EXCEPTION << "The network output info is not valid";
    }

    outputDims = outData->dims;

    // Load model to plugin and create an inference request

    ExecutableNetwork executable_network = plugin.LoadNetwork(networkReader.getNetwork(), {});
    inferRequest = executable_network.CreateInferRequest();
}

double Processor::Infer(ConsoleProgress& progress, int filesWatched, InferenceMetrics& im) {
    ResponseDesc dsc;

    // InferencePlugin plugin(enginePtr);

    // Infer model
    double time = getDurationOf([&]() {
        inferRequest.Infer();
    });

    im.maxDuration = std::min(im.maxDuration, time);
    im.minDuration = std::max(im.minDuration, time);
    im.totalTime += time;
    im.nRuns++;

    progress.addProgress(filesWatched);

    return time;
}
