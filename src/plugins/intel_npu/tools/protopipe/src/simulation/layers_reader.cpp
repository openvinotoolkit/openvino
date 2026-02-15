//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simulation/layers_reader.hpp"
#include "scenario/inference.hpp"
#include "utils/error.hpp"
#include "utils/logger.hpp"

OpenVINOLayersReader& getOVReader() {
    static OpenVINOLayersReader reader;
    return reader;
}

static std::string getModelFileName(const InferenceParams& params) {
    if (std::holds_alternative<OpenVINOParams>(params)) {
        const auto& ov_params = std::get<OpenVINOParams>(params);
        if (std::holds_alternative<OpenVINOParams::ModelPath>(ov_params.path)) {
            return std::get<OpenVINOParams::ModelPath>(ov_params.path).model;
        } else {
            ASSERT(std::holds_alternative<OpenVINOParams::BlobPath>(ov_params.path));
            return std::get<OpenVINOParams::BlobPath>(ov_params.path).blob;
        }
    } else if (std::holds_alternative<ONNXRTParams>(params)) {
        return std::get<ONNXRTParams>(params).model_path;
    } else {
        THROW_ERROR("Unsupported model parameters type!");
    }
    // NB: Unreachable
    ASSERT(false);
}

InOutLayers LayersReader::readLayers(const InferenceParams& params) {
    LOG_INFO() << "Reading model " << getModelFileName(params) << std::endl;
    if (std::holds_alternative<OpenVINOParams>(params)) {
        const auto& ov = std::get<OpenVINOParams>(params);
        return getOVReader().readLayers(ov);
    }
    ASSERT(std::holds_alternative<ONNXRTParams>(params));
    const auto& ort = std::get<ONNXRTParams>(params);
    // NB: Using OpenVINO to read the i/o layers information for *.onnx model
    OpenVINOParams ov;
    ov.path = OpenVINOParams::ModelPath{ort.model_path, ""};
    return getOVReader().readLayers(ov, true /* use_results_names */);
}
