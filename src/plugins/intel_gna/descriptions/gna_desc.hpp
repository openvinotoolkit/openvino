// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <numeric>
#include <functional>

#include "ie_precision.hpp"
#include "ie_input_info.hpp"
#include "ie_algorithm.hpp"

#include "backend/dnn_types.h"
#include "gna_plugin_config.hpp"

namespace GNAPluginNS {

struct GnaDesc {
    // common OV properties
    std::string friendly_name;
    std::string tensor_name;
    InferenceEngine::Layout layout;
    InferenceEngine::SizeVector dims;
    InferenceEngine::Precision precision;
    InferenceEngine::Precision blob_precision;

    // gna specific properties
    double scale_factor = GNAPluginNS::kScaleFactorDefault;
    intel_dnn_orientation_t orientation = kDnnUnknownOrientation;
    uint32_t num_bytes_per_element = 0;
    uint32_t num_elements = 0;
    uint32_t allocated_size = 0;
    std::vector<void *> ptrs = {};  // ptr per each infer request

    // help methods
    uint32_t getRequiredSize() {
        return num_elements * num_bytes_per_element;
    }

    uint32_t getAllocatedSize() {
        return allocated_size;
    }

    void setPrecision(InferenceEngine::Precision precision) {
        this->precision = precision;
        this->num_bytes_per_element = precision.size();
    }
};

struct InputDesc : GnaDesc {
    InputDesc() {}
    InputDesc(const InferenceEngine::InputInfo::Ptr inputInfo) {
        this->precision = inputInfo->getPrecision();
        this->blob_precision = inputInfo->getPrecision();
        this->layout = inputInfo->getLayout();
        this->dims = inputInfo->getTensorDesc().getDims();
        this->friendly_name = inputInfo->name();
        this->tensor_name = inputInfo->name();
        this->num_bytes_per_element = precision.size();
        this->num_elements = InferenceEngine::details::product(dims.begin(), dims.end());
    }
};

struct OutputDesc : GnaDesc {
    OutputDesc() {}
    OutputDesc(const InferenceEngine::DataPtr outputData) {
        this->precision = outputData->getPrecision();
        this->blob_precision = outputData->getPrecision();
        this->layout = outputData->getLayout();
        this->dims = outputData->getTensorDesc().getDims();
        this->friendly_name = outputData->getName();
        this->tensor_name = outputData->getName();
        this->num_bytes_per_element = precision.size();
        this->num_elements = InferenceEngine::details::product(dims.begin(), dims.end());
    }
};

typedef std::map<std::string, InputDesc> GnaInputs;
typedef std::map<std::string, OutputDesc> GnaOutputs;

}  // namespace GNAPluginNS
