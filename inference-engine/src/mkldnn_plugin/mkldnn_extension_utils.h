// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Convinience wrapper class for handling MKL-DNN memory formats.
 * @file mkldnn_extension_utils.h
 */
#pragma once

#include <string>

#include "mkldnn.hpp"
#include "cpu_memory_desc.h"

namespace MKLDNNPlugin {

class MKLDNNExtensionUtils {
public:
    static uint8_t sizeOfDataType(mkldnn::memory::data_type dataType);
    static mkldnn::memory::data_type IEPrecisionToDataType(const InferenceEngine::Precision& prec);
    static InferenceEngine::Precision DataTypeToIEPrecision(mkldnn::memory::data_type dataType);
    static std::string getReorderArgs(const MemoryDesc &parentDesc, const MemoryDesc &childDesc);

    //TODO : move to common utils
    static InferenceEngine::Precision getMaxPrecision(std::vector<InferenceEngine::Precision> precisions);

    // TODO [DS]: remove
    static InferenceEngine::TensorDesc getUninitTensorDesc(const InferenceEngine::TensorDesc& desc);
    static bool initTensorsAreEqual(const InferenceEngine::TensorDesc &desc1, const InferenceEngine::TensorDesc &desc2);
};

}  // namespace MKLDNNPlugin
