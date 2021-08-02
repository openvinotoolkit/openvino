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
    static InferenceEngine::SizeVector convertToSizeVector(const mkldnn::memory::dims& dims);
    static std::vector<dnnl::memory::dim> convertToDnnlDims(const InferenceEngine::SizeVector& dims);
};

}  // namespace MKLDNNPlugin
