// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Convinience wrapper class for handling MKL-DNN memory formats.
 * @file mkldnn_extension_utils.h
 */
#pragma once

#include <string>

#include "mkldnn.hpp"
#include "mkldnn_memory.h"

namespace MKLDNNPlugin {

namespace MKLDNNExtensionUtils {
    uint8_t sizeOfDataType(mkldnn::memory::data_type dataType);
    mkldnn::memory::data_type IEPrecisionToDataType(InferenceEngine::Precision prec);
    InferenceEngine::Precision DataTypeToIEPrecision(mkldnn::memory::data_type dataType);
    InferenceEngine::TensorDesc getUninitTensorDesc(const InferenceEngine::TensorDesc& desc);
    bool initTensorsAreEqual(const InferenceEngine::TensorDesc &desc1, const InferenceEngine::TensorDesc &desc2);
};

}  // namespace MKLDNNPlugin
