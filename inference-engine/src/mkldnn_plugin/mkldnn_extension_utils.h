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
#include "details/ie_pre_allocator.hpp"

namespace MKLDNNPlugin {

class MKLDNNExtensionUtils {
public:
    static uint8_t sizeOfDataType(mkldnn::memory::data_type dataType);
    static mkldnn::memory::data_type IEPrecisionToDataType(InferenceEngine::Precision prec);
    static InferenceEngine::Precision DataTypeToIEPrecision(mkldnn::memory::data_type dataType);
    static InferenceEngine::TensorDesc getUninitTensorDesc(const InferenceEngine::TensorDesc& desc);
    static bool initTensorsAreEqual(const InferenceEngine::TensorDesc &desc1, const InferenceEngine::TensorDesc &desc2);
    static std::string getReorderArgs(const InferenceEngine::TensorDesc &parentDesc, const InferenceEngine::TensorDesc &childDesc);
    static size_t getRealElementCount(const InferenceEngine::TensorDesc desc);
    static bool isZeroOffsetDataPadding(const InferenceEngine::TensorDesc &desc);
    static bool isDefaultStrides(const InferenceEngine::TensorDesc &desc);
    static std::tuple<std::shared_ptr<InferenceEngine::IAllocator>, uint8_t *> allocRealSizeMem(const InferenceEngine::TensorDesc &desc,
                                                                                                const InferenceEngine::Precision prec);
};

}  // namespace MKLDNNPlugin
