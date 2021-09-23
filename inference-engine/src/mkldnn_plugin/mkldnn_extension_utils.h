// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Convinience wrapper class for handling MKL-DNN memory formats.
 * @file mkldnn_extension_utils.h
 */
#pragma once

#include <string>

#include "memory_desc/cpu_memory_desc.h"
#include "mkldnn.hpp"

namespace MKLDNNPlugin {

class DnnlMemoryDesc;

class MKLDNNExtensionUtils {
public:
    static uint8_t sizeOfDataType(mkldnn::memory::data_type dataType);
    static mkldnn::memory::data_type IEPrecisionToDataType(const InferenceEngine::Precision& prec);
    static InferenceEngine::Precision DataTypeToIEPrecision(mkldnn::memory::data_type dataType);
    static Dim convertToDim(const dnnl::memory::dim& dim);
    static dnnl::memory::dim convertToDnnlDim(const Dim& dim);
    static VectorDims convertToVectorDims(const mkldnn::memory::dims& dims);
    static std::vector<dnnl::memory::dim> convertToDnnlDims(const VectorDims& dims);
    static mkldnn::memory::format_tag GetPlainFormatByRank(size_t rank);

    /**
     * @brief Creates DnnlBlockedMemoryDesc if desc is blocked, otherwise DnnlMemoryDesc
     * @param desc mkldnn::memory::desc from which one of the descriptors will be created
     * @return pointer to DnnlBlockedMemoryDesc or DnnlMemoryDesc
     */
    static std::shared_ptr<DnnlMemoryDesc> makeDescriptor(const mkldnn::memory::desc& desc);
    static size_t getMemSizeForDnnlDesc(mkldnn::memory::desc desc);
};

}  // namespace MKLDNNPlugin
