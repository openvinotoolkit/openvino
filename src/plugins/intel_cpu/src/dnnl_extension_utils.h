// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Convinience wrapper class for handling OneDNN memory formats.
 * @file dnnl_extension_utils.h
 */
#pragma once

#include <string>

#include "onednn/dnnl.h"
#include "memory_desc/cpu_memory_desc.h"

namespace ov {
namespace intel_cpu {

class DnnlMemoryDesc;

class DnnlExtensionUtils {
public:
    static uint8_t sizeOfDataType(dnnl::memory::data_type dataType);
    static dnnl::memory::data_type IEPrecisionToDataType(const InferenceEngine::Precision& prec);
    static InferenceEngine::Precision DataTypeToIEPrecision(dnnl::memory::data_type dataType);
    static Dim convertToDim(const dnnl::memory::dim &dim);
    static dnnl::memory::dim convertToDnnlDim(const Dim &dim);
    static VectorDims convertToVectorDims(const dnnl::memory::dims& dims);
    static std::vector<dnnl::memory::dim> convertToDnnlDims(const VectorDims& dims);
    static dnnl::memory::format_tag GetPlainFormatByRank(size_t rank);

    /**
     * @brief Creates DnnlBlockedMemoryDesc if desc is blocked, otherwise DnnlMemoryDesc
     * @param desc dnnl::memory::desc from which one of the descriptors will be created
     * @return pointer to DnnlBlockedMemoryDesc or DnnlMemoryDesc
     */
    static std::shared_ptr<DnnlMemoryDesc> makeDescriptor(const dnnl::memory::desc &desc);

    /**
     * @brief Helper function that creates DnnlBlockedMemoryDesc from defined dnnl::memory::desc and undefined shape.
     * It uses desc as an basis for the new undefined one. Specifically, type, layout, precision, blocks, extra data will be preserved.
     * @param desc dnnl::memory::desc dnnl desc which will be used as a basis of the new descriptor
     * @param shape a new undefined shape
     * @return pointer to the created DnnlBlockedMemoryDesc
     * @note Obly blocked descriptors are allowed at the moment
     */

    static std::shared_ptr<DnnlBlockedMemoryDesc> makeUndefinedDesc(const dnnl::memory::desc &desc, const Shape& shape);
    static size_t getMemSizeForDnnlDesc(const dnnl::memory::desc& desc);
};

}   // namespace intel_cpu
}   // namespace ov
