// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_layouts.h>
#include <ie_blob.h>
#include "mkldnn/ie_mkldnn.h"

namespace MKLDNNPlugin {

class MemoryDesc;
class DnnlMemoryDesc;
class BlockedMemoryDesc;
class DnnlBlockedMemoryDesc;
class CpuBlockedMemoryDesc;
class MKLDNNMemory;

class MemoryDescUtils {
public:
    /**
     * @brief Converts MemoryDesc to DnnlMemoryDesc
     * @param desc MemoryDesc to be converted
     * @return converted DnnlMemoryDesc
     */
    static std::unique_ptr<DnnlMemoryDesc> convertToDnnlMemoryDesc(const MemoryDesc& desc);

    /**
     * @brief Converts BlockedMemoryDesc to DnnlMemoryDesc
     * @param desc BlockedMemoryDesc to be converted
     * @return converted DnnlMemoryDesc
     */
    static std::unique_ptr<DnnlMemoryDesc> convertToDnnlMemoryDesc(const CpuBlockedMemoryDesc& desc);

    /**
     * @brief Converts InferenceEngine::TensorDesc to DnnlBlockedMemoryDesc
     * @param desc InferenceEngine::TensorDesc to be converted
     * @return converted DnnlBlockedMemoryDesc
     */
    static std::unique_ptr<DnnlBlockedMemoryDesc> convertToDnnlBlockedMemoryDesc(const InferenceEngine::TensorDesc& desc);

    /**
     * @brief Converts MemoryDesc to BlockedMemoryDesc
     * @param desc MemoryDesc to be converted
     * @return converted BlockedMemoryDesc
     */
    static std::unique_ptr<BlockedMemoryDesc> convertToBlockedMemoryDesc(const MemoryDesc& desc);

    /**
     * @brief Creates BlockedMemoryDesc with offsetPadding and strides of UNDEFINED_DIM size
     * @param desc is the MemoryDesc to be cloned
     * @return pointer to the new MemoryDesc
     */
    static std::unique_ptr<MemoryDesc> cloneWithUndefStridesAndOffset(const MemoryDesc& desc);

    /**
     * @brief Creates MemoryDesc with offsetPadding of 0 size and default strides
     * @param desc is the MemoryDesc to be cloned
     * @return pointer to the new MemoryDesc
     */
    static std::unique_ptr<MemoryDesc> cloneWithDefaultStridesAndOffset(const MemoryDesc* desc);

    /**
     * @brief Creates InferenceEngine::Blob from MemoryDesc
     * @param desc MemoryDesc from which will be created InferenceEngine::Blob
     * @return pointer to InferenceEngine::Blob
     */
    static InferenceEngine::Blob::Ptr createBlob(const MemoryDesc& memDesc);

    /**
     * @brief Creates InferenceEngine::Blob from MKLDNNMemory with the memory reuse
     * @param desc MKLDNNMemory from which will be created InferenceEngine::Blob
     * @return pointer to InferenceEngine::Blob
     */
    static InferenceEngine::Blob::Ptr interpretAsBlob(const MKLDNNMemory& mem);

    /**
     * @brief Converts MemoryDesc to InferenceEngine::TensorDesc
     * @param desc MemoryDesc to be converted
     * @return converted InferenceEngine::TensorDesc
     */
    static InferenceEngine::TensorDesc convertToTensorDesc(const MemoryDesc& desc);

    /**
     * @brief Converts dim to string, undefined dim represented as ?
     * @param dim Dim to be converted
     * @return dim as string
     */
    static std::string dim2str(size_t dim);

    /**
     * @brief Converts dims to string, undefined dim represented as ?
     * @param dim Dims to be converted
     * @return dims as string
     */
    static std::string dims2str(const std::vector<size_t>& dims);
};

}  // namespace MKLDNNPlugin
