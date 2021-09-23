// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <ie_layouts.h>

#include "cpu_types.h"
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
    static std::shared_ptr<DnnlMemoryDesc> convertToDnnlMemoryDesc(const std::shared_ptr<MemoryDesc>& desc);

    /**
     * @brief Converts MemoryDesc to DnnlBlockedMemoryDesc
     * @param desc MemoryDesc to be converted
     * @return converted DnnlBockedMemoryDesc
     */
    static DnnlBlockedMemoryDesc convertToDnnlBlockedMemoryDesc(const MemoryDesc& desc);

    /**
     * @brief Converts InferenceEngine::TensorDesc to CpuBlockedMemoryDesc
     * @param desc InferenceEngine::TensorDesc to be converted
     * @return converted CpuBlockedMemoryDesc
     */
    static CpuBlockedMemoryDesc convertToCpuBlockedMemoryDesc(const InferenceEngine::TensorDesc& desc);

    /**
     * @brief Converts InferenceEngine::TensorDesc to DnnlBlockedMemoryDesc
     * @param desc InferenceEngine::TensorDesc to be converted
     * @return converted DnnlBlockedMemoryDesc
     */
    static DnnlBlockedMemoryDesc convertToDnnlBlockedMemoryDesc(const InferenceEngine::TensorDesc& desc);

    /**
     * @brief Converts MemoryDesc to BlockedMemoryDesc
     * @param desc MemoryDesc to be converted
     * @return converted BlockedMemoryDesc
     */
    static std::shared_ptr<BlockedMemoryDesc> convertToBlockedMemoryDesc(const std::shared_ptr<MemoryDesc>& desc);

    /**
     * @brief Creates BlockedMemoryDesc with offsetPadding and strides of UNDEFINED_DIM size
     * @param desc is the MemoryDesc to be cloned
     * @return pointer to the new MemoryDesc
     */
    static std::shared_ptr<MemoryDesc> cloneWithUndefStridesAndOffset(const MemoryDesc& desc);

    /**
     * @brief Creates MemoryDesc with offsetPadding of 0 size and default strides
     * @param desc is the MemoryDesc to be cloned
     * @return pointer to the new MemoryDesc
     */
    static std::shared_ptr<MemoryDesc> cloneWithDefaultStridesAndOffset(const MemoryDesc& desc);

    /**
     * @brief Creates MemoryDesc with specified precision
     * @param desc is the MemoryDesc to be cloned
     * @return pointer to the new MemoryDesc
     */
    static std::shared_ptr<MemoryDesc> cloneWithNewPrecision(const MemoryDesc& desc,
                                                             const InferenceEngine::Precision prec);

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
    static std::string dim2str(Dim dim);

    /**
     * @brief Converts dims to string, undefined dim represented as ?
     * @param dim Dims to be converted
     * @return dims as string
     */
    static std::string dims2str(const VectorDims& dims);
};

}  // namespace MKLDNNPlugin
