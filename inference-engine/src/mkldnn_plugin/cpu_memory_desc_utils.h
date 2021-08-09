// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_layouts.h>
#include <ie_blob.h>

namespace MKLDNNPlugin {
class MKLDNNMemoryDesc;
class BlockedMemoryDesc;
class MKLDNNMemory;

class MemoryDescUtils {
public:
    /**
     * @brief Converts MemoryDesc to InferenceEngine::TensorDesc
     * @param desc MemoryDesc to be converted
     * @return converted InferenceEngine::TensorDesc
     */
    static InferenceEngine::TensorDesc convertToTensorDesc(const MemoryDesc& desc);

    /**
     * @brief Converts MemoryDesc to MKLDNNMemoryDesc
     * @param desc MemoryDesc to be converted
     * @return converted MKLDNNMemoryDesc
     */
    static MKLDNNMemoryDesc convertToMKLDNNMemoryDesc(const MemoryDesc& desc);

    /**
     * @brief Converts BlockedMemoryDesc to MKLDNNMemoryDesc
     * @param desc BlockedMemoryDesc to be converted
     * @return converted MKLDNNMemoryDesc
     */
    static MKLDNNMemoryDesc convertToMKLDNNMemoryDesc(const BlockedMemoryDesc& desc);

    /**
     * @brief Converts InferenceEngine::TensorDesc to MKLDNNMemoryDesc
     * @param desc InferenceEngine::TensorDesc to be converted
     * @return converted MKLDNNMemoryDesc
     */
    static MKLDNNMemoryDesc convertToMKLDNNMemoryDesc(const InferenceEngine::TensorDesc& desc);

    /**
     * @brief Converts MemoryDesc to BlockedMemoryDesc
     * @param desc MemoryDesc to be converted
     * @return converted BlockedMemoryDesc
     */
    static BlockedMemoryDesc convertToBlockedDescriptor(const MemoryDesc& desc);

    /**
     * @brief Converts MKLDNNMemoryDesc to BlockedMemoryDesc
     * @param desc MKLDNNMemoryDesc to be converted
     * @return converted BlockedMemoryDesc
     */
    static BlockedMemoryDesc convertToBlockedDescriptor(const MKLDNNMemoryDesc& inpDesc);

    /**
     * @brief Creates MKLDNNMemoryDesc with offset0 of UNDEFINED_DIM size
     * @param desc modifiable MKLDNNMemoryDesc
     * @return pointer to MKLDNNMemoryDesc
     */
    static MemoryDescPtr applyUndefinedOffset(const MKLDNNMemoryDesc& desc);

    /**
     * @brief Creates BlockedMemoryDesc with offsetPadding, strides of UNDEFINED_DIM size and offsetPaddingToData of 0 size
     * @param desc modifiable BlockedMemoryDesc
     * @return pointer to BlockedMemoryDesc
     */
    static MemoryDescPtr applyUndefinedOffset(const BlockedMemoryDesc& desc);

    /**
     * @brief Creates MemoryDesc with offsetPadding of 0 size
     * @param desc modifiable MemoryDesc
     * @return pointer to MemoryDesc
     */
    static MemoryDescPtr resetOffset(const MemoryDesc* desc);

    /**
     * @brief Creates InferenceEngine::Blob from MKLDNNMemory
     * @param desc MKLDNNMemory from which will be created InferenceEngine::Blob
     * @return pointer to InferenceEngine::Blob
     */
    static InferenceEngine::Blob::Ptr interpretAsBlob(const MKLDNNMemory& mem);
};

}  // namespace MKLDNNPlugin
