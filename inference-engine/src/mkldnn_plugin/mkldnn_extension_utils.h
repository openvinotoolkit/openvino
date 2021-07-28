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
#include "mkldnn_memory.h"

namespace MKLDNNPlugin {


/**
 * Partial tensor descriptor
 *
 * Represent a classes of layout. As example Plain, TailC, CBlocked and other.
 *
 * The tensor are in one layout family if they have same PartialBlkDesc.
 *
 * Any tensor will have same PartialBlkDesc as it subview tensor.
 *
 * PartialBlkDesc plus Dims allow to reconstruct real tensorDesc (dense representation).
 */
class PartialBlkDesc {
public:
    /**
     * Check if this partial blocking desc will lead to additional zero padding
     * for real tensor with provided dims
     *
     * Example: dims [2, 3, 8, 8] with blocking by 16 for second dim. Will lead
     *          to effective dims [2, 16, 8, 8] with zeroing all values
     *          [:, 3:16, :, :]
     *
     * @param dims to check on zero auto padding
     * @return true if provided dims will use auto padding. Otherwise false.
     */
    bool isAutoExtendedWith(const InferenceEngine::SizeVector &dims) const;

    /**
     * Construct PartialBlkDesc from provided TensorDesc
     *
     * PartialBlkDesc has less expressiveness power so some information from TensorDesc will be dropped.
     * The different TensorDesc object will has equal PartialBlkDesc.
     *
     * @param desc to extract PartialBlkDesc information about kind of layout
     * @return PartialBlkDesc object corresponds layout described in desc
     */
    static PartialBlkDesc extractFrom(const InferenceEngine::TensorDesc &desc);

    /** Construct plain PartialBlkDesc based on dims information */
    static PartialBlkDesc makePlain(const InferenceEngine::SizeVector &dims);

    /** Construct blocked Channel PartialBlkDesc based on dims information */
    static PartialBlkDesc makeCBlocked(const InferenceEngine::SizeVector &dims, size_t block_size);

    /** Construct per Channel PartialBlkDesc based on dims information */
    static PartialBlkDesc makeTailC(const InferenceEngine::SizeVector &dims);

    /** Compare operators. Allow to use it as key for std::map */
    bool operator == (const PartialBlkDesc& it) const;
    bool operator < (const PartialBlkDesc& it) const;

private:
    PartialBlkDesc() = default;
    InferenceEngine::SizeVector outer_order;
    InferenceEngine::SizeVector inner_blk_size;
    InferenceEngine::SizeVector inner_blk_idxes;
};

class MKLDNNExtensionUtils {
public:
    static uint8_t sizeOfDataType(mkldnn::memory::data_type dataType);
    static mkldnn::memory::data_type IEPrecisionToDataType(InferenceEngine::Precision prec);
    static InferenceEngine::Precision DataTypeToIEPrecision(mkldnn::memory::data_type dataType);
    static InferenceEngine::TensorDesc getUninitTensorDesc(const InferenceEngine::TensorDesc& desc);
    static bool initTensorsAreEqual(const InferenceEngine::TensorDesc &desc1, const InferenceEngine::TensorDesc &desc2);
    static std::string getReorderArgs(const InferenceEngine::TensorDesc &parentDesc, const InferenceEngine::TensorDesc &childDesc);
    static InferenceEngine::Precision getMaxPrecision(std::vector<InferenceEngine::Precision> precisions);
};

}  // namespace MKLDNNPlugin
