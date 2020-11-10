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

class PartialBlkDesc {
public:
    bool operator == (const PartialBlkDesc& it) const;
    bool operator < (const PartialBlkDesc& it) const;

    bool isAutoExtendedWith(const InferenceEngine::SizeVector &dims) const;

    static PartialBlkDesc extractFrom(const InferenceEngine::TensorDesc &desc);
    static PartialBlkDesc makePlain(const InferenceEngine::SizeVector &dims);

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
};

}  // namespace MKLDNNPlugin
