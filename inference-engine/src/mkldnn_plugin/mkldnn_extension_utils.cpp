// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_extension_utils.h"
#include <limits>
#include <vector>

using namespace mkldnn;
using namespace MKLDNNPlugin;

uint8_t MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type dataType) {
    switch (dataType) {
    case mkldnn::memory::data_type::f32:
        return 4;
    case mkldnn::memory::data_type::s32:
        return 4;
    case mkldnn::memory::data_type::s16:
        return 2;
    case mkldnn::memory::data_type::s8:
        return 1;
    case mkldnn::memory::data_type::u8:
        return 1;
    case mkldnn::memory::data_type::data_undef:
        return 0;
    default:
        THROW_IE_EXCEPTION << "Unsupported data type.";
    }
}

memory::data_type MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision prec) {
    switch (prec) {
        case InferenceEngine::Precision::FP32:
            return memory::f32;
        case InferenceEngine::Precision::I32:
            return memory::s32;
        case InferenceEngine::Precision::I16:
            return memory::s16;
        case InferenceEngine::Precision::I8:
            return memory::s8;
        case InferenceEngine::Precision::U8:
            return memory::u8;

        default: {
            THROW_IE_EXCEPTION << "The plugin does not support " << prec.name();
        }
    }
}

InferenceEngine::Precision MKLDNNExtensionUtils::DataTypeToIEPrecision(memory::data_type dataType) {
    switch (dataType) {
        case memory::f32:
            return InferenceEngine::Precision(InferenceEngine::Precision::FP32);
        case memory::s32:
            return InferenceEngine::Precision::I32;
        case memory::s16:
            return InferenceEngine::Precision::I16;
        case memory::s8:
            return InferenceEngine::Precision::I8;
        case memory::u8:
            return InferenceEngine::Precision::U8;

        default: {
            THROW_IE_EXCEPTION << "Unsupported data type.";
        }
    }
}

InferenceEngine::TensorDesc MKLDNNExtensionUtils::getUninitTensorDesc(const InferenceEngine::TensorDesc &desc) {
    std::vector<size_t> notInitArr;
    std::vector<size_t> zeroArr;
    for (size_t i = 0; i < desc.getBlockingDesc().getBlockDims().size(); i++) {
        notInitArr.push_back(std::numeric_limits<size_t>::max());
        zeroArr.push_back(0);
    }
    // MKLDNN doesn't support offset_padding_to_data[i] != 0 (assert(src_d_blk.offset_padding_to_data[d] == 0);)
    return desc.getLayout() == InferenceEngine::Layout::ANY ? desc :
           InferenceEngine::TensorDesc(desc.getPrecision(), desc.getDims(),
                                       {desc.getBlockingDesc().getBlockDims(), desc.getBlockingDesc().getOrder(),
                                        std::numeric_limits<size_t>::max(), zeroArr, notInitArr});
}

bool MKLDNNExtensionUtils::initTensorsAreEqual(InferenceEngine::TensorDesc desc1, InferenceEngine::TensorDesc desc2) {
    if (desc1.getDims() != desc2.getDims() || desc1.getPrecision() != desc2.getPrecision())
        return false;
    if (desc1.getLayout() == InferenceEngine::Layout::ANY || desc2.getLayout() == InferenceEngine::Layout::ANY)
        return true;
    bool batch1 = desc1.getDims()[0] == 1;
    const auto& in1Block = desc1.getBlockingDesc();
    const auto& in2Block = desc2.getBlockingDesc();
    size_t uninitNum = std::numeric_limits<size_t>::max();
    if (in1Block.getBlockDims().size() != in2Block.getBlockDims().size())
        return false;
    for (size_t i = 0; i < in1Block.getBlockDims().size(); i++) {
        if (in1Block.getBlockDims()[i] != in2Block.getBlockDims()[i] &&
                in1Block.getBlockDims()[i] != uninitNum && in2Block.getBlockDims()[i] != uninitNum)
            return false;
        if (in1Block.getOffsetPaddingToData()[i] != in2Block.getOffsetPaddingToData()[i] &&
                in1Block.getOffsetPaddingToData()[i] != uninitNum && in2Block.getOffsetPaddingToData()[i] != uninitNum)
            return false;
        if (i >= batch1 && in1Block.getStrides()[i] != in2Block.getStrides()[i] &&
                in1Block.getStrides()[i] != uninitNum && in2Block.getStrides()[i] != uninitNum)
            return false;
        if (in1Block.getOrder()[i] != in2Block.getOrder()[i] &&
                in1Block.getOrder()[i] != uninitNum && in2Block.getOrder()[i] != uninitNum)
            return false;
    }
    return !(in1Block.getOffsetPadding() != in2Block.getOffsetPadding() &&
        in1Block.getOffsetPadding() != uninitNum && in2Block.getOffsetPadding() != uninitNum);
}
