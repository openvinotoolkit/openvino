// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_extension_utils.h"
#include "utils/general_utils.h"
#include <limits>
#include <vector>
#include <numeric>

using namespace mkldnn;
using namespace MKLDNNPlugin;

uint8_t MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type dataType) {
    switch (dataType) {
    case mkldnn::memory::data_type::f32:
        return 4;
    case mkldnn::memory::data_type::s32:
        return 4;
    case mkldnn::memory::data_type::bf16:
        return 2;
    case mkldnn::memory::data_type::s8:
        return 1;
    case mkldnn::memory::data_type::u8:
        return 1;
    case mkldnn::memory::data_type::bin:
        return 1;
    case mkldnn::memory::data_type::undef:
        return 0;
    default:
        IE_THROW() << "Unsupported data type.";
    }
}

memory::data_type MKLDNNExtensionUtils::IEPrecisionToDataType(const InferenceEngine::Precision& prec) {
    switch (prec) {
        case InferenceEngine::Precision::FP32:
            return memory::data_type::f32;
        case InferenceEngine::Precision::I32:
            return memory::data_type::s32;
        case InferenceEngine::Precision::BF16:
            return memory::data_type::bf16;
        case InferenceEngine::Precision::I8:
            return memory::data_type::s8;
        case InferenceEngine::Precision::U8:
        case InferenceEngine::Precision::BOOL:
            return memory::data_type::u8;
        case InferenceEngine::Precision::BIN:
            return memory::data_type::bin;
        case InferenceEngine::Precision::UNSPECIFIED:
            return memory::data_type::undef;
        default: {
            IE_THROW() << "The plugin does not support " << prec.name();
        }
    }
}

InferenceEngine::Precision MKLDNNExtensionUtils::DataTypeToIEPrecision(memory::data_type dataType) {
    switch (dataType) {
        case memory::data_type::f32:
            return InferenceEngine::Precision::FP32;
        case memory::data_type::s32:
            return InferenceEngine::Precision::I32;
        case memory::data_type::bf16:
            return InferenceEngine::Precision::BF16;
        case memory::data_type::s8:
            return InferenceEngine::Precision::I8;
        case memory::data_type::u8:
            return InferenceEngine::Precision::U8;
        case memory::data_type::bin:
            return InferenceEngine::Precision::BIN;
        case memory::data_type::undef:
            return InferenceEngine::Precision::UNSPECIFIED;
        default: {
            IE_THROW() << "Unsupported data type.";
        }
    }
}

InferenceEngine::SizeVector MKLDNNExtensionUtils::convertToSizeVector(const mkldnn::memory::dims& dims) {
    return InferenceEngine::SizeVector(dims.begin(), dims.end());
}

std::vector<dnnl::memory::dim> MKLDNNExtensionUtils::convertToDnnlDims(const InferenceEngine::SizeVector& dims) {
    return std::vector<dnnl::memory::dim>(dims.begin(), dims.end());;
}
