// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_extension_utils.h"
#include "utils/general_utils.h"
#include <vector>
#include "memory_desc/dnnl_blocked_memory_desc.h"

using namespace dnnl;

namespace ov {
namespace intel_cpu {

uint8_t DnnlExtensionUtils::sizeOfDataType(dnnl::memory::data_type dataType) {
    switch (dataType) {
    case dnnl::memory::data_type::f32:
        return 4;
    case dnnl::memory::data_type::s32:
        return 4;
    case dnnl::memory::data_type::bf16:
        return 2;
    case dnnl::memory::data_type::s8:
        return 1;
    case dnnl::memory::data_type::u8:
        return 1;
    case dnnl::memory::data_type::bin:
        return 1;
    case dnnl::memory::data_type::undef:
        return 0;
    default:
        IE_THROW() << "Unsupported data type.";
    }
}

memory::data_type DnnlExtensionUtils::IEPrecisionToDataType(const InferenceEngine::Precision& prec) {
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

InferenceEngine::Precision DnnlExtensionUtils::DataTypeToIEPrecision(memory::data_type dataType) {
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

Dim DnnlExtensionUtils::convertToDim(const dnnl::memory::dim &dim) {
    return dim == DNNL_RUNTIME_DIM_VAL ?  Shape::UNDEFINED_DIM : static_cast<size_t>(dim);
}
dnnl::memory::dim DnnlExtensionUtils::convertToDnnlDim(const Dim &dim) {
    return dim == Shape::UNDEFINED_DIM ? DNNL_RUNTIME_DIM_VAL : static_cast<dnnl::memory::dim>(dim);
}

VectorDims DnnlExtensionUtils::convertToVectorDims(const memory::dims& dims) {
    std::vector<size_t> vecResult;
    vecResult.reserve(dims.size());
    std::back_insert_iterator<std::vector<size_t>> itr(vecResult);
    std::transform(dims.begin(), dims.end(), itr, convertToDim);
    return vecResult;
}

memory::dims DnnlExtensionUtils::convertToDnnlDims(const VectorDims& dims) {
    memory::dims vecResult;
    vecResult.reserve(dims.size());
    std::back_insert_iterator<memory::dims> itr(vecResult);
    std::transform(dims.begin(), dims.end(), itr, convertToDnnlDim);
    return vecResult;
}

memory::format_tag DnnlExtensionUtils::GetPlainFormatByRank(size_t rank) {
    switch (rank) {
        case 0:
        case 1:
            return memory::format_tag::a;
        case 2:
            return memory::format_tag::ab;
        case 3:
            return memory::format_tag::abc;
        case 4:
            return memory::format_tag::abcd;
        case 5:
            return memory::format_tag::abcde;
        case 6:
            return memory::format_tag::abcdef;
        default:
            return memory::format_tag::undef;
    }
}

DnnlMemoryDescPtr DnnlExtensionUtils::makeDescriptor(const dnnl::memory::desc &desc) {
    if (desc.data.format_kind == dnnl_blocked) {
        return std::shared_ptr<DnnlBlockedMemoryDesc>(new DnnlBlockedMemoryDesc(desc));
    } else {
        return std::shared_ptr<DnnlMemoryDesc>(new DnnlMemoryDesc(desc));
    }
}

size_t DnnlExtensionUtils::getMemSizeForDnnlDesc(const dnnl::memory::desc& desc) {
    auto tmpDesc = desc;
    const auto offset0 = tmpDesc.data.offset0;
    tmpDesc.data.offset0 = 0;
    size_t size = tmpDesc.get_size();
    if (size == DNNL_RUNTIME_SIZE_VAL)
        return MemoryDesc::UNDEFINED_SIZE;
    size += offset0 * sizeOfDataType(tmpDesc.data_type());
    return size;
}

std::shared_ptr<DnnlBlockedMemoryDesc> DnnlExtensionUtils::makeUndefinedDesc(const memory::desc &desc, const Shape &shape) {
    if (desc.data.format_kind == dnnl_blocked) {
        return std::shared_ptr<DnnlBlockedMemoryDesc>(new DnnlBlockedMemoryDesc(desc, shape));
    } else {
        IE_THROW(Unexpected) << "Cannot make undefined descriptor. Only dnnl_blocked type is allowed.";
    }
}

}   // namespace intel_cpu
}   // namespace ov
