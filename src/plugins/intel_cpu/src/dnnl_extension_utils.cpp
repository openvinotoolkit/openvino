// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_extension_utils.h"

#include "utils/general_utils.h"
#include <oneapi/dnnl/dnnl.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "onednn/iml_type_mapper.h"
#include <common/primitive_desc.hpp>
#include <common/primitive_desc_iface.hpp>

#include <vector>

using namespace dnnl;

namespace ov {
namespace intel_cpu {

uint8_t DnnlExtensionUtils::sizeOfDataType(dnnl::memory::data_type dataType) {
    switch (dataType) {
    case dnnl::memory::data_type::f32:
    case dnnl::memory::data_type::s32:
        return 4;
    case dnnl::memory::data_type::bf16:
    case dnnl::memory::data_type::f16:
        return 2;
    case dnnl::memory::data_type::s8:
    case dnnl::memory::data_type::u8:
    case dnnl::memory::data_type::bin:
    case dnnl::memory::data_type::nf4:
    case dnnl::memory::data_type::s4:
    case dnnl::memory::data_type::u4:
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
        case InferenceEngine::Precision::FP16:
            return memory::data_type::f16;
        case InferenceEngine::Precision::NF4:
            return memory::data_type::nf4;
        case InferenceEngine::Precision::I4:
            return memory::data_type::s4;
        case InferenceEngine::Precision::U4:
            return memory::data_type::u4;
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
        case memory::data_type::f16:
            return InferenceEngine::Precision::FP16;
        case memory::data_type::nf4:
            return InferenceEngine::Precision::NF4;
        case memory::data_type::s4:
            return InferenceEngine::Precision::I4;
        case memory::data_type::u4:
            return InferenceEngine::Precision::U4;
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
    std::vector<size_t> vecResult(dims.size());
    std::transform(dims.begin(), dims.end(), vecResult.begin(), convertToDim);
    return vecResult;
}

VectorDims DnnlExtensionUtils::convertToVectorDims(const dnnl::impl::dims_t dims, const int ndims) {
    return VectorDims(dims, dims + ndims);
}

memory::dims DnnlExtensionUtils::convertToDnnlDims(const VectorDims& dims) {
    memory::dims vecResult(dims.size());
    std::transform(dims.begin(), dims.end(), vecResult.begin(), convertToDnnlDim);
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
    return makeDescriptor(desc.get());
}

DnnlMemoryDescPtr DnnlExtensionUtils::makeDescriptor(const_dnnl_memory_desc_t desc) {
    if (desc->format_kind == dnnl::impl::format_kind_t::dnnl_blocked) {
        return std::shared_ptr<DnnlBlockedMemoryDesc>(new DnnlBlockedMemoryDesc(desc));
    } else {
        return std::shared_ptr<DnnlMemoryDesc>(new DnnlMemoryDesc(desc));
    }
}

size_t DnnlExtensionUtils::getMemSizeForDnnlDesc(const dnnl::memory::desc& desc) {
    auto tmpDesc = desc;

    const auto offset0 = tmpDesc.get()->offset0;
    tmpDesc.get()->offset0 = 0;

    size_t size = tmpDesc.get_size();
    if (size == DNNL_RUNTIME_SIZE_VAL)
        return MemoryDesc::UNDEFINED_SIZE;

    size += offset0 * sizeOfDataType(tmpDesc.get_data_type());
    return size;
}

std::shared_ptr<DnnlBlockedMemoryDesc> DnnlExtensionUtils::makeUndefinedDesc(const memory::desc &desc, const Shape &shape) {
    if (desc.get_format_kind() == memory::format_kind::blocked) {
        return std::shared_ptr<DnnlBlockedMemoryDesc>(new DnnlBlockedMemoryDesc(desc, shape));
    } else {
        IE_THROW(Unexpected) << "Cannot make undefined descriptor. Only dnnl_blocked type is allowed.";
    }
}

DnnlMemoryDescPtr DnnlExtensionUtils::query_md(const const_dnnl_primitive_desc_t& pd, const dnnl::query& what, int idx) {
    auto query = dnnl::convert_to_c(what);
    const auto* cdesc = dnnl_primitive_desc_query_md(pd, query, idx);

    if (!cdesc)
        IE_THROW() << "query_md failed for query=" << query << " idx=" << idx << ".";

    return DnnlExtensionUtils::makeDescriptor(cdesc);
}

std::string DnnlExtensionUtils::query_impl_info_str(const const_dnnl_primitive_desc_t& pd) {
    const char *res;
    dnnl_status_t status = dnnl_primitive_desc_query(pd, dnnl_query_impl_info_str, 0, &res);
    if (status != dnnl_success)
        IE_THROW() << "query_impl_info_str failed.";
    return std::string(res);
}

bool DnnlExtensionUtils::find_implementation(dnnl::primitive_desc& desc, impl_desc_type impl_type) {
    return DnnlExtensionUtils::find_implementation(desc,
                                                   [impl_type](impl_desc_type cur_impl_type){
                                                       return cur_impl_type == impl_type;
                                                   });
}

dnnl_memory_desc_t DnnlExtensionUtils::clone_desc(const_dnnl_memory_desc_t cdesc) {
    dnnl_memory_desc_t cloned_md = nullptr;
    dnnl_memory_desc_clone(&cloned_md, cdesc);
    return cloned_md;
}

dnnl_primitive_desc_t DnnlExtensionUtils::clone_primitive_desc(const_dnnl_primitive_desc_t cprim_desc) {
    dnnl_primitive_desc_t cloned_md = nullptr;
    dnnl_primitive_desc_clone(&cloned_md, cprim_desc);
    return cloned_md;
}

const char* DnnlExtensionUtils::query_pd_info(const_dnnl_primitive_desc_t pd) {
    return pd->info();
}

dnnl::algorithm DnnlExtensionUtils::convertToDnnlAlgorithm(Algorithm alg) {
    switch (alg) {
        case Algorithm::EltwiseRelu: return dnnl::algorithm::eltwise_relu;
        case Algorithm::EltwiseTanh: return dnnl::algorithm::eltwise_tanh;
        case Algorithm::EltwiseElu: return dnnl::algorithm::eltwise_elu;
        case Algorithm::EltwiseAbs: return dnnl::algorithm::eltwise_abs;
        case Algorithm::EltwiseSqrt: return dnnl::algorithm::eltwise_sqrt;
        case Algorithm::EltwiseSwish: return dnnl::algorithm::eltwise_swish;
        case Algorithm::EltwiseHswish: return dnnl::algorithm::eltwise_hardswish;
        case Algorithm::EltwiseSoftRelu: return dnnl::algorithm::eltwise_soft_relu;
        case Algorithm::EltwiseMish: return dnnl::algorithm::eltwise_mish;
        case Algorithm::EltwiseExp: return dnnl::algorithm::eltwise_exp;
        case Algorithm::EltwiseGeluErf: return dnnl::algorithm::eltwise_gelu_erf;
        case Algorithm::EltwiseGeluTanh: return dnnl::algorithm::eltwise_gelu_tanh;
        case Algorithm::EltwiseSigmoid: return dnnl::algorithm::eltwise_logistic;
        case Algorithm::EltwiseClamp: return dnnl::algorithm::eltwise_clip;
        case Algorithm::EltwisePowerStatic: return dnnl::algorithm::eltwise_pow;
        case Algorithm::EltwiseHsigmoid: return dnnl::algorithm::eltwise_hsigmoid;
        case Algorithm::EltwiseRoundHalfToEven: return dnnl::algorithm::eltwise_round_half_to_even;
        case Algorithm::EltwiseRoundHalfAwayFromZero: return dnnl::algorithm::eltwise_round_half_away_from_zero;
        case Algorithm::EltwiseAdd: return dnnl::algorithm::binary_add;
        case Algorithm::EltwiseMultiply: return dnnl::algorithm::binary_mul;
        case Algorithm::EltwiseSubtract: return dnnl::algorithm::binary_sub;
        case Algorithm::EltwiseDivide: return dnnl::algorithm::binary_div;
        case Algorithm::EltwiseMaximum: return dnnl::algorithm::binary_max;
        case Algorithm::EltwiseMinimum: return dnnl::algorithm::binary_min;
        case Algorithm::EltwiseEqual: return dnnl::algorithm::binary_eq;
        case Algorithm::EltwiseNotEqual: return dnnl::algorithm::binary_ne;
        case Algorithm::EltwiseGreater: return dnnl::algorithm::binary_gt;
        case Algorithm::EltwiseGreaterEqual: return dnnl::algorithm::binary_ge;
        case Algorithm::EltwiseLess: return dnnl::algorithm::binary_lt;
        case Algorithm::EltwiseLessEqual: return dnnl::algorithm::binary_le;
        case Algorithm::EltwisePrelu: return dnnl::algorithm::binary_prelu;
        case Algorithm::ReduceMax: return dnnl::algorithm::reduction_max;
        case Algorithm::ReduceMin: return dnnl::algorithm::reduction_min;
        case Algorithm::ReduceSum: return dnnl::algorithm::reduction_sum;
        case Algorithm::ReduceMean: return dnnl::algorithm::reduction_mean;
        case Algorithm::FQCommon: return dnnl::algorithm::quantization_quantize_dequantize;
        case Algorithm::FQQuantization: return dnnl::algorithm::quantization_quantize;
        case Algorithm::FQBinarization: return dnnl::algorithm::binarization_depthwise;
        default: return dnnl::algorithm::undef;
    }
}

bool DnnlExtensionUtils::isUnarySupportedAsPostOp(Algorithm alg) {
#if defined(OV_CPU_WITH_ACL)
    return one_of(alg, Algorithm::EltwiseRelu,
                       Algorithm::EltwiseTanh,
                       Algorithm::EltwiseElu,
                       Algorithm::EltwiseAbs,
                       Algorithm::EltwiseSqrt,
                       Algorithm::EltwiseSoftRelu,
                       Algorithm::EltwiseSigmoid);
#elif defined(OPENVINO_ARCH_X86_64)
    return one_of(alg, Algorithm::EltwiseRelu,
                       Algorithm::EltwiseGeluErf,
                       Algorithm::EltwiseGeluTanh,
                       Algorithm::EltwiseElu,
                       Algorithm::EltwiseSigmoid,
                       Algorithm::EltwiseClamp,
                       Algorithm::EltwiseTanh,
                       Algorithm::EltwiseSwish,
                       Algorithm::EltwiseHswish,
                       Algorithm::EltwiseMish,
                       Algorithm::EltwiseHsigmoid,
                       Algorithm::EltwiseRoundHalfToEven,
                       Algorithm::EltwiseRoundHalfAwayFromZero,
                       Algorithm::EltwiseAbs,
                       Algorithm::EltwiseSqrt,
                       Algorithm::EltwiseSoftRelu);
#else
    return false;
#endif
}

}   // namespace intel_cpu
}   // namespace ov
