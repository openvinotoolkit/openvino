// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_extension_utils.h"

#include <common/primitive_desc.hpp>
#include <common/primitive_desc_iface.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "cpu_memory.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "onednn/iml_type_mapper.h"
#include "utils/general_utils.h"

using namespace dnnl;

namespace ov::intel_cpu {

uint8_t DnnlExtensionUtils::sizeOfDataType(dnnl::memory::data_type dataType) {
    switch (dataType) {
    case dnnl::memory::data_type::f64:
        return 8;
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
    case dnnl::memory::data_type::f8_e8m0:
    case dnnl::memory::data_type::f8_e4m3:
    case dnnl::memory::data_type::f8_e5m2:
    case dnnl::memory::data_type::f4_e2m1:
        return 1;
    case dnnl::memory::data_type::undef:
        return 0;
    default:
        OPENVINO_THROW("Unsupported data type.");
    }
}

std::optional<dnnl::memory::data_type> DnnlExtensionUtils::ElementTypeToDataType(
    const ov::element::Type& elementType,
    DnnlExtensionUtils::nothrow_tag) noexcept {
    switch (elementType) {
    case ov::element::f32:
        return memory::data_type::f32;
    case ov::element::i32:
        return memory::data_type::s32;
    case ov::element::bf16:
        return memory::data_type::bf16;
    case ov::element::i8:
        return memory::data_type::s8;
    case ov::element::u8:
    case ov::element::boolean:
        return memory::data_type::u8;
    case ov::element::u1:
        return memory::data_type::bin;
    case ov::element::f16:
        return memory::data_type::f16;
    case ov::element::nf4:
        return memory::data_type::nf4;
    case ov::element::i4:
        return memory::data_type::s4;
    case ov::element::u4:
        return memory::data_type::u4;
    case ov::element::f8e8m0:
        return memory::data_type::f8_e8m0;
    case ov::element::f8e4m3:
        return memory::data_type::f8_e4m3;
    case ov::element::f8e5m2:
        return memory::data_type::f8_e5m2;
    case ov::element::f4e2m1:
        return memory::data_type::f4_e2m1;
    case ov::element::dynamic:
        return memory::data_type::undef;
    default: {
        return {};
    }
    }
}

dnnl::memory::data_type DnnlExtensionUtils::ElementTypeToDataType(const ov::element::Type& elementType,
                                                                  DnnlExtensionUtils::throw_tag) {
    auto&& result = ElementTypeToDataType(elementType, nothrow_tag{});
    OPENVINO_ASSERT(result, "CPU plugin does not support ", elementType.to_string(), " for use with oneDNN.");
    return result.value();
}

ov::element::Type DnnlExtensionUtils::DataTypeToElementType(const dnnl::memory::data_type& dataType) {
    switch (dataType) {
    case memory::data_type::f32:
        return ov::element::f32;
    case memory::data_type::s32:
        return ov::element::i32;
    case memory::data_type::bf16:
        return ov::element::bf16;
    case memory::data_type::s8:
        return ov::element::i8;
    case memory::data_type::u8:
        return ov::element::u8;
    case memory::data_type::bin:
        return ov::element::u1;
    case memory::data_type::f16:
        return ov::element::f16;
    case memory::data_type::f64:
        return ov::element::f64;
    case memory::data_type::nf4:
        return ov::element::nf4;
    case memory::data_type::s4:
        return ov::element::i4;
    case memory::data_type::u4:
        return ov::element::u4;
    case memory::data_type::f8_e8m0:
        return ov::element::f8e8m0;
    case memory::data_type::f8_e4m3:
        return ov::element::f8e4m3;
    case memory::data_type::f8_e5m2:
        return ov::element::f8e5m2;
    case memory::data_type::f4_e2m1:
        return ov::element::f4e2m1;
    case memory::data_type::undef:
        return ov::element::dynamic;
    default: {
        OPENVINO_THROW("Unsupported data type.");
    }
    }
}

Dim DnnlExtensionUtils::convertToDim(const dnnl::memory::dim& dim) {
    return dim == DNNL_RUNTIME_DIM_VAL ? Shape::UNDEFINED_DIM : static_cast<size_t>(dim);
}
dnnl::memory::dim DnnlExtensionUtils::convertToDnnlDim(const Dim& dim) {
    return dim == Shape::UNDEFINED_DIM ? DNNL_RUNTIME_DIM_VAL : static_cast<dnnl::memory::dim>(dim);
}

VectorDims DnnlExtensionUtils::convertToVectorDims(const memory::dims& dims) {
    std::vector<size_t> vecResult(dims.size());
    std::transform(dims.begin(), dims.end(), vecResult.begin(), convertToDim);
    return vecResult;
}

VectorDims DnnlExtensionUtils::convertToVectorDims(const dnnl::impl::dims_t dims, const int ndims) {
    return {dims, dims + ndims};
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

DnnlMemoryDescPtr DnnlExtensionUtils::makeDescriptor(const dnnl::memory::desc& desc) {
    return makeDescriptor(desc.get());
}

DnnlMemoryDescPtr DnnlExtensionUtils::makeDescriptor(const_dnnl_memory_desc_t desc) {
    if (desc->format_kind == dnnl::impl::format_kind_t::dnnl_blocked) {
        return std::shared_ptr<DnnlBlockedMemoryDesc>(new DnnlBlockedMemoryDesc(desc));
    }
    return std::shared_ptr<DnnlMemoryDesc>(new DnnlMemoryDesc(desc));
}

size_t DnnlExtensionUtils::getMemSizeForDnnlDesc(const dnnl::memory::desc& desc) {
    OPENVINO_ASSERT(IMPLICATION(desc.get_format_kind() == dnnl::memory::format_kind::blocked, desc.get()->offset0 == 0),
                    "Unexpected non zero offset for a dnnl blocked memory desc");

    size_t size = desc.get_size();
    if (size == DNNL_RUNTIME_SIZE_VAL) {
        return MemoryDesc::UNDEFINED_SIZE;
    }

    return size;
}

std::shared_ptr<DnnlBlockedMemoryDesc> DnnlExtensionUtils::makeUndefinedDesc(const memory::desc& desc,
                                                                             const Shape& shape) {
    if (desc.get_format_kind() == memory::format_kind::blocked) {
        return std::shared_ptr<DnnlBlockedMemoryDesc>(new DnnlBlockedMemoryDesc(desc, shape));
    }
    OPENVINO_THROW("Unexpected: Cannot make undefined descriptor. Only dnnl_blocked type is allowed.");
}

DnnlMemoryDescPtr DnnlExtensionUtils::query_md(const const_dnnl_primitive_desc_t& pd,
                                               const dnnl::query& what,
                                               int idx) {
    auto query = dnnl::convert_to_c(what);
    const auto* cdesc = dnnl_primitive_desc_query_md(pd, query, idx);

    if (!cdesc) {
        OPENVINO_THROW("query_md failed for query=", query, " idx=", idx, ".");
    }

    return DnnlExtensionUtils::makeDescriptor(cdesc);
}

std::string DnnlExtensionUtils::query_impl_info_str(const const_dnnl_primitive_desc_t& pd) {
    const char* res;
    dnnl_status_t status = dnnl_primitive_desc_query(pd, dnnl_query_impl_info_str, 0, reinterpret_cast<void*>(&res));
    if (status != dnnl_success) {
        OPENVINO_THROW("query_impl_info_str failed.");
    }
    return res;
}

bool DnnlExtensionUtils::find_implementation(dnnl::primitive_desc& desc, impl_desc_type impl_type) {
    return DnnlExtensionUtils::find_implementation(desc, [impl_type](impl_desc_type cur_impl_type) {
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

bool DnnlExtensionUtils::isUnarySupportedAsPostOp(Algorithm alg) {
#if defined(OV_CPU_WITH_ACL)
    return one_of(alg,
                  Algorithm::EltwiseRelu,
                  Algorithm::EltwiseTanh,
                  Algorithm::EltwiseElu,
                  Algorithm::EltwiseAbs,
                  Algorithm::EltwiseSqrt,
                  Algorithm::EltwiseSoftRelu,
                  Algorithm::EltwiseSigmoid,
                  Algorithm::EltwiseClamp);
#elif defined(OPENVINO_ARCH_X86_64)
    return one_of(alg,
                  Algorithm::EltwiseRelu,
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

std::string DnnlExtensionUtils::computeWeightsStringHash(const std::shared_ptr<const IMemory>& memory,
                                                         const std::shared_ptr<DnnlMemoryDesc>& dstDesc) {
    const auto desc_hash = dnnl::impl::primitive_hashing::get_md_hash(*dstDesc->getDnnlDesc().get());
    return std::to_string(desc_hash) + "_" + std::to_string(reinterpret_cast<uint64_t>(memory->getData()));
}

}  // namespace ov::intel_cpu
