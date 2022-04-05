// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_memory_desc.h"
#include <dnnl_extension_utils.h>
#include <common/memory_desc_wrapper.hpp>
#include <onednn/dnnl.h>

namespace ov {
namespace intel_cpu {

DnnlMemoryDesc::DnnlMemoryDesc(const dnnl::memory::desc& desc) :
    MemoryDesc(Shape(DnnlExtensionUtils::convertToVectorDims(desc.dims())), Dnnl), desc(desc) {
    if (desc.data.format_kind == dnnl::impl::format_kind::any)
        IE_THROW(Unexpected) << "Memory format any is prohibited!";
}

bool DnnlMemoryDesc::canComputeMemSizeZeroDims() const {
    return getShape().hasZeroDims() && desc.data.offset0 != DNNL_RUNTIME_DIM_VAL;
}

size_t DnnlMemoryDesc::getCurrentMemSizeImp() const {
    return DnnlExtensionUtils::getMemSizeForDnnlDesc(desc);
}

size_t DnnlMemoryDesc::getElementOffset(size_t elemNumber) const {
    dnnl::impl::memory_desc_wrapper wrapped(desc.data);
    return wrapped.off_l(elemNumber);
}

bool DnnlMemoryDesc::isCompatible(const MemoryDesc &rhs) const {
    if (MemoryDescType::Dnnl == rhs.getType()) {
        return this->desc == rhs.as<DnnlMemoryDesc>()->desc;
    } else {
        return false;
    }
}

// TODO: add serialization for packed format
std::string DnnlMemoryDesc::serializeFormat() const {
    if (desc.data.format_kind == dnnl_format_kind_wino) {
        switch (desc.data.format_desc.wino_desc.wino_format) {
            case dnnl_wino_memory_format_t::dnnl_wino_wei_aaOIoi: return "wino_aaOIoi";
            case dnnl_wino_memory_format_t::dnnl_wino_wei_aaOio: return "wino_aaOio";
            case dnnl_wino_memory_format_t::dnnl_wino_wei_aaOBiOo: return "wino_aaOBiOo";
            case dnnl_wino_memory_format_t::dnnl_wino_wei_OBaaIBOIio: return "wino_OBaaIBOIio";
            default: return "wino_undef";
        }
    }
    return "undef";
}

bool DnnlMemoryDesc::isDefinedImp() const {
    dnnl::impl::memory_desc_wrapper wrappedThis(desc.data);

    if (wrappedThis.has_runtime_dims_or_strides()) {
        return false;
    }

    return wrappedThis.offset0() != DNNL_RUNTIME_DIM_VAL;
}

InferenceEngine::Precision DnnlMemoryDesc::getPrecision() const {
    return DnnlExtensionUtils::DataTypeToIEPrecision(desc.data_type());
}

MemoryDescPtr DnnlMemoryDesc::cloneWithNewDimsImp(const VectorDims &dims) const {
    IE_THROW(Unexpected) << "Cannot clone non blocked oneDNN desc with new dims";
}

size_t DnnlMemoryDesc::getMaxMemSize() const {
    if (shape.isDynamic()) {
        IE_THROW() << "Can't compute max mem size for DnnlMemoryDesc with dynaimc shape";
    }

    return getCurrentMemSize();
}

MemoryDescPtr DnnlMemoryDesc::cloneWithNewPrecision(const InferenceEngine::Precision prec) const {
    auto newDesc = std::make_shared<DnnlMemoryDesc>(*this);
    newDesc->setPrecision(prec);
    return newDesc;
}

}   // namespace intel_cpu
}   // namespace ov
