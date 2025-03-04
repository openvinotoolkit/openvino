// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_memory_desc.h"

#include <common/memory_desc.hpp>
#include <common/memory_desc_wrapper.hpp>

#include "dnnl_extension_utils.h"
#include "onednn/dnnl.h"

namespace ov::intel_cpu {

DnnlMemoryDesc::DnnlMemoryDesc(const dnnl::memory::desc& desc) : DnnlMemoryDesc(desc.get()) {}

DnnlMemoryDesc::DnnlMemoryDesc(const_dnnl_memory_desc_t cdesc)
    : MemoryDesc(Shape(DnnlExtensionUtils::convertToVectorDims(cdesc->dims, cdesc->ndims)), Dnnl),
      desc(DnnlExtensionUtils::clone_desc(cdesc)) {
    if (getFormatKind() == dnnl::memory::format_kind::any) {
        OPENVINO_THROW("Unexpected: Memory format any is prohibited!");
    }
}

ov::element::Type DnnlMemoryDesc::getPrecision() const {
    return DnnlExtensionUtils::DataTypeToElementType(getDataType());
}

MemoryDescPtr DnnlMemoryDesc::clone() const {
    return std::make_shared<DnnlMemoryDesc>(*this);
}

MemoryDescPtr DnnlMemoryDesc::cloneWithNewPrecision(const ov::element::Type prec) const {
    auto newDesc = std::make_shared<DnnlMemoryDesc>(*this);
    newDesc->setPrecision(prec);
    return newDesc;
}

bool DnnlMemoryDesc::isCompatible(const MemoryDesc& rhs) const {
    if (MemoryDescType::Dnnl & rhs.getType()) {
        auto* dnnMemDesc = rhs.as<DnnlMemoryDesc>();
        return isCompatible(*dnnMemDesc);
    }
    return false;
}

bool DnnlMemoryDesc::isCompatible(const DnnlMemoryDesc& rhs) const {
    return this->desc == rhs.desc;
}

std::string DnnlMemoryDesc::serializeFormat() const {
    dnnl::impl::memory_desc_wrapper wrapped(desc.get());
    if (wrapped.is_wino_desc()) {
        switch (desc.get()->format_desc.wino_desc.wino_format) {
        case dnnl::impl::wino_memory_format_t::wino_wei_aaOio:
            return "wino_aaOio";
        case dnnl::impl::wino_memory_format_t::wino_wei_aaOBiOo:
            return "wino_aaOBiOo";
        case dnnl::impl::wino_memory_format_t::wino_wei_OBaaIBOIio:
            return "wino_OBaaIBOIio";
        default:
            return "wino_undef";
        }
    } else if (wrapped.is_rnn_packed_desc()) {
        switch (desc.get()->format_desc.rnn_packed_desc.format) {
        case dnnl::impl::rnn_packed_format::ldigo_p:
            return "packed_ldigo";
        case dnnl::impl::rnn_packed_format::ldgoi_p:
            return "packed_ldgoi";
        case dnnl::impl::rnn_packed_format::ldio_p:
            return "packed_ldio";
        default:
            return "packed_undef";
        }
    }
    return "undef";
}

size_t DnnlMemoryDesc::getMaxMemSize() const {
    if (shape.isDynamic()) {
        OPENVINO_THROW("Can't compute max mem size for DnnlMemoryDesc with dynamic shape");
    }

    return getCurrentMemSize();
}

dnnl::memory::data_type DnnlMemoryDesc::getDataType() const {
    return desc.get_data_type();
}

dnnl::memory::format_kind DnnlMemoryDesc::getFormatKind() const {
    return desc.get_format_kind();
}

bool DnnlMemoryDesc::hasEmptyExtraData() const {
    dnnl::impl::memory_desc_wrapper wrapped(desc.get());
    return wrapped.extra().flags == dnnl_memory_extra_flag_none;
}

bool DnnlMemoryDesc::canComputeMemSizeZeroDims() const {
    if (!getShape().hasZeroDims()) {
        return false;
    }

    dnnl::impl::memory_desc_wrapper wrapped(desc.get());
    return getShape().hasZeroDims() && wrapped.offset0() != DNNL_RUNTIME_DIM_VAL;
}

size_t DnnlMemoryDesc::getCurrentMemSizeImp() const {
    return DnnlExtensionUtils::getMemSizeForDnnlDesc(desc);
}

size_t DnnlMemoryDesc::getElementOffset(size_t elemNumber) const {
    dnnl::impl::memory_desc_wrapper wrapped(desc.get());
    return wrapped.off_l(elemNumber);
}

bool DnnlMemoryDesc::isDefinedImp() const {
    dnnl::impl::memory_desc_wrapper wrappedThis(desc.get());

    if (wrappedThis.has_runtime_dims_or_strides()) {
        return false;
    }

    return wrappedThis.offset0() != DNNL_RUNTIME_DIM_VAL;
}

MemoryDescPtr DnnlMemoryDesc::cloneWithNewDimsImp(const VectorDims& dims) const {
    OPENVINO_THROW("Unexpected: Cannot clone non blocked oneDNN desc with new dims");
}

size_t DnnlMemoryDesc::getOffsetPadding() const {
    dnnl::impl::memory_desc_wrapper wrap(desc.get());
    return DnnlExtensionUtils::convertToDim(wrap.offset0());
}

}  // namespace ov::intel_cpu
