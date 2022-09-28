// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_blocked_memory_desc.h"
#include <dnnl_extension_utils.h>

namespace ov {
namespace intel_cpu {

class DnnlMemoryDesc;

using DnnlMemoryDescPtr = std::shared_ptr<DnnlMemoryDesc>;
using DnnlMemoryDescCPtr = std::shared_ptr<const DnnlMemoryDesc>;

class DnnlMemoryDesc : public virtual MemoryDesc {
public:
    dnnl::memory::data_type getDataType() const {
        return static_cast<dnnl::memory::data_type>(desc.data.data_type);
    }

    dnnl_format_kind_t getFormatKind() const {
        return desc.data.format_kind;
    }

    MemoryDescPtr clone() const override {
        return std::make_shared<DnnlMemoryDesc>(*this);
    }

    std::string serializeFormat() const override;

    InferenceEngine::Precision getPrecision() const override;

    bool isCompatible(const MemoryDesc& rhs) const override;

    size_t getMaxMemSize() const override;

    const dnnl::memory::desc& getDnnlDesc() const {
        return desc;
    }

    bool hasLayoutType(LayoutType layoutType) const override { return false; }

    virtual bool isSame(dnnl::memory::format_tag fmt) const { return false; }

    bool hasEmptyExtraData() const { return desc.data.extra.flags == dnnl_memory_extra_flag_none; }

    MemoryDescPtr cloneWithNewPrecision(const InferenceEngine::Precision prec) const override;

protected:
    DnnlMemoryDesc() {}
    static constexpr size_t UNREACHABLE_DIM = std::numeric_limits<size_t>::max();

    dnnl::memory::desc desc;

    void setPrecision(InferenceEngine::Precision prc) override {
        desc.data.data_type = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(prc));
    }

private:
    explicit DnnlMemoryDesc(const dnnl::memory::desc& desc);

    size_t getElementOffset(size_t elemNumber) const override;

    bool canComputeMemSizeZeroDims() const override;
    size_t getCurrentMemSizeImp() const override;
    bool isDefinedImp() const override;
    MemoryDescPtr cloneWithNewDimsImp(const VectorDims& dims) const override;

    friend DnnlMemoryDescPtr DnnlExtensionUtils::makeDescriptor(const dnnl::memory::desc &desc);
};

}   // namespace intel_cpu
}   // namespace ov

