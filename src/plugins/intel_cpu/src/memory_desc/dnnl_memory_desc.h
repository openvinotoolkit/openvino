// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl_common_types.h>
#include <oneapi/dnnl/dnnl_types.h>

#include <common/memory_desc.hpp>
#include <cstddef>
#include <limits>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <string>

#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

class DnnlMemoryDesc;

using DnnlMemoryDescPtr = std::shared_ptr<DnnlMemoryDesc>;
using DnnlMemoryDescCPtr = std::shared_ptr<const DnnlMemoryDesc>;

class DnnlMemoryDesc : public virtual MemoryDesc {
public:
    ov::element::Type getPrecision() const override;

    MemoryDescPtr clone() const override;

    MemoryDescPtr cloneWithNewPrecision(ov::element::Type prec) const override;

    bool isCompatible(const MemoryDesc& rhs) const override;
    bool isCompatible(const DnnlMemoryDesc& rhs) const;

    bool hasLayoutType([[maybe_unused]] LayoutType layoutType) const override {
        return false;
    }

    std::string serializeFormat() const override;

    size_t getMaxMemSize() const override;

    virtual bool isSame([[maybe_unused]] dnnl::memory::format_tag fmt) const {
        return false;
    }

    const dnnl::memory::desc& getDnnlDesc() const {
        return desc;
    }

    dnnl::memory::data_type getDataType() const;

    dnnl::memory::format_kind getFormatKind() const;

    bool hasEmptyExtraData() const;

    size_t getOffsetPadding() const override;

protected:
    DnnlMemoryDesc() {}
    static constexpr size_t UNREACHABLE_DIM = std::numeric_limits<size_t>::max();

    dnnl::memory::desc desc;

    void setPrecision(ov::element::Type prc) override {
        desc.get()->data_type = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(prc));
    }

private:
    explicit DnnlMemoryDesc(const dnnl::memory::desc& desc);
    explicit DnnlMemoryDesc(const_dnnl_memory_desc_t cdesc);

    size_t getElementOffset(size_t elemNumber) const override;

    bool canComputeMemSizeZeroDims() const override;
    size_t getCurrentMemSizeImp() const override;
    bool isDefinedImp() const override;
    MemoryDescPtr cloneWithNewDimsImp(const VectorDims& dims) const override;

    friend DnnlMemoryDescPtr DnnlExtensionUtils::makeDescriptor(const dnnl::memory::desc& desc);
    friend DnnlMemoryDescPtr DnnlExtensionUtils::makeDescriptor(const_dnnl_memory_desc_t desc);
};

}  // namespace ov::intel_cpu
