// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_blocked_memory_desc.h"
#include "mkldnn_extension_utils.h"

namespace MKLDNNPlugin {

class DnnlMemoryDesc;

using DnnlMemoryDescPtr = std::unique_ptr<DnnlMemoryDesc>;
using DnnlMemoryDescCPtr = std::unique_ptr<const DnnlMemoryDesc>;

class DnnlMemoryDesc : public virtual MemoryDesc {
public:
    mkldnn::memory::data_type getDataType() const {
        return static_cast<mkldnn::memory::data_type>(desc.data.data_type);
    }

    dnnl_format_kind_t getFormatKind() const {
        return desc.data.format_kind;
    }

    std::unique_ptr<MemoryDesc> clone() const override {
        return MKLDNNPlugin::make_unique<DnnlMemoryDesc>(*this);
    }

    std::string serializeFormat() const override;

    InferenceEngine::Precision getPrecision() const override;

    void setPrecision(InferenceEngine::Precision prc) override;

    bool isCompatible(const MemoryDesc& rhs) const override;

    size_t getMaxMemSize() const override;

    const mkldnn::memory::desc& getDnnlDesc() const {
        return desc;
    }

    bool hasLayoutType(LayoutType layoutType) const override { return false; }

    virtual bool isSame(mkldnn::memory::format_tag fmt) const { return false; }

    bool hasEmptyExtraData() const { return desc.data.extra.flags == dnnl_memory_extra_flag_none; }

protected:
    DnnlMemoryDesc() {}
    static constexpr size_t UNREACHABLE_DIM = std::numeric_limits<size_t>::max();

    mkldnn::memory::desc desc;

private:
    explicit DnnlMemoryDesc(const mkldnn::memory::desc& desc);

    size_t getElementOffset(size_t elemNumber) const override;

    size_t getCurrentMemSizeImp() const override;
    bool isDefinedImp() const override;
    std::unique_ptr<MemoryDesc> cloneWithNewDimsImp(const std::vector<size_t>& dims) const override;

    friend DnnlMemoryDescPtr MKLDNNExtensionUtils::makeDescriptor(const mkldnn::memory::desc &desc);
};

}  // namespace MKLDNNPlugin
