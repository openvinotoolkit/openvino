// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "blocked_memory_desc.h"
#include "mkldnn_memory.h"
#include "mkldnn_extension_utils.h"

namespace MKLDNNPlugin {

class DnnlBlockedMemoryDesc : public BlockedMemoryDesc, public DnnlMemoryDesc {
public:
    // Creates planar DnnlBlockedMemoryDesc
    DnnlBlockedMemoryDesc(InferenceEngine::Precision prc, const Shape& shape);

    DnnlBlockedMemoryDesc(const Shape& shape, mkldnn::memory::data_type dataType, mkldnn::memory::format_tag format);

    MemoryDescPtr clone() const override {
        return MKLDNNPlugin::make_unique<DnnlBlockedMemoryDesc>(*this);
    }

    bool isCompatible(const MemoryDesc& rhs) const override;
    bool isCompatible(const DnnlBlockedMemoryDesc& rhs) const;
    bool isCompatible(const CpuBlockedMemoryDesc& rhs) const;

    const std::vector<size_t>& getBlockDims() const override;

    const std::vector<size_t>& getOrder() const override;

    const std::vector<size_t>& getOffsetPaddingToData() const override;

    size_t getOffsetPadding() const override;

    const std::vector<size_t>& getStrides() const override;

    bool hasLayoutType(LayoutType layoutType) const override;

    bool blocksExtended() const override;

    bool isSame(mkldnn::memory::format_tag fmt) const override;

    std::string serializeFormat() const override;

    size_t getMaxMemSize() const override;

    size_t getPaddedElementsCount() const override;

private:
    DnnlBlockedMemoryDesc(InferenceEngine::Precision prc, const Shape& shape, const std::vector<size_t>& blockedDims,
                            const std::vector<size_t>& order, size_t offsetPadding = 0, const std::vector<size_t>& offsetPaddingToData = {},
                            const std::vector<size_t>& strides = {});

    DnnlBlockedMemoryDesc(const mkldnn::memory::desc& mdesc);

    std::unique_ptr<MemoryDesc> cloneWithNewDimsImp(const std::vector<size_t>& dims) const override;

    bool isPlainFormat() const;
    bool isBlockedCFormat(size_t blk_size = UNREACHABLE_DIM) const;
    bool isTailCFormat() const;

    /**
     * Try to define original format tag use on creation
     *
     * @return format tag if was able to define it
     */
    mkldnn::memory::format_tag getFormat() const;

    friend DnnlMemoryDescPtr MKLDNNExtensionUtils::makeDescriptor(const mkldnn::memory::desc &desc);
    friend class MemoryDescUtils;
};

} // namespace MKLDNNPlugin
