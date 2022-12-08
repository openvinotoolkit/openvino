// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "blocked_memory_desc.h"
#include <cpu_memory.h>
#include <dnnl_extension_utils.h>

namespace ov {
namespace intel_cpu {

class DnnlBlockedMemoryDesc : public BlockedMemoryDesc, public DnnlMemoryDesc {
public:
    // Creates planar DnnlBlockedMemoryDesc
    DnnlBlockedMemoryDesc(InferenceEngine::Precision prc, const Shape& shape, const VectorDims& strides = {});

    DnnlBlockedMemoryDesc(const Shape& shape, dnnl::memory::data_type dataType, dnnl::memory::format_tag format);

    MemoryDescPtr clone() const override {
        return std::make_shared<DnnlBlockedMemoryDesc>(*this);
    }

    bool isCompatible(const MemoryDesc& rhs) const override;
    bool isCompatible(const BlockedMemoryDesc& rhs, CmpMask cmpMask) const override;
    bool isCompatible(const CpuBlockedMemoryDesc &rhs, CmpMask cmpMask = BLOCKED_DESC_FULL_MASK) const;
    bool isCompatible(const DnnlBlockedMemoryDesc &rhs, CmpMask cmpMask = BLOCKED_DESC_FULL_MASK) const;

    const VectorDims& getBlockDims() const override {
        return blockedDims;
    }

    const VectorDims& getOrder() const override {
        return order;
    }

    const VectorDims& getOffsetPaddingToData() const override {
        return offsetPaddingToData;
    }

    size_t getOffsetPadding() const override {
        return DnnlExtensionUtils::convertToDim(desc.data.offset0);
    }

    const VectorDims& getStrides() const override {
        return strides;
    }

    bool hasLayoutType(LayoutType layoutType) const override;

    bool isSame(dnnl::memory::format_tag fmt) const override;

    std::string serializeFormat() const override;

    size_t getMaxMemSize() const override;

    bool blocksExtended() const override;

    size_t getPaddedElementsCount() const override;

    MemoryDescPtr cloneWithNewPrecision(const InferenceEngine::Precision prec) const override;

    using DnnlMemoryDesc::setPrecision;
    using DnnlMemoryDesc::getPrecision;

private:
    DnnlBlockedMemoryDesc(InferenceEngine::Precision prc, const Shape& shape, const VectorDims& blockedDims,
                          const VectorDims& order, size_t offsetPadding = 0, const VectorDims& offsetPaddingToData = {},
                          const VectorDims& strides = {});

    explicit DnnlBlockedMemoryDesc(const dnnl::memory::desc& mdesc);

    // Creates DnnlBlockedMemoryDesc using the shape parameter as a true shape but all other params (layout, blocks, etc.) are used from the mdesc, but
    // the mdesc own shape is ignored. The main purpose of this constructor is making dynamic descriptor from some dummy mdesc, which stores info about
    // layout, blocking, strides, etc., and the provided dynamic shape.
    DnnlBlockedMemoryDesc(const dnnl::memory::desc& mdesc, const Shape& shape);

    MemoryDescPtr cloneWithNewDimsImp(const VectorDims& dims) const override;

    bool isPlainFormat() const;
    bool isBlockedCFormat(size_t blk_size = UNREACHABLE_DIM) const;
    bool isTailCFormat() const;

    // WA: we need to initialize blocked params into ctor to avoid bugs when we calculate these params in throughput mode
    // TODO [DS]: should be reimplemented to avoid useless calculation
    void initBlockedParams() {
        initBlockDims();
        initStrides();
        initOffsetPadding();
    }

    void initBlockDims();
    void initStrides();
    void initOffsetPadding();

    void recomputeDefaultStrides();

    friend DnnlMemoryDescPtr DnnlExtensionUtils::makeDescriptor(const dnnl::memory::desc &desc);
    friend std::shared_ptr<DnnlBlockedMemoryDesc> DnnlExtensionUtils::makeUndefinedDesc(const dnnl::memory::desc &desc, const Shape& shape);
    friend class MemoryDescUtils;
};

using DnnlBlockedMemoryDescPtr = std::shared_ptr<DnnlBlockedMemoryDesc>;
using DnnlBlockedMemoryDescCPtr = std::shared_ptr<const DnnlBlockedMemoryDesc>;

}   // namespace intel_cpu
}   // namespace ov
