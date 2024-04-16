// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dnnl_memory_desc.h"
#include "memory_desc/blocked_memory_desc.h"
#include "openvino/util/util.hpp"
#include "dnnl_extension_utils.h"
#include <common/memory_desc_wrapper.hpp>

namespace ov {
namespace intel_cpu {

class CpuBlockedMemoryDesc;

OPENVINO_DISABLE_WARNING_MSVC_BEGIN(4250)  // Visual Studio warns us about inheritance via dominance but it's done intentionally
                                           // so turn it off
class DnnlBlockedMemoryDesc : public BlockedMemoryDesc, public DnnlMemoryDesc {
public:
    // Creates planar DnnlBlockedMemoryDesc
    DnnlBlockedMemoryDesc(ov::element::Type prc, const Shape& shape, const VectorDims& strides = {});

    DnnlBlockedMemoryDesc(const Shape& shape, dnnl::memory::data_type dataType, dnnl::memory::format_tag format);

    MemoryDescPtr clone() const override {
        return std::make_shared<DnnlBlockedMemoryDesc>(*this);
    }

    bool isCompatible(const MemoryDesc& rhs) const override;
    bool isCompatible(const BlockedMemoryDesc& rhs, CmpMask cmpMask) const override;
    bool isCompatible(const CpuBlockedMemoryDesc &rhs, CmpMask cmpMask = FULL_MASK) const;
    bool isCompatible(const DnnlBlockedMemoryDesc &rhs, CmpMask cmpMask = FULL_MASK) const;

    const VectorDims& getBlockDims() const override {
        return blockedDims;
    }

    const VectorDims& getOrder() const override {
        return order;
    }

    const VectorDims& getOffsetPaddingToData() const override {
        return offsetPaddingToData;
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

    MemoryDescPtr cloneWithNewPrecision(const ov::element::Type prec) const override;

    using DnnlMemoryDesc::setPrecision;
    using DnnlMemoryDesc::getPrecision;

private:
    DnnlBlockedMemoryDesc(ov::element::Type prc, const Shape& shape, const VectorDims& blockedDims,
                          const VectorDims& order, size_t offsetPadding = 0, const VectorDims& offsetPaddingToData = {},
                          const VectorDims& strides = {});

    // Creates DnnlBlockedMemoryDesc using the shape parameter as a true shape but all other params (layout, blocks, etc.) are used from the mdesc, but
    // the mdesc own shape is ignored. The main purpose of this constructor is making dynamic descriptor from some dummy mdesc, which stores info about
    // layout, blocking, strides, etc., and the provided dynamic shape.
    DnnlBlockedMemoryDesc(const dnnl::memory::desc& mdesc, const Shape& shape);

    explicit DnnlBlockedMemoryDesc(const_dnnl_memory_desc_t cdesc);

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

    friend DnnlMemoryDescPtr DnnlExtensionUtils::makeDescriptor(const_dnnl_memory_desc_t desc);
    friend std::shared_ptr<DnnlBlockedMemoryDesc> DnnlExtensionUtils::makeUndefinedDesc(const dnnl::memory::desc &desc, const Shape& shape);
    friend class MemoryDescUtils;
};
OPENVINO_DISABLE_WARNING_MSVC_END(4250)

using DnnlBlockedMemoryDescPtr = std::shared_ptr<DnnlBlockedMemoryDesc>;
using DnnlBlockedMemoryDescCPtr = std::shared_ptr<const DnnlBlockedMemoryDesc>;

}   // namespace intel_cpu
}   // namespace ov
