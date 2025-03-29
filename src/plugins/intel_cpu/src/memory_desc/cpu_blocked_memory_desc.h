// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "blocked_memory_desc.h"
#include "dnnl_extension_utils.h"

namespace ov {
namespace intel_cpu {

class DnnlBlockedMemoryDesc;

class CpuBlockedMemoryDesc : public BlockedMemoryDesc {
public:
    CpuBlockedMemoryDesc(ov::element::Type prc, const Shape& shape);

    CpuBlockedMemoryDesc(ov::element::Type prc,
                         const Shape& shape,
                         const VectorDims& blockedDims,
                         const VectorDims& order,
                         size_t offsetPadding = 0,
                         const VectorDims& offsetPaddingToData = {},
                         const VectorDims& strides = {});

    MemoryDescPtr clone() const override {
        return std::make_shared<CpuBlockedMemoryDesc>(*this);
    }

    bool isCompatible(const MemoryDesc& rhs) const override;
    bool isCompatible(const BlockedMemoryDesc& rhs, CmpMask cmpMask) const override;
    bool isCompatible(const CpuBlockedMemoryDesc& rhs, CmpMask cmpMask = BlockedMemoryDesc::FULL_MASK) const;
    bool isCompatible(const DnnlBlockedMemoryDesc& rhs, CmpMask cmpMask = BlockedMemoryDesc::FULL_MASK) const;

    ov::element::Type getPrecision() const override {
        return precision;
    }

    const VectorDims& getBlockDims() const override {
        return blockedDims;
    }

    /**
     * @brief Returns the vector of order
     *
     * @return order
     */
    const VectorDims& getOrder() const override {
        return order;
    }

    /**
     * @brief Returns the per-dimension offset vector
     *
     * @return offsets
     */
    const VectorDims& getOffsetPaddingToData() const override {
        return offsetPaddingToData;
    }
    /**
     * @brief Returns the offset to the current memory block
     *
     * @return offset
     */
    size_t getOffsetPadding() const override {
        return offsetPadding;
    }

    /**
     * @brief Returns strides for each dimension
     *
     * @return strides
     */
    const VectorDims& getStrides() const override {
        return strides;
    }

    bool blocksExtended() const override;

    bool hasLayoutType(LayoutType layoutType) const override;

    size_t getMaxMemSize() const override;

    size_t getPaddedElementsCount() const override;

    MemoryDescPtr cloneWithNewPrecision(const ov::element::Type prec) const override;

private:
    size_t getElementOffset(size_t elemNumber) const override;
    bool canComputeMemSizeZeroDims() const override;
    size_t getCurrentMemSizeImp() const override;
    size_t getOffset(const VectorDims& v) const;
    bool isPlainFormat() const;
    bool isBlockedCFormat(size_t blk_size) const;
    bool isTailCFormat() const;
    bool isDefinedImp() const override;
    MemoryDescPtr cloneWithNewDimsImp(const VectorDims& dims) const override;

    void setPrecision(ov::element::Type prc) override {
        precision = prc;
    }

private:
    ov::element::Type precision;
    size_t offsetPadding;
};

using CpuBlockedMemoryDescPtr = std::shared_ptr<CpuBlockedMemoryDesc>;
using CpuBlockedMemoryDescCPtr = std::shared_ptr<const CpuBlockedMemoryDesc>;

}  // namespace intel_cpu
}  // namespace ov
