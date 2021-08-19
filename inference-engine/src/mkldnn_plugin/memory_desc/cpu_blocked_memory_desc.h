// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "blocked_memory_desc.h"
#include "utils/general_utils.h"

namespace MKLDNNPlugin {

class CpuBlockedMemoryDesc : public BlockedMemoryDesc {
public:
    CpuBlockedMemoryDesc(InferenceEngine::Precision prc, const Shape& shape);

    CpuBlockedMemoryDesc(InferenceEngine::Precision prc, const Shape& shape, const std::vector<size_t>& blockedDims,
                         const std::vector<size_t>& order, size_t offsetPadding = 0, const std::vector<size_t>& offsetPaddingToData = {},
                         const std::vector<size_t>& strides = {});

    MemoryDescPtr clone() const override {
        return MKLDNNPlugin::make_unique<CpuBlockedMemoryDesc>(*this);
    }

    bool isCompatible(const MemoryDesc& rhs) const override;
    bool isCompatible(const CpuBlockedMemoryDesc &rhs) const;
    bool isCompatible(const DnnlBlockedMemoryDesc &rhs) const;

    InferenceEngine::Precision getPrecision() const override {
        return precision;
    }

    void setPrecision(InferenceEngine::Precision prc) override {
        precision = std::move(prc);
    }

    const std::vector<size_t>& getBlockDims() const override {
        return blockedDims;
    }

    /**
     * @brief Returns the vector of order
     *
     * @return order
     */
    const std::vector<size_t>& getOrder() const override {
        return order;
    }

    /**
     * @brief Returns the per-dimension offset vector
     *
     * @return offsets
     */
    const std::vector<size_t>& getOffsetPaddingToData() const override {
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
    const std::vector<size_t>& getStrides() const override {
        return strides;
    }

    bool blocksExtended() const override;

    bool hasLayoutType(LayoutType layoutType) const override;

    std::string serializeFormat() const override;

    size_t getMaxMemSize() const override;

    size_t getPaddedElementsCount() const override;

private:
    size_t getElementOffset(size_t elemNumber) const override;
    size_t getCurrentMemSizeImp() const override;
    size_t getOffset(const InferenceEngine::SizeVector& v) const;
    bool isPlainFormat() const;
    bool isBlockedCFormat(size_t blk_size) const;
    bool isTailCFormat() const;
    bool isDefinedImp() const override;
    std::unique_ptr<MemoryDesc> cloneWithNewDimsImp(const std::vector<size_t>& dims) const override;

private:
    InferenceEngine::Precision precision;
    size_t offsetPadding;
    mutable VectorDims paddedDims;
};

} // namespace MKLDNNPlugin
