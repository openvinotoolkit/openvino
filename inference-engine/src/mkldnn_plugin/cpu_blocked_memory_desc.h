// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory_desc.h"

namespace MKLDNNPlugin {
class BlockedMemoryDesc : public MemoryDesc {
public:
    BlockedMemoryDesc(InferenceEngine::Precision prc, const std::vector<size_t>& dims);

    BlockedMemoryDesc(InferenceEngine::Precision prc, const std::vector<size_t>& dims, const std::vector<size_t>& blockedDims,
                      const std::vector<size_t>& order, size_t offsetPadding = 0, const std::vector<size_t>& offsetPaddingToData = {},
                      const std::vector<size_t>& strides = {});

    MemoryDescPtr clone() const override {
        return make_unique<BlockedMemoryDesc>(*this);
    }

    bool isDefined() const override;

    bool isCompatible(const MemoryDesc& rhs) const override {
        try {
            const auto& blockingDesc = dynamic_cast<const BlockedMemoryDesc&>(rhs);
            return isCompatible(blockingDesc);
        }
        catch (const std::bad_cast& excp) {
            //IE_THROW() << "Cannot check compatibility with this type of memory descriptor";
            return false;
        }
    }

    bool isCompatible(const BlockedMemoryDesc& rhs) const;

    const std::vector<size_t>& getBlockDims() const {
        return blockedDims;
    }

    /**
     * @brief Returns the vector of order
     *
     * @return order
     */
    const std::vector<size_t>& getOrder() const {
        return order;
    }

    /**
     * @brief Returns the per-dimension offset vector
     *
     * @return offsets
     */
    const std::vector<size_t>& getOffsetPaddingToData() const {
        return offsetPaddingToData;
    }

    /**
     * @brief Returns the offset to the current memory block
     *
     * @return offset
     */
    size_t getOffsetPadding() const {
        return offsetPadding;
    }

    /**
     * @brief Returns strides for each dimension
     *
     * @return strides
     */
    const std::vector<size_t>& getStrides() const {
        return strides;
    }

    size_t getOffset(size_t elemNumber) const override;

private:
    size_t getMemSizeImp() const override;
    size_t getOffset(const InferenceEngine::SizeVector& v) const;

private:
    std::vector<size_t> blockedDims;
    std::vector<size_t> strides;
    std::vector<size_t> order;
    std::vector<size_t> offsetPaddingToData;
    size_t offsetPadding;
};
} // namespace MKLDNNPlugin
