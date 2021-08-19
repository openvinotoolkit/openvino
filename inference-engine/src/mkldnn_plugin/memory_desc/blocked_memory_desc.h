// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory_desc.h"

namespace MKLDNNPlugin {

class BlockedMemoryDesc : public virtual MemoryDesc {
public:
    BlockedMemoryDesc() {}

    /**
     * @brief Returns the blocked dimensions
     *
     * @return blocked dimensions
     */
    virtual const std::vector<size_t>& getBlockDims() const = 0;

    /**
     * @brief Returns the vector of order
     *
     * @return order
     */
    virtual const std::vector<size_t>& getOrder() const = 0;

    /**
     * @brief Returns the per-dimension offset vector
     *
     * @return offsets
     */
    virtual const std::vector<size_t>& getOffsetPaddingToData() const = 0;

    /**
     * @brief Returns the offset to the current memory block
     *
     * @return offset
     */
    virtual size_t getOffsetPadding() const = 0;

    /**
     * @brief Returns strides for each dimension
     *
     * @return strides
     */
    virtual const std::vector<size_t>& getStrides() const = 0;

    /**
     * @brief Check that desc has padded dims
     *
     * @return true if exist padded dims, otherwise false
     */
    virtual bool blocksExtended() const = 0;

    /**
     * @brief Compute number of elements taking into account padded dims
     *
     * @return number of elements taking into account padded dims
     */
    virtual size_t getPaddedElementsCount() const = 0;

protected:
    /**
     * @brief Check descs on compatibility
     * WARNING: Check only BlockedMemoryDesc specific attributes like: strides, order etc.
     * Doesn't perform type check for descs
     * Doesn't perform descs specific attributes check
     * @return true if compatible, otherwise false
     */
    bool isCompatible(const BlockedMemoryDesc &rhs) const;

    mutable VectorDims blockedDims;
    mutable std::vector<size_t> strides;
    mutable std::vector<size_t> order;
    mutable std::vector<size_t> offsetPaddingToData;
};

using BlockedMemoryDescPtr = std::unique_ptr<BlockedMemoryDesc>;
using BlockedMemoryDescCPtr = std::unique_ptr<const BlockedMemoryDesc>;

} // namespace MKLDNNPlugin
