// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for data layouts and conversion between them
 *
 * @file ie_layouts.h
 */
#pragma once

#include <algorithm>

#include "ie_api.h"
#include "ie_common.h"
#include "ie_precision.hpp"

namespace InferenceEngine {

/**
 * @brief This class describes blocking layouts
 */
class INFERENCE_ENGINE_API_CLASS(BlockingDesc) {
public:
    /**
     * @brief The default constructor which creates empty blocking descriptor
     */
    BlockingDesc();
    /**
     * @brief The constructor which allows to create blocking descriptors for standard layouts
     *
     * @param dims real dimensions
     * @param layout memory layout
     */
    BlockingDesc(const SizeVector& dims, Layout layout);
    /**
     * @brief The constructor allows to create blocking descriptors for blocked memory
     *
     * @param blocked_dims blocked dimensions
     * @param order the order of dimensions
     */
    BlockingDesc(const SizeVector& blocked_dims, const SizeVector& order);
    /**
     * @brief The constructor allows to create blocking descriptors for blocked memory
     *
     * @param blocked_dims blocked dimensions
     * @param order the order of dimensions
     * @param offset offset to the current memory block
     */
    BlockingDesc(const SizeVector& blocked_dims, const SizeVector& order, size_t offset);
    /**
     * @brief The constructor allows to create blocking descriptors for blocked memory
     *
     * @param blocked_dims blocked dimensions
     * @param order the order of dimensions
     * @param offset offset to the current memory block
     * @param dimOffsets per-dimension offset from the padding to actual data,
     */
    BlockingDesc(const SizeVector& blocked_dims, const SizeVector& order, size_t offset, const SizeVector& dimOffsets);
    /**
     * @brief The constructor allows to create blocking descriptors for blocked memory
     *
     * @param blocked_dims blocked dimensions
     * @param order the order of dimensions
     * @param offset offset to the current memory block
     * @param dimOffsets per-dimension offset from the padding to actual data,
     * @param strides strides for each dimension
     */
    BlockingDesc(const SizeVector& blocked_dims, const SizeVector& order, size_t offset,
                 const SizeVector& dimOffsets, const SizeVector& strides);

    /**
     * @brief Returns the blocked dimensions vector
     *
     * @return blocked dimensions
     */
    const SizeVector& getBlockDims() const {
        return blockedDims;
    }

    /**
     * @brief Returns the vector of order
     *
     * @return order
     */
    const SizeVector& getOrder() const {
        return order;
    }

    /**
     * @brief Returns the per-dimension offset vector
     *
     * @return offsets
     */
    const SizeVector& getOffsetPaddingToData() const {
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
    const SizeVector& getStrides() const {
        return strides;
    }

    /**
     * @brief The comparison operator for the BlockingDesc
     *
     * @param rhs object to compare
     * @return true if objects are equal
     */
    bool operator==(const BlockingDesc& rhs) const;
    /**
     * @brief The comparison operator for the BlockingDesc
     *
     * @param rhs object to compare
     * @return true if objects aren't equal
     */
    bool operator!=(const BlockingDesc& rhs) const;

protected:
    /**
     * @brief Fills tensor descriptor based on blocking dimensions and specific order
     * @param blocked_dims A vector representing blocking dimensions
     * @param order A vector with specific dims order
     */
    void fillDesc(const SizeVector& blocked_dims, const SizeVector& order);

private:
    /** Blocked dimensions. */
    SizeVector blockedDims;
    /** Strides for blocked dimensions */
    SizeVector strides;
    /** The order of blocked dimensions **/
    SizeVector order;
    /** Per-dimension offset from the padding to actual data, the top-level
     * tensor with offsets applied must lie within the padding area. */
    SizeVector offsetPaddingToData;
    /** Offset from memory origin to the current block, non-zero only in
     * a description of a memory sub-block. */
    size_t offsetPadding;
};

/**
 * @brief This class defines Tensor description
 */
class INFERENCE_ENGINE_API_CLASS(TensorDesc) {
public:
    /**
     * @brief The constructor creates the tensor descriptor using blocking descriptor
     *
     * @param precision memory precision
     * @param dims memory dimensions
     * @param blockDesc blocking descriptor
     */
    TensorDesc(const Precision& precision, const SizeVector& dims, const BlockingDesc& blockDesc);
    /**
     * @brief The constructor creates the tensor descriptor using standard layout
     *
     * @param precision memory precision
     * @param dims memory dimensions
     * @param layout memory layout
     */
    TensorDesc(const Precision& precision, const SizeVector& dims, Layout layout);
    /**
     * @brief The constructor creates the empty tensor descriptor with precision and layout
     *
     * @param precision memory precision
     * @param layout memory layout
     */
    TensorDesc(const Precision& precision, Layout layout);
    /**
     * @brief The default constructor which creates empty tensor descriptor
     */
    TensorDesc();

    /**
     * @brief Reshapes the tensor descriptor
     *
     * @param dims new dimensions
     * @param layout new layout if it is necessary
     */
    void reshape(const SizeVector& dims, Layout layout = Layout::ANY);
    /**
     * @brief Reshapes the tensor descriptor
     *
     * @param dims new dimensions
     * @param blockDesc new blocking descriptor
     */
    void reshape(const SizeVector& dims, const BlockingDesc& blockDesc);

    /**
     * @brief Returns the vector of dimensions
     *
     * @return dimensions
     */
    SizeVector& getDims() {
        return dims;
    }
    /**
     * @brief Returns the constant vector of dimensions
     *
     * @return dimensions
     */
    const SizeVector& getDims() const noexcept {
        return dims;
    }
    /**
     * @brief Sets dimensions
     *
     * @param dims new dimensions
     */
    void setDims(const SizeVector& dims);

    /**
     * @brief Returns the memory layout
     *
     * @return layout
     */
    Layout getLayout() const {
        return layout;
    }

    /**
     * @brief Sets the layout
     *
     * @param l memory layout
     */
    void setLayout(Layout l);

    /**
     * @brief Returns the memory precision
     *
     * @return precision
     */
    const Precision& getPrecision() const {
        return precision;
    }

    /**
     * @brief Sets the memory precision
     *
     * @param p precision
     */
    void setPrecision(const Precision& p) {
        precision = p;
    }

    /**
     * @brief Returns the blocking descriptor
     *
     * @return blocking descriptor
     */
    const BlockingDesc& getBlockingDesc() const {
        return blockingDesc;
    }

    /**
     * @brief The comparison operator for the TensorDesc
     *
     * @param rhs object to compare
     * @return true if objects are equal
     */
    bool operator==(const TensorDesc& rhs) const;
    /**
     * @brief The comparison operator for the TensorDesc
     *
     * @param rhs object to compare
     * @return true if objects aren't equal
     */
    bool operator!=(const TensorDesc& rhs) const;

    /**
     * @brief Calculates offset for the vector of dimensions
     *
     * @param v vector of dimensions
     * @return offset
     */
    size_t offset(const SizeVector& v) const;
    /**
     * @brief Calculates offset for the local offset
     *
     * @param l local offset
     * @return offset
     */
    size_t offset(size_t l) const;

    /**
     * @brief Returns the standard layout for dimensions
     *
     * @param dims the vector of dimensions
     * @return the standard memory layout
     */
    static Layout getLayoutByDims(const SizeVector& dims);

    /**
     * @brief Returns the standard layout for the specified tensor rank
     *
     * @param rank of the requested layout
     * @return the standard memory layout
     */
    static Layout getLayoutByRank(size_t rank);

private:
    /**
     * Memory layout
     */
    Layout layout;
    /**
     * @brief blob's dimensions
     */
    SizeVector dims;
    /**
     * @brief memory precision
     */
    Precision precision;
    /**
     * Detailed information about layout construction
     */
    BlockingDesc blockingDesc;
};

/**
 * @brief This structure describes ROI data for image-like tensors.
 */
struct ROI {
    size_t id = 0;      //!< ID of a ROI (offset over batch dimension)
    size_t posX = 0;    //!< W upper left coordinate of ROI
    size_t posY = 0;    //!< H upper left coordinate of ROI
    size_t sizeX = 0;   //!< W size of ROI
    size_t sizeY = 0;   //!< H size of ROI

    ROI() = default;

    /**
     * @brief Creates a ROI objects with given parameters
     * @param id ID of a ROI (offset over batch dimension)
     * @param posX W upper left coordinate of ROI
     * @param posY H upper left coordinate of ROI
     * @param sizeX W size of ROI
     * @param sizeY H size of ROI
     */
    ROI(size_t id, size_t posX, size_t posY, size_t sizeX, size_t sizeY) :
        id(id), posX(posX), posY(posY), sizeX(sizeX), sizeY(sizeY) {
    }
};

/**
 * @brief Creates a TensorDesc object for ROI.
 *
 * @param origDesc original TensorDesc object.
 * @param roi An image ROI object inside of the original object.
 * @param useOrigMemDesc Flag to use original memory description (strides/offset).
 *     Should be set if the new TensorDesc describes shared memory.
 *
 * @return A newly created TensorDesc object representing ROI.
 */
INFERENCE_ENGINE_API_CPP(TensorDesc) make_roi_desc(
        const TensorDesc& origDesc,
        const ROI& roi,
        bool useOrigMemDesc);

}  // namespace InferenceEngine
