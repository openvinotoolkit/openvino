// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for data layouts and conversion between them
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
     * @param dims real dimensions
     * @param layout memory layout
     */
    BlockingDesc(const SizeVector& dims, Layout layout);
    /**
     * @brief The constructor allows to create blocking descriptors for blocked memory
     * @param blocked_dims blocked dimensions
     * @param order the order of dimensions
     */
    BlockingDesc(const SizeVector& blocked_dims, const SizeVector& order);
    /**
     * @brief The constructor allows to create blocking descriptors for blocked memory
     * @param blocked_dims blocked dimensions
     * @param order the order of dimensions
     * @param offset offset to the current memory block
     */
    BlockingDesc(const SizeVector& blocked_dims, const SizeVector& order, size_t offset);
    /**
     * @brief The constructor allows to create blocking descriptors for blocked memory
     * @param blocked_dims blocked dimensions
     * @param order the order of dimensions
     * @param offset offset to the current memory block
     * @param dimOffsets per-dimension offset from the padding to actual data,
     */
    BlockingDesc(const SizeVector& blocked_dims, const SizeVector& order, size_t offset, SizeVector dimOffsets);
    /**
     * @brief The constructor allows to create blocking descriptors for blocked memory
     * @param blocked_dims blocked dimensions
     * @param order the order of dimensions
     * @param offset offset to the current memory block
     * @param dimOffsets per-dimension offset from the padding to actual data,
     * @param strides strides for each dimension
     */
    BlockingDesc(const SizeVector& blocked_dims, const SizeVector& order, size_t offset, SizeVector dimOffsets, SizeVector strides);
    /**
     * @brief Destructor
     */
    virtual ~BlockingDesc() = default;

    /**
     * @brief Returns the blocked dimensions vector
     * @return blocked dimensions
     */
    const SizeVector& getBlockDims() const {
        return blockedDims;
    }

    /**
     * @brief Returns the vector of order
     * @return order
     */
    const SizeVector& getOrder() const {
        return order;
    }

    /**
     * @brief Returns the per-dimension offset vector
     * @return offsets
     */
    const SizeVector& getOffsetPaddingToData() const {
        return offsetPaddingToData;
    }

    /**
     * @brief Returns the offset to the current memory block
     * @return offset
     */
    size_t getOffsetPadding() const {
        return offsetPadding;
    }

    /**
     * @brief Returns strides for each dimension
     * @return strides
     */
    const SizeVector& getStrides() const {
        return strides;
    }

    /**
     * @brief The comparison operator for the BlockingDesc
     * @param rhs object to compare
     * @return true if objects are equal
     */
    bool operator == (const BlockingDesc& rhs) const;
    /**
     * @brief The comparison operator for the BlockingDesc
     * @param rhs object to compare
     * @return true if objects aren't equal
     */
    bool operator != (const BlockingDesc& rhs) const;

protected:
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
     * @param precision memory precision
     * @param dims memory dimensions
     * @param blockDesc blocking descriptor
     */
    TensorDesc(const Precision& precision, SizeVector dims, const BlockingDesc& blockDesc);
    /**
     * @brief The constructor creates the tensor descriptor using standard layout
     * @param precision memory precision
     * @param dims memory dimensions
     * @param layout memory layout
     */
    TensorDesc(const Precision& precision, SizeVector dims, Layout layout);
    /**
     * @brief The constructor creates the empty tensor descriptor with precision and layout
     * @param precision memory precision
     * @param layout memory layout
     */
    TensorDesc(const Precision& precision, Layout layout);
    /**
     * @brief The default constructor which creates empty tensor descriptor
     */
    TensorDesc();
    /**
     * @brief Destructor
     */
    virtual ~TensorDesc() = default;

    /**
     * @brief Reshapes the tensor descriptor
     * @param dims new dimensions
     * @param layout new layout if it is necessary
     */
    void reshape(const SizeVector &dims, Layout layout = Layout::ANY);
    /**
     * @brief Reshapes the tensor descriptor
     * @param dims new dimensions
     * @param blockDesc new blocking descriptor
     */
    void reshape(const SizeVector &dims, const BlockingDesc &blockDesc);

    /**
     * @brief Returns the vector of dimensions
     * @return dimensions
     */
    SizeVector& getDims() {
        return dims;
    }
    /**
     * @brief Returns the constant vector of dimensions
     * @return dimensions
     */
    const SizeVector& getDims() const noexcept {
        return dims;
    }
    /**
     * @brief Sets dimensions
     * @param dims new dimensions
     */
    void setDims(const SizeVector& dims);

    /**
     * @brief Returns the memory layout
     * @return layout
     */
    Layout getLayout() const {
        return layout;
    }

    /**
     * @brief Sets the layout
     * @param l memory layout
     */
    void setLayout(Layout l) {
        bool inconsistentLayout = true;
        switch (l) {
            case Layout::SCALAR:
                inconsistentLayout = !dims.empty();
                break;
            case Layout::C:
                inconsistentLayout = dims.size() != 1;
                break;
            case Layout::BLOCKED:
            case Layout::ANY:
                inconsistentLayout = false;
                break;
            case Layout::NCDHW:
            case Layout::NDHWC:
                inconsistentLayout = dims.size() != 5;
                break;
            case Layout::OIHW:
            case Layout::NCHW:
            case Layout::NHWC:
                inconsistentLayout = dims.size() != 4;
                break;
            case Layout::CHW:
                inconsistentLayout = dims.size() != 3;
                break;
            case Layout::CN:
            case Layout::NC:
            case Layout::HW:
                inconsistentLayout = dims.size() != 2;
                break;
            default:
                break;
        }
        if (inconsistentLayout)
            THROW_IE_EXCEPTION << "Size of dims(" << std::to_string(dims.size()) << ") and format(" << l << ") are inconsistent.";
        layout = l;
    }

    /**
     * @brief Returns the memory precision
     * @return precision
     */
    const Precision& getPrecision() const {
        return precision;
    }

    /**
     * @brief Sets the memory precision
     * @param p precision
     */
    void setPrecision(const Precision& p) {
        precision = p;
    }

    /**
     * @brief Returns the blocking descriptor
     * @return blocking descriptor
     */
    const BlockingDesc& getBlockingDesc() const {
        return blockingDesc;
    }

    /**
     * @brief The comparison operator for the TensorDesc
     * @param rhs object to compare
     * @return true if objects are equal
     */
    bool operator == (const TensorDesc& rhs) const;
    /**
     * @brief The comparison operator for the TensorDesc
     * @param rhs object to compare
     * @return true if objects aren't equal
     */
    bool operator != (const TensorDesc& rhs) const;

    /**
     * @brief Calculates offset for the vector of dimensions
     * @param v vector of dimensions
     * @return offset
     */
    size_t offset(const SizeVector& v) const;
    /**
     * @brief Calculates offset for the local offset
     * @param l local offset
     * @return offset
     */
    size_t offset(size_t l) const;

    /**
     * @brief Returns the standard layout for dimensions
     * @param dims the vector of dimensions
     * @return the standard memory layout
     */
    static Layout getLayoutByDims(SizeVector dims);
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
 * @deprecated Deprecated since provides dims in reverse order
 */
INFERENCE_ENGINE_DEPRECATED
static const size_t I_N = 3;

/**
 * @deprecated Deprecated since provides dims in reverse order
 */
INFERENCE_ENGINE_DEPRECATED
static const size_t I_C = 2;

/**
 * @deprecated Deprecated since provides dims in reverse order
 */
INFERENCE_ENGINE_DEPRECATED
static const size_t I_H = 1;

/**
 * @deprecated Deprecated since provides dims in reverse order
 */
INFERENCE_ENGINE_DEPRECATED
static const size_t I_W = 0;

/**
 * @deprecated Uses TensorDesc working with layouts
 * @brief This class helps calculating offset in different layouts
 */
class INFERENCE_ENGINE_DEPRECATED INFERENCE_ENGINE_API_CLASS(LayoutOffsetCounter) {
private:
    Layout _layout;
    SizeVector _dims;

    size_t _dims_count;

    /**
     * @brief Stores multipliers that are calculated during the LayoutOffsetCounter construction.
     * The multipliers are used for conversion.
     */
    SizeVector _muls;
public:
    /**
     * @brief A default constructor
     * @param dims Tensor dimension array (reverse NCHW order as in the IR: w,h,c,n)
     */
    LayoutOffsetCounter(Layout layout, SizeVector dims);

    IE_SUPPRESS_DEPRECATED_START
    /**
     * @brief A copy constructor
     */
    LayoutOffsetCounter(const LayoutOffsetCounter & l);

    /**
     * @brief A copy assignment operator
     * @param l A value to copy from
     */
    LayoutOffsetCounter & operator = (const LayoutOffsetCounter & l);
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief A destructor
     */
    ~LayoutOffsetCounter();

    /**
     * @brief Calculates an offset for the specified layout
     * @param pos Tensor position array (reverse NCHW order as in the IR: w,h,c,n)
     */
    size_t Offset(SizeVector pos);
};

/**
 * @deprecated Please use TensorDesc for conversion
 */
template<typename T>
INFERENCE_ENGINE_DEPRECATED
void ConvertLayout(Layout sourceLayout, Layout destLayout, const T* sourceBuffer, T* destBuffer, SizeVector dims) {
    IE_SUPPRESS_DEPRECATED_START
    if (dims.size() == 0) return;

    SizeVector pos(dims.size(), 0);
    LayoutOffsetCounter srcOffsetCounter(sourceLayout, dims);
    LayoutOffsetCounter destOffsetCounter(destLayout, dims);

    while (true) {
        // Setting the current item
        size_t ps = srcOffsetCounter.Offset(pos);
        size_t pd = destOffsetCounter.Offset(pos);

        destBuffer[pd] = sourceBuffer[ps];

        // Advancing pos
        size_t caret = 0;
        pos[caret]++;
        while (pos[caret] >= dims[caret]) {
            pos[caret] = 0;
            caret++;
            if (caret == pos.size()) {
                // We have finished converting
                return;
            }
            pos[caret]++;
        }
    }
    IE_SUPPRESS_DEPRECATED_END
}

}  // namespace InferenceEngine
