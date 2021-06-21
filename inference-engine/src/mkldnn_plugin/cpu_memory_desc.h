// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <ie_precision.hpp>
#include "cpu_shape.h"
#include "utils/general_utils.h"

namespace MKLDNNPlugin {

enum MemoryDescType {
    Undef,
    Blocked,
    Mkldnn
};

// TODO [DS]: Seems like should be pure class
class MemoryDesc {
public:
    MemoryDescType getType() const {
        return type;
    }

    const Shape& getShape() const {
        return shape;
    }

    const InferenceEngine::Precision& getPrecision() const {
        return precision;
    }

    virtual std::unique_ptr<MemoryDesc> clone() const = 0;

    // InitTensorsAreEqual
    virtual bool isCompatible(const MemoryDesc& rhs) const = 0;

    // Checks that all dimensions, offsets, strides, etc are defined (!= UNDEFINED_DIM)
    virtual bool isDefined() const = 0;

    // TODO [DS]: Can we generalize this? Like isCompatible(format::ncsp) or smt like that.
    bool isPlainFormat() const {
        IE_THROW() << "[DS] Unimplemented";
    }

    template <typename T,
            typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
            typename std::enable_if<std::is_base_of<MemoryDesc, T>::value, int>::type = 0>
    T* as() {
        T* casted = dynamic_cast<T*>(this);
        if (!casted)
            IE_THROW() << "Cannot dynamically cast MemoryDesc";
        return dynamic_cast<T*>(this);
    }

    template <typename T,
            typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
            typename std::enable_if<std::is_base_of<MemoryDesc, T>::value, int>::type = 0>
    const T* as() const {
        T* casted = dynamic_cast<T*>(this);
        if (!casted)
            IE_THROW() << "Cannot dynamically cast MemoryDesc";
        return dynamic_cast<const T*>(this);
    }

    const void setPrecision(InferenceEngine::Precision prc) {
        precision = prc;
    }

protected:
    MemoryDesc() : shape(std::vector<size_t>()), precision(InferenceEngine::Precision::UNSPECIFIED), type(Undef) {}

    MemoryDesc(const Shape& shape, const InferenceEngine::Precision& precision, MemoryDescType type)
            : shape(shape), precision(precision), type(type) {}

    MemoryDesc(const std::vector<size_t>& dims, const InferenceEngine::Precision& precision, MemoryDescType type)
            : shape(dims), precision(precision), type(type) {}

    MemoryDescType type;
    Shape shape;
    InferenceEngine::Precision precision;
};

class BlockedMemoryDesc : public MemoryDesc {
public:
    BlockedMemoryDesc(InferenceEngine::Precision prc, const std::vector<size_t>& dims) : MemoryDesc(dims, prc, Blocked) {
        order.resize(dims.size());
        std::iota(order.begin(), order.end(), 0);
        blockedDims = dims;
        offsetPadding = 0;
        offsetPaddingToData.resize(dims.size(), 0);
        strides.resize(order.size());
        strides[strides.size() - 1] = 1;
        for (size_t i = 2; i <= order.size(); i++) {
            strides[strides.size() - i] = strides[strides.size() - (i - 1)] * blockedDims[blockedDims.size() - (i - 1)];
        }
    }

    BlockedMemoryDesc(InferenceEngine::Precision prc, const std::vector<size_t>& dims, const std::vector<size_t>& blockedDims,
                      const std::vector<size_t>& order, size_t offsetPadding = 0, const std::vector<size_t>& offsetPaddingToData = {},
                      const std::vector<size_t>& strides = {}) : MemoryDesc(dims, prc, Blocked) {
        this->order = order;
        this->blockedDims = blockedDims;
        this->offsetPadding = offsetPadding;

        if (offsetPaddingToData.empty()) {
            this->offsetPaddingToData.resize(order.size());
            this->offsetPaddingToData[order.size() - 1] = 0;
            for (size_t i = 2; i <= order.size(); i++) {
                this->offsetPaddingToData[order.size() - i] = 0;
            }
        } else {
            this->offsetPaddingToData = offsetPaddingToData;
        }

        if (strides.empty()) {
            this->strides.resize(order.size());
            this->strides[order.size() - 1] = 1;
            for (size_t i = 2; i <= order.size(); i++) {
                this->strides[order.size() - i] = this->strides[order.size() - (i - 1)] * this->blockedDims[blockedDims.size() - (i - 1)];
            }
        } else {
            this->strides = strides;
        }
    }

    std::unique_ptr<MemoryDesc> clone() const override {
        return make_unique<BlockedMemoryDesc>(*this);
    }

    bool isDefined() const override {
        // TODO [DS]: Introduce isDefined status into base class to speedup the method

        bool defined = true;
        defined = defined && std::none_of(blockedDims.cbegin(), blockedDims.cend(), [](size_t val) { return val == Shape::UNDEFINED_DIM; });
        defined = defined && std::none_of(strides.cbegin(), strides.cend(), [](size_t val) { return val == Shape::UNDEFINED_DIM; });
        defined = defined && std::none_of(order.cbegin(), order.cend(), [](size_t val) { return val == Shape::UNDEFINED_DIM; });
        defined = defined && std::none_of(offsetPaddingToData.cbegin(), offsetPaddingToData.cend(), [](size_t val) { return val == Shape::UNDEFINED_DIM; });
        defined = defined && offsetPadding != Shape::UNDEFINED_DIM;

        return defined;
    }

    bool isCompatible(const MemoryDesc& rhs) const override {
        if (auto blockingDesc = dynamic_cast<const BlockedMemoryDesc*>(&rhs)) {
            return isCompatible(*blockingDesc);
        } else {
            IE_THROW() << "Cannot check compatibility with this type of memory descriptor";
        }

        return false;
    }

    bool isCompatible(const BlockedMemoryDesc& rhs) const {
        auto isEqualOrUndefined = [](const std::vector<size_t> lhs, const std::vector<size_t>& rhs) {
            if (lhs.size() != rhs.size())
                return false;
            
            for (size_t i = 0; i < lhs.size(); i++) {
                if (lhs[i] != rhs[i] && lhs[i] != Shape::UNDEFINED_DIM && rhs[i] != Shape::UNDEFINED_DIM)
                    return false;
            }

            return true;
        };
        
        if (this->getShape() != rhs.getShape() || this->getPrecision() != rhs.getPrecision())
            return false;

        if (!isEqualOrUndefined(this->getBlockDims(), rhs.getBlockDims())) {
            return false;
        }

        if (!isEqualOrUndefined(this->getOffsetPaddingToData(), rhs.getOffsetPaddingToData())) {
            return false;
        }

        if (!isEqualOrUndefined(this->getStrides(), rhs.getStrides())) {
            return false;
        }

        if (this->getOrder() != rhs.getOrder()) {
            return false;
        }

        return !(this->getOffsetPadding() != rhs.getOffsetPadding() &&
                 this->getOffsetPadding() != Shape::UNDEFINED_DIM && rhs.getOffsetPadding() != Shape::UNDEFINED_DIM);
    }

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

private:
    std::vector<size_t> blockedDims;
    std::vector<size_t> strides;
    std::vector<size_t> order;
    std::vector<size_t> offsetPaddingToData;
    size_t offsetPadding;
};

}  // namespace MKLDNNPlugin
