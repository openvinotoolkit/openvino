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
    Blocked,
    Mkldnn
};

enum class LayoutType : unsigned {
    nspc,      // general per channels format
    ncsp,      // general planar
    nCsp8c,    // general channels blocked by 8
    nCsp16c    // general channels blocked by 16
};

class MemoryDesc {
public:
    MemoryDescType getType() const {
        return type;
    }

    const Shape& getShape() const {
        return shape;
    }

    virtual ~MemoryDesc() = default;

    virtual InferenceEngine::Precision getPrecision() const = 0;

    virtual void setPrecision(InferenceEngine::Precision prc) = 0;

    virtual std::unique_ptr<MemoryDesc> clone() const = 0;

    virtual bool isCompatible(const MemoryDesc& rhs) const = 0;

    // Checks that all dimensions, offsets, strides, etc are defined (!= UNDEFINED_DIM)
    virtual bool isDefined() const = 0;

    virtual bool hasLayoutType(LayoutType layoutType) const = 0;

    virtual std::string serializeFormat() const = 0;

    /**
     * @brief Get minimal required memory size in bytes.
     * @return return minimal required memory size in bytes or UNDEFINED_SIZE in case undefined descriptor
     */
    size_t getCurrentSize() const {
        size_t retVal = UNDEFINED_SIZE;
        if (isDefined()) {
            retVal = getMemSizeImp();
        }
        return retVal;
    }

    template <typename T,
            typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
            typename std::enable_if<std::is_base_of<MemoryDesc, T>::value, int>::type = 0>
    T* as() {
        T* casted = dynamic_cast<T*>(this);
        if (!casted)
            IE_THROW() << "Cannot dynamically cast MemoryDesc";
        return casted;
    }

    template <typename T,
            typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
            typename std::enable_if<std::is_base_of<MemoryDesc, T>::value, int>::type = 0>
    const T* as() const {
        const T* casted = dynamic_cast<const T*>(this);
        if (!casted)
            IE_THROW() << "Cannot dynamically cast MemoryDesc";
        return casted;
    }

    static constexpr size_t UNDEFINED_SIZE = std::numeric_limits<size_t>::max();

protected:
    MemoryDesc(const Shape& shape, MemoryDescType type)
            : shape(shape), type(type) {}

    MemoryDesc(const std::vector<size_t>& dims, MemoryDescType type)
            : shape(dims), type(type) {}

    virtual size_t getMemSizeImp() const = 0;

    // Get offset to the n'th element. Returns physical index of the element by the logical one considering padding, layout, blocking etc.
    virtual size_t getElementOffset(size_t elemNumber) const = 0;

    MemoryDescType type;
    Shape shape;

    friend class BlobDumper;
    // WA: optimizedNspc2Ncsp used getElementOffset inside implementation
    friend class MKLDNNSplitNode;
};

using MemoryDescPtr = std::unique_ptr<MemoryDesc>;
using MemoryDescConstPtr = std::unique_ptr<const MemoryDesc>;

}  // namespace MKLDNNPlugin
