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

    virtual ~MemoryDesc() = default;

    virtual std::unique_ptr<MemoryDesc> clone() const = 0;

    // InitTensorsAreEqual
    virtual bool isCompatible(const MemoryDesc& rhs) const = 0;

    // Checks that all dimensions, offsets, strides, etc are defined (!= UNDEFINED_DIM)
    virtual bool isDefined() const = 0;

    // Get offset to the n'th element. Returns physical index of the element by the logical one considering padding, layout, blocking etc.
    virtual size_t getOffset(size_t elemNumber) const = 0;

    // TODO [DS]: Can we generalize this? Like isCompatible(format::ncsp) or smt like that.
    bool isPlainFormat() const {
        IE_THROW() << "[DS] Unimplemented";
    }


    // Get minimal requared memory size in bytes. Can be undefined
    size_t getMemSize() const {
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

    void setPrecision(InferenceEngine::Precision prc) {
        precision = prc;
    }

protected:
    MemoryDesc() : shape(std::vector<size_t>()), precision(InferenceEngine::Precision::UNSPECIFIED), type(Undef) {}

    MemoryDesc(const Shape& shape, const InferenceEngine::Precision& precision, MemoryDescType type)
            : shape(shape), precision(precision), type(type) {}

    MemoryDesc(const std::vector<size_t>& dims, const InferenceEngine::Precision& precision, MemoryDescType type)
            : shape(dims), precision(precision), type(type) {}

    virtual size_t getMemSizeImp() const = 0;

public:
    static constexpr size_t UNDEFINED_SIZE = std::numeric_limits<size_t>::max();

protected:
    MemoryDescType type;
    Shape shape;
    InferenceEngine::Precision precision;
};

using MemoryDescPtr = std::unique_ptr<MemoryDesc>;
using MemoryDescConsPtr = std::unique_ptr<const MemoryDesc>;

}  // namespace MKLDNNPlugin
