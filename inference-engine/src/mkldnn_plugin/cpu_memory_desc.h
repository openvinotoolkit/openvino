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
        return casted;
    }

    template <typename T,
            typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
            typename std::enable_if<std::is_base_of<MemoryDesc, T>::value, int>::type = 0>
    const T* as() const {
        T* casted = dynamic_cast<T*>(this);
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

    MemoryDescType type;
    Shape shape;
    InferenceEngine::Precision precision;
};

using MemoryDescPtr = std::unique_ptr<MemoryDesc>;
using MemoryDescConsPtr = std::unique_ptr<const MemoryDesc>;

}  // namespace MKLDNNPlugin
