// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_shape.h"
#include "cpu_types.h"
#include "openvino/core/type/element_type.hpp"

/**
 * @brief
 *
 * MemoryDesc - the descriptor of tensor representation in memory. Describes all required information
 * for proper allocation and handling tensor in some buffer. The real memory is not present, just description.
 * This object answers on question how and where data with logical index [x1, x2, .. xN] placed in real buffer.
 * In the simplest case it describe a mapping between "logical offset" and "real offset".
 *
 */

namespace ov {
namespace intel_cpu {
namespace node {
class Split;
}   // namespace node

class MemoryDesc;

using MemoryDescPtr = std::shared_ptr<MemoryDesc>;
using MemoryDescCPtr = std::shared_ptr<const MemoryDesc>;
using VecMemoryDescs = std::vector<MemoryDescPtr>;

enum MemoryDescType {
    Undef = 0,
    Blocked = 1,
    Dnnl = 1 << 1,
    DnnlBlocked = Blocked | Dnnl,
    Empty = 1 << 2,
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

    virtual ov::element::Type getPrecision() const = 0;

    virtual MemoryDescPtr clone() const = 0;

    /**
     * @brief Returns the offset to the current memory block
     *
     * @return offset
     */
    virtual size_t getOffsetPadding() const = 0;

    /**
     * @brief Clone descriptor with new dims.
     * Throws an exception if relaxedCheck is false and some of the new dims conflicts with the internal shape (i.e. its defined dims ,rank, upper bounds)
     * or if internal shape and dims have different ranks
     * @param dims new dims
     * @param relaxedCheck flag which defined must we check dims with internal desc on compatibility
     * @return MemoryDescPtr with new dims
     */
    MemoryDescPtr cloneWithNewDims(const VectorDims& dims, bool relaxedCheck = false) const {
        if (relaxedCheck) {
            if (getShape().getRank() != dims.size()) {
                OPENVINO_THROW("ParameterMismatch: Can not clone with new dims, ranks mistmatch. Descriptor's rank: ",
                               getShape().getRank(),
                               " is incompatible with provided rank of dimensions: ",
                               dims.size(),
                               ".");
            }
        } else if (!getShape().isCompatible(dims)) {
            OPENVINO_THROW("ParameterMismatch: Can not clone with new dims. Descriptor's shape: ",
                           getShape().toString(),
                           " is incompatible with provided dimensions: ",
                           dims2str(dims),
                           ".");
        }

        return cloneWithNewDimsImp(dims);
    }

    virtual MemoryDescPtr cloneWithNewPrecision(const ov::element::Type prec) const = 0;

    virtual bool isCompatible(const MemoryDesc& rhs) const = 0;

    // Checks that all dimensions, offsets, strides, etc are defined (!= UNDEFINED_DIM)
    bool isDefined() const {
        if (descStatus::Unknown == status) {
            status = isDefinedImp() ? descStatus::Defined : descStatus::Undefined;
        }
        return descStatus::Defined == status;
    }

    virtual bool hasLayoutType(LayoutType layoutType) const = 0;

    virtual std::string serializeFormat() const = 0;

    // Get memory upper bound if possible. Can be undefined
    virtual size_t getMaxMemSize() const = 0;

    /**
     * @brief Get minimal required memory size in bytes.
     * @return return minimal required memory size in bytes or UNDEFINED_SIZE in case undefined descriptor
     */
    size_t getCurrentMemSize() const {
        size_t retVal = UNDEFINED_SIZE;
        if (canComputeMemSize()) {
            retVal = getCurrentMemSizeImp();
        }
        return retVal;
    }

    bool hasDefinedMaxSize() const {
        return getMaxMemSize() != MemoryDesc::UNDEFINED_SIZE;
    }

    bool empty() const {
        return type == Empty;
    }

    template <typename T,
            typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
            typename std::enable_if<std::is_base_of<MemoryDesc, T>::value, int>::type = 0>
    T* as() {
        T* casted = dynamic_cast<T*>(this);
        if (!casted)
            OPENVINO_THROW("Cannot dynamically cast MemoryDesc");
        return casted;
    }

    template <typename T,
            typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
            typename std::enable_if<std::is_base_of<MemoryDesc, T>::value, int>::type = 0>
    const T* as() const {
        const T* casted = dynamic_cast<const T*>(this);
        if (!casted)
            OPENVINO_THROW("Cannot dynamically cast MemoryDesc");
        return casted;
    }

    static constexpr size_t UNDEFINED_SIZE = std::numeric_limits<size_t>::max();

protected:
    MemoryDesc() : type(MemoryDescType::Undef) {}
    MemoryDesc(Shape shape, MemoryDescType type)
            : type(type), shape(std::move(shape)) {}

    MemoryDesc(const VectorDims& dims, MemoryDescType type)
            : type(type), shape(dims) {}

    virtual void setPrecision(ov::element::Type prc) = 0;

    virtual size_t getCurrentMemSizeImp() const = 0;

    // Get offset to the n'th element. Returns physical index of the element by the logical one considering padding, layout, blocking etc.
    virtual size_t getElementOffset(size_t elemNumber) const = 0;

    virtual bool canComputeMemSizeZeroDims() const = 0;
    virtual bool isDefinedImp() const = 0;

    bool canComputeMemSize() const {
        return isDefined() || canComputeMemSizeZeroDims();
    }

    virtual MemoryDescPtr cloneWithNewDimsImp(const VectorDims& dims) const = 0;

    MemoryDescType type;
    Shape shape;

    mutable enum class descStatus : uint8_t {
        Unknown,
        Defined,
        Undefined,
    } status = descStatus::Unknown;

    friend class BlobDumper;
    // WA: optimizedNspc2Ncsp used getElementOffset inside implementation
    friend class node::Split;
};

}   // namespace intel_cpu
}   // namespace ov
