// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory_desc.h"
#include "cpu_shape.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {

/**
 * @brief Represents an empty memory descriptor.
 *
 * The main purpose is to create an empty Memory.
 * Empty Memory is used to generalize passing an optional memory (such as bias)
 * when both pointer to the memory data and nullptr are valid
 */
class EmptyMemoryDesc : public MemoryDesc {
public:
    EmptyMemoryDesc() : MemoryDesc(Shape{0}, Empty) {
        /* status never changes for an empty memory desc
         * so "define" beforehand to ensure isDefined() is thread safe */
        status = MemoryDesc::descStatus::Defined;
    }

    MemoryDescPtr clone() const override {
        return std::make_shared<EmptyMemoryDesc>(*this);
    }

    bool isCompatible(const MemoryDesc& rhs) const override {
        return everyone_is(this->getType(), rhs.getType(), Empty);
    };

    ov::element::Type getPrecision() const override {
        return ov::element::dynamic;
    }

    size_t getOffsetPadding() const override {
        return 0;
    }

    bool hasLayoutType(LayoutType layoutType) const override {
        return false;
    }

    std::string serializeFormat() const override {
        return "empty";
    }

    size_t getMaxMemSize() const override {
        return 0;
    }

    MemoryDescPtr cloneWithNewPrecision(const ov::element::Type prec) const override {
        OPENVINO_ASSERT(prec == ov::element::dynamic,
                        "Clone an empty memory desc with defined precision: ",
                        prec,
                        " is prohibited");
        return clone();
    }

private:
    size_t getElementOffset(size_t elemNumber) const override {
        return 0;
    }
    bool canComputeMemSizeZeroDims() const override {
        return false;
    }
    size_t getCurrentMemSizeImp() const override {
        return 0;
    }
    size_t getOffset(const VectorDims& v) const {
        return 0;
    }
    bool isDefinedImp() const override {
        return true;
    }
    MemoryDescPtr cloneWithNewDimsImp(const VectorDims& dims) const override {
        OPENVINO_THROW("Clone an empty memory desc with any new dimensions is prohibited");
    }

    void setPrecision(ov::element::Type prc) override {
        OPENVINO_THROW("Setting any precision (", prc, ") for an empty memory desc is prohibited");
    }
};

using EmptyMemoryDescPtr = std::shared_ptr<EmptyMemoryDesc>;
using EmptyMemoryDescCPtr = std::shared_ptr<const EmptyMemoryDesc>;

}  // namespace intel_cpu
}  // namespace ov
