// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "cpu_memory_desc.h"
#include "cpu_shape.h"
#include "cpu_types.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

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
        return all_of(this->getType(), rhs.getType(), Empty);
    };

    ov::element::Type getPrecision() const override {
        return ov::element::dynamic;
    }

    size_t getOffsetPadding() const override {
        return 0;
    }

    bool hasLayoutType([[maybe_unused]] LayoutType layoutType) const override {
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
    size_t getElementOffset([[maybe_unused]] size_t elemNumber) const override {
        return 0;
    }
    bool canComputeMemSizeZeroDims() const override {
        return false;
    }
    size_t getCurrentMemSizeImp() const override {
        return 0;
    }
    static size_t getOffset([[maybe_unused]] const VectorDims& v) {
        return 0;
    }
    bool isDefinedImp() const override {
        return true;
    }
    MemoryDescPtr cloneWithNewDimsImp([[maybe_unused]] const VectorDims& dims) const override {
        OPENVINO_THROW("Clone an empty memory desc with any new dimensions is prohibited");
    }

    void setPrecision(ov::element::Type prc) override {
        OPENVINO_THROW("Setting any precision (", prc, ") for an empty memory desc is prohibited");
    }
};

using EmptyMemoryDescPtr = std::shared_ptr<EmptyMemoryDesc>;
using EmptyMemoryDescCPtr = std::shared_ptr<const EmptyMemoryDesc>;

}  // namespace ov::intel_cpu
