// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_shape.h"
#include "utils/general_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"

namespace ov {
namespace intel_cpu {

bool Shape::isCompatible(const VectorDims &vecDims) const {
    if (getRank() != vecDims.size()) {
        return false;
    }

    auto comparator = [](Dim lhs, Dim rhs) {
        return (lhs == rhs) || (lhs == Shape::UNDEFINED_DIM);
    };

    if (!std::equal(getDims().begin(), getDims().end(), vecDims.begin(), comparator)) {
        return false;
    }

    if (!std::equal(getMaxDims().begin(), getMaxDims().end(), vecDims.begin(), [](Dim lhs, Dim rhs) { return lhs >= rhs; })) {
        return false;
    }

    if (!std::equal(getMinDims().begin(), getMinDims().end(), vecDims.begin(), [](Dim lhs, Dim rhs) { return lhs <= rhs; })) {
        return false;
    }
    return true;
}

std::string Shape::toString() const  {
    std::stringstream output;
    output << "{";

    size_t i = 0;
    do {
        if (dims[i] == Shape::UNDEFINED_DIM) {
            output << MemoryDescUtils::dim2str(minDims[i]) << " - " << MemoryDescUtils::dim2str(maxDims[i]);
        } else {
            output << dims[i];
        }
    } while (++i < dims.size() && output << ", ");

    output << "}";
    return output.str();
}

}   // namespace intel_cpu
}   // namespace ov
