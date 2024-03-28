// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_shape.h"
#include "utils/general_utils.h"

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
            output << dim2str(minDims[i]) << " - " << dim2str(maxDims[i]);
        } else {
            output << dims[i];
        }
    } while (++i < dims.size() && output << ", ");

    output << "}";
    return output.str();
}

Shape mergeShapes(const Shape& lhs, const Shape& rhs) {
    OPENVINO_ASSERT(lhs.getRank() == rhs.getRank(),
        "Couldn't merge shapes of different ranks: shape 1:",
        lhs.toString(),
        " shape 2: ",
        rhs.toString());

    const auto& lhsMinDims = lhs.getMinDims();
    const auto& lhsMaxDims = lhs.getMaxDims();
    const auto& rhsMinDims = rhs.getMinDims();
    const auto& rhsMaxDims = rhs.getMaxDims();

    VectorDims resultMinDims(lhsMinDims.size());
    VectorDims resultMaxDims(lhsMaxDims.size());

    for (size_t i = 0; i < resultMinDims.size(); ++i) {
        resultMinDims[i] = std::max(lhsMinDims[i], rhsMinDims[i]);
        resultMaxDims[i] = std::min(lhsMaxDims[i], rhsMaxDims[i]);
        OPENVINO_ASSERT(resultMinDims[i] <= resultMaxDims[i], "Couldn't merge shapes as the dims intervals are not overlapping.");
    }
    return Shape{resultMinDims, resultMaxDims};
}

}   // namespace intel_cpu
}   // namespace ov
