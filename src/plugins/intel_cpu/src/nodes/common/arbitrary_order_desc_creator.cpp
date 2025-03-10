// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arbitrary_order_desc_creator.h"

#include "utils/general_utils.h"

namespace ov::intel_cpu {

ArbitraryOrderDescCreator::ArbitraryOrderDescCreator(VectorDims order) : m_order(std::move(order)) {
    OPENVINO_ASSERT(std::adjacent_find(m_order.begin(), m_order.end()) == m_order.end(),
                    "Can't construct ArbitraryOrderDescCreator, order vector contains repetitive elements",
                    vec2str(m_order));
}

CpuBlockedMemoryDesc ArbitraryOrderDescCreator::createDesc(const ov::element::Type& precision,
                                                           const Shape& srcShape) const {
    auto&& dims = srcShape.getDims();
    OPENVINO_ASSERT(dims.size() == m_order.size(),
                    "Couldn't create a tensor descriptor, shape and order size mismatch. Shape: ",
                    vec2str(dims),
                    " order: ",
                    vec2str(m_order));

    VectorDims blkDims(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        blkDims[i] = dims[m_order[i]];
    }

    return {precision, srcShape, blkDims, m_order};
}

size_t ArbitraryOrderDescCreator::getMinimalRank() const {
    return m_order.size();
}

}  // namespace ov::intel_cpu
