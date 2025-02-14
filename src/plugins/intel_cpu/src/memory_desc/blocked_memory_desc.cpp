// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blocked_memory_desc.h"

#include <unordered_set>

#include "utils/general_utils.h"

namespace ov::intel_cpu {

/* c++11 requires to have a definition in cpp file */
constexpr BlockedMemoryDesc::CmpMask BlockedMemoryDesc::FULL_MASK;
constexpr BlockedMemoryDesc::CmpMask BlockedMemoryDesc::EMPTY_MASK;
constexpr BlockedMemoryDesc::CmpMask BlockedMemoryDesc::SKIP_OFFSET_MASK;
constexpr size_t BlockedMemoryDesc::OFFSET_MASK_POS;

bool BlockedMemoryDesc::isCompatibleInternal(const BlockedMemoryDesc& rhs, CmpMask cmpMask) const {
    if (this->getShape() != rhs.getShape() || this->getPrecision() != rhs.getPrecision()) {
        return false;
    }

    if (!dimsEqualWeak(this->getBlockDims(), rhs.getBlockDims())) {
        return false;
    }

    if (!dimsEqualWeak(this->getOffsetPaddingToData(), rhs.getOffsetPaddingToData())) {
        return false;
    }

    auto& thisStrides = this->getStrides();
    auto& rhsStrides = rhs.getStrides();

    if (thisStrides.size() != rhsStrides.size()) {
        return false;
    }

    for (size_t i = 0; i < thisStrides.size(); i++) {
        if (cmpMask.test(i) && !dimsEqualWeak(thisStrides[i], rhsStrides[i])) {
            return false;
        }
    }

    if (!dimsEqualWeak(this->getOrder(), rhs.getOrder())) {
        return false;
    }

    if (cmpMask.test(OFFSET_MASK_POS)) {
        return dimsEqualWeak(this->getOffsetPadding(), rhs.getOffsetPadding());
    }

    return true;
}

std::string BlockedMemoryDesc::serializeFormat() const {
    std::stringstream result;
    char startLetter = 'a';
    std::unordered_set<size_t> blockedAxis;
    const auto& order = getOrder();
    const auto& shape = getShape();
    for (size_t i = shape.getRank(); i < order.size(); ++i) {
        blockedAxis.insert(order[i]);
    }

    for (size_t i = 0; i < shape.getRank(); ++i) {
        char nextLetter = startLetter + order[i];
        if (blockedAxis.count(i)) {
            nextLetter = toupper(nextLetter);
        }
        result << nextLetter;
    }

    const auto& blkDims = getBlockDims();
    for (size_t i = shape.getRank(); i < order.size(); ++i) {
        result << blkDims[i] << static_cast<char>(startLetter + order[i]);
    }

    return result.str();
}

}  // namespace ov::intel_cpu
