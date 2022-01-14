// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blocked_memory_desc.h"
#include "utils/general_utils.h"

using namespace MKLDNNPlugin;

bool BlockedMemoryDesc::isCompatibleInternal(const BlockedMemoryDesc &rhs, uint32_t cmpMask) const {
    if (this->getShape() != rhs.getShape() || this->getPrecision() != rhs.getPrecision())
        return false;

    if (!dimsEqualWeak(this->getBlockDims(), rhs.getBlockDims())) {
        return false;
    }

    if (!dimsEqualWeak(this->getOffsetPaddingToData(), rhs.getOffsetPaddingToData())) {
        return false;
    }

    if (!dimsEqualWeak(this->getStrides(), rhs.getStrides())) {
        return false;
    }

    auto& thisStrides = this->getStrides();
    auto& rhsStrides = rhs.getStrides();

    if (thisStrides.size() != rhsStrides.size())
        return false;

    for (size_t i = 0; i < thisStrides.size(); i++) {
        if ((cmpMask & (1u << i)) && !dimsEqualWeak(thisStrides[i], rhsStrides[i]))
            return false;
    }

    if (!dimsEqualWeak(this->getOrder(), rhs.getOrder())) {
        return false;
    }

    constexpr uint32_t offsetMask = 0x80000000u;
    if (cmpMask & offsetMask) {
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
        result << blkDims[i] << char(startLetter + order[i]);
    }

    return result.str();
}
