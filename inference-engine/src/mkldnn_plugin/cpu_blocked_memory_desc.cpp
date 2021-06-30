// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_blocked_memory_desc.h"
#include "mkldnn_memory.h"
#include "utils/cpu_utils.hpp"

using namespace MKLDNNPlugin;

BlockedMemoryDesc::BlockedMemoryDesc(InferenceEngine::Precision prc, const std::vector<size_t>& dims) : MemoryDesc(dims, prc, Blocked) {
    order.resize(dims.size());
    std::iota(order.begin(), order.end(), 0);
    blockedDims = dims;
    offsetPadding = 0;
    offsetPaddingToData.resize(dims.size(), 0);
    strides.resize(order.size());
    strides[strides.size() - 1] = 1;
    for (size_t i = 2; i <= order.size(); i++) {
        strides[strides.size() - i] = strides[strides.size() - (i - 1)] * blockedDims[blockedDims.size() - (i - 1)];
    }
}

BlockedMemoryDesc::BlockedMemoryDesc(InferenceEngine::Precision prc, const std::vector<size_t>& dims, const std::vector<size_t>& blockedDims,
                  const std::vector<size_t>& order, size_t offsetPadding, const std::vector<size_t>& offsetPaddingToData,
                  const std::vector<size_t>& strides) : MemoryDesc(dims, prc, Blocked) {
    if (std::any_of(order.begin(), order.end(), [](size_t val) { return val == Shape::UNDEFINED_DIM; })) {
        IE_THROW() << "BlockedMemoryDesc do not support undefined order.";
    }

    if (std::any_of(blockedDims.begin() + dims.size(), blockedDims.end(), [](size_t val) { return val == Shape::UNDEFINED_DIM; })) {
        IE_THROW() << "BlockedMemoryDesc do not support undefined blocks.";
    }

    this->order = order;
    this->blockedDims = blockedDims;
    this->offsetPadding = offsetPadding;

    if (offsetPaddingToData.empty() && !order.empty()) {
        this->offsetPaddingToData.resize(order.size());
        this->offsetPaddingToData[order.size() - 1] = 0;
        for (size_t i = 2; i <= order.size(); i++) {
            this->offsetPaddingToData[order.size() - i] = 0;
        }
    } else {
        this->offsetPaddingToData = offsetPaddingToData;
    }

    if (strides.empty() && !order.empty()) {
        if (std::any_of(this->blockedDims.begin(), this->blockedDims.end(), [](size_t val) { return val == Shape::UNDEFINED_DIM; })) {
            this->strides.resize(order.size(), Shape::UNDEFINED_DIM);
        } else {
            this->strides.resize(order.size());
            this->strides[order.size() - 1] = 1;
            for (size_t i = 2; i <= order.size(); i++) {
                this->strides[order.size() - i] = this->strides[order.size() - (i - 1)] * this->blockedDims[blockedDims.size() - (i - 1)];
            }
        }
    } else {
        this->strides = strides;
    }
}

bool BlockedMemoryDesc::isDefined() const {
    // TODO [DS]: Introduce isDefined status into base class to speedup the method

    bool defined = true;
    defined = defined && std::none_of(blockedDims.cbegin(), blockedDims.cend(), [](size_t val) { return val == Shape::UNDEFINED_DIM; });
    defined = defined && std::none_of(strides.cbegin(), strides.cend(), [](size_t val) { return val == Shape::UNDEFINED_DIM; });
    defined = defined && std::none_of(order.cbegin(), order.cend(), [](size_t val) { return val == Shape::UNDEFINED_DIM; });
    defined = defined && std::none_of(offsetPaddingToData.cbegin(), offsetPaddingToData.cend(), [](size_t val) { return val == Shape::UNDEFINED_DIM; });
    defined = defined && offsetPadding != Shape::UNDEFINED_DIM;

    return defined;
}

bool BlockedMemoryDesc::isCompatible(const MemoryDesc& rhs) const {
    const MemoryDesc* pRhs = &rhs;
    if (auto blockingDesc = dynamic_cast<const BlockedMemoryDesc*>(pRhs)) {
        return isCompatible(*blockingDesc);
    } else if (auto mkldnnDesc = dynamic_cast<const MKLDNNMemoryDesc*>(pRhs)) {
        return mkldnnDesc->isCompatible(*this);
    } else {
        //IE_THROW() << "Cannot check compatibility with this type of memory descriptor";
        return false;
    }
}

bool BlockedMemoryDesc::isCompatible(const BlockedMemoryDesc& rhs) const {
    if (this->getShape() != rhs.getShape() || this->getPrecision() != rhs.getPrecision())
        return false;

    if (!isEqualOrUndefined(this->getBlockDims(), rhs.getBlockDims())) {
        return false;
    }

    if (!isEqualOrUndefined(this->getOffsetPaddingToData(), rhs.getOffsetPaddingToData())) {
        return false;
    }

    if (!isEqualOrUndefined(this->getStrides(), rhs.getStrides())) {
        return false;
    }

    if (this->getOrder() != rhs.getOrder()) {
        return false;
    }

    return !(this->getOffsetPadding() != rhs.getOffsetPadding() &&
             this->getOffsetPadding() != Shape::UNDEFINED_DIM && rhs.getOffsetPadding() != Shape::UNDEFINED_DIM);
}

size_t BlockedMemoryDesc::getMemSizeImp() const {
    int64_t e_size = getOffsetPadding() + 1;  // size in bytes (from begin of data to last element)
    for (int j = 0; j < getBlockDims().size(); j++)
        e_size += (getBlockDims()[j] - 1) * getStrides()[j];

    // In some cases computational formula above doesn't work properly (e.g. for OhIw8o4i layout).
    // This WA allows to limit the size of allocated memory from below.
    // TODO: need to properly investigate the root cause of incorrect computations
    int64_t min_size = 1;
    for (int64_t dim : getBlockDims()) {
        min_size *= dim;
    }
    e_size = std::max(e_size, min_size);

    e_size *= getPrecision() == InferenceEngine::Precision::BIN ? 1 : getPrecision().size();

    return e_size;
}

size_t BlockedMemoryDesc::getOffset(const InferenceEngine::SizeVector& v) const {
    InferenceEngine::SizeVector off_v = v;

    size_t n_blocked_dims = order.size();
    if (blockedDims.size() != n_blocked_dims || strides.size() != n_blocked_dims) {
        IE_THROW() << "Cannot calculate offset. Incorrect primitive descriptor!";
    }
    InferenceEngine::SizeVector blockedShift(n_blocked_dims);
    for (size_t i = 1; i <= n_blocked_dims; i++) {
        blockedShift[n_blocked_dims - i] = off_v[order[n_blocked_dims - i]] % blockedDims[n_blocked_dims - i];
        off_v[order[n_blocked_dims - i]] /= blockedDims[n_blocked_dims - i];
    }
    size_t offset = getOffsetPadding();
    for (size_t d = 0; d < n_blocked_dims; ++d) {
        const size_t p = blockedShift[d] + getOffsetPaddingToData()[d];
        offset += p * strides[d];
    }
    return offset;
}

size_t BlockedMemoryDesc::getOffset(size_t elemNumber) const {
    // TODO [mkutakov]: rewrite to support dynamic shapes
    auto& dims = shape.getStaticDims();
    size_t n_dims = dims.size();
    InferenceEngine::SizeVector pos(n_dims);
    for (size_t rd = 1; rd <= n_dims; ++rd) {
        const size_t d = n_dims - rd;
        const size_t cur_dim = dims[d];
        pos[d] = elemNumber % cur_dim;
        elemNumber /= cur_dim;
    }
    return getOffset(pos);
}

bool BlockedMemoryDesc::checkGeneralLayout(GeneralLayout layoutType) const {
    switch (layoutType) {
        case GeneralLayout::ncsp:
            return isPlainFormat();
        case GeneralLayout::nspc:
            return isTailCFormat();
        case GeneralLayout::nCsp8c:
            return isBlockedCFormat(8);
        case GeneralLayout::nCsp16c:
            return isBlockedCFormat(16);
        default:
            return false;
    }
}

bool BlockedMemoryDesc::isPlainFormat() const {
    if (shape.getRank() != order.size()) {
        return false;
    }
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] != i) {
            return false;
        }
    }
    return true;
}

bool BlockedMemoryDesc::isBlockedCFormat(size_t blk_size) const {
    if ((order.size() - shape.getRank()) != 1) {
        return false;
    }
    for (size_t i = 0; i < order.size() - 1; ++i) {
        if (order[i] != i) {
            return false;
        }
    }
    if (order.back() != 1) {
        return false;
    }
    if (blockedDims.back() != blk_size) {
        return false;
    }
    return true;
}

bool BlockedMemoryDesc::isTailCFormat() const {
    if (shape.getRank() < 3) {
        return false;
    }
    if (shape.getRank() != order.size()) {
        return false;
    }
    if (!std::is_sorted(order.begin(), order.end()--)) {
        return false;
    }
    if (order.back() != 1) {
        return false;
    }
    return true;
}
