// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blocked_memory_desc.h"
#include "utils/general_utils.h"

using namespace MKLDNNPlugin;

bool BlockedMemoryDesc::isCompatible(const BlockedMemoryDesc &rhs) const {
    if (this->getShape() != rhs.getShape() || this->getPrecision() != rhs.getPrecision())
        return false;

    if (!dimsEqualWeak(this->getBlockDims(), rhs.getBlockDims())) {
        return false;
    }

    if (!dimsEqualWeak(this->getOffsetPaddingToData(), rhs.getOffsetPaddingToData())) {
        return false;
    }

    // this check needed to avoid inserting unnecessary reorders if the memory is used in place and the batch size is equal to 1
    size_t skipAxis = this->getShape().getRank() > 0 && this->getShape().getDims().front() == 1 ? 0 :
            Shape::UNDEFINED_DIM; //ignore batch axis if batch size == 1
    if (!dimsEqualWeak(this->getStrides(), rhs.getStrides(), skipAxis)) {
        return false;
    }

    if (!dimsEqualWeak(this->getOrder(), rhs.getOrder())) {
        return false;
    }

    return dimsEqualWeak(this->getOffsetPadding(), rhs.getOffsetPadding());
}

bool BlockedMemoryDesc::isPhycicalMemCompatible(const std::shared_ptr<const BlockedMemoryDesc> rhsMemDesc) const {
    if (this->getShape() != rhsMemDesc->getShape() || this->getPrecision() != rhsMemDesc->getPrecision())
        return false;

    // padding check
    bool isZeroDimsPaddings =
        std::all_of(this->getOffsetPaddingToData().begin(), this->getOffsetPaddingToData().end(), [](size_t x){ return x == 0; }) &&
        std::all_of(rhsMemDesc->getOffsetPaddingToData().begin(), rhsMemDesc->getOffsetPaddingToData().end(), [](size_t x){ return x == 0; });
    bool isZeroTensorPadding = this->getOffsetPadding() == 0 && rhsMemDesc->getOffsetPadding() == 0;
    bool isSameElementsCount = this->getPaddedElementsCount() == rhsMemDesc->getPaddedElementsCount();
    if (!isZeroDimsPaddings || !isZeroTensorPadding || !isSameElementsCount)
        return false;

    // stride check
    const auto lhsBlockDims = this->getBlockDims();
    std::vector<size_t> lhsStridesDefault(lhsBlockDims.size());
    lhsStridesDefault[lhsBlockDims.size() - 1] = 1;
    for (size_t i = 2; i <= lhsBlockDims.size(); i++) {
        lhsStridesDefault[lhsBlockDims.size() - i] = lhsStridesDefault[lhsBlockDims.size() - (i - 1)] * lhsBlockDims[lhsBlockDims.size() - (i - 1)];
    }

    auto rhsBlockDims = rhsMemDesc->getBlockDims();
    std::vector<size_t> rhsStridesDefault(rhsBlockDims.size());
    rhsStridesDefault[rhsBlockDims.size() - 1] = 1;
    for (size_t i = 2; i <= rhsBlockDims.size(); i++) {
        rhsStridesDefault[rhsBlockDims.size() - i] =
             rhsStridesDefault[rhsBlockDims.size() - (i - 1)] * rhsBlockDims[rhsBlockDims.size() - (i - 1)];
    }

    bool isDenseTensor = dimsEqualStrong(lhsStridesDefault, this->getStrides()) &&
                         dimsEqualStrong(rhsStridesDefault, rhsMemDesc->getStrides());
    if (!isDenseTensor)
        return false;

    auto getCleanDim = [&](std::vector<size_t> dims, std::vector<size_t> flag) {
        if (dims.size() != flag.size())
            return dims;
        std::vector<size_t> ret;
        for (int i = 0; i < dims.size(); i++) {
            if (flag[i] != 1) {
                ret.push_back(dims[i]);
            }
        }
        return ret;
    };

    // block dim check
    auto lhsBlockDimsClean = getCleanDim(lhsBlockDims, lhsBlockDims);
    auto rhsBlockDimsClean = getCleanDim(rhsBlockDims, rhsBlockDims);
    if (lhsBlockDimsClean.size() != rhsBlockDimsClean.size()) {
        return false;
    } else {
        for (int i = 0; i < lhsBlockDimsClean.size(); i++) {
            if (lhsBlockDimsClean[i] != rhsBlockDimsClean[i]) {
                return false;
            }
        }
    }

    // order check
    auto lhsOrderClean = getCleanDim(this->getOrder(), lhsBlockDims);
    auto rhsOrderClean = getCleanDim(rhsMemDesc->getOrder(), rhsBlockDims);
    if (lhsOrderClean.size() != rhsOrderClean.size()) {
        return false;
    } else {
        for (int i = 0; i < lhsOrderClean.size(); i++) {
            if (lhsOrderClean[i] != rhsOrderClean[i]) {
                return false;
            }
        }
    }

    return true;
}
