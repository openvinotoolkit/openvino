// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blocked_memory_desc.h"

#include "utils/general_utils.h"

using namespace MKLDNNPlugin;

bool BlockedMemoryDesc::isCompatible(const BlockedMemoryDesc& rhs) const {
    if (this->getShape() != rhs.getShape() || this->getPrecision() != rhs.getPrecision())
        return false;

    if (!dimsEqualWeak(this->getBlockDims(), rhs.getBlockDims())) {
        return false;
    }

    if (!dimsEqualWeak(this->getOffsetPaddingToData(), rhs.getOffsetPaddingToData())) {
        return false;
    }

    // this check needed to avoid inserting unnecessary reorders if the memory is used in place and the batch size is
    // equal to 1
    size_t skipAxis = this->getShape().getRank() > 0 && this->getShape().getDims().front() == 1
                          ? 0
                          : Shape::UNDEFINED_DIM;  // ignore batch axis if batch size == 1
    if (!dimsEqualWeak(this->getStrides(), rhs.getStrides(), skipAxis)) {
        return false;
    }

    if (!dimsEqualWeak(this->getOrder(), rhs.getOrder())) {
        return false;
    }

    return dimsEqualWeak(this->getOffsetPadding(), rhs.getOffsetPadding());
}
