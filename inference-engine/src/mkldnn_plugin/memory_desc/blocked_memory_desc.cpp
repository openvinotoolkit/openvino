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

    if (!dimsEqualWeak(this->getStrides(), rhs.getStrides())) {
        return false;
    }

    if (!dimsEqualWeak(this->getOrder(), rhs.getOrder())) {
        return false;
    }

    return dimsEqualWeak(this->getOffsetPadding(), rhs.getOffsetPadding());
}
