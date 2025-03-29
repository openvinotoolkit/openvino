// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "blocked_desc_creator.h"

namespace ov {
namespace intel_cpu {

class ArbitraryOrderDescCreator : public BlockedDescCreator {
public:
    ArbitraryOrderDescCreator(VectorDims order);

    CpuBlockedMemoryDesc createDesc(const ov::element::Type& precision, const Shape& srcShape) const override;
    size_t getMinimalRank() const override;

private:
    VectorDims m_order;
};

}  // namespace intel_cpu
}  // namespace ov
