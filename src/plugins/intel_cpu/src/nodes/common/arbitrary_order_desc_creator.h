// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "blocked_desc_creator.h"
#include "cpu_shape.h"
#include "cpu_types.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "openvino/core/type/element_type.hpp"

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
