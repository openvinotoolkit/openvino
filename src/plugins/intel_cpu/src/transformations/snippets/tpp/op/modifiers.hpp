// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "snippets/op/memory_access.hpp"

namespace ov {
namespace intel_cpu {

/**
 * @interface TensorProcPrim
 * @brief TensorProcPrim is modifier to mark operations supported with TPP
 * @ingroup snippets
 */
class TensorProcessingPrimitive : virtual public snippets::modifier::MemoryAccess {
};

} // namespace intel_cpu
} // namespace ov
