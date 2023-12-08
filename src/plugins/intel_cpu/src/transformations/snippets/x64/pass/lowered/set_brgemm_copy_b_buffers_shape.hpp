// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface SetBrgemmCopyBBuffersShape
 * @brief Sets the allocation shape for the Buffers after BrgemmCopyB node using BrgemmCopyB parameters
 * @ingroup snippets
 */
class SetBrgemmCopyBBuffersShape: public snippets::lowered::pass::Pass {
public:
    SetBrgemmCopyBBuffersShape() = default;
    OPENVINO_RTTI("SetBrgemmCopyBBuffersShape", "Pass");
    bool run(snippets::lowered::LinearIR& linear_ir) override;
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
