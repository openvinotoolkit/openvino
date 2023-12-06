// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

class UpdateSubtensors : public pass::SubgraphPass {
public:
    UpdateSubtensors(size_t tail_size);
    OPENVINO_RTTI("UpdateSubtensors", "Pass")
    bool run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override;

private:
    size_t m_tail_size;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov