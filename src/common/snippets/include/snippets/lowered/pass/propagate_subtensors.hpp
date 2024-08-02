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
/**
 * @interface UpdateSubtensors
 * @brief The pass updates subtensors of all operations in Loop based on tail size.
 * Firstly, the pass updates subtensors of all Loop input ports.
 * After that, shape inference infrastructure is used to update subtensors of all ops in Loop body
 * @param m_offset - offset which must be set
 * @ingroup snippets
 */
class UpdateSubtensors : public pass::RangedPass {
public:
    UpdateSubtensors(size_t tail_size);
    OPENVINO_RTTI("UpdateSubtensors", "RangedPass")
    bool run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override;
    std::shared_ptr<pass::PassBase> merge(const std::shared_ptr<pass::PassBase>& other) override;

private:
    size_t m_tail_size;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov