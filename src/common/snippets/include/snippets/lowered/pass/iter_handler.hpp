// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/lowered/pass/pass_pipeline.hpp"
#include "snippets/op/loop.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

class SetSingleIterationWithWorkAmount : public pass::SubgraphPass {
public:
    SetSingleIterationWithWorkAmount(size_t work_amount);
    OPENVINO_RTTI("SetSingleIterationWithWorkAmount", "Pass")
    bool run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override;

private:
    size_t m_work_amount;
};

class UpdateMemoryAccessOps : public pass::SubgraphPass {
public:
    UpdateMemoryAccessOps(size_t count);
    OPENVINO_RTTI("UpdateMemoryAccessOps", "Pass")
    bool run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override;

private:
    size_t m_count;
};

class ReduceWorkAmount : public pass::SubgraphPass {
public:
    ReduceWorkAmount(size_t reduce_value);
    OPENVINO_RTTI("ReduceWorkAmount", "Pass")
    bool run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override;

private:
    size_t m_reduce_value;
};

class ZeroFinalizationOffsets : public pass::SubgraphPass {
public:
    OPENVINO_RTTI("ZeroFinalizationOffsets", "Pass")
    bool run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override;
};

class SetFillOffset : public pass::SubgraphPass {
public:
    SetFillOffset(size_t offset);
    OPENVINO_RTTI("SetFillOffset", "Pass")
    bool run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override;

private:
    size_t m_offset;
};

class TransformInnerSplitLoop : public pass::SubgraphPass {
public:
    TransformInnerSplitLoop(size_t tail_size);
    OPENVINO_RTTI("TransformInnerSplitLoop", "Pass")
    bool run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override;

private:
    size_t m_tail_size;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov