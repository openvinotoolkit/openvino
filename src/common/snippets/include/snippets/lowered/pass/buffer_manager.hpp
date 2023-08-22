// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

class BufferManager : public PassPipeline {
public:
    BufferManager();

    void run(lowered::LinearIR& linear_ir) override;
    void propagate_offset(const LinearIR& linear_ir, const ExpressionPtr& buffer_expr, const size_t offset) const;
    size_t get_scratchpad_size() const { return m_scratchpad_size; }

private:
    void initialization(const LinearIR& linear_ir);

    size_t m_scratchpad_size = 0;
    bool m_enable_optimizations = true;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
