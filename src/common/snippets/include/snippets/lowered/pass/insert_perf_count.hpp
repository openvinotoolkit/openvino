// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <map>
#include <string>

#include "openvino/core/rtti.hpp"
#include "snippets/lowered/linear_ir.hpp"
#ifdef SNIPPETS_DEBUG_CAPS

#    pragma once

#    include "pass.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface InsertPerfCount
 * @brief Insert PerfCountBegin node after last parameter and insert PerfCountEnd node before first result.
 *  This is a illustration transformation to enable perf count in snippets.
 *  Developers could modify this to insert perf count pairs around interested sequence of nodes.
 * @ingroup snippets
 */
class InsertPerfCount : public RangedPass {
public:
    OPENVINO_RTTI("InsertPerfCount", "", RangedPass);
    explicit InsertPerfCount(std::map<std::string, std::string> boundary_op_names);
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

private:
    std::map<std::string, std::string> m_boundary_op_names;
};

}  // namespace ov::snippets::lowered::pass

#endif  // SNIPPETS_DEBUG_CAPS
