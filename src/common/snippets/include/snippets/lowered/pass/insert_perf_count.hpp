// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

#pragma once

#include "pass.hpp"

#include "snippets/op/perf_count.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface InsertPerfCount
 * @brief Insert PerfCountBegin node after last parameter and insert PerfCountEnd node before first result.
 *  This is a illustration transformation to enable perf count in snippets.
 *  Developers could modify this to insert perf count pairs around interested sequence of nodes.
 * @ingroup snippets
 */
class InsertPerfCount: public Pass {
public:
    OPENVINO_RTTI("InsertPerfCount", "Pass")
    InsertPerfCount() = default;
    bool run(LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
#endif  // SNIPPETS_DEBUG_CAPS
