// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/op/loop.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface InsertTailLoop
 * @brief Injects tail-processing loop after a vector loop if required.
 *  Additional optimizations are performed if a loop body is executed only once.
 * @ingroup snippets
 */
class InsertTailLoop : public Pass {
public:
    OPENVINO_RTTI("InsertTailLoop", "Pass")
    bool run(LinearIR& linear_ir) override;

private:
    static std::shared_ptr<op::LoopEnd> create_tail_loop(LinearIR& linear_ir,
                                                         LinearIR::constExprIt vector_begin,
                                                         LinearIR::constExprIt vector_end,
                                                         LinearIR::constExprIt& tail_begin,
                                                         LinearIR::constExprIt& tail_end,
                                                         const std::shared_ptr<op::LoopEnd>& vector_loop_end,
                                                         bool need_vector_loop,
                                                         size_t tail_size, const std::vector<int64_t>& tail_finalization_offsets);
    static void tail_transformations(LinearIR& linear_ir,
                                     LinearIR::constExprIt tail_begin,
                                     LinearIR::constExprIt tail_end,
                                     size_t tail_size);
    static bool optimize_single_evaluation(const std::shared_ptr<op::LoopEnd>& loop);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
