// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS

#pragma once

#include "snippets/itt.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/specific_loop_iter_handlers.hpp"
#include "snippets/lowered/pass/iter_handler.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface InsertPerfCountVerbose
 * @brief Brgemm parameters dump pass
 * @ingroup snippets
 */
class InsertPerfCountVerbose : public snippets::lowered::pass::RangedPass {
public:
    InsertPerfCountVerbose(const std::string& subgraph_name) : m_subgraph_name(subgraph_name) {}
    OPENVINO_RTTI("InsertPerfCountVerbose", "", RangedPass);

    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;

private:
    std::string collect_params(const ov::snippets::lowered::ExpressionPtr& brgemm_expr,
                               const snippets::lowered::LinearIR& linear_ir);

    std::string m_subgraph_name;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

#endif  // SNIPPETS_DEBUG_CAPS
