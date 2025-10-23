// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "openvino/core/rtti.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"
#ifdef SNIPPETS_DEBUG_CAPS

#    pragma once

#    include <utility>

#    include "snippets/itt.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface InsertPerfCountVerbose
 * @brief Brgemm parameters dump pass
 * @ingroup snippets
 */
class InsertPerfCountVerbose : public snippets::lowered::pass::RangedPass {
public:
    explicit InsertPerfCountVerbose(std::string subgraph_name) : m_subgraph_name(std::move(subgraph_name)) {}
    OPENVINO_RTTI("InsertPerfCountVerbose", "", RangedPass);

    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;

private:
    std::string collect_params(const ov::snippets::lowered::ExpressionPtr& brgemm_expr,
                               const snippets::lowered::LinearIR& linear_ir);

    std::string m_subgraph_name;
};

}  // namespace ov::snippets::lowered::pass

#endif  // SNIPPETS_DEBUG_CAPS
