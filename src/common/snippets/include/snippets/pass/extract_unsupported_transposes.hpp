// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/rtti.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "subgraph_pass.hpp"

namespace ov::snippets::pass {

/**
 * @interface ExtractUnsupportedTransposes
 * @brief Moves up unsupported Transposes on Parameter outputs from body
 * @ingroup snippets
 */
class ExtractUnsupportedTransposes : public CommonOptimizations::SubgraphPass {
public:
    OPENVINO_RTTI("ExtractUnsupportedTransposes", "0");
    explicit ExtractUnsupportedTransposes(CommonOptimizations::Config::TransposeSupportCallback transpose_support_cb)
        : SubgraphPass("ExtractUnsupportedTransposes"),
          m_transpose_support_cb(std::move(transpose_support_cb)) {}

    bool run_on_subgraph(const std::shared_ptr<op::Subgraph>& subgraph) override;

private:
    CommonOptimizations::Config::TransposeSupportCallback m_transpose_support_cb;
};

}  // namespace ov::snippets::pass
