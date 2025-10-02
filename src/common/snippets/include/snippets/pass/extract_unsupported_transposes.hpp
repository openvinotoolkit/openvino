// Copyright (C) 2018-2025 Intel Corporation
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
    ExtractUnsupportedTransposes() : SubgraphPass("ExtractUnsupportedTransposes") {}

    bool run_on_subgraph(const std::shared_ptr<op::Subgraph>& subgraph) override;
};

}  // namespace ov::snippets::pass
