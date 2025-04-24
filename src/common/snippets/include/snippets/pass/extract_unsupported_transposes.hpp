// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "subgraph_pass.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface ExtractUnsupportedTransposes
 * @brief Moves up unsupported Transposes on Parameter outputs from body
 * @ingroup snippets
 */
class ExtractUnsupportedTransposes: public CommonOptimizations::SubgraphPass {
public:
    OPENVINO_RTTI("ExtractUnsupportedTransposes", "0");
    ExtractUnsupportedTransposes() = default;

    bool run_on_subgraph(const std::shared_ptr<op::Subgraph>& subgraph) override;
};


} // namespace pass
} // namespace snippets
} // namespace ov
