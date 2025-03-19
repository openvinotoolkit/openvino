// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "subgraph_pass.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface ExtractConstants
 * @brief Moves up Constants which aren't scalars outside of the Subgraph's body and replaces them with Parameters inside body
 * @ingroup snippets
 */
class ExtractConstants: public CommonOptimizations::SubgraphPass {
public:
    OPENVINO_RTTI("ExtractConstants", "0");
    ExtractConstants() = default;

    bool run_on_subgraph(const std::shared_ptr<op::Subgraph>& subgraph) override;
};


} // namespace pass
} // namespace snippets
} // namespace ov
