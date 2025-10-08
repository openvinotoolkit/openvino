// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/rtti.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "snippets/snippets_visibility.hpp"
#include "subgraph_pass.hpp"

namespace ov::snippets::pass {

/**
 * @interface ExtractConstants
 * @brief Moves up Constants which aren't scalars outside of the Subgraph's body and replaces them with Parameters
 * inside body
 * @ingroup snippets
 */
class SNIPPETS_API ExtractConstants : public CommonOptimizations::SubgraphPass {
public:
    OPENVINO_RTTI("ExtractConstants", "0");
    ExtractConstants() : SubgraphPass("ExtractConstants") {}

    bool run_on_subgraph(const std::shared_ptr<op::Subgraph>& subgraph) override;
};

}  // namespace ov::snippets::pass
