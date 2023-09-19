// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <typeinfo>

#include "snippets/op/subgraph.hpp"

#include "openvino/pass/pass.hpp"


namespace ov {
namespace snippets {
namespace pass {
/**
 * @brief Base class for Subgraph passes
 * @ingroup snippets
 */
class SubgraphPass : public ov::pass::PassBase {
public:
    OPENVINO_RTTI("ov::pass::SubgraphPass");
    ~SubgraphPass() override = default;
    virtual bool run_on_subgraph(const std::shared_ptr<op::Subgraph>& subgraph) = 0;
};


} // namespace pass
} // namespace snippets
} // namespace ov
