// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/pass/graph_rewrite.hpp"

#include <memory>

namespace vpu {

class ExtractBatch: public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;

    explicit ExtractBatch(std::unordered_set<ngraph::Node::type_info_t> targets);
    bool run_on_function(std::shared_ptr<ngraph::Function> function) override;

private:
    std::unordered_set<ngraph::Node::type_info_t> targets;
};

}  // namespace vpu
