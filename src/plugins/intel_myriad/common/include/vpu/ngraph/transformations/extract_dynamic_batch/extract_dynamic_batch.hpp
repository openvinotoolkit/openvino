// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/pass/graph_rewrite.hpp"

#include <memory>

namespace vpu {

class ExtractBatch: public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("ExtractBatch", "0");

    explicit ExtractBatch(std::unordered_set<ngraph::Node::type_info_t> targets);
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;

private:
    std::unordered_set<ngraph::Node::type_info_t> targets;
};

}  // namespace vpu
