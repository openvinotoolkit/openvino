// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class NGRAPH_API SmartReshape;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::SmartReshape : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("SmartReshape", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};
