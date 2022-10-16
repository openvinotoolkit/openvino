// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API CommonOptimizations;

}  // namespace pass
}  // namespace ov

class ov::pass::CommonOptimizations : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("CommonOptimizations", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>& f) override;
};

namespace ngraph {
namespace pass {
using ov::pass::CommonOptimizations;
}  // namespace pass
}  // namespace ngraph
