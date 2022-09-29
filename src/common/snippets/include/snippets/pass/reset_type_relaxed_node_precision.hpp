// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface ResetTypeRelaxedNodePrecision
 * @brief Reset precision for type relaxed nodes inside body to align precision between nodes.
 *        Should be called after all Convert insertions
 * @ingroup snippets
 */
class ResetTypeRelaxedNodePrecision: public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("ResetTypeRelaxedNodePrecision", "0");
    ResetTypeRelaxedNodePrecision(const ov::element::Type exec_type = ov::element::f32);
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
private:
    ov::element::Type exec_type;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
