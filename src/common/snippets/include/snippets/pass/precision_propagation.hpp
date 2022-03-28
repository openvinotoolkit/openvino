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
 * @interface PrecisionPropagation
 * @brief Propagate precision inside body to align precision between nodes. Should be called after all Convert insertions
 * @ingroup snippets
 */
class PrecisionPropagation: public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("PrecisionPropagation", "0");
    PrecisionPropagation(const ov::element::Type default_type = ov::element::f32);
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
private:
    ov::element::Type default_type;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
