// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>


namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API CommonOptimizations;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::CommonOptimizations: public ngraph::pass::FunctionPass {
    bool m_low_precision_enabled;
public:
    NGRAPH_RTTI_DECLARATION;
    explicit CommonOptimizations(bool low_precision_enabled = true) : m_low_precision_enabled(low_precision_enabled) {}
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};
