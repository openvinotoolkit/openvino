// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/manager.hpp>


namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API TensorIteratorTransformations;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::TensorIteratorTransformations: public ngraph::pass::FunctionPass {
public:
    explicit TensorIteratorTransformations(const ngraph::pass::Manager& manager) {
        m_external_manager = manager;
    }
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
private:
    ngraph::pass::Manager m_external_manager;
};
