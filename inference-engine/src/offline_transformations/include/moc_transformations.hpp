// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class MOCTransformations;

}  // namespace pass
}  // namespace ngraph

/**
 * @brief This transformation is an entry point for nGraph transformations that will be
 * applied inside MOC. And in future this transformations container will be filled
 * with transformations pipeline but now it remains empty.
 */

class ngraph::pass::MOCTransformations: public ngraph::pass::FunctionPass {
    bool m_cf;

public:
    NGRAPH_RTTI_DECLARATION;
    explicit MOCTransformations(bool cf) : m_cf(cf) {}

    bool run_on_function(std::shared_ptr<ngraph::Function>) override;
};
