// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class POTTransformations;

}  // namespace pass
}  // namespace ngraph

/**
 * @brief This transformation is an entry point for nGraph transformations that will be
 * executed inside POT.
 */

class ngraph::pass::POTTransformations: public ngraph::pass::FunctionPass {
    std::string m_device;

public:
    NGRAPH_RTTI_DECLARATION;
    explicit POTTransformations(std::string device) : m_device(std::move(device)) {}

    bool run_on_function(std::shared_ptr<ngraph::Function>) override;
};
