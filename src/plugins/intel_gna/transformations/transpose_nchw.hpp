// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief TODO
 */
class SubstituteGNAConvolution : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SubstituteGNAConvolution();
};

class SubstituteGNAMaxPool : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SubstituteGNAMaxPool();
};

class TransposeNCHW : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
     bool run_on_model(const std::shared_ptr<ngraph::Function>& f) override;
};

} // namespace GNAPluginNS
