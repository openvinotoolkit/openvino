// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

// @brief TO DO

class HandleMultiConnectedLayerToConcat : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HandleMultiConnectedLayerToConcat();
};

class InsertCopyBeforeMemoryLayer : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    InsertCopyBeforeMemoryLayer();
};

class InsertCopyBetweenCropConcat : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    InsertCopyBetweenCropConcat();
};

class InsertCopyBetweenSplitConcat : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    InsertCopyBetweenSplitConcat();
};

class HandleLayerConnectedToConcatOrMemory : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_model(const std::shared_ptr<ngraph::Function>& f) override;
};

class HandleNonComputationalSubgraphs : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_model(const std::shared_ptr<ngraph::Function>& f) override;
};


} // namespace GNAPluginNS

