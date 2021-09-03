// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

class ConvertUnalignedConcatIntoGnaGraph : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertUnalignedConcatIntoGnaGraph();
    bool unaligned_concat_converted = false;
};

class RemoveTrivialStrideConcatPattern : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    RemoveTrivialStrideConcatPattern();
};

class TransformConcatForGna : public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    TransformConcatForGna();

    bool unalignedConcatIntoGnaGraphConverted() const;

private:
    std::shared_ptr<ConvertUnalignedConcatIntoGnaGraph> convUCPass;
};

} // namespace GNAPluginNS
