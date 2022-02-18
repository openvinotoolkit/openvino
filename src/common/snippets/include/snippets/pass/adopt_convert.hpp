// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>


namespace ngraph {
namespace snippets {
namespace pass {

class AdoptConvert: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    explicit AdoptConvert(const std::vector<ov::element::Type> supported_output_types);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
