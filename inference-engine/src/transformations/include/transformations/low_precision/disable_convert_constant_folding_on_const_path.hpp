// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API DisableConvertConcstantFoldingOnConstPath;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::DisableConvertConcstantFoldingOnConstPath : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    DisableConvertConcstantFoldingOnConstPath(
        const std::vector<ngraph::element::Type>& inputPrecisions = {});
};
