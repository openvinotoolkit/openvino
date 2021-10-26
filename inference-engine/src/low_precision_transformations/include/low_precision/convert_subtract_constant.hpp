// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <low_precision/lpt_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API ConvertSubtractConstant;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

class ngraph::pass::low_precision::ConvertSubtractConstant : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertSubtractConstant(const std::vector<ngraph::element::Type>& constantPrecisions = {});
};
