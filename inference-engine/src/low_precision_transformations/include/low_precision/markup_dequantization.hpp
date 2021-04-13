// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "common/operation_precision_restriction.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API MarkupDequantizations;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

// Transformation is used to add customization options runtime
class ngraph::pass::low_precision::MarkupDequantizations : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MarkupDequantizations();
};
