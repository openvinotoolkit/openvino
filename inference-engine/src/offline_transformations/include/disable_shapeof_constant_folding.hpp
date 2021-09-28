// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class DisableShapeOfConstantFolding;

}  // namespace pass
}  // namespace ngraph


class ngraph::pass::DisableShapeOfConstantFolding: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    DisableShapeOfConstantFolding();
};
