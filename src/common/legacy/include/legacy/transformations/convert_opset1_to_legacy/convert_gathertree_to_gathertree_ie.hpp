// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

#include <ngraph/op/gather_tree.hpp>
#include <legacy/ngraph_ops/gather_tree_ie.hpp>

namespace ngraph {
namespace pass {

class ConvertGatherTreeToGatherTreeIEMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertGatherTreeToGatherTreeIEMatcher: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGatherTreeToGatherTreeIEMatcher", "0");
    ConvertGatherTreeToGatherTreeIEMatcher();
};
