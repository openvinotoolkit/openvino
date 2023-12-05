// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <legacy/ngraph_ops/gather_tree_ie.hpp>
#include <memory>
#include <ngraph/op/gather_tree.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class ConvertGatherTreeToGatherTreeIEMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertGatherTreeToGatherTreeIEMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGatherTreeToGatherTreeIEMatcher", "0");
    ConvertGatherTreeToGatherTreeIEMatcher();
};
