// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API FuseFilteringBoxesBySize;
class TRANSFORMATIONS_API RemoveFilteringBoxesBySize;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::FuseFilteringBoxesBySize : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("FuseFilteringBoxesBySize", "0");
    FuseFilteringBoxesBySize();
};

class ngraph::pass::RemoveFilteringBoxesBySize : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("RemoveFilteringBoxesBySize", "0");
    RemoveFilteringBoxesBySize();
};
