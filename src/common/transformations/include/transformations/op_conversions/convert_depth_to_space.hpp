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

class TRANSFORMATIONS_API ConvertDepthToSpace;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertDepthToSpace : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertDepthToSpace", "0");
    ConvertDepthToSpace();
};
