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

class TRANSFORMATIONS_API ConvertShapeOf3;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertShapeOf3 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertShapeOf3", "0");
    ConvertShapeOf3();
};
