// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertDepthToSpace;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertDepthToSpace: public ngraph::pass::GraphRewrite {
public:
    ConvertDepthToSpace() : GraphRewrite() {
        convert_depth_to_space();
    }

private:
    void convert_depth_to_space();
};
