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

class TRANSFORMATIONS_API ConvertTopK3;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertTopK3: public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertTopK3() : GraphRewrite() {
        convert_topk3();
    }

private:
    void convert_topk3();
};
