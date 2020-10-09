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

class TRANSFORMATIONS_API ConvertPriorBox;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertPriorBox: public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPriorBox() : GraphRewrite() {
        convert_prior_box();
        convert_prior_box_clustered();
    }

private:
    void convert_prior_box();

    void convert_prior_box_clustered();
};
