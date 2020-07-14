// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/ngraph.hpp>

namespace ngraph {
namespace pass {

class MyPatternBasedTransformation;

}  // namespace pass
}  // namespace ngraph

// ! [graph_rewrite:template_transformation_hpp]
// template_pattern_transformation.hpp
class ngraph::pass::MyPatternBasedTransformation: public ngraph::pass::GraphRewrite {
public:
    MyPatternBasedTransformation() : GraphRewrite() {
        transform();
    }

private:
    void transform();
};
// ! [graph_rewrite:template_transformation_hpp]
