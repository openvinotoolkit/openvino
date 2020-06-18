// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>

// ! [graph_rewrite:template_transformation_hpp]
// template_pattern_transformation.hpp
class MyPatternBasedTransformation: public ngraph::pass::GraphRewrite {
public:
    MyPatternBasedTransformation() : GraphRewrite() {
        transform();
    }

private:
    void transform();
};
// ! [graph_rewrite:template_transformation_hpp]