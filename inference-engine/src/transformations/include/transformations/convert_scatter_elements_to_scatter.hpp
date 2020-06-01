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

class TRANSFORMATIONS_API ConvertScatterElementsToScatter;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertScatterElementsToScatter: public ngraph::pass::GraphRewrite {
public:
    ConvertScatterElementsToScatter() : GraphRewrite() {
        convert_scatter_elements_to_scatter();
    }

private:
    void convert_scatter_elements_to_scatter();
};
