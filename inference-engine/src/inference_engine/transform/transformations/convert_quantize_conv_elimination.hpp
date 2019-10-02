// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class ConvertElimination;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertElimination: public ngraph::pass::GraphRewrite {
public:
    ConvertElimination() : GraphRewrite() {
        convert_elimination();
    }

private:
    void convert_elimination();
};
