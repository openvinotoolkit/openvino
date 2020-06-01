// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertSeluToSeluIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertSeluToSeluIE: public ngraph::pass::GraphRewrite {
public:
    ConvertSeluToSeluIE() : GraphRewrite() {
        convert_selu();
    }

private:
    void convert_selu();
};
