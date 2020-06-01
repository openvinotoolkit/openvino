// Copyright (C) 2020 Intel Corporation
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

class TRANSFORMATIONS_API ConvertSqrtToPowerIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertSqrtToPowerIE: public ngraph::pass::GraphRewrite {
public:
    ConvertSqrtToPowerIE() : GraphRewrite() {
        convert_sqrt();
    }

private:
    void convert_sqrt();
};

