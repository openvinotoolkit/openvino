// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConstantEltwiseReduction;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConstantEltwiseReduction: public ngraph::pass::GraphRewrite {
public:
    ConstantEltwiseReduction() : GraphRewrite() {
        constant_multiply_reduction();
        constant_add_reduction();
    }

private:
    void constant_multiply_reduction();
    void constant_add_reduction();
};
