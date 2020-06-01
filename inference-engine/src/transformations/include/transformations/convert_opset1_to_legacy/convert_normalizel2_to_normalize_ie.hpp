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

class TRANSFORMATIONS_API ConvertNormalizeL2WithMulToNormalizeIE;
class TRANSFORMATIONS_API ConvertNormalizeL2ToNormalizeIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertNormalizeL2WithMulToNormalizeIE: public ngraph::pass::GraphRewrite {
public:
    ConvertNormalizeL2WithMulToNormalizeIE() : GraphRewrite() {
        convert_normalize_l2_with_mul();
    }

private:
    void convert_normalize_l2_with_mul();
};

class ngraph::pass::ConvertNormalizeL2ToNormalizeIE: public ngraph::pass::GraphRewrite {
public:
    ConvertNormalizeL2ToNormalizeIE() : GraphRewrite() {
        convert_normalize_l2();
    }

private:
    void convert_normalize_l2();
};
