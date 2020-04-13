// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>


namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertNormalizeL2WithMulToNormalizeIE);
class INFERENCE_ENGINE_API_CLASS(ConvertNormalizeL2ToNormalizeIE);

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
