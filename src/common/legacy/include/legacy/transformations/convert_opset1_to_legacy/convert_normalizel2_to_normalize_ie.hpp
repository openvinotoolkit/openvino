// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>


namespace ngraph {
namespace pass {

class ConvertNormalizeL2WithMulToNormalizeIE;
class ConvertNormalizeL2ToLegacyMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertNormalizeL2WithMulToNormalizeIE: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertNormalizeL2WithMulToNormalizeIE", "0");
    ConvertNormalizeL2WithMulToNormalizeIE();
};

class ngraph::pass::ConvertNormalizeL2ToLegacyMatcher: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertNormalizeL2ToLegacyMatcher", "0");
    ConvertNormalizeL2ToLegacyMatcher();
};
