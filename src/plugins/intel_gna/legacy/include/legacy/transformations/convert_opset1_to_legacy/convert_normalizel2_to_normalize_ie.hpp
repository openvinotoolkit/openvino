// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class ConvertNormalizeL2WithMulToNormalizeIE;
class ConvertNormalizeL2ToLegacyMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertNormalizeL2WithMulToNormalizeIE : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertNormalizeL2WithMulToNormalizeIE", "0");
    ConvertNormalizeL2WithMulToNormalizeIE();
};

class ngraph::pass::ConvertNormalizeL2ToLegacyMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertNormalizeL2ToLegacyMatcher", "0");
    ConvertNormalizeL2ToLegacyMatcher();
};
