// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>


namespace ov {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertNormalizeL2WithMulToNormalizeIE);
class INFERENCE_ENGINE_API_CLASS(ConvertNormalizeL2ToLegacyMatcher);

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertNormalizeL2WithMulToNormalizeIE: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertNormalizeL2WithMulToNormalizeIE();
};

class ov::pass::ConvertNormalizeL2ToLegacyMatcher: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertNormalizeL2ToLegacyMatcher();
};
