// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <set>
#include <string>
#include <vector>
#include <memory>
#include <map>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertInterpolateToInterpOrResampleMatcher);

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertInterpolateToInterpOrResampleMatcher: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertInterpolateToInterpOrResampleMatcher();
};
