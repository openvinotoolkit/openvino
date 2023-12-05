// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <map>
#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <set>
#include <string>
#include <vector>

namespace ngraph {
namespace pass {

class ConvertInterpolateToInterpOrResampleMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertInterpolateToInterpOrResampleMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertInterpolateToInterpOrResampleMatcher", "0");
    ConvertInterpolateToInterpOrResampleMatcher();
};
