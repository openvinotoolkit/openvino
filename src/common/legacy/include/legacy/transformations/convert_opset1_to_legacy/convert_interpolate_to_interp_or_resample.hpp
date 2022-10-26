// Copyright (C) 2018-2022 Intel Corporation
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

namespace ngraph {
namespace pass {

class ConvertInterpolateToInterpOrResampleMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertInterpolateToInterpOrResampleMatcher: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertInterpolateToInterpOrResampleMatcher", "0");
    ConvertInterpolateToInterpOrResampleMatcher();
};
