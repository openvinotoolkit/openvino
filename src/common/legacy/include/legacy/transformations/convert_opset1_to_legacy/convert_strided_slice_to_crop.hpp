// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <algorithm>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class ConvertStridedSliceToCropMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertStridedSliceToCropMatcher: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertStridedSliceToCropMatcher", "0");
    ConvertStridedSliceToCropMatcher();
};
