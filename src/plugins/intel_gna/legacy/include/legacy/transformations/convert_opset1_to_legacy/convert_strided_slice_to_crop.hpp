// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <algorithm>
#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class ConvertStridedSliceToCropMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertStridedSliceToCropMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertStridedSliceToCropMatcher", "0");
    ConvertStridedSliceToCropMatcher();
};
