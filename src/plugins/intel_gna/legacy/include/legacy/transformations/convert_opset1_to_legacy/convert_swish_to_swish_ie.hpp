// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <string>
#include <vector>

namespace ngraph {
namespace pass {
class ConvertSwishToSwishIEMatcher;
}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertSwishToSwishIEMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertSwishToSwishIEMatcher", "0");
    ConvertSwishToSwishIEMatcher();
};
