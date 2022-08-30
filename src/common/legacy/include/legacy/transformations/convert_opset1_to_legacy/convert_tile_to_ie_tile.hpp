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

class ConvertTileToLegacyMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertTileToLegacyMatcher: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertTileToLegacyMatcher", "0");
    ConvertTileToLegacyMatcher();
};
