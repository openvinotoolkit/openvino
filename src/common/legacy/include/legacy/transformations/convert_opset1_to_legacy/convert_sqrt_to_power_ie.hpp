// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class ConvertSqrtToPowerIEMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertSqrtToPowerIEMatcher: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertSqrtToPowerIEMatcher", "0");
    ConvertSqrtToPowerIEMatcher();
};

