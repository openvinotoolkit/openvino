// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>
#include <legacy/ngraph_ops/pad_ie.hpp>

#include "ngraph/op/lrn.hpp"
#include "ngraph/op/constant.hpp"

namespace ngraph {
namespace pass {

class ConvertPadToLegacyMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertPadToLegacyMatcher: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertPadToLegacyMatcher", "0");
    ConvertPadToLegacyMatcher();
};
