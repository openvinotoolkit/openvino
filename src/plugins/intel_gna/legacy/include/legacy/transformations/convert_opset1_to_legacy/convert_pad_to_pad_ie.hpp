// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <legacy/ngraph_ops/pad_ie.hpp>
#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <string>
#include <vector>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/lrn.hpp"

namespace ngraph {
namespace pass {

class ConvertPadToLegacyMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertPadToLegacyMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertPadToLegacyMatcher", "0");
    ConvertPadToLegacyMatcher();
};
