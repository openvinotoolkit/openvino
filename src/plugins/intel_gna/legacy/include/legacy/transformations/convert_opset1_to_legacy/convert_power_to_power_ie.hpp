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

class ConvertPowerToPowerIEMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertPowerToPowerIEMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertPowerToPowerIEMatcher", "0");
    ConvertPowerToPowerIEMatcher();
};
