// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <legacy/ngraph_ops/gather_ie.hpp>
#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <string>
#include <vector>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/squeeze.hpp"
#include "ngraph/op/unsqueeze.hpp"

namespace ngraph {
namespace pass {

class ConvertGatherToGatherIEMatcher;

}  // namespace pass
}  // namespace ngraph

/*
 * Description:
 *     This transformation converts opset1::Gather to legacy GatherIE
 *     GatherIE takes axes as value and if indices input has empty shape (scalar)
 *     we unsqueeze indices input and squeeze GatherIE output.
 */

class ngraph::pass::ConvertGatherToGatherIEMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGatherToGatherIEMatcher", "0");
    ConvertGatherToGatherIEMatcher();
};
