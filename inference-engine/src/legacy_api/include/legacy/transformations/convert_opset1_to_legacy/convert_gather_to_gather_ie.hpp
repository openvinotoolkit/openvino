// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>
#include <legacy/ngraph_ops/gather_ie.hpp>

#include "ngraph/op/gather.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/squeeze.hpp"
#include "ngraph/op/unsqueeze.hpp"


namespace ov {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertGatherToGatherIEMatcher);

}  // namespace pass
}  // namespace ov

/*
 * Description:
 *     This transformation converts opset1::Gather to legacy GatherIE
 *     GatherIE takes axes as value and if indices input has empty shape (scalar)
 *     we unsqueeze indices input and squeeze GatherIE output.
 */

class ov::pass::ConvertGatherToGatherIEMatcher : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGatherToGatherIEMatcher();
};
