// Copyright (C) 2018-2021 Intel Corporation
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

namespace ov {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertPadToLegacyMatcher);

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertPadToLegacyMatcher: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPadToLegacyMatcher();
};
