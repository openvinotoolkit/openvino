// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertPriorBox);
class INFERENCE_ENGINE_API_CLASS(ConvertPriorBoxToLegacy);
class INFERENCE_ENGINE_API_CLASS(ConvertPriorBoxClusteredToLegacy);

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertPriorBoxToLegacy : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPriorBoxToLegacy();
};

class ov::pass::ConvertPriorBoxClusteredToLegacy : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPriorBoxClusteredToLegacy();
};

class ov::pass::ConvertPriorBox: public ov::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPriorBox() {
        add_matcher<ov::pass::ConvertPriorBoxToLegacy>();
        add_matcher<ov::pass::ConvertPriorBoxClusteredToLegacy>();
    }
};
