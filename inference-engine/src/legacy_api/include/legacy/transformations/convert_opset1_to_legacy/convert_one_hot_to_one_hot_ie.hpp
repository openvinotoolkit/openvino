// Copyright (C) 2018-2020 Intel Corporation
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

class INFERENCE_ENGINE_API_CLASS(ConvertOneHotToOneHotIEMatcher);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertOneHotToOneHotIEMatcher: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertOneHotToOneHotIEMatcher();

    void detect_output_type(const std::shared_ptr<Function> & f);

private:
    element::Type m_output_type = element::Type_t::f32;
};