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

class ConvertOneHotToOneHotIEMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertOneHotToOneHotIEMatcher: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertOneHotToOneHotIEMatcher", "0");
    ConvertOneHotToOneHotIEMatcher();

    void detect_output_type(const std::shared_ptr<Function> & f);

private:
    element::Type m_output_type = element::Type_t::f32;
};
