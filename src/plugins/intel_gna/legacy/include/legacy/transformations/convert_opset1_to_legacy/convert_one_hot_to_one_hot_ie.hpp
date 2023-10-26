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

class ConvertOneHotToOneHotIEMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertOneHotToOneHotIEMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertOneHotToOneHotIEMatcher", "0");
    ConvertOneHotToOneHotIEMatcher();

    void detect_output_type(const std::shared_ptr<Function>& f);

private:
    element::Type m_output_type = element::Type_t::f32;
};
