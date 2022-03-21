// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>
#include <vector>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertMulticlassNmsToMulticlassNmsIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertMulticlassNmsToMulticlassNmsIE : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("public", "0");
    ConvertMulticlassNmsToMulticlassNmsIE(bool force_i32_output_type = true);
};
