// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

#include "ngraph/op/gelu.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertGELU;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertGELU : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGELU", "0");
    ConvertGELU();
};
