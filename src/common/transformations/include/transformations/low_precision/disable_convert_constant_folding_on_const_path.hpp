// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API DisableConvertConstantFoldingOnConstPath;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::DisableConvertConstantFoldingOnConstPath : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("DisableConvertConstantFoldingOnConstPath", "0");
    DisableConvertConstantFoldingOnConstPath(const element::TypeVector& inputPrecisions = {});
};
