// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API DisableConvertConstantFoldingOnConstPath;

}  // namespace pass
}  // namespace ov

class ov::pass::DisableConvertConstantFoldingOnConstPath : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("DisableConvertConstantFoldingOnConstPath", "0");
    DisableConvertConstantFoldingOnConstPath(const element::TypeVector& inputPrecisions = {});
};

namespace ngraph {
namespace pass {
using ov::pass::DisableConvertConstantFoldingOnConstPath;
}  // namespace pass
}  // namespace ngraph
